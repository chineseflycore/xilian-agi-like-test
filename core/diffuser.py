# -*- coding: utf-8 -*-
"""
core/diffuser.py — 三线扩散器 v7.3
====================================
显存预估: L2 ≈ 0.01GB（~0.5M 参数 × 4-bit ≈ 0.25MB 常驻）; L1/L3 纯 CPU（< 20MB RAM）

职责: 把"活跃协议码"沿语义邻接关系扩散, 产出扩散激活码（供路由/解码器/记忆检索）。

  L1 —— CPU 无参数: 汉明相似度邻接扩散（相似度 = 1 - hamming/16）, 无任何权重。
  L2 —— GPU 小 Transformer: 输入 soft 码序列 [B,S,16] → d=128 → 输出 16 维 logits,
        预测下一码位概率; 数据 = knowledge/xilian_copy.json 协议码序列（自回归 CE）。
  L3 —— CPU CSR: 热门码空间 (4096×4096) 稀疏计数矩阵, 由记忆池已见码转移计数构建,
        按行 top-k 采样扩散。

接口: diffuse(active_codes, k=8) -> [(code, score), ...]; set_mode("L1"|"L2"|"L3")
"""
import os
import time
import json
import threading
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

import config
from core import protocol
from core import quant

try:
    _HAS_TORCH = True
except Exception:            # pragma: no cover
    _HAS_TORCH = False


# ----------------------------------------------------------------------
# L1: CPU 无参数汉明扩散
# ----------------------------------------------------------------------
def _hamming_diffuse(active_codes, candidates, k: int):
    """对候选码按与活跃码的加权汉明相似度排序, 返回 top-k [(code, score)]。"""
    scores = []
    for c in candidates:
        s = sum(protocol.similarity(a, c) for a in active_codes) / max(1, len(active_codes))
        scores.append((int(c), float(s)))
    scores.sort(key=lambda kv: -kv[1])
    return scores[:k]


# ----------------------------------------------------------------------
# L2: GPU 小 Transformer（soft 码自回归）
# ----------------------------------------------------------------------
class _L2Net(torch.nn.Module):
    """L2 转移场网络（d=128, 含 Dropout 正则防过拟合）。"""

    def __init__(self, d_model=128, heads=4, layers=1):
        super().__init__()
        self.embed = torch.nn.Linear(16, d_model)
        self.pos = torch.nn.Parameter(torch.zeros(16, d_model))
        self.drop = torch.nn.Dropout(0.15)     # 语料仅 684 条 → 强正则
        self.blocks = torch.nn.ModuleList([])
        for _ in range(layers):
            self.blocks.append(torch.nn.Sequential())
            self.blocks[-1].add_module("ln1", torch.nn.LayerNorm(d_model))
            b = torch.nn.MultiheadAttention(d_model, heads, batch_first=True)
            self.blocks[-1].add_module("attn", b)
            self.blocks[-1].add_module("ln2", torch.nn.LayerNorm(d_model))
            self.blocks[-1].add_module("ffn", torch.nn.Sequential(
                torch.nn.Linear(d_model, d_model * 2), torch.nn.GELU(),
                torch.nn.Linear(d_model * 2, d_model)))
        self.out = torch.nn.Linear(d_model, 16)

    def forward(self, x):
        n = min(x.shape[1], self.pos.shape[0])     # 超长截断（防御, 不超过 16 步）
        h = self.drop(self.embed(x[:, :n]) + self.pos[:n])
        for blk in self.blocks:
            h = h + self.drop(blk.attn(blk.ln1(h), blk.ln1(h), blk.ln1(h),
                                       need_weights=False)[0])
            h = h + blk.ffn(blk.ln2(h))
        return self.out(h)


class Diffuser:
    """三线扩散器: L1/L2/L3 可热切换。"""

    def __init__(self, cfg=None, device=None, memory_pool=None):
        self.cfg = cfg if cfg is not None else config.get_config()
        self.device = device or self.cfg.device
        self.log = self.cfg.setup_logging().getChild("diffuser")
        self.memory = memory_pool            # 记忆池（L1 候选/L3 建图数据源）
        self.mode = "L2" if _HAS_TORCH else "L1"
        self.lock = threading.RLock()
        self._l2 = None
        self._l3 = None                      # CSR: (indptr, indices, values)
        self._history = deque(maxlen=8)      # 最近 soft 码序列
        self._load_l2()
        self._build_l3()

    # ==================================================================
    # L2 加载/训练
    # ==================================================================
    def _load_l2(self):
        if not _HAS_TORCH:
            return
        self._l2 = _L2Net(self.cfg.diffuser_l2_d_model, self.cfg.diffuser_l2_heads,
                          self.cfg.diffuser_l2_layers).to(self.device)
        if self.cfg.diffuser_quant != "fp32":
            self._l2.half()                 # fp16/nf4 模式（默认 fp32, 用户指定）
        if os.path.isfile(self.cfg.l2_diffuser_file):
            try:
                st = quant.load_state_atomic(self.cfg.l2_diffuser_file)
                if st.get("version") == "7.3":
                    sd = {}
                    for name, q in st["q"].items():
                        v = quant.dequantize_nf4(q.reshape(1, -1),
                                                 st["scale"][name].reshape(-1),
                                                 numel=st["numel"][name])
                        sd[name] = v.reshape(st["shape"][name])
                    self._l2.load_state_dict(sd, strict=True)
                    if self.cfg.diffuser_quant == "fp32":
                        self._l2.float()    # NF4 解量化 → fp32 常驻（用户指定 fp32）
                    self._l2.eval()
                    self.log.info("[DIFFUSER] L2 加载完成 (%s)",
                                  "FP32" if self.cfg.diffuser_quant == "fp32"
                                  else "FP16")
                    return
            except Exception:
                self.log.warning("[DIFFUSER] L2 损坏 → 重新训练")
        self.log.info("[DIFFUSER] L2 占位模型就绪 (未训练, 待 xilian_copy.json)")

    def _save_l2(self):
        if self._l2 is None:
            return
        st = {"version": "7.3", "q": {}, "scale": {}, "numel": {}, "shape": {}}
        for name, p in self._l2.state_dict().items():
            t = p.detach().float().reshape(1, -1)
            q, s = quant.quantize_nf4(t)
            st["q"][name] = q.reshape(-1).contiguous()
            st["scale"][name] = s.contiguous()
            st["numel"][name] = int(t.numel())
            st["shape"][name] = list(p.shape)
        quant.save_state_atomic(st, self.cfg.l2_diffuser_file)

    def train_l2(self, dataset_path: str = None, epochs: int = None):
        """L2 自回归训练: 文本语料 → 协议软码序列 → CE（语言建模转移场）。

        数据源 xilian_copy.json（顶层 list, 字段 xilian/user —— 昔涟语料）;
        训练完成后 NF4 落盘, 下次启动自动加载（trained 判定）。
        """
        if self._l2 is None:
            return False
        path = dataset_path or self.cfg.diffuser_dataset
        if not os.path.isfile(path):
            self.log.info("[DIFFUSER] 训练数据缺失 (%s) → 跳过 L2 训练", path)
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data.get("items", data) if isinstance(data, dict) else data
            texts = []
            for d in items:
                if not isinstance(d, dict):
                    continue
                t = (str(d.get("text") or d.get("xilian") or d.get("user") or "")
                     .strip())
                if t:
                    texts.append(t)
        except (OSError, ValueError):
            return False
        if not texts:
            self.log.warning("[DIFFUSER] 语料为空 → 跳过 L2 训练")
            return False
        epochs = epochs or self.cfg.diffuser_l2_epochs
        seqs = [self._text_to_soft_seq(t) for t in texts]
        seqs = [s for s in seqs if len(s) >= 2]
        if not seqs:
            return False
        # 防过拟合: 10% 验证集（seed 42 确定性切分）+ 早停(patience=2) + AdamW 权重衰减
        rng = np.random.RandomState(42)
        perm = rng.permutation(len(seqs))
        n_val = max(1, int(len(seqs) * 0.10))
        val_ids = set(perm[:n_val].tolist())
        train_seqs = [s for i, s in enumerate(seqs) if i not in val_ids]
        val_seqs = [s for i, s in enumerate(seqs) if i in val_ids]
        self._l2.train()
        self._l2.float()
        opt = torch.optim.AdamW(self._l2.parameters(), lr=self.cfg.diffuser_l2_lr,
                                weight_decay=1e-4)
        t0 = time.time()
        best_val, best_sd, patience = float("inf"), None, 0
        done_epochs = 0
        for ep in range(epochs):
            self._l2.train()
            rng.shuffle(train_seqs)
            total_loss = 0.0
            for s in train_seqs:
                x = torch.as_tensor(np.asarray(s[:-1], dtype=np.float32)
                                    .reshape(1, -1, 16), device=self.device)
                y = torch.as_tensor(np.asarray(s[1:], dtype=np.float32)
                                    .reshape(1, -1, 16), device=self.device)
                opt.zero_grad()
                logits = self._l2(x)
                loss = F.cross_entropy(logits.reshape(-1, 16), y.reshape(-1, 16))
                loss.backward()
                opt.step()
                total_loss += float(loss.item())
            train_loss = total_loss / max(1, len(train_seqs))
            self._l2.eval()
            with torch.no_grad():
                vloss = 0.0
                for s in val_seqs:
                    x = torch.as_tensor(np.asarray(s[:-1], dtype=np.float32)
                                        .reshape(1, -1, 16), device=self.device)
                    y = torch.as_tensor(np.asarray(s[1:], dtype=np.float32)
                                        .reshape(1, -1, 16), device=self.device)
                    logits = self._l2(x)
                    vloss += float(F.cross_entropy(
                        logits.reshape(-1, 16), y.reshape(-1, 16)).item())
                val_loss = vloss / max(1, len(val_seqs))
            done_epochs = ep + 1
            self.log.info("[DIFFUSER] L2 epoch %d/%d train=%.4f val=%.4f (%d/%d)",
                          done_epochs, epochs, train_loss, val_loss,
                          len(train_seqs), len(val_seqs))
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                best_sd = {k: v.clone()
                           for k, v in self._l2.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 2:
                    self.log.info("[DIFFUSER] 早停 (patience=2, best val=%.4f)",
                                  best_val)
                    break
        if best_sd is not None:
            self._l2.load_state_dict(best_sd)     # 恢复 best（非过拟合末轮）
        self._l2.eval()
        if self.cfg.diffuser_quant == "fp32":
            self._l2.float()                      # 扩散器 fp32 常驻（用户指定）
        else:
            self._l2.half()
        self._save_l2()
        self.log.info("[DIFFUSER] L2 训练完成 (%.1fs, best val=%.4f, %d epochs)",
                      time.time() - t0, best_val, done_epochs)
        return True

    def _text_to_soft_seq(self, text: str):
        """文本 → soft 码序列（规则编码 → code_to_soft）。"""
        code = self._rule_code(text)
        return [protocol.code_to_soft(c, 4.0) for c in code]

    @staticmethod
    def _rule_code(text: str):
        """规则编码: 文本 → 序列 of 16 位码（关键词 → 协议槽位）。"""
        codes = []
        from core.agi_core import _RULE_PATTERNS
        emo, sem = _RULE_PATTERNS(text)
        codes.append(protocol.encode(1, emo))          # 情感
        codes.append(protocol.encode(0, sem))          # 语义
        for ch in text[:8]:
            codes.append(protocol.encode(2, ord(ch) % 4096))
        return codes

    # ==================================================================
    # L3: CPU CSR（记忆码转移计数）
    # ==================================================================
    def _build_l3(self):
        """从记忆池/内存构建 4096 空间的转移计数 + 稀疏矩阵。"""
        n = self.cfg.diffuser_l3_codespace
        rows = {i: {} for i in range(n)}
        # 冷启动: 种子记忆内容 → 码序列转移
        mems = []
        if self.memory is not None:
            for m in getattr(self.memory, "all_disk_entries", lambda: [])()[:60]:
                mems.append(str(m.get("content", "")))
        for m in mems + []:
            codes = self._rule_code(m)
            for a, b in zip(codes, codes[1:]):
                k = (int(b) & 0xFFF) % n
                rows[(int(a) & 0xFFF) % n][k] = rows[(int(a) & 0xFFF) % n] \
                    .get(k, 0) + 1
        # CSR 三元组
        indptr = [0]
        indices, values = [], []
        for i in range(n):
            items = sorted(rows[i].items(), key=lambda kv: -kv[1])[:50]
            for k, v in items:
                indices.append(k)
                values.append(float(v))
            indptr.append(len(indices))
        self._l3 = (np.asarray(indptr, dtype=np.int64),
                    np.asarray(indices, dtype=np.int64),
                    np.asarray(values, dtype=np.float32))
        self.log.info("[DIFFUSER] L3 CSR 建图完成 (%d 非零边)", len(values))

    # ==================================================================
    # 训练状态判定（v7.3 版本戳, 过滤旧版遗留文件）
    # ==================================================================
    def is_trained(self) -> bool:
        """L2 是否已按 v7.3 训练（版本戳校验, 不被旧版遗留文件误判）。"""
        if not os.path.isfile(self.cfg.l2_diffuser_file):
            return False
        try:
            st = quant.load_state_atomic(self.cfg.l2_diffuser_file)
            return st.get("version") == "7.3"
        except Exception:
            return False

    # ==================================================================
    # 扩散接口
    # ==================================================================
    def set_mode(self, mode: str):
        if mode in ("L1", "L2", "L3"):
            self.mode = mode

    def diffuse(self, active_codes, k: int = 8, candidates=None) -> list:
        """活跃协议码 → 扩散候选 [(code, score), ...]。"""
        active_codes = [int(c) & 0xFFFF for c in active_codes]
        if self.mode == "L1":
            cands = candidates or self._l1_candidates()
            return _hamming_diffuse(active_codes, cands, k)
        if self.mode == "L3":
            return self._diffuse_l3(active_codes, k)
        return self._diffuse_l2(active_codes, k)

    def _l1_candidates(self) -> list:
        """候选码集: 协议槽位码 + 记忆协议码。"""
        cands = set()
        for t in range(5):
            for s in range(96):
                cands.add(protocol.encode(t, s))
        if self.memory is not None:
            for m in getattr(self.memory, "all_disk_entries", lambda: [])()[:200]:
                cands.add(protocol.encode(2, int(m.get("id", 0)) % 4096))
        return list(cands)

    def _diffuse_l2(self, active_codes, k: int) -> list:
        """L2 预测下一 soft 码 → 转 16 位码候选。"""
        if self._l2 is None or not active_codes:
            return []
        self._history.append(protocol.code_to_soft(active_codes[-1], 4.0))
        seq = np.asarray(list(self._history)[-8:], dtype=np.float32)
        pad = np.zeros((8 - seq.shape[0], 16), dtype=np.float32)
        seq = np.concatenate([pad, seq])[None]           # [1,8,16]
        with torch.no_grad():
            x = torch.as_tensor(seq,
                                dtype=(torch.float32 if self.cfg.diffuser_quant
                                       == "fp32" else torch.float16),
                                device=self.device)
            logits = self._l2(x).float()
            probs = torch.softmax(logits, dim=-1)[0, -1].cpu().numpy()
        code = protocol.soft_to_code(probs)
        return [(code, float(probs.max()))] + \
               _hamming_diffuse([code], self._l1_candidates(), k - 1)

    def _diffuse_l3(self, active_codes, k: int) -> list:
        """L3 CSR 行采样扩散。"""
        if self._l3 is None:
            return []
        indptr, indices, values = self._l3
        out = []
        for a in active_codes:
            r = int(a) & 0xFFF
            if r + 1 >= len(indptr):
                continue
            s, e = indptr[r], indptr[r + 1]
            if e <= s:
                continue
            idxs, vals = indices[s:e], values[s:e]
            top = np.argsort(-vals)[:k]
            out.extend([(int(indices[s + j]), float(values[s + j]))
                        for j in top])
        out.sort(key=lambda kv: -kv[1])
        return out[:k]

    # ==================================================================
    def get_stats(self) -> dict:
        return {"mode": self.mode,
                "l2_trained": os.path.isfile(self.cfg.l2_diffuser_file),
                "l3_edges": int(self._l3[2].size) if self._l3 else 0}
