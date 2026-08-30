# -*- coding: utf-8 -*-
"""
core/emotion_loop.py — 20 个 10M 情感回环模型（【硬性】 §三）
============================================================
VRAM: ~0.1GB（20 × 9.52M 参数 × 4-bit NF4 ≈ 95MB，全部常驻显存）。
本模块不加载 Qwen；输出 8 维"下一情感向量"（config.emotion_names）。

架构: 每个回环 = 3 层 MLP [16→3072→3072→8]（9,520,136 ≈ 10M 参数）。
  - 输入 [情感向量(8) ‖ 匹配器摘要(8)] = 16 维
  - 输出 8 维情感向量（joy/sadness/anger/fear/calm/trust/pain/loss）

分类（硬性）: 积极×5(id 0-4)、消极×5(5-9)、记忆×5(10-14)、随机×5(15-19)。
主导切换（硬性）: 每 60 秒切换主导权重 —— 本轮循环选择一类作为主导类别，
  主导类别内的某个回环获得更高输出权重（其余均分），实现"情绪转调"。

存储/前向: 与 MatcherCluster 相同 —— 每类堆叠 NF4 打包张量常驻显存，
  前向按 5 个回环一组解量化（块内 fp16 瞬态 ≈ 95MB），峰值受控。
"""
import os
import time
import threading

import numpy as np
import torch
import torch.nn.functional as F

import config
from core import quant


class EmotionLoops:
    """20 情感回环集群: 生成/加载 → 主导权重 → 情感步进（step）→ 状态查询。"""

    def __init__(self, cfg=None, device=None):
        self.cfg = cfg if cfg is not None else config.get_config()
        self.device = device or self.cfg.device
        self.log = config.get_config().setup_logging().getChild("emotion")
        self.lock = threading.RLock()
        self.arch = list(self.cfg.loop_arch)
        self._stack = {}          # layer → (packed (20,bytes), absmax (20,), numel)
        self._fp16 = {}           # 【性能】预解量化 fp16 常驻: layer → w(20,out,in)/b(20,out)
        self._categories = {}     # id → 类别名
        # 无条件积极回环（德缪歌锚点·情感侧）: positive 类前 positive_loops_locked 个, 锁定
        plo, phi = self.cfg.loop_categories["positive"]
        self.locked_ids = list(range(plo, min(plo + self.cfg.positive_loops_locked, phi)))
        self._prepare_dir()
        self._load_or_generate()
        self._emotion = np.asarray(self.cfg.emotion_baseline, dtype=np.float32)
        self._dom_weights = np.full(self.cfg.loop_count, 1.0 / self.cfg.loop_count,
                                    dtype=np.float32)
        self._dom_epoch = -1
        self._dom_meta = {"category": "随机", "loop": -1}

    # ------------------------------------------------------------------
    # 路径与生成
    # ------------------------------------------------------------------
    def _prepare_dir(self):
        os.makedirs(self.cfg.emotion_loops_dir, exist_ok=True)

    def loop_path(self, loop_id: int) -> str:
        return os.path.join(self.cfg.emotion_loops_dir, f"loop_{loop_id}_nf4.pt")

    def _category_of(self, loop_id: int) -> str:
        """回环 id → 类别（积极/消极/记忆/随机）。"""
        for cat, (lo, hi) in self.cfg.loop_categories.items():
            if lo <= loop_id < hi:
                return cat
        return "随机"

    def _build_default_loop(self, loop_id: int) -> dict:
        """默认（随机初始化）回环并原子落盘（占位模型生成逻辑）。

        无条件积极回环（locked_ids）: 输出层 bias 抬高 joy/trust 通道（+1.5），
        使输出天然偏向"温暖/接纳"方向 —— 与锚点匹配器共同构成人格恒定锚（永不更新）。
        """
        rng = torch.Generator().manual_seed(0x1000 + loop_id)
        params = {}
        for i in range(len(self.arch) - 1):
            wshape = (self.arch[i + 1], self.arch[i])
            bound = 1.0 / np.sqrt(self.arch[i])
            params[f"w{i}"] = (torch.rand(wshape, generator=rng, dtype=torch.float16)
                               * (2 * bound) - bound)
            params[f"b{i}"] = torch.zeros(self.arch[i + 1], dtype=torch.float16)
        if loop_id in self.locked_ids:
            out_dim = self.arch[-1]
            bias = torch.zeros(out_dim, dtype=torch.float16)
            for ch, name in enumerate(self.cfg.emotion_names):
                if name in ("joy", "trust"):
                    bias[ch] = 1.5          # 无条件偏积极/接纳
            params["b2"] = bias
        st = {"q": {}, "scale": {}, "numel": {}}
        for layer, t in params.items():
            q, s = quant.quantize_nf4(t.reshape(1, -1).float())
            st["q"][layer] = q.reshape(-1)
            st["scale"][layer] = s
            st["numel"][layer] = int(t.numel())
        st["id"] = loop_id
        st["category"] = self._category_of(loop_id)
        st["arch"] = list(self.arch)
        st["locked"] = loop_id in self.locked_ids
        quant.save_state_atomic(st, self.loop_path(loop_id))
        return st

    def _load_or_generate(self):
        """加载/生成 20 个回环并堆叠（常驻显存）。"""
        t0 = time.time()
        rows = {}
        for i in range(self.cfg.loop_count):
            st = None
            if os.path.isfile(self.loop_path(i)):
                try:
                    loaded = quant.load_state_atomic(self.loop_path(i))
                    if loaded.get("arch") == self.arch and "q" in loaded:
                        st = loaded
                except Exception:
                    st = None
            if st is None:
                st = self._build_default_loop(i)
            # 旧文件无 locked 标记且属于锁定集 → 重建（锚点升级识别）
            if i in self.locked_ids and not st.get("locked"):
                st = self._build_default_loop(i)
            for layer in st["q"]:
                rows.setdefault(layer, []).append(
                    (st["q"][layer], st["scale"][layer], st["numel"][layer]))
            self._categories[i] = st.get("category", self._category_of(i))
        for layer, lst in rows.items():
            self._stack[layer] = (torch.stack([r[0] for r in lst]).contiguous(),
                                  torch.stack([r[1] for r in lst]).reshape(-1).contiguous(),
                                  lst[0][2])
        if self.device.startswith("cuda"):
            for layer, (p, s, n) in self._stack.items():
                self._stack[layer] = (p.to(self.device), s.to(self.device), n)
        # 【性能】一次性解量化为 fp16 常驻（每心跳 step 不再重复解量化 2 亿个参数）
        if self._stack:
            t1 = time.time()
            for layer, (packed, scale, numel) in self._stack.items():
                if layer.startswith("w"):
                    i = int(layer[1:])
                    w = quant.dequantize_rows(packed, scale, numel)
                    self._fp16[layer] = w.reshape(self.cfg.loop_count,
                                                  *self._stack_shape_w(i))
                else:
                    b = quant.dequantize_rows(packed, scale, numel)
                    self._fp16[layer] = b.reshape(self.cfg.loop_count, -1)
            self.log.info("[EMOTION] 回环权重预解量化 fp16 常驻 (%.1fs, ≈%.0fMB)",
                          time.time() - t1,
                          sum(v.numel() for v in self._fp16.values()) * 2 / 1048576)
        self.log.info("[EMOTION] 20 个回环加载完成 (%.1fs, 堆叠 %s)", time.time() - t0,
                      {k: tuple(v[0].shape) for k, v in self._stack.items()})

    # ------------------------------------------------------------------
    # 主导权重（每 60 秒切换）
    # ------------------------------------------------------------------
    def _update_dominant(self, now=None):
        """按 60 秒周期切换主导回环（类别轮转 + 类别内随机）。

        权重设计: 主导类别的选定回环 0.65，其余 19 个均分 0.35，
          使输出以"主导情绪调性"为主、保留余量产生多声部叠加。
        """
        epoch = int((now if now is not None else time.time())
                    // self.cfg.loop_dominant_switch_s)
        if epoch == self._dom_epoch:
            return
        self._dom_epoch = epoch
        cats = list(self.cfg.loop_categories.keys())
        cat = cats[epoch % len(cats)]                       # 类别轮转
        lo, hi = self.cfg.loop_categories[cat]
        dom = int(np.random.randint(lo, hi))
        w = np.full(self.cfg.loop_count, 0.35 / (self.cfg.loop_count - 1), dtype=np.float32)
        w[dom] = 0.65
        # 德缪歌锚点·情感侧: 无条件积极回环权重保底（任何主导类别下都保留"温暖/接纳"底色）
        for lid in self.locked_ids:
            w[lid] = max(w[lid], self.cfg.positive_loop_min_weight)
        w = w / w.sum()                                     # 重归一（保持概率分布）
        self._dom_weights = w
        self._dom_meta = {"category": {"positive": "积极", "negative": "消极",
                                       "memory": "记忆", "random": "随机"}.get(cat, cat),
                          "loop": dom}
        self.log.info("[EMOTION] 主导切换 → 类别=%s 回环#%d (锚点回环权重=%.3f)",
                      self._dom_meta["category"], dom, float(w[self.locked_ids].mean()))

    # ------------------------------------------------------------------
    # 情感步进（每心跳）
    # ------------------------------------------------------------------
    def step(self, matcher_summary: np.ndarray, emotion: np.ndarray = None,
             dt: float = 1.0) -> np.ndarray:
        """情感回环步进: (当前情感, 匹配器摘要) → 下一情感向量（CPU ndarray (8,)）。

        流程:
          1. 更新主导权重（60s 周期）
          2. 前向全部 20 个回环（分块解量化，no_grad）
          3. 按主导权重加权混合 → 目标情感
          4. 残差融合（loop_emotion_decay）+ 漂移钳制（emotion_drift_limit）
        """
        self._update_dominant()
        em = np.asarray(emotion if emotion is not None else self._emotion,
                        dtype=np.float32).reshape(-1)
        summary = np.asarray(matcher_summary, dtype=np.float32).reshape(-1)[:8]
        x = np.concatenate([em, summary]).astype(np.float32)   # 16 维
        with self.lock, torch.no_grad():
            xt = torch.as_tensor(x, dtype=torch.float16, device=self.device).unsqueeze(0)
            out_chunks = []
            n = self.cfg.loop_count
            for off in range(0, n, self.cfg.loop_chunk):
                size = min(self.cfg.loop_chunk, n - off)
                h = xt.unsqueeze(1).expand(1, size, -1)         # (1, size, 16)
                for i in range(len(self.arch) - 1):
                    w = self._fp16[f"w{i}"][off:off + size]     # (size, out, in) 已解量化
                    # einsum 处理 批量×回环 双维矩阵乘（bmm 不广播批维）
                    h = torch.einsum("bki,koi->bko", h, w)
                    b = self._fp16[f"b{i}"][off:off + size]     # (size, out) 已解量化
                    h = h + b.unsqueeze(0)
                    if i < len(self.arch) - 2:
                        h = F.relu(h)
                out_chunks.append(h.squeeze(-1))               # (1, size, 8)
            outs = torch.cat(out_chunks, dim=1).squeeze(0)     # (20, 8)
            weights = torch.as_tensor(self._dom_weights, dtype=torch.float16,
                                      device=self.device)
            mix = (outs * weights.unsqueeze(1)).sum(dim=0).float().cpu().numpy()
        # 残差融合 + 钳制（情感不突变: 每步最多漂移 drift_limit）
        blended = (self.cfg.loop_emotion_decay * em
                   + (1.0 - self.cfg.loop_emotion_decay) * mix)
        lim = float(self.cfg.emotion_drift_limit) * dt
        new_em = np.clip(em + np.clip(blended - em, -lim, lim), 0.0, 1.0)
        self._emotion = new_em.astype(np.float32)
        return self._emotion.copy()

    def _stack_shape_w(self, i: int) -> tuple:
        """第 i 层权重的 (out, in) 形状（解量化后 reshape 用）。"""
        return (self.arch[i + 1], self.arch[i])

    # ------------------------------------------------------------------
    # 查询/状态（调试窗口用）
    # ------------------------------------------------------------------
    def current_emotion(self) -> np.ndarray:
        """当前情感向量（CPU 拷贝）。"""
        return self._emotion.copy()

    def dominant_info(self) -> dict:
        """当前主导类别/回环（调试窗口显示）。"""
        return dict(self._dom_meta)

    def baseline(self) -> np.ndarray:
        return np.asarray(self.cfg.emotion_baseline, dtype=np.float32)
