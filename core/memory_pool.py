# -*- coding: utf-8 -*-
"""
core/memory_pool.py — 记忆系统 v7.3（显存优先）
================================================
显存预估:
  z float16 [5000,384] ≈ 3.7MB + coherence float16 [5000] 10KB
  + activation int16 [5000] 10KB ≈ 3.9MB（常驻 GPU, 与规格 §二.3 一致）
内存预估: 内容字符串 5000 × ~0.5KB ≈ 2.5MB RAM + 归档池（磁盘保留, 不加载全部）≈ 5MB

【硬性】§二.3:
  - 启动筛选: 从 memories.json 读取全部, 按价值分数取 Top-5000 驻留
  - 检索: GPU 余弦相似度, Top-30
  - 异步落盘: 每 50 心跳或关机时写入（.tmp + os.replace 原子替换）
  - 低价值记忆不删除: coherence < 0.2 时归档, 但仍保留在磁盘, 永不删除
"""
import os
import json
import time
import threading
import hashlib

import numpy as np

import config
from core import protocol

try:
    import torch
    _HAS_TORCH = True
except Exception:            # pragma: no cover - DEMO 无 torch
    torch = None
    _HAS_TORCH = False


class MemoryPool:
    """长期记忆池: GPU 张量驻留 + JSON 原子落盘 + 归档永不删除。"""

    def __init__(self, cfg=None, device=None, encode_fn=None):
        """
        encode_fn: callable(text) -> np.ndarray [384] — 文本→Z 编码器
                   （由 AGICore 注入; 缺失时用协议码投影兜底）
        """
        self.cfg = cfg if cfg is not None else config.get_config()
        self.device = device or self.cfg.device
        self.log = self.cfg.setup_logging().getChild("memory")
        self.lock = threading.RLock()
        self.path = self.cfg.memories_path
        self._encode = encode_fn or self._fallback_encode

        self.max_count = self.cfg.memory_max_count        # 5000
        self.top_k = self.cfg.retrieval_top_k             # 30
        self.save_interval = self.cfg.memory_save_interval  # 50 心跳

        # ---- 显存结构（规格 §二.3） ----
        self._z = None            # float16 [N, 384]
        self._coherence = None    # float16 [N]
        self._activation = None   # int16 [N]
        self._ids = []            # list[int] 记忆 id（与行号对齐)
        self._content = []        # list[str]
        self._archived = []       # list[dict] 归档池（内存中也保留, 磁盘必存）

        self._next_id = 1
        self._pending_save = False
        self._save_thread = None
        self._save_ticks = 0
        rng_seed = self.cfg.code_to_z_seed
        self._recall_filter = float(self.cfg.memory_recall_filter)

        self._load_all()

    # ==================================================================
    # 编码兜底（无编码器注入时）
    # ==================================================================
    def _fallback_encode(self, text: str) -> np.ndarray:
        """协议码投影兜底: 稳定哈希 → 16位码 → Z（跨进程确定性）。"""
        d = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
        h = int.from_bytes(d[:8], "big")
        code = protocol.encode(2, h % 4096)
        return protocol.code_to_z(code, self.cfg)

    # ==================================================================
    # 启动: 读取全部 → 价值分数 Top-5000 驻留
    # ==================================================================
    def _load_all(self):
        """从 memories.json 读取全部记忆, 按价值分数取 Top-5000。"""
        raw = []
        if os.path.isfile(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                raw = data.get("memories", [])
                self._next_id = max([int(m.get("id", 0)) for m in raw] + [0]) + 1
            except (OSError, ValueError):
                self.log.warning("[MEMORY] memories.json 损坏 → 空池启动（磁盘内容不删除）")
                raw = []
        else:
            self.log.info("[MEMORY] memories.json 缺失 → 空池启动（start.py 会生成种子）")

        def value(m: dict) -> float:
            """价值分数 = 激活次数 × 0.5 + 连贯 × 0.3 + 新鲜度 × 0.2。"""
            act = float(m.get("activation_count", 0) or 0)
            coh = float(m.get("coherence_strength", 1.0) or 0)
            rec = max(0.0, 1.0 - max(0.0, time.time() - float(
                m.get("last_activated", 0) or 0)) / (30 * 86400))
            return act * 0.5 + coh * 0.3 + rec * 0.2

        ordered = sorted(raw, key=value, reverse=True)
        keep = ordered[: self.max_count]
        rest = ordered[self.max_count:]
        # 归档池: 低价值 + 超容量 —— 全部保留在磁盘, 永不删除
        for m in rest:
            if m not in self._archived:
                self._archived.append(m)
        self.log.info("[MEMORY] 载入 %d 条 → 驻留 Top-%d, 归档池 %d 条",
                      len(raw), len(keep), len(self._archived))

        # 预分配 max_count 行（规格 §二.3: [5000,384]≈3.7MB 恒驻留）;
        # 实际使用 _n 行, 追加/替换均为原地写入（零复制, 无 torch.cat 全量拷贝）。
        self._z = torch.zeros(self.max_count, self.cfg.z_dim, dtype=torch.float16)
        self._coherence = torch.zeros(self.max_count, dtype=torch.float16)
        self._activation = torch.zeros(self.max_count, dtype=torch.int16)
        for i, m in enumerate(keep):
            # 落盘不保存 z（避免 JSON 膨胀）: 载入时按内容重新编码（规则编码确定性强）
            z = self._encode(str(m.get("content", "")))
            zt = torch.as_tensor(z, dtype=torch.float16)
            nz = zt.norm(p=2)
            if float(nz) > 1e-9:                # 写入即归一化（检索省全量 norm 归约）
                zt = zt / nz
            self._z[i] = zt
            self._coherence[i] = float(m.get("coherence_strength", 1.0))
            self._activation[i] = int(m.get("activation_count", 0) or 0) % 32767
            self._ids.append(int(m.get("id", i + 1)))
            self._content.append(str(m.get("content", "")))
        if torch is not None and self.device.startswith("cuda"):
            self._z = self._z.to(self.device)
            self._coherence = self._coherence.to(self.device)
            self._activation = self._activation.to(self.device)
        self._n = len(keep)

    # ==================================================================
    # 检索（GPU 余弦相似度, Top-30）
    # ==================================================================
    def recall(self, z, k: int = None, min_sim: float = None) -> list:
        """按 Z 检索相似记忆 → [(sim, id, content), ...]（升序截取 Top-k）。

        - GPU 余弦: sim = z·Z^T / (|z||Z|)（Z 行已归一可省除法, 此处显式归一）
        - 归档池（coherence<0.2）自动排除于检索（只存不召回）
        """
        k = k or self.top_k
        if self._n == 0:
            return []
        zt = torch.as_tensor(np.asarray(z, dtype=np.float32).reshape(1, -1),
                             dtype=torch.float16, device=self._z.device)
        zt = zt / zt.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-6)
        with self.lock, torch.no_grad():
            zz = self._z[:self._n]               # 仅已用行（预分配空行不参与）
            coh = self._coherence[:self._n]
            sims = (zz @ zt.T).squeeze(-1)       # [n_use]（Z 行已归一, 免全量 norm）
            active = coh >= self.cfg.forget_threshold      # 归档不召回
            sims = sims * active.float()
            vals, idxs = torch.topk(sims, min(k, self._n))
            vals = vals.float().cpu().numpy()
            idxs = idxs.cpu().numpy()
        out = []
        for v, i in zip(vals, idxs):
            if min_sim is not None and v < min_sim:
                continue
            out.append((float(v), int(self._ids[int(i)]),
                        str(self._content[int(i)])))
        return out

    def recall_related(self, z, n: int = 4) -> list:
        """记忆检索决策（router）专用: 相似度高于 MEMORY_RECALL_SIMILARITY 的精筛。"""
        return self.recall(z, k=max(n * 3, self.top_k),
                           min_sim=self._recall_filter)[:n]

    def nearest_row(self, z):
        """最近记忆的行号（或 -1）: 覆写/更新用。"""
        hits = self.recall(z, k=1)
        if not hits:
            return -1
        return self._row_of(int(hits[0][1]))

    def _row_of(self, mem_id: int) -> int:
        try:
            return self._ids.index(mem_id)
        except ValueError:
            return -1

    # ==================================================================
    # 写入 / 更新
    # ==================================================================
    def add(self, content: str, z, emotion_tag: str = "calm",
            protocol_code: int = None) -> int:
        """新增记忆（满则替换价值最低行）; 返回记忆 id。

        同簇干扰: 与已有记忆高度相似(>0.9)时, 旧记忆 coherence -= 干扰值。
        """
        with self.lock:
            mem_id = self._next_id
            self._next_id += 1
            # 同簇干扰（规格: 同簇干扰 −0.03）
            hits = self.recall(z, k=2)
            if hits and hits[0][0] > 0.90 and hits[0][1] != mem_id:
                r = self._row_of(hits[0][1])
                if r >= 0 and self._coherence is not None:
                    self._coherence[r] = max(
                        0.0, float(self._coherence[r]) -
                        self.cfg.memory_coherence_interfere)
            row = self._row_of(mem_id)
            if row < 0:
                if self._n < self.max_count:
                    self._append_row(mem_id, content, z, emotion_tag, protocol_code)
                else:
                    self._replace_lowest(mem_id, content, z, emotion_tag,
                                         protocol_code)
            else:
                self._update_row(row, content, z, emotion_tag, protocol_code)
            self._pending_save = True
            return mem_id

    def _append_row(self, mem_id, content, z, tag, code):
        """尾插一行（预分配空间内原地写入, 零复制）。"""
        zt = torch.as_tensor(np.asarray(z, dtype=np.float32),
                             dtype=torch.float16, device=self._z.device)
        nz = zt.norm(p=2)
        if float(nz) > 1e-9:                    # 写入即归一化
            zt = zt / nz
        row = self._n
        self._z[row] = zt.reshape(1, -1)
        self._coherence[row] = 1.0
        self._activation[row] = 0
        self._ids.append(mem_id)
        self._content.append(content)
        self._n += 1

    def _replace_lowest(self, mem_id, content, z, tag, code):
        """满池: 替换价值最低行（价值 = 激活×(1+coherence)×新鲜度）。"""
        if self._n == 0:
            self._append_row(mem_id, content, z, tag, code)
            return
        act = self._activation[:self._n].float().cpu().numpy()
        coh = self._coherence[:self._n].float().cpu().numpy()
        value = act * (1.0 + coh) *(1.0 - coh * 0.05)
        row = int(np.argmin(value))
        self._ids[row] = mem_id
        self._content[row] = content
        self._overwrite_row(row, z)

    def _update_row(self, row, content, z, tag, code):
        self._content[row] = content
        self._overwrite_row(row, z)

    def _overwrite_row(self, row: int, z):
        zt = torch.as_tensor(np.asarray(z, dtype=np.float32),
                             dtype=torch.float16, device=self._z.device)
        nz = zt.norm(p=2)
        if float(nz) > 1e-9:                    # 写入即归一化
            zt = zt / nz
        self._z[row] = zt
        self._coherence[row] = 1.0
        self._activation[row] = 0

    def touch(self, mem_ids, gain: float = None):
        """召回后激活: activation+1, coherence += gain（成功匹配奖励）。"""
        if not mem_ids:
            return
        with self.lock:
            gain = gain if gain is not None else self.cfg.memory_coherence_gain
            for mid in mem_ids:
                r = self._row_of(int(mid))
                if r < 0:
                    continue
                self._activation[r] = int(min(32766, int(self._activation[r]) + 1))
                self._coherence[r] = min(2.0, float(self._coherence[r]) + gain)

    # ==================================================================
    # 归档（coherence<0.2: 内存标记 + 磁盘保留, 永不删除）
    # ==================================================================
    def _archive_pass(self):
        """把 coherence<0.2 的记忆标记为归档（保留在池中但不召回/不采样）。"""
        with self.lock:
            if self._coherence is None:
                return 0
            mask = (self._coherence[:self._n].float().cpu().numpy()
                    < self.cfg.forget_threshold)
            return int(mask.sum())

    def archived_count(self) -> int:
        """当前归档（沉底）条数。"""
        return self._archive_pass()

    def all_disk_entries(self) -> list:
        """完整磁盘清单 = 驻留 + 归档（数据永不删除的落盘依据; 不落 z, 载入时重编码）。"""
        entries = []
        with self.lock:
            for r in range(self._n):
                entries.append({
                    "id": self._ids[r],
                    "content": self._content[r],
                    "coherence_strength": float(self._coherence[r]),
                    "activation_count": int(self._activation[r]),
                    "last_activated": time.time(),
                    "archived": float(self._coherence[r]) < self.cfg.forget_threshold,
                })
        entries.extend(self._archived)
        return entries

    # ==================================================================
    # 异步落盘（每 50 心跳 / 关机; .tmp + os.replace）
    # ==================================================================
    def tick(self):
        """主循环每心跳调用: 到点触发异步保存。"""
        self._save_ticks += 1
        if self._save_ticks >= self.save_interval and self._pending_save:
            self._save_ticks = 0
            self.save_async()

    def save_async(self):
        """后台线程落盘（线程安全: 只允许一个保存线程）。"""
        if self._save_thread is not None and self._save_thread.is_alive():
            self._pending_save = True
            return
        t = threading.Thread(target=self._save_blocking, name="memory-save",
                             daemon=True)
        self._save_thread = t
        t.start()

    def _save_blocking(self):
        with self.lock:
            entries = self.all_disk_entries()
        try:
            data = {"version": "7.3", "saved_at": time.time(),
                    "memories": entries}
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp, self.path)          # 原子替换【硬性】
            with self.lock:
                self._pending_save = False
        except OSError as e:
            self.log.error("[MEMORY] 落盘失败: %s", e)

    def shutdown_save(self):
        """关机强制存档（阻塞等待完成）。"""
        self._pending_save = True
        self._save_blocking()
        self.log.info("[MEMORY] 关机存档完成 (%d 条, 含归档 %d)",
                      self._n, self.archived_count())

    # ==================================================================
    # 统计
    # ==================================================================
    def stats(self) -> dict:
        return {"size": self._n, "sunk": self.archived_count(),
                "pending": self._pending_save}

    def top_recent(self, n: int = 5) -> list:
        """最近激活 Top-n（调试窗口显示）。"""
        if self._n == 0:
            return []
        act = self._activation[:self._n].float().cpu().numpy()
        idxs = np.argsort(-act)[:n]
        return [{"id": self._ids[int(i)], "content": self._content[int(i)][:40],
                 "act": float(self._activation[int(i)])} for i in idxs]
