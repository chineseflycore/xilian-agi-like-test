# -*- coding: utf-8 -*-
"""
memory.py — 记忆系统（三层结构，规范 §4.3）
===========================================
内存/显存预估（规范 §九.1）:
  - FAISS 索引: 10000 条 × 512 维 × 4B ≈ 20MB（CPU RAM）
  - numpy 降级索引（无 faiss 时）: 同上 ≈ 20MB
  - 磁盘压缩导出: faiss_path 下 .npz/.jsonl 文件（按需）
  - 显存: 0MB —— 记忆系统强制 CPU，不占用 GPU 显存（规范 §二）

三层结构（规范 §4.3）:
  第一层: FAISS 短期记忆（可删除）—— 最近对话状态向量，上限 cfg.max_faiss_entries=10000，
          超限压缩: 最旧 50% 导出磁盘并从索引移除（规范 §4.3 / §九.10）
  第二层: 扩散器连接权重（长期记忆/本能）—— 同一模式被成功检索使用 > cfg.migration_threshold(10)
          次 → 写入 L3 扩散器权重，之后不再检索直接响应（规范 §4.3 第二层）
  第三层: 随机召回层（自发激活触发器）—— 静默期随机抽取 1~3 条注入扩散器，
          产生“乱码”→ 感性模块判定 → 主动输出（规范 §4.3 第三层 / §十.26）

接口（line_controller.py / server.py / formatting.py 调用）:
  add(vector, text, emotion) -> id
  search(query, k=3) -> dict {"vectors","ids","similarities"}（兼容多种返回形态）
  random_recall(k=3) -> [{"vector","text","id","similarity"}]
  clear() / stats() / compress_if_needed() / persist() / load()

降级链（规范 §九.2）:
  faiss 未安装 → numpy 暴力余弦检索（接口一致）；faiss_path 不可写 → 内存运行。
"""

# 环境引导: 脚本目录与 vendor 入 sys.path（嵌入式 Python 隔离模式必需；常规 CPython no-op）
import os as _os, sys as _sys
_BASE = _os.path.dirname(_os.path.abspath(__file__))
if _BASE not in _sys.path:
    _sys.path.insert(0, _BASE)
try:
    import numpy  # noqa: F401
except ImportError:
    _V = _os.path.join(_BASE, "vendor")
    if _os.path.isdir(_V) and _V not in _sys.path:
        _sys.path.insert(0, _V)

import os
import json
import time

import numpy as np

import config
from common_utils import get_logger, Timer, safe_import, normalize, cosine, stable_hash

# 迁移计数阈值默认 10 次（规范 §4.3 第二层，可配置 cfg.migration_threshold）
_DEFAULT_MIGRATION = 10


class MemorySystem:
    """三层记忆系统: FAISS 短期记忆 + 迁移阈值 + 随机召回层。"""

    def __init__(self, cfg=None, translator=None, l3=None):
        self.cfg = cfg if cfg is not None else config.get_config()
        self.log = get_logger("memory")
        self.translator = translator      # 语义翻译器（写入向量用；可选）
        self.l3 = l3                      # L3 扩散器（长期记忆/本能写入目标；可选）
        self.dim = self.cfg.STATE_DIM

        # ---- 存储 ----
        self._vectors = None              # numpy (n, 512) float32（faiss 与降级共用来源）
        self._texts = []                  # 每条记忆的原文
        self._emotions = []               # 每条记忆的 8 维情绪向量（可选）
        self._timestamps = []             # 写入时间戳（压缩时按最旧导出）
        self._use_count = {}              # 检索使用计数（迁移阈值判定）
        self._migrated = set()            # 已迁移到 L3 的记忆 id（不再参与检索加权）

        self._faiss = None                # faiss.IndexFlatIP（惰性初始化）
        self._try_init_faiss()
        self.load()                       # 启动时从磁盘恢复（幂等）

    # ================================================================
    # FAISS 初始化（降级链）
    # ================================================================
    def _try_init_faiss(self):
        """尝试初始化 FAISS（IndexFlatIP 内积 ≈ 余弦，向量已归一化）。
        失败 → numpy 暴力检索降级（接口一致，规范 §九.2）。"""
        try:
            faiss = safe_import("faiss")
            if faiss is None:
                raise ImportError("faiss 未安装")
            if self._vectors is None:
                self._vectors = np.zeros((0, self.dim), dtype=np.float32)
            self._faiss = faiss.IndexFlatIP(self.dim)
            self.log.info("[MEM] FAISS IndexFlatIP(dim=%d) 就绪 (CPU)", self.dim)
        except Exception as e:
            self._faiss = None
            self.log.warning("[MEM] FAISS 不可用(%s)，降级 numpy 暴力检索", e)

    # ================================================================
    # 写入
    # ================================================================
    def add(self, vector, text="", emotion=None, meta=None) -> int:
        """写入一条记忆，返回 id。

        向量自动归一化（FAISS 内积 ≈ 余弦）；超限自动压缩（规范 §九.10）。
        """
        try:
            v = np.asarray(vector, dtype=np.float32).reshape(-1)
            if v.size != self.dim:
                self.log.warning("[MEM] 向量维度 %d ≠ %d，拒绝写入", v.size, self.dim)
                return -1
            v = normalize(v)
            if self._vectors is None:
                self._vectors = v.reshape(1, -1)
            else:
                self._vectors = np.vstack([self._vectors, v.reshape(1, -1)])
            self._texts.append(str(text or ""))
            self._emotions.append(None if emotion is None else np.asarray(emotion, dtype=np.float32))
            self._timestamps.append(time.time())
            idx = len(self._texts) - 1
            if self._faiss is not None:
                self._faiss.add(v.reshape(1, -1))
            self.compress_if_needed()
            return idx
        except Exception as e:
            self.log.error("[MEM] 写入失败: %s", e, exc_info=True)
            return -1

    # ================================================================
    # 检索（第一层 + 迁移计数，规范 §4.3）
    # ================================================================
    def search(self, query, k: int = 3) -> dict:
        """检索 top-k 相似记忆。

        返回 dict {"vectors": np.ndarray(k,512), "ids": [int], "similarities": [float]}
        （line_controller._parse_memory_result 兼容该形态；也兼容 tuple/ndarray）。
        命中记忆的使用计数 +1；计数 ≥ 迁移阈值 → 尝试写入 L3 长期记忆（规范 §4.3 第二层）。
        """
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        if q.size != self.dim:
            q = normalize(np.resize(q, self.dim) if q.size else np.zeros(self.dim, np.float32))
        q = normalize(q)
        n = len(self._texts)
        if n == 0:
            return {"vectors": np.zeros((0, self.dim), np.float32), "ids": [], "similarities": []}

        with Timer("memory.search") as t:
            if self._faiss is not None:
                sims, ids = self._faiss.search(q.reshape(1, -1), min(k, n))
                sims = sims[0].astype(np.float32)
                ids = ids[0].astype(np.int64).tolist()
            else:
                # numpy 降级: 余弦暴力检索
                sims_all = np.asarray(
                    [cosine(q, self._vectors[i]) for i in range(n)], dtype=np.float32)
                order = np.argsort(-sims_all)[:k]
                ids = order.tolist()
                sims = sims_all[order]

        vecs = np.stack([self._vectors[i] for i in ids]) if ids else np.zeros((0, self.dim), np.float32)
        sims = np.clip(np.asarray(sims, dtype=np.float32), -1.0, 1.0)
        for i, sid in enumerate(ids):
            if sims[i] >= 0.3:           # 低相似度命中不计数（避免噪声迁移）
                self._bump_usage(sid)
        self.log.info("[MEM] 检索 k=%d 命中=%d 耗时=%.1fms", k, len(ids), t.elapsed)
        return {"vectors": vecs, "ids": ids, "similarities": sims.tolist()}

    def _bump_usage(self, sid: int):
        """检索使用计数 +1；达到迁移阈值 → 写入 L3 长期记忆（规范 §4.3 第二层）。"""
        if sid in self._migrated:
            return
        self._use_count[sid] = self._use_count.get(sid, 0) + 1
        threshold = int(getattr(self.cfg, "migration_threshold", _DEFAULT_MIGRATION))
        if self._use_count[sid] >= threshold:
            self._migrate_to_l3(sid)

    def _migrate_to_l3(self, sid: int):
        """同一模式被成功检索使用 ≥ 阈值次数 → 写入 L3 连接权重（本能固化）。"""
        if self.l3 is None or sid >= len(self._texts):
            return
        self._migrated.add(sid)
        try:
            vec = self._vectors[sid]
            if hasattr(self.l3, "hebbian_update"):
                self.l3.hebbian_update(vec, reward=1.0)
                self.log.info("[MEM] 记忆 #%d 已迁移写入 L3 长期记忆（使用≥%d 次）",
                              sid, getattr(self.cfg, "migration_threshold", _DEFAULT_MIGRATION))
        except Exception as e:
            self.log.warning("[MEM] L3 迁移失败: %s", e)

    # ================================================================
    # 随机召回层（第三层，自发激活触发器，规范 §4.3 / §十.26）
    # ================================================================
    def random_recall(self, k: int = None) -> list:
        """静默期随机抽取 1~k 条记忆作为自发激活触发信号。

        返回 [{"vector": (512,), "text": str, "id": int, "similarity": float}]。
        无记忆 → 返回空列表（禁止索引空列表，规范 §3.1 同类约束）。
        """
        n = len(self._texts)
        k = int(k or getattr(self.cfg, "random_recall_count", 3))
        if n == 0:
            self.log.info("[RECALL] 无记忆可召回，返回空列表")
            return []
        k = max(1, min(k, n))
        rng = np.random.RandomState(int(time.time()) % (2 ** 31))
        picked = rng.choice(n, size=k, replace=False)
        out = []
        for i in picked.tolist():
            out.append({
                "vector": self._vectors[i].copy(),
                "text": self._texts[i],
                "id": int(i),
                "similarity": 1.0,
            })
        self.log.info("[RECALL] 随机召回 %d 条（自发激活触发信号）", k)
        return out

    # ================================================================
    # 超限压缩（规范 §4.3 第一层 / §九.10）
    # ================================================================
    def compress_if_needed(self):
        """条目超上限（cfg.max_faiss_entries=10000）→ 最旧 50% 导出磁盘并从索引移除。"""
        cap = int(getattr(self.cfg, "max_faiss_entries", 10000))
        if len(self._texts) <= cap:
            return
        n_drop = len(self._texts) // 2
        # 最旧 n_drop 条（按写入时间）
        order = np.argsort(self._timestamps)
        drop_ids = sorted(order[:n_drop].tolist())
        self._export_oldest(drop_ids)
        self._remove_ids(drop_ids)
        self.log.warning(
            "[MEM] 超上限 %d，已压缩: 导出最旧 %d 条到磁盘并从索引移除", cap, n_drop)

    def _export_oldest(self, ids):
        """把指定记忆导出到磁盘（faiss_path 下 npz/jsonl，路径基于 BASE_DIR）。"""
        try:
            os.makedirs(self.cfg.faiss_path, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            arr = np.stack([self._vectors[i] for i in ids]) if ids else np.zeros((0, self.dim), np.float32)
            meta = [{"id": i, "text": self._texts[i],
                     "emotion": (self._emotions[i].tolist()
                                 if self._emotions[i] is not None else None),
                     "timestamp": self._timestamps[i]} for i in ids]
            np.savez_compressed(
                os.path.join(self.cfg.faiss_path, f"compressed_{ts}.npz"),
                vectors=arr, meta=json.dumps(meta, ensure_ascii=False))
            self.log.info("[MEM] 已导出 %d 条记忆 → %s", len(ids), self.cfg.faiss_path)
        except Exception as e:
            self.log.warning("[MEM] 导出失败（压缩仅内存执行）: %s", e)

    def _remove_ids(self, ids):
        """从全部存储结构中移除指定 id（重建索引）。"""
        keep = sorted(set(range(len(self._texts))) - set(ids))
        if not keep:
            self._vectors = np.zeros((0, self.dim), np.float32)
            self._texts, self._emotions, self._timestamps = [], [], []
        else:
            self._vectors = self._vectors[keep]
            self._texts = [self._texts[i] for i in keep]
            self._emotions = [self._emotions[i] for i in keep]
            self._timestamps = [self._timestamps[i] for i in keep]
        # 重建 FAISS 索引（压缩后旧索引失效）
        if self._faiss is not None:
            try:
                faiss = safe_import("faiss")
                new_idx = faiss.IndexFlatIP(self.dim)
                if len(self._texts):
                    new_idx.add(self._vectors)
                self._faiss = new_idx
            except Exception:
                self._faiss = None

    # ================================================================
    # 持久化 / 清空 / 统计
    # ================================================================
    def persist(self):
        """把当前索引导出到磁盘（供重启恢复；路径基于 BASE_DIR）。"""
        try:
            os.makedirs(self.cfg.faiss_path, exist_ok=True)
            arr = self._vectors if self._vectors is not None else np.zeros((0, self.dim), np.float32)
            meta = [{"text": self._texts[i],
                     "emotion": (self._emotions[i].tolist()
                                 if self._emotions[i] is not None else None),
                     "timestamp": self._timestamps[i],
                     "use_count": self._use_count.get(i, 0)} for i in range(len(self._texts))]
            np.savez_compressed(
                os.path.join(self.cfg.faiss_path, "memory_index.npz"),
                vectors=arr, meta=json.dumps(meta, ensure_ascii=False))
            self.log.info("[MEM] 索引已持久化 → %s", self.cfg.faiss_path)
        except Exception as e:
            self.log.warning("[MEM] 持久化失败: %s", e)

    def load(self):
        """启动时从磁盘恢复索引（幂等；文件不存在则跳过）。"""
        p = os.path.join(self.cfg.faiss_path, "memory_index.npz")
        if not os.path.isfile(p):
            return
        try:
            data = np.load(p, allow_pickle=True)
            vecs = data["vectors"].astype(np.float32)
            meta = json.loads(str(data["meta"]))
            self._vectors = vecs
            self._texts = [m.get("text", "") for m in meta]
            self._emotions = [None if m.get("emotion") is None
                              else np.asarray(m["emotion"], dtype=np.float32) for m in meta]
            self._timestamps = [m.get("timestamp", 0.0) for m in meta]
            self._use_count = {i: m.get("use_count", 0) for i, m in enumerate(meta)}
            if self._faiss is not None and len(self._texts):
                self._faiss.add(self._vectors)
            self.log.info("[MEM] 从磁盘恢复 %d 条记忆", len(self._texts))
        except Exception as e:
            self.log.warning("[MEM] 恢复失败: %s", e)

    def clear(self):
        """清空短期记忆（格式化系统调用；长期 L3 权重不在本方法内清）。"""
        self._vectors = np.zeros((0, self.dim), np.float32)
        self._texts, self._emotions, self._timestamps = [], [], []
        self._use_count, self._migrated = {}, set()
        if self._faiss is not None:
            try:
                faiss = safe_import("faiss")
                self._faiss = faiss.IndexFlatIP(self.dim)
            except Exception:
                self._faiss = None
        self.log.info("[MEM] 短期记忆已清空（格式化流程）")

    def stats(self) -> dict:
        return {
            "entries": len(self._texts),
            "max_entries": getattr(self.cfg, "max_faiss_entries", 10000),
            "migrated_to_l3": len(self._migrated),
            "faiss": self._faiss is not None,
            "path": self.cfg.faiss_path,
        }


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # DEMO 自检（无 faiss 时走 numpy 降级，流程验证）
    _cfg = config.get_config()
    _mem = MemorySystem(_cfg, translator=None, l3=None)
    _rng = np.random.RandomState(7)
    for i in range(5):
        _mem.add(normalize(_rng.normal(0.0, 1.0, _cfg.STATE_DIM)),
                 text=f"记忆样本 {i}: 翁法罗斯的花海很安静", emotion=None)
    _q = _mem._vectors[0] if _mem._vectors is not None and len(_mem._texts) else None
    if _q is not None:
        _res = _mem.search(_q, k=3)
        print("[SELFTEST] search → ids=%s sims=%s" % (_res["ids"],
              [round(float(s), 3) for s in _res["similarities"]]))
    _rc = _mem.random_recall(2)
    print("[SELFTEST] random_recall → %d 条" % len(_rc))
    print("[SELFTEST] stats →", _mem.stats())
