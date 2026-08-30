# -*- coding: utf-8 -*-
"""
core/connection_network.py — 认知单元间连接网络（AGI v2 核心）
=============================================================
内存/显存预估:
  - 连接表: N×N 稀疏（默认 100 单元 → 最多 9900 条连接），float32 ≈ < 1MB
  - 每条连接含: 权重(0~1)、试错计数、最后更新时间 —— 全部 numpy 数组
  - 显存: 0MB —— 连接网络强制 CPU 管理（规格: 连接网络在 CPU 内存管理）

设计（AGI v2 "模型之间通过巨大的外部连接网络通信，连接方式通过试错建立"）:
  · 单元本身固定（认知单元 core/cognitive_units.py），单元间连接可塑
  · link(a, b, init)   : 建立/更新 a→b 连接
  · activate(a, x)     : 信号经连接网络传播（a 的输出按连接权重驱动 b 的输入）
  · trial_update(路径, reward) : 试错建立/强化连接（Hebbian 式:
      reward>0 → 该路径上连接权重上升；reward<0 → 衰减；权重<阈值 → 连接断开）
  · 断开/重连: 长期低权重的连接自动"遗忘"（规范 §4.7.4 时间衰减精神）

接口:
  ConnectionNetwork(cfg) / register_unit(uid, name) / link / unlink /
  activate / trial_update / decay_all / stats / save / load
"""

import os
import json
import time

import numpy as np

import config
from common_utils import get_logger, clamp

logger = get_logger("connection_network")

# 连接断开阈值（权重低于此值 → 试错后自动断开）
_UNLINK_THRESHOLD = 0.05
# 连接学习率（试错更新步长）
_LR = 0.1


class ConnectionNetwork:
    """单元间连接网络：稀疏邻接表（numpy），权重可塑，试错驱动。"""

    def __init__(self, cfg=None, max_units: int = 100):
        self.cfg = cfg if cfg is not None else config.get_config()
        self.max_units = int(max_units)
        # 单元注册表: uid(int) → {"name", "type", "obj"(认知单元, 可选)}
        self._units = {}
        # 邻接表: (a,b) → {"weight","trials","last_ts"}
        self._edges = {}
        self._path = os.path.join(config.BASE_DIR, "data_cache", "connection_network.json")
        self.load()

    # ------------------------------------------------------------------
    # 单元注册
    # ------------------------------------------------------------------
    def register_unit(self, uid: int, name: str, unit_type: str = "diffuser",
                      obj=None) -> bool:
        """注册一个认知单元。uid 超出 max_units → 拒绝。"""
        if uid in self._units:
            logger.info("[NET] 单元 %d 已存在（%s）", uid, name)
            return True
        if len(self._units) >= self.max_units:
            logger.warning("[NET] 单元数已达上限 %d，拒绝注册 %s", self.max_units, name)
            return False
        self._units[uid] = {"name": str(name), "type": str(unit_type), "obj": obj}
        logger.info("[NET] 注册单元 %d: %s（%s）", uid, name, unit_type)
        return True

    # ------------------------------------------------------------------
    # 连接建立/断开
    # ------------------------------------------------------------------
    def link(self, a: int, b: int, weight: float = 0.5) -> dict:
        """建立/更新 a→b 连接（权重 0~1）。"""
        if a not in self._units or b not in self._units:
            raise KeyError("单元未注册: %d→%d" % (a, b))
        key = (int(a), int(b))
        w = clamp(float(weight), 0.0, 1.0)
        self._edges[key] = {"weight": w, "trials": 0, "last_ts": time.time()}
        return {"a": a, "b": b, "weight": w}

    def unlink(self, a: int, b: int) -> bool:
        key = (int(a), int(b))
        if key in self._edges:
            del self._edges[key]
            logger.info("[NET] 断开连接 %d→%d（试错淘汰）", a, b)
            return True
        return False

    def outgoing(self, uid: int) -> list:
        """uid 的出边列表 [(b, weight)]。"""
        return [(b, e["weight"]) for (a, b), e in self._edges.items() if a == uid]

    def incoming(self, uid: int) -> list:
        """uid 的入边列表 [(a, weight)]。"""
        return [(a, e["weight"]) for (a, b), e in self._edges.items() if b == uid]

    # ------------------------------------------------------------------
    # 信号传播（核心: 单元间通信，规格 "巨大的外部连接网络"）
    # ------------------------------------------------------------------
    def activate(self, source_uid: int, signal_vec) -> dict:
        """source 单元的输出信号沿连接网络传播。

        返回 {"propagated": {target_uid: 加权信号副本}, "active_links": int}。
        signal_vec 为 512 维 numpy；每个出边目标获得 signal × weight 的输入
        （由目标单元自行处理，如作为状态向量预热）。
        """
        signal_vec = np.asarray(signal_vec, dtype=np.float32).reshape(-1)
        propagated, active = {}, 0
        for b, w in self.outgoing(source_uid):
            propagated[b] = (signal_vec * float(w)).astype(np.float32)
            active += 1
        return {"propagated": propagated, "active_links": active}

    # ------------------------------------------------------------------
    # 试错学习（规格: "连接方式通过试错建立"）
    # ------------------------------------------------------------------
    def trial_update(self, path: list, reward: float, lr: float = _LR) -> dict:
        """一次试错的结果反馈：强化/弱化路径上的所有连接。

        reward ∈ [-1, 1]（感性模块判定通过 → 正，拒绝 → 负）。
        Hebbian 式: Δw = lr × reward × w × (1 - w)（保持 0~1）
        权重 < _UNLINK_THRESHOLD → 断开该连接（"这条路走不通"）。
        返回 {"updated": n, "unlinked": m}。
        """
        updated, unlinked = 0, 0
        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            key = (int(a), int(b))
            e = self._edges.get(key)
            if e is None:
                continue
            w = e["weight"]
            # 逻辑斯蒂增长式更新（0~1 有界）
            w_new = w + lr * float(reward) * w * (1.0 - w)
            w_new = clamp(w_new, 0.0, 1.0)
            e["weight"] = w_new
            e["trials"] += 1
            e["last_ts"] = time.time()
            updated += 1
            if w_new < _UNLINK_THRESHOLD:
                del self._edges[key]
                unlinked += 1
        if updated:
            logger.info("[NET] 试错反馈 reward=%+.2f: 更新 %d 连接, 断开 %d",
                        reward, updated, unlinked)
        return {"updated": updated, "unlinked": unlinked}

    # ------------------------------------------------------------------
    # 时间衰减 / 统计 / 持久化
    # ------------------------------------------------------------------
    def decay_all(self, rate: float = 0.02, min_weight: float = _UNLINK_THRESHOLD):
        """全局连接权重时间衰减（规范 §4.7.4）; 低于阈值自动断开。"""
        for key in list(self._edges.keys()):
            e = self._edges[key]
            e["weight"] = clamp(e["weight"] * (1.0 - rate), 0.0, 1.0)
            if e["weight"] < min_weight:
                del self._edges[key]

    def stats(self) -> dict:
        return {"units": len(self._units), "edges": len(self._edges),
                "max_units": self.max_units}

    def save(self):
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            payload = {"units": {str(k): v for k, v in self._units.items()},
                       "edges": {f"{a},{b}": e for (a, b), e in self._edges.items()}}
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
        except Exception as e:
            logger.warning("[NET] 保存失败: %s", e)

    def load(self):
        try:
            if os.path.isfile(self._path):
                with open(self._path, "r", encoding="utf-8") as f:
                    p = json.load(f)
                self._units = {int(k): v for k, v in p.get("units", {}).items()}
                self._edges = {tuple(map(int, k.split(","))): v
                               for k, v in p.get("edges", {}).items()}
                logger.info("[NET] 已恢复: %d 单元, %d 连接",
                            len(self._units), len(self._edges))
        except Exception as e:
            logger.warning("[NET] 读取失败（用空网络）: %s", e)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    _net = ConnectionNetwork()
    _net.register_unit(0, "L3-主扩散", "diffuser_l3")
    _net.register_unit(1, "情感", "emotion")
    _net.link(0, 1, 0.6)
    _v = np.random.RandomState(0).randn(512).astype(np.float32)
    _r = _net.activate(0, _v)
    print("[SELF] activate:", _r["active_links"], "条传播")
    _u = _net.trial_update([0, 1], reward=0.8)
    print("[SELF] trial_update:", _u)
    print("[SELF] stats:", _net.stats())
