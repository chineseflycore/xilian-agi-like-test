# -*- coding: utf-8 -*-
"""
core/cognitive_units.py — 认知单元注册与管理（AGI v2 核心）
==========================================================
内存/显存预估:
  - 单元规格: 1M 参数 ≈ 4MB（浮点权重）/ 单元
  - 最多 ~100 单元，**懒加载**（显存预算内按需装载，规格: 路由层常驻 GPU，
    连接网络 CPU）; 单单元显存: L3≈4MB / L2≈4MB / 情感≈143MB / 翻译器≈3.2GB(常驻)
  - 单元元数据表（名称/类型/状态）≈ < 1KB

设计（AGI v2 "每个认知单元 1M 参数模型，内部动态稀疏连接"）:
  · 单元 = 现有扩散器/情感/翻译器等实例的轻量封装
  · UnitRegistry: 注册 → 懒加载 → 激活 → 释放（显存预算控制器）
  · 每个单元暴露统一接口: feed(x) → 输出向量（512 维）
  · 单元间通信交给 core/connection_network.ConnectionNetwork（试错建立连接）

接口:
  UnitRegistry(cfg, max_units=100) /
    register(uid, name, kind, factory=None) / load(uid) / unload(uid) /
    feed(uid, x) / active_stats() / vram_budget_check()
"""

import os
import sys

import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

import config
from common_utils import get_logger, safe_import, normalize

logger = get_logger("cognitive_units")

# 单元类型 → 显存预估（MB，规范 §二 显存峰值 < 4.5GB 预算）
UNIT_VRAM_MB = {
    "diffuser_l1": 2.0,      # CPU numpy
    "diffuser_l2": 4.0,      # GPU torch transformer
    "diffuser_l3": 404.0,    # CPU 1M CSR（不含 GPU 推理副本）
    "emotion_40m": 143.0,    # GPU 40M 情感模型
    "translator": 3200.0,    # GPU Qwen3.5-0.8B 常驻
    "memory": 20.0,          # FAISS 10000×512
    "limb": 1.0,             # 规则
    "sentiment": 1.0,        # 规则
}


class CognitiveUnit:
    """单元封装: 统一 feed() 接口（内部动态稀疏连接由 diffuser 自身保证）。"""

    def __init__(self, uid, name, kind, obj=None):
        self.uid = int(uid)
        self.name = name
        self.kind = kind
        self.obj = obj            # 底层实现（L1/L2/L3/EmotionModel/Translator...）
        self.loaded = obj is not None
        self.last_output = None   # 最近一次 feed 输出（供连接网络传播）

    def feed(self, x) -> "np.ndarray":
        """统一输入接口: x(512,) → 输出(512,)；单元类型不同做适配。"""
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        o = self.obj
        if o is None:
            return np.zeros_like(x)
        # 扩散器系列: run() 返回 dict / diffuse() 返回 tuple / forward() 返回向量
        for fn, args in ((getattr(o, "run", None), (x,)),
                         (getattr(o, "diffuse", None), (x,)),
                         (getattr(o, "forward", None), (x,))):
            if fn is None:
                continue
            try:
                out = fn(*args)
                if isinstance(out, dict):
                    v = out.get("vector")
                elif isinstance(out, (tuple, list)):
                    v = out[0]
                else:
                    v = out
                v = np.asarray(v, dtype=np.float32).reshape(-1)
                if v.size:
                    self.last_output = normalize(v[:512] if v.size >= 512
                                                 else np.resize(v, 512))
                    return self.last_output
            except Exception:
                continue
        # 情感模型: from_text 需要文本 → 这里用 from_state（状态向量 → 情绪）
        if o is not None and hasattr(o, "from_state"):
            try:
                ev = o.from_state(x)
                self.last_output = normalize(np.resize(ev, 512).astype(np.float32))
                return self.last_output
            except Exception:
                pass
        self.last_output = normalize(x)
        return self.last_output

    @property
    def vram_mb(self) -> float:
        return UNIT_VRAM_MB.get(self.kind, 1.0)


class UnitRegistry:
    """认知单元注册表: 懒加载 + 显存预算控制（最多 ~100 单元）。"""

    def __init__(self, cfg=None, max_units: int = 100,
                 vram_budget_mb: float = 4500.0):
        self.cfg = cfg if cfg is not None else config.get_config()
        self.max_units = int(max_units)
        self.vram_budget_mb = float(vram_budget_mb)
        self._units = {}          # uid → CognitiveUnit

    # ------------------------------------------------------------------
    def register(self, uid: int, name: str, kind: str = "diffuser_l3",
                 factory=None) -> int:
        """注册一个单元（factory 为可调用对象，懒加载时调用创建底层对象）。"""
        if uid in self._units:
            return uid
        if len(self._units) >= self.max_units:
            logger.warning("[UNIT] 单元数达上限 %d", self.max_units)
            return -1
        self._units[uid] = CognitiveUnit(uid, name, kind, obj=None)
        self._units[uid].factory = factory
        logger.info("[UNIT] 注册 %d: %s（%s, 懒加载, ~%.0fMB）",
                    uid, name, kind, UNIT_VRAM_MB.get(kind, 1.0))
        return uid

    def load(self, uid: int) -> bool:
        """懒加载单元（显存预算检查）。"""
        u = self._units.get(uid)
        if u is None:
            return False
        if u.loaded:
            return True
        if not self.vram_budget_check(u.vram_mb):
            logger.warning("[UNIT] 显存预算不足，拒绝加载 %d（%s, +%.0fMB）",
                           uid, u.name, u.vram_mb)
            return False
        try:
            if callable(getattr(u, "factory", None)):
                u.obj = u.factory()
            u.loaded = u.obj is not None
            if u.loaded:
                logger.info("[UNIT] 已加载 %d: %s", uid, u.name)
            return u.loaded
        except Exception as e:
            logger.warning("[UNIT] 加载 %d 失败: %s", uid, e)
            return False

    def unload(self, uid: int):
        """释放单元（腾出显存；连接网络中的连接保留，重载后继续）。"""
        u = self._units.get(uid)
        if u is None:
            return
        u.obj = None
        u.loaded = False
        u.last_output = None
        logger.info("[UNIT] 已卸载 %d: %s（连接网络保留）", uid, u.name)

    def feed(self, uid: int, x) -> "np.ndarray":
        u = self._units.get(uid)
        if u is None or not self.load(uid):
            return np.zeros(np.asarray(x).reshape(-1).size, dtype=np.float32)
        return u.feed(x)

    def vram_budget_check(self, add_mb: float) -> bool:
        """估算当前已加载单元显存 + add_mb 是否在预算内。"""
        used = sum(u.vram_mb for u in self._units.values() if u.loaded)
        return (used + float(add_mb)) <= self.vram_budget_mb

    def active_stats(self) -> dict:
        loaded = [u for u in self._units.values() if u.loaded]
        return {"registered": len(self._units), "loaded": len(loaded),
                "vram_used_mb": round(sum(u.vram_mb for u in loaded), 1),
                "vram_budget_mb": self.vram_budget_mb}

    def get(self, uid: int):
        return self._units.get(uid)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 自检（不参与正式运行）
    reg = UnitRegistry()
    reg.register(0, "L3-主扩散", "diffuser_l3")
    reg.register(1, "L1-快速扩散", "diffuser_l1")
    print("[SELF] active_stats:", reg.active_stats())
    x = np.random.RandomState(0).randn(512).astype(np.float32)
    print("[SELF] feed(未加载) →", reg.feed(0, x).shape, "(无 factory 则零向量)")
