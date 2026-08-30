# -*- coding: utf-8 -*-
"""
core/controller.py — 认知控制器（AGI v2 核心编排）
==================================================
内存/显存预估:
  - 本模块无权重，仅编排逻辑 + 小缓冲（< 1MB / 0 显存）
  - 依赖: 翻译器常驻 GPU（≈3.2GB）、单元注册表（懒加载）、连接网络（CPU）

设计（AGI v2 双层面智能）:
  1. 单元内部: 1M 参数扩散器（动态稀疏连接，复用 diffuser_l1/l2/l3）
  2. 单元之间: ConnectionNetwork 试错连接（激活单元集合 → 信号传播 → 融合）

流程:
  route(x, emotion, text) → 选单元集合（路由层，基于 router.py 的 mode 决策
    + 连接网络传播目标）→ 各单元 feed → 加权融合 → 512 输出
  report(verdict) → 感性/验证结果回灌连接网络（trial_update，试错建立连接）

接口:
  CognitiveController(cfg, translator, router, units, net) /
    forward(state, emotion_vec, text) / report(reward) / stats()
"""

import os
import sys

import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

import config
from common_utils import get_logger, normalize, clamp

logger = get_logger("cognitive_controller")


class CognitiveController:
    """认知控制器：单元激活决策 + 连接网络传播 + 融合。"""

    def __init__(self, cfg=None, translator=None, router=None,
                 units=None, net=None):
        self.cfg = cfg if cfg is not None else config.get_config()
        self.translator = translator
        self.router = router
        self.units = units        # UnitRegistry
        self.net = net            # ConnectionNetwork
        # 单元 id 约定（与 units 注册表对齐）
        self.UNIT_L3 = 0
        self.UNIT_L1 = 1
        self.UNIT_L2 = 2
        self.UNIT_EMO = 3

    # ------------------------------------------------------------------
    # 路由层: 决定激活哪些单元及连接方式（规格 "路由层决定激活单元与连接"）
    # ------------------------------------------------------------------
    def decide_units(self, state_vector, emotion_vec, text: str = "") -> dict:
        """选单元集合。基于 router.route 的 line 决策 + 连接网络可达单元。

        返回 {"active": [uid...], "mode": "L1/L2/L3", "net_targets": [uid...]}
        """
        route = {}
        if self.router is not None:
            route = self.router.route(state_vector, emotion_vec, text)
        mode = route.get("line_used", self.cfg.default_mode)

        # 单元激活: 主扩散器 + 连接网络中的传播目标
        active = []
        if mode == "L1" and self.units is not None:
            active.append(self.UNIT_L1)
        elif mode == "L2" and self.units is not None:
            active.append(self.UNIT_L2)
        else:
            active.append(self.UNIT_L3)

        # 连接网络传播目标: 主单元的出边
        net_targets = []
        if self.net is not None:
            main = self.UNIT_L3 if self.UNIT_L3 in active else active[0]
            net_targets = [b for b, _ in self.net.outgoing(main)]

        # 情感单元按需激活（情绪判断需要）
        if self.units is not None and self.UNIT_EMO not in active:
            active.append(self.UNIT_EMO)

        logger.info("[CTRL] 激活单元=%s mode=%s 网络目标=%s",
                    active, mode, net_targets)
        return {"active": active, "mode": mode, "net_targets": net_targets}

    # ------------------------------------------------------------------
    # 前向: 状态 → 单元扩散 → 连接网络传播 → 融合输出
    # ------------------------------------------------------------------
    def forward(self, state_vector, emotion_vec, text: str = "") -> dict:
        """完整认知前向。

        返回 {"vector": (512,), "confidence": float, "mode": str,
              "active": [...], "propagated": {uid: vec}}
        """
        dec = self.decide_units(state_vector, emotion_vec, text)
        outputs = {}
        if self.units is not None:
            for uid in dec["active"]:
                try:
                    if uid == self.UNIT_EMO:
                        # 情感单元直接输出情绪向量扩展
                        ev = np.asarray(emotion_vec, dtype=np.float32).reshape(-1)
                        outputs[uid] = normalize(np.resize(ev, 512).astype(np.float32))
                    else:
                        outputs[uid] = self.units.feed(uid, state_vector)
                except Exception as e:
                    logger.warning("[CTRL] 单元 %d 前向失败: %s", uid, e)

        # 连接网络传播: 主单元输出 → 目标单元输入（信号叠加）
        propagated = {}
        if self.net is not None and outputs:
            main = dec["active"][0] if dec["active"] else self.UNIT_L3
            if main in outputs:
                pr = self.net.activate(main, outputs[main])
                propagated = pr["propagated"]

        # 融合: 主扩散输出为主，传播目标信号按权重叠加
        vec = np.zeros(self.cfg.STATE_DIM, dtype=np.float32)
        conf = 0.5
        if outputs:
            main = dec["active"][0] if dec["active"] else self.UNIT_L3
            vec = outputs.get(main, vec)
            conf = 0.55
            for uid, pv in propagated.items():
                w = 0.2
                for b, ww in (self.net.outgoing(main) if self.net else []):
                    if b == uid:
                        w = ww * 0.5
                vec = vec + pv * float(w)
        vec = normalize(vec)
        return {"vector": vec, "confidence": clamp(conf, 0.0, 1.0),
                "mode": dec["mode"], "active": dec["active"],
                "propagated": propagated}

    # ------------------------------------------------------------------
    # 试错回灌（规格: 连接方式通过试错建立）
    # ------------------------------------------------------------------
    def report(self, reward: float, path: list = None) -> dict:
        """把感性/验证结果回灌连接网络（正 reward 强化路径连接）。

        reward ∈ [-1, 1]（感性 pass → +，reject → -）。
        path: 本次激活单元路径（默认取最近一次 active）。
        """
        if self.net is None:
            return {"updated": 0, "unlinked": 0}
        if path is None:
            path = getattr(self, "_last_active", [self.UNIT_L3])
        self._last_active = list(path or [])
        return self.net.trial_update(path, reward)

    def stats(self) -> dict:
        return {
            "units": self.units.active_stats() if self.units else None,
            "network": self.net.stats() if self.net else None,
            "last_active": getattr(self, "_last_active", None),
        }
