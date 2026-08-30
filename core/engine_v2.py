# -*- coding: utf-8 -*-
"""
core/engine_v2.py — 昔涟AGI v2 主装配（多认知单元 + 连接网络 + 控制器）
======================================================================
内存/显存预估:
  - 复用 server.Engine（Qwen3.5-0.8B ≈ 3.2GB 显存等）
  - UnitRegistry 懒加载单元 + ConnectionNetwork（CPU < 1MB）
  - 总显存 ≈ 引擎基线 + 已激活单元（预算 4.5GB，规格 §硬件约束）

AGI v2 双层面智能（规格 §核心设计哲学）:
  1. 单元内部: 1M 参数扩散器（动态稀疏连接）
  2. 单元之间: ConnectionNetwork 试错连接（路由决定激活，传播融合）

接口:
  EngineV2(cfg) -> chat_v2(message) / controller / units / net / engine
"""

import os
import sys

import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

import config
from common_utils import get_logger, normalize

logger = get_logger("engine_v2")


class EngineV2:
    """AGI v2 装配: 旧引擎（翻译/路由/情感/记忆/痛觉）+ 单元 + 连接网络 + 控制器。"""

    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else config.get_config()
        from server import Engine
        self.engine = Engine(self.cfg)          # 旧引擎（翻译器/路由/情感/记忆等）

        # ---- 认知单元注册表（懒加载，工厂指向旧引擎实例）----
        from core.cognitive_units import UnitRegistry
        self.units = UnitRegistry(self.cfg, max_units=100,
                                  vram_budget_mb=4500.0)
        self.units.register(0, "L3-主扩散", "diffuser_l3",
                            factory=lambda: self.engine.l3)
        self.units.register(1, "L1-快速扩散", "diffuser_l1",
                            factory=lambda: self.engine.l1)
        self.units.register(2, "L2-Transformer", "diffuser_l2",
                            factory=lambda: self.engine.l2)
        self.units.register(3, "情感-40M", "emotion_40m",
                            factory=lambda: self.engine.emotion)

        # ---- 单元间连接网络（试错建立，CPU）----
        from core.connection_network import ConnectionNetwork
        self.net = ConnectionNetwork(self.cfg, max_units=100)
        # 初始连接: L3 → 情感（默认通路）; L1 → L3（快速扩散接入主路）
        self.net.register_unit(0, "L3-主扩散", "diffuser_l3")
        self.net.register_unit(1, "L1-快速扩散", "diffuser_l1")
        self.net.register_unit(2, "L2-Transformer", "diffuser_l2")
        self.net.register_unit(3, "情感-40M", "emotion_40m")
        self.net.link(0, 3, 0.6)
        self.net.link(1, 0, 0.4)

        # ---- 认知控制器 ----
        from core.controller import CognitiveController
        self.controller = CognitiveController(
            self.cfg, translator=self.engine.translator,
            router=self.engine.router, units=self.units, net=self.net)
        self.engine.controller = self.controller    # 供 CLI /status 展示
        logger.info("[V2] 认知控制器就绪: %s", self.controller.stats())

    # ------------------------------------------------------------------
    def chat_v2(self, message: str) -> dict:
        """AGI v2 对话路径: 感知融合 → 控制器多单元前向 → 生成 → 感性 → 试错回灌。

        与旧 Engine.chat 的差异: 扩散阶段走 CognitiveController.forward
        （多单元 + 连接网络传播），并返回单元激活/网络信息。
        """
        eng = self.engine
        try:
            emotion = eng.emotion.from_text(message)
            text_vec = eng.translator.encode([message])[0]
            state = eng.fusion.fuse(text_vec, None, None)

            # 多单元认知前向（控制器）
            ctrl = self.controller.forward(state, emotion, message)

            # 路由（供管线标签与直接输出判定）
            route = eng.router.route(ctrl["vector"], emotion, message)

            # 生成（深度人设 + few-shot，分场景温度）
            scenario = eng._pick_scenario(message)
            persona = getattr(self.cfg, "persona_system", None) or self.cfg.anchor_generation_prompt
            from cloud_model_client import SEED_EXAMPLES
            candidate = eng.translator.decode(
                f"{message}\n{scenario['prompt']}", persona=persona,
                few_shots=SEED_EXAMPLES, temperature=scenario["temperature"])

            # 感性判定 + 试错回灌连接网络（reward）
            verdict = eng.sentiment.judge(state, emotion, eng._history_text(), candidate)
            reward = 0.8 if verdict["verdict"] == "pass" else -0.5
            ctrl_rpt = self.controller.report(reward, path=ctrl["active"])

            # 记忆 + 好感度
            eng.memory.add(state, text=message, emotion=emotion)
            eng.affection.update(emotion, user_text=message, reward=reward / 2.0)
            eng.affection.save()

            return {
                "response_text": candidate,
                "line_used": route["line_used"],
                "emotion": eng._emotion_dict(emotion),
                "affection": eng.affection.value,
                "affection_level": eng.affection.level(),
                "pain_value": round(float(eng.pain.avg_pain), 3),
                "sentiment_verdict": verdict["verdict"],
                "active_units": ctrl["active"],
                "net_active_links": len(ctrl.get("propagated", {})),
                "net_update": ctrl_rpt,
            }
        except Exception as e:
            logger.error("[V2] chat_v2 异常: %s", e, exc_info=True)
            return {"response_text": "（沉默）……我有些恍惚，让我缓一缓。",
                    "error": repr(e)}

    def status(self) -> dict:
        return {**self.engine.status(), "v2": self.controller.stats()}

    def shutdown(self):
        try:
            self.net.save()
        except Exception:
            pass
        self.engine.shutdown()


if __name__ == "__main__":
    v2 = EngineV2()
    r = v2.chat_v2("你好，昔涟。")
    print("[SELF]", r["response_text"][:80])
    print("[SELF] units:", r["active_units"], "| net:", r["net_active_links"])
    v2.shutdown()
