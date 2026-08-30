# -*- coding: utf-8 -*-
"""
affection.py — 好感度系统（借鉴 xilian-agent 的 affection/好感度设计）
======================================================================
内存/显存预估: 无模型权重，仅一个 float 值 + 少量计数器，< 1KB；0 显存。

设计（对齐"昔涟与伙伴的关系成长"人设闭环，参考 yaosqee/xilian-agent）:
  · 好感度 0~100，初始 30（中立偏暖）
  · 关系等级: 陌生(<20) / 相识(20~40) / 熟悉(40~60) / 亲近(60~80) / 挚爱(>80)
  · 更新规则:
      - 对话情绪 valence>0 且 intensity 高 → 好感上升；valence<0 强烈 → 下降
      - 用户输入含亲近/感谢/陪伴信号 → 额外加分
      - 长期无互动 → 缓慢时间衰减（"想念却渐渐模糊"）
  · 影响面:
      - limb_controller 动作柔和度（挚爱更亲昵，陌生更疏离）
      - server /chat 输出 affection 字段
      - sentiment 判定可参考（关系近 → 更宽容，规范 §十.27 经验式成长）

接口:
  value / level() / update(emotion_vec, user_text="") / decay_if_idle()
  save() / load()（data_cache/affection.json）
"""

import os
import json
import time

import numpy as np

import config
from common_utils import get_logger, clamp

logger = get_logger("affection")

# 关系等级阈值与文案（借鉴 xilian 关系距离概念）
LEVELS = [
    (80.0, "挚爱",   "像牵住彼此名字的两个人，话可以说到很深的地方。"),
    (60.0, "亲近",   "已经可以放心地把脆弱交到对方手里。"),
    (40.0, "熟悉",   "开始记得对方的小习惯，语气更放松了。"),
    (20.0, "相识",   "还在慢慢试探彼此的故事。"),
    (0.0,  "陌生",   "礼貌而疏离，保持着故事化的距离。"),
]


class AffectionSystem:
    """好感度：伙伴关系成长（持久化到 data_cache/affection.json）。"""

    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else config.get_config()
        self.value = 30.0                     # 初始中立偏暖
        self.total_interactions = 0
        self.last_interaction_ts = time.time()
        self._path = os.path.join(config.BASE_DIR, "data_cache", "affection.json")
        self.load()

    # ------------------------------------------------------------------
    # 关系等级
    # ------------------------------------------------------------------
    def level(self) -> str:
        for thr, name, _ in LEVELS:
            if self.value >= thr:
                return name
        return "陌生"

    def level_desc(self) -> str:
        for thr, name, desc in LEVELS:
            if self.value >= thr:
                return desc
        return LEVELS[-1][2]

    # ------------------------------------------------------------------
    # 更新（对话后调用）
    # ------------------------------------------------------------------
    def update(self, emotion_vec, user_text: str = "", reward: float = 0.0) -> dict:
        """根据情绪向量与用户输入更新好感度。

        情绪影响（借鉴 PAD: valence 主导亲近/疏离）:
          valence > 0.3 且 intensity > 0.3 → +0.6~+1.5
          valence < -0.3 且 intensity > 0.6 → -0.5~-1.2（强烈负面）
          低活性发呆 → 微降（"沉默久了，距离会悄悄变远"）
        用户输入信号: 感谢/喜欢/陪伴/明天见 → 额外 +0.5；攻击/命令 → -0.8
        """
        e = np.asarray(emotion_vec, dtype=np.float32).reshape(-1)
        delta = 0.0
        if e.size >= 8:
            valence = float(e[7])
            intensity = float(e[5])
            if valence > 0.3 and intensity > 0.3:
                delta += 0.6 + 0.9 * float(np.clip(valence, 0, 1))
            elif valence < -0.3 and intensity > 0.6:
                delta -= 0.5 + 0.7 * float(np.clip(-valence, 0, 1))
            elif intensity < 0.2:
                delta -= 0.15                        # 低活性/发呆（规范 §4.6）
        text = (user_text or "")
        if any(w in text for w in ("谢谢", "喜欢", "陪伴", "明天见", "想你", "晚安")):
            delta += 0.5
        if any(w in text for w in ("滚", "闭嘴", "讨厌", "垃圾")):
            delta -= 0.8
        delta += float(reward)
        old = self.value
        self.value = clamp(self.value + delta, 0.0, 100.0)
        self.total_interactions += 1
        self.last_interaction_ts = time.time()
        if abs(delta) >= 0.3:
            logger.info("[AFFECTION] 好感 %+.2f → %.1f（%s）", delta, self.value, self.level())
        return {"old": round(old, 2), "new": round(self.value, 2),
                "delta": round(delta, 2), "level": self.level()}

    # ------------------------------------------------------------------
    # 时间衰减（静默期/长时间不互动，规范 §4.7.4 时间衰减）
    # ------------------------------------------------------------------
    def decay_if_idle(self, idle_threshold_s: float = 86400.0) -> float:
        """超过 idle 阈值未互动 → 好感缓慢衰减（每天最多 -1）。"""
        idle = time.time() - self.last_interaction_ts
        if idle > idle_threshold_s:
            days = idle / 86400.0
            drop = min(1.0 * days, 5.0)
            self.value = clamp(self.value - drop, 0.0, 100.0)
            logger.info("[AFFECTION] 长时间未互动（%.1f 天），好感 -%.1f → %.1f",
                        days, drop, self.value)
            return drop
        return 0.0

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------
    def save(self):
        try:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump({"value": self.value,
                           "total_interactions": self.total_interactions,
                           "last_interaction_ts": self.last_interaction_ts},
                          f, ensure_ascii=False, indent=1)
        except Exception as e:
            logger.warning("[AFFECTION] 保存失败: %s", e)

    def load(self):
        try:
            if os.path.isfile(self._path):
                with open(self._path, "r", encoding="utf-8") as f:
                    d = json.load(f)
                self.value = clamp(float(d.get("value", 30.0)), 0.0, 100.0)
                self.total_interactions = int(d.get("total_interactions", 0))
                self.last_interaction_ts = float(d.get("last_interaction_ts", time.time()))
                logger.info("[AFFECTION] 已恢复: 好感 %.1f（%s），累计 %d 次互动",
                            self.value, self.level(), self.total_interactions)
        except Exception as e:
            logger.warning("[AFFECTION] 读取失败（用初始值）: %s", e)

    def stats(self) -> dict:
        return {"affection": round(self.value, 2), "level": self.level(),
                "level_desc": self.level_desc(),
                "total_interactions": self.total_interactions}


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 自检（不参与正式运行）
    _a = AffectionSystem()
    _e = np.zeros(8, dtype=np.float32)
    _e[5], _e[7] = 0.8, 0.7        # intensity=0.8 valence=0.7
    print("[SELF] update(+):", _a.update(_e, "谢谢你陪我说了这么多"))
    _e[5], _e[7] = 0.9, -0.8
    print("[SELF] update(-):", _a.update(_e, ""))
    print("[SELF] stats:", _a.stats())
