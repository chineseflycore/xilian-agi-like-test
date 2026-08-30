# -*- coding: utf-8 -*-
"""
sentiment_module.py — 感性模块（新增，独立于理性验证层，规范 §4.1/§十.25/27）
================================================================
内存/显存预估（规范 §九.1）:
  · 本模块不加载任何模型权重（emotion_model 由外部持有，不复制）。
  · 常驻仅保存一张经验表（特征 → 命中率，小 dict）与若干计数器，< 1MB RAM。
  · 无 GPU 显存占用；运行期临时分配 1~2 个 512 维向量（约 4KB）。

设计哲学（规范 §4.1 感性模块规格）:
  · 独立于验证层/批判层运行，不经过理性计算，直接做出价值判断。
  · 接收: 当前状态向量、情感向量、最近 5 轮对话摘要、候选文本。
  · 输出: "通过 / 拒绝 / 重新生成" 三类感性判定 + 强度分数 + 理由。
  · 冲突仲裁: 当感性判定与验证层评分冲突时，默认信任感性
    —— 权重 cfg.sentiment_weight=0.6 vs 验证层 0.4（规范 §4.1、§十.25）。

经验性可成长（规范 §十.27）:
  · 内部维护经验表: 特征(tag) → 历史判定命中率 {"total","good"}。
  · feedback(verdict, outcome) 在真实结果回落后更新经验权重；
    后续 judge 会用经验命中率调制判定强度（score），实现"随系统成长而变化"。

判定依据（经验式，不经过理性计算，规范 §1.3）:
  1. 情感混乱（arousal > 0.8 且 valence < -0.3）→ 直接 pass（允许直接输出）
  2. 候选与身份锚点余弦 < cfg.anchor_similarity_threshold → 倾向 regenerate
  3. 候选过短 / 与近期内容重复 → reject
  4. 其余 → pass

降级策略（关键路径 try-except，规范 §九.2）:
  · torch 不可用、emotion_model 缺失、锚点未初始化时均以 numpy/伪嵌入降级，
    保证 DEMO（纯 CPU 流程验证）可运行。

TODO(V3.4): 经验表落盘持久化（config.BASE_DIR 下）；情感模型输出接入真实情绪向量。
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

import numpy as np

from config import get_config
from common_utils import get_logger, normalize, cosine, clamp, pseudo_embedding

_LOG = get_logger("sentiment_module")

# 8 维情绪向量下标约定（规范 §3.2）: [0..4] 五类概率, [5] intensity, [6] arousal, [7] valence
_AROUSAL_IDX = 6
_VALENCE_IDX = 7

# 情感混乱阈值（规范 §1.3 直接输出触发条件）
_CHAOS_AROUSAL = 0.8
_CHAOS_VALENCE = -0.3

# 经验调制强度上限（score = (1-w)*规则分 + w*经验命中率）
_EXP_BLEND = 0.4


class SentimentModule:
    """感性模块: 关键时刻不经过理性计算，直接拍板；经验表随成长更新。"""

    def __init__(self, cfg, emotion_model=None, translator=None):
        self.cfg = cfg if cfg is not None else get_config()
        self.emotion_model = emotion_model   # 外部情感模型（40M，GPU），本模块不复制
        self.translator = translator         # 语义翻译器（REAL 身份锚定用真实 Qwen 编码）
        # 经验表: 特征(tag) → {"total": 出现次数, "good": 判定正确次数}
        self._experience = {}
        # 最近一次判定的特征标签（供 feedback 更新经验表用）
        self._last_tags = []
        self._total = 0                       # 累计判定次数
        self._verdict_counts = {"pass": 0, "reject": 0, "regenerate": 0}
        self._last_score = 0.5                # 最近一次判定强度（供冲突仲裁使用）

    # ------------------------------------------------------------------
    # 主入口: judge(...) -> {"verdict", "score", "reason"}
    # ------------------------------------------------------------------
    def judge(self, state_vector, emotion_vec, recent_summary, candidate_text) -> dict:
        """感性判定（经验式，不经过理性计算）。

        参数:
          state_vector : 512 维状态向量（numpy float32）
          emotion_vec  : 8 维情绪向量 [angry,sad,happy,neutral,fear,intensity,arousal,valence]
          recent_summary: 最近 5 轮对话摘要（字符串）
          candidate_text: 待判定候选文本（扩散器/翻译器的输出）

        返回:
          {"verdict": "pass"/"reject"/"regenerate", "score": 0~1, "reason": str}
        """
        self._total += 1
        text = (candidate_text or "").strip()
        tags, reasons = [], []
        vec = np.asarray(emotion_vec, dtype=np.float32) if emotion_vec is not None else None

        # ---- 规则 1: 情感混乱 → 直接放行（规范 §1.3 允许直接输出）----
        if vec is not None and vec.size >= 8:
            arousal, valence = float(vec[_AROUSAL_IDX]), float(vec[_VALENCE_IDX])
            if arousal > _CHAOS_AROUSAL and valence < _CHAOS_VALENCE:
                tags.append("chaos")
                reasons.append(
                    "情感混乱(arousal=%.2f>%.1f, valence=%.2f<%.1f)：感性直接放行，绕过理性验证"
                    % (arousal, _CHAOS_AROUSAL, valence, _CHAOS_VALENCE))
                return self._finalize("pass", 0.90, tags, reasons)

        # ---- 规则 2: 身份锚定 —— 余弦 < 阈值 → 倾向重新生成（规范 §4.7.1）----
        anchor = self.cfg.anchor_vector
        if anchor is None:
            # 锚点未初始化（DEMO）: 用伪嵌入替代真实 Qwen 编码（规范 §4.8 降级边界）
            anchor = pseudo_embedding(self.cfg.anchor_generation_prompt, self.cfg.STATE_DIM)
        # 候选文本向量: REAL 用 Qwen 真实编码（人设锚定的正确实现，防止人设偏离）；
        # DEMO 才用伪嵌入（仅流程验证，不具备真实语义）
        if self.translator is not None and not self.translator.is_demo:
            cand_vec = self.translator.encode([text])[0]
        else:
            cand_vec = pseudo_embedding(text, self.cfg.STATE_DIM)
        cos_anchor = cosine(cand_vec, anchor)
        if cos_anchor < self.cfg.anchor_similarity_threshold:
            tags.append("off_anchor")
            base = clamp(cos_anchor, 0.2, 0.6)   # 偏离越远分数越低
            reasons.append("身份锚定余弦=%.3f < 阈值%.2f → 建议重新生成"
                           % (cos_anchor, self.cfg.anchor_similarity_threshold))
            return self._finalize("regenerate", base, tags, reasons)

        # ---- 规则 3: 候选过短 / 重复 → reject ----
        if len(text) < 4:
            tags.append("too_short")
            reasons.append("候选过短(仅%d字) → 拒绝输出" % len(text))
            return self._finalize("reject", 0.25, tags, reasons)
        if self._is_repetitive(text, recent_summary):
            tags.append("repeated")
            reasons.append("候选内容重复（与近期摘要或自身重复）→ 拒绝输出")
            return self._finalize("reject", 0.30, tags, reasons)

        # ---- 规则 4: 默认放行 ----
        tags.append("normal")
        reasons.append("未见明显风险信号，感性判定通过")
        return self._finalize("pass", 0.72, tags, reasons)

    # ------------------------------------------------------------------
    # 经验融合与最终封装
    # ------------------------------------------------------------------
    def _finalize(self, verdict, base_score, tags, reasons):
        """规则分(权重 1-EXP_BLEND) 与 经验命中率(权重 EXP_BLEND) 融合成最终 score。
        经验命中率高 → 该情形下历史判定正确率高 → 强度上调；
        经验命中率低 → 历史判定经常出错 → 强度下调（保守）。
        score 语义: 判定强度（越高越倾向放行/肯定）；verdict 是最终结论。
        """
        exp_factor = self._experience_factor(tags)          # 0~1
        score = (1.0 - _EXP_BLEND) * float(base_score) + _EXP_BLEND * exp_factor
        if abs(exp_factor - 0.5) > 0.2:
            reasons.append("经验调制(命中率%.2f)已生效" % exp_factor)
        self._last_tags = list(tags)
        self._last_score = clamp(float(score), 0.0, 1.0)
        self._verdict_counts[verdict] = self._verdict_counts.get(verdict, 0) + 1
        _LOG.info("[SENTIMENT] verdict=%s score=%.3f tags=%s",
                  verdict, score, "/".join(tags) or "-")
        return {"verdict": verdict, "score": clamp(float(score), 0.0, 1.0),
                "reason": "；".join(reasons)}

    def _experience_factor(self, tags):
        """对触发特征的经验命中率求平均；未见过的特征按 0.5（中性）计。"""
        rates = [self._hit_rate(t) for t in tags]
        return float(np.mean(rates)) if rates else 0.5

    def _hit_rate(self, key):
        e = self._experience.get(key)
        if not e or e.get("total", 0) == 0:
            return 0.5
        return e["good"] / e["total"]

    # ------------------------------------------------------------------
    # 经验成长（规范 §十.27）: feedback(verdict, outcome)
    # ------------------------------------------------------------------
    def feedback(self, verdict, outcome) -> dict:
        """根据实际结果更新经验权重。
        参数:
          verdict: 之前的判定 "pass"/"reject"/"regenerate"
          outcome: bool，True=判定事后被证明正确，False=错误
        更新对象: 判定本身 + 该次判定的特征标签（特征 → 历史判定命中率，规范 §4.1）。
        """
        outcome = bool(outcome)
        keys = [str(verdict)] + [str(t) for t in getattr(self, "_last_tags", []) or []]
        for key in keys:
            e = self._experience.setdefault(key, {"total": 0, "good": 0})
            e["total"] += 1
            if outcome:
                e["good"] += 1
        _LOG.info("[SENTIMENT] feedback verdict=%s outcome=%s → 经验表已更新", verdict, outcome)
        return {"ok": True, "updated_keys": keys}

    # ------------------------------------------------------------------
    # 与验证层冲突仲裁（规范 §4.1、§十.25）: 信任感性 0.6 vs 验证层 0.4
    # ------------------------------------------------------------------
    def blend_with_validator(self, validator_score) -> dict:
        """把感性判定分数与验证层分数(0~100)按 0.6/0.4 权重融合。
        冲突时感性拥有最终决定权（0.6 > 0.4）：
          · 感性判"reject"而验证层高分 → 融合分被感性压到低位 → 最终拒稿；
          · 感性判"pass"而验证层低分 → 融合分仍受感性托底 → 允许输出。
        返回 {"sentiment_score", "validator_score", "blended", "sentiment_won": bool}。
        """
        s = float(getattr(self, "_last_score", 0.5))
        v = float(validator_score or 0.0) / 100.0
        w = float(self.cfg.sentiment_weight)      # 默认 0.6（hoyotool.ini [Sentiment]）
        blended = w * s + (1.0 - w) * v
        _LOG.info("[SENTIMENT] 冲突仲裁: 感性%.2f(权重%.1f) vs 验证%.2f(权重%.1f) → %.2f",
                  s, w, v, 1.0 - w, blended)
        return {"sentiment_score": s, "validator_score": v,
                "blended": clamp(blended, 0.0, 1.0), "sentiment_won": bool(s >= v)}

    # ------------------------------------------------------------------
    # 内部小工具
    # ------------------------------------------------------------------
    @staticmethod
    def _is_repetitive(text, recent_summary) -> bool:
        """重复检测: ① 自身 3-gram 去重率过低（复读机式输出）；② 与近期摘要整句重合。"""
        grams = [text[i:i + 3] for i in range(max(0, len(text) - 2))]
        if grams and len(set(grams)) / len(grams) < 0.5:
            return True
        if recent_summary and text in (recent_summary or ""):
            return True
        return False

    def stats(self) -> dict:
        """感性模块状态摘要。"""
        exp = {k: {"total": v["total"],
                   "hit_rate": round(v["good"] / v["total"], 3) if v["total"] else 0.0}
               for k, v in self._experience.items()}
        return {
            "total_judgements": self._total,
            "verdict_counts": dict(self._verdict_counts),
            "experience_size": len(self._experience),
            "experience": exp,
            "sentiment_weight": float(self.cfg.sentiment_weight),
        }


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # DEMO 自检：纯 CPU 流程验证（规范 §六.1）
    _cfg = get_config()
    _sm = SentimentModule(_cfg, emotion_model=None)
    _sv = np.random.RandomState(1).randn(512).astype(np.float32)
    # 正常情绪 → 默认 pass
    _r1 = _sm.judge(_sv, np.zeros(8, dtype=np.float32), "最近5轮摘要", "（轻声）我记得，你说花海很安静。")
    # 情感混乱 → 直接 pass
    _ev = np.zeros(8, dtype=np.float32)
    _ev[6], _ev[7] = 0.95, -0.6
    _r2 = _sm.judge(_sv, _ev, "最近5轮摘要", "无论输出什么都行……")
    _sm.feedback(_r2["verdict"], True)
    print("[SELFTEST] pass_case=%s chaos_case=%s stats=%s" % (
        _r1["verdict"], _r2["verdict"], _sm.stats()["total_judgements"]))
