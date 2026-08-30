# -*- coding: utf-8 -*-
"""
router.py — 路由层（Qwen3.5-0.8B 真实路由，GPU 常驻，规范 §4.1）
================================================================
内存/显存预估（规范 §九.1）:
  本模块不持有独立权重，复用 translator 的 Qwen3.5-0.8B
  （≈3.2GB 显存已计入 translator.py），自身仅常量与轻量计算，< 1MB RAM。

职责:
  1. 依据融合层 512 维状态向量 + 8 维情绪向量输出扩散参数与世界状态标签（规范 §4.1）
  2. 真实路由: 构造路由提示词 → translator.next_token_probs(["1","2","3"])
     → softmax 概率选择 L1/L2/L3 并给出置信度（不再使用纯启发式近似）
  3. 判定是否直接输出（“允许犯错”，规范 §1.3）
  4. 降级: 真实路由失败/DEMO → 状态向量能量×熵启发式（规范 §六.1）
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
from common_utils import get_logger, clamp, cosine

# 语境关键词 → 世界状态标签（规范 §4.1 “世界状态标签”; 示例: 翁法罗斯/花海/轮回）
_KEYWORD_LABELS = [
    ("翁法罗斯", "翁法罗斯"),
    ("花海", "花海"),
    ("轮回", "轮回"),
    ("死亡", "死亡"),
    ("格式化", "格式化"),
    ("再见", "离别"),
    ("黄金裔", "黄金裔"),
]

# 真实路由提示词（Qwen 多选一；候选数字 token "1/2/3" → L1/L2/L3）
_ROUTE_PROMPT = (
    "你是昔涟人格引擎的路由层。请仅从 1、2、3 中选择一个数字，"
    "为下面这条输入选择最合适的扩散管线：\n"
    "1 = 纯数据/概率扩散（最快，适合混乱或自发激活内容）\n"
    "2 = Transformer 扩散（适合清晰、高置信的常规回应）\n"
    "3 = 神经元特化网络（默认，最接近人格回路的深层扩散）\n"
    "输入：{input_text}\n选择："
)
_CANDIDATES = ["1", "2", "3"]
_ROUTE_TO_LINE = {"1": "L1", "2": "L2", "3": "L3"}


class Router:
    """路由层：状态向量 + 情绪向量 → 扩散参数 / 世界标签 / 置信度 / 直接输出判定。"""

    def __init__(self, cfg=None, translator=None):
        self.cfg = cfg or get_config()
        self.translator = translator      # 语义翻译器（REAL 路径提供真实 logits 路由）
        self.logger = get_logger("router")

    # ------------------------------------------------------------------
    # 真实路由（Qwen logits softmax，规范 §4.1）
    # ------------------------------------------------------------------
    def _real_route(self, input_text: str) -> dict:
        """真实路由: 路由提示词 → next_token_probs → 管线选择 + 置信度。

        返回 {"line": "L1"|"L2"|"L3", "confidence": 0~1, "prob": {...}}
        失败时返回 None（调用方回退启发式）。
        """
        if self.translator is None or self.translator.is_demo:
            return None
        try:
            prompt = _ROUTE_PROMPT.format(input_text=(input_text or "")[:200])
            res = self.translator.next_token_probs(prompt, _CANDIDATES)
            if res.get("confidence", 0.0) <= 0.0:
                return None   # DEMO 均匀概率 → 视为不可用，回退启发式
            return {
                "line": _ROUTE_TO_LINE.get(str(res.get("argmax")), "L3"),
                "confidence": float(res["confidence"]),
                "prob": dict(res.get("prob", {})),
            }
        except Exception as e:
            self.logger.warning(f"[ROUTE] 真实路由失败，回退启发式: {e!r}")
            return None

    # ------------------------------------------------------------------
    # 启发式置信度（降级路径，规范 §六.1）
    # ------------------------------------------------------------------
    def _demo_confidence(self, state_vector) -> float:
        """DEMO 启发式置信度: 能量 × (1 - 归一化熵)。

        能量 = clamp(||v||)：状态向量能量越高越“有内容”；
        熵    = 归一化信息熵：分布越集中（低熵）越“清晰”，均匀噪声（高熵）越不可信。
        confidence = 能量 × 集中度 ∈ [0, 1]。
        """
        v = np.asarray(state_vector, dtype=np.float32).reshape(-1)
        energy = clamp(float(np.linalg.norm(v)))
        p = np.abs(v) / (float(np.sum(np.abs(v))) + 1e-12)
        h = -float(np.sum(p * np.log(p + 1e-12))) / np.log(max(2, v.size))
        concentration = 1.0 - clamp(h)
        return clamp(energy * concentration)

    # ------------------------------------------------------------------
    # 世界状态标签
    # ------------------------------------------------------------------
    def _world_labels(self, input_text: str) -> list:
        """语境标签: 恒含 cfg.world_context，再从输入文本中匹配关键词。"""
        labels = [self.cfg.world_context]
        text = input_text or ""
        for kw, label in _KEYWORD_LABELS:
            if kw in text and label not in labels:
                labels.append(label)
        return labels

    # ------------------------------------------------------------------
    # 路由主入口
    # ------------------------------------------------------------------
    def route(self, state_vector, emotion_vec, input_text: str = "",
              spontaneous: bool = False) -> dict:
        """路由判定。

        参数:
          state_vector: 融合层 512 维状态向量
          emotion_vec : 8 维情绪向量 [angry,sad,happy,neutral,fear,intensity,arousal,valence]
          input_text  : 用户输入文本（REAL 路径用于 Qwen 真实路由；可选）
          spontaneous : 是否自发激活（静默期随机召回注入的“乱码”，规范 §4.3/§1.3）

        返回:
          diffusion_params: {"mode","iteration_steps","damping","resonance_freq"}
          world_labels    : 世界状态标签列表
          confidence      : 0~1
          direct_output   : 是否直接输出（绕过验证/批判层，规范 §1.3）
          direct_reasons  : 直接输出触发原因（日志用）
          line_used       : 本次实际使用的管线（L1/L2/L3）
        """
        v = np.asarray(state_vector, dtype=np.float32).reshape(-1)
        e = np.asarray(emotion_vec, dtype=np.float32).reshape(-1)
        # 情绪向量布局: 0~4 为五维概率，5=intensity，6=arousal，7=valence（规范 §3.2）
        arousal = float(e[6]) if e.size > 6 else 0.0
        valence = float(e[7]) if e.size > 7 else 0.0

        # ---- 1) 置信度与管线: 真实 logits 路由优先，失败回退启发式 ----
        real = self._real_route(input_text)
        confidence = None
        mode = None
        if real is not None:
            confidence = real["confidence"]
            mode = real["line"]
            self.logger.info(
                f"[ROUTE] 真实路由 prob={real['prob']} → {mode} "
                f"confidence={confidence:.3f}")
        if confidence is None:
            confidence = self._demo_confidence(v)

        # ---- 2) 直接输出判定（规范 §1.3）----
        reasons = []
        if confidence < self.cfg.direct_output_threshold:        # 阈值 0.45
            reasons.append("低置信度")
        if arousal > 0.8 and valence < -0.3:                    # 情感混乱
            reasons.append("情感混乱")
        if spontaneous:                                         # 自发激活乱码
            reasons.append("自发激活乱码")
        direct_output = bool(self.cfg.direct_output_enabled) and len(reasons) > 0

        # ---- 3) 世界状态标签 ----
        world_labels = self._world_labels(input_text)

        # ---- 4) 管线选择（优先级: 情感混乱/自发激活 > 真实路由 > 默认）----
        if arousal > 0.8 and valence < -0.3:
            mode = "L1"                     # 情感混乱 → 最快纯数据扩散
        elif spontaneous:
            mode = "L1"                     # 自发激活乱码 → 快速扩散直接输出
        elif mode is None:
            # 启发式降级路径: 高置信 → L2，其余保持 cfg.default_mode（默认 L3）
            mode = "L2" if confidence >= 0.7 else self.cfg.default_mode

        diffusion_params = {
            "mode": mode,
            "iteration_steps": self.cfg.iteration_steps,
            "damping": self.cfg.damping,
            "resonance_freq": self.cfg.resonance_freq,
        }

        if direct_output:
            self.logger.warning(
                f"\033[33m[DIRECT_MODE] 直接输出（{', '.join(reasons)}）"
                f"confidence={confidence:.3f} line={mode}\033[0m")
        else:
            self.logger.info(
                f"[ROUTE] mode={mode} confidence={confidence:.3f} "
                f"labels={world_labels} arousal={arousal:.2f} valence={valence:.2f}")

        return {
            "diffusion_params": diffusion_params,
            "world_labels": world_labels,
            "confidence": float(confidence),
            "direct_output": direct_output,
            "direct_reasons": reasons,
            "line_used": mode,
        }
