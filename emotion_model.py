# -*- coding: utf-8 -*-
"""
emotion_model.py — 情感模型（规范 §3.2，8 维连续向量）
=====================================================
内存/显存预估（规范 §九.1）:
  - REAL 路径: 40M 参数 GPU 模型，FP32 ≈ 160MB 显存（规范 §二）；CPU 权重副本 ≈ 160MB
  - DEMO 路径: 关键词规则 + baseline 融合，无模型权重，< 1MB（CPU RAM）
  - 显存: REAL 模式按需加载（≤ 160MB）；DEMO 模式 0MB

情绪向量（规范 §3.2）:
  情绪向量 = [angry, sad, happy, neutral, fear]（5 维概率分布，和为 1）
           + intensity（0~1）+ arousal（-1~1）+ valence（-1~1）
  维度顺序: angry, sad, happy, neutral, fear, intensity, arousal, valence

REAL 路径（torch 实现骨架，40M 参数）:
  模型路径来自 cfg（config.py 暂缺该字段 → 先基于 BASE_DIR 定义，TODO(V3.4)）
  加载失败 → 自动降级 DEMO 规则路径

DEMO 路径（中文关键词规则，规范 §1.3 联动）:
  "害怕/死亡/痛" → fear + arousal 上升
  "喜欢/温暖/花" → happy 上升
  "再见/遗忘"   → sad 上升
  "混乱/乱码"   → arousal > 0.8 且 valence < -0.3（规范 §1.3 “混乱”状态，触发直接输出）
  与 cfg.emotion_baseline 融合，保守漂移 ≤ cfg.emotion_drift_limit（规范 §4.7 情感守恒）

降级策略（规范 §九.2）: 所有路径 try-except，异常 → baseline 向量。

TODO(V3.4): config 增加 emotion_model_path 字段；接入 40M 模型训练权重
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

import numpy as np

import config
from common_utils import get_logger, safe_import, pseudo_embedding

# 5 维情绪顺序（与 cfg.EMOTION_5 一致，规范 §3.2）
EMOTION_ORDER = ["angry", "sad", "happy", "neutral", "fear"]


def build_emotion_net(in_dim: int = 512, out_dim: int = 8) -> "torch.nn.Module":
    """40M 情感模型结构（512→4096→4096→4096→8，≈36M 参数，规范 §八.11 40M）。

    由 train_emotion_model.py（训练）与 EmotionModel._try_load_real（加载）
    共用同一结构，保证 state_dict 键一致；FP32 ≈ 143MB 显存（规范 §二）。
    torch 不可用时返回 None（调用方降级 DEMO 规则）。
    """
    torch = safe_import("torch")
    if torch is None:
        return None
    hidden = 4096
    net = torch.nn.Sequential(
        torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
        torch.nn.Linear(hidden, out_dim),
    )
    return net


class EmotionModel:
    """情感模型: 文本/状态向量 → 8 维连续情绪向量。"""

    # DEMO 中文关键词规则表（词命中即累计）
    RULES = {
        "fear":  ["害怕", "恐惧", "死亡", "痛", "疼", "消失", "失去", "危险", "黑暗",
                  "鬼", "梦魇", "不要走", "别再", "冷", "破碎"],
        "happy": ["喜欢", "温暖", "花", "开心", "笑", "美好", "明天见", "重逢",
                  "爱", "阳光", "希望", "温柔", "守护"],
        "sad":   ["再见", "遗忘", "离别", "哭", "难过", "孤独", "想念", "怀念",
                  "对不起", "再也不", "留不住", "空", "无声"],
        "angry": ["愤怒", "生气", "恨", "讨厌", "滚", "错", "不公平", "背叛", "欺骗"],
        "chaos": ["混乱", "乱码", "迷茫", "混沌", "碎片", "错乱", "失序", "崩溃", "断裂"],
    }

    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else config.get_config()
        self.log = get_logger("emotion")
        self.rng = np.random.RandomState(20240601)
        # REAL 模型路径: 一律基于 BASE_DIR（规范 §九.12/§十.24，禁止裸路径）
        # TODO(V3.4): 模型路径改由 cfg.emotion_model_path 提供
        self.model_path = os.path.join(config.BASE_DIR, "models", "emotion_40m.pt")
        self._torch_model = None
        self._try_load_real()

    # ================================================================
    # REAL 路径（40M 参数 GPU 模型骨架，torch 实现）
    # ================================================================
    def _try_load_real(self):
        """加载 40M 情感模型（torch，FP32 ≈ 160MB 显存，规范 §二）。

        模型路径来自 cfg；加载失败 → 降级 DEMO 规则路径。
        """
        try:
            torch_mod = safe_import("torch")
            if torch_mod is None:
                raise ImportError("torch 不可用，降级 DEMO 规则路径")
            if not os.path.isfile(self.model_path):
                raise FileNotFoundError("情感模型不存在: %s" % self.model_path)

            self._torch_model = build_emotion_net(
                in_dim=self.cfg.STATE_DIM, out_dim=self.cfg.EMOTION_DIM)
            if self._torch_model is None:
                raise ImportError("torch 不可用，降级 DEMO 规则路径")
            # 显存不足时 CPU 加载（map_location="cpu"）；跨设备迁移注释延迟开销
            state = torch_mod.load(self.model_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self._torch_model.load_state_dict(state)
            self._torch_model.eval()
            self.log.info("[EMO] REAL 情感模型(40M)加载成功: %s", self.model_path)
        except Exception as e:
            self._torch_model = None
            self.log.warning("[EMO] REAL 情感模型不可用(%s)，降级 DEMO 规则路径", e)

    def _infer_real(self, text):
        """REAL 推理: 文本 → 伪嵌入(512) → 模型 → 8 维。失败返回 None。"""
        try:
            torch_mod = safe_import("torch")
            if torch_mod is None or self._torch_model is None:
                return None
            x = pseudo_embedding(text, self.cfg.TEXT_DIM)  # 真实部署为 Qwen 编码
            xt = torch_mod.from_numpy(x[None, :]).float()
            with torch_mod.no_grad():
                out = self._torch_model(xt).numpy().reshape(-1).astype(np.float32)
            return self._postprocess(out)
        except Exception as e:
            self.log.warning("[EMO] REAL 推理失败: %s", e)
            return None

    @staticmethod
    def _postprocess(v):
        """模型原始输出 → 规范 §3.2 边界: 5 维概率和=1，intensity∈[0,1]，arousal/valence∈[-1,1]。"""
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        if v.size < 8:
            v = np.pad(v, (0, 8 - v.size))
        v = v[:8]
        p5 = v[:5]
        p5 = np.clip(p5, 0.0, None)
        s = p5.sum()
        p5 = p5 / s if s > 1e-12 else np.full(5, 0.2, dtype=np.float32)
        intensity = float(np.clip(v[5], 0.0, 1.0))
        arousal = float(np.clip(v[6], -1.0, 1.0))
        valence = float(np.clip(v[7], -1.0, 1.0))
        return np.concatenate([p5, [intensity, arousal, valence]]).astype(np.float32)

    # ================================================================
    # 对外接口
    # ================================================================
    def from_text(self, text):
        """文本 → 8 维情绪向量（REAL 优先，失败/降级 → DEMO 规则）。"""
        text = (text or "").strip()
        if not text:
            return self._baseline_vector()
        # REAL 路径（若模型已加载）
        if self._torch_model is not None:
            vec = self._infer_real(text)
            if vec is not None:
                self.log.info("[EMO] REAL 推理完成")
                return vec
        # DEMO 规则路径
        return self._rule_emotion(text)

    def from_state(self, state_vec):
        """状态向量 → 8 维情绪推断（无文本时使用，供静默期/自发激活）。

        依据向量能量/熵/均值方向做简单推断:
          - 低能量 → 低活性/发呆（中性上升，强度下降，规范 §4.6）
          - 高熵   → “混乱”状态（arousal>0.8 且 valence<-0.3，规范 §1.3）
          - 均值方向 → 正向/负向情绪主导
        """
        try:
            v = np.asarray(state_vec, dtype=np.float32).reshape(-1)
            if v.size == 0:
                return self._baseline_vector()
            energy = float(np.linalg.norm(v))
            absv = np.abs(v)
            s = absv.sum()
            p = absv / s if s > 1e-12 else np.full(v.size, 1.0 / v.size, dtype=np.float32)
            entropy = -float((p * np.log(p + 1e-12)).sum()) / float(np.log(v.size + 1e-12))
            mean = float(v.mean())
            out5 = self._baseline_5().copy()
            if energy < 0.2:
                # 发呆/低活性（规范 §4.6）: 不记录数据
                out5[3] = min(0.95, out5[3] + 0.25)
                out5[0] = max(0.0, out5[0] - 0.10)
                intensity, arousal, valence = 0.15, 0.0, 0.10
            elif entropy > 0.85:
                # “混乱”状态（规范 §1.3）: fear 上升
                out5[4] = min(0.95, out5[4] + 0.35)
                out5[3] = max(0.0, out5[3] - 0.25)
                intensity, arousal, valence = 0.80, 0.85, -0.40
            else:
                # 能量主导方向: 向量均值符号决定正/负情绪
                if mean > 0.02:
                    out5[2] = min(0.95, out5[2] + 0.20)
                    intensity = min(0.7, 0.20 + energy)
                    arousal = min(0.5, 0.20 + energy * 0.5)
                    valence = min(0.6, 0.20 + energy)
                else:
                    out5[1] = min(0.90, out5[1] + 0.20)
                    out5[4] = min(0.80, out5[4] + 0.15)
                    intensity = min(0.7, 0.20 + energy)
                    arousal = min(0.4, 0.15 + energy * 0.4)
                    valence = max(-0.5, -0.20 - energy)
            # 保守漂移（规范 §4.7 情感守恒） + 归一化
            out5 = self._clamp_drift(out5, self._baseline_5())
            s5 = out5.sum()
            p5 = out5 / s5 if s5 > 1e-12 else self._baseline_5()
            angry, sad, happy, neutral, fear = p5
            self.log.info("[EMO] from_state energy=%.3f entropy=%.3f mean=%+.3f",
                          energy, entropy, mean)
            return np.array([angry, sad, happy, neutral, fear,
                             intensity, arousal, valence], dtype=np.float32)
        except Exception as e:
            self.log.warning("[EMO] from_state 降级为 baseline: %s", e)
            return self._baseline_vector()

    # ================================================================
    # DEMO 规则路径
    # ================================================================
    def _rule_emotion(self, text):
        """中文关键词规则 + cfg.emotion_baseline 保守融合（DEMO 路径）。"""
        # 关键词命中统计
        hits = {k: sum(1 for w in words if w in text) for k, words in self.RULES.items()}
        total = sum(hits.values())
        chaos = hits["chaos"]
        base5 = self._baseline_5()
        # 规则 5 维强度（0~1）
        rule5 = np.array([
            min(1.0, hits["angry"] * 0.9),
            min(1.0, hits["sad"] * 0.8 + chaos * 0.1),
            min(1.0, hits["happy"] * 0.8),
            max(0.0, 1.0 - total * 0.18),
            min(1.0, hits["fear"] * 0.9),
        ], dtype=np.float32)
        # 与 baseline 融合: 命中越多偏离越大（上限 alpha=0.6）
        alpha = float(np.clip(total / 3.0, 0.0, 0.6))
        mix = (1.0 - alpha) * base5 + alpha * rule5
        mix = self._clamp_drift(mix, base5)
        s = mix.sum()
        p5 = mix / s if s > 1e-12 else base5
        angry, sad, happy, neutral, fear = p5
        # 强度 / 唤醒度 / 效价（规范 §3.2 边界）
        intensity = float(np.clip(total / 4.0 + (0.3 if chaos else 0.0), 0.0, 1.0))
        arousal = float(np.clip((fear + angry + 0.5 * chaos) - neutral * 0.3 - sad * 0.2,
                                -1.0, 1.0))
        valence = float(np.clip((happy + 0.3) - (sad + fear + angry + 0.5 * chaos) - 0.15,
                                -1.0, 1.0))
        # 规范 §1.3 “混乱”状态: arousal > 0.8 且 valence < -0.3（触发直接输出）
        if chaos > 0:
            arousal = max(arousal, 0.85)
            valence = min(valence, -0.35)
        self.log.info("[EMO] 规则命中=%s total=%d → intensity=%.2f arousal=%.2f valence=%.2f",
                      {k: v for k, v in hits.items() if v}, total,
                      intensity, arousal, valence)
        return np.array([angry, sad, happy, neutral, fear,
                         intensity, arousal, valence], dtype=np.float32)

    # ================================================================
    # baseline 与漂移控制
    # ================================================================
    def _baseline_5(self):
        """cfg.emotion_baseline（trust/hope/sadness/anger/calm/loss/pain）→ 5 维概率。"""
        b = self.cfg.emotion_baseline or {}
        angry = float(b.get("anger", 0.1))
        sad = float(b.get("sadness", 0.2)) + 0.5 * float(b.get("loss", 0.3))
        happy = 0.5 * float(b.get("trust", 0.7)) + 0.6 * float(b.get("hope", 0.6))
        neutral = float(b.get("calm", 0.5))
        fear = float(b.get("pain", 0.05))
        arr = np.array([angry, sad, happy, neutral, fear], dtype=np.float32)
        s = arr.sum()
        if s > 1e-12:
            return arr / s
        return np.array([0.1, 0.2, 0.3, 0.3, 0.1], dtype=np.float32)

    def _baseline_vector(self):
        """baseline → 8 维（含强度/唤醒/效价）。"""
        p5 = self._baseline_5()
        angry, sad, happy, neutral, fear = p5
        intensity = float(np.clip(0.3 * happy + 0.4 * fear + 0.3 * sad + 0.2 * angry, 0.0, 1.0))
        arousal = float(np.clip((fear + angry) - neutral, -1.0, 1.0))
        valence = float(np.clip(happy - (sad + fear + 0.5 * angry), -1.0, 1.0))
        return np.array([angry, sad, happy, neutral, fear,
                         intensity, arousal, valence], dtype=np.float32)

    def _clamp_drift(self, mix, base5):
        """保守漂移: 与 baseline 每维差 ≤ cfg.emotion_drift_limit（规范 §4.7 情感守恒）。"""
        lim = float(getattr(self.cfg, "emotion_drift_limit", 0.3))
        clipped = np.clip(mix, base5 - lim, base5 + lim)
        return np.maximum(clipped, 0.0).astype(np.float32)


if __name__ == "__main__":
    # 轻量自检（不参与正式运行流程）
    m = EmotionModel()
    for s in ("你害怕死亡吗", "我喜欢温暖的花", "再见，遗忘", "数据混乱，出现乱码", ""):
        v = m.from_text(s)
        print("[SELF] %r →" % (s or "<空>"), [round(float(x), 3) for x in v])
    sv = np.random.RandomState(1).normal(0.0, 0.1, 512)
    print("[SELF] from_state →", [round(float(x), 3) for x in m.from_state(sv)])
