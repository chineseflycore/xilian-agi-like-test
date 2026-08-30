# -*- coding: utf-8 -*-
"""
perception_fusion.py — 多模态注意力融合层（规范 §3.4）
=====================================================
内存/显存预估（规范 §九.1）:
  - 投影矩阵: 3 模态 × 3 组(Q/K/V) × 512×512 float32 ≈ 9.4MB（CPU RAM）
  - 注意力中间缓冲: K×512 float32 ≈ 6KB（CPU RAM，K ≤ 3）
  - 显存: 0MB —— 纯 numpy 实现，CPU 运行，不占用 GPU 显存（规范 §二）

融合公式（规范 §3.4）:
  文本(512) + 听觉(512) + 视觉(512)
    → 各模态投影到 query/key/value
    → softmax 注意力加权（query = 各模态 query 的全局综合向量）
    → 512 维状态向量（L2 归一化）

跳过规则:
  任一输入为 None 或零向量（范数 < 1e-6）→ 自动跳过该模态
  全部跳过 → 返回 512 维零向量
  单模态 → 注意力退化为恒等（权重=1），输出该模态投影后归一化

降级策略（规范 §九.2）: 全程 try-except，异常 → 512 维零向量。

TODO(V3.4): 接入可训练投影（配合路由层微调）；输出注意力权重供可视化/诊断
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

import config
from common_utils import get_logger, normalize


class FusionLayer:
    """多模态注意力融合层（纯 numpy 实现，无 torch 依赖）。"""

    # 模态投影初始化随机种子（进程内稳定）
    _SEED = 20240502
    # 零向量判定阈值（范数低于该值视为“无此模态”）
    _ZERO_EPS = 1e-6

    def __init__(self, cfg=None, dim: int = 512):
        self.cfg = cfg if cfg is not None else config.get_config()
        self.dim = int(dim)
        self.log = get_logger("perception.fusion")
        rng = np.random.RandomState(self._SEED)
        scale = 1.0 / np.sqrt(self.dim)
        # 每模态独立 Q/K/V 投影（随机初始化；TODO(V3.4): 训练可学习投影）
        self.proj = {}
        for m in ("text", "audio", "visual"):
            self.proj[m] = {
                "q": rng.normal(0.0, scale, (self.dim, self.dim)).astype(np.float32),
                "k": rng.normal(0.0, scale, (self.dim, self.dim)).astype(np.float32),
                "v": rng.normal(0.0, scale, (self.dim, self.dim)).astype(np.float32),
            }
        # 缩放因子 √d 防止 softmax 饱和
        self.attn_scale = 1.0 / np.sqrt(self.dim)

    # ================================================================
    def fuse(self, text_vec=None, audio_vec=None, visual_vec=None):
        """注意力融合: 文本(512) + 听觉(512) + 视觉(512) → 512 维。

        任一输入为 None 或零向量 → 自动跳过该模态（规范 §3.4 边界）。
        返回: np.ndarray (512,)，float32，L2 归一化。
        """
        try:
            # ① 收集有效模态（None / 维度不匹配 / 零向量 → 跳过）
            mods = []
            for name, vec in (("text", text_vec),
                              ("audio", audio_vec),
                              ("visual", visual_vec)):
                if vec is None:
                    continue
                v = np.asarray(vec, dtype=np.float32).reshape(-1)
                if v.size != self.dim:
                    self.log.warning("[FUSE] %s 模态维度 %d ≠ %d，跳过",
                                     name, v.size, self.dim)
                    continue
                if float(np.linalg.norm(v)) < self._ZERO_EPS:
                    self.log.info("[FUSE] %s 模态为零向量，自动跳过", name)
                    continue
                mods.append((name, v))
            # ② 无有效模态 → 零向量
            if not mods:
                self.log.info("[FUSE] 无有效模态，返回 512 维零向量")
                return np.zeros(self.dim, dtype=np.float32)
            # ③ 投影到 query/key/value
            q = np.stack([self.proj[m]["q"] @ v for m, v in mods])   # (K, D)
            k = np.stack([self.proj[m]["k"] @ v for m, v in mods])   # (K, D)
            vv = np.stack([self.proj[m]["v"] @ v for m, v in mods])  # (K, D)
            # ④ 全局查询向量 = 各模态 query 均值（当前“综合意图”）
            q_ref = q.mean(axis=0)                                   # (D,)
            scores = (q_ref @ k.T) * self.attn_scale                 # (K,)
            # 数值稳定 softmax（减最大值防溢出）
            exp = np.exp(scores - float(scores.max()))
            weights = exp / exp.sum()                                # (K,)
            # ⑤ 注意力加权输出 → 归一化
            out = (weights @ vv).astype(np.float32)                  # (D,)
            out = normalize(out)
            names = [m for m, _ in mods]
            self.log.info("[FUSE] 模态=%s 注意力权重=%s", names,
                          [round(float(w), 3) for w in weights])
            return out
        except Exception as e:
            # 关键路径 try-except 降级（规范 §九.2）
            self.log.error("[FUSE] 融合异常，降级为 512 维零向量: %s", e, exc_info=True)
            return np.zeros(self.dim, dtype=np.float32)


if __name__ == "__main__":
    # 轻量自检（不参与正式运行流程）
    f = FusionLayer()
    rng = np.random.RandomState(3)
    t = normalize(rng.normal(0.0, 1.0, 512))
    a = normalize(rng.normal(0.0, 1.0, 512))
    v = normalize(rng.normal(0.0, 1.0, 512))
    out3 = f.fuse(t, a, v)
    print("[SELF] 三模态 →", out3.shape, "范数=", round(float(np.linalg.norm(out3)), 3))
    out1 = f.fuse(t, None, None)
    print("[SELF] 单模态 →", out1.shape, "范数=", round(float(np.linalg.norm(out1)), 3))
    out0 = f.fuse(None, None, None)
    print("[SELF] 全空   →", out0.shape, "范数=", round(float(np.linalg.norm(out0)), 3))
