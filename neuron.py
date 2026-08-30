# -*- coding: utf-8 -*-
"""
neuron.py — L3 神经元元数据管理（规范 §4.2 L3 神经元特化网络 / §4.4 痛觉系统）
=============================================================================
内存/显存预估（规范 §九.1）—— 1,000,000 神经元 × 5 个 numpy 元数据数组（CPU 常驻）:
  thresholds       float32 × 1M ≈ 4.0 MB   （0.3~0.7 均匀随机阈值，规范 §4.2）
  pain_marks       int32   × 1M ≈ 4.0 MB   （0~10 痛觉标记，达 l3_pain_max 进入半休眠，规范 §4.4）
  forget_counters  int32   × 1M ≈ 4.0 MB   （遗忘计数器：未激活则累积，驱动连接遗忘衰减）
  last_activation  float64 × 1M ≈ 8.0 MB   （最后激活时间戳，用于时间衰减判断）
  half_dormant     bool    × 1M ≈ 1.0 MB   （派生：pain_marks >= l3_pain_max，按需计算）
  ─────────────────────────────────────────
  合计 ≈ 21 MB RAM；无 GPU 显存占用（L3 为纯 CPU 网络，规范 §4.2）。
  连接权重矩阵（二维 CSR (N, N*50)）的存储估算见 diffuser_l3.py 顶部注释。

职责（供 diffuser_l3.py 使用）:
  1. 阈值初始化：l3_threshold_range=(0.3, 0.7) 区间均匀分布（float32）
  2. 痛觉标记更新：标记 +1（上限 l3_pain_max=10）→ 达上限的神经元进入“半休眠”
     （半休眠神经元在扩散中跳过激活，规范 §4.4“标记累积到 10 时进入半休眠”）
  3. 遗忘衰减：未激活神经元遗忘计数器累积，越过遗忘窗口的神经元返回给调用方
     做权重衰减（连接权重衰减由 diffuser_l3.py 在 CSR 上向量化执行）
  4. 格式化支持（规范 §4.5）：“清空权重、保留痛觉标记（留疤）” → reset_counts()

实现约束（规范 §十.20）:
  · 纯 numpy 数组存储元数据（禁止 Python 对象列表）
  · 关键路径 try-except 降级（规范 §九.2）
  · 禁止 torch（本机仅 numpy/scipy）
"""

import os
import sys

# 本地引导：确保本文件目录与 vendor 目录（numpy/scipy 离线依赖）可被导入
_BASE = os.path.dirname(os.path.abspath(__file__))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)
_VENDOR = os.path.join(_BASE, "vendor")
if os.path.isdir(_VENDOR) and _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

import numpy as np

from config import get_config
from common_utils import get_logger


# ----------------------------------------------------------------------
# 模块级辅助函数（可直接独立使用）
# ----------------------------------------------------------------------
def init_thresholds(count: int, lo: float = 0.3, hi: float = 0.7,
                    rng: np.random.Generator = None) -> np.ndarray:
    """初始化 count 个神经元的激活阈值：lo~hi 区间均匀分布（规范 §4.2）。

    返回 float32 一维数组。rng 缺省时使用独立可复现的生成器。
    """
    if rng is None:
        rng = np.random.default_rng(20240601)
    thr = rng.uniform(float(lo), float(hi), size=int(count)).astype(np.float32)
    # 防御：确保阈值落在合法区间内（浮点边界抖动保护）
    np.clip(thr, float(lo), float(hi), out=thr)
    return thr


def update_pain_marks(pain_marks: np.ndarray, indices,
                      pain_max: int = 10) -> np.ndarray:
    """痛觉标记批量 +1，上限钳制为 pain_max（规范 §4.4）。

    参数:
      pain_marks: int32 一维数组（0~pain_max）
      indices   : 本次被标记的神经元索引（会自动去重、剔除越界）
      pain_max  : 标记上限（cfg.l3_pain_max=10）
    返回:
      本次新达到上限（半休眠）的神经元索引数组 —— 调用方据此做
      “阈值上升 / 权重衰减 / 跳过激活”等后续处理。
    """
    idx = np.asarray(indices, dtype=np.int64).reshape(-1)
    idx = np.unique(idx[idx >= 0])
    if idx.size == 0:
        return np.empty(0, dtype=np.int64)
    pm = pain_marks
    n = pm.size
    idx = idx[idx < n]
    if idx.size == 0:
        return np.empty(0, dtype=np.int64)
    # 只对未达上限的神经元 +1（已满的保持 10，即“留疤”上限不再叠加）
    sub = idx[pm[idx] < pain_max]
    if sub.size:
        pm[sub] = np.minimum(pm[sub] + 1, pain_max)
    # 本次新达上限者 = 标记后恰好等于 pain_max 的神经元
    newly = sub[pm[sub] >= pain_max]
    return newly.astype(np.int64, copy=False)


def forgetting_decay(forget_counters: np.ndarray, active_mask: np.ndarray,
                     forget_window: int = 100) -> np.ndarray:
    """遗忘计数更新（规范 §4.2 每神经元含“遗忘计数器”）。

    · 本次激活（active_mask=True）的神经元：遗忘计数器清零，last_activation 刷新
    · 未激活的神经元：遗忘计数器 +1
    · 计数器越过 forget_window 的神经元视为“遗忘临界”：计数器复位，
      返回其索引 —— 调用方应对其连接权重施加小幅度衰减（遗忘衰减）。
    """
    counters = forget_counters
    act = np.asarray(active_mask, dtype=bool)
    counters[act] = 0
    if (~act).any():
        counters[~act] = counters[~act] + 1
    ripe = np.flatnonzero(counters >= forget_window)
    if ripe.size:
        counters[ripe] = 0
    return ripe.astype(np.int64, copy=False)


# ----------------------------------------------------------------------
# NeuronMeta：L3 神经元元数据容器（diffuser_l3.py 使用的主入口）
# ----------------------------------------------------------------------
class NeuronMeta:
    """L3 神经元元数据管理：阈值 / 痛觉标记 / 遗忘计数器 / 最后激活时间。

    全部元数据以 numpy 数组存储（禁止 Python 对象列表，规范 §十.20），
    供 diffuser_l3.py 的扩散、Hebbian 更新、痛觉与遗忘流程向量化使用。

    半休眠判定（规范 §4.4）：pain_marks >= pain_max 的神经元不参与扩散激活，
    由 half_dormant 属性按需计算（N 元素布尔比较，1M 规模 < 1ms）。
    """

    def __init__(self, count: int, threshold_range=(0.3, 0.7), pain_max: int = 10,
                 forget_window: int = 100, rng: np.random.Generator = None,
                 logger=None):
        self.logger = logger or get_logger("neuron")
        self.count = int(count)
        self.pain_max = int(pain_max)
        self.forget_window = int(forget_window)
        self.threshold_range = tuple(threshold_range)
        if rng is None:
            rng = np.random.default_rng(20240601)
        # 阈值：0.3~0.7 均匀分布（规范 §4.2）
        self._thresholds = init_thresholds(self.count, *self.threshold_range, rng=rng)
        # 痛觉标记：0~10（规范 §4.4），int32 便于向量化比较与累加
        self._pain_marks = np.zeros(self.count, dtype=np.int32)
        # 遗忘计数器：未激活累积（规范 §4.2）
        self._forget_counters = np.zeros(self.count, dtype=np.int32)
        # 最后激活时间：-1 表示从未激活（规范 §4.2 / §4.7.4 时间衰减）
        self._last_activation = np.full(self.count, -1.0, dtype=np.float64)
        self.logger.info(
            "[NEURON] 元数据就绪 count=%d pain_max=%d forget_window=%d 内存≈%.1fMB",
            self.count, self.pain_max, self.forget_window, self.memory_bytes() / 1048576.0)

    # ------------------------------------------------------------------
    # 属性访问
    # ------------------------------------------------------------------
    @property
    def thresholds(self) -> np.ndarray:
        return self._thresholds

    @thresholds.setter
    def thresholds(self, value: np.ndarray):
        """设置阈值数组（load_state 恢复用；校验长度与类型）。"""
        value = np.asarray(value, dtype=np.float32)
        if value.size != self.count:
            raise ValueError("thresholds 长度 %d ≠ 神经元数 %d" % (value.size, self.count))
        self._thresholds = value.reshape(-1)

    @property
    def pain_marks(self) -> np.ndarray:
        return self._pain_marks

    @pain_marks.setter
    def pain_marks(self, value: np.ndarray):
        """设置痛觉标记（load_state 恢复"留疤"用）。"""
        value = np.asarray(value, dtype=np.int32)
        if value.size != self.count:
            raise ValueError("pain_marks 长度 %d ≠ 神经元数 %d" % (value.size, self.count))
        self._pain_marks = value.reshape(-1)

    @property
    def forget_counters(self) -> np.ndarray:
        return self._forget_counters

    @property
    def last_activation(self) -> np.ndarray:
        return self._last_activation

    @property
    def half_dormant(self) -> np.ndarray:
        """半休眠掩码：痛觉标记已达上限的神经元（扩散中跳过激活，规范 §4.4）。"""
        return self._pain_marks >= self.pain_max

    # ------------------------------------------------------------------
    # 痛觉标记（规范 §4.4）
    # ------------------------------------------------------------------
    def mark_pain(self, indices) -> np.ndarray:
        """对指定神经元痛觉标记 +1，返回本次新达上限（半休眠）的索引。

        阈值上升（l3_pain_mark_up）与权重衰减（l3_pain_weight_decay）由
        diffuser_l3.py 依据本方法返回的索引执行（涉及 CSR 权重矩阵）。
        """
        return update_pain_marks(self._pain_marks, indices, self.pain_max)

    def relieve_pain(self, indices, amount: int = 1):
        """痛觉缓解：标记 -amount（下限 0），供“奖励/纠正”路径使用（可选扩展）。

        注意：半休眠神经元缓解后重新参与激活（pain_marks < pain_max）。
        """
        idx = np.asarray(indices, dtype=np.int64).reshape(-1)
        idx = np.unique(idx[(idx >= 0) & (idx < self.count)])
        if idx.size:
            self._pain_marks[idx] = np.maximum(self._pain_marks[idx] - int(amount), 0)

    # ------------------------------------------------------------------
    # 遗忘衰减（规范 §4.2 / §4.7.4 时间衰减）
    # ------------------------------------------------------------------
    def apply_forgetting(self, active_mask: np.ndarray, now: float = None) -> np.ndarray:
        """按本轮激活掩码更新遗忘计数器与最后激活时间。

        参数:
          active_mask: 本轮实际激活（电位超过阈值）的神经元布尔掩码
          now        : 当前时间戳（默认 time.time()），写入 last_activation
        返回:
          越过遗忘窗口、等待调用方做权重衰减的神经元索引；
          其遗忘计数器已复位（避免反复触发）。
        """
        if now is None:
            import time
            now = time.time()
        # 激活神经元刷新最后激活时间
        act = np.asarray(active_mask, dtype=bool)
        if act.any():
            self._last_activation[act] = float(now)
        # 未激活神经元累积遗忘计数，越过窗口者返回
        ripe = forgetting_decay(self._forget_counters, act, self.forget_window)
        if ripe.size:
            self.logger.info("[NEURON] %d 个神经元进入遗忘临界（权重待衰减）", ripe.size)
        return ripe

    # ------------------------------------------------------------------
    # 格式化支持（规范 §4.5：“清空权重、保留痛觉标记”）
    # ------------------------------------------------------------------
    def reset_counts(self):
        """清空遗忘计数器与最后激活时间，但保留痛觉标记（“留疤”）。

        格式化系统（formatting.py）清空连接权重时调用本方法，
        保证痛觉记忆在“向死而生”流程后依然存在（规范 §4.5 第 3 条）。
        """
        self._forget_counters.fill(0)
        self._last_activation.fill(-1.0)
        self.logger.info("[NEURON] 遗忘/激活计数已清空，痛觉标记保留（留疤）")

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------
    def memory_bytes(self) -> int:
        """元数据数组占用内存（字节）。"""
        return int(self._thresholds.nbytes + self._pain_marks.nbytes
                   + self._forget_counters.nbytes + self._last_activation.nbytes)

    def summary(self) -> str:
        """一行统计摘要（启动日志 / 自检用）。"""
        pm = self._pain_marks
        return ("[NEURON] count=%d 阈值=%.3f~%.3f 平均痛觉=%.3f 半休眠=%d 内存=%.1fMB"
                % (self.count, float(self._thresholds.min()), float(self._thresholds.max()),
                   float(pm.mean()), int(self.half_dormant.sum()),
                   self.memory_bytes() / 1048576.0))


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 自检：小规模神经元（2000），保证秒级完成（规范 §九.1 / §十.3）
    _cfg = get_config()
    _rng = np.random.default_rng(42)
    _meta = NeuronMeta(2000, threshold_range=_cfg.l3_threshold_range,
                       pain_max=_cfg.l3_pain_max, rng=_rng)
    print("[SELFTEST]", _meta.summary())

    # 1) 阈值初始化区间检查（规范 §4.2: 0.3~0.7）
    lo, hi = _cfg.l3_threshold_range
    assert float(_meta.thresholds.min()) >= lo - 1e-4 and float(_meta.thresholds.max()) <= hi + 1e-4
    print("[SELFTEST] 阈值区间 OK: min=%.4f max=%.4f" %
          (float(_meta.thresholds.min()), float(_meta.thresholds.max())))

    # 2) 痛觉标记：标记 100 个神经元 ×10 次（l3_pain_max）→ 全部达上限进入半休眠（规范 §4.4）
    _idx = np.arange(100, dtype=np.int64)
    _newly = np.empty(0, dtype=np.int64)
    for _ in range(_cfg.l3_pain_max):
        _newly = _meta.mark_pain(_idx)
    assert _newly.size == 100, "第 %d 轮应新达上限 100 个，实际 %d" % (_cfg.l3_pain_max, _newly.size)
    assert int(_meta.pain_marks[_idx].min()) == _cfg.l3_pain_max
    assert int(_meta.half_dormant.sum()) == 100
    print("[SELFTEST] 痛觉标记 OK: 100 神经元达上限=%d，半休眠=%d，平均痛觉=%.3f" %
          (_cfg.l3_pain_max, int(_meta.half_dormant.sum()), float(_meta.pain_marks.mean())))

    # 3) 遗忘衰减：前 50 个神经元标记为未激活并累积超过窗口 → 返回遗忘临界索引
    _act = np.ones(2000, dtype=bool)
    _act[:50] = False
    _ripe = np.empty(0, dtype=np.int64)
    for _ in range(_meta.forget_window + 1):
        _r = _meta.apply_forgetting(_act, now=float(_))
        if _r.size:
            _ripe = _r   # 越过窗口的那一轮返回临界索引（随后计数器复位）
    assert set(_ripe.tolist()) == set(range(50))
    # 临界者计数器已复位（随后最后一个循环轮次仅 +1；未复位则应为 101）
    assert int(_meta.forget_counters[:50].max()) <= 1
    print("[SELFTEST] 遗忘衰减 OK: %d 个神经元进入遗忘临界，计数器已复位" % _ripe.size)

    # 4) 格式化支持：reset_counts 保留痛觉标记、清空遗忘/激活（规范 §4.5）
    _meta.reset_counts()
    assert int(_meta.pain_marks[_idx].min()) == _cfg.l3_pain_max   # 留疤
    assert int(_meta.forget_counters.sum()) == 0
    assert float(_meta.last_activation.max()) == -1.0
    print("[SELFTEST] reset_counts OK: 痛觉标记保留（留疤），遗忘/激活已清空")
    print("[SELFTEST] neuron.py 全部通过 OK")
