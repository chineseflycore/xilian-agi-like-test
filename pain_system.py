# -*- coding: utf-8 -*-
"""
pain_system.py — L3 痛觉系统（规范 §4.4 痛觉系统 / §4.5 格式化系统联动）
========================================================================
内存/显存预估（规范 §九.1）:
  - 本模块纯 CPU、纯 numpy 实现，禁止 torch（规范 §4.4）→ 显存占用 0MB（VRAM n/a）。
  - 内部状态（numpy float32，n = cfg.l3_neuron_count_eff，默认 1,000,000；
    支持 PHILIA_L3_SCALE=0.1 缩放 → 100,000）:
      · 痛觉运行计数 marks:   n × float32 = 4n 字节
      · 神经元阈值 thr:       n × float32 = 4n 字节
      · 平均连接强度 wavg:    n × float32 = 4n 字节（权重衰减的轻量镜像）
      · 半休眠掩码（派生）:   n × bool    = n 字节
    全量 1M 神经元 ≈ 13MB CPU RAM；缩放 0.1 ≈ 1.3MB。自检用小规模（200）。
  - REAL 路径（l3 存在）: 痛觉主状态由 diffuser_l3.L3Diffuser 持有
    （其权重/索引 ~400MB 见该文件头）；本模块仅委托 mark_pain / 读取 avg_pain，
    内部镜像仅在降级回退时更新，不重复持有大矩阵。

职责（规范 §4.4）:
  1. update_from_error(): 错误路径上的激活神经元子集被标记 —— 仅作用于
     MC Dropout 掩码激活的子集（比例 cfg.l3_pain_update_ratio < 10%）:
       · 痛觉标记 +1（达 cfg.l3_pain_max=10 进入“半休眠”，该神经元激活被抑制）
       · 阈值上升 cfg.l3_pain_mark_up（0.1/标记）
       · 权重衰减 cfg.l3_pain_weight_decay（委托 l3.mark_pain 或内部模拟）
  2. avg_pain: 全局平均痛觉。l3 存在时从 l3 读取，否则从内部状态计算。
  3. should_check_format(): avg_pain > cfg.pain_threshold(8.0) → True（§4.5 格式化检查）
  4. should_format():       avg_pain > cfg.pain_format_trigger(10.0) → True（§4.5 格式化）
  5. reset(): 清空内部运行计数（格式化后调用；“留疤”与否由 formatting.py 决定，
     本方法只清运行计数，不判断留疤；formatting.py 可先读 pain_marks() 再回填）。

降级策略（规范 §九.2）:
  - l3 为 None（DEMO/独立运行）: 内部 numpy 镜像完整模拟，保证流程可验证。
  - l3 存在但 mark_pain / avg_pain / reset_pain 调用失败: 记录告警并回退
    内部镜像，绝不向上抛异常。

L3 接口契约（与 diffuser_l3.L3Diffuser 对齐，另一子代理并行产出）:
  - mark_pain(indices, mark_up=None, weight_decay=None, pain_max=None)
      → 对 indices 内神经元执行: 标记 +1、阈值上升 mark_up、权重 ×(1-weight_decay)；
        标记 ≥ pain_max 进入半休眠。兼容简化签名 mark_pain(indices)。
  - avg_pain: float 属性或可调用（全局平均痛觉）；缺失时回退 l3.pain 数组均值。
  - pain: np.ndarray 形状 (n,)，痛觉标记（可选，供 avg_pain 回退与留疤导出）。
  - reset_pain(): 可选，reset() 时同步调用。

TODO(V3.4): 痛觉标记的“留疤”持久化（格式化后保留部分标记，规范 §4.5 ③）
"""
import os
import sys

# 本文件与 config.py / common_utils.py 同目录；确保其可导入，并触发
# common_utils 把 vendor/（numpy/scipy）注入 sys.path —— 必须先于 import numpy
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

import common_utils                     # noqa: E402  先导入: 自动把 vendor/(numpy/scipy) 注入 sys.path
import numpy as np                      # noqa: E402  纯 numpy，禁止 torch（规范 §4.4）
import config                           # noqa: E402  配置唯一数据源（规范 §八.1）
from common_utils import get_logger, clamp, Timer   # noqa: E402  通用工具


class PainSystem:
    """L3 痛觉系统: 错误路径上的 MC Dropout 激活神经元子集被标记（规范 §4.4）。

    与格式化系统联动（规范 §4.5）:
      · 全局平均痛觉 > 8.0  → 触发格式化检查（should_check_format）
      · 全局平均痛觉 > 10.0 → 触发格式化（should_format: “再见，昔涟”→“你好，世界”）

    设计说明 —— 痛觉标记的运行计数不封顶:
      cfg.l3_pain_max=10 是“半休眠”阈值（标记 ≥10 该神经元激活被抑制），
      不是计数器的绝对上限；运行计数继续累加（10、11、12… 疤痕加深），
      使全局平均痛觉能够越过 10.0 —— 否则均值被 10 封顶，§4.5 的格式化
      触发器（>10.0）永远不会成立，流程无法验证。
    """

    def __init__(self, cfg=None, l3=None, neuron_count=None):
        """
        cfg: Config 对象（None → config.get_config() 单例）
        l3:  L3 扩散器对象（diffuser_l3.L3Diffuser，可选）
        neuron_count: 内部镜像神经元规模（None → 优先取 l3.pain 长度，
                      否则取 cfg.l3_neuron_count_eff；自检可显式传小规模）
        """
        self.cfg = cfg if cfg is not None else config.get_config()
        self.log = get_logger("pain")
        self.l3 = l3

        # ---- 神经元规模: 显式参数 > l3 推导 > 配置 ----
        if neuron_count is None:
            neuron_count = 0
            if l3 is not None:
                try:
                    neuron_count = int(len(np.asarray(getattr(l3, "pain", []))))
                except Exception:
                    neuron_count = 0
            if not neuron_count:
                neuron_count = int(getattr(self.cfg, "l3_neuron_count_eff", 1_000_000))
        self.n = max(1, int(neuron_count))

        # ---- 内部镜像状态（DEMO/独立运行与降级回退共用）----
        self._marks = np.zeros(self.n, dtype=np.float32)      # 痛觉运行计数（不封顶，见类注释）
        lo, hi = getattr(self.cfg, "l3_threshold_range", (0.3, 0.7))
        self._threshold = np.full(self.n, (lo + hi) / 2.0, dtype=np.float32)  # 阈值初始取区间中点
        self._wavg = np.ones(self.n, dtype=np.float32)        # 平均连接强度（权重衰减的轻量镜像）

        self._rng = np.random.RandomState(20240415)           # 固定种子 → 自检可复现
        self._last_marked = 0                                 # 最近一次实际标记的神经元数（观测用）

        self.log.info("[PAIN] 初始化: n=%d 神经元, l3=%s, ratio=%.3f, pain_max=%s",
                      self.n, "接入L3" if l3 is not None else "独立模拟",
                      getattr(self.cfg, "l3_pain_update_ratio", 0.1),
                      getattr(self.cfg, "l3_pain_max", 10))

    # ================================================================
    # 核心: 错误更新（规范 §4.4）
    # ================================================================
    def update_from_error(self, neuron_indices=None, error_vector=None):
        """错误路径上的激活神经元子集被标记（规范 §4.4）。

        参数:
          neuron_indices: 错误路径上的神经元索引（int 数组/列表，可为 None）
          error_vector:   逐神经元误差向量（与 neuron_indices 二选一；给出时按
                          |error| 最大的 top-k 作为错误路径候选）

        流程:
          1. 抽取 MC Dropout 掩码激活子集（比例 cfg.l3_pain_update_ratio，<10%）
          2. marked = 掩码子集 ∩ 错误路径候选；交集为空 → 本次不标记（语义真实）
          3. 对 marked 内神经元: 痛觉 +1、阈值 +l3_pain_mark_up、
             权重 ×(1-l3_pain_weight_decay)；≥ l3_pain_max 进入半休眠

        返回: 本次实际标记的神经元数（int，恒 ≤ 10% 神经元规模）。
        """
        # ---- ① MC Dropout 掩码激活子集（<10%，规范 §4.4）----
        ratio = clamp(float(getattr(self.cfg, "l3_pain_update_ratio", 0.1)), 0.0, 1.0)
        mask_size = max(1, min(self.n, int(round(self.n * ratio))))
        mask = self._rng.choice(self.n, size=mask_size, replace=False)  # 掩码激活子集

        # ---- ② 确定错误路径候选 ----
        cand = None
        if neuron_indices is not None and len(neuron_indices):
            cand = np.asarray(neuron_indices, dtype=np.int64).ravel()
            cand = cand[(cand >= 0) & (cand < self.n)]          # 越界索引防御性过滤
        elif error_vector is not None:
            ev = np.asarray(error_vector, dtype=np.float32).ravel()
            if ev.size < self.n:
                ev = np.pad(ev, (0, self.n - ev.size))          # 短向量补零
            else:
                ev = ev[:self.n]                                # 长向量截断
            topk = min(mask_size, self.n)
            cand = np.argsort(-np.abs(ev))[:topk]               # |error| 最强 top-k
        else:
            cand = mask  # 无任何输入: 掩码自身充当错误路径（流程验证兜底）

        marked = np.intersect1d(mask, cand)
        if marked.size == 0:
            self.log.info("[PAIN] 错误路径与 MC Dropout 掩码无交集，本次不标记（交集=∅）")
            self._last_marked = 0
            return 0
        self._last_marked = int(marked.size)

        # ---- ③ 委托 L3 本体；失败 → 降级内部模拟（规范 §九.2）----
        if self.l3 is not None and self._l3_mark(marked):
            return marked.size          # L3 负责阈值上升/权重衰减/半休眠，镜像不动
        self._simulate_mark(marked)     # 独立运行 / 降级回退: 内部 numpy 模拟
        return marked.size

    # ================================================================
    # 标记执行路径
    # ================================================================
    def _l3_mark(self, marked) -> bool:
        """委托 diffuser_l3 的 mark_pain（接口契约见文件头）。

        失败（异常/缺方法/签名不匹配）→ 返回 False，由调用方回退内部模拟。
        """
        try:
            fn = getattr(self.l3, "mark_pain", None)
            if not callable(fn):
                return False
            try:
                fn(indices=marked,
                   mark_up=self.cfg.l3_pain_mark_up,
                   weight_decay=self.cfg.l3_pain_weight_decay,
                   pain_max=self.cfg.l3_pain_max)
            except TypeError:
                fn(marked)               # 兼容简化签名 mark_pain(indices)
            return True
        except Exception as e:
            self.log.warning("[PAIN] l3.mark_pain 调用失败(%s)，回退内部模拟", e)
            return False

    def _simulate_mark(self, marked):
        """内部 numpy 模拟（DEMO/独立运行，规范 §4.4）:
           - 痛觉运行计数 +1（≥ cfg.l3_pain_max=10 → 半休眠，见 half_dormant_mask）
           - 阈值 + cfg.l3_pain_mark_up（0.1/标记）
           - 平均连接强度 ×(1 - cfg.l3_pain_weight_decay)（真实稀疏权重由 L3 持有，
             此处以每神经元标量强度近似权重衰减效果）
        """
        self._marks[marked] += 1.0
        self._threshold[marked] += float(self.cfg.l3_pain_mark_up)
        self._wavg[marked] *= (1.0 - float(self.cfg.l3_pain_weight_decay))

    # ================================================================
    # 痛觉观测（规范 §4.4 / §4.5）
    # ================================================================
    @property
    def avg_pain(self) -> float:
        """全局平均痛觉（常规 0~10 尺度；极端累积可越过 10 → 触发格式化）。

        l3 存在时优先从 l3 读取（avg_pain 属性/方法，或 l3.pain 数组均值），
        读取失败 → 回退内部状态计算（规范 §九.2）。
        """
        if self.l3 is not None:
            try:
                v = getattr(self.l3, "avg_pain", None)
                if v is None:
                    pain = getattr(self.l3, "pain", None)
                    if pain is not None:
                        v = float(np.asarray(pain, dtype=np.float32).mean())
                if v is not None:
                    return float(v() if callable(v) else v)
            except Exception as e:
                self.log.warning("[PAIN] 读取 l3.avg_pain 失败(%s)，回退内部状态", e)
        return float(self._marks.mean())

    def should_check_format(self) -> bool:
        """avg_pain > cfg.pain_threshold(8.0) → True（规范 §4.5 格式化检查）。"""
        return self.avg_pain > float(self.cfg.pain_threshold)

    def should_format(self) -> bool:
        """avg_pain > cfg.pain_format_trigger(10.0) → True（规范 §4.5 触发格式化）。"""
        return self.avg_pain > float(self.cfg.pain_format_trigger)

    # ================================================================
    # 半休眠（规范 §4.4: 标记达上限 → 该神经元激活被抑制）
    # ================================================================
    @property
    def half_dormant_mask(self) -> "np.ndarray":
        """半休眠布尔掩码: 痛觉标记 ≥ cfg.l3_pain_max(10) 的神经元（激活被抑制）。"""
        return self._marks >= float(getattr(self.cfg, "l3_pain_max", 10))

    @property
    def half_dormant_count(self) -> int:
        """当前处于半休眠状态的神经元数量。"""
        return int(np.count_nonzero(self.half_dormant_mask))

    def suppress_activations(self, activations=None):
        """半休眠神经元激活被抑制（规范 §4.4）:
           - activations=None: 返回激活许可掩码（~half_dormant_mask），供调用方过滤
           - activations 给出: 返回新数组（不改动入参），半休眠位置置 0
        长度与 self.n 不一致时补零/截断防御；异常 → 原样返回（降级，规范 §九.2）。
        """
        try:
            mask = self.half_dormant_mask
            if activations is None:
                return ~mask
            act = np.asarray(activations, dtype=np.float32).ravel().copy()
            if act.size < self.n:
                act = np.pad(act, (0, self.n - act.size))
            else:
                act = act[:self.n]
            act[mask] = 0.0
            return act
        except Exception as e:
            self.log.warning("[PAIN] suppress_activations 失败(%s)，原样返回", e)
            return activations

    # ================================================================
    # 运行计数清零（规范 §4.5 格式化后调用）
    # ================================================================
    def reset(self):
        """格式化后清空运行计数（规范 §4.5）:
           - 内部镜像归零（标记清零、阈值回初始中点、连接强度回 1.0）
           - l3 存在时同步调用其 reset_pain()（若有）

           注意: 是否“留疤”由 formatting.py 决定，本方法只清运行计数；
           需保留疤痕时先读 pain_marks() 导出，reset 后按需回填。
        """
        self._marks[:] = 0.0
        lo, hi = getattr(self.cfg, "l3_threshold_range", (0.3, 0.7))
        self._threshold[:] = (lo + hi) / 2.0
        self._wavg[:] = 1.0
        self._last_marked = 0
        if self.l3 is not None:
            try:
                fn = getattr(self.l3, "reset_pain", None)
                if callable(fn):
                    fn()
            except Exception as e:
                self.log.warning("[PAIN] l3.reset_pain 调用失败(%s)", e)

    def pain_marks(self) -> "np.ndarray":
        """导出痛觉标记副本（供 formatting.py 读取“留疤”，规范 §4.5 ③）。

        l3 存在时优先导出 l3.pain 的副本，否则导出内部镜像副本。
        """
        if self.l3 is not None:
            try:
                pain = getattr(self.l3, "pain", None)
                if pain is not None:
                    return np.asarray(pain, dtype=np.float32).copy()
            except Exception:
                pass
        return self._marks.copy()

    # ================================================================
    # 观测辅助
    # ================================================================
    @property
    def last_marked(self) -> int:
        """最近一次 update_from_error 实际标记的神经元数（观测用）。"""
        return int(self._last_marked)

    def summary(self) -> str:
        """启动/运行日志摘要（供 main.py / line_controller 调用）。"""
        return ("[PAIN] avg=%.3f check=%s format=%s 半休眠=%d/%d"
                % (self.avg_pain, self.should_check_format(), self.should_format(),
                   self.half_dormant_count, self.n))


if __name__ == "__main__":
    # ================================================================
    # 轻量自检（规范 §九.3 启动自检的子集，不参与正式运行流程）
    # 模拟 200 个神经元，多次 update_from_error，打印 avg_pain /
    # should_check_format / should_format 的演变。
    # ================================================================
    # 兼容 GBK/UTF-8 控制台: 输出一律 UTF-8，无法编码的字符以 ? 替换，避免自检崩溃
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("=" * 70)
    print("pain_system.py 自检: L3 痛觉系统（规范 §4.4 / §4.5 联动）")
    print("=" * 70)

    cfg = config.get_config()
    used_fields = {k: getattr(cfg, k) for k in (
        "pain_threshold", "pain_format_trigger", "l3_pain_max", "l3_pain_mark_up",
        "l3_pain_weight_decay", "l3_pain_update_ratio", "l3_neuron_count_eff",
        "l3_threshold_range")}
    print("[SELF] 用到的 config 字段:", used_fields)

    N = 200
    ps = PainSystem(cfg, l3=None, neuron_count=N)
    mask_size = max(1, int(round(N * cfg.l3_pain_update_ratio)))
    print("[SELF] 初始: avg_pain=%.3f check=%s format=%s 半休眠=%d"
          " (n=%d, 掩码规模=%d = %.1f%%)"
          % (ps.avg_pain, ps.should_check_format(), ps.should_format(),
             ps.half_dormant_count, ps.n, mask_size, 100.0 * mask_size / N))

    # ---- 场景 A: 错误路径 = 前 75% 神经元，连续 200 步标记 ----
    err_path = np.arange(0, int(N * 0.75))
    trigger_at = None
    with Timer("pain_update_loop") as tm:
        for step in range(1, 201):
            ps.update_from_error(neuron_indices=err_path)
            if ps.should_format() and trigger_at is None:
                trigger_at = step
            if step % 20 == 0:
                star = " ★格式化区" if (trigger_at is not None and step >= trigger_at) else ""
                print("[SELF] step=%3d | avg_pain=%6.3f | check=%s | format=%s"
                      " | 半休眠=%3d | 本次标记=%2d%s"
                      % (step, ps.avg_pain, ps.should_check_format(), ps.should_format(),
                         ps.half_dormant_count, ps.last_marked, star))
    print("[SELF] 200 步 update_from_error 总耗时: %.1f ms（平均 %.3f ms/步）"
          % (tm.elapsed, tm.elapsed / 200.0))
    print("[SELF] should_check_format() 首次为 True 于 avg>%.1f（§4.5 格式化检查）"
          % cfg.pain_threshold)
    print("[SELF] should_format() 首次为 True 于 step=%s（avg>%.1f → 触发格式化，§4.5）"
          % (trigger_at, cfg.pain_format_trigger))
    assert trigger_at is not None and ps.avg_pain > cfg.pain_format_trigger
    assert ps.should_check_format() and ps.should_format()
    assert ps.half_dormant_count > 0, "应有神经元进入半休眠"
    print("[SELF] 检查点 A 通过: 标记累积 → avg_pain 越过 8.0/10.0，半休眠生效")

    # ---- 场景 B: error_vector 路径（无显式索引）+ <10% 子集约束 ----
    ps2 = PainSystem(cfg, l3=None, neuron_count=N)
    ev = np.zeros(N, dtype=np.float32)
    ev[::5] = 3.0                                    # 周期性强误差（5 的倍数神经元）
    for _ in range(30):
        ps2.update_from_error(error_vector=ev)
    max_marked = 0
    for _ in range(50):                              # 无输入: 掩码自身充当错误路径
        max_marked = max(max_marked, ps2.update_from_error())
    print("[SELF] error_vector 路径: avg_pain=%.3f check=%s format=%s"
          "（误差集中在 5 的倍数神经元）"
          % (ps2.avg_pain, ps2.should_check_format(), ps2.should_format()))
    assert max_marked <= mask_size, "单次标记数不得超过 <10% 子集"
    print("[SELF] 检查点 B 通过: 单次标记数恒 ≤ %d（<10%% 子集约束，§4.4）" % max_marked)

    # ---- 场景 C: reset() 清空运行计数（格式化后，§4.5）----
    before = ps.avg_pain
    ps.reset()
    print("[SELF] reset(): avg_pain %.3f → %.3f, check=%s format=%s, 半休眠=%d"
          % (before, ps.avg_pain, ps.should_check_format(), ps.should_format(),
             ps.half_dormant_count))
    assert ps.avg_pain == 0.0 and not ps.should_format() and not ps.should_check_format()

    # ---- 场景 D: l3 委托路径（模拟 diffuser_l3 的痛觉接口）----
    class _MockL3:
        """模拟 diffuser_l3.L3Diffuser 的痛觉接口（真实文件由另一子代理并行产出）。"""

        def __init__(self, n):
            self.pain = np.zeros(n, dtype=np.float32)
            self.threshold = np.full(n, 0.5, dtype=np.float32)
            self._w = np.ones(n, dtype=np.float32)
            self.reset_calls = 0

        @property
        def avg_pain(self):
            return float(self.pain.mean())

        def mark_pain(self, indices=None, mark_up=None, weight_decay=None, pain_max=None):
            idx = np.asarray(indices, dtype=np.int64)
            self.pain[idx] = np.minimum(self.pain[idx] + 1.0, float(pain_max or 10.0))
            self.threshold[idx] += float(mark_up or 0.1)
            self._w[idx] *= (1.0 - float(weight_decay or 0.02))
            return int(idx.size)

        def reset_pain(self):
            self.reset_calls += 1
            self.pain[:] = 0.0
            self.threshold[:] = 0.5

    mock = _MockL3(N)
    ps3 = PainSystem(cfg, l3=mock, neuron_count=N)
    ps3.update_from_error(neuron_indices=np.arange(0, 60))
    print("[SELF] l3 委托路径: avg_pain=%.3f（来自 l3） | 被标记神经元=%d | 阈值峰值=%.2f"
          % (ps3.avg_pain, int(np.count_nonzero(mock.pain)), mock.threshold.max()))
    assert ps3.avg_pain > 0.0 and mock.pain.max() == 1.0 and ps3.avg_pain == mock.avg_pain
    ps3.reset()
    print("[SELF] l3 委托 reset(): mock.reset_calls=%d, mock.pain.sum=%.1f"
          % (mock.reset_calls, float(mock.pain.sum())))
    assert mock.reset_calls == 1 and mock.pain.sum() == 0.0

    # ---- 场景 E: 半休眠激活抑制 ----
    ps4 = PainSystem(cfg, l3=None, neuron_count=N)
    for _ in range(120):
        ps4.update_from_error()                      # 无输入: 全量掩码标记 → 快速积累
    act = np.ones(N, dtype=np.float32)
    act_sum_before = float(act.sum())
    suppressed = ps4.suppress_activations(act)
    print("[SELF] 半休眠激活抑制: 半休眠=%d 个, 抑制前激活和=%.1f → 抑制后=%.1f"
          % (ps4.half_dormant_count, act_sum_before, float(suppressed.sum())))
    assert float(suppressed.sum()) == float(N - ps4.half_dormant_count)
    assert float(act.sum()) == act_sum_before, "suppress_activations 不得改动入参"

    print("=" * 70)
    print("[SELF] 全部检查点通过 [OK]  pain_system.py 自检结束")
    print("=" * 70)
