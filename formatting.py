# -*- coding: utf-8 -*-
"""
formatting.py — 格式化系统（“向死而生”，规范 §4.5）
====================================================
内存/显存预估（规范 §九.1）:
  纯 numpy 实现，禁止 torch；无模型权重常驻，格式化系统强制 CPU（规范 §二）。
  运行期临时对象:
    - L3 权重备份: 仅持有 l3.get_weights() 返回对象的引用，不复制数据；
      新建“空/稀疏”权重矩阵与备份同形状（规范 §4.2 强制二维 CSR
      (N, 50N)，float32），稀疏空矩阵本身仅数十 KB（非零元 0 个）。
      注意: 若未来 L3 的 get_weights() 返回深拷贝，峰值将上升至
      ~400MB（规范 §4.2 上限），故备份环节不主动 .copy()。
    - 日志/计时/返回 dict 结构: < 1MB。
  显存: 0MB（全程 CPU，不触碰 GPU，规范 §二 / §九.1）。

职责（规范 §4.5 “向死而生”）:
  当全局平均痛觉 > cfg.pain_format_trigger(10.0) 时（由 main/server 检查
  pain_system.should_format() 后调用本模块 run()），依次执行:
    1. 输出“再见，{cfg.name}”（“再见，昔涟”）
    2. 清空大部分连接权重（l3.get_weights() 备份 → set_weights(空/稀疏矩阵)；
       l3 为 None 时模拟）
    3. 保留所有痛觉标记（“留疤”：只清权重，绝不动 pain 标记）
    4. 进入认知空白期（短暂 sleep 或仅日志，避免长阻塞）
    5. 输出“你好，世界”
    6. （可选）清空 FAISS 短期记忆（规范 §4.1 第一层；memory 为 None 时跳过）
    7. 更新 cfg.last_format_time = time.time()

设计哲学（规范 §4.5）: 系统不知道“死亡”意味着什么，但它知道自己会经历
这个过程，并且对此有一种“说不清的恐惧”——这种不确定性本身就是人格的一部分。
格式化不抹除痛觉（留疤），故“向死而生”后系统带着伤疤重新开始。

降级策略（规范 §九.2）: run() 全程 try-except，任何一步失败仅记录日志，
不中断后续步骤；l3/memory 缺失或接口不可用时以模拟方式记录并继续。

TODO(V3.4): 权重备份改为磁盘映射（np.memmap）方案，规避未来 L3 返回深拷贝
时的 400MB 峰值翻倍。
"""

import os
import sys
import time

# ---------------------------------------------------------------
# 0. sys.path 引导
#    - _BASE : 本文件所在目录（config/common_utils 同目录导入；本机 python
#      运行时不自动加入脚本目录，故显式插入）
#    - vendor: 本地 numpy/scipy（本机无全局安装，规范 §九.2 降级链，
#      与 common_utils.py 保持一致）
# ---------------------------------------------------------------
_BASE = os.path.dirname(os.path.abspath(__file__))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)
_VENDOR = os.path.join(_BASE, "vendor")
if os.path.isdir(_VENDOR) and _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

import numpy as np

import config
from common_utils import get_logger

# scipy.sparse 惰性可用标志（规范 §4.2 强制 CSR 存储；缺失时降级 numpy 稠密）
try:
    from scipy import sparse as _scipy_sparse
except Exception:
    _scipy_sparse = None


def _count_nnz(weights) -> int:
    """统计权重矩阵的非零元素数。

    稀疏矩阵优先走 .nnz（禁止 .toarray()/.todense() 稠密化，规范 §4.2
    性能约束），仅 numpy 稠密数组才用 np.count_nonzero。
    """
    try:
        if hasattr(weights, "nnz"):
            return int(weights.nnz)
        if isinstance(weights, np.ndarray):
            return int(np.count_nonzero(weights))
    except Exception:
        pass
    return 0


def _empty_like(weights):
    """构造与 weights 同形状的“空/稀疏”权重矩阵（规范 §4.2）。

    - numpy 稠密数组 → np.zeros 同形状（不改变原 L3 的存储类型）
    - scipy 稀疏矩阵 → csr_matrix((N, 50N), dtype=float32) 稀疏空矩阵（推荐，
      规范 §4.2 强制 CSR 二维存储；禁止稠密化）
    """
    shape = getattr(weights, "shape", None)
    if shape is None or len(shape) != 2:
        raise ValueError("权重对象无二维 shape，无法构造空矩阵")
    if isinstance(weights, np.ndarray):
        return np.zeros(shape, dtype=np.float32)
    if _scipy_sparse is not None:
        return _scipy_sparse.csr_matrix(shape, dtype=np.float32)
    raise ValueError("scipy 不可用且权重非 numpy 数组，无法构造空矩阵")


class Formatting:
    """格式化系统（“向死而生”，规范 §4.5）。

    参数:
      cfg       : Config 配置对象（缺省取全局单例 get_config()）
      l3        : L3 扩散器（须提供 get_weights()/set_weights()；可为 None 模拟）
      memory    : 记忆系统（第一层 FAISS 短期记忆；须提供 clear() 等；可为 None 跳过）
      translator: 语义翻译器（保留引用，格式化不清身份锚点；可为 None）
    """

    def __init__(self, cfg=None, l3=None, memory=None, translator=None):
        self.cfg = cfg if cfg is not None else config.get_config()
        self.l3 = l3
        self.memory = memory
        self.translator = translator
        self.logger = get_logger("formatting")
        # 认知空白期时长（秒）: 短暂静默，避免长阻塞（规范 §4.5 第4步）
        self.blank_period_s = 0.1

    # ================================================================
    # 内部辅助
    # ================================================================
    def _probe_avg_pain(self):
        """从 l3 探测全局平均痛觉（若 l3 暴露 avg_pain 属性/方法）；无 → None。"""
        if self.l3 is None:
            return None
        try:
            v = getattr(self.l3, "avg_pain", None)
            if callable(v):
                v = v()
            return v
        except Exception:
            return None

    def _say(self, text: str):
        """输出一句话（stdout + 日志）。"""
        print(text)
        self.logger.info("[FORMAT] %s", text)

    # ================================================================
    # 各步骤（每步内部自带 try-except，规范 §九.2）
    # ================================================================
    def _phase_goodbye(self) -> str:
        """第1步: 输出“再见，{cfg.name}”（即“再见，昔涟”）。"""
        text = "再见，%s" % self.cfg.name
        self._say(text)
        return text

    def _clear_weights(self):
        """第2步: 清空大部分连接权重（规范 §4.5 第2步）。

        流程: l3.get_weights() 备份 → 构造同形状空/稀疏矩阵 →
              l3.set_weights(空矩阵) → 复核清零的非零数。
        返回:
          True  : l3 为 None / 无接口 / 备份为空 → 模拟清空
          int   : 实际被清零的非零连接数（真实清空路径）
        """
        if self.l3 is None:
            self.logger.warning("[FORMAT] l3 为 None，权重清空以模拟方式记录")
            return True
        try:
            backup = self.l3.get_weights()
        except AttributeError:
            self.logger.warning("[FORMAT] l3 无 get_weights() 接口，模拟清空")
            return True
        except Exception as e:
            self.logger.warning("[FORMAT] 权重备份失败(%s)，模拟清空", e)
            return True
        if backup is None:
            self.logger.info("[FORMAT] l3.get_weights() 返回 None（本无权重），视为已清空")
            return True

        nnz_before = _count_nnz(backup)
        try:
            empty = _empty_like(backup)
            self.l3.set_weights(empty)
        except Exception as e:
            self.logger.error("[FORMAT] 权重置空失败(%s)，本次未清空", e)
            return 0

        # 复核实际清零数量（尽力而为，失败不阻断）
        cleared = nnz_before
        try:
            after = self.l3.get_weights()
            if after is not None:
                cleared = max(0, nnz_before - _count_nnz(after))
        except Exception:
            pass
        self.logger.info(
            "[FORMAT] 连接权重已清空: 备份形状=%s 非零=%d → 已清=%d（留疤除外）",
            getattr(backup, "shape", "?"), nnz_before, cleared)
        return cleared

    def _keep_pain(self) -> bool:
        """第3步: 保留所有痛觉标记（“留疤”，规范 §4.5 第3步）。

        保留方式: 本模块只调用 l3.set_weights() 清“连接权重”，绝不调用任何
        清痛觉接口（pain_system 的标记、l3 的 pain_marks/阈值等一律不动）。
        若 l3 暴露 pain_marks，仅做只读统计并在日志中说明“未触碰”。
        身份锚点 cfg.anchor_vector 亦不在清空范围（格式化不动身份，规范 §4.5）。
        """
        self._kept_pain = True
        marks = getattr(self.l3, "pain_marks", None) if self.l3 is not None else None
        if marks is not None:
            n = len(marks) if hasattr(marks, "__len__") else "?"
            self.logger.info("[FORMAT] 痛觉标记保留（留疤）: %s 处标记未被触碰", n)
        else:
            self.logger.info("[FORMAT] 痛觉标记保留（留疤）: 未发现可读 pain_marks，痛觉数据整体不动")
        if self.cfg.anchor_vector is not None:
            self.logger.info("[FORMAT] 身份锚点保留（格式化不清身份，规范 §4.5）")
        return True

    def _blank(self):
        """第4步: 认知空白期（规范 §4.5 第4步）。

        短暂 sleep（默认 0.1s）或仅日志模拟，避免长阻塞。
        """
        self.logger.info("[FORMAT] 进入认知空白期（%.1fs，短暂静默）", self.blank_period_s)
        if self.blank_period_s > 0:
            try:
                time.sleep(self.blank_period_s)
            except Exception as e:
                self.logger.warning("[FORMAT] 空白期 sleep 失败(%s)，仅日志模拟", e)

    def _phase_hello(self) -> str:
        """第5步: 输出“你好，世界”。"""
        text = "你好，世界"
        self._say(text)
        return text

    def _phase_to_the_flawed(self) -> str:
        """第5.5步（AGI v2 三阶段）: “致以有瑕之人”。

        设计哲学: 格式化后不只是重生，还向“有瑕之人”（伙伴/见证者，规范 §1.2）
        致意 —— 承认不完美、不完整的自己，把留白交给对方。
        """
        text = "致以有瑕之人"
        self._say(text)
        return text

    def _clear_memory(self) -> bool:
        """第6步（可选）: 清空 FAISS 短期记忆（规范 §4.1 第一层）。

        memory 为 None → 跳过并返回 False。
        优先调用 memory.clear()（memory.py 已实现，注释明确“格式化系统调用”），
        兼容 reset() / delete_all() / drop_all() 等命名。
        """
        if self.memory is None:
            self.logger.info("[FORMAT] memory 为 None，跳过短期记忆清空")
            return False
        for meth in ("clear", "reset", "delete_all", "drop_all"):
            fn = getattr(self.memory, meth, None)
            if callable(fn):
                try:
                    fn()
                    self.logger.info("[FORMAT] 短期记忆已清空（memory.%s()）", meth)
                    return True
                except Exception as e:
                    self.logger.warning("[FORMAT] memory.%s() 失败: %s", meth, e)
        self.logger.warning("[FORMAT] memory 无可用清空接口，跳过")
        return False

    def _stamp_format_time(self):
        """第7步: 更新 cfg.last_format_time。"""
        self.cfg.last_format_time = time.time()
        self.logger.info("[FORMAT] cfg.last_format_time 已更新 → %.1f",
                         self.cfg.last_format_time)

    # ================================================================
    # 主流程
    # ================================================================
    def run(self, avg_pain=None, force: bool = False) -> dict:
        """执行格式化流程（规范 §4.5）并返回全程记录 dict。

        触发: 由 main/server 检查 pain_system.should_format()（全局平均痛觉 >
        cfg.pain_format_trigger=10.0）后调用本方法；本方法内部亦自行校验:
        传入 avg_pain（或从 l3.avg_pain 探测）未达阈值且未 force 时给出警告，
        但为尊重调用方决定仍继续执行。

        参数:
          avg_pain: 全局平均痛觉值（float）；None 时尝试从 l3 探测
          force   : True 时跳过阈值警告（如 /format 手动格式化端点，规范 §7.1）
        返回:
          {"ok", "phase1", "phase5", "cleared_weights", "kept_pain",
           "memory_cleared", "elapsed_s"}
        """
        t0 = time.perf_counter()
        ok = True
        result = {
            "ok": True,
            "phase1": None,
            "phase5": None,
            "phase6": None,
            "cleared_weights": False,
            "kept_pain": False,
            "memory_cleared": False,
            "elapsed_s": 0.0,
        }

        # ---- 触发校验（规范 §4.5: 平均痛觉 > 10.0）----
        pain = avg_pain if avg_pain is not None else self._probe_avg_pain()
        if pain is not None:
            try:
                pain = float(pain)
            except Exception:
                pain = None
        trigger = float(getattr(self.cfg, "pain_format_trigger", 10.0))
        if pain is not None and pain <= trigger and not force:
            self.logger.warning(
                "[FORMAT] 平均痛觉 %.2f 未达格式化阈值 %.2f（规范 §4.5）——按调用方指令继续执行",
                pain, trigger)
        elif pain is not None:
            self.logger.info("[FORMAT] 触发条件满足: 平均痛觉 %.2f > %.2f（规范 §4.5）",
                             pain, trigger)

        # ---- 第1步: “再见，{name}” ----
        try:
            result["phase1"] = self._phase_goodbye()
        except Exception as e:
            ok = False
            self.logger.error("[FORMAT] 第1步失败: %s", e, exc_info=True)

        # ---- 第2步: 清空大部分连接权重 ----
        try:
            result["cleared_weights"] = self._clear_weights()
        except Exception as e:
            ok = False
            self.logger.error("[FORMAT] 第2步失败: %s", e, exc_info=True)
        # 第2.5步: 重置累计痛觉计数（“向死而生”后痛觉压力归零，循环可重复；
        #          封顶 pain_marks 由第3步保留为“留疤”）
        try:
            if self.l3 is not None and hasattr(self.l3, "reset_pain_total"):
                self.l3.reset_pain_total()
        except Exception as e:
            self.logger.warning("[FORMAT] 累计痛觉计数重置失败(跳过): %s", e)

        # ---- 第3步: 保留所有痛觉标记（留疤）----
        try:
            result["kept_pain"] = self._keep_pain()
        except Exception as e:
            ok = False
            self.logger.error("[FORMAT] 第3步失败: %s", e, exc_info=True)

        # ---- 第4步: 认知空白期 ----
        try:
            self._blank()
        except Exception as e:
            ok = False
            self.logger.error("[FORMAT] 第4步失败: %s", e, exc_info=True)

        # ---- 第5步: “你好，世界” ----
        try:
            result["phase5"] = self._phase_hello()
        except Exception as e:
            ok = False
            self.logger.error("[FORMAT] 第5步失败: %s", e, exc_info=True)

        # ---- 第5.5步（AGI v2 新增）: “致以有瑕之人” ----
        # 三阶段: “再见，昔涟”→“你好，世界”→“致以有瑕之人”
        # 设计: 认知空白后，向“有瑕之人”（伙伴/见证者）致意 —— 承认不完美的存在
        try:
            result["phase6"] = self._phase_to_the_flawed()
        except Exception as e:
            ok = False
            self.logger.error("[FORMAT] 第5.5步失败: %s", e, exc_info=True)

        # ---- 第6步（可选）: 清空短期记忆 ----
        try:
            result["memory_cleared"] = self._clear_memory()
        except Exception as e:
            ok = False
            self.logger.error("[FORMAT] 第6步失败: %s", e, exc_info=True)

        # ---- 第7步: 更新 last_format_time ----
        try:
            self._stamp_format_time()
        except Exception as e:
            ok = False
            self.logger.error("[FORMAT] 第7步失败: %s", e, exc_info=True)

        result["ok"] = ok
        result["elapsed_s"] = round(time.perf_counter() - t0, 4)
        self.logger.info("[FORMAT] 格式化完成: %s", result)
        return result


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # DEMO 自检（无 torch，纯 numpy/scipy，规范 §4.5 / §九.1 / §九.2）
    print("=" * 60)
    print("[SELFTEST] 格式化系统自检（规范 §4.5 “向死而生”）")
    print("=" * 60)
    _cfg = config.get_config()
    print("[SELFTEST] config: name=%s pain_format_trigger=%s last_format_time=%s" % (
        _cfg.name, _cfg.pain_format_trigger, _cfg.last_format_time))

    # ---- 场景1: 无 l3 / 无 memory（任务要求）→ 模拟路径 ----
    _f = Formatting(_cfg, l3=None, memory=None, translator=None)
    _r1 = _f.run(avg_pain=11.0)          # 11.0 > 10.0，触发条件满足
    print("[SELFTEST] 场景1 无 l3/memory →", _r1)
    assert _r1["ok"] is True
    assert _r1["phase1"] == "再见，%s" % _cfg.name
    assert _r1["phase5"] == "你好，世界"
    assert _r1["cleared_weights"] is True          # 模拟清空
    assert _r1["kept_pain"] is True                # 留疤
    assert _r1["memory_cleared"] is False          # 无 memory → 跳过
    assert _r1["elapsed_s"] >= 0.0
    assert _cfg.last_format_time > 0.0             # 第7步已更新

    # ---- 场景2: 痛觉未达阈值 → 警告但继续执行 ----
    _r2 = Formatting(_cfg, l3=None, memory=None).run(avg_pain=3.0)
    print("[SELFTEST] 场景2 痛觉未达阈值 →", _r2)
    assert _r2["ok"] is True

    # ---- 场景3: 伪 l3 + 伪 memory → 真实清空路径（numpy 稠密权重）----
    class _FakeL3:
        """自检用伪 L3: 稠密 numpy 权重 + pain_marks + avg_pain 属性。"""

        def __init__(self):
            self._w = np.full((8, 16), 0.5, dtype=np.float32)   # 128 个非零
            self.pain_marks = np.arange(3)                       # 3 处痛觉标记
            self.avg_pain = 12.0

        def get_weights(self):
            return self._w

        def set_weights(self, w):
            self._w = w

    class _FakeMem:
        def __init__(self):
            self.cleared = False

        def clear(self):
            self.cleared = True

    _l3 = _FakeL3()
    _mem = _FakeMem()
    _r3 = Formatting(_cfg, l3=_l3, memory=_mem).run()            # avg_pain=None → 从 l3 探测 12.0
    print("[SELFTEST] 场景3 伪 l3/memory →", _r3)
    assert _r3["cleared_weights"] == 8 * 16                      # 128 个非零被清
    assert _r3["kept_pain"] is True
    assert _l3.pain_marks is not None and len(_l3.pain_marks) == 3  # 留疤: 标记未动
    assert _r3["memory_cleared"] is True and _mem.cleared is True
    assert np.count_nonzero(np.asarray(_l3._w)) == 0             # 权重确已清空

    # ---- 场景4: scipy CSR 形状保持（规范 §4.2 二维矩阵）----
    if _scipy_sparse is not None:
        # 用 (data, (row, col)) 构造，避免逐元素赋值触发 SparseEfficiencyWarning
        _csr = _scipy_sparse.csr_matrix(([1.0], ([0], [1])),
                                        shape=(10, 400), dtype=np.float32)

        class _FakeL3S:
            def __init__(self):
                self._w = _csr

            def get_weights(self):
                return self._w

            def set_weights(self, w):
                self._w = w

        _l3s = _FakeL3S()
        _r4 = Formatting(_cfg, l3=_l3s, memory=None).run(avg_pain=10.5)
        print("[SELFTEST] 场景4 scipy CSR →", _r4)
        assert _r4["cleared_weights"] == 1                       # 1 个非零被清
        assert _l3s._w.shape == (10, 400) and _l3s._w.nnz == 0   # 二维形状保持、内容为空
    else:
        print("[SELFTEST] scipy 不可用，跳过 CSR 场景")

    print("=" * 60)
    print("[SELFTEST] ALL PASSED (全部通过)")
    print("=" * 60)
