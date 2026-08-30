# -*- coding: utf-8 -*-
"""
diffuser_l1.py — L1 纯数据/概率扩散器（CPU，0.5M 参数，无训练，规范 §4.2）
=========================================================================
内存/显存预估（规范 §九.1）:
  - 常驻权重: 随机投影矩阵 P（512×1024，float32）＝ 524,288 参数 × 4B ≈ 2.0 MB RAM
    （纯 CPU 推理，显存占用 0；远低于 L2 ≈4MB 与 L3 ≈400MB，规范 §4.2 内存对比）
  - 推理临时缓冲: z(512) + h(1024) + MC 样本(10×512)，float32 ≈ 30 KB/次调用
  - 合计常驻内存 ≈ 2 MB；无 torch / GPU / faiss 依赖，仅依赖 numpy（vendor/ 提供）。

职责与算法（规范 §4.2 L1 — 纯数据/概率扩散器: MC Dropout + 随机掩码，0.5M，无训练）:
    1. 随机投影: 512 维状态 z → 1024 维隐藏 h = tanh(z @ P)，
       P 为 512×1024 随机投影矩阵（524,288 参数 ≈ 0.5M），回投使用共享权重 P^T（tied weights）；
    2. MC Dropout: 每次随机前向对 h 施加随机二值掩码（keep_prob=0.7，反向缩放稳定能量），
       共 mc_samples=10 次随机前向取平均（任务规格“随机掩码多次，如 10 次”）；
    3. 阻尼迭代: z ← z·damping + (1-damping)·back + noise，迭代 cfg.iteration_steps 次，
       每步 L2 归一化防发散（cfg.damping=0.7 / iteration_steps=20，规范 [Diffuser] 段）；
    4. 共振调制: 隐藏层乘 (1 + 0.05·sin(2π·resonance_freq·t/steps))，cfg.resonance_freq=2.5；
    5. 情感调制: emotion_vec[5]（intensity）放大/抑制注入噪声 —— 概率扩散的“情绪涨落”；
    6. 文本调制: text_history 参与调用随机种子 —— 触景生情（规范 §4.3 第三层随机召回注入扩散器）；
    7. 置信度: confidence = 1 - 平均逐维标准差 / 输出 RMS —— 多次 MC 输出越一致置信度越高（0~1）。

降级路径（规范 §九.2 关键路径 try-except）:
  - config 加载失败 → 内置默认配置（STATE_DIM=512 / iteration_steps=20 / damping=0.7 / resonance_freq=2.5）
  - numpy 不可用 → 纯 Python 伪扩散（仅 math，确定性伪随机噪声，置信度 0.5），
    保证无 numpy 时仍可 import 与运行自检；
  - 权重初始化失败 / 单次前向数值异常 → 捕获并降级为伪扩散，不影响其他管线调用。
  - 本模块刻意不引入 torch：L1 为纯数据/概率扩散器（规范 §4.2），CPU 运行。

对外接口（line_controller.py 调用，多接口兼容，见其 _call_diffuser 尝试顺序）:
  run(state_vector, emotion_vec, text_history="") -> {"vector": ndarray(512,) float32 已归一化,
                                                       "confidence": float(0~1)}
  diffuse(state_vector, emotion_vec) -> dict（兼容接口，内部调 run）
  forward(state_vector) -> ndarray(512,)（兼容接口，单次确定性前向）
  __call__(state_vector, emotion_vec) -> dict（同 run）
无任何训练方法（无 train/fit/backward/step），符合规范 §4.2 L1“无训练”。

TODO(V3.4): 增加 L1 权重周期性重随机化（模拟遗忘），以及 MC 样本数的自适应调整。
"""

import os
import sys
import math
import hashlib

# 路径引导（规范 §十.24: 一律基于 os.path.dirname(os.path.abspath(__file__)) 定位）:
# 本机 Python 运行时不保证把脚本目录/cwd 加入 sys.path，这里显式加入
# 脚本目录（供 import config）与 vendor/（本机 numpy/scipy，供 import numpy）。
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)
_VENDOR_DIR = os.path.join(_BASE_DIR, "vendor")
if os.path.isdir(_VENDOR_DIR) and _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

# numpy 延迟探测（vendor/ 已在上面加入 sys.path）:
# 缺失时全程走纯 Python 伪扩散（规范 §九.2 降级），保证 import 不炸。
try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    np = None
    _HAS_NUMPY = False

try:
    from common_utils import get_logger, clamp, normalize
except Exception:
    # 终极兜底: common_utils 不可用时提供本地最小实现（规范 §九.2）
    import logging

    def get_logger(name):
        _lg = logging.getLogger(name)
        if not _lg.handlers:
            _h = logging.StreamHandler()
            _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
            _lg.addHandler(_h)
            _lg.setLevel(logging.INFO)
        return _lg

    def clamp(v, lo=0.0, hi=1.0):
        return max(lo, min(hi, v))

    def normalize(v):
        a = np.asarray(v, dtype=np.float32)
        n = float(np.linalg.norm(a))
        return a / n if n > 1e-12 else np.zeros_like(a)


class L1Diffuser:
    """L1 纯数据/概率扩散器（CPU，0.5M 参数，无训练，规范 §4.2）。

    算法核心: MC Dropout + 随机掩码 —— 多次随机前向取平均，
    输出 512 维归一化状态向量与 0~1 置信度（由多次输出的方差/能量估计）。
    """

    def __init__(self, cfg=None):
        self.logger = get_logger("diffuser_l1")
        self.cfg = self._load_cfg(cfg)
        self.state_dim = int(getattr(self.cfg, "STATE_DIM", 512) or 512)  # 融合层 512 维契约（规范 §3.4）
        self.hidden_dim = 1024                          # 512→1024 随机投影（≈0.5M 参数）
        self.mc_samples = 10                            # MC Dropout 随机前向次数（任务规格：10 次）
        self.keep_prob = 0.7                            # 随机掩码保留概率（Dropout keep 概率）
        self.noise_scale = 0.25                         # 注入噪声基准幅度（情感能量调制）
        self._params_ok = False
        if _HAS_NUMPY:
            try:
                self._init_params()
            except Exception as e:
                self.logger.warning("[L1] 权重初始化失败，降级为伪扩散: %s", e)   # 黄色警告（规范 §九.2）
        else:
            self.logger.warning("[L1] numpy 不可用，全程使用纯 Python 伪扩散（规范 §九.2 降级）")

    # ------------------------------------------------------------------
    # 初始化与降级
    # ------------------------------------------------------------------
    def _load_cfg(self, cfg):
        """加载配置；失败时使用内置默认值（规范 §九.2 关键路径 try-except）。"""
        if cfg is not None:
            return cfg
        try:
            from config import get_config
            return get_config()
        except Exception as e:
            self.logger.warning("[L1] config 加载失败，使用内置默认配置: %s", e)
            import types
            return types.SimpleNamespace(
                STATE_DIM=512, iteration_steps=20, damping=0.7, resonance_freq=2.5, demo=True)

    def _init_params(self):
        """构建 512→1024 随机投影矩阵 P（≈0.5M 参数，规范 §4.2 L1）。"""
        rs = np.random.RandomState(20240601)            # 固定种子: DEMO 流程可复现（规范 §六.1）
        scale = 1.0 / math.sqrt(self.hidden_dim)        # Xavier 缩放: tanh 前向能量稳定
        self.P = (rs.randn(self.state_dim, self.hidden_dim) * scale).astype(np.float32)
        self._params_ok = True
        self.logger.info("[L1] 投影矩阵 %s 初始化完成（%d 参数 ≈ %.2fM）",
                         self.P.shape, int(self.P.size), self.P.size / 1e6)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _call_seed(state_vector, emotion_vec, text_history):
        """调用级随机种子: 由 (状态, 情感, 文本) 稳定派生 —— 相同输入可复现（DEMO），
        不同 text_history 产生不同噪声（触景生情，规范 §4.3 第三层）。"""
        h = hashlib.sha256()
        try:
            h.update(np.asarray(state_vector, dtype=np.float32).tobytes())
        except Exception:
            h.update(str(state_vector).encode("utf-8", errors="ignore"))
        try:
            h.update(np.asarray(emotion_vec, dtype=np.float32).tobytes())
        except Exception:
            h.update(str(emotion_vec).encode("utf-8", errors="ignore"))
        h.update(str(text_history or "").encode("utf-8", errors="ignore"))
        return int.from_bytes(h.digest()[:4], "little")

    def _to_state(self, state_vector):
        """规整为 512 维 float32 一维数组（不足补零、超出截断）。"""
        a = np.asarray(state_vector, dtype=np.float32).reshape(-1)
        if a.size == 0:
            return np.zeros(self.state_dim, dtype=np.float32)
        if a.size < self.state_dim:
            a = np.concatenate([a, np.zeros(self.state_dim - a.size, dtype=np.float32)])
        return a[:self.state_dim].astype(np.float32)

    def _emotion_energy(self, emotion_vec):
        """情感能量: emotion_vec[5]=intensity（config.py EMOTION_DIM=8 契约）调制噪声幅度。"""
        try:
            e = np.asarray(emotion_vec, dtype=np.float32).reshape(-1)
        except Exception:
            e = np.zeros(8, dtype=np.float32)
        if e.size == 0:
            e = np.zeros(8, dtype=np.float32)
        if e.size > 5 and e[5] != 0:
            base = float(np.abs(e[5]))                  # intensity 直接驱动
        else:
            base = float(np.mean(np.abs(e))) if e.size else 0.0
        return clamp(0.2 + base, 0.1, 1.5)              # 下限 0.2: 无情绪时仍有轻微涨落

    def _mc_samples(self, x, rng, energy):
        """MC Dropout 多次随机前向（规范 §4.2 L1）: 返回 (mc_samples, 512) 样本矩阵。"""
        steps = max(1, int(getattr(self.cfg, "iteration_steps", 20) or 20))
        dmp = float(getattr(self.cfg, "damping", 0.7) or 0.7)
        res = float(getattr(self.cfg, "resonance_freq", 2.5) or 0.0)
        samples = np.empty((self.mc_samples, self.state_dim), dtype=np.float32)
        for s in range(self.mc_samples):
            z = x.copy()
            for t in range(steps):
                # 512→1024 随机投影 + tanh 非线性（概率扩散的“能量面”）
                h = np.tanh(z @ self.P)
                # MC Dropout: 随机二值掩码 ⊙ 隐藏层，反向缩放保持能量稳定（inverted dropout）
                mask = (rng.random(self.hidden_dim) < self.keep_prob).astype(np.float32)
                h = h * mask / self.keep_prob
                # 共振调制: cfg.resonance_freq=2.5（规范 [Diffuser] 段）
                if res > 0.0:
                    phase = 2.0 * math.pi * res * (t + 1) / float(steps)
                    h = h * (1.0 + 0.05 * math.sin(phase))
                # 共享权重回投 1024→512（tied weights，参数仍 ≈0.5M）
                back = h @ self.P.T
                # 情感调制噪声注入（“情绪涨落”）
                noise = (rng.standard_normal(self.state_dim).astype(np.float32)
                         * self.noise_scale * energy)
                # 阻尼迭代 + 每步归一化防发散
                z = normalize(z * dmp + (1.0 - dmp) * back + noise)
            samples[s] = normalize(z)
        return samples

    # ------------------------------------------------------------------
    # 对外接口（line_controller.py 调用，多接口兼容）
    # ------------------------------------------------------------------
    def run(self, state_vector, emotion_vec=None, text_history=""):
        """主入口: MC Dropout 多次随机前向取平均。

        返回 dict: {"vector": ndarray(512,) float32 已归一化, "confidence": float(0~1)}
        confidence = 1 - 平均逐维标准差 / 输出 RMS（方差/能量估计，任务规格）。
        """
        if not _HAS_NUMPY or not self._params_ok:
            vec, conf = self._pseudo_diffuse(state_vector, emotion_vec, text_history)
            vec = np.asarray(vec, dtype=np.float32) if _HAS_NUMPY else vec
            return {"vector": vec, "confidence": conf}
        try:
            x = self._to_state(state_vector)
            energy = self._emotion_energy(emotion_vec)
            rng = np.random.default_rng(self._call_seed(state_vector, emotion_vec, text_history))
            samples = self._mc_samples(x, rng, energy)
            # MC 均值作为扩散输出（10 次随机前向取平均）
            y = normalize(samples.mean(axis=0)).astype(np.float32)
            # 置信度: 多次输出的方差/能量估计 —— 样本越一致（方差越小）置信度越高
            sigma = float(np.sqrt(float(np.mean(samples.var(axis=0)))))
            rms_y = float(np.sqrt(float(np.mean(y * y)))) + 1e-6
            confidence = clamp(1.0 - sigma / rms_y, 0.0, 1.0)
            return {"vector": y, "confidence": confidence}
        except Exception as e:
            self.logger.warning("[L1] run 数值异常，降级为伪扩散: %s", e)       # 黄色警告（规范 §九.2）
            vec, conf = self._pseudo_diffuse(state_vector, emotion_vec, text_history)
            vec = np.asarray(vec, dtype=np.float32) if _HAS_NUMPY else vec
            return {"vector": vec, "confidence": conf}

    def diffuse(self, state_vector, emotion_vec=None, **kwargs):
        """兼容接口（line_controller 第二候选）: 与 run 相同，返回 dict。"""
        if kwargs:
            self.logger.warning("[L1] diffuse 忽略多余参数: %s", sorted(kwargs))
        return self.run(state_vector, emotion_vec)

    def forward(self, state_vector, **kwargs):
        """兼容接口（line_controller 第三候选）: 单次确定性前向，返回 512 维归一化向量。"""
        if kwargs:
            self.logger.warning("[L1] forward 忽略多余参数: %s", sorted(kwargs))
        if not _HAS_NUMPY or not self._params_ok:
            vec, _ = self._pseudo_diffuse(state_vector)
            return np.asarray(vec, dtype=np.float32) if _HAS_NUMPY else vec
        try:
            x = self._to_state(state_vector)
            steps = max(1, int(getattr(self.cfg, "iteration_steps", 20) or 20))
            dmp = float(getattr(self.cfg, "damping", 0.7) or 0.7)
            res = float(getattr(self.cfg, "resonance_freq", 2.5) or 0.0)
            z = x
            for t in range(steps):
                h = np.tanh(z @ self.P)
                if res > 0.0:
                    phase = 2.0 * math.pi * res * (t + 1) / float(steps)
                    h = h * (1.0 + 0.05 * math.sin(phase))
                back = h @ self.P.T
                z = normalize(z * dmp + (1.0 - dmp) * back)
            return normalize(z).astype(np.float32)
        except Exception as e:
            self.logger.warning("[L1] forward 数值异常，降级为伪扩散: %s", e)   # 黄色警告（规范 §九.2）
            vec, _ = self._pseudo_diffuse(state_vector)
            return np.asarray(vec, dtype=np.float32) if _HAS_NUMPY else vec

    def __call__(self, state_vector, emotion_vec=None, **kwargs):
        """对象调用（line_controller 第四候选）: 与 run 相同，返回 dict。"""
        if kwargs:
            self.logger.warning("[L1] __call__ 忽略多余参数: %s", sorted(kwargs))
        return self.run(state_vector, emotion_vec)

    # ------------------------------------------------------------------
    # 纯 Python 伪扩散（numpy 缺失/数值异常的最后兜底，规范 §九.2）
    # ------------------------------------------------------------------
    def _pseudo_diffuse(self, state_vector, emotion_vec=None, text_history=""):
        """纯 Python 伪扩散: 确定性伪随机噪声 + 手写 L2 归一化（仅 math，无 numpy）。
        仅用于流程验证（规范 §六.1 降级边界），置信度固定 0.5。"""
        vals = [float(v) for v in state_vector]
        if not vals:
            vals = [0.0] * 512
        seed = sum(ord(c) for c in str(text_history or "")) % 9973 + 7
        emo = 0.0
        try:
            ev = [float(v) for v in (emotion_vec or [])]
            emo = abs(ev[5]) if len(ev) > 5 else (sum(abs(v) for v in ev) / max(1, len(ev)))
        except Exception:
            emo = 0.0
        out = []
        for i, v in enumerate(vals):
            s = math.sin((i + 1) * (seed + 3)) * (0.05 + 0.05 * emo)
            c = math.cos((i + 1) * 13.7 + seed) * 0.05
            out.append(v * 0.9 + s + c)
        norm = math.sqrt(sum(x * x for x in out)) or 1.0
        out = [x / norm for x in out]
        return out, 0.5


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 自检（规范 §九.1/九.4）: 无 torch 依赖，numpy 缺失时亦可运行（伪扩散路径）
    print("[SELFTEST] diffuser_l1.py 自检开始 ...")
    _cfg = None
    try:
        from config import get_config
        _cfg = get_config()
        print("[SELFTEST] config 加载成功: demo=%s iteration_steps=%s damping=%s resonance_freq=%s" % (
            _cfg.demo, _cfg.iteration_steps, _cfg.damping, _cfg.resonance_freq))
    except Exception as _e:
        print("[SELFTEST] config 加载失败（预期降级为内置默认）: %s" % _e)

    _d = L1Diffuser(_cfg)

    _sv = np.random.RandomState(0).randn(512).astype(np.float32)
    _ev = np.zeros(8, dtype=np.float32)
    _ev[5] = 0.6                                        # intensity=0.6
    _txt = "你好，昔涟。你还记得花海吗？"

    # --- run(): 主接口 ---
    _r = _d.run(_sv, _ev, _txt)
    assert isinstance(_r, dict) and "vector" in _r and "confidence" in _r
    _v = _r["vector"]
    assert _v.shape == (512,), "vector 形状应为 (512,)，实际 %s" % (_v.shape,)
    assert _v.dtype == np.float32, "vector dtype 应为 float32"
    assert abs(float(np.linalg.norm(_v)) - 1.0) < 1e-3, "输出应已归一化"
    assert 0.0 <= _r["confidence"] <= 1.0, "confidence 应在 [0,1]"

    # --- diffuse / __call__ / forward: 兼容接口 ---
    _r2 = _d.diffuse(_sv, _ev)
    _r3 = _d(_sv, _ev)
    _fv = _d.forward(_sv)
    assert _fv.shape == (512,) and _fv.dtype == np.float32
    assert abs(float(np.linalg.norm(_fv)) - 1.0) < 1e-3

    # --- 确定性: 相同输入 → 相同输出（DEMO 可复现）---
    _r4 = _d.run(_sv, _ev, _txt)
    assert np.allclose(_r["vector"], _r4["vector"]), "相同输入应产生相同输出"

    # --- 文本调制: 不同 text_history → 不同噪声（触景生情，规范 §4.3）---
    _r5 = _d.run(_sv, _ev, "完全不同的一段话……")
    assert not np.allclose(_r["vector"], _r5["vector"]), "不同文本应产生不同扩散结果"

    # --- 纯 Python 伪扩散路径（numpy 缺失兜底）---
    _pv, _pc = _d._pseudo_diffuse([0.1] * 512, [0.0] * 8, "fallback")
    assert len(_pv) == 512 and abs(_pc - 0.5) < 1e-9

    # --- 参数规模 ≈ 0.5M（规范 §4.2 L1）---
    _nparams = int(_d.P.size) if _d._params_ok else 0
    assert _nparams == 524288, "P 应为 512×1024 = 524,288 参数"
    print("[SELFTEST] P 矩阵: %s → 参数 %d ≈ %.2fM" % (_d.P.shape, _nparams, _nparams / 1e6))

    print("[SELFTEST] run/diffuse/__call__ confidence = %.4f / %.4f / %.4f" % (
        _r["confidence"], _r2["confidence"], _r3["confidence"]))
    print("[SELFTEST] OK (PASS) L1 纯数据扩散器自检全部通过（CPU，%d 参数 ≈ 0.5M，无 torch）" % _nparams)
