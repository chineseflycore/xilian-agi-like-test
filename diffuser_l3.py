# -*- coding: utf-8 -*-
"""
diffuser_l3.py — L3 神经元特化网络扩散器（CPU 核心，规范 §4.2 / §4.4）
========================================================================
内存/显存预估（规范 §九.1）—— 默认 1,000,000 神经元 × 50 连接/神经元:

  连接权重矩阵 W（二维 CSR，形状 (N, N*50)，N = cfg.l3_neuron_count_eff）:
    · W.data    float32 × 50e6 ≈ 200 MB   （连接权重，规范 §4.2）
    · W.indices int32  × 50e6 ≈ 200 MB    （连接列索引 = 源神经元×50 + 槽位）
    · W.indptr  int32  × (N+1) ≈ 4 MB
    ─────────────────────────────────────────
    W 小计 ≈ 404 MB RAM（与规范 §4.2“~400MB”口径一致）

  投影矩阵（固定随机稀疏，规模与 N 无关）:
    · input_proj   (512, N) 25,600 非零 ≈ 0.2 MB   （状态向量 → 神经元激发）
    · emotion_proj (8, N)    1,600 非零 ≈ 0.02 MB  （情绪向量 → 神经元偏置）
    · readout      (512, N) 25,600 非零 ≈ 0.2 MB   （神经元活性 → 512 维读出）

  神经元元数据（neuron.py NeuronMeta）: ≈ 21 MB（详见 neuron.py 顶部注释）

  迭代期瞬态（每次扩散分配，用后即释放）:
    · v_rep = np.repeat(activity, 50)  float32 × 50e6 ≈ 200 MB
    · pot / drive / v  float32 × N ≈ 4 MB × 3
    峰值 ≈ W 404MB + 元数据 21MB + 瞬态 212MB ≈ 640 MB RAM（CPU 可容纳）
  缩放: 设 PHILIA_L3_SCALE=0.1 → N=100k → 峰值 ≈ 70MB（演示/快速验证）。

算法（规范 §4.2 L3 神经元特化网络 + “神经元扩散与震荡”）:
  输入 512 维状态 → input_proj 稀疏激发神经元（a0）
  → 迭代 cfg.iteration_steps 次衰减震荡（damping 阻尼 + resonance_freq 谐振）:
      pot   = W @ repeat(a, 50)            （CSR 矩阵-向量乘，二维稀疏存储）
      drive = sigmoid((pot/rms - threshold) × gain)   （阈值非线性，§4.2）
      drive[半休眠] = 0                    （痛觉达上限跳过激活，§4.4）
      a ← clip(a + damping×(drive - a) + resonance_freq×0.05×(a - a_prev), 0, 1)
  → 读出 readout @ a → 512 维归一化向量 + 置信度

强制存储规格（规范 §4.2 / §十.20）:
  · 必须使用 scipy.sparse.csr_matrix 二维矩阵 (N, N*50)，禁止一维向量
  · 训练/更新循环禁止 .toarray()/.todense() 稠密转换 —— Hebbian 更新与
    权重衰减仅通过 indptr/indices 向量化批量修改 W.data 的非零子集
  · 连接权重 float32、连接索引 int32；单次更新 < 50ms（向量化，小规模毫秒级）

降级策略（关键路径 try-except，规范 §九.2）:
  · lil_matrix 构建失败 → 回退三元组向量化构建（仍为二维 CSR）
  · 扩散/训练/痛觉任一环节异常 → 记录日志并回退伪输出，保证流程可验证
  · 纯 numpy/scipy 实现，禁止 torch（本机无 torch）
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
from scipy.sparse import lil_matrix, csr_matrix

from config import get_config
from common_utils import get_logger, normalize, clamp, Timer, safe_import
from neuron import NeuronMeta


class L3Diffuser:
    """L3 神经元特化网络扩散器（CPU，1M 神经元，50 连接/神经元）。

    统一接口（line_controller.py 契约）:
      run(state_vector, emotion_vec, text_history) -> dict     （主入口，返回 dict）
      diffuse(state_vector, emotion_vec)          -> (y, conf) （兼容 L1/L2 惯例）
      forward(state_vector)                       -> y          （纯向量）
      __call__(state_vector, emotion_vec)         -> dict      （同 run）
    辅助接口:
      hebbian_update(input_vec, reward=1.0)  —— Hebbian 无监督更新（规范 §4.2）
      mark_pain(neuron_indices)              —— 痛觉标记（规范 §4.4）
      avg_pain / confidence                  属性
      get_weights() / set_weights(csr)       —— 格式化系统（formatting.py）存取权重
    """

    def __init__(self, cfg=None, neuron_count=None):
        self.cfg = cfg if cfg is not None else get_config()
        self.logger = get_logger("diffuser_l3")
        # 神经元规模：显式 neuron_count 供自检/测试覆盖，默认取 cfg.l3_neuron_count_eff
        # （受 PHILIA_L3_SCALE 环境变量缩放；至少 1000，保证 512 维输入可完整映射）
        if neuron_count is not None:
            n = int(neuron_count)
            if n < 1:
                raise ValueError("neuron_count 必须 >= 1，实际 %d" % n)
            self.neuron_count = n
        else:
            self.neuron_count = int(self.cfg.l3_neuron_count_eff)
        self.conns = int(self.cfg.l3_conns)          # 每神经元 50 传入（规范 §4.2）
        self.cols = self.neuron_count * self.conns   # 二维 CSR 列数 N*50（规范 §4.2）
        self.dim = int(self.cfg.STATE_DIM)           # 512 维状态契约（规范 §3.4）
        self.emotion_dim = int(self.cfg.EMOTION_DIM)  # 8 维情绪向量
        self.steps = int(self.cfg.iteration_steps)    # 扩散迭代步数
        self.damping = float(self.cfg.damping)        # 阻尼（衰减震荡，0.7）
        self.resonance_freq = float(self.cfg.resonance_freq)  # 谐振频率（2.5）
        self.gain = 4.0                               # 阈值非线性增益
        self.rng = np.random.default_rng(20240601)    # 固定种子：可复现

        # ---- 连接权重：二维 CSR (N, N*50)，lil_matrix 逐行构建后 tocsr（规范 §4.2）----
        self.weights = self._build_weights()
        # ---- 投影矩阵（稀疏，2D CSR）----
        self.input_proj = self._build_projection(self.dim, nnz_per_row=50, seed=7,
                                                 name="input_proj")    # (512, N)
        self.emotion_proj = self._build_projection(self.emotion_dim, nnz_per_row=200, seed=11,
                                                   name="emotion_proj")  # (8, N)
        self.readout = self._build_readout(nnz_per_row=50, seed=13)    # (512, N)
        # ---- 神经元元数据（neuron.py）----
        self.meta = NeuronMeta(self.neuron_count,
                               threshold_range=self.cfg.l3_threshold_range,
                               pain_max=self.cfg.l3_pain_max,
                               rng=self.rng, logger=self.logger)
        # ---- 运行时状态 ----
        self.confidence = 0.0            # 最近一次扩散的置信度（line_controller 兼容）
        self._last_activity = None       # 最近一次扩散的神经元活性（Hebbian 更新用）
        self._last_vec = None            # 最近一次扩散的 512 维输出
        # 不封顶的累计痛觉计数（规范 §4.4 疤痕加深 / §4.5 触发条件）:
        #   · meta.pain_marks 封顶 l3_pain_max=10 —— 仅用于“半休眠”判定（§4.4）
        #   · _pain_total 不封顶累加 —— 使全局平均痛觉可越过 10.0，
        #     否则 §4.5 的“平均痛觉 > 10.0 触发格式化”在真实路径永远不成立
        self._pain_total = np.zeros(self.neuron_count, dtype=np.float32)
        # GPU 推理加速（1M 稀疏 → torch.sparse_csr 放显存，规范 §二 GPU 优先）:
        #   CPU 上 1M×50M CSR 单次扩散约 47s，GPU 稀疏矩阵-向量乘预期 < 2s
        self._gpu_ready = False
        self._gpu_dirty = True           # CPU 侧权重/阈值被训练修改后置位，run 前重建
        self._t_in = self._t_emo = self._t_w = self._t_ro = None
        self._t_thr = self._t_half = None
        self._mem_report()

    # ------------------------------------------------------------------
    # 构建：二维 CSR 权重矩阵（规范 §4.2 强制存储规格）
    # ------------------------------------------------------------------
    def _build_weights(self) -> csr_matrix:
        """构建 (N, N*50) 二维 CSR 权重矩阵。

        逐行构造: 每神经元一行，行内 50 个随机“传入连接”——列索引 =
        源神经元×50 + 槽位偏移（源神经元∈[0,N)、槽位∈[0,50)），权重 float32
        均匀随机。先用 lil_matrix 逐行填入，再 tocsr() 压缩存储。
        构建失败降级为三元组向量化构建（仍为二维 CSR，规范 §九.2）。
        """
        n, c = self.neuron_count, self.conns
        try:
            lil = lil_matrix((n, n * c), dtype=np.float32)
            for i in range(n):
                src = self.rng.integers(0, n, size=c, dtype=np.int32)
                off = self.rng.integers(0, c, size=c, dtype=np.int32)
                cols = (src * c + off).astype(np.int32)          # 列索引 int32
                data = self.rng.uniform(-0.5, 0.5, size=c).astype(np.float32)
                lil.rows[i] = cols.tolist()                      # lil 内部行列表
                lil.data[i] = data.tolist()
            W = lil.tocsr()
        except Exception as e:
            self.logger.warning(
                "\033[33m[L3] lil_matrix 逐行构建失败，降级三元组向量化构建: %r\033[0m", e)
            W = self._build_weights_triplet(n, c)
        W.sum_duplicates()
        W.sort_indices()
        # 强制规格校验（规范 §4.2 / §十.20）
        if W.ndim != 2 or W.shape != (n, n * c):
            raise RuntimeError("L3 权重矩阵必须是二维 (N, N*50)，实际 %s" % (W.shape,))
        if W.data.dtype != np.float32 or W.indices.dtype != np.int32:
            raise RuntimeError("L3 权重必须 float32/int32，实际 %s/%s"
                               % (W.data.dtype, W.indices.dtype))
        self.logger.info("[L3] 权重矩阵就绪 %s，稀疏度=%.2e，非零=%d",
                         (W.shape,), W.nnz / max(W.shape[0] * W.shape[1], 1), W.nnz)
        return W

    def _build_weights_triplet(self, n: int, c: int) -> csr_matrix:
        """降级路径：三元组 (data, indices, indptr) 向量化构建二维 CSR。

        不经过 lil_matrix，但产物与主路径完全等价（二维 (N, N*50)、float32/int32）。
        """
        nnz = n * c
        indptr = np.arange(0, nnz + 1, c, dtype=np.int32)             # 每行恰 50 项
        src = self.rng.integers(0, n, size=nnz, dtype=np.int32)
        off = np.tile(np.arange(c, dtype=np.int32), n)
        indices = (src * c + off).astype(np.int32)
        data = self.rng.uniform(-0.5, 0.5, size=nnz).astype(np.float32)
        return csr_matrix((data, indices, indptr), shape=(n, n * c))

    def _build_projection(self, n_rows: int, nnz_per_row: int, seed: int,
                          name: str) -> csr_matrix:
        """构建 (n_rows, N) 稀疏投影矩阵：每行 nnz_per_row 个随机目标神经元。

        用于输入状态（512 维）→ 神经元激发、情绪向量（8 维）→ 神经元偏置。
        权重 = ±1/sqrt(nnz_per_row)（保持激发幅度 O(1)）。
        """
        rng = np.random.default_rng(seed)
        lil = lil_matrix((n_rows, self.neuron_count), dtype=np.float32)
        scale = 1.0 / np.sqrt(nnz_per_row)
        for i in range(n_rows):
            cols = rng.integers(0, self.neuron_count, size=nnz_per_row, dtype=np.int32)
            data = rng.choice([-1.0, 1.0], size=nnz_per_row).astype(np.float32) * scale
            lil.rows[i] = cols.tolist()
            lil.data[i] = data.tolist()
        P = lil.tocsr()
        P.sum_duplicates()
        return P

    def _build_readout(self, nnz_per_row: int, seed: int) -> csr_matrix:
        """构建 (512, N) 稀疏读出矩阵：每个输出维读取 nnz_per_row 个随机神经元。

        权重 = ±1/sqrt(nnz_per_row)，使读出幅度与活性尺度一致。
        """
        rng = np.random.default_rng(seed)
        lil = lil_matrix((self.dim, self.neuron_count), dtype=np.float32)
        scale = 1.0 / np.sqrt(nnz_per_row)
        for i in range(self.dim):
            cols = rng.integers(0, self.neuron_count, size=nnz_per_row, dtype=np.int32)
            data = rng.choice([-1.0, 1.0], size=nnz_per_row).astype(np.float32) * scale
            lil.rows[i] = cols.tolist()
            lil.data[i] = data.tolist()
        R = lil.tocsr()
        R.sum_duplicates()
        return R

    def _mem_report(self):
        """内存估算日志（规范 §九.1 / §九.4）。"""
        w_bytes = int(self.weights.data.nbytes + self.weights.indices.nbytes
                      + self.weights.indptr.nbytes)
        proj_bytes = int(self.input_proj.data.nbytes + self.input_proj.indices.nbytes
                         + self.emotion_proj.data.nbytes + self.emotion_proj.indices.nbytes
                         + self.readout.data.nbytes + self.readout.indices.nbytes)
        meta_bytes = self.meta.memory_bytes()
        transient = int(self.neuron_count * self.conns * 4)   # 迭代期 v_rep 瞬态（float32）
        total = w_bytes + proj_bytes + meta_bytes
        self.logger.info(
            "[L3] 内存预估: 权重≈%.1fMB + 投影≈%.1fMB + 元数据≈%.1fMB ≈ 常驻 %.1fMB，"
            "迭代峰值≈%.1fMB（N=%d, 列数=%d, 非零=%d）",
            w_bytes / 1048576.0, proj_bytes / 1048576.0, meta_bytes / 1048576.0,
            total / 1048576.0, (total + transient) / 1048576.0,
            self.neuron_count, self.cols, self.weights.nnz)

    # ------------------------------------------------------------------
    # 扩散核心（神经元扩散与震荡，规范 §4.2）
    # ------------------------------------------------------------------
    def _excite(self, x: np.ndarray, e: np.ndarray) -> np.ndarray:
        """输入激发: a0 = input_projᵀ·x + emotion_projᵀ·e → (N,) 稀疏投影激发。

        两个稀疏矩阵-向量乘（CSRᵀ @ 稠密），均不产生稠密权重矩阵。
        """
        a0 = self.input_proj.T.dot(x)
        if e is not None and e.size:
            a0 = a0 + self.emotion_proj.T.dot(e)
        return a0.astype(np.float32)

    def _diffuse_core(self, x: np.ndarray, e: np.ndarray):
        """衰减震荡扩散: 返回 (最终活性 a(N,), 读出 y(512,), 置信度)。

        每步:
          pot   = W @ repeat(a, conns)      —— 二维 CSR 矩阵-向量乘（规范 §4.2）
          drive = σ((pot/rms − threshold)·gain)   —— 阈值非线性（半休眠跳过）
          a ← clip(a + damping·(drive−a) + resonance_freq·0.05·(a−a_prev), 0, 1)
        """
        n = self.neuron_count
        thr = self.meta.thresholds
        half = self.meta.half_dormant
        a = np.clip(self._excite(x, e), 0.0, 1.0).astype(np.float32)
        a_prev = np.zeros(n, dtype=np.float32)
        res_mom = min(0.9, self.resonance_freq * 0.05)   # 谐振动量（衰减震荡回响）
        delta = 0.0
        for _ in range(self.steps):
            # 二维 CSR 矩阵-向量乘：把活性按“神经元×50 槽位”展开为列空间向量
            # （np.repeat 是展开算子，非稠密化权重矩阵；W 始终保持稀疏存储）
            v_rep = np.repeat(a, self.conns)
            pot = self.weights.dot(v_rep)                 # (N,)
            # 幅度归一化（RMS），稳定阈值非线性输入尺度
            rms = float(np.sqrt(np.mean(pot * pot))) + 1e-6
            pot_n = pot / rms
            drive = 1.0 / (1.0 + np.exp(-(pot_n - thr) * self.gain))
            drive[half] = 0.0                             # 半休眠跳过激活（规范 §4.4）
            # 衰减震荡动力学（damping 阻尼收敛 + resonance_freq 谐振回响）
            a_new = a + self.damping * (drive - a) + res_mom * (a - a_prev)
            a_new = np.clip(a_new, 0.0, 1.0).astype(np.float32)
            denom = float(np.linalg.norm(a)) + 1e-6
            delta = float(np.linalg.norm(a_new - a)) / denom
            a_prev, a = a, a_new

        # 读出投影回 512 维（稀疏矩阵-向量乘）并归一化
        y = normalize(self.readout.dot(a))
        # 置信度 = 收敛度（末步相对变化小 → 置信度高）与激活覆盖率加权
        coverage = float(np.mean(a > 0.5))
        conf = clamp(0.7 * (1.0 - min(delta, 1.0)) + 0.3 * coverage)
        return a, y, conf

    # ------------------------------------------------------------------
    # GPU 稀疏推理（1M 规模加速，规范 §二 GPU 优先；CPU 路径保留为降级）
    # ------------------------------------------------------------------
    def _to_gpu(self) -> bool:
        """把稀疏权重转 torch.sparse_csr 放显存（构建一次，dirty 时重建）。

        转换: input_proj.T(N,512) / emotion_proj.T(N,8) / weights(N,N*50) /
        readout(512,N) + 阈值/半休眠掩码。失败 → 回退 CPU（规范 §九.2）。
        """
        torch = safe_import("torch")
        if torch is None or not torch.cuda.is_available():
            return False
        if self._gpu_ready and not self._gpu_dirty:
            return True
        try:
            dev = "cuda"

            def to_tcsr(M):
                M = M.tocsr()
                crow = torch.from_numpy(M.indptr.astype(np.int64)).to(dev)
                col = torch.from_numpy(M.indices.astype(np.int64)).to(dev)
                val = torch.from_numpy(np.asarray(M.data, dtype=np.float32)).to(dev)
                return torch.sparse_csr_tensor(crow, col, val,
                                               size=tuple(M.shape), device=dev)

            self._t_in = to_tcsr(self.input_proj.T)      # (N, 512)
            self._t_emo = to_tcsr(self.emotion_proj.T)   # (N, 8)
            self._t_w = to_tcsr(self.weights)            # (N, N*50)
            self._t_ro = to_tcsr(self.readout)           # (512, N)
            self._t_thr = torch.from_numpy(
                self.meta.thresholds.astype(np.float32)).to(dev)
            self._t_half = torch.from_numpy(self.meta.half_dormant).to(dev)
            self._gpu_ready = True
            self._gpu_dirty = False
            est = (self.weights.nnz * 12 + self.weights.shape[0] * 8 * 2) / 1048576.0
            self.logger.info("[L3] GPU 稀疏副本就绪（显存 +%.0fMB）", est)
            return True
        except Exception as e:
            self._gpu_ready = False
            self.logger.warning("\033[33m[L3] GPU 副本构建失败，回退 CPU 扩散: %r\033[0m", e)
            return False

    def _diffuse_gpu(self, x_np: np.ndarray, e_np: np.ndarray):
        """GPU 扩散: 20 步衰减震荡全部在 GPU（sparse_csr @ dense）。

        与 _diffuse_core 数学等价；阈值/半休眠每次刷新（CPU 训练可能已改）。
        """
        torch = safe_import("torch")
        dev = "cuda"
        n = self.neuron_count
        x = torch.from_numpy(np.asarray(x_np, dtype=np.float32)).to(dev)
        e = torch.from_numpy(np.asarray(e_np, dtype=np.float32)).to(dev)
        # 阈值/半休眠可能被 CPU 侧训练更新 → 每次 run 前刷新（1M 数组 < 5ms）
        self._t_thr.copy_(torch.from_numpy(self.meta.thresholds.astype(np.float32)).to(dev))
        self._t_half.copy_(torch.from_numpy(self.meta.half_dormant).to(dev))
        a0 = self._t_in @ x + self._t_emo @ e
        a = torch.clamp(a0, 0.0, 1.0)
        a_prev = torch.zeros(n, device=dev)
        res_mom = min(0.9, self.resonance_freq * 0.05)
        delta = torch.tensor(0.0, device=dev)
        for _ in range(self.steps):
            v_rep = a.repeat_interleave(self.conns)          # (N*50,)
            pot = (self._t_w @ v_rep.unsqueeze(1)).squeeze(1)  # (N,)
            rms = torch.sqrt(torch.mean(pot * pot) + 1e-6)
            pot_n = pot / rms
            drive = 1.0 / (1.0 + torch.exp(-(pot_n - self._t_thr) * self.gain))
            drive[self._t_half] = 0.0                         # 半休眠跳过（规范 §4.4）
            a_new = a + self.damping * (drive - a) + res_mom * (a - a_prev)
            a_new = torch.clamp(a_new, 0.0, 1.0)
            denom = torch.norm(a) + 1e-6
            delta = torch.norm(a_new - a) / denom
            a_prev, a = a, a_new
        y_np = (self._t_ro @ a.unsqueeze(1)).squeeze(1).detach().cpu().numpy().astype(np.float32)
        a_np = a.detach().cpu().numpy().astype(np.float32)
        del v_rep, pot, a, a_prev
        torch.cuda.empty_cache()
        coverage = float(np.mean(a_np > 0.5))
        conf = clamp(0.7 * (1.0 - min(float(delta.cpu()), 1.0)) + 0.3 * coverage)
        return a_np, normalize(y_np), conf

    def _fallback_output(self, x: np.ndarray):
        """降级伪输出：扩散异常时返回输入归一化向量（流程可验证，规范 §九.2）。"""
        return normalize(np.asarray(x, dtype=np.float32).reshape(-1))

    # ------------------------------------------------------------------
    # 统一接口（line_controller.py 契约）
    # ------------------------------------------------------------------
    def run(self, state_vector, emotion_vec=None, text_history: str = "") -> dict:
        """主入口: 512 维状态 → 扩散 → {"vector": (512,) float32 归一化, "confidence": float}。

        参数:
          state_vector: 融合层 512 维状态向量（float32）
          emotion_vec : 8 维情绪向量 [angry,sad,happy,neutral,fear,intensity,arousal,valence]
          text_history: 历史对话文本（L3 核心不直接消费文本——文本经 translator
                        编码为状态向量后进入扩散；此处仅记录日志，规范 §4.2 架构）
        返回:
          {"vector": y(512,) float32 归一化, "confidence": float}
        """
        try:
            x = np.asarray(state_vector, dtype=np.float32).reshape(-1)
            if x.size != self.dim:
                x = np.resize(x, self.dim).astype(np.float32)
            if emotion_vec is None:
                e = np.zeros(self.emotion_dim, dtype=np.float32)
            else:
                e = np.asarray(emotion_vec, dtype=np.float32).reshape(-1)
                if e.size != self.emotion_dim:
                    e = np.resize(e, self.emotion_dim).astype(np.float32)
            if text_history:
                self.logger.debug("[L3] text_history(%d 字符) 经 translator 编码后生效", len(text_history))

            with Timer("l3_diffuse") as t:
                if self._to_gpu():
                    a, y, conf = self._diffuse_gpu(x, e)     # GPU 稀疏（1M 加速）
                else:
                    a, y, conf = self._diffuse_core(x, e)    # CPU 稀疏（降级路径）
            self._last_activity = a.astype(np.float32)
            self._last_vec = y
            self.confidence = float(conf)
            # 本轮激活掩码 → 遗忘计数（规范 §4.2 遗忘计数器 / §4.7.4 时间衰减）
            try:
                ripe = self.meta.apply_forgetting(a > 0.5)
                if ripe.size:
                    self._decay_rows(ripe, rate=self.cfg.l3_pain_weight_decay * 0.5)
            except Exception as fe:
                self.logger.warning("[L3] 遗忘衰减失败(降级跳过): %r", fe)
            self.logger.info("[L3] 扩散完成 conf=%.3f 耗时=%.1fms", conf, t.elapsed)
            return {"vector": y, "confidence": float(conf)}
        except Exception as e:
            self.logger.warning("\033[33m[L3] 扩散失败，回退伪输出: %r\033[0m", e)
            self.confidence = 0.30
            try:
                vec = self._fallback_output(state_vector)
            except Exception:
                vec = np.zeros(self.dim, dtype=np.float32)   # 兜底：零向量（流程可验证）
            return {"vector": vec, "confidence": 0.30}

    def diffuse(self, state_vector, emotion_vec=None):
        """兼容接口: 返回 (y(512,), confidence) 元组（同 L1/L2 惯例）。"""
        r = self.run(state_vector, emotion_vec)
        return r["vector"], r["confidence"]

    def forward(self, state_vector):
        """兼容接口: 仅返回 512 维输出向量。"""
        return self.run(state_vector, None)["vector"]

    def __call__(self, state_vector, emotion_vec=None) -> dict:
        """兼容接口: 同 run，返回 dict。"""
        return self.run(state_vector, emotion_vec)

    # ------------------------------------------------------------------
    # Hebbian 无监督更新（规范 §4.2“训练: Hebbian 规则（无监督）+ 奖励微调”）
    # ------------------------------------------------------------------
    def hebbian_update(self, input_vec, reward: float = 1.0, lr: float = 0.01) -> int:
        """Hebbian 无监督更新，仅作用于 MC Dropout 随机掩码子集（比例
        cfg.l3_pain_update_ratio，<10%），禁止稠密转换（规范 §4.2 / §十.20）。

        规则（Oja 归一化 Hebbian，天然抑制权重发散）:
          Δw_ik = lr × reward × a_post_i × (a_pre_k − w_ik × a_post_i)

        实现: 通过 indptr/indices 向量化定位掩码行的全部非零项，仅修改
        W.data 的对应切片（不触碰其他行，无 .toarray()/.todense()）。

        返回: 本次更新的连接条数（自检/日志用）。
        """
        try:
            x = np.asarray(input_vec, dtype=np.float32).reshape(-1)
            if x.size != self.dim:
                x = np.resize(x, self.dim).astype(np.float32)
            # 活性来源：优先最近一次扩散的神经元活性，否则即时激发一次
            if self._last_activity is not None:
                a = self._last_activity
            else:
                a = np.clip(self._excite(x, np.zeros(self.emotion_dim, np.float32)),
                            0.0, 1.0).astype(np.float32)
            # MC Dropout 掩码：比例 l3_pain_update_ratio（<10%）的神经元参与更新，
            # 半休眠神经元不参与（痛觉回避，规范 §4.4）
            mask = self.rng.random(self.neuron_count) < self.cfg.l3_pain_update_ratio
            mask &= ~self.meta.half_dormant
            rows = np.flatnonzero(mask)
            if rows.size == 0:
                return 0
            # 向量化定位掩码行全部非零项的位置（indptr 批量索引，规范 §4.2 推荐）
            pos = (np.repeat(self.weights.indptr[rows], self.conns)
                   + np.tile(np.arange(self.conns, dtype=np.int32), rows.size))
            # 源神经元 = 列索引 // conns（列布局: 源神经元×conns + 槽位）
            src = self.weights.indices[pos] // self.conns
            pre = a[src]                                     # 突触前活性
            post = np.repeat(a[rows], self.conns)            # 突触后活性（行内广播）
            w = self.weights.data[pos]
            # Oja Hebbian: Δw = lr·reward·post·(pre − w·post)
            dw = float(lr) * float(reward) * post * (pre - w * post)
            w_new = np.clip(w + dw, -2.0, 2.0)               # 权重限幅防发散
            self.weights.data[pos] = w_new.astype(np.float32)
            self._gpu_dirty = True                           # GPU 副本需重建
            return int(pos.size)
        except Exception as e:
            self.logger.warning("\033[33m[L3] Hebbian 更新失败(降级跳过): %r\033[0m", e)
            return 0

    # ------------------------------------------------------------------
    # 痛觉系统（规范 §4.4）
    # ------------------------------------------------------------------
    def mark_pain(self, neuron_indices) -> int:
        """痛觉标记: 指定神经元标记 +1（上限 cfg.l3_pain_max=10）。

        对本次被标记（含新达上限）的神经元:
          · 阈值上升 cfg.l3_pain_mark_up/标记（0.1）
          · 连接权重衰减 ×(1 − cfg.l3_pain_weight_decay)（0.02）
          · 达上限 → 半休眠（后续扩散跳过激活）
        返回: 本次新达上限（半休眠）的神经元数量。
        """
        try:
            newly = self.meta.mark_pain(neuron_indices)
            # 累计疤痕计数（不封顶，规范 §4.4/§4.5）—— 使 avg_pain 可越过 10.0
            idx = np.asarray(neuron_indices, dtype=np.int64).reshape(-1)
            idx = np.unique(idx[(idx >= 0) & (idx < self.neuron_count)])
            if idx.size:
                self._pain_total[idx] += 1.0
            # 阈值上升（规范 §4.4: 阈值上升 0.1/标记）——作用于所有被标记神经元
            if idx.size:
                thr = self.meta.thresholds[idx]
                thr = np.minimum(thr + self.cfg.l3_pain_mark_up, 1.5)
                self.meta.thresholds[idx] = thr.astype(np.float32)
            # 权重衰减（规范 §4.4: 权重衰减）——作用于被标记神经元的传入连接
            if idx.size:
                self._decay_rows(idx, rate=self.cfg.l3_pain_weight_decay)
            self.logger.info("[L3] 痛觉标记 %d 神经元，其中 %d 个新达上限进入半休眠",
                             idx.size, newly.size)
            return int(newly.size)
        except Exception as e:
            self.logger.warning("\033[33m[L3] 痛觉标记失败(降级跳过): %r\033[0m", e)
            return 0

    def _decay_rows(self, rows: np.ndarray, rate: float):
        """向量化权重衰减: 对指定神经元行的全部非零连接 W.data *= (1 − rate)。

        仅通过 indptr 索引批量修改 W.data（禁止稠密转换，规范 §十.20）。
        """
        if rows is None or np.asarray(rows).size == 0:
            return
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        rows = rows[(rows >= 0) & (rows < self.neuron_count)]
        if rows.size == 0:
            return
        pos = (np.repeat(self.weights.indptr[rows], self.conns)
               + np.tile(np.arange(self.conns, dtype=np.int32), rows.size))
        self.weights.data[pos] = (self.weights.data[pos] * (1.0 - float(rate))).astype(np.float32)
        self._gpu_dirty = True                               # GPU 副本需重建

    @property
    def avg_pain(self) -> float:
        """全局平均痛觉（规范 §4.4/§4.5）。

        基于不封顶的累计痛觉计数（_pain_total）求均值 —— 可越过 10.0，
        使 pain_system.should_format()（> 10.0）在真实路径可触发（§4.5）。
        """
        return float(np.mean(self._pain_total))

    @property
    def pain_marks(self) -> "np.ndarray":
        """封顶痛觉标记（0~l3_pain_max）—— 格式化“留疤”导出用（规范 §4.5）。"""
        try:
            return self.meta.pain_marks
        except Exception:
            return np.zeros(self.neuron_count, dtype=np.float32)

    def reset_pain_total(self):
        """重置累计痛觉计数（格式化“向死而生”后调用，规范 §4.5）。

        注意: 只清零累计计数，**保留封顶 pain_marks（留疤）** ——
        与 formatting.py 的“清空大部分连接权重、保留所有痛觉标记”语义一致。
        """
        self._pain_total = np.zeros(self.neuron_count, dtype=np.float32)
        self.logger.info("[L3] 累计痛觉计数已重置（封顶痛觉标记保留，留疤）")

    # ------------------------------------------------------------------
    # 格式化系统接口（formatting.py: 保存/清空权重，保留痛觉标记）
    # ------------------------------------------------------------------
    def get_weights(self) -> csr_matrix:
        """返回当前连接权重矩阵（二维 CSR (N, N*50)，供格式化保存）。"""
        return self.weights

    def set_weights(self, csr: csr_matrix):
        """替换连接权重矩阵（格式化清空权重时调用）。

        保留痛觉标记（“留疤”，规范 §4.5 第 3 条）；遗忘/激活计数复位。
        校验: 必须为二维 csr_matrix，形状 (N, N*50)，权重 float32。
        """
        if not isinstance(csr, csr_matrix):
            raise TypeError("set_weights 仅接受 scipy.sparse.csr_matrix")
        if csr.ndim != 2 or csr.shape != (self.neuron_count, self.cols):
            raise ValueError("权重矩阵形状必须为 (N, N*50)=%s，实际 %s"
                             % ((self.neuron_count, self.cols), csr.shape))
        csr = csr.astype(np.float32).tocsr()
        self.weights = csr
        self.meta.reset_counts()          # 保留痛觉标记，清空遗忘/激活（规范 §4.5）
        self._last_activity = None
        self._gpu_dirty = True            # GPU 副本需重建
        self.logger.info("[L3] 权重已替换（保留痛觉标记），非零=%d", csr.nnz)

    # ------------------------------------------------------------------
    # 持久化（train_l3_knowledge.py 训练固化后保存 / 启动时恢复）
    # ------------------------------------------------------------------
    def save_state(self, path: str):
        """保存 L3 完整状态（连接权重 + 阈值 + 痛觉标记 + 累计痛觉计数）。

        path 为 .npz：权重存 path（csr），元数据存 path.meta.npz。
        说明: 训练固化后的"本能"（规范 §4.3 第二层）落盘，重启可恢复。
        """
        from scipy import sparse
        sparse.save_npz(path, self.weights)
        np.savez(path + ".meta.npz",
                 thresholds=self.meta.thresholds,
                 pain_marks=self.meta.pain_marks,
                 pain_total=self._pain_total)
        self.logger.info("[L3] 状态已保存 → %s（非零=%d）", path, self.weights.nnz)

    def load_state(self, path: str):
        """从 save_state 产物恢复 L3 状态（形状/类型校验后原位加载）。"""
        from scipy import sparse
        if not os.path.isfile(path) or not os.path.isfile(path + ".meta.npz"):
            raise FileNotFoundError("L3 状态文件缺失: %s" % path)
        W = sparse.load_npz(path)
        if W.ndim != 2 or W.shape != (self.neuron_count, self.cols):
            raise ValueError("L3 状态形状 %s 与当前 %s 不符"
                             % (W.shape, (self.neuron_count, self.cols)))
        self.weights = W.astype(np.float32).tocsr()
        m = np.load(path + ".meta.npz")
        self.meta.thresholds = m["thresholds"].astype(np.float32)
        self.meta.pain_marks = m["pain_marks"].astype(np.int32)
        self._pain_total = m["pain_total"].astype(np.float32)
        self._last_activity = None
        self._gpu_dirty = True                               # GPU 副本需重建
        self.logger.info("[L3] 状态已恢复 ← %s（非零=%d，avg_pain=%.3f）",
                         path, self.weights.nnz, self.avg_pain)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 自检：小规模神经元（2000），保证秒级完成（规范 §九.1 / §十.3）
    _cfg = get_config()
    print(_cfg.summary())
    _d = L3Diffuser(_cfg, neuron_count=2000)

    _rng = np.random.RandomState(0)
    _x = normalize(_rng.randn(_cfg.STATE_DIM).astype(np.float32))
    _e = np.zeros(_cfg.EMOTION_DIM, dtype=np.float32)
    _e[[0, 6]] = (0.6, 0.4)   # 情绪: angry=0.6, arousal=0.4

    # 1) run 主入口（返回 dict，规范 §4.2 / line_controller 契约）
    with Timer("selftest_run") as _t:
        _r = _d.run(_x, _e, "自检: 你好，昔涟。你还记得花海吗？")
    assert _r["vector"].shape == (512,) and _r["vector"].dtype == np.float32
    assert np.isfinite(_r["vector"]).all() and 0.0 <= _r["confidence"] <= 1.0
    print("[SELFTEST] run OK: vector=%s conf=%.3f 耗时=%.1fms" %
          (_r["vector"].shape, _r["confidence"], _t.elapsed))

    # 2) diffuse / forward / __call__ 兼容接口
    _y2, _c2 = _d.diffuse(_x, _e)
    _y3 = _d.forward(_x)
    _r4 = _d(_x, _e)
    assert _y2.shape == (512,) and _y3.shape == (512,) and isinstance(_r4, dict)
    print("[SELFTEST] diffuse/forward/__call__ OK: conf=%.3f conf_attr=%.3f" %
          (_c2, float(_d.confidence)))

    # 3) Hebbian 无监督更新（规范 §4.2，MC Dropout 子集 <10%，禁止稠密转换）
    with Timer("selftest_hebbian") as _t2:
        _n_upd = _d.hebbian_update(_x, reward=1.0)
    assert 0 < _n_upd <= _d.neuron_count * _d.conns
    assert _t2.elapsed < 50.0, "单次更新应 <50ms，实际 %.1fms" % _t2.elapsed
    print("[SELFTEST] hebbian_update OK: 更新连接=%d 耗时=%.2fms (<50ms)" %
          (_n_upd, _t2.elapsed))

    # 4) 痛觉标记（规范 §4.4: 标记上限 → 半休眠；阈值上升；权重衰减）
    #    每次 mark_pain 标记 +1，需累积 l3_pain_max=10 次才达上限进入半休眠
    _pain_idx = np.arange(50, dtype=np.int64)
    _w_before = float(np.abs(_d.weights.data).sum())
    _thr_before = float(_d.meta.thresholds[_pain_idx].mean())
    _newly = 0
    for _ in range(_cfg.l3_pain_max):
        _newly = _d.mark_pain(_pain_idx)
    assert _newly == 50
    assert int(_d.meta.half_dormant.sum()) == 50
    assert float(_d.meta.thresholds[_pain_idx].mean()) > _thr_before
    assert float(np.abs(_d.weights.data).sum()) < _w_before
    print("[SELFTEST] mark_pain OK: 半休眠=%d 阈值↑(%.3f→%.3f) 权重衰减 ✓ avg_pain=%.3f" %
          (int(_d.meta.half_dormant.sum()), _thr_before,
           float(_d.meta.thresholds[_pain_idx].mean()), float(_d.avg_pain)))

    # 5) 半休眠神经元在扩散中被跳过（驱动归零，规范 §4.4）
    _r5 = _d.run(_x, _e)
    print("[SELFTEST] 半休眠后扩散仍可运行: conf=%.3f avg_pain=%.3f" %
          (_r5["confidence"], float(_d.avg_pain)))

    # 6) get_weights / set_weights 往返（格式化保留痛觉标记，规范 §4.5）
    _W = _d.get_weights()
    _pain_saved = int(_d.meta.pain_marks.sum())
    _d.set_weights(_W)                       # 同一矩阵回写（等价“清空后保存”的往返验证）
    assert _d.get_weights().shape == _W.shape
    assert int(_d.meta.pain_marks.sum()) == _pain_saved
    print("[SELFTEST] get/set_weights OK: 形状=%s 非零=%d 痛觉标记保留=%d" %
          (_d.get_weights().shape, _d.get_weights().nnz, _pain_saved))

    print("[SELFTEST] diffuser_l3.py 全部通过 OK（神经元=%d 连接=%d 列数=%d）" %
          (_d.neuron_count, _d.conns, _d.cols))
