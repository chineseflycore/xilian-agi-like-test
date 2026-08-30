# -*- coding: utf-8 -*-
"""
diffuser_l2.py — L2 Transformer 扩散器（GPU，~1M 参数，双缓冲训练，规范 §4.2 L2）
==================================================================================
内存/显存预估（规范 §九.1）:
  REAL（torch 可用）:
    · 主模型 self.model: 2 层 Transformer（隐藏 cfg.l2_hidden=256、4 头、输入 512 折叠为
      16×32 序列），参数 ≈ 0.8~1.0M，FP32 ≈ 3.2~4MB/份 —— 与规范口径“1M（~4MB）”一致。
    · 双缓冲（规范 §十.15）: GPU 主模型（显存 ~4MB）+ CPU 训练副本 self.model_cpu
      （内存 ~4MB）+ SGD 优化器状态（~8MB）→ 本模块合计 ~16MB；显存峰值增量 < 5MB，
      不挤占 Qwen 路由层（3.2GB）与情感模型（160MB）的预算（规范 §二 峰值 <4.5GB）。
  DEMO（无 torch / PHILIA_DEMO=1）:
    · 随机正交投影 P(512×512) ≈ 1MB + 门控向量/情绪投影 ≈ 64KB → 合计 ~1.1MB RAM，
      无训练参数、不占显存。
  torch 通过 common_utils.safe_import 惰性获取：本机无 torch 时模块仍可 import，
  自动降级 numpy 伪扩散器（黄色警告日志 + [DEMO] 标记）。

双缓冲训练（规范 §4.2 L2 / §十.15）:
  · 推理用 GPU 主模型 self.model（常驻）；训练用 CPU 副本 self.model_cpu。
  · train_step() 只更新 CPU 副本（禁止触碰 GPU 主模型），记录改进指标；
    swap() 依据 cfg.l2_swap_threshold（0.02）决定是否“原子替换”——
    以 Python 对象引用切换（self.model = 新对象）完成，无中间态、无并发撕裂。
  · DEMO（numpy 伪扩散器）不可训练: train_step 为 no-op（日志 + 返回 0.0），
    符合规范 §六.1 降级边界（降级仅用于流程验证）。

TODO(V3.4): ① 真 Transformer 扩散器结构对齐 Qwen 隐藏层；② run() 中 text_history
  的语义调制（当前仅 DEMO 路径以伪嵌入做弱调制）；③ swap 改进指标改为验证集评估。
"""

import copy

import os
import sys

# 本机 Python 以 safe_path 模式运行（脚本目录不会自动进入 sys.path）:
# 显式把本文件所在目录加入 sys.path，保证 `python diffuser_l2.py` 自检
# 与 `from diffuser_l2 import L2Diffuser` 两种用法都能导入同项目模块（规范 §九.12 路径定位）
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# 注意导入顺序: common_utils 会先把 vendor/（本地 numpy/scipy）加入 sys.path，
# 故必须先于 numpy 导入（本机无全局 numpy，仅 vendor/ 可用，规范 §二 环境约束）
from config import get_config
from common_utils import (get_logger, safe_import, clamp, normalize,
                          gpu_mem_mb, pseudo_embedding)

import numpy as np

__all__ = ["L2Diffuser"]


class L2Diffuser:
    """L2 Transformer 扩散器（双缓冲训练，规范 §4.2 L2）。

    统一接口（line_controller.py / trainer.py 调用契约）:
      run(state_vector, emotion_vec, text_history="") -> {"vector", "confidence"}
      diffuse(state_vector, emotion_vec)                  -> 同上（兼容）
      forward(state_vector)                               -> 512 维向量（兼容）
      __call__(state_vector, emotion_vec)                 -> 同 run
      train_step(inputs, targets)                         -> loss float
      swap()                                              -> {"swapped", "improvement"}
    """

    def __init__(self, cfg=None):
        # 配置缺省用模块级单例（config.get_config，规范 §八.1 集中配置）
        self.cfg = cfg or get_config()
        self.dim = int(getattr(self.cfg, "STATE_DIM", 512))   # 512 维状态向量契约（规范 §3.4）
        self.logger = get_logger("diffuser_l2")
        self.confidence = 0.55          # 兼容 line_controller._coerce_output 的兜底置信度
        self.last_loss = 0.0            # 最近一次 train_step 的 loss（调试用）
        self._pending_improvement = 0.0  # train_step 累积的改进指标，供 swap() 原子替换判断

        # torch 惰性获取（规范 §九.2 关键路径降级）：本机无 torch → None → 走 numpy DEMO
        torch = safe_import("torch")
        self.use_torch = bool(torch is not None and not self.cfg.demo)
        self.model = None       # GPU/CPU 主模型（推理用，双缓冲“主”）
        self.model_cpu = None   # CPU 训练副本（双缓冲“副”，train_step 更新它）
        if self.use_torch:
            self._init_torch(torch)
        else:
            self._init_numpy()
        self.logger.info("[L2] VRAM=%.0fMB", gpu_mem_mb())

    # ------------------------------------------------------------------
    # torch 路径（2 层 Transformer，隐藏 256，4 头，~1M 参数 / ~4MB）
    # ------------------------------------------------------------------
    def _init_torch(self, torch):
        """构建 torch 主模型 + CPU 训练副本 + 优化器；任一环节失败降级 numpy（§九.2）。"""
        try:
            import torch.nn as nn
            # 设备选择: 有 CUDA 用 GPU（GTX 1060 6GB），否则 CPU；FP32 强制（规范 §十.1）
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self._build_torch_model(nn).to(self.device).eval()
            # 双缓冲: 深拷贝 CPU 副本用于训练（规范 §十.15），主模型不参与训练
            self.model_cpu = copy.deepcopy(self.model).to("cpu").eval()
            self.lr = 0.01
            self.optimizer = torch.optim.SGD(self.model_cpu.parameters(), lr=self.lr)
            n_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(
                "[L2] torch Transformer 扩散器就绪 device=%s 参数=%s (~%.2fMB FP32)",
                self.device, format(n_params, ","), n_params * 4 / 1024 / 1024)
        except Exception as e:
            self.logger.warning(
                "\033[33m[L2] torch 初始化失败，降级 numpy 伪扩散器: %r\033[0m", e)
            self.use_torch = False
            self._init_numpy()

    def _build_torch_model(self, nn):
        """手写 2 层小型 Transformer（隐藏 256，4 头；输入 512 折叠为 16×32 序列）。

        结构: Linear(32→256) + 2×[LayerNorm + MultiheadAttention(256,4头) + FFN(GELU)] + Linear(256→32)
        参数估算（FFN 宽度=隐藏 256）:
          投影 2×32×256 ≈ 16K
          + 2 层 × (注意力 4×256×256 + LayerNorm×2 + FFN 2×256×256) ≈ 2×395K
          ≈ 0.81M 参数 ≈ 3.2MB FP32（规范 §4.2 口径 “1M / ~4MB”，含优化器状态后 ~16MB 总量）。
        手写而非 nn.TransformerEncoderLayer: 显式控制维度并规避 torch 版本 API 漂移。
        """
        hidden = int(self.cfg.l2_hidden)     # 256
        heads = int(self.cfg.l2_heads)       # 4
        layers = int(self.cfg.l2_layers)     # 2
        seq_len = 16
        emb_in = self.dim // seq_len         # 32（512 = 16 × 32）

        class _Block(nn.Module):
            """单层: 预归一化注意力 + 残差 + FFN + 残差（类 Pre-LN Transformer）。"""

            def __init__(self, h, hd):
                super().__init__()
                self.ln1 = nn.LayerNorm(h)
                self.attn = nn.MultiheadAttention(h, hd, batch_first=True)
                self.ln2 = nn.LayerNorm(h)
                self.ffn = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, h))

            def forward(self, x):
                a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
                x = x + a
                x = x + self.ffn(self.ln2(x))
                return x

        class _L2Net(nn.Module):
            """512 维去噪器: (B,512) → (B,512)，逐 token 序列化后过 Transformer。"""

            def __init__(self, dim_, h, hd, n_layers, sl, ei):
                super().__init__()
                self.seq_len = sl
                self.emb_in = ei
                self.in_proj = nn.Linear(ei, h)
                self.blocks = nn.ModuleList([_Block(h, hd) for _ in range(n_layers)])
                self.out_proj = nn.Linear(h, ei)

            def forward(self, x):
                b = x.shape[0]
                h = x.view(b, self.seq_len, self.emb_in)   # (B,16,32)
                h = self.in_proj(h)                        # (B,16,256)
                for blk in self.blocks:
                    h = blk(h)
                h = self.out_proj(h)                       # (B,16,32)
                return h.reshape(b, -1)                    # (B,512)

        return _L2Net(self.dim, hidden, heads, layers, seq_len, emb_in)

    # ------------------------------------------------------------------
    # numpy 降级路径（DEMO: 随机正交投影 + 门控 伪扩散器，规范 §6.1 降级边界）
    # ------------------------------------------------------------------
    def _init_numpy(self):
        """DEMO 伪扩散器: 随机正交投影 P(512×512) + 门控向量 + 8→512 情绪投影。

        无训练参数（train_step 为 no-op），仅用于流程验证（规范 §六.1）。
        迭代 z ← damping·z + (1-damping)·(P@z ⊙ gate) + 0.05·情绪偏置，收缩映射
        → 收敛到由 P/门控/情绪偏置刻画的稳定方向，模拟 Transformer 扩散的稳态。
        """
        self.use_torch = False
        self.model = None
        self.model_cpu = None
        rng = np.random.default_rng(20240601)               # 固定种子: DEMO 流程可复现
        # 随机正交投影: 高斯矩阵 QR 正交化（512² 规模，初始化一次，<100ms）
        a = rng.standard_normal((self.dim, self.dim)).astype(np.float32)
        q, _ = np.linalg.qr(a)
        self.P = q.astype(np.float32)                       # 512×512 正交矩阵
        self.gate = rng.uniform(0.3, 0.9, self.dim).astype(np.float32)   # 门控向量
        self.emotion_proj = (rng.standard_normal((8, self.dim)) * 0.1).astype(np.float32)  # 8→512
        self.damping = float(self.cfg.damping)              # 阻尼系数（cfg.damping=0.7）
        self.logger.warning(
            "\033[33m[DEMO] L2 numpy 伪扩散器（随机正交投影 + 门控，无训练，规范 §6.1）\033[0m")

    # ------------------------------------------------------------------
    # 推理入口（line_controller.py 多接口兼容契约）
    # ------------------------------------------------------------------
    def run(self, state_vector, emotion_vec, text_history="") -> dict:
        """主入口: 情绪调制扩散 → {"vector": (512,) float32 归一化, "confidence": float}。

        emotion_vec 按规范 §3.2 布局: [angry,sad,happy,neutral,fear,intensity,arousal,valence]；
        text_history 做弱语义调制（DEMO 路径以伪嵌入注入，规范 §4.3 随机召回注入同款思路；
        REAL 路径预留 TODO(V3.4) 接入文本编码器）。
        """
        y, conf = self._diffuse(state_vector, emotion_vec, text_history=text_history)
        self.confidence = float(conf)
        return {"vector": np.asarray(y, dtype=np.float32), "confidence": float(conf)}

    def diffuse(self, state_vector, emotion_vec) -> dict:
        """兼容接口: 同 run（line_controller.py 尝试顺序第 2 位）。"""
        return self.run(state_vector, emotion_vec)

    def forward(self, state_vector) -> "np.ndarray":
        """兼容接口: 单次去噪前向 → 512 维归一化向量（line_controller.py 尝试顺序第 3 位）。"""
        x = self._coerce_input(state_vector)
        if self.use_torch and self.model is not None:
            try:
                torch = safe_import("torch")
                # 跨设备迁移 .to(device): 仅 (512,) 小张量，延迟 < 1ms（规范 §九.5）
                xt = torch.from_numpy(x).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    out = self.model(xt)
                return normalize(out.detach().cpu().numpy().reshape(-1))
            except Exception as e:
                self.logger.warning(
                    "\033[33m[L2] forward torch 失败，降级 numpy: %r\033[0m", e)
                if not hasattr(self, "P"):
                    self._init_numpy()
        # numpy 单次伪投影
        return normalize(self.P @ x)

    def __call__(self, state_vector, emotion_vec) -> dict:
        """兼容接口: 同 run（line_controller.py 尝试顺序第 4 位）。"""
        return self.run(state_vector, emotion_vec)

    # ------------------------------------------------------------------
    # 扩散核心（torch / numpy 双实现，关键路径 try-except 降级，规范 §九.2）
    # ------------------------------------------------------------------
    def _diffuse(self, state_vector, emotion_vec, steps=None, text_history=""):
        """统一扩散入口: 迭代去噪 + 情绪调制 → (y(512,), confidence)。"""
        steps = int(steps) if steps is not None else int(self.cfg.iteration_steps)
        x = self._coerce_input(state_vector)
        if self.use_torch and self.model is not None:
            try:
                return self._diffuse_torch(x, emotion_vec, steps)
            except Exception as e:
                self.logger.warning(
                    "\033[33m[L2] torch 扩散失败，降级 numpy: %r\033[0m", e)
                if not hasattr(self, "P"):
                    self._init_numpy()          # torch 模式下未建 numpy 状态 → 惰性构建
        return self._diffuse_numpy(x, emotion_vec, steps, text_history)

    def _diffuse_torch(self, x, emotion_vec, steps):
        """torch 扩散: 迭代 h ← normalize(gate·model(h) + (1-gate)·h)（情绪强度调制残差）。

        gate = 情绪强度映射: 强度高 → 更激进去噪，强度低 → 保守保持原状；
        每步归一化防止随机初始化模型输出发散（与 L1 每步归一化同款策略）。
        confidence = 1 - 末次迭代相对变化率（收敛程度）。
        """
        torch = safe_import("torch")
        gate = self._emotion_gate(emotion_vec)
        # 跨设备迁移 .to(device): 仅 (512,) 小张量，延迟 < 1ms（规范 §九.5）
        h = torch.from_numpy(x).unsqueeze(0).to(self.device)
        self.model.eval()
        prev = h
        delta = 0.0
        with torch.no_grad():
            for _ in range(steps):
                out = self.model(h)
                h = gate * out + (1.0 - gate) * h          # 情绪强度调制残差
                h = h / (h.norm() + 1e-6)                  # 每步归一化防发散
                delta = float((h - prev).norm().item()) / (float(h.norm().item()) + 1e-6)
                prev = h
        # 跨设备迁移 .cpu(): 仅 (512,) 小张量，延迟 < 1ms（规范 §九.5）
        y = h.detach().cpu().numpy().reshape(-1)
        return normalize(y), clamp(1.0 - delta)

    def _diffuse_numpy(self, x, emotion_vec, steps, text_history=""):
        """DEMO 伪扩散: z ← damping·z + (1-damping)·(P@z ⊙ gate) + 0.05·情绪偏置（每步归一化）。

        情绪偏置 = ev(8) @ emotion_proj(8×512)（情绪向量 → 状态空间投影），强度越大门控越强；
        文本历史以伪嵌入弱调制初始状态（规范 §4.3 “触景生情”注入思路，DEMO 仅流程验证）。
        收缩映射 → 稳定收敛，confidence = 1 - 末次迭代相对变化率。
        """
        ev = np.asarray(emotion_vec, dtype=np.float32).reshape(-1)
        if ev.size < 8:                                    # 补齐规范 §3.2 的 8 维布局
            ev = np.concatenate([ev, np.zeros(8 - ev.size, dtype=np.float32)])
        intensity = self._emotion_intensity(ev)
        gate = np.clip(self.gate + 0.2 * intensity, 0.0, 1.0)          # 情绪调制门控
        # 情绪偏置 = ev(8) @ emotion_proj(8×512) → (512,)（情绪向量 → 状态空间投影）
        eb = (ev @ self.emotion_proj).astype(np.float32)
        z = x.astype(np.float32)
        if text_history:                                   # 文本历史弱调制（TODO(V3.4) 换真编码器）
            z = z + 0.05 * pseudo_embedding(text_history, self.dim)
        prev = z
        delta = 0.0
        for _ in range(steps):
            z = self.damping * z + (1.0 - self.damping) * ((self.P @ z) * gate) + 0.05 * eb
            z = normalize(z)
            delta = float(np.linalg.norm(z - prev)) / (float(np.linalg.norm(z)) + 1e-6)
            prev = z
        return normalize(z), clamp(1.0 - delta)

    # ------------------------------------------------------------------
    # 双缓冲训练（规范 §4.2 L2 / §十.15）
    # ------------------------------------------------------------------
    def train_step(self, inputs, targets) -> float:
        """训练 CPU 副本一步（MSE + SGD），返回 loss float。

        双缓冲语义:
          · 只更新 self.model_cpu（CPU 副本）与优化器，禁止触碰 GPU 主模型 self.model；
          · 将相对改进 (loss_before - loss_after)/(loss_before+eps) 累计到
            self._pending_improvement，由 swap() 依据 cfg.l2_swap_threshold 决策原子替换。
        DEMO（numpy 伪扩散器）: 不可训练，黄色日志标记 [DEMO] 并返回 0.0（no-op，§六.1）。
        """
        if not (self.use_torch and self.model_cpu is not None):
            self.logger.warning(
                "\033[33m[DEMO] L2 train_step 为 no-op（numpy 伪扩散器不可训练），返回 0.0\033[0m")
            self._pending_improvement = 0.0
            self.last_loss = 0.0
            return 0.0
        torch = safe_import("torch")
        try:
            x = self._coerce_input(inputs)
            t = self._coerce_input(targets)
            xt = torch.from_numpy(x).unsqueeze(0)     # CPU 张量（副本本来就在 CPU）
            tt = torch.from_numpy(t).unsqueeze(0)
            self.model_cpu.eval()
            with torch.no_grad():
                loss_before = float(((self.model_cpu(xt) - tt) ** 2).mean().item())
            for g in self.optimizer.param_groups:     # 学习率可调（trainer.py 可传参场景）
                g["lr"] = float(self.lr)
            self.model_cpu.train()
            self.optimizer.zero_grad()
            pred = self.model_cpu(xt)
            loss = ((pred - tt) ** 2).mean()          # MSE 去噪损失
            loss.backward()
            self.optimizer.step()                     # 仅 CPU 副本参数被更新
            self.model_cpu.eval()
            with torch.no_grad():
                loss_after = float(((self.model_cpu(xt) - tt) ** 2).mean().item())
            improvement = (loss_before - loss_after) / (loss_before + 1e-9)
            self._pending_improvement = max(self._pending_improvement, improvement)
            self._train_steps = getattr(self, "_train_steps", 0) + 1
            self.last_loss = float(loss_after)
            return float(loss_after)
        except Exception as e:
            # 关键路径降级（规范 §九.2）: 训练失败不崩溃，转 no-op
            self.logger.warning("\033[33m[L2] torch 训练失败，降级 no-op: %r\033[0m", e)
            return 0.0

    def swap(self) -> dict:
        """原子替换（规范 §十.15）: 若 CPU 副本改进 ≥ cfg.l2_swap_threshold 则替换主模型。

        替换方式为 Python 对象引用切换（self.model = 新对象）→ 原子、无中间态；
        返回 {"swapped": bool, "improvement": float}。improvement 取 train_step 累计最大值。
        """
        imp = float(self._pending_improvement)
        self._pending_improvement = 0.0               # 一次性消费，避免重复替换
        threshold = float(self.cfg.l2_swap_threshold)  # 0.02
        if imp < threshold:
            self.logger.info("[L2] swap 跳过: 改进 %.4f < 阈值 %.4f", imp, threshold)
            return {"swapped": False, "improvement": imp}
        if self.use_torch and self.model_cpu is not None:
            torch = safe_import("torch")
            try:
                # 跨设备迁移 .to(device): 仅 ~4MB 权重，延迟 < 10ms（规范 §九.5）
                new_model = copy.deepcopy(self.model_cpu).to(self.device).eval()
                self.model = new_model                # 原子替换（引用切换，无中间态）
                self.logger.info("[L2] 双缓冲原子替换完成（CPU 训练副本 → 主模型，改进 %.4f）", imp)
                return {"swapped": True, "improvement": imp}
            except Exception as e:
                self.logger.warning("\033[33m[L2] swap 失败: %r\033[0m", e)
                return {"swapped": False, "improvement": imp}
        self.logger.info("[L2] DEMO 模式无训练副本，swap no-op")
        return {"swapped": False, "improvement": 0.0}

    # ------------------------------------------------------------------
    # 持久化（训练完成后的权重保存 / 启动恢复，规范 §4.3 长期记忆）
    # ------------------------------------------------------------------
    def save_weights(self, path: str):
        """保存 L2 主模型权重（torch state_dict）→ path（如 models/l2_weights.pt）。

        保存的是双缓冲"主模型"（swap 后的最新权重）；结构信息一并写入。
        DEMO（无 torch）→ 仅记录日志。
        """
        torch = safe_import("torch")
        if not (self.use_torch and self.model is not None and torch is not None):
            self.logger.warning("[L2] 无 torch 主模型，跳过保存")
            return
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            torch.save({"state_dict": self.model.state_dict(),
                        "dim": self.dim, "hidden": self.cfg.l2_hidden,
                        "layers": self.cfg.l2_layers, "heads": self.cfg.l2_heads,
                        "trained_steps": getattr(self, "_train_steps", 0)},
                       path)
            self.logger.info("[L2] 权重已保存 → %s", path)
        except Exception as e:
            self.logger.warning("\033[33m[L2] 保存失败: %r\033[0m", e)

    def load_weights(self, path: str) -> bool:
        """从 save_weights 产物恢复主模型与 CPU 训练副本。

        返回是否成功；形状不匹配/文件缺失 → False（不崩溃，保留初始权重）。
        """
        torch = safe_import("torch")
        if not (self.use_torch and torch is not None) or not os.path.isfile(path):
            return False
        try:
            state = torch.load(path, map_location="cpu")
            sd = state.get("state_dict", state)
            # 校验结构（维度一致才加载）
            if int(state.get("hidden", self.cfg.l2_hidden)) != self.cfg.l2_hidden:
                raise ValueError("L2 保存的 hidden 与当前配置不符")
            self.model.load_state_dict(sd)
            self.model.eval()
            if self.model_cpu is not None:
                self.model_cpu.load_state_dict(sd)
                self.model_cpu.eval()
            self.logger.info("[L2] 权重已恢复 ← %s（steps=%s）",
                             path, state.get("trained_steps", "?"))
            return True
        except Exception as e:
            self.logger.warning("\033[33m[L2] 加载失败: %r\033[0m", e)
            return False

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def _coerce_input(self, v) -> "np.ndarray":
        """规整输入为 self.dim 维 float32 一维数组（长度不足补零、超出截断）。"""
        a = np.asarray(v, dtype=np.float32).reshape(-1)
        if a.size == 0:
            return np.zeros(self.dim, dtype=np.float32)
        if a.size != self.dim:
            a = np.resize(a, self.dim).astype(np.float32)
        return a

    @staticmethod
    def _emotion_intensity(emotion_vec) -> float:
        """情绪强度: 规范 §3.2 布局第 6 维为 intensity（0~1）；长度不足时取均值兜底。"""
        ev = np.asarray(emotion_vec, dtype=np.float32).reshape(-1)
        if ev.size >= 6:
            return clamp(float(ev[5]))
        if ev.size >= 1:
            return clamp(float(np.mean(ev)))
        return 0.5

    def _emotion_gate(self, emotion_vec) -> float:
        """情绪强度 → 去噪门控系数 [0.2, 1.0]（强度 0.5 → 门控 0.75）。"""
        return clamp(0.5 + 0.5 * self._emotion_intensity(emotion_vec), 0.2, 1.0)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 自检（规范 §九.3）: 无 torch 时验证 DEMO 降级路径可 import、接口齐全、数值契约成立
    import time

    _cfg = get_config()
    _t0 = time.perf_counter()
    _d = L2Diffuser(_cfg)
    _init_ms = (time.perf_counter() - _t0) * 1000.0

    _sv = np.random.RandomState(0).randn(512).astype(np.float32)
    _ev = np.array([0.1, 0.2, 0.4, 0.2, 0.1, 0.6, 0.3, 0.5], dtype=np.float32)  # 规范 §3.2 布局
    _txt = "你好，昔涟。你还记得花海吗？"

    # 1) run: dict 契约 + 数值约束
    _r = _d.run(_sv, _ev, _txt)
    assert isinstance(_r, dict) and set(_r) >= {"vector", "confidence"}
    assert _r["vector"].shape == (512,) and _r["vector"].dtype == np.float32
    assert abs(float(np.linalg.norm(_r["vector"])) - 1.0) < 1e-3, "输出必须为单位向量"
    assert 0.0 <= _r["confidence"] <= 1.0

    # 2) 接口齐全性（line_controller.py 尝试顺序契约: run/diffuse/forward/__call__）
    _outs = {
        "run": _d.run(_sv, _ev, _txt),
        "diffuse": _d.diffuse(_sv, _ev),
        "forward": _d.forward(_sv),
        "__call__": _d(_sv, _ev),
    }
    assert _outs["diffuse"]["vector"].shape == (512,)
    assert _outs["forward"].shape == (512,)
    assert _outs["__call__"]["vector"].shape == (512,)

    # 3) 双缓冲: train_step 不触碰主模型 + swap 阈值决策
    _loss = _d.train_step(_sv, _sv)
    _sw = _d.swap()
    assert isinstance(_sw, dict) and set(_sw) >= {"swapped", "improvement"}
    assert _sw["swapped"] in (True, False) and _sw["improvement"] >= 0.0
    assert _loss >= 0.0

    # 4) 情绪/文本调制: 不同情绪向量应产生可区分输出（流程验证，非语义验证）
    _ev_high = _ev.copy(); _ev_high[5] = 1.0
    _r_high = _d.run(_sv, _ev_high, "")
    _diff_norm = float(np.linalg.norm(_r["vector"] - _r_high["vector"]))

    print("[SELFTEST] L2Diffuser 自检通过")
    print(f"  mode        : {'REAL(torch)' if _d.use_torch else 'DEMO(numpy 伪扩散器)'}")
    print(f"  初始化耗时  : {_init_ms:.1f}ms  VRAM={gpu_mem_mb():.0f}MB")
    print(f"  run         : vector={_r['vector'].shape} {_r['vector'].dtype} "
          f"norm={np.linalg.norm(_r['vector']):.4f} confidence={_r['confidence']:.3f}")
    print(f"  接口         : run/diffuse/forward/__call__ 全部可用")
    print(f"  train_step  : loss={_loss:.4f}  swap={_sw}")
    print(f"  情绪调制     : 强度 0.6 vs 1.0 输出差异 L2={_diff_norm:.4f}")
    print(f"  用到的配置   : STATE_DIM={_cfg.STATE_DIM} l2_hidden={_cfg.l2_hidden} "
          f"l2_layers={_cfg.l2_layers} l2_heads={_cfg.l2_heads} "
          f"l2_swap_threshold={_cfg.l2_swap_threshold} iteration_steps={_cfg.iteration_steps} "
          f"damping={_cfg.damping}")
