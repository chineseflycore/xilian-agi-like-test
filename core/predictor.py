# -*- coding: utf-8 -*-
"""
core/predictor.py — 因果预测器 v7.3（Z 序列预测）
==================================================
显存预估: ~0.02GB（~0.73M 参数: 磁盘 4-bit ≈ 0.4MB, 运行时 fp16 1.5MB 常驻）
本模块预测"下一个心跳的 Z 潜向量"（因果: 只用过去, 不看未来）。

架构（1 层 Transformer, 4 头, d=256）:
  输入: 过去 ≤10 心跳的 Z 序列 (10,384) + 情感向量 (8) → [B,10,392]
  embed(392→256) + 位置嵌入 → pre-norm MHA(4头) → FFN(256→512→256)
  → out_proj(256→384) → L2 归一 → Z_pred ∈ R^384

用途:
  - 深度思考（API 模式 local_deep）: 迭代 2~3 轮, Z_final = 0.75·Z_match + 0.25·Z_pred
  - 常规心跳: λ=0.12 混合（Z_mix = (1-λ)·Z_match + λ·Z_pred）
  - 休眠期每 5 周期 MSE 更新（数据源: 工作记忆中的 Z 序列）

存储: 4-bit NF4 打包落盘（quant.save_state_atomic）; 运行时 fp16 常驻（参数极小）。
"""
import os
import math

import numpy as np
import torch
import torch.nn.functional as F

import config
from core import quant


class _Block(torch.nn.Module):
    """pre-norm Transformer 块: MHA + FFN。"""

    def __init__(self, d_model: int, heads: int, ffn: int):
        super().__init__()
        self.ln1 = torch.nn.LayerNorm(d_model)
        self.attn = torch.nn.MultiheadAttention(d_model, heads,
                                                batch_first=True)
        self.ln2 = torch.nn.LayerNorm(d_model)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, ffn), torch.nn.GELU(),
            torch.nn.Linear(ffn, d_model))

    def forward(self, x):
        h = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                          need_weights=False)[0]
        return h + self.ffn(self.ln2(h))


class _PredictorNet(torch.nn.Module):
    """预测器网络本体: embed → 位置嵌入 → Transformer 块 → 输出头。"""

    def __init__(self, d_model: int, heads: int, layers: int,
                 in_dim: int, out_dim: int, seq: int):
        super().__init__()
        self.embed = torch.nn.Linear(in_dim, d_model)
        self.pos = torch.nn.Parameter(torch.zeros(seq, d_model),
                                      requires_grad=True)
        self.blocks = torch.nn.ModuleList([
            _Block(d_model, heads, d_model * 2)
            for _ in range(layers)])
        self.out = torch.nn.Linear(d_model, out_dim)

    def forward(self, x):
        h = self.embed(x) + self.pos[:x.shape[1]]
        for blk in self.blocks:
            h = blk(h)
        return self.out(h)


class ZPredictor:
    """因果预测器: 预测下一心跳 Z（384 维, 与 matcher 同表示域）。"""

    def __init__(self, cfg=None, device=None):
        self.cfg = cfg if cfg is not None else config.get_config()
        self.device = device or self.cfg.device
        self.log = self.cfg.setup_logging().getChild("predictor")
        self.path = self.cfg.predictor_model_file

        d = self.cfg.predictor_d_model
        self.d_model = d
        self.seq = self.cfg.predictor_seq
        self.in_dim = self.cfg.z_dim + len(self.cfg.emotion_names)   # 392
        self.model = _PredictorNet(d, self.cfg.predictor_heads,
                                   self.cfg.predictor_layers,
                                   self.in_dim, self.cfg.z_dim, self.seq)
        self.model.to(self.device)
        self.model.half()                 # fp16 常驻（显存优先）

        self._mse = float("nan")
        self._train_count = 0
        self._load_or_generate()

    # ==================================================================
    # 存储（NF4 4-bit 落盘 / fp16 常驻）
    # ==================================================================
    def _load_or_generate(self):
        """加载 predictor_nf4.pt; 缺失/格式不符 → 随机初始化并落盘。"""
        if os.path.isfile(self.path):
            try:
                st = quant.load_state_atomic(self.path)
                if st.get("version") == "7.3" and st.get("d_model") == self.d_model:
                    self._dequant_load(st)
                    self._mse = float(st.get("mse", 0.0))
                    self._train_count = int(st.get("train_count", 0))
                    self.log.info("[PREDICTOR] 加载完成 (d=%d mse=%.4f)",
                                  self.d_model, self._mse)
                    return
            except Exception:
                self.log.warning("[PREDICTOR] 模型损坏 → 重新生成占位")
        self._init_weights()
        self.save()
        self.log.info("[PREDICTOR] 生成随机占位模型 (%.2fM 参数)",
                      sum(p.numel() for p in self.model.parameters()) / 1e6)

    def _init_weights(self):
        """Xavier 初始化（≥2 维权重 Xavier; 1 维 bias/pos 归零, 占位模型生成逻辑）。

        注: nn.MultiheadAttention 的 in_proj_weight 为 2 维 → Xavier; 其余 1 维归零。
        """
        for name, p in self.model.named_parameters():
            if p.ndim >= 2:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)

    def save(self):
        """NF4 逐层量化落盘（原子写, 与全系统 4-bit 一致）。"""
        st = {"version": "7.3", "d_model": self.d_model,
              "mse": self._mse, "train_count": self._train_count,
              "q": {}, "scale": {}, "numel": {}, "shape": {}}
        for name, p in self.model.state_dict().items():
            t = p.detach().float().reshape(1, -1)
            q, s = quant.quantize_nf4(t)
            st["q"][name] = q.reshape(-1).contiguous()
            st["scale"][name] = s.contiguous()
            st["numel"][name] = int(t.numel())
            st["shape"][name] = list(p.shape)
        quant.save_state_atomic(st, self.path)

    def _dequant_load(self, st: dict):
        """从 NF4 state 恢复到 fp16 模型。"""
        sd = {}
        for name, q in st["q"].items():
            v = quant.dequantize_nf4(q.reshape(1, -1), st["scale"][name].reshape(-1),
                                     numel=st["numel"][name])
            sd[name] = v.reshape(st["shape"][name])
        self.model.load_state_dict(sd, strict=True)
        self.model.half()
        self.model.eval()

    # ==================================================================
    # 前向
    # ==================================================================
    def predict(self, z_seq, emotion) -> np.ndarray:
        """输入过去 Z 序列 [ (≤10), 384] + 情感 [8] → Z_pred [384]（L2 归一）。"""
        seq = self._pack(z_seq, emotion)
        self.model.eval()
        with torch.no_grad():
            out = self.model(seq)[:, -1]
            n = torch.linalg.norm(out, dim=-1, keepdim=True).clamp(min=1e-6)
            out = out / n
        return out.float().cpu().numpy().reshape(self.cfg.z_dim)

    def _pack(self, z_seq, emotion) -> torch.Tensor:
        """拼接序列并 pad 到定长 self.seq。"""
        zs = np.asarray(z_seq, dtype=np.float32)
        if zs.ndim == 1:
            zs = zs.reshape(1, -1)
        if zs.shape[1] != self.cfg.z_dim:
            pad = self.cfg.z_dim - zs.shape[1]
            zs = np.pad(zs, ((0, 0), (0, max(0, pad))))
        em = np.asarray(emotion, dtype=np.float32).reshape(1, -1)
        feats = np.concatenate([zs, np.tile(em, (zs.shape[0], 1))], axis=1)
        if feats.shape[0] < self.seq:
            feats = np.concatenate([feats,
                                    np.zeros((self.seq - feats.shape[0],
                                              feats.shape[1]), dtype=np.float32)])
        feats = feats[-self.seq:]
        return torch.as_tensor(feats, dtype=torch.float16,
                               device=self.device).unsqueeze(0)

    # ==================================================================
    # 训练（休眠期每 5 周期 MSE 更新）
    # ==================================================================
    def update(self, sequence: list, target_z, emotion, steps: int = 8):
        """MSE 微调: sequence = [z0..z9]（长度不足自动 pad 截断）, target = z10。"""
        self.model.train()
        self.model.float()                # 训练用 fp32（权重微小, 显存无压力）
        opt = torch.optim.Adam(self.model.parameters(),
                               lr=self.cfg.predictor_lr)
        loss_val = 0.0
        for _ in range(steps):
            seq = self._pack(sequence, emotion).float()
            tgt = torch.as_tensor(np.asarray(target_z, dtype=np.float32)
                                  .reshape(1, -1), device=self.device)
            opt.zero_grad()
            out = self.model(seq)[:, -1]
            n = torch.linalg.norm(out, dim=-1, keepdim=True).clamp(min=1e-6)
            out = out / n
            loss = F.mse_loss(out, tgt)
            loss.backward()
            # 权重裁剪（防震荡）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.cfg.predictor_weight_clip)
            opt.step()
            loss_val = float(loss.item())
        self.model.eval()
        self.model.half()
        self._mse = loss_val
        self._train_count += 1
        self.save()
        return loss_val

    # ==================================================================
    def get_stats(self) -> dict:
        return {"mse": self._mse, "train_count": self._train_count}
