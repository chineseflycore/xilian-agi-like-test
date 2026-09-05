# -*- coding: utf-8 -*-
"""
projector.py — 昔涟AGI v8.0 轻量投影层（Z 向量 → MiniMind Thinker 语义空间）
================================================================================
内存/显存预估: 权重 < 1M 参数（FP32 约 4MB / FP16 约 2MB），一般不进 GPU 常驻。

【设计角色】
  昔涟认知引擎输出的 768 维 Z 向量，与 MiniMind-O Thinker 的 768 维 hidden 状态
  虽同维度，但处于【不同语义空间】。本模块提供一个 1~2 层 MLP 的「翻译层」，
  把 Z 向量对齐到 Thinker 的输入语义域，避免「翻译事故」。

【训练策略】（配合 scripts/）
  冻结 Thinker 与 Talker，仅训练本投影层；训练对为 (Z 向量, 期望回复文本)，
  损失为最终输出的交叉熵。训练完成后把 state_dict 落盘到
  models/minimind_o/projector.pt。

【置信度过滤】
  投影层输出后接一个轻量二分类器（置信度头），判断当前 Z 是否在训练分布内。
  低于 threshold 时调用方应回退到纯文本规则模式（不做模型生成），
  降低「域外输入导致胡言乱语」的风险。

【接口】
  - Projector(in_dim, hidden_dim, out_dim, num_layers, confidence_threshold)
  - .forward(z)             -> 投影后的高维向量 [*, out_dim]
  - .confidence(z)          -> 分布内置信度 [*, 1]
  - .project_with_conf(z)   -> (proj, conf) 一步返回
  - .save(path) / load(path)
  - Linear Migrate 兼容: 提供 upgrade_384_to_768(W, mean) 用于旧记忆向量迁移
"""
from __future__ import annotations

import os
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("xilian.projector")


# ---------------------------------------------------------------------------
# 投影层网络
# ---------------------------------------------------------------------------
class _ProjectorMLP(nn.Module):
    """轻量 MLP：Linear->LayerNorm->GELU ... ->Linear。参数量 < 1M。"""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers >= 1 (至少 1 层映射)")
        layers = []
        prev = in_dim
        for i in range(num_layers):
            mid = hidden_dim if i < num_layers - 1 else out_dim
            layers.append(nn.Linear(prev, mid))
            if i < num_layers - 1:
                layers.append(nn.LayerNorm(mid))
                layers.append(nn.GELU())
            prev = mid
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ConfidenceHead(nn.Module):
    """轻量二分类器：把投影特征映射到 [0,1] 标量置信度。"""

    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logit = self.net(x).squeeze(-1)
        return torch.sigmoid(logit)


class Projector(nn.Module):
    """轻量投影层 + 置信度过滤，封装成可直接使用的高层对象。"""

    def __init__(
        self,
        in_dim: int = 768,
        hidden_dim: int = 256,
        out_dim: int = 768,
        num_layers: int = 2,
        confidence_threshold: float = 0.7,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.confidence_threshold = float(confidence_threshold)

        self.proj = _ProjectorMLP(in_dim, hidden_dim, out_dim, num_layers)
        self.conf_head = _ConfidenceHead(out_dim)

    # ------------------------------------------------------------------
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """投影: [*, in_dim] -> [*, out_dim]，输出前做 L2 归一化换取稳定性。"""
        out = self.proj(z)
        n = torch.linalg.norm(out, dim=-1, keepdim=True).clamp(min=1e-6)
        return out / n

    def confidence(self, z: torch.Tensor) -> torch.Tensor:
        """置信度: [*, in_dim] -> [*] 概率。"""
        with torch.no_grad():
            proj = self.forward(z)
            return self.conf_head(proj)

    def project_with_conf(self, z: torch.Tensor):
        """一次返回 (投影向量, 置信度)。

        _ConfidenceHead.forward 内部已对 logit 做 sigmoid 返回 [0,1]，
        这里不重复 sigmoid，避免双重 sigmoid 导致置信度塌陷到 ~0.5 失真。
        """
        proj = self.forward(z)
        conf = self.conf_head(proj)
        return proj, conf

    @torch.no_grad()
    def filter_callable(self, z: np.ndarray):
        """供规则路径使用的纯 NumPy 接口：返回 (proj512, conf, in_dist)。"""
        if isinstance(z, np.ndarray):
            z_t = torch.as_tensor(z, dtype=torch.float32)
        else:
            z_t = torch.as_tensor(z, dtype=torch.float32)
        single = z_t.ndim == 1
        if single:
            z_t = z_t.unsqueeze(0)
        # 规则 Z 是 NumPy(CPU)，投影层可能在 GPU 上 —— 对齐设备避免 device mismatch
        device = next(self.parameters()).device
        if z_t.device != device:
            z_t = z_t.to(device)
        proj, conf = self.project_with_conf(z_t)
        proj_np = proj.cpu().numpy().reshape(-1, self.out_dim) if not single else proj.cpu().numpy().reshape(-1)
        conf_np = conf.cpu().numpy().reshape(-1)
        if single:
            conf_np = conf_np[0]
        in_dist = bool((conf_np >= self.confidence_threshold).all())
        return proj_np, float(np.mean(conf_np)), in_dist

    # ------------------------------------------------------------------
    # 持久化（原子保存）
    # ------------------------------------------------------------------
    def save(self, path: str):
        """原子保存 state_dict 到 path（先写 .tmp 再 os.replace）。"""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = path + ".tmp"
        torch.save(
            {
                "state_dict": self.state_dict(),
                "meta": {
                    "in_dim": self.in_dim,
                    "hidden_dim": self.hidden_dim,
                    "out_dim": self.out_dim,
                    "num_layers": self.num_layers,
                    "confidence_threshold": self.confidence_threshold,
                },
            },
            tmp,
        )
        os.replace(tmp, path)
        log.info("[PROJECTOR] 已保存: %s (%d params)", path, self._param_count())

    @classmethod
    def load(cls, path: str, device: str = "cpu", map_location: str = None):
        """加载 state_dict，若 meta 缺失则用默认参数。"""
        payload = torch.load(path, map_location=map_location or device, weights_only=False)
        meta = payload.get("meta", {})
        obj = cls(
            in_dim=meta.get("in_dim", 768),
            hidden_dim=meta.get("hidden_dim", 256),
            out_dim=meta.get("out_dim", 768),
            num_layers=meta.get("num_layers", 2),
            confidence_threshold=meta.get("confidence_threshold", 0.7),
        )
        obj.load_state_dict(payload["state_dict"])
        obj.to(device)
        obj.eval()
        log.info("[PROJECTOR] 已加载: %s", path)
        return obj

    def _param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# 数据迁移工具：384 维旧记忆向量 → 768 维（线性投影初始化）
# ---------------------------------------------------------------------------
def upgrade_384_to_768(W384, mean384=None, seed: int = 0x7CA1):
    """把 v7.3 的 384 维投影矩阵/编码头升维到 768 维（线性延拓，保信息不丢）。

    策略：
      - W384 形状 (384, D) 或 (D, 384)，统一转成 (B, A) 理解。
      - 新矩阵 = 上块(W384) ，下块用固定 seed 的随机小值填充，保证可复现。
      - 若提供 mean384，同理延拓 mean768（上=mean384，下=零）。

    返回 (W768, mean768)。可用作旧记忆中 384 维向量升级的线性基础。
    """
    W = np.asarray(W384, dtype=np.float32)
    if W.ndim != 2:
        raise ValueError("W384 必须是二维矩阵")
    # (B, A)，B 为「目标特征维度」，A 为原维度
    B, A = W.shape
    rng = np.random.RandomState(seed)
    G = rng.normal(0.0, (1.0 / np.sqrt(B)) * 0.05, (B, A)).astype(np.float32)
    # 做一个简单的对矩阵（若 384->768 用 2 倍分段延拓）
    W768 = np.zeros((B * 2, A), dtype=np.float32)
    W768[:B, :] = W
    W768[B:, :] = G  # 下块用噪声保底，避免结构丢失
    if mean384 is not None:
        mean = np.asarray(mean384, dtype=np.float32).reshape(-1)
        mean768 = np.concatenate([mean, np.zeros_like(mean)])
        return W768, mean768
    return W768


# ---------------------------------------------------------------------------
# 便捷函数：构建默认 Projector（从 config）
# ---------------------------------------------------------------------------
def build_default_projector(cfg) -> Projector:
    """根据 config.projector_conf 构建 Projector 实例。"""
    conf = getattr(cfg, "projector_conf", {})
    kwargs = {
        "in_dim": int(getattr(cfg, "z_dim", 768)),
        "hidden_dim": int(conf.get("hidden_dim", 256)),
        "out_dim": int(getattr(cfg, "minimind_hidden_dim", 768)),
        "num_layers": int(conf.get("hidden_layers", 2)),
        "confidence_threshold": float(conf.get("confidence_threshold", 0.7)),
    }
    obj = Projector(**kwargs)
    log.info("[PROJECTOR] 默认构建: %s", kwargs)
    return obj
