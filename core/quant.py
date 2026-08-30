# -*- coding: utf-8 -*-
"""
core/quant.py — NF4 4-bit 量化工具（小模型专用，纯 torch 实现）
===============================================================
说明: 匹配器/情感回环/路由等小模型按【硬性】要求以 4-bit NF4 常驻显存。
  Qwen 走 bitsandbytes（config.bnb_quant_kwargs），本模块实现与 bnb 同款
  NF4 码本的"打包量化"（每字节装 2 个 4-bit 值），供自定义小模型使用:
  避免为 8000 个 0.1M 线性层实例化 bnb 模块的过高开销。

存储格式（每层权重一张 "量化矩阵"）:
  q      : uint8 (..., numel/2)  —— 每字节低 4 bit = 第 2i 个权重，高 4 bit = 第 2i+1 个
  absmax : float32 (...)         —— 该张量的最大绝对值（NF4 归一化尺度）
  码本   : NF4_CODEBOOK（16 级，与 bitsandbytes 官方码本一致）

访存: 显存驻留仅 q + absmax（400MB 量级，满足"总约 0.4GB"预算）；
  前向时按块解量化到 fp16 临时代理（块内瞬态 < 100MB，峰值受控）。

原子保存: save_state_atomic() 先写 .tmp，成功后 os.replace（【硬性】 §十四）。
"""
import os
import torch

# ----------------------------------------------------------------------
# NF4 码本（bitsandbytes 官方 16 级，升序）
# ----------------------------------------------------------------------
NF4_CODEBOOK = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.1921844780445099, -0.11626319587230682,
    -0.0545236012339592, 0.01851463690304756, 0.07869209349155426,
    0.12533240390729904, 0.20851804316043854, 0.33390918374061584,
    0.539726048707962, 0.8593318462371826, 1.0,
], dtype=torch.float32)


# ----------------------------------------------------------------------
# 量化 / 解量化
# ----------------------------------------------------------------------
def quantize_nf4(t: torch.Tensor) -> tuple:
    """把 float 张量量化为 (packed_uint8, absmax)。

    算法:
      1. absmax = max|t|（为 0 时置 1，避免除零）
      2. t_norm = t / absmax ∈ [-1, 1]
      3. 对每个值在 NF4_CODEBOOK 中找最近码 → 4-bit 索引
      4. 两个 4-bit 索引打包进一个 uint8（低 nibble 先）

    返回的 packed 与原张量维度一致（最后一维减半），absmax 同形状（除最后一维）。
    """
    if t.dtype not in (torch.float16, torch.float32):
        t = t.float()
    absmax = t.abs().amax(dim=-1, keepdim=True)
    safe = torch.clamp(absmax, min=1e-8)
    normed = t / safe
    # 最近码本索引: 码本升序 → 用相邻码本中点的 bucketize 定位（O(n log16)，免距离矩阵）
    cb = NF4_CODEBOOK.to(t.device)
    boundaries = (cb[1:] + cb[:-1]) / 2.0
    idx = torch.bucketize(normed, boundaries).to(torch.uint8)   # 0..15
    if idx.shape[-1] % 2 != 0:
        idx = torch.cat([idx, torch.zeros_like(idx[..., :1])], dim=-1)  # 奇数补 0
    idx = idx.reshape(*idx.shape[:-1], -1, 2)
    packed = idx[..., 0] | (idx[..., 1] << 4)       # 低 nibble = 第 2i 个
    return packed.contiguous(), absmax.reshape(-1).contiguous()


def dequantize_nf4(packed: torch.Tensor, absmax: torch.Tensor, numel: int = None,
                   rows: int = 1) -> torch.Tensor:
    """解量化 packed_uint8 (N, bytes) + absmax (N,) → fp16 张量 (N, numel)。

    参数:
      packed : uint8 (N, bytes)，每字节低 nibble = 第 2i 个权重、高 nibble = 第 2i+1 个
      absmax : float32 (N,) 每行的 NF4 归一化尺度
      numel  : 每行权重数（奇数值时行内独立填充，跨行不错位）
      rows   : 行数（默认 1；以 packed.shape[0] 为准，语义冗余）
    """
    if packed.dim() == 1:                              # 单行 (bytes,) → (1, bytes)
        packed = packed.unsqueeze(0)
    n = packed.shape[0]
    cb = NF4_CODEBOOK.to(packed.device)
    lo = (packed & 0xF).to(torch.long)                  # (N, bytes)
    hi = ((packed >> 4) & 0xF).to(torch.long)
    v = torch.empty((n, packed.shape[1], 2), dtype=torch.float32, device=packed.device)
    v[..., 0] = cb[lo]
    v[..., 1] = cb[hi]
    v = v.reshape(n, -1)                                # (N, bytes*2)
    if numel is not None:
        v = v[:, :numel]                                # 每行截断（奇数行独立补齐，不错位）
    scale = absmax.reshape(-1)
    if scale.numel() == 1:
        v = v * scale[0]
    elif scale.numel() == n:
        v = v * scale.unsqueeze(1)
    else:
        v = v * scale[:n].unsqueeze(1)
    return v.to(torch.float16)


def dequantize_rows(packed: torch.Tensor, absmax: torch.Tensor, col: int) -> torch.Tensor:
    """批量解量化 (N, bytes) 打包张量 → (N, col) float16。"""
    return dequantize_nf4(packed, absmax, numel=col)


# ----------------------------------------------------------------------
# 原子保存 / 加载
# ----------------------------------------------------------------------
def save_state_atomic(state: dict, path: str):
    """原子保存: torch.save 到 path.tmp → os.replace(path)。

    防止中途崩溃写出半截 .pt（【硬性】 §十四 原子保存）。
    """
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_state_atomic(path: str) -> dict:
    """加载 .pt state（路径不存在返回空 dict）。"""
    if not os.path.isfile(path):
        return {}
    return torch.load(path, map_location="cpu", weights_only=True)


# ----------------------------------------------------------------------
def self_test():
    """量化往返自检（--selftest 用）。"""
    torch.manual_seed(0)
    x = torch.randn(8, 1024, dtype=torch.float16) * 0.2
    q, s = quantize_nf4(x)
    x2 = dequantize_nf4(q, s, numel=x.numel()).view_as(x)
    rel_err = float(((x.float() - x2.float()) ** 2).mean().sqrt() / x.float().abs().mean())
    assert rel_err < 0.2, rel_err           # NF4 4-bit 相对误差应在 ~5-15% 量级
    assert q.shape[-1] == x.shape[-1] // 2 and q.dtype == torch.uint8
    return rel_err
