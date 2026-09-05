# -*- coding: utf-8 -*-
"""
scripts/migrate_z_384_to_768.py — 昔涟AGI v8.0 记忆向量迁移工具
================================================================================
作用: 把 v7.3 的 384 维 Z 向量/编码头/匹配器权重，线性延拓升维到 768 维，
      对齐 MiniMind-O Thinker 的语义空间，做到「不浪费已有积累」。

【重要】本脚本默认【只预览不写入】（--dry-run）。它不会主动改动你的权重，
      只有显式加 --apply 才会落盘，且落盘用原子保存（.tmp + os.replace），
      权重会额外备份一份到 <path>.bak_384。

【三种迁移模式】
  1. 编码头 head.pt（Qwen 桥的 project 矩阵）: project [D,384] 扩展为 [D,768]
  2. 匹配器合并版 matchers_combined.pt: NF4 打包 uint8 [N, bytes384] → [N, bytes768]
  3. 记忆池向量: memory_pools 下的 384 维 z 向量 → 768 维

【运行】
  python scripts/migrate_z_384_to_768.py --dry-run          # 只预览
  python scripts/migrate_z_384_to_768.py --apply --dir <path>
"""
from __future__ import annotations

import os
import sys
import argparse
import logging

import numpy as np

# 支持从项目根 import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="[MIGRATE] %(message)s")
log = logging.getLogger("migrate")

D384 = 384
D768 = 768


def _bytes_for(total_params: int) -> int:
    """NF4 打包：每字节 2 参数。"""
    return (total_params + 1) // 2


def upgrade_linear_w(W384: np.ndarray, seed: int = 0x7CA1) -> np.ndarray:
    """把线性矩阵 [B,384] / [B,K] 延拓到 [B,768]（下块随机保底）。"""
    W = np.asarray(W384, dtype=np.float32)
    if W.ndim != 2 or W.shape[1] != D384:
        raise ValueError(f"预期 [B,384]，实际 {W.shape}")
    rng = np.random.RandomState(seed)
    B, _ = W.shape
    scale = 0.05 * (1.0 / np.sqrt(B))
    G = rng.normal(0.0, scale, (B, D768 - D384)).astype(np.float32)
    return np.concatenate([W, G], axis=1)


def upgrade_nf4_block(q384: np.ndarray, total_old: int, total_new: int) -> np.ndarray:
    """把 NF4 打包 uint8 行从 384 维升到 768 维。

    由于 NF4 是 4-bit 打包（每字节 2 参数），直接扩展字节数即可：
      新行 = 旧行 + 随机填充的额外字节（对应新增维度）。
    这里采用「信息无损扩展」：旧 384 维部分原样保留，新增维度用 0 均值噪声对齐。
    """
    q = np.asarray(q384, dtype=np.uint8).reshape(-1)
    old_bytes = _bytes_for(total_old)
    new_bytes = _bytes_for(total_new)
    out = np.zeros(new_bytes, dtype=np.uint8)
    out[:min(old_bytes, new_bytes)] = q[:min(old_bytes, new_bytes)]
    # 新增部分用随机 4-bit 值（范围 0-15）填充，保证不全 0（避免解量化退化为同值）
    rng = np.random.RandomState(0x7CA1)
    pad = rng.randint(0, 16, size=new_bytes - old_bytes).astype(np.uint8)
    out[old_bytes:] = pad
    return out


def migrate_head(head_path: str, dry: bool = True):
    """迁移 Qwen 编码头 head.pt 的 project [384->768]。"""
    import torch
    if not os.path.isfile(head_path):
        log.warning("编码头不存在: %s", head_path)
        return
    st = torch.load(head_path, map_location="cpu", weights_only=False)
    key = "project" if "project" in st else "weight"
    w = np.asarray(st[key], dtype=np.float32)
    if w.ndim != 2 or w.shape[-1] != D384:
        log.info("编码头 %s 维度 %s，非 384 维，跳过", head_path, w.shape)
        return
    w768 = upgrade_linear_w(w)
    new_st = dict(st)
    new_st[key] = w768
    new_st["z_dim"] = D768
    new_st["migrated_from"] = D384
    backup = head_path + ".bak_384"
    log.info("[HEAD] %s → %s (dry=%s)", w.shape, w768.shape, dry)
    if dry:
        return
    # 备份原始
    import shutil
    shutil.copy2(head_path, backup)
    # 原子保存
    tmp = head_path + ".tmp"
    torch.save(new_st, tmp)
    os.replace(tmp, head_path)
    log.info("[HEAD] 已迁移并保存（原文件备份 %s）", backup)


def migrate_combined(pt_path: str, total_old: int, total_new: int, dry: bool = True):
    """迁移匹配器合并版 matchers_combined.pt（NF4 打包扩展）。"""
    import torch, shutil
    if not os.path.isfile(pt_path):
        log.warning("匹配器合并版不存在: %s", pt_path)
        return
    st = torch.load(pt_path, map_location="cpu", weights_only=False)
    changed = False
    for k, v in list(st.items()):
        arr = np.asarray(v)
        if arr.ndim == 2 and arr.shape[1] == _bytes_for(total_old):
            arr768 = np.stack([
                upgrade_nf4_block(row, total_old, total_new) for row in arr
            ])
            st[k] = torch.as_tensor(arr768, dtype=v.dtype) if hasattr(v, "dtype") else arr768
            changed = True
            log.info("[COMBINED] 键 %s: %s → %s (dry=%s)", k, arr.shape, arr768.shape, dry)
    if not changed:
        log.info("[COMBINED] 未发现需迁移的 384 维打包张量，跳过: %s", pt_path)
        return
    if dry:
        return
    shutil.copy2(pt_path, pt_path + ".bak_384")
    tmp = pt_path + ".tmp"
    torch.save(st, tmp)
    os.replace(tmp, pt_path)
    log.info("[COMBINED] 已迁移（原文件备份 %s.bak_384）", pt_path)


def migrate_pool_vectors(pool_dir: str, dry: bool = True):
    """迁移记忆池下的 384 维 z 向量到 768 维。"""
    import torch
    if not os.path.isdir(pool_dir):
        log.warning("记忆池目录不存在: %s", pool_dir)
        return
    for root, _, files in os.walk(pool_dir):
        for fn in files:
            if not fn.endswith(".pt"):
                continue
            p = os.path.join(root, fn)
            try:
                st = torch.load(p, map_location="cpu", weights_only=False)
            except Exception:
                continue
            if not isinstance(st, dict):
                continue
            for k, v in list(st.items()):
                arr = np.asarray(v, dtype=np.float32)
                if arr.ndim == 1 and arr.shape[0] == D384:
                    st[k] = torch.as_tensor(upgrade_linear_w(arr.reshape(1, D384)).reshape(-1))
                    log.info("[POOL] %s key=%s: 384→768 (dry=%s)", fn, k, dry)
                elif arr.ndim == 2 and arr.shape[1] == D384:
                    st[k] = torch.as_tensor(upgrade_linear_w(arr))
                    log.info("[POOL] %s key=%s: [%s]→768 (dry=%s)", fn, k, arr.shape, dry)
            if dry:
                continue
            import shutil
            shutil.copy2(p, p + ".bak_384")
            torch.save(st, p + ".tmp")
            os.replace(p + ".tmp", p)
            log.info("[POOL] 已迁移: %s", p)


def main(argv=None):
    ap = argparse.ArgumentParser(description="v8.0 Z 向量 384→768 迁移工具")
    ap.add_argument("--dir", default=None,
                    help="扫描目录（默认工作区 models/ 与 memory_pools/）")
    ap.add_argument("--apply", action="store_true",
                    help="真正落盘（默认仅 dry-run 预览）")
    ap.add_argument("--heads", action="store_true", help="处理编码头 head.pt")
    ap.add_argument("--combined", action="store_true", help="处理匹配器合并版")
    ap.add_argument("--pools", action="store_true", help="处理记忆池向量")
    args = ap.parse_args(argv)

    base = args.dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base, "models")
    pool_dir = os.path.join(base, "memory_pools")
    total_old = 115073   # v7.3 [384,256,64,1]
    total_new = 213377   # v8.0 [768,256,64,1]

    do_all = not (args.heads or args.combined or args.pools)
    if do_all or args.heads:
        for d in (os.path.join(models_dir, "qwen_encoder"),
                  os.path.join(models_dir, "MiniMind-O-0.1B")):
            migrate_head(os.path.join(d, "head.pt"), dry=not args.apply)
    if do_all or args.combined:
        for d in (os.path.join(models_dir, "matchers"),):
            migrate_combined(os.path.join(d, "matchers_combined.pt"),
                             total_old, total_new, dry=not args.apply)
    if do_all or args.pools:
        migrate_pool_vectors(pool_dir, dry=not args.apply)

    log.info("迁移扫描完成（apply=%s）。如需落盘请加 --apply。", args.apply)


if __name__ == "__main__":
    main()
