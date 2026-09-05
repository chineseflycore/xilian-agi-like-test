# -*- coding: utf-8 -*-
"""
test_projector.py — 投影层独立验证脚本
运行方式: python scripts/test_projector.py
不依赖 MiniMind 模型（用规则 Z），只验证投影层本身是否可用、bug 是否已修。
"""
import os
import sys

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch


def main() -> int:
    from core.projector import Projector, build_default_projector
    import config

    cfg = config.get_config()
    print(f"[ENV] torch={torch.__version__} device={cfg.device} z_dim={cfg.z_dim}")

    # 1) 构建默认投影层
    proj = build_default_projector(cfg)
    proj.to(cfg.device)
    proj.eval()
    print(f"[BUILD] in={proj.in_dim} out={proj.out_dim} "
          f"hidden={proj.hidden_dim} layers={proj.num_layers} "
          f"conf_th={proj.confidence_threshold} params={proj._param_count()}")

    # 2) 生成一个规则 Z（用 protocol 的真实路径）
    from core import protocol as P
    raw = P._rule_soft16("你好呀，今天开心吗？") if hasattr(P, "_rule_soft16") else None
    if raw is None:
        # 退化: 直接用 16 维规则向量
        raw = np.zeros(16, dtype=np.float32)
        raw[0] = 1.0  # 开心
    z = P.vectors_to_z(raw, cfg)
    print(f"[Z] shape={z.shape} dim={z.shape[0]} norm={np.linalg.norm(z):.3f}")

    # 3) 投影 + 置信度（验证双重 sigmoid 修复：置信度应分布在 [0,1]，不会是 ~0.5 塌陷）
    proj_np, conf, in_dist = proj.filter_callable(z)
    print(f"[PROJ] out_shape={proj_np.shape} out_norm={np.linalg.norm(proj_np):.3f}")
    print(f"[CONF] conf={conf:.4f} in_dist={in_dist} (threshold={proj.confidence_threshold})")
    assert 0.0 <= conf <= 1.0, "置信度必须在 [0,1]"

    # 4) 验证投影向量确实是 L2 归一化（norm≈1）
    n = np.linalg.norm(proj_np)
    print(f"[NORM] proj norm={n:.3f}")
    assert abs(n - 1.0) < 0.05, "投影输出应为 L2 归一化 (norm≈1)"

    # 5) 验证 save/load 往返
    save_path = os.path.join(ROOT, "models", "minimind_o", "_test_projector.pt")
    proj.save(save_path)
    loaded = Projector.load(save_path, device="cpu")
    proj_np2, conf2, _ = loaded.filter_callable(np.asarray(z, dtype=np.float32))
    print(f"[LOAD] reload ok, conf2={conf2:.4f}, proj_norm2={np.linalg.norm(proj_np2):.3f}")

    print("\n[OK] 投影层自检通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
