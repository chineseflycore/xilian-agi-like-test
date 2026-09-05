# -*- coding: utf-8 -*-
"""
test_projector_agiconnect.py — 验证投影层在 AGICore 里的接入
运行: python scripts/test_projector_agiconnect.py
设 XILIAN_ENCODER=rule 强制规则编码（不加载 MiniMind），更贴近「接规则路径 Z」。
"""
import os
import sys

THIS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(THIS)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 强制规则编码：不加载 MiniMind（与「接规则路径最稳」一致）
os.environ["XILIAN_ENCODER"] = "rule"

import numpy as np
import torch


def main() -> int:
    import config
    cfg = config.get_config()

    print(f"[ENV] encoder={cfg.encoding_backend} z_dim={cfg.z_dim} device={cfg.device}")
    print(f"[ENV] projector_conf={cfg.projector_conf}")

    # 构造 AGICore（会装配组件；用轻量构造避免触发 MiniMind）
    from core.agi_core import AGICore
    core = AGICore(cfg, api_mode=False)

    # 确保投影层存在
    proj = core._ensure_projector()
    print(f"[ENSURE] projector={proj is not None} "
          f"in={proj.in_dim if proj else '-'} out={proj.out_dim if proj else '-'}")

    # 生成规则路径 Z（模拟真实输入）
    from core import protocol as P
    raw = np.zeros(16, dtype=np.float32)
    raw[0] = 1.0
    z0 = P.vectors_to_z(raw, cfg)
    print(f"[Z0] dim={z0.shape[0]} norm={np.linalg.norm(z0):.3f}")

    # 调 _project_z
    proj_z, conf, in_dist = core._project_z(z0, apply_conf=True)
    print(f"[PROJECT] out_dim={np.asarray(proj_z).shape[-1]} "
          f"out_norm={np.linalg.norm(proj_z):.3f} conf={conf:.4f} in_dist={in_dist}")

    # 断言
    assert np.asarray(proj_z).shape[-1] == cfg.z_dim, "投影后维度应为 z_dim"
    assert 0.0 <= conf <= 1.0, "置信度必须在 [0,1]"

    # 再测 apply_conf=False
    proj_z2 = core._project_z(z0, apply_conf=False)
    print(f"[PROJECT2] (no conf) out_dim={np.asarray(proj_z2).shape[-1]}")

    # 测 batch: 多行 Z
    zbatch = np.vstack([z0, z0])
    pz, c, d = core._project_z(zbatch, apply_conf=True)
    print(f"[BATCH] out shape={np.asarray(pz).shape} conf={c:.4f} in_dist={d}")

    print("\n[OK] AGICore 投影层接入验证通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
