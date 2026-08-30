# -*- coding: utf-8 -*-
"""
selftest_real.py — 真实模型自检脚本（在装有 torch + transformers 的机器上运行）
==============================================================================
用途: 在您的 GTX 1060 6GB（或任意带 torch 的机器）上验证"真实代码路径"——
真实加载 Qwen3.5-0.8B（联网核实: https://huggingface.co/Qwen/Qwen3.5-0.8B），
并跑通: 编码 → 身份锚点 → 生成 → 路由 logits → （可选）视觉编码。

运行前提:
  1. pip install torch==2.0.1+cu116 --index-url https://download.pytorch.org/whl/cu116
     （GTX 1060 最高支持 CUDA 11.6，规范 §二；现代 GPU 可装新版并设
       PHILIA_SKIP_CUDA_CHECK=1 跳过自检）
  2. pip install transformers>=4.40.0 numpy scipy
  3. 首次运行会从 HuggingFace 下载模型（约 3.2GB），请确保网络可用

运行方式:
  python selftest_real.py                # 真实加载并自检
  PHILIA_MODEL=Qwen/Qwen3.5-0.8B-Base python selftest_real.py   # 换模型
  显存不足时: 设 PHILIA_DEMO=1 走降级链；或改 config 的 use_amp/use_4bit

自检项:
  1. 加载 Qwen3.5-0.8B（FP32 / AMP / 4bit 按配置）
  2. encode(): 文本 → 512 维（mean pooling + 投影）
  3. 身份锚点 encode_anchor()（规范 §4.8）
  4. decode(): 真实生成（chat 模板，昔涟人设）
  5. next_token_probs(): 路由 logits softmax（规范 §4.1）
  6. encode_image(): Qwen3.5-0.8B-V 视觉编码（CPU 惰性加载，规范 §3.3；可跳过）
"""

# 环境引导: 优先使用已安装的 numpy，缺失时回退本项目 vendor/（离线沙箱）
try:
    import numpy  # noqa: F401
except ImportError:
    import os as _os
    import sys as _sys
    _V = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "vendor")
    if _os.path.isdir(_V) and _V not in _sys.path:
        _sys.path.insert(0, _V)

import os
import sys

# 让脚本目录可导入（嵌入式 Python ._pth 隔离模式下必需；常规 CPython 为 no-op）
_THIS = os.path.dirname(os.path.abspath(__file__))
if _THIS not in sys.path:
    sys.path.insert(0, _THIS)

import numpy as np

from config import get_config
from common_utils import get_logger, vram_summary, Timer

logger = get_logger("selftest_real")


def section(name: str):
    print("\n" + "=" * 64)
    print(f"  {name}")
    print("=" * 64)


def main() -> int:
    cfg = get_config()
    cfg.ensure_dirs()
    if cfg.demo:
        logger.warning("当前为 DEMO 模式（未检测到 torch/transformers 或 PHILIA_DEMO=1）。"
                       "请先在带 torch 的环境运行本脚本。")
        return 2

    torch = __import__("torch")
    print(f"torch {torch.__version__} | cuda_avail={torch.cuda.is_available()} "
          f"| {vram_summary()}")

    # ---- 1) 加载真实模型 ----
    section("1) 加载 Qwen3.5-0.8B（真实模型）")
    from translator import Translator
    with Timer("load_model") as t:
        tr = Translator(cfg)
    if tr.is_demo:
        logger.error("模型加载失败，已降级 DEMO —— 请检查网络/transformers 版本/显存。")
        return 1
    print(f"加载耗时 {t.elapsed/1000:.1f}s | device={tr.device} | hidden={tr.hidden_dim} "
          f"| {vram_summary()}")

    # ---- 2) encode ----
    section("2) encode(): 文本 → 512 维")
    texts = ["花海很安静。", "你会害怕死亡吗？", "三千万世的轮回，我都记得。"]
    vecs = tr.encode(texts)
    print(f"shape={vecs.shape} dtype={vecs.dtype}")
    for i, v in enumerate(vecs):
        print(f"  [{texts[i][:10]}] norm={float(np.linalg.norm(v)):.3f}")
    assert vecs.shape == (3, 512)

    # ---- 3) 身份锚点 ----
    section("3) 身份锚点 encode_anchor()（规范 §4.8）")
    anchor = tr.encode_anchor()
    print(f"anchor dim={anchor.shape[0]} norm={float(np.linalg.norm(anchor)):.3f}")

    # ---- 4) 真实生成 ----
    section("4) decode(): 真实生成（昔涟人设）")
    from cloud_model_client import SEED_EXAMPLES
    system_hint = cfg.anchor_generation_prompt
    prompt = (f"{system_hint}\n\n用户：你好，昔涟。你还记得花海吗？\n昔涟：")
    with Timer("generate") as tg:
        reply = tr.decode(prompt)
    print(f"生成耗时 {tg.elapsed/1000:.1f}s")
    print(f"回复: {reply}")

    # ---- 5) 路由 logits ----
    section("5) next_token_probs(): 真实路由 softmax（规范 §4.1）")
    from router import Router
    r = Router(cfg, translator=tr)
    state = tr.encode(["用户消息"])[0]
    emotion = np.zeros(8, dtype=np.float32)
    emotion[3] = 1.0                    # neutral
    route = r.route(state, emotion, "聊聊翁法罗斯的轮回")
    print(f"route: line={route['line_used']} confidence={route['confidence']:.3f} "
          f"labels={route['world_labels']}")

    # ---- 6) 视觉编码（可选，Qwen3.5-0.8B-V 需另行下载）----
    section("6) encode_image(): Qwen3.5-0.8B-V 视觉编码（可选）")
    try:
        from PIL import Image
        img = Image.new("RGB", (64, 64), (120, 160, 200))
        vv = tr.encode_image(img)
        print(f"visual vec norm={float(np.linalg.norm(vv)):.3f} "
              f"(非零={bool(np.linalg.norm(vv) > 1e-6)})")
    except Exception as e:
        print(f"跳过视觉自检（需要 Qwen3.5-0.8B-V 模型与 Pillow）: {e!r}")

    # ---- 汇总 ----
    section("结果汇总")
    print(f"模型: {cfg.qwen_model_name}")
    print(f"设备: {tr.device} | {vram_summary()}")
    print(f"身份锚点已写入 cfg.anchor_vector: {cfg.anchor_vector is not None}")
    print("✓ 真实模型路径自检完成 —— 可运行 python main.py 启动完整人格引擎。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
