# -*- coding: utf-8 -*-
"""
first_run.py — 首次预启动配置向导（检查环境/模型/API Key，输出修复指引）
=========================================================================
内存/显存预估: 纯检查脚本，无模型权重，< 1MB。

由 first_run.bat 调用；也可直接:  Python\python.exe first_run.py
检查项:
  1. 便携版 Python 3.11.9
  2. torch（CUDA 可用性）
  3. transformers / modelscope
  4. Qwen3.5-0.8B 模型是否已下载（models\Qwen3.5-0.8B\config.json）
  5. hoyotool.ini [Cloud].api_key 是否配置（训练数据生成用，可后配）
  6. （可选）faiss-cpu / silero-vad
退出码: 0 = 全部就绪可启动；1 = 存在缺项（已打印修复指引）
"""

import os
import sys

_BASE = os.path.dirname(os.path.abspath(__file__))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)
try:
    import numpy  # noqa: F401
except ImportError:
    _V = os.path.join(_BASE, "vendor")
    if os.path.isdir(_V) and _V not in sys.path:
        sys.path.insert(0, _V)

import configparser

PY = os.path.join(_BASE, "Python", "python.exe")
INI = os.path.join(_BASE, "hoyotool.ini")
MODEL_DIR = os.path.join(_BASE, "models", "Qwen3.5-0.8B")
MODEL_CFG = os.path.join(MODEL_DIR, "config.json")


def check(ok: bool, name: str, hint: str = ""):
    mark = "[OK] " if ok else "[缺] "
    print(("  " + mark + name).ljust(56) + (("  ← " + hint) if hint and not ok else ""))
    return ok


def main() -> int:
    print()
    print("=" * 62)
    print("  昔涟人格引擎 - 首次预启动配置向导")
    print("=" * 62)
    all_ok = True

    # 1) 便携版 Python
    all_ok &= check(os.path.isfile(PY), "便携版 Python 3.11.9",
                    "缺失 Python\\ 目录（embeddable 版解压到项目下）")

    # 2) torch + CUDA
    torch_ok = False
    try:
        import torch
        torch_ok = torch.cuda.is_available()
        if torch_ok:
            print(f"  [OK] torch {torch.__version__} | CUDA 可用: "
                  f"{torch.cuda.get_device_name(0)}")
        else:
            print(f"  [缺] torch {torch.__version__} 但 CUDA 不可用，"
                  f"请装 CUDA 版: pip install torch==2.6.0+cu126 "
                  f"--index-url https://download.pytorch.org/whl/cu126")
    except Exception:
        print("  [缺] torch 未安装")
        print("       修复: Python\\python.exe -m pip install torch==2.6.0+cu126 "
              "--index-url https://download.pytorch.org/whl/cu126")
    all_ok &= torch_ok

    # 3) transformers / modelscope
    try:
        import transformers  # noqa: F401
        print(f"  [OK] transformers {transformers.__version__}")
    except Exception:
        all_ok &= check(False, "transformers", "Python\\python.exe -m pip install transformers")
    try:
        import modelscope  # noqa: F401
        print("  [OK] modelscope")
    except Exception:
        all_ok &= check(False, "modelscope",
                        "Python\\python.exe -m pip install modelscope")

    # 4) 模型
    model_ok = os.path.isfile(MODEL_CFG)
    if model_ok:
        size = sum(os.path.getsize(os.path.join(dp, f))
                   for dp, _, fs in os.walk(MODEL_DIR) for f in fs) / 1048576
        print(f"  [OK] Qwen3.5-0.8B 模型已下载（{size:.0f} MB）")
    else:
        all_ok &= check(False, "Qwen3.5-0.8B 模型",
                        "Python\\python.exe -c \"from modelscope import snapshot_download; "
                        "snapshot_download('Qwen/Qwen3.5-0.8B', local_dir=r'%s')\"" % MODEL_DIR)

    # 5) API Key（hoyotool.ini [Cloud].api_key；可后配，不影响启动）
    api_ok = False
    try:
        cp = configparser.ConfigParser()
        cp.read(INI, encoding="utf-8")
        key = cp.get("Cloud", "api_key", fallback="").strip()
        api_ok = bool(key)
    except Exception:
        key = ""
    if api_ok:
        print("  [OK] API Key 已配置（hoyotool.ini [Cloud]）")
    else:
        print("  [提示] API Key 未配置（hoyotool.ini [Cloud].api_key）—— "
              "训练数据生成将走降级；不影响引擎启动/对话")
        print("        配置: 编辑 hoyotool.ini 填入 DeepSeek Key（platform.deepseek.com 获取）")

    # 6) 可选依赖
    for pkg, why in (("faiss", "FAISS 记忆索引加速（无则 numpy 降级）"),
                     ("silero_vad", "VAD 语音检测（无则听觉返回零向量）")):
        try:
            __import__(pkg)
            print(f"  [OK] {pkg}（{why}）")
        except Exception:
            print(f"  [可选] {pkg} 未装（{why}，可后补）")

    print()
    if all_ok and api_ok:
        print("  ✅ 全部就绪！运行 start.bat 启动引擎。")
        return 0
    if all_ok:
        print("  ✅ 运行就绪（API Key 可后补）。运行 start.bat 启动引擎。")
        return 0
    print("  ⚠️  存在缺项，请按上方 [缺]/[提示] 指引修复后重跑本向导。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
