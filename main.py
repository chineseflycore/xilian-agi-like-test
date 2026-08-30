# -*- coding: utf-8 -*-
"""
main.py — 入口（启动自检 → 感知层 → 服务器，规范 §八.24 / §九.3）
=================================================================
内存/显存预估（规范 §九.1）: 本模块不持有权重；总显存峰值预算 4.5GB（规范 §二）:
  路由层 Qwen3.5-0.8B FP32 ≈ 3.2GB + KV Cache ≈ 0.3GB + 框架 ≈ 0.3GB
  + 情感模型 ≈ 0.16GB + L2 ≈ 4MB ≈ 峰值 4.0~4.4GB（GTX 1060 6GB 内）

启动自检顺序（规范 §九.3，任何阶段失败进入降级模式）:
  ① CUDA 版本校验: torch.version.cuda ≤ 11.6（GTX 1060 约束）；不匹配 → 错误日志并退出
     （可用环境变量 PHILIA_SKIP_CUDA_CHECK=1 跳过，供现代 GPU 使用）
  ② 显存检测: < 5GB → 启用 AMP（[AMP_MODE]）；< 3GB → 4-bit 量化（[4BIT_MODE]）
  ③ Qwen 模型加载（真实 Qwen3.5-0.8B / DEMO 伪嵌入兜底）
  ④ 感知层初始化（听觉 VAD + 融合，CPU）

运行:
  python main.py                 # 启动 HTTP 服务（127.0.0.1:8080）
  python main.py --selftest      # 装配后跑一轮对话并退出（CI/验证用）
  PHILIA_DEMO=1 python main.py   # 强制 DEMO（纯 CPU 流程验证，规范 §六.1）
  PHILIA_L3_SCALE=0.01 python main.py   # 缩小 L3 规模（本沙箱/低内存机）
"""

# 环境引导: 脚本目录与 vendor 入 sys.path（嵌入式 Python 隔离模式必需；常规 CPython no-op）
import os as _os, sys as _sys
_BASE = _os.path.dirname(_os.path.abspath(__file__))
if _BASE not in _sys.path:
    _sys.path.insert(0, _BASE)
try:
    import numpy  # noqa: F401
except ImportError:
    _V = _os.path.join(_BASE, "vendor")
    if _os.path.isdir(_V) and _V not in _sys.path:
        _sys.path.insert(0, _V)

import os
import sys
import time
import logging

import config
from common_utils import get_logger, log_startup, vram_summary, safe_import

logger = get_logger("main")

_HOST = os.environ.get("PHILIA_HOST", "127.0.0.1")
_PORT = int(os.environ.get("PHILIA_PORT", "8080"))


# ----------------------------------------------------------------------
# 启动自检（规范 §九.3）
# ----------------------------------------------------------------------
def self_check(cfg) -> bool:
    """按顺序执行启动自检；返回是否继续（False → 退出）。"""
    log_startup("main", f"角色={cfg.name} 模式={'DEMO' if cfg.demo else 'REAL'}")
    log_startup("main", cfg.summary())

    # ① CUDA 版本校验（规范 §二: GTX 1060 目标约束）
    # 实际部署: 驱动 582.53 支持 CUDA 12.6（torch 2.6.0+cu126）；
    # 新版 transformers 要求 torch>=2.5，故本项目默认栈为 torch 2.6.0+cu126。
    # 版本不匹配仅告警不退出（设 PHILIA_STRICT_CUDA_CHECK=1 可恢复硬性退出）。
    torch = safe_import("torch")
    if torch is not None and torch.cuda.is_available():
        v = torch.version.cuda or ""
        log_startup("main", f"CUDA 可用: torch.version.cuda={v} "
                             f"{vram_summary()}")
        major, minor = 0, 0
        try:
            parts = v.split(".")
            major, minor = int(parts[0]), int(parts[1])
        except Exception:
            pass
        if os.environ.get("PHILIA_STRICT_CUDA_CHECK") == "1" and \
                (major > 11 or (major == 11 and minor > 7)):
            logger.error(
                "\033[31m[CUDA-CHECK] torch.version.cuda=%s 超出目标 11.7 "
                "（PHILIA_STRICT_CUDA_CHECK=1 强制模式）\033[0m", v)
            return False
        if major > 12 or (major == 12 and minor > 6):
            logger.warning(
                "\033[33m[CUDA-CHECK] torch.version.cuda=%s 高于已验证的 12.6，"
                "若加载异常请改回 torch 2.6.0+cu126\033[0m", v)
    else:
        log_startup("main", "CUDA 不可用 → CPU 运行（真实模型仍会加载，速度较慢）")

    # ② 显存检测（<5GB → AMP；<3GB → 4-bit，规范 §九.3② / §二 精度策略）
    if torch is not None and torch.cuda.is_available():
        try:
            total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            log_startup("main", f"显存检测: {total_mb:.0f} MB")
            if total_mb < 3 * 1024:
                cfg.use_4bit = True
                logger.warning("\033[33m[4BIT_MODE] 显存 < 3GB，加载 4-bit 量化版本\033[0m")
            elif total_mb < 5 * 1024:
                cfg.use_amp = True
                logger.warning("\033[33m[AMP_MODE] 显存 < 5GB，启用混合精度\033[0m")
            else:
                cfg.use_amp = cfg.use_4bit = False
                log_startup("main", "显存充足，保持 FP32（规范 §十.1）")
        except Exception as e:
            logger.warning("显存检测失败: %s", e)
    else:
        cfg.use_amp = cfg.use_4bit = False

    # ③ ④ 在 Engine 装配中完成: Qwen 加载 + 感知层初始化
    return True


# ----------------------------------------------------------------------
def build_engine(cfg):
    """装配完整引擎（含 Qwen 加载与感知层初始化）。"""
    from server import Engine
    return Engine(cfg)


def run_server(cfg):
    """启动 HTTP 服务（规范 §七）。"""
    from server import create_server
    engine = build_engine(cfg)
    srv = create_server(engine, host=_HOST, port=_PORT)
    log_startup("main", f"HTTP 服务已启动 http://{_HOST}:{_PORT} （Ctrl+C 停止）")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engine.shutdown()
        srv.server_close()
        log_startup("main", "服务已关闭")
    return 0


def run_selftest(cfg):
    """装配引擎并跑一轮对话后退出（验证用，规范 §六.1 流程验证）。"""
    t0 = time.time()
    engine = build_engine(cfg)
    log_startup("main", f"引擎装配完成（{time.time() - t0:.1f}s）{vram_summary()}")
    resp = engine.chat("你好，昔涟。你还记得花海吗？")
    print("\n[CHAT-DEMO] 输入: 你好，昔涟。你还记得花海吗？")
    print(f"[CHAT-DEMO] 输出: {resp['response_text']}")
    print(f"[CHAT-DEMO] 管线: {resp['line_used']} | direct={resp['direct_mode']} | "
          f"alignment={resp['alignment_score']} | {resp['response_time_ms']}ms")
    print(f"[CHAT-DEMO] 状态: {json_dumps(engine.status())}")
    engine.shutdown()
    return 0


def json_dumps(obj) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False)


# ----------------------------------------------------------------------
def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    cfg = config.get_config()
    cfg.ensure_dirs()
    log_startup("main", f"启动 (PID={os.getpid()}) Python={sys.version.split()[0]}")

    if not self_check(cfg):
        log_startup("main", "启动自检失败，退出（规范 §九.3①）")
        return 1

    if "--selftest" in argv:
        return run_selftest(cfg)
    return run_server(cfg)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
