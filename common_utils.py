# -*- coding: utf-8 -*-
"""
common_utils.py — 通用工具（日志 / 计时 / 显存报告 / 安全导入 / 向量运算）
==========================================================================
内存/显存预估: 纯工具函数，无模型权重，占用 < 1MB。
"""
import os
import sys
import time
import hashlib
import logging
from datetime import datetime

# 环境引导: 优先使用全局已安装的 numpy（与 torch 2.0.1 兼容需 numpy 1.x），
# 全局缺失时才回退 vendor/（本项目离线沙箱内置的 numpy 2.x / scipy）。
# 注意: 若无条件把 vendor 插到 sys.path 最前，torch 2.0.1（numpy 1.x 编译）会与
# vendor numpy 2.x 冲突崩溃（"compiled using NumPy 1.x cannot run in NumPy 2.x"）。
_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
try:
    import numpy  # noqa: F401
    _USE_VENDOR = False
except ImportError:
    _USE_VENDOR = True
if _USE_VENDOR and os.path.isdir(_VENDOR) and _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """带时间戳的进程内统一 logger（同时输出到文件与 stdout）。"""
    logger = logging.getLogger(f"philia.{name}")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    try:
        fh = logging.FileHandler(os.path.join(LOG_DIR, f"philia_{datetime.now():%Y%m%d}.log"),
                                 encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    except Exception:
        pass
    return logger


class Timer:
    """轻量计时上下文管理器。"""

    def __init__(self, name: str = "op"):
        self.name = name
        self.elapsed = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = (time.perf_counter() - self._t0) * 1000.0  # ms
        return False


def gpu_mem_mb() -> float:
    """返回 torch.cuda.memory_allocated()（MB）；无 CUDA 时返回 0。"""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def vram_summary() -> str:
    """启动日志用显存摘要。"""
    try:
        import torch
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024 * 1024)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
            return f"VRAM {alloc:.0f}/{total:.0f} MB"
    except Exception:
        pass
    return "VRAM n/a (CPU only)"


def safe_import(name: str, package: str = None):
    """安全导入: 失败返回 None 并记录（供降级路径使用）。"""
    try:
        return __import__(name if package is None else package)
    except Exception as e:
        return None


def normalize(v):
    """L2 归一化（兼容 numpy 与 torch）。"""
    import numpy as np
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.zeros_like(v)


def cosine(a, b) -> float:
    """余弦相似度。"""
    import numpy as np
    a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def stable_hash(*parts) -> str:
    """稳定哈希（训练缓存键，规范 §六.1）。"""
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8", errors="ignore"))
    return h.hexdigest()[:16]


def clamp(v, lo=0.0, hi=1.0) -> float:
    return max(lo, min(hi, v))


def log_startup(module: str, note: str = ""):
    """带时间戳的启动日志（规范 §九.4）。"""
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [BOOT] {module}: {note}")


def pseudo_embedding(text: str, dim: int = 512, seed: int = 42) -> "np.ndarray":
    """DEMO 模式伪嵌入: 字符 n-gram 哈希投影 → dim 维单位向量。
    仅用于流程验证，不具备真实语义（规范 §六.1 降级边界）。
    """
    import numpy as np
    rng = np.random.RandomState(seed)
    vec = np.zeros(dim, dtype=np.float32)
    text = (text or "").strip()
    if not text:
        return vec
    for gram_len in (1, 2, 3):
        for i in range(max(0, len(text) - gram_len + 1)):
            gram = text[i:i + gram_len]
            h = int(hashlib.md5(gram.encode("utf-8", errors="ignore")).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if (h >> 16) & 1 else -1.0
            vec[idx] += sign
    return normalize(vec)


def parse_turns(text: str, max_tokens: int = 512) -> str:
    """裁剪上下文到 ≤max_tokens（验证/批判层精简上下文，规范 §四.1）。"""
    if len(text) <= max_tokens:
        return text
    return text[-max_tokens:]
