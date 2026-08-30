# -*- coding: utf-8 -*-
"""
scripts/shutdown.py — 关闭并存档（【硬性】 §十二）
==================================================
职责: 向运行中的昔涟AGI 发送「退出并存档」信号 → 等待 GUI 进程执行
  agi_core.shutdown()（强制存档记忆/情感、释放显存、退出进程）；
  同时对本机 knowledge/memories.json 做一次防御性原子重写（报平安）。

信令机制（跨进程）:
  运行中的 GUI 每 200ms 轮询 knowledge/shutdown.signal 文件；
  shutdown.py 写入该文件（PID + 时间戳），GUI 收到后自动触发
  ChatWindow._on_close() → core.shutdown() → destroy。
"""
import os
import sys
import json
import time

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

import config


def send_signal(cfg) -> str:
    """写入关闭信号文件（GUI 轮询识别）。"""
    signal_path = os.path.join(cfg.knowledge_dir, "shutdown.signal")
    with open(signal_path, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "ts": time.time(), "reason": "shutdown.py"},
                  f, ensure_ascii=False)
    return signal_path


def defensive_save(cfg) -> bool:
    """防御性存档: 对 memories.json 做原子重写（.tmp → replace），保证断电不丢。

    若 GUI 进程仍在运行，它随后会执行真正的 core.shutdown()（含情感状态与
    模型释放），本步仅确保"即使核心崩溃，记忆文件也是最新且格式合法的"。
    """
    if not os.path.isfile(cfg.memories_path):
        return False
    try:
        with open(cfg.memories_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["saved_at"] = time.time()              # 刷新存档时间戳
        data["defensive_backup"] = True
        tmp = cfg.memories_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
        os.replace(tmp, cfg.memories_path)
        return True
    except Exception:
        return False


def main():
    cfg = config.get_config()
    cfg.setup_logging()
    log = cfg.setup_logging().getChild("shutdown")
    log.info("昔涟AGI 关闭请求……")

    # 1) 发送关闭信号
    sig = send_signal(cfg)
    log.info("信号已写入 %s（GUI 将在 1s 内自动退出并存档）", sig)

    # 2) 等待 GUI 完成关闭（最多 15s: 存档 ~秒级 + 显存释放）
    ok = False
    for _ in range(30):
        time.sleep(0.5)
        if not os.path.isfile(cfg.memories_path + ".tmp"):
            ok = True
            break
        # 仍存在 .tmp 说明正在写 → 等待完成
    log.info("等待完成（tmp 已清理）" if ok else "等待超时，执行防御性存档")

    # 3) 防御性存档
    if defensive_save(cfg):
        log.info("记忆池防御性存档完成 → %s", cfg.memories_path)
    else:
        log.info("无需防御性存档（文件不存在或写入失败）")
    log.info("关闭完成。昔涟曾在过去等你。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
