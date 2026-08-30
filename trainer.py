# -*- coding: utf-8 -*-
"""
trainer.py — 异步训练器（规范 §八.19 / §十.15，CPU 后台双缓冲）
================================================================
内存/显存预估（规范 §九.1）:
  - 任务队列: 训练样本缓冲（numpy 数组，批大小 ≤ 32，< 1MB）
  - L2 CPU 副本: 由 diffuser_l2 持有（≈4MB）；本模块仅持有队列与线程
  - 显存: 0MB —— 训练强制 CPU 后台执行，不占用 GPU 显存（规范 §二）

职责:
  1. 后台守护线程消费训练任务（L2 用 CPU 副本训练 + L3 Hebbian 更新）
  2. 双缓冲: 训练只动 CPU 副本，完成后 l2.swap() 原子替换 GPU 主模型（规范 §十.15）
  3. 训练循环禁止将 L3 csr_matrix 稠密转换（规范 §十.20，交由 L3 自身保证）
  4. DEMO（无 torch / 无 l2/l3）: 线程为空转并打日志，流程可验证（规范 §六.1）

接口:
  submit_l2(x, target) / submit_hebbian(vec, reward) / start() / stop() / stats()
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

import threading
import queue
import time

import numpy as np

import config
from common_utils import get_logger


class AsyncTrainer:
    """异步训练器: L2 CPU 副本训练 + L3 Hebbian 更新，原子替换。"""

    def __init__(self, cfg=None, l2=None, l3=None):
        self.cfg = cfg if cfg is not None else config.get_config()
        self.log = get_logger("trainer")
        self.l2 = l2                    # L2 Transformer 扩散器（双缓冲）
        self.l3 = l3                    # L3 神经元特化网络（Hebbian）
        self._queue = queue.Queue(maxsize=256)
        self._thread = None
        self._stop = threading.Event()
        self._stats = {"l2_steps": 0, "hebbian_updates": 0, "swaps": 0, "dropped": 0}

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self):
        """启动后台训练线程（守护线程，进程退出自动终止）。"""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._worker, name="philia-trainer", daemon=True)
        self._thread.start()
        self.log.info("[TRAIN] 异步训练线程已启动（CPU 后台）")

    def stop(self, timeout: float = 3.0):
        """请求停止并等待线程退出。"""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self.log.info("[TRAIN] 训练线程已停止（处理 %d 个任务）",
                      self._stats["l2_steps"] + self._stats["hebbian_updates"])

    # ------------------------------------------------------------------
    # 任务提交
    # ------------------------------------------------------------------
    def submit_l2(self, x, target, lr: float = 0.01):
        """提交一条 L2 训练样本（CPU 副本训练，规范 §十.15 双缓冲）。"""
        try:
            self._queue.put_nowait(("l2", (x, target, lr)))
        except queue.Full:
            self._stats["dropped"] += 1
            self.log.warning("[TRAIN] 训练队列已满，丢弃 1 条 L2 样本")

    def submit_hebbian(self, vec, reward: float = 1.0):
        """提交一条 L3 Hebbian 更新（无监督，规范 §4.2）。"""
        try:
            self._queue.put_nowait(("hebbian", (vec, reward)))
        except queue.Full:
            self._stats["dropped"] += 1
            self.log.warning("[TRAIN] 训练队列已满，丢弃 1 条 Hebbian 更新")

    # ------------------------------------------------------------------
    # 后台工作线程
    # ------------------------------------------------------------------
    def _worker(self):
        """消费队列: l2 → CPU 副本 train_step + swap 原子替换；hebbian → l3 更新。"""
        while not self._stop.is_set():
            try:
                kind, payload = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if kind == "l2":
                    self._do_l2(payload)
                elif kind == "hebbian":
                    self._do_hebbian(payload)
            except Exception as e:
                self.log.warning("[TRAIN] 任务执行失败: %s", e)
            finally:
                self._queue.task_done()

    def _do_l2(self, payload):
        """L2 双缓冲训练: CPU 副本 train_step → swap() 原子替换（规范 §十.15）。"""
        if self.l2 is None:
            return
        x, target, lr = payload
        x = np.asarray(x, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        loss = self._call_l2_train(self.l2, x, target, lr)
        self._stats["l2_steps"] += 1
        # 原子替换（改进 ≥ cfg.l2_swap_threshold 才替换）
        swapped = False
        swap_fn = getattr(self.l2, "swap", None)
        if callable(swap_fn):
            try:
                r = swap_fn()
                if isinstance(r, dict):
                    swapped = bool(r.get("swapped", False))
                else:
                    swapped = bool(r)
            except Exception as e:
                self.log.warning("[TRAIN] swap 失败: %s", e)
        if swapped:
            self._stats["swaps"] += 1
        self.log.info("[TRAIN] L2 step loss=%.4f swapped=%s", float(loss or 0.0), swapped)

    def _do_hebbian(self, payload):
        """L3 Hebbian 更新（规范 §4.2，禁止稠密转换由 L3 内部保证）。"""
        if self.l3 is None:
            return
        vec, reward = payload
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        ok = False
        fn = getattr(self.l3, "hebbian_update", None)
        if callable(fn):
            try:
                fn(vec, reward=reward)
                ok = True
            except TypeError:
                try:
                    fn(vec, reward)
                    ok = True
                except Exception as e:
                    self.log.warning("[TRAIN] hebbian_update 调用失败: %s", e)
        if ok:
            self._stats["hebbian_updates"] += 1

    @staticmethod
    def _call_l2_train(l2, x, target, lr):
        """多形态调用 L2.train_step: train_step(x, target, lr) 或 train_step(x, target)。"""
        fn = getattr(l2, "train_step", None)
        if not callable(fn):
            return None
        try:
            return fn(x, target, lr=lr)
        except TypeError:
            try:
                return fn(x, target)
            except TypeError:
                return fn(x)
        except Exception as e:
            return e

    # ------------------------------------------------------------------
    # 统计
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return dict(self._stats, queue_size=self._queue.qsize(),
                    alive=bool(self._thread is not None and self._thread.is_alive()))


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # DEMO 自检: 无 l2/l3 时线程空转不崩溃（规范 §六.1 流程验证）
    _cfg = config.get_config()
    _t = AsyncTrainer(_cfg, l2=None, l3=None)
    _t.start()
    _t.submit_l2(np.zeros(512, np.float32), np.zeros(512, np.float32))
    _t.submit_hebbian(np.zeros(512, np.float32), reward=1.0)
    time.sleep(0.5)
    _t.stop()
    print("[SELFTEST] trainer stats →", _t.stats())
