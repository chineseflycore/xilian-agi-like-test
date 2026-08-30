# -*- coding: utf-8 -*-
"""
core/train_manager.py — 训练管理器（Web UI 训练控制面板后端）
=============================================================
内存/显存预估:
  - 曲线历史: loss/temp/lr 环形缓冲（各 500 点）≈ < 1MB
  - 后台训练线程: 复用 diffuser（L3 Hebbian / L2 双缓冲），显存按训练目标
  - 检查点: models/checkpoints/ckpt_{step}.npz（L3 状态，≈400MB/份，按需保留）

功能（规格 "Web UI 训练控制面板"）:
  - 训练曲线（Loss / 温度 / 学习率）历史 + JSON API
  - 实时训练日志（环形缓冲）
  - 手动停止 / 恢复训练（线程事件）
  - 检查点列表 + 从 Web 选择检查点做推理测试

接口:
  TrainManager(cfg) / start(mode="l3") / stop() / pause() / resume() /
  set_temperature(t) / set_lr(lr) / snapshot() / list_checkpoints() /
  test_checkpoint(path, x) / status()
"""

import os
import sys
import time
import threading
from collections import deque

import numpy as np

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

import config
from common_utils import get_logger, safe_import, normalize

logger = get_logger("train_manager")

_CURVE_LEN = 500


class TrainManager:
    """后台训练控制器（Web 面板可停止/恢复/调参/选检查点测试）。"""

    def __init__(self, cfg=None, l3=None, l2=None):
        self.cfg = cfg if cfg is not None else config.get_config()
        self.l3 = l3
        self.l2 = l2
        self._thread = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._paused.set()          # 初始暂停（由 Web 面板 start 激活）
        self.curves = {"loss": deque(maxlen=_CURVE_LEN),
                       "temperature": deque(maxlen=_CURVE_LEN),
                       "lr": deque(maxlen=_CURVE_LEN)}
        self.logs = deque(maxlen=200)
        self.temperature = 0.6
        self.lr = 0.01
        self.step = 0
        self.running = False
        self.ckpt_dir = os.path.join(self.cfg.model_cache_dir, "checkpoints")
        os.makedirs(self.ckpt_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def _log(self, msg: str):
        self.logs.append("[%s] %s" % (time.strftime("%H:%M:%S"), msg))
        logger.info("[TRAINMGR] %s", msg)

    def start(self, mode: str = "l3"):
        """启动训练线程（mode: l3=Hebbian 知识固化 / l2=Transformer 双缓冲）。"""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._paused.set()
        self.running = True
        self._thread = threading.Thread(target=self._worker, args=(mode,),
                                        name="train-mgr", daemon=True)
        self._thread.start()
        self._log("训练已启动（mode=%s, temp=%.2f, lr=%.4f）" % (mode, self.temperature, self.lr))

    def stop(self):
        self._stop.set()
        self._paused.set()
        self.running = False
        self._log("训练已停止")

    def pause(self):
        self._paused.clear()
        self._log("训练已暂停")

    def resume(self):
        self._paused.set()
        self._log("训练已恢复")

    def set_temperature(self, t: float):
        self.temperature = clamp_f(float(t), 0.0, 2.0)

    def set_lr(self, lr: float):
        self.lr = clamp_f(float(lr), 1e-6, 1.0)

    # ------------------------------------------------------------------
    def _worker(self, mode: str):
        """训练循环: 每步记录 loss/温度/lr 曲线；每 50 步存检查点。"""
        step = 0
        while not self._stop.is_set():
            self._paused.wait()               # 暂停时阻塞
            try:
                if mode == "l2" and self.l2 is not None:
                    x = normalize(np.random.RandomState(step).randn(self.cfg.STATE_DIM).astype(np.float32))
                    loss = self.l2.train_step(x, x)
                    self.l2.swap()
                    loss = float(loss) if loss is not None else 0.0
                elif self.l3 is not None:
                    x = normalize(np.random.RandomState(step).randn(self.cfg.STATE_DIM).astype(np.float32))
                    n = self.l3.hebbian_update(x, reward=1.0, lr=self.lr)
                    loss = float(n) / 1e6      # 连接更新量作为 loss 曲线指标
                else:
                    loss = 0.0
                self.step += 1
                step = self.step
                self.curves["loss"].append(float(loss))
                self.curves["temperature"].append(float(self.temperature))
                self.curves["lr"].append(float(self.lr))
                if step % 50 == 0:
                    self._save_ckpt(step)
                    self._log("step=%d loss=%.5f temp=%.2f lr=%.4f" %
                              (step, float(loss), self.temperature, self.lr))
            except Exception as e:
                self._log("训练步异常: %s" % e)
                time.sleep(1.0)
        self.running = False

    def _save_ckpt(self, step: int):
        if self.l3 is None or not hasattr(self.l3, "save_state"):
            return
        try:
            path = os.path.join(self.ckpt_dir, "ckpt_%06d.npz" % step)
            self.l3.save_state(path)
            self._log("检查点已保存: %s" % path)
        except Exception as e:
            self._log("检查点保存失败: %s" % e)

    # ------------------------------------------------------------------
    def list_checkpoints(self) -> list:
        if not os.path.isdir(self.ckpt_dir):
            return []
        out = []
        for f in sorted(os.listdir(self.ckpt_dir)):
            if f.endswith(".npz") and not f.endswith(".meta.npz"):
                p = os.path.join(self.ckpt_dir, f)
                out.append({"path": p, "step": f.split("_")[-1].split(".")[0],
                            "size_mb": round(os.path.getsize(p) / 1048576, 1)})
        return out

    def test_checkpoint(self, path: str, x=None) -> dict:
        """从 Web 选择检查点 → 加载 L3 → 一次扩散推理测试。"""
        if self.l3 is None:
            return {"error": "L3 不可用"}
        try:
            self.l3.load_state(path)
        except Exception as e:
            return {"error": str(e)}
        if x is None:
            x = normalize(np.random.RandomState(42).randn(self.cfg.STATE_DIM).astype(np.float32))
        r = self.l3.run(x, np.zeros(self.cfg.EMOTION_DIM, dtype=np.float32))
        return {"vector_norm": round(float(np.linalg.norm(r["vector"])), 3),
                "confidence": round(float(r["confidence"]), 3),
                "avg_pain": round(float(self.l3.avg_pain), 4)}

    def snapshot(self) -> dict:
        return {
            "running": self.running,
            "paused": not self._paused.is_set(),
            "step": self.step,
            "temperature": self.temperature,
            "lr": self.lr,
            "curves": {k: list(v) for k, v in self.curves.items()},
            "logs": list(self.logs),
            "checkpoints": self.list_checkpoints(),
        }


def clamp_f(v, lo, hi):
    return max(lo, min(hi, v))
