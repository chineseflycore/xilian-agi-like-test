# -*- coding: utf-8 -*-
"""
core/router.py — 路由模型 v7.3（5 类认知决策）
================================================
显存预估: ~0.03GB（53.6M ≈ 50M 参数 × 4-bit NF4 ≈ 27MB 常驻; 逐层解量化瞬态 < 15MB）
本模块不加载 Qwen; 对"下一步动作"做 5 类决策。

架构: 4 层 MLP [392→8192→6144→5]（53,601,589 ≈ 50M 参数）。
  输入(392) = Z(384) ‖ 情感向量(8)     —— v7.3 直接使用编码器潜向量
  输出(5)   = softmax: 0直接/1扩散/2情感调制/3记忆检索/4回环

存储: models/router/router_nf4.pt（单模型, NF4 打包; 缺失或 arch 不符时自动生成占位）。
"""
import os
import time
import threading

import numpy as np
import torch
import torch.nn.functional as F

import config
from core import quant

try:
    _HAS_TORCH = True
except Exception:            # pragma: no cover
    _HAS_TORCH = False


class Router:
    """路由模型: 决策前向（Z+情感 → 5 类动作概率）→ 状态查询。"""

    def __init__(self, cfg=None, device=None):
        self.cfg = cfg if cfg is not None else config.get_config()
        self.device = device or self.cfg.device
        self.log = self.cfg.setup_logging().getChild("router")
        self.lock = threading.RLock()
        self.arch = list(self.cfg.router_arch)            # [392, 8192, 6144, 5]
        self.path = self.cfg.router_model_file
        self._stack = {}                                  # layer → (q, scale, numel)
        self._on_gpu = False
        self._last_decision = {"decision": 0, "name": "direct", "probs": []}
        self._prepare_dir()
        self._load_or_generate()

    def _prepare_dir(self):
        os.makedirs(self.cfg.router_dir, exist_ok=True)

    # ==================================================================
    # 存储（NF4 4-bit 常驻）
    # ==================================================================
    def _load_or_generate(self):
        t0 = time.time()
        st = None
        if os.path.isfile(self.path):
            try:
                st = quant.load_state_atomic(self.path)
            except Exception:
                st = None
        if st is None or st.get("arch") != self.arch or "q" not in st:
            self.log.info("[ROUTER] 生成随机占位路由 (%s)…", self.arch)
            st = self._build_default()
            self._save(st)
        self._stack = {}
        for layer, q in st["q"].items():
            self._stack[layer] = (q.reshape(-1), st["scale"][layer].reshape(-1),
                                  int(st["numel"][layer]))
            if self.device.startswith("cuda"):
                self._stack[layer] = (self._stack[layer][0].to(self.device),
                                      self._stack[layer][1].to(self.device),
                                      self._stack[layer][2])
        self._on_gpu = self.device.startswith("cuda")
        total = st["numel"].get("total", 0)
        self.log.info("[ROUTER] 就绪 (%.1fs, %s, %.1fM 参数)", time.time() - t0,
                      "GPU" if self._on_gpu else "CPU", total / 1e6)

    def _build_default(self) -> dict:
        """随机初始化全层（1/sqrt(in) 均匀）并 NF4 打包。"""
        rng = np.random.RandomState(self.cfg.router_seed)
        st = {"version": "7.3", "arch": list(self.arch),
              "q": {}, "scale": {}, "numel": {}}
        total = 0
        for i in range(len(self.arch) - 1):
            in_d, out_d = self.arch[i], self.arch[i + 1]
            bound = 1.0 / np.sqrt(in_d)
            w = torch.as_tensor(rng.uniform(-bound, bound, (out_d, in_d)),
                                dtype=torch.float32)
            b = torch.zeros(out_d, dtype=torch.float32)
            for tag, t in (("w", w), ("b", b)):
                q, s = quant.quantize_nf4(t.reshape(1, -1))
                st["q"][f"w{i}" if tag == "w" else f"b{i}"] = q.reshape(-1)
                st["scale"][f"w{i}" if tag == "w" else f"b{i}"] = s
                st["numel"][f"w{i}" if tag == "w" else f"b{i}"] = int(t.numel())
                total += int(t.numel())
        st["numel"]["total"] = total
        return st

    def _save(self, st: dict):
        tmp = self.path + ".tmp"
        torch.save(st, tmp)
        os.replace(tmp, self.path)

    # ==================================================================
    # 前向（逐层解量化, 瞬态 < 15MB）
    # ==================================================================
    def decide(self, z, emotion) -> dict:
        """(Z [384], 情感 [8]) → 决策 dict {decision, name, probs}。"""
        if not _HAS_TORCH:
            return self._last_decision
        feats = np.concatenate([np.asarray(z, dtype=np.float32).reshape(-1),
                                np.asarray(emotion, dtype=np.float32).reshape(-1)])
        x = torch.as_tensor(feats, dtype=torch.float16,
                            device=self.device).unsqueeze(0)
        with self.lock, torch.no_grad():
            for i in range(len(self.arch) - 1):
                wq, ws, wn = self._stack[f"w{i}"]
                w = quant.dequantize_nf4(wq.unsqueeze(0), ws.reshape(-1),
                                         numel=wn).reshape(
                    self.arch[i + 1], self.arch[i])
                bq, bs, bn = self._stack[f"b{i}"]
                b = quant.dequantize_nf4(bq.unsqueeze(0), bs.reshape(-1),
                                         numel=bn).reshape(-1)
                x = F.linear(x, w.half()) + b.half()
                if i < len(self.arch) - 2:
                    x = F.relu(x)
        probs = torch.softmax(x, dim=-1).float().cpu().numpy().reshape(-1)
        dec = int(np.argmax(probs))
        self._last_decision = {
            "decision": dec,
            "name": self.cfg.router_decisions[dec],
            "probs": probs.tolist(),
        }
        return self._last_decision

    def record_code(self, code: int):
        """记录输入协议码（历史缓冲, 供日志/规划层使用）。"""
        with self.lock:
            self._last_code = int(code) & 0xFFFF

    def get_last(self) -> dict:
        return dict(self._last_decision)
