# -*- coding: utf-8 -*-
"""
core/working_memory.py — 工作记忆（前额叶缓存, v6 阶段一）
==========================================================
内存预估: < 100MB（轮次变长时以 z 序列与文本为主; 显存 0）。

职责（与长期记忆池互补, 借鉴人类工作记忆模型）:
  - 本会话所有轮次完整保留（不限容量, 元规则）—— 长任务（代码/长文）连续处理
  - 会话内容: 用户文本 + 协议码 + Z 潜向量 + 昔涟回复（含时间戳）
  - 新会话启动: 从长期记忆池拉 Top-30 预加载到背景区（背景是"静态参考", 不进轮次流）
  - 运行中每 N 心跳自动补充背景区（长期池 → 背景）
  - 解码注入: prompt_context() 按 token 上限截断头部（保尾部近因）
  - 持久化: knowledge/working_memory.json（原子写; 存档轮次上限可配, 内存不限）
  - 微调数据源: 休眠期匹配器/预测器更新一律取本模块（不用长期池直接训练）
"""
import json
import os
import time
import threading

import numpy as np

import config

try:
    import torch
except ImportError:
    torch = None


class WorkMemory:
    """工作记忆: 轮次流（当前会话）+ 背景区（长期池预加载参考）。"""

    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else config.get_config()
        self.log = config.get_config().setup_logging().getChild("workmem")
        self.lock = threading.RLock()
        self.path = self.cfg.working_memory_path
        self.turns = []            # [{role, text, protocol_code, z(list), reply, ts}]
        self.background = []       # [{content, protocol_code, emotion_tag}]（Top-30 预加载）
        # 【去显存】轮次 Z 的显存镜像（(N,256) fp32, ~5000轮×2.5MB; 无 CUDA 降级 None）
        self._device = self.cfg.device if self.cfg.device.startswith("cuda") else None
        # 【性能】预分配环形缓冲: 覆写式写入, 消除每轮 torch.cat 全量复制
        self._z_t = None          # (cap, z_dim) fp32 常驻; None = 无 CUDA 或降级
        self._z_cap = 2048        # 环形容量（超出覆盖最旧; ≥ 训练序列 11 与 prompt 需求）
        self._z_head = 0          # 下一写入位
        self._z_count = 0         # 有效行数（≤ cap）
        self.load()

    # ------------------------------------------------------------------
    # 读写
    # ------------------------------------------------------------------
    def add_turn(self, user_text: str, protocol_code: int, z: np.ndarray,
                 reply_text: str = ""):
        """追加一轮完整工作记忆（用户输入 + 协议码 + Z + 昔涟回复）。Z 同步镜像到显存。"""
        zv = np.asarray(z, dtype=np.float32).reshape(-1)
        with self.lock:
            self.turns.append({
                "role": "user",
                "text": str(user_text)[:2000],
                "protocol_code": int(protocol_code) & 0xFFFF,
                "z": [round(float(x), 5) for x in zv],
                "reply": str(reply_text)[:2000],
                "ts": time.time(),
            })
            if self._device is not None:
                try:
                    import torch
                    zv = zv[: self.cfg.z_dim]
                    if zv.size < self.cfg.z_dim:
                        zv = np.pad(zv, (0, self.cfg.z_dim - zv.size))
                    if self._z_t is None:
                        self._z_t = torch.zeros((self._z_cap, self.cfg.z_dim),
                                                device=self._device)
                    self._z_t[self._z_head] = torch.as_tensor(zv, device=self._device)
                    self._z_head = (self._z_head + 1) % self._z_cap
                    self._z_count = min(self._z_count + 1, self._z_cap)
                except Exception:
                    self._z_t = None
                    self._z_head = 0
                    self._z_count = 0

    def load_background(self, memories: list):
        """会话启动: 用长期记忆 Top-30 填充背景区（预加载静态参考）。"""
        with self.lock:
            self.background = []
            for m in memories:
                self.background.append({
                    "content": str(m.get("content", ""))[:300],
                    "protocol_code": int(m.get("protocol_code", 0)),
                    "emotion_tag": m.get("emotion_tag", "calm"),
                })
            self.log.info("[WORKMEM] 背景区预加载 %d 条（长期池 Top-%d）",
                          len(self.background), self.cfg.retrieval_top_k)

    def inject_background(self, memories: list, cap: int = None):
        """运行中补充背景区（每 N 心跳, 取长期池样本去重追加, 上限计数）。"""
        cap = cap or self.cfg.retrieval_top_k
        with self.lock:
            seen = {b["protocol_code"] for b in self.background}
            for m in memories:
                code = int(m.get("protocol_code", 0))
                if code in seen:
                    continue
                seen.add(code)
                self.background.append({
                    "content": str(m.get("content", ""))[:300],
                    "protocol_code": code,
                    "emotion_tag": m.get("emotion_tag", "calm"),
                })
            if len(self.background) > cap:
                self.background = self.background[-cap:]

    # ------------------------------------------------------------------
    # 查询（解码注入 / 微调数据源 / 监控）
    # ------------------------------------------------------------------
    def prompt_context(self, max_tokens: int = None) -> str:
        """解码器注入上下文: 背景区 + 近期轮次, 超 token 上限截断头部（保近因）。

        v6.1: 长度上限/背景条数/轮次数全部走 config（提升上下文容量, 由调用方权衡生成速度）。
        """
        max_tokens = max_tokens or min(self.cfg.working_memory_token_limit,
                                       self.cfg.working_memory_prompt_chars)
        with self.lock:
            parts = [f"（背景）{b['content']}"
                     for b in self.background[: self.cfg.working_memory_prompt_bg]]
            parts += [f"（{t['role']}）{t['text']}"
                      for t in self.turns[-self.cfg.working_memory_prompt_turns:]]
            if self.turns and self.turns[-1].get("reply"):
                parts.append(f"（昔涟上次说）{self.turns[-1]['reply'][:160]}")
            text = "\n".join(parts)
            if len(text) > max_tokens:
                text = "…（前文截断）…" + text[-max_tokens:]
            return text

    def z_sequence(self, n: int = 12):
        """最近 n 轮 Z 向量序列（预测器训练/推理数据源; 显存镜像优先, 无则 numpy）。

        返回: torch.Tensor (m, z_dim)（有显存镜像时在 GPU 上; 否则 CPU 张量）。
        """
        with self.lock:
            if torch is not None and self._z_t is not None and self._z_count > 0:
                nn = min(n, self._z_count)
                idx = (self._z_head - nn + np.arange(nn)) % self._z_cap
                return self._z_t[idx].clone()
            zs = [np.asarray(t.get("z", []), dtype=np.float32)
                  for t in self.turns if t.get("z")]
        if not zs:
            if torch is None:
                return np.zeros((0, self.cfg.z_dim), dtype=np.float32)
            return torch.zeros((0, self.cfg.z_dim), dtype=torch.float32)
        seq = np.stack([z.reshape(-1)[: self.cfg.z_dim]
                        if z.size >= self.cfg.z_dim
                        else np.pad(z.reshape(-1), (0, self.cfg.z_dim - z.size))[: self.cfg.z_dim]
                        for z in zs])
        if torch is None:
            return seq[-n:]
        return torch.as_tensor(seq[-n:], dtype=torch.float32)

    def codes_recent(self, n: int = 12) -> list:
        """最近 n 轮的协议码列表（匹配器微调正样本来源）。

        注意: 合法协议码可能为 0（语义#0 问候）—— 过滤必须用 is not None, 不可用真值判断。
        """
        with self.lock:
            return [int(t["protocol_code"]) for t in self.turns[-n:]
                    if t.get("protocol_code") is not None]

    def length(self) -> int:
        with self.lock:
            return len(self.turns)

    def est_tokens(self) -> int:
        """估算工作记忆 token 占用（监控显示用: 近似按字符/2）。"""
        with self.lock:
            chars = sum(len(t.get("text", "")) + len(t.get("reply", "")) for t in self.turns)
            return int(chars / 2)

    def stats(self) -> dict:
        with self.lock:
            return {
                "turns": len(self.turns),
                "tokens_est": self.est_tokens(),
                "background": len(self.background),
                "z_used": sum(1 for t in self.turns if t.get("z")),
            }

    # ------------------------------------------------------------------
    # 持久化（原子写; 幂等）
    # ------------------------------------------------------------------
    def save(self):
        """持久化工作记忆（原子 .tmp → replace; 轮次数上限存档, 内存不删）。"""
        with self.lock:
            limit = self.cfg.working_memory_persist_turns
            data = {
                "version": "6.0",
                "saved_at": time.time(),
                "turns": self.turns[-limit:],
                "background": self.background[: self.cfg.retrieval_top_k],
            }
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
            os.replace(tmp, self.path)
            self.log.info("[WORKMEM] 已存档 %d 轮 / 背景 %d 条", len(data["turns"]),
                          len(data["background"]))
        except Exception as e:
            self.log.warning("[WORKMEM] 存档失败: %s", e)

    def load(self):
        """启动恢复（缺失/损坏 → 空工作记忆, 不崩溃）; 重建显存 Z 镜像。"""
        if not os.path.isfile(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            with self.lock:
                self.turns = [t for t in data.get("turns", []) if isinstance(t, dict)]
                self.background = [b for b in data.get("background", [])
                                   if isinstance(b, dict)]
                # 【去显存】重建 GPU 镜像（旧 z 按当前 cfg.z_dim 截断/补零, 兼容跨版本存档）
                if self._device is not None:
                    try:
                        import torch
                        rows = []
                        for t in self.turns:
                            z = np.asarray(t.get("z", []), dtype=np.float32)
                            if z.size == 0:
                                continue
                            if z.size >= self.cfg.z_dim:
                                rows.append(z[: self.cfg.z_dim])
                            else:
                                rows.append(np.pad(z, (0, self.cfg.z_dim - z.size)))
                        if rows:
                            rows = rows[-self._z_cap:]
                            arr = torch.as_tensor(np.stack(rows), device=self._device)
                            self._z_t = torch.zeros((self._z_cap, self.cfg.z_dim),
                                                    device=self._device)
                            self._z_t[: arr.shape[0]] = arr
                            self._z_head = arr.shape[0] % self._z_cap
                            self._z_count = arr.shape[0]
                        else:
                            self._z_t = None
                            self._z_head = 0
                            self._z_count = 0
                    except Exception:
                        self._z_t = None
                        self._z_head = 0
                        self._z_count = 0
            self.log.info("[WORKMEM] 恢复 %d 轮 / 背景 %d 条 (z_镜像=%s)",
                          len(self.turns), len(self.background),
                          "显存" if self._z_t is not None else "CPU降级")
        except Exception as e:
            self.log.warning("[WORKMEM] 恢复失败(%s), 空工作记忆启动", e)
