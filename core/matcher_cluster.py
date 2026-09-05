# -*- coding: utf-8 -*-
"""
core/matcher_cluster.py — 匹配器集群 v8.0（双存储策略 + 动态激活）[Z_DIM=768]
==================================================================
显存预估:
  大版本（合并）常驻: q uint8 [4000, 106689] ≈ 427MB + scale fp32 [4000] 16KB
  前向分块瞬态: 256 行 × 213377 fp16 ≈ 109MB（分块解量化, 峰值受控）→ 总 < 0.55GB

【硬性】§一.3 双存储策略:
  硬盘存储两种格式:
    1. 大版本（合并）  models/matchers/matchers_combined.pt
                      单个 [4000, total_params] 规格语义: 实现为 NF4 4-bit 打包
                      张量 [4000, ceil(total_params/2)] uint8（每字节 2 个 4-bit 值,
                      与 quant.py 全系统一致）, 整行即一个匹配器 —— 一次性
                      torch.load + 视图切片分配, 零拷贝。
    2. 小版本（独立）  models/matchers/individual/matcher_0000.pt ~ matcher_3999.pt
                      每个 ≈ 0.06MB(115,073 参数 × 4-bit NF4) 约 0.1MB 级
  启动流程:
    ① 检查 individual/ 下是否存在任何 .pt
    ② 若存在, 且 matchers_combined.pt 缺失或任何小文件比大文件新（mtime 比较）
       → 先合并所有小文件生成新大版本（.tmp + os.replace 原子写入）
    ③ 加载 matchers_combined.pt 到显存（一次性, 视图切片分配, 零拷贝）
    首次启动: individual/ 为空 → 从随机初始化生成 4000 个小版本 → 合并大版本。
  更新流程（休眠期微调）:
    ① 修改大张量中对应匹配器的切片权重（内存中替换该行）
    ② 该匹配器权重单独保存到 individual/matcher_XXXX.pt（覆盖, 原子写）
    ③ combined_dirty = True, 不立即重新合并 —— 下次启动时自动合并

【硬性】§二.2 动态激活策略:
  温度分级（每 100 心跳重算）:
    Hot  (>50 次/100心跳): 每心跳参与
    Warm (10~50 次)      : 每 3 心跳参与一次
    Cold (<10 次)        : 每 10 心跳参与一次
  全域激活兜底: Hot+Warm < 2000 → 强制全量运行（冷却 1000 心跳）
  无输入降频: 心跳间隔 2.0s, 仅运行 Hot
  随机回环保活: 4 个随机回环, 每心跳在 Cold 中随机抽取 10~20 个强制激活
"""
import os
import time
import glob
import math
import random
import threading

import numpy as np

import config
from core import quant

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:            # pragma: no cover - DEMO 无 torch 环境
    torch = None
    F = None
    _HAS_TORCH = False


# ----------------------------------------------------------------------
# 层布局: 单匹配器参数为一维向量, 按偏移切分（由 MATCHER_ARCH 计算）
# ----------------------------------------------------------------------
def _compute_layout(arch):
    """由层宽 arch 计算 (name, offset, size) 布局表。

    arch = [768, 256, 64, 1] → w0,b0,w1,b1,w2,b2 依次拼接:
      w0 [256,768] 196608 + b0 [256] 256 + w1 [64,256] 16384 + b1 [64] 64 +
      w2 [1,64] 64 + b2 [1] 1 = 213,377
    """
    layout = []
    off = 0
    for i in range(len(arch) - 1):
        w_size = arch[i + 1] * arch[i]
        b_size = arch[i + 1]
        layout.append((f"w{i}", off, w_size))
        off += w_size
        layout.append((f"b{i}", off, b_size))
        off += b_size
    return layout


class MatcherCluster:
    """4000 匹配器集群: 双存储 / 温度分级 / 全域兜底 / 随机回环保活 / 休眠微调。"""

    def __init__(self, cfg=None, device=None):
        self.cfg = cfg if cfg is not None else config.get_config()
        self.device = device or self.cfg.device
        self.log = self.cfg.setup_logging().getChild("matcher")
        self.lock = threading.RLock()

        self.count = self.cfg.matcher_count                # 4000
        self.arch = list(self.cfg.matcher_arch)            # [768,256,64,1]
        self.numel = self.cfg.matcher_total_params         # 213377
        self.bytes = self.cfg.matcher_bytes                # 106689
        self.layout = _compute_layout(self.arch)
        self.in_dim = self.arch[0]

        # 显存常驻: 合并大张量（NF4 打包 uint8）+ 每行 scales
        self._q = None            # [count, bytes] uint8
        self._scale = None        # [count] float32
        self._on_gpu = False      # 是否加载到 GPU

        # 激活状态
        self._act_counts = np.zeros(self.count, dtype=np.int32)     # 窗口内激活次数
        self._window_beats = 0                                       # 当前窗口心跳数
        self._hot_mask = np.zeros(self.count, dtype=bool)
        self._warm_mask = np.zeros(self.count, dtype=bool)
        self._cold_mask = np.ones(self.count, dtype=bool)
        self._full_burst_beats = 0                                   # 全域激活剩余心跳
        self._rngs = [random.Random(0x1100 + i)
                      for i in range(self.cfg.random_loop_count)]    # 4 个随机回环
        self._activation_scale = 1.0    # 防逃逸: 匹配器激活加权（planner 可下调）

        self.combined_dirty = False     # 更新后标记, 不做立即合并（下次启动合并）
        self._updates_since_merge = 0
        self._last_active_ids = []
        self._last_top = None      # (ids ndarray, scores ndarray) 最近一次激活原始结果

        os.makedirs(self.cfg.matchers_individual_dir, exist_ok=True)
        os.makedirs(self.cfg.matchers_dir, exist_ok=True)
        self._boot_load_or_generate()

    # ==================================================================
    # 路径
    # ==================================================================
    def individual_path(self, idx: int) -> str:
        return os.path.join(self.cfg.matchers_individual_dir,
                            f"matcher_{idx:04d}.pt")

    # ==================================================================
    # 启动流程（【硬性】双存储）
    # ==================================================================
    def _boot_load_or_generate(self):
        """启动: ① 检查 individual ② 合并（若需要）③ 加载大版本到显存。"""
        t0 = time.time()
        ind_files = sorted(glob.glob(os.path.join(self.cfg.matchers_individual_dir,
                                                  "matcher_*.pt")))
        combined = self.cfg.matchers_combined_path

        if not ind_files:
            # ---- 首次启动: individual/ 为空 → 随机初始化生成 4000 个 ----
            self.log.info("[MATCHER] individual/ 为空 → 生成 %d 个默认匹配器（占位模型）",
                          self.count)
            self._generate_all()
            ind_files = sorted(glob.glob(os.path.join(
                self.cfg.matchers_individual_dir, "matcher_*.pt")))

        # ---- ② 合并判定: 大版本缺失 或 任何小文件比大文件新（mtime 比较） ----
        need_merge = not os.path.isfile(combined)
        if not need_merge:
            try:
                c_mtime = os.path.getmtime(combined)
                newest = max(os.path.getmtime(p) for p in ind_files)
                if newest > c_mtime:
                    need_merge = True
            except OSError:
                need_merge = True
        if need_merge:
            self.log.info("[MATCHER] 触发合并: %s",
                          "大版本缺失" if not os.path.isfile(combined)
                          else "存在更新的小文件(mtime)")
            self._merge_individual(combined)

        # ---- ③ 加载大版本到显存（一次性 + 视图切片, 零拷贝） ----
        self._load_combined(combined)
        n = len(glob.glob(os.path.join(self.cfg.matchers_individual_dir,
                                       "matcher_*.pt")))
        self.log.info("[MATCHER] 就绪: count=%d numel=%d bytes=%d 小文件=%d 启动耗时 %.1fs",
                      self.count, self.numel, self.bytes, n, time.time() - t0)

    # ------------------------------------------------------------------
    # 生成 4000 个默认（随机初始化）匹配器
    # ------------------------------------------------------------------
    def _build_default_params(self, idx: int):
        """随机初始化一个匹配器的参数（按层, 1/sqrt(in) 均匀分布）。"""
        rng = np.random.RandomState(0xA11CE + idx)
        params = {}
        for i in range(len(self.arch) - 1):
            in_d, out_d = self.arch[i], self.arch[i + 1]
            bound = 1.0 / math.sqrt(in_d)
            params[f"w{i}"] = (rng.uniform(-bound, bound, (out_d, in_d))
                               .astype(np.float32))
            params[f"b{i}"] = np.zeros(out_d, dtype=np.float32)
        return params

    def _flatten_params(self, params: dict) -> np.ndarray:
        """层参数字典 → 一维 float32 向量（按 LAYOUT 顺序拼接）。"""
        buf = np.empty(self.numel, dtype=np.float32)
        for name, off, size in self.layout:
            buf[off:off + size] = params[name].reshape(-1).astype(np.float32)
        return buf

    def _generate_all(self):
        """生成全部 4000 个小版本文件（原子写, 首次启动一次性）。"""
        t0 = time.time()
        for i in range(self.count):
            flat = self._flatten_params(self._build_default_params(i))
            q, s = quant.quantize_nf4(torch.as_tensor(flat).reshape(1, -1))
            state = {"version": "7.3", "id": i, "arch": list(self.arch),
                     "numel": self.numel, "q": q.reshape(-1).contiguous(),
                     "scale": s.contiguous(),
                     "meta": {"tier": "cold", "created": time.time()}}
            quant.save_state_atomic(state, self.individual_path(i))
            if (i + 1) % 500 == 0:
                self.log.info("[MATCHER] 生成进度 %d/%d (%.0fs)", i + 1, self.count,
                              time.time() - t0)
        self.log.info("[MATCHER] %d 个小版本生成完成 (%.1fs)", self.count,
                      time.time() - t0)

    # ------------------------------------------------------------------
    # 合并: 小文件 → 大版本（原子写入 .tmp + os.replace）
    # ------------------------------------------------------------------
    def _merge_individual(self, combined_path: str, validate_count: int = None):
        """读取全部小文件, 堆叠为 [count, bytes] 大张量, 原子写入。

        约 1~2s（4000 × 57KB 顺序读 + 一次 230MB 连续写）。
        """
        n = validate_count or self.count
        qs, scales, ids = [], [], []
        missing = []
        for i in range(n):
            p = self.individual_path(i)
            if not os.path.isfile(p):
                missing.append(i)
                continue
            st = quant.load_state_atomic(p)
            if not st or st.get("numel") != self.numel:
                missing.append(i)          # 损坏/旧格式 → 待重生成
                continue
            qs.append(st["q"].reshape(-1))
            scales.append(st["scale"].reshape(-1))
            ids.append(st.get("id", i))
        # 缺失的用随机默认补位（永不静默坏档）
        for i in missing:
            self.log.warning("[MATCHER] 小文件缺失/损坏 → 重新生成 #%d", i)
            flat = self._flatten_params(self._build_default_params(i))
            q, s = quant.quantize_nf4(torch.as_tensor(flat).reshape(1, -1))
            qs.append(q.reshape(-1))
            scales.append(s.reshape(-1))
            ids.append(i)
            quant.save_state_atomic(
                {"version": "7.3", "id": i, "arch": list(self.arch),
                 "numel": self.numel, "q": q.reshape(-1).contiguous(),
                 "scale": s.contiguous(), "meta": {}}, self.individual_path(i))
        q = torch.stack(qs).contiguous()          # [4000, bytes]
        scale = torch.stack(scales).reshape(-1).contiguous()
        state = {"version": "7.3", "count": len(q), "numel": self.numel,
                 "arch": list(self.arch), "q": q, "scale": scale,
                 "merged_at": time.time()}
        tmp = combined_path + ".tmp"
        torch.save(state, tmp)                    # 先写 tmp
        os.replace(tmp, combined_path)            # 再原子替换【硬性】
        self.log.info("[MATCHER] 大版本已合并 → %s (%.1f MB)",
                      combined_path, q.numel() / 1e6)

    # ------------------------------------------------------------------
    # 加载大版本到显存（一次性, 视图切片零拷贝）
    # ------------------------------------------------------------------
    def _load_combined(self, combined_path: str):
        t0 = time.time()
        st = quant.load_state_atomic(combined_path)
        if not st or st.get("count") != self.count or st.get("numel") != self.numel:
            # 版本不匹配（如 v6 的 8000×0.1M 大包或旧格式）→ 全量重建
            self.log.warning("[MATCHER] 大版本与 v7.3 规格不匹配 (count=%s numel=%s) "
                             "→ 重建小版本并重新合并",
                             st.get("count") if st else None,
                             st.get("numel") if st else None)
            self._rebuild_all()
            st = quant.load_state_atomic(combined_path)
        self._q = st["q"]                          # [4000, bytes] uint8 —— 行即视图
        self._scale = st["scale"]
        if self.device.startswith("cuda") and _HAS_TORCH:
            self._q = self._q.to(self.device, non_blocking=True)
            self._scale = self._scale.to(self.device, non_blocking=True)
            self._on_gpu = True
            torch.cuda.synchronize()
        self.log.info("[MATCHER] 大版本加载完成 (%.1fs, %s)", time.time() - t0,
                      "GPU" if self._on_gpu else "CPU")

    def _rebuild_all(self):
        """规格不匹配时的全量重建: 清空 individual → 重新生成 → 合并。"""
        for p in glob.glob(os.path.join(self.cfg.matchers_individual_dir,
                                        "matcher_*.pt")):
            try:
                os.remove(p)
            except OSError:
                pass
        self._generate_all()
        self._merge_individual(self.cfg.matchers_combined_path,
                               validate_count=self.count)

    # ==================================================================
    # 温度分级（每 100 心跳重算, 【硬性】§二.2）
    # ==================================================================
    def on_heartbeat(self, idle: bool = False) -> dict:
        """每心跳调用: 推进窗口计数, 重算温度分级, 决定本次激活策略。

        返回策略 dict: {"full": bool, "ids": np.ndarray|None, "mode": str}
        """
        with self.lock:
            self._window_beats += 1
            if self._window_beats >= self.cfg.matcher_tier_refresh:
                self._window_beats = 0
                self._refresh_tiers()
            if self._full_burst_beats > 0:
                self._full_burst_beats -= 1
                return {"full": True, "ids": None, "mode": "full"}
            hot_n = int(self._hot_mask.sum())
            warm_n = int(self._warm_mask.sum())
            if (hot_n + warm_n) < self.cfg.full_activation_threshold \
                    and self._full_burst_beats == 0:
                # 全域激活兜底（【硬性】）: Hot+Warm < 2000 → 全量, 冷却 1000 心跳
                self._full_burst_beats = self.cfg.full_activation_cooldown
                self.log.info("[MATCHER] 全域激活兜底 (hot+warm=%d<%d) → 冷却 %d 心跳",
                              hot_n + warm_n, self.cfg.full_activation_threshold,
                              self.cfg.full_activation_cooldown)
                return {"full": True, "ids": None, "mode": "full"}
            if idle:
                # 无输入降频: 仅 Hot 参与
                return {"full": False, "ids": np.where(self._hot_mask)[0].astype(
                    np.int64), "mode": "idle"}
            beat = self._window_beats
            cadence = self.cfg.matcher_tier_cadence
            ids = []
            if beat % cadence["hot"] == 0:
                ids.append(np.where(self._hot_mask)[0])
            if beat % cadence["warm"] == 0:
                ids.append(np.where(self._warm_mask)[0])
            if beat % cadence["cold"] == 0:
                ids.append(np.where(self._cold_mask)[0])
            # 随机回环保活: 每心跳在 Cold 中随机抽取 10~20 个强制激活
            cold_idx = np.where(self._cold_mask)[0]
            if cold_idx.size:
                k = min(self.cfg.random_activation_max,
                        max(self.cfg.random_activation_min, int(cold_idx.size)))
                rng = self._rngs[beat % len(self._rngs)]
                pick = np.asarray(rng.sample(list(cold_idx.tolist()), k), dtype=np.int64)
                ids.append(pick)
            if ids:
                merged = np.concatenate(ids)
                return {"full": False, "ids": np.unique(merged), "mode": "tier"}
            return {"full": False, "ids": np.empty(0, dtype=np.int64), "mode": "tier"}

    def _refresh_tiers(self):
        """依据窗口内激活次数重算 hot/warm/cold。"""
        hot = self._act_counts > self.cfg.hot_threshold
        warm = (~hot) & (self._act_counts >= self.cfg.warm_threshold)
        cold = ~(hot | warm)
        self._hot_mask, self._warm_mask, self._cold_mask = hot, warm, cold
        self._act_counts[:] = 0                       # 新窗口清零
        self.log.debug("[MATCHER] 温度分级: hot=%d warm=%d cold=%d",
                       int(hot.sum()), int(warm.sum()), int(cold.sum()))

    # ==================================================================
    # 前向（分块解量化, 峰值受控）
    # ==================================================================
    def _forward_block(self, zt, rows_q, rows_scale) -> torch.Tensor:
        """对若干匹配器行做前向: 解量化 → 三层线性链 → score [B]。"""
        w = quant.dequantize_nf4(rows_q, rows_scale, numel=self.numel)   # [B, numel] fp16
        bsize = w.shape[0]
        h = zt.expand(bsize, -1).contiguous()          # [B, 384] fp16
        for name, off, size in self.layout:
            part = w[:, off:off + size]                # [B, size]
            if name.startswith("w"):
                layer = int(name[1:])
                wi = part.reshape(bsize, self.arch[layer + 1], self.arch[layer])
                h = torch.einsum("bki,bi->bk", wi, h)          # [B, out]
            else:
                h = h + part                                     # bias [B, out]
                if int(name[1:]) < len(self.arch) - 2:
                    h = F.relu(h)
        return h.squeeze(-1)                           # [B] 原始分数（无 sigmoid）

    def _dequant_block(self, ids: np.ndarray):
        """取大张量中 ids 对应行（视图切片, 零拷贝）→ (q块, scale块)。"""
        ids_t = torch.as_tensor(ids, dtype=torch.long, device=self._q.device)
        return self._q[ids_t], self._scale[ids_t]

    def activate(self, z, strategy=None, force_full: bool = False) -> dict:
        """激活一批匹配器, 返回 {scores, active_ids, full, mode}。

        z: [384] float32/float16 向量（CPU ndarray 或 GPU tensor）
        strategy: on_heartbeat() 的返回; force_full 用于全域兜底/手动全量
        """
        if strategy is None:
            strategy = {"full": force_full, "ids": None,
                        "mode": "force" if force_full else "manual"}
        with self.lock:
            if not _HAS_TORCH:
                return self._fake_activate(z, strategy)
            if force_full or strategy.get("full"):
                ids = np.arange(self.count, dtype=np.int64)
                mode = "full"
            else:
                ids = np.asarray(strategy.get("ids", []), dtype=np.int64)
                mode = strategy.get("mode", "tier")
            if ids.size == 0:
                self._last_active_ids = []
                self._last_top = None
                return {"scores": None, "active_ids": [], "full": False, "mode": mode}

            zt = torch.as_tensor(np.asarray(z, dtype=np.float32).reshape(1, -1),
                                 dtype=torch.float16, device=self._q.device)
            scores = torch.empty(ids.shape[0], dtype=torch.float16,
                                 device=self._q.device)
            chunk = 256                      # 分块: 单块 fp16 瞬态 ≈ 59MB
            with torch.no_grad():
                for off in range(0, ids.shape[0], chunk):
                    sub = ids[off:off + chunk]
                    qb, sb = self._dequant_block(sub)
                    scores[off:off + chunk] = self._forward_block(zt, qb, sb)
            scores = scores.float().cpu().numpy()
            scores = np.clip(scores * self._activation_scale, -8.0, 8.0)

            # 激活计数（供温度分级）
            self._act_counts[ids] += 1
            self._last_active_ids = ids.tolist()
            self._last_top = (ids, scores)
            return {"scores": scores, "active_ids": ids.tolist(),
                    "full": mode == "full", "mode": mode}

    def _fake_activate(self, z, strategy):    # pragma: no cover - 无 torch 降级
        """无 torch 降级: 给假分数, 保证流程不断链。"""
        ids = (np.arange(self.count) if strategy.get("full")
               else np.asarray(strategy.get("ids") or [0], dtype=np.int64))
        rng = random.Random()
        scores = (np.array([0.35 + 0.3 * rng.random() for _ in range(ids.size)],
                           dtype=np.float32) * self._activation_scale)
        self._act_counts[ids] += 1
        self._last_active_ids = ids.tolist()
        return {"scores": scores, "active_ids": ids.tolist(),
                "full": bool(strategy.get("full")), "mode": strategy.get("mode", "fake")}

    # ==================================================================
    # top-k 查询（供情感摘要 / 回复选择 / 思维链渲染）
    # ==================================================================
    def top_matchers(self, k: int = 5, positive: bool = True) -> list:
        """最近一次激活中分数最高的 k 个匹配器 [(idx, score), ...]。"""
        if self._last_top is None or self._last_top[0].size == 0:
            return []
        ids, scores = self._last_top
        # stable + 取反: 与旧 sorted(items, key=..., reverse=positive) 的并列顺序完全一致
        order = np.argsort(-scores if positive else scores, kind="stable")
        order = order[:k]
        return [(int(ids[i]), float(scores[i])) for i in order]

    def summary_vector(self, k: int = 8) -> np.ndarray:
        """匹配器摘要（k 维软分布）: 供情感回环输入的 16 维拼接部分。"""
        top = self.top_matchers(k=k)
        if not top:
            return np.full(k, 1.0 / k, dtype=np.float32)
        vals = np.clip([v for _, v in top] + [0.0] * (k - len(top)), 0.0, 1.0)[:k]
        total = float(vals.sum()) + 1e-8
        return (vals / total).astype(np.float32)

    # ==================================================================
    # 休眠期微调（【硬性】§一.3 更新流程）
    # ==================================================================
    def fine_tune(self, idx: int, samples, steps: int = None) -> dict:
        """微调单个匹配器: 优化三层 MLP 使 (z→score) 逼近样本目标。

        samples: [(z [384], target float)]; 更新流程（硬性）:
          ① 修改大张量中对应匹配器切片权重（内存替换该行）
          ② 单独保存到 individual/matcher_XXXX.pt（原子覆盖）
          ③ combined_dirty = True（不立即合并, 下次启动自动合并）
        """
        if not _HAS_TORCH or not samples:
            return None
        steps = steps or self.cfg.fine_tune_steps
        with self.lock:
            q_row = self._q[idx].cpu()
            s_row = self._scale[idx].cpu().reshape(1)
            row = quant.dequantize_nf4(q_row.unsqueeze(0), s_row,
                                       numel=self.numel)[0].float()
            params = {}
            for name, off, size in self.layout:
                part = row[off:off + size].float()
                if name.startswith("w"):
                    layer = int(name[1:])
                    params[name] = torch.nn.Parameter(
                        part.reshape(self.arch[layer + 1], self.arch[layer]).clone())
                else:
                    params[name] = torch.nn.Parameter(
                        part.reshape(self.arch[int(name[1:]) + 1]).clone())
            z0 = torch.as_tensor(np.asarray(samples[0][0], dtype=np.float32))
            before = self._score_params(params, z0.reshape(1, -1)).item()

            zt = torch.stack([torch.as_tensor(np.asarray(z, dtype=np.float32))
                              for z, _ in samples])                    # [S, 384]
            targets = torch.tensor([float(t) for _, t in samples],
                                   dtype=torch.float32).unsqueeze(1)
            opt = torch.optim.Adam(list(params.values()), lr=3e-3)
            for _ in range(steps):
                opt.zero_grad()
                h = zt
                for name in self._layer_names():
                    if name.startswith("w"):
                        h = F.linear(h, params[name])
                    else:
                        h = h + params[name]
                        if int(name[1:]) < len(self.arch) - 2:
                            h = F.relu(h)
                loss = F.mse_loss(torch.sigmoid(h), targets)
                loss.backward()
                opt.step()
            after = self._score_params(params, z0.reshape(1, -1)).item()

            # ① 修改大张量对应切片（内存）
            flat = self._flatten_torch_params(params)
            nq, ns = quant.quantize_nf4(flat.reshape(1, -1))
            self._q[idx] = nq.reshape(-1).to(self._q.device)
            self._scale[idx] = ns.reshape(-1).to(self._scale.device)
            # ② 独立小文件原子覆盖
            quant.save_state_atomic(
                {"version": "7.3", "id": idx, "arch": list(self.arch),
                 "numel": self.numel, "q": nq.reshape(-1).cpu().contiguous(),
                 "scale": ns.reshape(-1).cpu().contiguous(),
                 "meta": {"fine_tuned": time.time()}}, self.individual_path(idx))
            # ③ 标记 dirty（【硬性】: 不立即合并）
            self.combined_dirty = True
            self._updates_since_merge += 1
        return {"idx": idx, "before": before, "after": after,
                "samples": len(samples), "steps": steps}

    def _layer_names(self):
        return [name for name, _, _ in self.layout]

    def _score_params(self, params, zt) -> torch.Tensor:
        """按当前参数打分（无梯度）。"""
        h = zt
        with torch.no_grad():
            for name in self._layer_names():
                if name.startswith("w"):
                    h = F.linear(h, params[name])
                else:
                    h = h + params[name]
                    if int(name[1:]) < len(self.arch) - 2:
                        h = F.relu(h)
        return h

    def _flatten_torch_params(self, params) -> torch.Tensor:
        """层参数字典（torch Parameter）→ 一维 float32（按 LAYOUT 顺序, 微调路径）。"""
        buf = torch.empty(self.numel, dtype=torch.float32)
        for name, off, size in self.layout:
            buf[off:off + size] = params[name].reshape(-1).float()
        return buf

    # ==================================================================
    # 防逃逸: 匹配器激活加权（任务规划层告警时调用）
    # ==================================================================
    def adjust_activation_scale(self, factor: float) -> float:
        """调整激活加权（默认 1.0; <1 抑制, >1 增强）。

        任务规划层发现"走云端比例 > 60%"时调用, 抑制云端倾向响应。
        """
        with self.lock:
            self._activation_scale *= float(factor)
            self._activation_scale = float(np.clip(self._activation_scale, 0.2, 3.0))
        self.log.warning("[MATCHER] 激活加权调整 → %.3f (防逃逸)", self._activation_scale)
        return self._activation_scale

    # ==================================================================
    # 状态 / 统计
    # ==================================================================
    def get_stats(self) -> dict:
        """调试窗口 / 状态快照用。"""
        return {
            "active_last": len(self._last_active_ids),
            "hot": int(self._hot_mask.sum()),
            "warm": int(self._warm_mask.sum()),
            "cold": int(self._cold_mask.sum()),
            "full_burst": self._full_burst_beats > 0,
            "full_burst_beats": self._full_burst_beats,
            "combined_dirty": self.combined_dirty,
            "updates": self._updates_since_merge,
            "activation_scale": self._activation_scale,
            "vram_mb": (self._q.numel() / 1e6) if self._q is not None else 0.0,
        }
