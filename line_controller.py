# -*- coding: utf-8 -*-
"""
line_controller.py — 三线切换控制器（规范 §八.10、§十.9/18）
================================================================
内存/显存预估（规范 §九.1）:
  · 本模块不持有任何模型权重，仅保存少量 512 维 numpy 向量（float32）。
  · 常驻占用 < 1MB RAM；无 GPU 显存占用。
  · L3 融合路径临时分配 2~3 个 512 维向量（约 6KB），随函数返回即释放。

职责:
  1. 运行时管线切换 L1/L2/L3，同步更新 cfg.current_line（规范 §十.9，对应 POST /switch）
  2. L1（纯数据/概率扩散）/ L2（Transformer 扩散）模式直接调用对应扩散器
  3. L3 模式下执行 FAISS-L3 加权融合：FAISS 检索记忆向量权重 0.3 + L3 扩散输出
     权重 0.7，融合后归一化（规范 §十.18）
  4. 冲突解决"332"原则（规范 §4.7.5）：多模态两个一致则优先 —— 当 FAISS 记忆向量
     与 L3 输出方向余弦 > 0.85 时视为"一致"，信任融合结果并抬高置信度；
     不一致时仍按 0.3/0.7 权重折中（权重本身即代表"多数侧优先"的折中立场）

降级策略（关键路径 try-except，规范 §九.2）:
  · diffuser / memory 为 None 或接口缺失时（如独立 DEMO 运行），回退为确定性伪输出
    并打印日志，保证纯 CPU 流程可验证（规范 §六.1 降级路径）。
  · torch 不可用时全程使用 numpy，无需 torch。

TODO(V3.4): 融合权重 0.3/0.7 提升为 hoyotool.ini 可配置项；增加融合置信度在线校准。
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

import numpy as np

from config import get_config
from common_utils import get_logger, normalize, cosine, clamp, Timer

_LOG = get_logger("line_controller")

_VALID_LINES = ("L1", "L2", "L3")

# FAISS 与 L3 的加权融合权重（规范 §十.18: FAISS 0.3 / L3 0.7）
_FAISS_WEIGHT = 0.3
_L3_WEIGHT = 0.7

# "332"原则一致阈值：两个模态（记忆 vs 扩散）方向余弦超过该值视为"一致"（规范 §4.7.5）
_AGREE_THRESHOLD = 0.85


class LineController:
    """三线切换控制器：把状态向量分发到当前管线，并负责 L3 的 FAISS 加权融合。"""

    def __init__(self, cfg, l1=None, l2=None, l3=None, memory=None):
        self.cfg = cfg if cfg is not None else get_config()
        self.l1 = l1          # L1 纯数据/概率扩散器（CPU，0.5M，无训练）
        self.l2 = l2          # L2 Transformer 扩散器（GPU，双缓冲训练）
        self.l3 = l3          # L3 神经元特化网络（CPU，1M 神经元）
        self.memory = memory  # 记忆系统（FAISS 短期记忆 + 随机召回层）
        # 初始化时校验 cfg.current_line 合法性，非法则回落默认管线
        line = str(getattr(self.cfg, "current_line", None) or "").strip().upper()
        if line not in _VALID_LINES:
            self.cfg.current_line = str(getattr(self.cfg, "default_mode", "L3")).upper()
            if self.cfg.current_line not in _VALID_LINES:
                self.cfg.current_line = "L3"

    # ------------------------------------------------------------------
    # 管线切换（规范 §十.9）
    # ------------------------------------------------------------------
    def switch(self, line: str) -> dict:
        """校验并切换当前管线，同步 cfg.current_line。
        支持 "L1"/"L2"/"L3"；非法值抛出 ValueError（由 server 层转 400）。
        """
        line = str(line or "").strip().upper()
        if line not in _VALID_LINES:
            raise ValueError("无效管线: %r，仅支持 %s" % (line, "/".join(_VALID_LINES)))
        self.cfg.current_line = line
        _LOG.info("[SWITCH] 管线切换 → %s（运行时生效，下一轮 run 起用）", line)
        return {"line": line, "ok": True}

    # ------------------------------------------------------------------
    # 主入口：run(state_vector, emotion_vec, text_history) -> dict
    # ------------------------------------------------------------------
    def run(self, state_vector, emotion_vec, text_history: str) -> dict:
        """按 cfg.current_line 运行当前管线，返回融合后的 512 维向量与元信息。

        参数:
          state_vector: 融合层输出的 512 维状态向量（numpy float32）
          emotion_vec : 8 维情绪向量 [angry,sad,happy,neutral,fear,intensity,arousal,valence]
          text_history: 历史对话文本（字符串）

        返回:
          {"vector": y(512,), "line_used": str, "confidence": float, "memory_hits": [...]}
        """
        line = str(getattr(self.cfg, "current_line", "L3") or "L3").strip().upper()
        if line not in _VALID_LINES:
            line = "L3"  # 兜底：任何非法状态回落 L3
        state_vector = np.asarray(state_vector, dtype=np.float32)

        with Timer("line_%s" % line) as t:
            if line == "L1":
                y, conf = self._call_diffuser(self.l1, state_vector, emotion_vec, text_history)
                hits = []
            elif line == "L2":
                y, conf = self._call_diffuser(self.l2, state_vector, emotion_vec, text_history)
                hits = []
            else:  # L3：FAISS-L3 加权融合
                y, conf, hits = self._run_l3_fusion(state_vector, emotion_vec, text_history)

        # 统一规整为 512 维单位向量（保证输出契约）
        y = self._to_dim(y, state_vector)
        if y is None:
            _LOG.warning("[LINE] 输出向量非法，回退伪输出")
            y = self._fallback_output(state_vector)
        y = normalize(y)

        _LOG.info("[LINE:%s] conf=%.3f memory_hits=%d 耗时=%.1fms",
                  line, conf, len(hits), t.elapsed)
        return {
            "vector": y,
            "line_used": line,
            "confidence": clamp(float(conf), 0.0, 1.0),
            "memory_hits": hits,
        }

    # ------------------------------------------------------------------
    # L3：FAISS-L3 加权融合（规范 §十.18）+ "332"冲突解决（规范 §4.7.5）
    # ------------------------------------------------------------------
    def _run_l3_fusion(self, state_vector, emotion_vec, text_history):
        """融合流程:
          1. FAISS 检索 top-k 相似记忆向量 → 按相似度加权平均 → faiss_vec
          2. L3 扩散器输出 → l3_vec（含置信度）
          3. 加权融合 y = 0.3*faiss_vec + 0.7*l3_vec，再 L2 归一化
          4. "332"原则: 当 cosine(faiss_vec, l3_vec) > 0.85（两模态方向一致）时,
             信任融合结果（抬升置信度）；不一致时按权重折中（0.3/0.7 本身即折中）。
        """
        faiss_vec, hits = self._faiss_retrieve(state_vector, k=3)
        l3_out, l3_conf = self._call_diffuser(self.l3, state_vector, emotion_vec, text_history)
        l3_vec = normalize(l3_out)

        if faiss_vec is None:
            # FAISS 不可用/无命中 → 权重临时退化为 0/1（纯 L3），注释说明降级
            _LOG.info("[FUSION] FAISS 无检索结果，退化为纯 L3")
            return l3_vec, l3_conf, []

        faiss_vec = normalize(faiss_vec)
        # 模态间一致性（记忆 vs 扩散的方向夹角）
        agree = cosine(faiss_vec, l3_vec)
        if agree > _AGREE_THRESHOLD:
            _LOG.info("[FUSION] 记忆与L3方向一致(cos=%.3f>%.2f) → 信任融合（332原则）",
                      agree, _AGREE_THRESHOLD)

        # 加权融合（FAISS 0.3 / L3 0.7）后归一化（规范 §十.18）
        fused = normalize(_FAISS_WEIGHT * faiss_vec + _L3_WEIGHT * l3_vec)

        # 置信度: 0.3*记忆平均相似度 + 0.7*L3 置信度
        faiss_sim = float(np.mean([h.get("similarity", 0.0) for h in hits])) if hits else 0.0
        conf = clamp(_FAISS_WEIGHT * faiss_sim + _L3_WEIGHT * float(l3_conf), 0.0, 1.0)
        if agree > _AGREE_THRESHOLD:
            # 两个一致则优先：抬升信任度（规范 §4.7.5 冲突解决）
            conf = max(conf, 0.85)
        return fused, conf, hits

    # ------------------------------------------------------------------
    # FAISS 检索（多接口兼容 + 降级）
    # ------------------------------------------------------------------
    def _faiss_retrieve(self, query_vec, k: int = 3):
        """从记忆系统检索相似向量。
        返回 (融合后的记忆向量或 None, memory_hits 列表)。
        兼容 memory.search/query/retrieve/recall 等多种接口，均失败返回 (None, [])。
        """
        if self.memory is None:
            return None, []
        query = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        for mname in ("search", "query", "retrieve", "recall"):
            fn = getattr(self.memory, mname, None)
            if fn is None:
                continue
            try:
                res = fn(query, k=k)
                vecs, ids, sims = self._parse_memory_result(res, k)
                if vecs is None or len(vecs) == 0:
                    continue
                sims = np.asarray([s if s is not None else 0.0 for s in sims], dtype=np.float32)
                sims = np.clip(sims, 0.0, 1.0)
                if float(sims.sum()) < 1e-9:
                    sims = np.ones_like(sims) / max(len(sims), 1)  # 无相似度信息 → 均权
                # 相似度加权平均得到"代表性记忆向量"
                fused = normalize(
                    np.sum(np.asarray(vecs, dtype=np.float32) * sims[:, None], axis=0))
                hits = [{"id": ids[i] if i < len(ids) else i, "similarity": float(sims[i])}
                        for i in range(len(vecs))]
                return fused, hits
            except Exception as e:
                _LOG.warning("[FAISS] %s 检索失败: %s", mname, e)
        return None, []

    @staticmethod
    def _parse_memory_result(res, k):
        """把各种 memory 接口返回值规整为 (vectors, ids, sims)。
        支持 dict / tuple / list / 纯 ndarray；距离自动换算为相似度 sim=1/(1+d)。
        """
        vecs, ids, sims = None, [], []
        if isinstance(res, dict):
            # 注意: 不能用 `a or b` 对 ndarray 做真值判断（numpy 会抛
            # "truth value of an array is ambiguous"），必须显式 None 判断
            vecs = res.get("vectors")
            if vecs is None:
                vecs = res.get("vecs")
            if vecs is None:
                vecs = res.get("embeddings")
            ids = res.get("ids")
            if ids is None:
                ids = res.get("indices")
            ids = list(ids) if ids is not None else []
            if "similarities" in res:
                sims = res.get("similarities")
            elif "sims" in res:
                sims = res.get("sims")
            elif "scores" in res:
                sims = res.get("scores")
            elif "distances" in res:
                d = list(res.get("distances") or res.get("dists") or [])
                sims = [1.0 / (1.0 + float(x)) for x in d]
            elif "dists" in res:
                d = list(res.get("dists") or [])
                sims = [1.0 / (1.0 + float(x)) for x in d]
            sims = list(sims) if sims is not None else []
        elif isinstance(res, (tuple, list)) and len(res) >= 1:
            vecs = res[0]
            if len(res) >= 2 and res[1] is not None:
                second = res[1]
                # 第二元素: 整型 → ids；浮点 → 距离（再换算相似度）
                try:
                    arr = np.asarray(second)
                    if arr.size and np.issubdtype(arr.dtype, np.integer):
                        ids = list(arr)
                    elif arr.size:
                        sims = [1.0 / (1.0 + float(x)) for x in arr]
                except Exception:
                    pass
            if len(res) >= 3 and res[2] is not None:
                try:
                    d = np.asarray(res[2])
                    if d.size:
                        sims = [1.0 / (1.0 + float(x)) for x in d]
                except Exception:
                    pass
        elif isinstance(res, np.ndarray):
            vecs = res
        if vecs is None:
            return None, ids, sims
        vecs = np.asarray(vecs, dtype=np.float32)
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        vecs = vecs[:k]
        ids = (ids[:k] if ids else list(range(len(vecs))))
        sims = (sims[:k] if sims else [None] * len(vecs))
        return vecs, ids, sims

    # ------------------------------------------------------------------
    # 扩散器调用（多接口兼容 + 降级）
    # ------------------------------------------------------------------
    def _call_diffuser(self, diffuser, state_vector, emotion_vec, text_history):
        """调用扩散器，返回 (输出向量, 置信度)。接口缺失/异常 → 伪输出兜底。"""
        if diffuser is None:
            _LOG.warning("[DIFFUSER] 扩散器为 None，回退伪输出")
            return self._fallback_output(state_vector), 0.40
        attempts = (
            ("run", (state_vector, emotion_vec, text_history)),
            ("diffuse", (state_vector, emotion_vec)),
            ("forward", (state_vector,)),
            ("__call__", (state_vector, emotion_vec)),
        )
        for name, args in attempts:
            fn = getattr(diffuser, name, None)
            if fn is None:
                continue
            try:
                out = fn(*args)
                vec, conf = self._coerce_output(out, diffuser, state_vector)
                if vec is not None:
                    return vec, conf
            except Exception as e:
                _LOG.warning("[DIFFUSER] %s 调用失败: %s", name, e)
        _LOG.warning("[DIFFUSER] 接口均不可用，回退伪输出")
        return self._fallback_output(state_vector), 0.40

    @staticmethod
    def _coerce_output(out, diffuser, state_vector):
        """把扩散器输出规整为 (向量, 置信度)；tuple/dict/纯向量均可。"""
        conf = None
        if isinstance(out, tuple):
            vec = out[0]
            if len(out) >= 2 and isinstance(out[1], (int, float)):
                conf = float(out[1])
        elif isinstance(out, dict):
            # 注意: 不能用 `a or b` 对 ndarray 做真值判断（numpy 抛
            # "truth value of an array is ambiguous"），必须显式 None 判断
            vec = out.get("vector")
            if vec is None:
                vec = out.get("y")
            c = out.get("confidence")
            if isinstance(c, (int, float)):
                conf = float(c)
        else:
            vec = out
        vec = LineController._to_dim(vec, state_vector)
        if vec is None:
            return None, 0.0
        if conf is None:
            d = getattr(diffuser, "confidence", None)
            conf = float(d) if isinstance(d, (int, float)) else 0.55
        return vec, conf

    @staticmethod
    def _to_dim(v, ref_vec, dim=None):
        """把向量规整为 float32 一维数组；长度不足补零、超出截断、非法返回 None。"""
        if v is None:
            return None
        dim = dim if dim is not None else int(ref_vec.size if hasattr(ref_vec, "size") else len(ref_vec))
        try:
            a = np.asarray(v, dtype=np.float32).reshape(-1)
        except Exception:
            return None
        if a.size == 0:
            return None
        if a.size < dim:
            a = np.concatenate([a, np.zeros(dim - a.size, dtype=np.float32)])
        return a[:dim]

    def _fallback_output(self, state_vector):
        """确定性伪扩散输出：状态向量 + 依赖 text_history 的固定噪声（DEMO 流程验证）。"""
        seed = int(np.abs(hash(str(state_vector.tobytes()))) % (2 ** 31))
        rng = np.random.RandomState(seed)
        noise = rng.randn(len(state_vector)).astype(np.float32)
        return normalize(0.7 * np.asarray(state_vector, dtype=np.float32) + 0.3 * noise)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # DEMO 自检：无扩散器/记忆系统时验证流程可跑通（规范 §六.1）
    _cfg = get_config()
    _lc = LineController(_cfg, None, None, None, None)
    _lc.switch("L3")
    _sv = np.random.RandomState(0).randn(512).astype(np.float32)
    _ev = np.zeros(8, dtype=np.float32)
    _r = _lc.run(_sv, _ev, "你好，昔涟。你还记得花海吗？")
    print("[SELFTEST] line_used=%s vector=%s conf=%.3f hits=%d" % (
        _r["line_used"], _r["vector"].shape, _r["confidence"], len(_r["memory_hits"])))
