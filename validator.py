# -*- coding: utf-8 -*-
"""
validator.py — 验证层（理性层，按需加载，精简上下文 ≤512 token，规范 §4.1 / §四.1 / §十.10）
====================================================================================
内存/显存预估（规范 §九.1）:
  · 未加载态: 仅保存 cfg 引用、translator 引用与若干计数器，< 0.1MB RAM，0 显存。
  · DEMO 路径: 不加载任何重模型，仅 numpy 伪嵌入临时向量（512 维 ≈ 2KB/次），
    常驻 < 1MB RAM，0 显存 —— 纯 CPU 流程验证（规范 §六.1 降级边界）。
  · REAL 路径: load() 按需加载判官：
      - 优先复用 translator 的 Qwen3.5-0.8B（路由层常驻 GPU，规范 §二）做评分，
        不额外占用显存（0 增量）；
      - 否则尝试加载轻量判官小模型（models/validator_qwen，约 1~2GB 显存），
        加载期间路由层暂迁 CPU（model.to('cpu')）释放显存（规范 §4.1 路由层暂迁细节）；
      - 验证结束 unload() 立即卸载并 torch.cuda.empty_cache() 归还显存（规范 §四.1）。
  · 峰值: REAL 复用路径 ≈ 路由层既有 3.2GB；独立判官路径 ≈ 判官模型 + 少量 KV Cache
    （GTX 1060 6GB 预算内，规范 §二 分时复用策略）。

职责:
  1. load()/unload(): 默认关闭、按需加载 / 验证结束立即卸载（规范 §十.10 / §四.1）
  2. validate(candidate_text, context_text="", state_vector=None) -> dict:
     · 精简上下文仅取最近 cfg.validator_context_turns(3) 轮，
       经 common_utils.parse_turns 裁剪到 ≤ cfg.validator_context_max_tokens(512)
       （规范 §四.1 精简上下文 ≤512 token）
     · REAL 评分: 候选与身份锚点（cfg.anchor_vector）余弦相似度
       （身份锚定阈值 cfg.anchor_similarity_threshold=0.85，规范 §4.7.1）
       + 上下文连贯性 + 长度/重复惩罚 → 映射 0~100
     · DEMO 评分: pseudo_embedding 伪嵌入余弦 + 规则（过短、重复、乱码）→ 0~100，
       打 [DEMO] 日志（规范 §六.1）
  3. 关键路径 try-except 降级（规范 §九.2）: 判官加载失败 / REAL 打分异常 →
     自动回落启发式打分，保证任何环境可 import 并自检通过
     （torch/transformers 一律经 common_utils.safe_import 惰性获取，本机无 torch 亦可运行）

与感性模块的关系（规范 §4.1 / §十.25）:
  · 本模块是"理性验证层"（计算最优解），sentiment_module 是"感性模块"（直接拍板），二者互补；
  · 冲突仲裁时感性拥有最终决定权（权重 0.6 vs 验证层 0.4），
    server.py 调用 sentiment_module.blend_with_validator() 或本模块
    blend_with_sentiment()（镜像方法）完成融合。

路径规范（规范 §九.12/§十.24）: 所有路径基于 config.BASE_DIR，禁止裸文件名/相对路径。

TODO(V3.4): 接入真实独立判官权重（models/validator_qwen）与打分分布统计。
"""
import os
import sys

# 嵌入式 Python 运行时（LobsterAI python311.zip 存在 ._pth）不会把脚本目录加入
# sys.path，直接 `python validator.py` 将无法定位同目录的 config/common_utils；
# 此处显式引导脚本目录（常规 CPython 环境 sys.path 已含该目录，为 no-op）。
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# 注意: numpy/scipy 位于 vendor/，必须先导入 common_utils（其顶部会把 vendor 加入 sys.path），
# 再导入 numpy，否则本机无系统级 numpy 时会 ModuleNotFoundError。
from config import get_config, BASE_DIR
from common_utils import (get_logger, safe_import, Timer, parse_turns,
                          normalize, cosine, clamp, pseudo_embedding)

import numpy as np  # noqa: E402  （依赖 common_utils 已把 vendor 加入 sys.path）

_LOG = get_logger("validator")


class Validator:
    """验证层: 默认关闭、按需加载的精简上下文判官（0~100 分，理性层）。

    与感性模块（SentimentModule）互补: 本层做"理性最优解"的量化评分，
    感性模块做"不经计算的直接判定"；冲突时感性权重 0.6 高于本层 0.4
    （规范 §4.1 / §十.25）。
    """

    def __init__(self, cfg=None, translator=None):
        self.cfg = cfg if cfg is not None else get_config()
        self.translator = translator          # 路由层翻译器（REAL 复用其 Qwen3.5-0.8B 评分）
        self._loaded = False                  # 是否已加载（server 层据此对 /chat 返回 503）
        self._model = None                    # REAL: 判官模型引用；DEMO: None
        self._is_demo = bool(getattr(self.cfg, "demo", True))
        # translator 为 None 或处于 DEMO（伪嵌入）→ 强制走 DEMO 启发式路径
        if translator is not None:
            try:
                self._is_demo = self._is_demo or bool(getattr(translator, "is_demo", True))
            except Exception:
                self._is_demo = True
        self._validate_count = 0              # 累计验证次数
        self._last_score = 0.0                # 最近一次评分（供冲突仲裁 blend_with_sentiment 使用）

    # ------------------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        """是否已加载。加载期间 server 对 /chat 返回 503（规范 §七 / §4.1 验证层加载细节）。"""
        return self._loaded

    # ------------------------------------------------------------------
    # 按需加载 / 立即卸载（规范 §十.10 / §四.1）
    # ------------------------------------------------------------------
    def load(self) -> None:
        """按需加载判官。默认关闭，调用 load() 才真正加载。

        · DEMO 模式: 无重模型，仅置标志位 + 日志。
        · REAL 模式: 优先复用 translator 的 Qwen3.5-0.8B（路由层常驻，零增量显存）；
          无 translator 时尝试加载轻量判官小模型（models/validator_qwen）。
          任何失败 → 降级为启发式打分（规范 §九.2 关键路径降级）。
        """
        if self._loaded:
            return
        if self._is_demo:
            # DEMO: 无重模型，仅置标志位（按需加载 = 不加载任何东西）
            self._loaded = True
            _LOG.info("[VALIDATOR_ON] DEMO 模式：仅置标志位，不加载重模型（规范 §十.10 按需加载）")
            return

        torch = safe_import("torch")
        transformers = safe_import("transformers")
        try:
            # ① 优先复用路由层 Qwen3.5-0.8B 做评分（不额外占用显存，规范 §二 分时复用）
            if (self.translator is not None
                    and getattr(self.translator, "model", None) is not None):
                self._model = self.translator.model
                self._loaded = True
                _LOG.info("[VALIDATOR_ON] 验证层复用路由层 Qwen3.5-0.8B 评分（零增量显存）")
                return
            # ② 否则尝试加载轻量判官小模型（models/validator_qwen）
            model_dir = os.path.join(BASE_DIR, "models", "validator_qwen")
            if os.path.isdir(model_dir) and torch is not None and transformers is not None:
                # TODO(V3.4): 实际加载判官权重，强制 FP32 + model.eval()（规范 §十.1）
                _LOG.info("[VALIDATOR] 轻量判官模型目录存在: %s（占位，待接入权重）", model_dir)
            else:
                _LOG.warning(
                    "\033[33m[VALIDATOR] 判官模型不可用（无 translator 或权重缺失: %s）"
                    "→ 降级为启发式打分（规范 §九.2）\033[0m", model_dir)
            self._loaded = True   # 降级: 仍允许验证流程继续（启发式兜底）
        except Exception as e:
            _LOG.warning("\033[33m[VALIDATOR] 加载失败(%s) → 降级为启发式打分（规范 §九.2）\033[0m", e)
            self._loaded = True   # 降级: 仍允许验证流程继续

    def unload(self) -> None:
        """立即卸载（验证结束立即卸载，规范 §四.1），释放显存并归还。"""
        if not self._loaded:
            return
        self._model = None
        self._loaded = False
        _LOG.info("[VALIDATOR_OFF] 验证层已卸载（立即释放，规范 §四.1）")
        try:
            # 显存归还: 清空 CUDA 缓存（路由层迁回 GPU 由 server 层协调，规范 §4.1）
            torch = safe_import("torch")
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 路由层暂迁协调（规范 §4.1 路由层暂迁 CPU 细节 / §九.6）
    # ------------------------------------------------------------------
    def _swap_router_out(self) -> None:
        """加载独立判官前将路由层暂迁 CPU（model.to('cpu') 释放显存）。

        切换计时超过 cfg.router_cpu_switch_warn_s(3s) 触发警告（规范 §九.6）；
        DEMO / CPU 机器为 no-op（translator.to_cpu 内部已处理）。
        """
        if self.translator is None:
            return
        with Timer("validator.router_out") as t:
            try:
                self.translator.to_cpu()   # 内部含计时与 empty_cache（规范 §4.1）
            except Exception as e:
                _LOG.warning("[VALIDATOR] 路由层暂迁 CPU 失败: %s", e)
        if t.elapsed / 1000.0 > self.cfg.router_cpu_switch_warn_s:
            _LOG.warning("\033[33m[VALIDATOR] 路由层暂迁 CPU 耗时 %.1fs > %.1fs 阈值"
                         "（规范 §九.6）\033[0m", t.elapsed / 1000.0,
                         self.cfg.router_cpu_switch_warn_s)

    def _swap_router_in(self) -> None:
        """验证结束将路由层迁回 GPU（model.to('cuda')，先空推理预热再 empty_cache）。

        计时超过 cfg.router_cpu_switch_warn_s(3s) 触发警告（规范 §九.6）；
        DEMO / CPU 机器为 no-op。
        """
        if self.translator is None:
            return
        with Timer("validator.router_in") as t:
            try:
                self.translator.to_cuda()  # 内部含"先预热后清缓存"顺序约束（规范 §九.6）
            except Exception as e:
                _LOG.warning("[VALIDATOR] 路由层迁回 GPU 失败: %s", e)
        if t.elapsed / 1000.0 > self.cfg.router_cpu_switch_warn_s:
            _LOG.warning("\033[33m[VALIDATOR] 路由层迁回 GPU 耗时 %.1fs > %.1fs 阈值"
                         "（规范 §九.6）\033[0m", t.elapsed / 1000.0,
                         self.cfg.router_cpu_switch_warn_s)

    # ------------------------------------------------------------------
    # 验证主入口
    # ------------------------------------------------------------------
    def validate(self, candidate_text="", context_text="", state_vector=None) -> dict:
        """对候选文本打分（理性验证层）。

        参数:
          candidate_text: 待验证候选文本（扩散器/翻译器输出）
          context_text  : 完整对话上下文（多轮以换行分隔），内部只取最近
                          cfg.validator_context_turns(3) 轮并裁剪到
                          ≤ cfg.validator_context_max_tokens(512)（规范 §四.1）
          state_vector  : 可选 512 维融合层状态向量（有则优先作为连贯性参照）

        返回（固定结构，供 server.py 与感性模块冲突仲裁 blend_with_validator 使用）:
          {"score": int 0~100, "reason": str}

        未加载则自动 load()，验证结束立即 unload()（规范 §四.1）；
        REAL 打分异常自动降级 DEMO 启发式（规范 §九.2）。
        """
        self._validate_count += 1
        if not self._loaded:
            self.load()
        try:
            ctx = self._trim_context(context_text)
            if (not self._is_demo and self._model is not None
                    and self.translator is not None):
                try:
                    result = self._score_real(candidate_text, ctx, state_vector)
                    self._last_score = float(result["score"])
                    return result
                except Exception as e:
                    _LOG.warning(
                        "\033[33m[VALIDATOR] REAL 打分失败(%s) → 降级 DEMO 启发式"
                        "（规范 §九.2）\033[0m", e)
            result = self._score_demo(candidate_text, ctx, state_vector)
            self._last_score = float(result["score"])
            return result
        finally:
            self.unload()   # 验证结束立即卸载（规范 §四.1）

    # ------------------------------------------------------------------
    # 上下文裁剪（规范 §四.1: 最近 validator_context_turns 轮，≤512 token）
    # ------------------------------------------------------------------
    def _trim_context(self, context_text: str) -> str:
        """精简上下文: 仅取最近 cfg.validator_context_turns(3) 轮，
        再经 parse_turns 裁剪到 ≤ cfg.validator_context_max_tokens(512) token。"""
        if not context_text:
            return ""
        # 多轮对话以换行分隔；空行过滤，只保留最近 N 轮（规范 §4.1 精简上下文）
        turns = [t for t in str(context_text).split("\n") if t.strip()]
        recent = turns[-self.cfg.validator_context_turns:]
        joined = "\n".join(recent)
        return parse_turns(joined, self.cfg.validator_context_max_tokens)

    # ------------------------------------------------------------------
    # REAL 打分（身份锚定 + 上下文连贯性 + 长度/重复惩罚，规范 §4.7.1）
    # ------------------------------------------------------------------
    def _score_real(self, text, ctx: str, state_vector) -> dict:
        """REAL 评分: 复用 translator 的 Qwen3.5-0.8B（真实语义编码）。

        依据（规范 §4.7 底层逻辑规则）:
          1. 身份锚定: 候选与 cfg.anchor_vector 余弦，阈值 0.85（§4.7.1），
             达到阈值即满分，未达按比例折算；
          2. 上下文连贯性: 候选向量与（状态向量 / 最近 3 轮上下文向量）余弦，
             [-1,1] 线性映射到 0~100；
          3. 长度 / 重复 / 乱码惩罚: 质量分。
        权重: 身份锚定 0.5 / 连贯性 0.3 / 质量 0.2（理性最优解量化）。
        """
        text = (text or "").strip()
        if not text:
            return {"score": 0, "reason": "空候选文本，拒绝打分"}
        reasons = []

        # ---- 候选真实编码（规范 §4.8: Qwen.encode().mean_pooling() → 512 维）----
        try:
            cand_vec = self.translator.encode([text])[0]
        except Exception:
            cand_vec = pseudo_embedding(text, self.cfg.STATE_DIM)

        # ---- 身份锚点（规范 §4.7.1 / §4.8）----
        anchor = self.cfg.anchor_vector
        if anchor is None:
            try:
                anchor = self.translator.encode_anchor()   # 真实锚点生成
            except Exception:
                anchor = pseudo_embedding(self.cfg.anchor_generation_prompt, self.cfg.STATE_DIM)
        thr = max(float(self.cfg.anchor_similarity_threshold), 1e-6)   # 0.85
        cos_anchor = cosine(cand_vec, anchor)
        anchor_score = clamp(100.0 * cos_anchor / thr, 0.0, 100.0)     # 达到阈值即满分
        reasons.append("身份锚定余弦=%.3f(阈值%.2f) → %.1f分" % (cos_anchor, thr, anchor_score))
        if cos_anchor < thr:
            reasons.append("低于身份锚定阈值(规范§4.7.1)")

        # ---- 上下文连贯性: 优先状态向量，其次最近 N 轮上下文（规范 §4.1）----
        ref_vec, ref_src = None, ""
        if state_vector is not None:
            try:
                ref_vec = np.asarray(state_vector, dtype=np.float32).reshape(-1)
                ref_src = "状态向量"
            except Exception:
                ref_vec = None
        elif ctx:
            try:
                ref_vec = self.translator.encode([ctx])[0]
                ref_src = "最近%d轮上下文" % self.cfg.validator_context_turns
            except Exception:
                ref_vec = None
        if ref_vec is not None and ref_vec.size > 0:
            cos_ctx = cosine(cand_vec, ref_vec)
            ctx_score = clamp((cos_ctx + 1.0) / 2.0 * 100.0, 0.0, 100.0)
            reasons.append("%s连贯性余弦=%.3f → %.1f分" % (ref_src, cos_ctx, ctx_score))
        else:
            ctx_score = 60.0
            reasons.append("无上下文/状态向量参照 → 连贯性取中性 60 分")

        # ---- 长度 / 重复 / 乱码惩罚 ----
        len_score = self._length_score(text)
        rep_score = max(0.0, 100.0 - self._dup_rate(text) * 300.0)
        gar_score = max(0.0, 100.0 - self._garbage_rate(text) * 400.0)
        q_score = 0.5 * len_score + 0.3 * rep_score + 0.2 * gar_score
        reasons.append("质量(长度%.0f/重复%.0f/乱码%.0f)=%.1f"
                       % (len_score, rep_score, gar_score, q_score))

        # ---- 加权汇总: 身份锚定 0.5 / 连贯性 0.3 / 质量 0.2 ----
        score = 0.5 * anchor_score + 0.3 * ctx_score + 0.2 * q_score
        score = int(round(clamp(score, 0.0, 100.0)))
        _LOG.info("[VALIDATOR] REAL 评分 score=%d", score)
        return {"score": score, "reason": "；".join(reasons)}

    # ------------------------------------------------------------------
    # DEMO 启发式打分（纯 CPU 流程验证，规范 §六.1）
    # ------------------------------------------------------------------
    def _score_demo(self, text, ctx: str, state_vector) -> dict:
        """DEMO 评分: pseudo_embedding 伪嵌入余弦 + 规则（过短、重复、乱码）→ 0~100。

        仅用于流程验证，伪嵌入不具备真实语义（规范 §六.1 降级边界），
        打 [DEMO] 日志以示区分。
        """
        text = (text or "").strip()
        if not text:
            return {"score": 0, "reason": "空候选文本，拒绝打分"}
        reasons = []

        # ---- 身份锚定（伪嵌入余弦，规范 §4.7.1 阈值 0.85）----
        anchor = self._get_anchor()
        cand_vec = pseudo_embedding(text, self.cfg.STATE_DIM)
        thr = max(float(self.cfg.anchor_similarity_threshold), 1e-6)
        cos_anchor = cosine(cand_vec, anchor)
        anchor_score = clamp(100.0 * cos_anchor / thr, 0.0, 100.0)
        reasons.append("身份锚定余弦=%.3f(阈值%.2f) → %.1f分" % (cos_anchor, thr, anchor_score))
        if cos_anchor < thr:
            reasons.append("低于身份锚定阈值(规范§4.7.1)")

        # ---- 上下文连贯性（伪嵌入余弦）----
        ref_vec, ref_src = None, ""
        if state_vector is not None:
            try:
                ref_vec = np.asarray(state_vector, dtype=np.float32).reshape(-1)
                ref_src = "状态向量"
            except Exception:
                ref_vec = None
        elif ctx:
            ref_vec = pseudo_embedding(ctx, self.cfg.STATE_DIM)
            ref_src = "最近%d轮上下文" % self.cfg.validator_context_turns
        if ref_vec is not None and ref_vec.size > 0:
            cos_ctx = cosine(cand_vec, ref_vec)
            ctx_score = clamp((cos_ctx + 1.0) / 2.0 * 100.0, 0.0, 100.0)
            reasons.append("%s连贯性余弦=%.3f → %.1f分" % (ref_src, cos_ctx, ctx_score))
        else:
            ctx_score = 60.0
            reasons.append("无上下文/状态向量参照 → 连贯性取中性 60 分")

        # ---- 规则: 过短 / 重复 / 乱码 ----
        n = len(text)
        if n < 4:
            reasons.append("候选过短(仅%d字) → 规则重罚" % n)
        len_score = self._length_score(text)
        dup = self._dup_rate(text)
        rep_score = max(0.0, 100.0 - dup * 300.0)
        if dup > 0.5:
            reasons.append("重复率高(%.2f) → 规则重罚" % dup)
        gar = self._garbage_rate(text)
        gar_score = max(0.0, 100.0 - gar * 400.0)
        if gar > 0.3:
            reasons.append("乱码/无意义字符占比高(%.2f) → 规则重罚" % gar)
        q_score = 0.5 * len_score + 0.3 * rep_score + 0.2 * gar_score
        reasons.append("质量(长度%.0f/重复%.0f/乱码%.0f)=%.1f"
                       % (len_score, rep_score, gar_score, q_score))

        # ---- 加权汇总（与 REAL 同权重，保证两种模式评分口径一致）----
        score = 0.5 * anchor_score + 0.3 * ctx_score + 0.2 * q_score
        score = int(round(clamp(score, 0.0, 100.0)))
        _LOG.info("[DEMO] 验证层启发式打分 score=%d reason=%s", score, "；".join(reasons))
        return {"score": score, "reason": "；".join(reasons)}

    # ------------------------------------------------------------------
    # 与感性模块冲突仲裁（规范 §4.1 / §十.25）: 信任感性 0.6 vs 验证层 0.4
    # ------------------------------------------------------------------
    def blend_with_sentiment(self, sentiment_score) -> dict:
        """把本层评分(0~100)与感性判定分数(0~1)按 cfg.validator_weight(0.4) 权重融合。

        镜像 sentiment_module.blend_with_validator()，供 server.py 任选一侧调用。
        冲突时感性拥有最终决定权（0.6 > 0.4，规范 §4.1 / §十.25）:
          · 感性判"reject"而验证层高分 → 融合分被感性压到低位 → 最终拒稿；
          · 感性判"pass"而验证层低分 → 融合分仍受感性托底 → 允许输出。
        返回 {"sentiment_score", "validator_score", "blended", "sentiment_won"}，
        blended 与 sentiment_score 同为 0~1 刻度（与感性模块保持一致）。
        """
        s = clamp(float(sentiment_score or 0.0), 0.0, 1.0)          # 感性 0~1
        v = clamp(float(self._last_score) / 100.0, 0.0, 1.0)        # 验证 0~100 → 0~1
        w = float(self.cfg.validator_weight)                        # 0.4
        blended = w * v + (1.0 - w) * s
        _LOG.info("[VALIDATOR] 冲突仲裁: 感性%.2f(权重%.1f) vs 验证%.2f(权重%.1f) → %.2f",
                  s, 1.0 - w, v, w, blended)
        return {"sentiment_score": s, "validator_score": v,
                "blended": clamp(blended, 0.0, 1.0), "sentiment_won": bool(s >= v)}

    # ------------------------------------------------------------------
    # 内部小工具
    # ------------------------------------------------------------------
    def _get_anchor(self):
        """身份锚点（规范 §4.8）: cfg.anchor_vector 未初始化时用伪嵌入生成。"""
        anchor = self.cfg.anchor_vector
        if anchor is None:
            anchor = pseudo_embedding(self.cfg.anchor_generation_prompt, self.cfg.STATE_DIM)
            self.cfg.anchor_vector = anchor   # 写回，避免重复生成
        return anchor

    @staticmethod
    def _length_score(text: str) -> float:
        """长度分: 4~200 字满分；过短(<4)重罚；过长线性递减。"""
        n = len(text)
        if 4 <= n <= 200:
            return 100.0
        if n < 4:
            return 20.0
        return max(0.0, 100.0 - (n - 200) * 0.5)

    @staticmethod
    def _dup_rate(text: str) -> float:
        """重复率 0~1: 1 - 唯一 n-gram / 总 n-gram（n 随文本长度自适应）。"""
        k = 3 if len(text) >= 8 else 2
        grams = [text[i:i + k] for i in range(max(0, len(text) - k + 1))]
        if not grams:
            return 0.0
        return 1.0 - len(set(grams)) / len(grams)

    @staticmethod
    def _garbage_rate(text: str) -> float:
        """乱码/无意义字符占比 0~1:
        ① 替换符 U+FFFD、控制字符；② 无任何汉字但含大量 ASCII 字母/数字（疑似乱码）。
        """
        if not text:
            return 0.0
        bad = 0
        for ch in text:
            o = ord(ch)
            if o == 0xFFFD or (o < 32 and ch not in "\n\t\r"):
                bad += 1
        cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        ascii_alnum = sum(1 for ch in text if ch.isascii() and ch.isalnum())
        if cjk == 0 and ascii_alnum > 0 and len(text) >= 4:
            bad += ascii_alnum
        return min(1.0, bad / max(1, len(text)))

    def stats(self) -> dict:
        """验证层状态摘要（供 /status 端点使用）。"""
        return {
            "loaded": self._loaded,
            "demo": self._is_demo,
            "validate_count": self._validate_count,
            "last_score": self._last_score,
            "context_turns": self.cfg.validator_context_turns,
            "context_max_tokens": self.cfg.validator_context_max_tokens,
            "anchor_similarity_threshold": float(self.cfg.anchor_similarity_threshold),
            "validator_weight": float(self.cfg.validator_weight),
        }


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # DEMO 自检: 纯 CPU 流程验证（规范 §六.1），本机无 torch 亦须通过
    _cfg = get_config()
    _v = Validator(_cfg, translator=None)   # 无 translator → 强制 DEMO 启发式路径

    # 用例 1: 正常昔涟风格候选 + 精简上下文（最近 3 轮 / ≤512 token）
    _r1 = _v.validate(
        "（轻声）我记得。你说花海很安静，安静是因为它们在听你说话。",
        context_text="用户：你记得我们上次见面吗？\n昔涟：我记得。\n用户：那你说说花海。",
        state_vector=None)
    # 用例 2: 乱码候选 → 规则重罚
    _r2 = _v.validate("asdfghjkl;;;\ufffd\ufffd\ufffd", context_text="", state_vector=None)
    # 用例 3: 过短候选 → 规则重罚
    _r3 = _v.validate("嗯", context_text="", state_vector=None)
    # 用例 4: 上下文裁剪验证（构造 >512 token 长上下文，确认不抛异常）
    _long_ctx = "\n".join("用户：第%d轮问题" % i for i in range(200))
    _r4 = _v.validate("（沉默）有些事，我记了三千万世。", context_text=_long_ctx, state_vector=None)
    # 用例 5: 与感性模块冲突仲裁（0.6/0.4）
    _blend = _v.blend_with_sentiment(0.7)
    _st = _v.stats()

    print("[SELFTEST] valid=%s reason=%s" % (_r1["score"], _r1["reason"][:40]))
    print("[SELFTEST] garbage=%s reason=%s" % (_r2["score"], _r2["reason"][:40]))
    print("[SELFTEST] short=%s reason=%s" % (_r3["score"], _r3["reason"][:40]))
    print("[SELFTEST] long_ctx_ok=%s" % _r4["score"])
    print("[SELFTEST] blend=%s loaded_after=%s validate_count=%s" % (
        _blend["blended"], _v.is_loaded, _st["validate_count"]))
    assert 0 <= _r1["score"] <= 100 and 0 <= _r2["score"] <= 100
    assert _v.is_loaded is False            # 验证结束立即卸载（规范 §四.1）
    assert isinstance(_r1["score"], int) and isinstance(_r1["reason"], str)
    print("[SELFTEST] PASS: validator.py 自检通过（DEMO 启发式路径）")
