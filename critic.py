# -*- coding: utf-8 -*-
"""
critic.py — 批判层（默认关闭，按需加载，精简上下文 ≤512 token，规范 §4.1 / §四.1 / §十.10）
========================================================================================
内存/显存预估（规范 §九.1）:
  · DEMO 路径: 不加载任何重模型，仅标志位与若干计数器，< 1MB RAM，0 显存。
  · REAL 路径: 复用路由层 translator 持有的 Qwen3.5-0.8B（0.8B × FP32 ≈ 3.2GB，
    常驻 GPU，由 translator 持有，本模块只引用不复制权重，≈0 额外显存）。
    批判期间路由层按规范暂迁 CPU 释放显存（§4.1 路由层暂迁细节，由 server 协调）；
    批判结束立即 unload()，显存归还，路由层迁回 GPU。
  · 运行期临时分配: 1~2 个 512 维伪嵌入向量（约 4KB）与上下文裁剪副本（≤512 token）。

职责:
  1. load()/unload(): 默认关闭，按需加载 / 立即卸载（规范 §4.1）
  2. critique(candidate_text, context_text="") -> {"score","issues","verdict"}:
     · 精简上下文: 仅用最近 cfg.validator_context_turns(3) 轮，parse_turns 裁剪
       到 ≤ cfg.validator_context_max_tokens(512)（规范 §四.1）
     · REAL: 复用 translator 的 Qwen3.5-0.8B 生成式批判（提示词要求指出问题并
       输出 0-100 分），失败自动回退 DEMO 规则（规范 §九.2）
     · DEMO: 规则批判五维度 —— ①身份漂移 ②重复/复读 ③过短/空洞
       ④情感混乱乱码(arousal 关键词) ⑤风格(诗意/留白/柔和，标点密度启发式)
     · score 语义: 越高越好（0~100）；verdict: score>=60 → accept，否则 revise
  3. 关键路径 try-except 降级（规范 §九.2）: 模型加载/生成失败 → 自动回落 DEMO 规则

降级判定:
  · translator 为 None / translator.is_demo / cfg.demo → 一律走 DEMO 规则路径，
    并打印 [DEMO] 日志（规范 §六.1 降级边界）。

路径规范（规范 §九.12/§十.24）: 所有路径基于 config.BASE_DIR，禁止裸文件名/相对路径。

TODO(V3.4): 批判结果接入修正动作映射（issue 类型 → 重新生成参数调整）。
"""
import os
import re
import sys

# 本机 Python 运行时为隔离模式（sys.path 不含脚本目录/工作目录），
# 显式加入项目根目录，保证 `python critic.py` 可直接导入兄弟模块
# （config / common_utils）；重复导入无副作用。
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from config import get_config
from common_utils import (get_logger, safe_import, parse_turns,
                          cosine, clamp, pseudo_embedding, gpu_mem_mb)

_LOG = get_logger("critic")

# ---- DEMO 规则批判阈值 ----
_MIN_LEN = 8                 # 低于该字数视为“过短/空洞”（与 validator.py 一致）
_DUP_RATE_LIMIT = 0.30       # n-gram 去重率低于该值视为“复读”
_REVISE_SCORE = 60           # score >= 60 → accept，否则 revise（任务规格 §十.10）
_MAX_ISSUES = 5              # 单次批判最多输出 issue 条数

# ④ 情感混乱/乱码（arousal 相关关键词 + 替换符乱码信号）
_AROUSAL_NOISE_WORDS = ("乱码", "嘶吼", "尖叫", "抓狂", "崩溃",
                        "胡言乱语", "啊啊啊", "疯了", "发疯", "狂笑")
# ⑤ 风格启发式: 诗意/留白/柔和 的正向信号（柔和词 + 留白标点）
_SOFT_WORDS = ("轻声", "低语", "温柔", "安静", "缓缓", "轻轻",
               "呢喃", "柔和", "静谧", "轻叹")
_SOFT_PUNCS = ("…", "～", "——")        # 留白类标点（省略号/波浪线/破折号）
_HARD_PUNCS = ("！", "？", "!", "?")    # 激烈类标点（感叹/疑问 → 不够柔和）


class Critic:
    """批判层: 默认关闭、按需加载的精简上下文审查器（0~100 分 + issue 清单 + 判定）。"""

    def __init__(self, cfg=None, translator=None):
        self.cfg = cfg if cfg is not None else get_config()
        self.translator = translator      # 路由层翻译器（REAL 路径复用其 Qwen3.5-0.8B）
        self._loaded = False              # 默认关闭（规范 §4.1: 验证/批判层默认关闭）
        self._model = None                # REAL: 复用 translator.model 的引用；DEMO: None
        self._total = 0                   # 累计批判次数
        self._verdict_counts = {"accept": 0, "revise": 0}
        self._last = None                 # 最近一次批判结果（供 server 层展示/调试）

    # ------------------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        """是否已加载。server 层据此在加载期间对 /chat 返回 503（规范 §四.1）。"""
        return self._loaded

    def _use_demo(self) -> bool:
        """降级判定: translator 为 None / is_demo / cfg.demo → 走 DEMO 规则路径。"""
        tr = self.translator
        if tr is None:
            return True
        return bool(getattr(tr, "is_demo", True)) or bool(self.cfg.demo)

    # ------------------------------------------------------------------
    # 按需加载 / 立即卸载（规范 §4.1: 默认关闭，批判结束立即卸载）
    # ------------------------------------------------------------------
    def load(self) -> None:
        """按需加载。DEMO 仅置标志位（无重模型）；REAL 复用 translator 的 Qwen3.5-0.8B。"""
        if self._loaded:
            return
        if self._use_demo():
            # DEMO: 无重模型，仅置标志位 + [DEMO] 日志（规范 §六.1 降级边界）
            self._loaded = True
            _LOG.info("[CRITIC_ON] [DEMO] 模式：仅置标志位，不加载重模型")
            return
        # ---- REAL 模式：复用路由层 translator 的 Qwen3.5-0.8B（不复制权重）----
        try:
            safe_import("torch")          # torch/transformers 惰性获取；不可用也不抛异常
            tr = self.translator
            if tr is not None and getattr(tr, "model", None) is not None:
                # 复用同一模型对象 → 显存开销 ≈ 0（批判层不独占权重，规范 §4.1）
                self._model = tr.model
                _LOG.info("[CRITIC_ON] 复用路由层 Qwen3.5-0.8B 作为批判模型")
            else:
                # translator 无真实模型（权重缺失/加载失败）→ 降级启发式（规范 §九.2）
                _LOG.warning("[CRITIC] translator 无真实模型 → 降级为 DEMO 规则批判")
                self._model = None
            self._loaded = True
            _LOG.info("[CRITIC_ON] 批判层已加载，当前显存 %.0fMB", gpu_mem_mb())
        except Exception as e:
            _LOG.warning("[CRITIC] 加载失败(%s) → 降级为 DEMO 规则批判", e)
            self._loaded = True           # 降级: 仍允许批判流程继续

    def unload(self) -> None:
        """立即卸载（批判结束立即卸载，规范 §4.1 / §四.1），释放引用与显存。"""
        if not self._loaded:
            return
        self._model = None                # 只释放本模块引用，不卸载 translator 持有的模型
        self._loaded = False
        _LOG.info("[CRITIC_OFF] 批判层已卸载")
        try:
            torch = safe_import("torch")
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()  # 归还显存（路由层迁回 GPU 由 server 处理）
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 批判主入口
    # ------------------------------------------------------------------
    def critique(self, candidate_text, context_text="") -> dict:
        """批判候选文本，返回 {"score", "issues", "verdict"}。

        参数:
          candidate_text: 待批判候选回复（扩散器/翻译器输出）
          context_text  : 精简上下文文本（仅使用最近 3 轮，≤512 token，规范 §四.1）
                          —— 兼容传入多行文本或轮次列表

        返回:
          {"score": 0~100（越高越好）, "issues": [str 中文说明, ...],
           "verdict": "accept"|"revise"}，其中 score>=60 → accept。
        未加载则自动 load()，批判结束立即 unload()（规范 §4.1）。
        """
        if not self._loaded:
            self.load()
        try:
            ctx = self._trim_context(context_text)
            if not self._use_demo() and self._model is not None:
                result = self._critique_real(candidate_text, ctx)
            else:
                result = self._critique_demo(candidate_text, ctx)
            # score 语义: 越高越好（0~100）；verdict: >=60 accept，否则 revise
            score = clamp(round(float(result.get("score", 0.0)), 1), 0.0, 100.0)
            verdict = "accept" if score >= _REVISE_SCORE else "revise"
            result = {"score": score, "issues": list(result.get("issues", [])),
                      "verdict": verdict}
            self._total += 1
            self._verdict_counts[verdict] = self._verdict_counts.get(verdict, 0) + 1
            self._last = result
            _LOG.info("[CRITIC] score=%.1f verdict=%s issues=%d",
                      score, verdict, len(result["issues"]))
            return result
        finally:
            self.unload()   # 批判结束立即卸载（规范 §4.1 / §四.1）

    # ------------------------------------------------------------------
    # 上下文裁剪（规范 §四.1: 最近 3 轮，≤512 token）
    # ------------------------------------------------------------------
    def _trim_context(self, context_text) -> str:
        """精简上下文: 仅取最近 cfg.validator_context_turns 轮，parse_turns 裁剪
        到 ≤ cfg.validator_context_max_tokens(512)。"""
        if not context_text:
            return ""
        if isinstance(context_text, (list, tuple)):
            turns = [str(t) for t in context_text]
        else:
            # 多轮对话通常以换行分隔；无换行则视为单轮文本
            turns = [ln.strip() for ln in str(context_text).splitlines() if ln.strip()]
        turns = turns[-max(1, int(self.cfg.validator_context_turns)):]
        joined = "\n".join(turns)
        return parse_turns(joined, int(self.cfg.validator_context_max_tokens))

    # ------------------------------------------------------------------
    # REAL 生成式批判（复用 translator 的 Qwen3.5-0.8B，失败回退 DEMO）
    # ------------------------------------------------------------------
    def _critique_real(self, text, ctx: str) -> dict:
        """REAL 路径: 构造批判提示词交给 Qwen 生成式批判（规范 §4.1）。

        提示词要求模型指出候选回复的问题并输出 0-100 分；解析失败/生成失败
        → 自动回退 DEMO 规则（规范 §九.2 关键路径降级）。
        """
        prompt = (
            "你是批判层。请审查下面的候选回复，指出其中的问题"
            "（身份漂移/重复复读/过短空洞/情感混乱/风格不符等），"
            "逐条列出，最后一行输出 0-100 分（越高越好）。\n"
            "上下文:\n%s\n候选回复:\n%s\n批判结果:"
            % (ctx or "（无）", (text or "").strip()))
        try:
            tr = self.translator
            raw = tr.decode(prompt) if tr is not None else ""
            if not raw or not str(raw).strip():
                raise RuntimeError("批判生成结果为空")
            score = self._parse_real_score(str(raw))
            if score is None:
                raise RuntimeError("未能从模型输出解析 0-100 分数")
            issues = self._parse_real_issues(str(raw))
            return {"score": score, "issues": issues}
        except Exception as e:
            _LOG.warning("[CRITIC] REAL 生成式批判失败(%s) → 回退 DEMO 规则批判", e)
            return self._critique_demo(text, ctx)

    @staticmethod
    def _parse_real_score(raw: str):
        """从模型输出提取 0~100 分: 优先“数字+分”，否则取最后一个独立整数。"""
        if not raw:
            return None
        m = re.search(r"(?<![\d-])(\d{1,3})\s*分", raw)
        if m:
            return clamp(float(m.group(1)), 0.0, 100.0)
        nums = re.findall(r"(?<!\d)(\d{1,3})(?!\d)", raw)
        if nums:
            return clamp(float(nums[-1]), 0.0, 100.0)
        return None

    @staticmethod
    def _parse_real_issues(raw: str) -> list:
        """把模型输出的问题行解析为中文 issue 列表（去行首序号、跳过分数行、限条数）。"""
        issues = []
        for ln in (raw or "").splitlines():
            s = re.sub(r"^[\s\d\.\-•·、]+", "", ln.strip())      # 去掉行首序号/装饰
            if not s or re.search(r"\d{1,3}\s*分", s):            # 跳过“xx分”分数行
                continue
            issues.append(s)
            if len(issues) >= _MAX_ISSUES:
                break
        return issues

    # ------------------------------------------------------------------
    # DEMO 规则批判（纯 CPU 流程验证，规范 §六.1）—— 五维度
    # ------------------------------------------------------------------
    def _critique_demo(self, text, ctx: str) -> dict:
        """DEMO 规则批判: 五维度启发式，每条 issue 是一句中文说明。"""
        text = (text or "").strip()
        if not text:
            return {"score": 0.0, "issues": ["候选为空：没有任何可批判的内容"]}

        issues = []

        # ① 身份漂移: 与身份锚点余弦 < 阈值 → 指出（规范 §4.7.1 身份锚定）
        #    注: DEMO 用伪嵌入近似（不具备真实语义，规范 §6.1 降级边界）
        anchor = self.cfg.anchor_vector
        if anchor is None:
            anchor = pseudo_embedding(self.cfg.anchor_generation_prompt, self.cfg.STATE_DIM)
        cos = cosine(pseudo_embedding(text, self.cfg.STATE_DIM), anchor)
        thr = max(float(self.cfg.anchor_similarity_threshold), 1e-6)
        drift = clamp(1.0 - cos / thr, 0.0, 1.0)
        if cos < thr:
            issues.append("身份漂移：候选与身份锚点余弦 %.3f 低于阈值 %.2f，"
                          "语气可能偏离昔涟的温柔坚定" % (cos, self.cfg.anchor_similarity_threshold))

        # ② 重复/复读: n-gram 去重率过低（复读机式输出）
        dup = self._dup_rate(text)
        if dup > _DUP_RATE_LIMIT:
            issues.append("内容复读：n-gram 去重率仅 %.2f，疑似同一句话反复说" % dup)

        # ③ 过短/空洞: 字数不足，信息量不够支撑一次完整回应
        if len(text) < _MIN_LEN:
            issues.append("过短空洞：仅 %d 字，信息量不足" % len(text))

        # ④ 情感混乱/乱码: arousal 相关关键词或替换符（U+FFFD）等乱码信号
        #    （情绪向量维度不可用时以文本关键词近似，见 sentiment_module 的 arousal 阈值）
        if self._is_noise(text):
            issues.append("情感混乱：文本含 arousal 相关混乱信号（乱码/嘶吼/崩溃类词），"
                          "情绪表达失控")

        # ⑤ 风格: 诗意、留白、柔和（标点密度 + 柔和词启发式）
        style = self._style_score(text)
        if style < 0.5:
            issues.append("风格不符：标点密度与长度启发式显示缺少诗意、留白与柔和感")

        # ---- 汇总分数（越高越好，0~100）----
        score = 100.0
        score -= 30.0 * drift                                # ① 身份漂移最多扣 30
        score -= 40.0 * dup                                  # ② 复读最多扣 40
        if len(text) < _MIN_LEN:                             # ③ 过短最多扣 25
            score -= 25.0 * (1.0 - len(text) / _MIN_LEN)
        if self._is_noise(text):                             # ④ 情感混乱扣 25
            score -= 25.0
        if style < 0.5:                                      # ⑤ 风格不符最多扣 20
            score -= 20.0 * (0.5 - style) * 2.0
        return {"score": clamp(score, 0.0, 100.0), "issues": issues[:_MAX_ISSUES]}

    # ------------------------------------------------------------------
    # 内部小工具
    # ------------------------------------------------------------------
    @staticmethod
    def _dup_rate(text: str) -> float:
        """重复率 0~1: 1 - 唯一 n-gram / 总 n-gram（n 随文本长度自适应）。"""
        k = 3 if len(text) >= 8 else 2
        grams = [text[i:i + k] for i in range(max(0, len(text) - k + 1))]
        if not grams:
            return 0.0
        return 1.0 - len(set(grams)) / len(grams)

    @staticmethod
    def _is_noise(text: str) -> bool:
        """情感混乱/乱码检测: arousal 关键词命中 或 替换符(U+FFFD) 乱码信号。"""
        for w in _AROUSAL_NOISE_WORDS:
            if w in text:
                return True
        if "\ufffd" in text:
            return True
        return False

    @staticmethod
    def _style_score(text: str) -> float:
        """风格启发式 0~1: 诗意/留白/柔和。

        正向信号: 柔和词（轻声/低语/温柔…）、留白标点（省略号/波浪线/破折号）、
                 适中的标点密度（每 10~40 字一个标点）；
        负向信号: 感叹/疑问标点密集（不够柔和）、文本过长过密（无留白）。
        """
        if not text:
            return 0.0
        n = len(text)
        soft_hits = sum(1 for w in _SOFT_WORDS if w in text)
        soft_punc = sum(text.count(p) for p in _SOFT_PUNCS)
        hard_punc = sum(text.count(p) for p in _HARD_PUNCS)
        score = 0.0
        score += min(1.0, soft_hits / 2.0) * 0.35            # 柔和词
        score += min(1.0, soft_punc / 2.0) * 0.30            # 留白标点
        density = (soft_punc + hard_punc) / max(n, 1)        # 标点密度
        score += 0.20 if 0.02 <= density <= 0.15 else 0.10   # 适中为佳
        score -= min(0.30, hard_punc * 0.05)                 # 感叹/疑问密集 → 减分
        if n > 120:                                          # 过长过密 → 缺留白
            score -= min(0.15, (n - 120) * 0.002)
        return clamp(score, 0.0, 1.0)

    # ------------------------------------------------------------------
    def stats(self) -> dict:
        """批判层状态摘要（供 server 层监控）。"""
        return {
            "loaded": self._loaded,
            "mode": "DEMO规则" if self._use_demo() else "REAL(Qwen生成式)",
            "total_critiques": self._total,
            "verdict_counts": dict(self._verdict_counts),
            "last": self._last,
        }


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # DEMO 自检：纯 CPU 流程验证（规范 §六.1）
    _cfg = get_config()
    _c = Critic(_cfg, translator=None)          # translator=None → 自动 DEMO 规则路径
    # 良好候选（含柔和词“轻声”、留白标点“……”）
    _good = _c.critique(
        "（轻声）我记得……你说花海太安静了。安静，是因为它们在听你说话。",
        "用户：你记得我们上次见面吗？\n昔涟：（垂眸）记得。你走后，我把那天的风也收好了。")
    # 复读候选
    _bad = _c.critique("好的好的好的好的好的好的好的好的！！！", "用户：在吗？")
    # 情感混乱/乱码候选
    _noise = _c.critique("啊啊啊啊嘶吼崩溃乱码！！！！！", "用户：你还好吗？")
    print("[SELFTEST] good=%s" % _good)
    print("[SELFTEST] bad=%s" % _bad)
    print("[SELFTEST] noise=%s" % _noise)
    print("[SELFTEST] stats=%s" % _c.stats())
    # 契约断言
    assert 0 <= _good["score"] <= 100 and _good["verdict"] in ("accept", "revise")
    assert _good["verdict"] == "accept", "良好候选应判 accept"
    assert _bad["score"] < 60 and _bad["verdict"] == "revise", "复读候选应判 revise"
    assert _noise["score"] < 60 and _noise["verdict"] == "revise", "混乱候选应判 revise"
    assert all(isinstance(i, str) and i for i in _good["issues"]), "issue 应为非空中文说明"
    assert _c.is_loaded is False, "批判结束应立即卸载（规范 §4.1）"
    print("[SELFTEST] OK: 契约断言全部通过（score/verdict/issue/按需卸载）")
