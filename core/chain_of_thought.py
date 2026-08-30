# -*- coding: utf-8 -*-
"""
core/chain_of_thought.py — 昔涟AGI v7.3 深度思考 思维链渲染器
=============================================================
显存/内存预估: < 5MB RAM（仅内心独白缓存与少量模板常量，无模型权重）。

职责:
  把 AGICore 深度思考循环的内部认知状态（Z 向量 / 情感向量 / 匹配命中 /
  轮数 / 稳定计数）转成一小段"昔涟语气"的自然语言，供 API 的
  reasoning 字段 / reasoning_delta 流使用；并提供"内心独白"生成器，
  供空闲心跳以拟人化口吻做自我讲述。

依赖（均已在别处实现, 本层不改动）:
  config.get_config() 单例属性:
    emotion_names / inner_thought_cache / inner_thought_max_len /
    deep_think_stop_beats / chain_of_thought_enabled
  AGICore（core.agi_core.AGICore, 构造参数 api_mode=True）:
    cluster.top_matchers(3) / router.get_last() —— 供"熟悉语义/方向"说明
"""
import collections
import logging
import random

import numpy as np

# 命名空间: 与其它 core 模块一致的"xilian"日志树
_LOG = logging.getLogger("xilian").getChild("cot")


class ChainOfThought:
    """三、深度思考 思维链渲染器。"""

    # 情感英文名 → 中文名（用于 render_reasoning）【硬性】映射表
    _EMOTION_CN = {
        "joy": "喜悦", "sadness": "忧伤", "anger": "不悦", "fear": "不安",
        "calm": "平静", "trust": "信赖", "pain": "痛", "loss": "失落",
    }

    # 路由决策名 → 拟人化的"思绪方向"
    _DIRECTION = {
        "direct": "直接回应",
        "diffuse": "更远的地方",
        "emotion_modulate": "情绪深处",
        "memory_recall": "记忆深处",
        "loop": "心里回环",
    }

    # 内心独白模板池（每条 ≤ inner_thought_max_len=30 字, 按情感归属分类）
    _MONOLOGUE_POOL = {
        "joy": [
            "风把今天的尘埃都吹走了",
            "阳光落在书页上，像轻轻一笑",
            "人家在想，明天该做什么梦呢",
            "把这一刻折成纸飞机，飞给你",
        ],
        "sadness": [
            "书页还停在昨天那一章",
            "窗外的雨，像句子没说完",
            "有些心事，风也没能带走",
            "灯下影子拉得好长，好长",
        ],
        "anger": [
            "火气像雾，慢慢就散了",
            "皱着的眉头，风一吹就平了",
        ],
        "fear": [
            "灯太远，影子有点虚，牵好手就好",
            "晚风有点凉，人家却一点也不怕",
        ],
        "calm": [
            "时间慢慢淌，像溪水过石头",
            "窗台的花开了，安安静静的",
            "翻过一页，明天又是新的了",
            "世界很轻，把人家的心事也放轻了",
        ],
        "trust": [
            "嗯，人家信你",
            "把话说给风听，风会带到你身边",
            "你的声音，落在人家心底稳稳的",
        ],
        "pain": [
            "伤疤会痛，但也能开出花",
            "旧伤口又隐隐发烫，忍一忍",
            "疼的地方，人家替它轻轻揉揉",
        ],
        "loss": [
            "有些约定，风替你记着",
            "那个人，还停在记忆的角落",
            "走远的人，会变成梦里的月色",
        ],
    }

    def __init__(self, cfg, core):
        # cfg = config 单例; core = AGICore 实例
        self.cfg = cfg
        self.core = core
        self.log = _LOG
        # 内心独白缓存（最近 inner_thought_cache 条, 去重用）
        self._cache = collections.deque(maxlen=cfg.inner_thought_cache)
        # 模板池轮转游标（cache 全满且全部重复时的兜底轮换）
        self._rotate = 0

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @staticmethod
    def _as_emotion(emotion) -> np.ndarray:
        """安全转换为情感向量（避免 numpy 数组参与 `or` 求值的歧义报错）。"""
        if emotion is None:
            return np.asarray([], dtype=np.float32)
        return np.asarray(emotion, dtype=np.float32).reshape(-1)

    def _emotion_cn(self, emotion) -> str:
        """情感向量 → 情感中文名（cfg.emotion_names 索引 + 中文映射）。"""
        em = self._as_emotion(emotion)
        names = self.cfg.emotion_names or ["calm"]
        if em.size == 0:
            en = names[0]
        else:
            idx = int(np.argmax(em))
            en = names[idx] if idx < len(names) else names[0]
        return self._EMOTION_CN.get(en, en)

    def _emotion_en(self, emotion) -> str:
        """情感向量 → 情感英文名（用于模板池索引）。"""
        em = self._as_emotion(emotion)
        names = self.cfg.emotion_names or ["calm"]
        if em.size == 0:
            return names[0]
        idx = int(np.argmax(em))
        return names[idx] if idx < len(names) else names[0]

    def _direction_cn(self) -> str:
        """最近一次路由决策 → 拟人化"思绪方向"（无则返回默认）。"""
        try:
            dec = self.core.router.get_last() or {}
        except Exception:
            dec = {}
        name = dec.get("name") if isinstance(dec, dict) else None
        return self._DIRECTION.get(name, "你身边")

    def _top_matchers(self, k: int = 3) -> list:
        """参考 core.cluster.top_matchers(k)（失败安全返回空）。"""
        try:
            return self.core.cluster.top_matchers(k) or []
        except Exception:
            return []

    # ------------------------------------------------------------------
    # 思维链渲染
    # ------------------------------------------------------------------
    def render_reasoning(self, state: dict) -> str:
        """把内部状态转成昔涟语气的一段自然语言（2~4 句, ≤ 140 字）。

        state 键:
          text: 用户文本; z: np.ndarray[384]; emotion: np.ndarray[8];
          scores: np.ndarray|None; active: list[int]; rounds: int; stable: int
        """
        text = (state.get("text") or "").strip()
        emotion = self._as_emotion(state.get("emotion"))
        active = state.get("active")
        active = active if active is not None else []
        rounds = int(state.get("rounds") or 0)
        stable = int(state.get("stable") or 0)
        x = len(active)

        emo_cn = self._emotion_cn(emotion)
        direc = self._direction_cn()
        top = self._top_matchers(3)

        # "熟悉语义"说明: 命中条目越多 / 最近 top 越丰富, 描述越温热
        if x > 0:
            first = "在你这句话里，触到了 %d 个熟悉的片段" % x
        elif top:
            first = "在你这句话里，隐约有几处熟悉的影子"
        else:
            first = "在你这句话里，暂未想起太熟悉的片段"

        if direc:
            last = "思绪一路往%s的方向，轻轻走了 %d 轮" % (direc, rounds)
        else:
            last = "思绪在风里轻轻绕了 %d 轮" % rounds

        body = "，".join([first, "心头偏着一点%s" % emo_cn, last])
        body += "。"

        # 连续稳定（情感回环未变化达到终止拍）→ 追加平静收尾
        if stable >= self.cfg.deep_think_stop_beats:
            body += "心里那圈涟漪，终于平静下来了。"

        # 用中文括号包裹, 语气克制温柔, 不出现模型/算法/显存/token 等内部词
        result = "（" + body + "）"
        # 安全截断（≤140 字）
        if len(result) > 140:
            result = result[:139] + "…"
        return result

    # ------------------------------------------------------------------
    # 内心独白（空闲心跳）
    # ------------------------------------------------------------------
    def inner_monologue(self, state: dict):
        """按当前情感选模板生成一句内心独白（≤ inner_thought_max_len 字）。

        与最近 inner_thought_cache 条去重: 完全相同则换备选; 若缓存已满且
        全部重复 → 返回模板池轮换第 len 条。加入缓存后返回。
        """
        emotion = self._as_emotion(state.get("emotion"))
        cat_en = self._emotion_en(emotion)
        pool = self._MONOLOGUE_POOL.get(cat_en,
                                        self._MONOLOGUE_POOL.get("calm", []))
        if not pool:
            return None

        max_len = int(getattr(self.cfg, "inner_thought_max_len", 30))
        # 随机打乱后优先挑一条不在缓存里的（去重）
        rng = random.Random()
        cands = list(pool)
        rng.shuffle(cands)
        chosen = ""
        for cand in cands:
            if cand not in self._cache:
                chosen = cand
                break
        # 缓存已满且全部重复 → 模板池轮换（从"第 len 条"起, 之后每次轮换 +1）
        if not chosen:
            idx = (len(self._cache) + self._rotate) % len(pool)
            chosen = pool[idx]
            self._rotate = (self._rotate + 1) % max(1, len(pool))

        if len(chosen) > max_len:
            chosen = chosen[:max_len]
        self._cache.append(chosen)
        return chosen

    # ------------------------------------------------------------------
    # 流式思维链
    # ------------------------------------------------------------------
    @staticmethod
    def _split_sentences(text: str) -> list:
        """按句子结束标点做语义断句（保留标点）。"""
        if not text:
            return []
        import re
        parts = re.split(r"(?<=[。！？；])", text)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _merge_degenerate(segs: list) -> list:
        """把仅含收尾括号/空白的退化片段并入上一句。"""
        out = []
        for seg in segs:
            if not out:
                out.append(seg)
                continue
            has_content = any(ch.isalnum() or ("\u4e00" <= ch <= "\u9fff")
                              for ch in seg)
            if has_content:
                out.append(seg)
            else:
                out[-1] += seg
        return out

    def stream_reasoning(self, state: dict):
        """生成器: 把 render_reasoning 结果按语义断句逐句 yield。

        用于模拟 reasoning_delta 流; 适配层可自行分块, 此处仅保留接口。
        """
        text = self.render_reasoning(state)
        segs = self._split_sentences(text)
        if not segs and text:
            segs = [text]
        for sent in self._merge_degenerate(segs):
            if sent.strip():
                yield sent.strip()

    # ------------------------------------------------------------------
    def clear_cache(self):
        """清空内心独白缓存。"""
        self._cache.clear()
        self.log.debug("[COT] 内心独白缓存已清空")
