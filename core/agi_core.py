# -*- coding: utf-8 -*-
"""
core/agi_core.py — 昔涟AGI v7.3 核心（hub / 心跳 / 深度思考 / 认知架构）
========================================================================
显存预算（GTX 1060 6GB, 峰值 < 4.5GB, 默认去KV化=LLM 桥关闭）:
  匹配器大张量 4-bit ≈ 0.23GB + 20 情感回环 ≈ 0.095GB + 路由 ≈ 0.027GB
  + 预测器 ≈ 0.002GB + 记忆池 ≈ 0.004GB + 前向瞬态 ≈ 0.2GB ≈ 峰值 0.6GB ✓
  （XILIAN_LLM=1 开启 Qwen 桥时: 编码器 0.8B ≈ 0.6GB + 解码器 2B ≈ 1.2GB,
   总峰值 ≈ 2.4~2.9GB 仍 < 4.5GB ✓ —— 默认关闭以贯彻"去KV化"策略）

运行形态:
  本地模式: 固定走完整认知架构（不启用任务规划, GUI 前端）
  API 模式: 启用任务规划层 + 双模式路由（本地深度思考 / 云端 DeepSeek）,
            AGICore.handle_chat* 是 Cyrene-Agent 适配层的核心接口

本地认知架构（每次输入）:
  文本 → 规则编码(可选 Qwen 编码头) → Z(384) → 匹配器集群动态激活
       → 20 情感回环步进 → 50M 路由 5 决策 → 记忆池 GPU 检索 Top-30
       → 预测器 λ 混合 → 回复生成（模板优先, 可选 Qwen 生成）
  无输入: 心跳持续流转（采样记忆 → Z → 激活 → 情感 → 内部独白…)
"""
import os
import re
import time
import json
import gc
import threading
import random
from collections import deque

import numpy as np

import config
from core import protocol
from core.matcher_cluster import MatcherCluster
from core.emotion_loop import EmotionLoops
from core.router import Router
from core.predictor import ZPredictor
from core.diffuser import Diffuser
from core.memory_pool import MemoryPool

try:
    import torch
    _HAS_TORCH = True
except Exception:            # pragma: no cover
    torch = None
    _HAS_TORCH = False


# ======================================================================
# 规则编码（无 Qwen 时的默认编码路径; 也是 diffuser 文本→码的规则源）
# ======================================================================
_EMOTION_WORDS = [
    (0, ["开心", "高兴", "喜欢", "哈哈", "真好", "太棒", "惊喜", "快乐", "幸福", "笑"]),
    (1, ["难过", "伤心", "哭", "失落", "沮丧", "委屈", "好累", "痛苦", "呜"]),
    (2, ["生气", "气死", "讨厌", "烦", "火大", "不行", "可恶", "过分"]),
    (3, ["害怕", "恐惧", "担心", "不安", "吓", "怕", "紧张"]),
    (4, ["平静", "静静", "没事", "还好", "嗯", "渐渐", "平常"]),
    (5, ["相信", "放心", "依靠", "信你", "交给你", "信任"]),
    (6, ["痛", "疼", "受伤", "伤口", "疼死"]),
    (7, ["失去", "想念", "离开", "别离", "逝去", "思念", "不在"]),
]
_SEMANTIC_WORDS = [
    (0, ["你好", "嗨", "初次见面", "久别重逢", "在吗", "早上好", "晚上好"]),
    (1, ["再见", "晚安", "拜拜", "走了", "下次", "休息了"]),
    (2, ["难过", "害怕", "陪我", "安慰", "抱抱", "陪陪我", "累"]),
    (3, ["为什么", "是什么", "怎么", "讲讲", "告诉", "如何", "哪里", "多少"]),
    (4, ["记得", "回忆", "以前", "花海", "约定", "曾经", "那时", "往事"]),
    (5, ["真棒", "厉害", "喜欢", "好看", "好美", "聪明", "优秀"]),
    (6, ["哈哈", "逗", "有趣", "玩笑", "干嘛", "嘻嘻", "调皮"]),
    (7, ["认真", "重要", "一定", "必须", "答应", "郑重"]),
]


def _RULE_PATTERNS(text: str):
    """规则编码: 文本 → (情感槽位 0..7, 语义槽位 0..7)。

    返回 emotion_id 与 semantic_id（关键词命中, 未命中取 4=平静 / 3=思考）。
    """
    t = text or ""
    emo = 4
    for slot, words in _EMOTION_WORDS:
        if any(w in t for w in words):
            emo = slot
            break
    sem = 3
    for slot, words in _SEMANTIC_WORDS:
        if any(w in t for w in words):
            sem = slot
            break
    return emo, sem


def _rule_soft16(text: str) -> np.ndarray:
    """规则文本 → 16 维 soft 向量（情感 8 维 0.75 权重 + 语义 8 维 0.25 权重）。"""
    emo, sem = _RULE_PATTERNS(text)
    v = np.zeros(16, dtype=np.float32)
    v[emo] = 0.75 * (1.0 + 0.1 * emo)
    v[8 + sem] = 0.25 * (1.0 + 0.1 * sem)
    return v


def _text_to_z(text: str, cfg) -> np.ndarray:
    """规则编码 → 384 维 Z（L2 归一）。"""
    return protocol.vectors_to_z(_rule_soft16(text), cfg)


# ----------------------------------------------------------------------
# API 模式输出净化: 剥离舞台指示/动作描写（（轻轻…）/（…））
# ----------------------------------------------------------------------
_ACTION_BRACKET_RE = re.compile(r"[（(][\s\S]{0,200}?[）)]", re.MULTILINE)


def _strip_action_brackets(text: str) -> str:
    """去除全部全/半角括号段（动作描写/舞台指示）, 并规范化空白。

    API 模式专用: 让回复以纯对话内容呈现（本地 GUI 保留原风格）。
    """
    out = _ACTION_BRACKET_RE.sub("", text)
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r"[ \t]+", " ", out)
    return out.strip()


# ======================================================================
# 工作记忆（前额叶缓存: 本轮完整上下文, 超出 token 上限截断头部）
# ======================================================================
class WorkingMemory:
    """工作记忆: 会话轮次 + token 估算; 超 8192 token 截断头部（【硬性】§二.4）。"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.turns = deque(maxlen=64)
        self.limit = cfg.working_memory_token_limit

    @staticmethod
    def _tokens(text: str) -> int:
        """近似 token 数: 中文 1 字 ≈ 1 token, 其余按 4 字符/token。"""
        cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        rest = max(0, len(text) - cjk)
        return cjk + int(rest / 4.0) + 1

    def add_user(self, text: str):
        self.turns.append(("user", text))
        self._trim()

    def add_assistant(self, text: str):
        self.turns.append(("assistant", text))
        self._trim()

    def _trim(self):
        """超限截断头部（保留最新轮次, 【硬性】§二.4）。"""
        total = sum(self._tokens(t) for _, t in self.turns)
        while total > self.limit and len(self.turns) > 1:
            _, dropped = self.turns.popleft()
            total -= self._tokens(dropped)

    def context_text(self, max_chars: int = 1500,
                     include_assistant: bool = True) -> str:
        """近期轮次文本（注入回复生成; 背景记忆由 agi_core 另行附加）。

        include_assistant=False: 仅注入用户轮次（防模型复读自身回复）。
        """
        parts = []
        for role, t in list(self.turns)[-8:]:
            if role == "assistant" and not include_assistant:
                continue
            who = "伙伴" if role == "user" else "昔涟"
            parts.append(f"{who}: {t}")
        text = "\n".join(parts)
        return text[-max_chars:] if len(text) > max_chars else text

    def tokens_est(self) -> int:
        return sum(self._tokens(t) for _, t in self.turns)

    def size(self) -> int:
        return len(self.turns)

    def sequence_zs(self, ks: int = 10) -> list:
        """最近 N 个用户输入的 Z 序列（供预测器）。"""
        zs = []
        for role, t in self.turns:
            if role == "user":
                zs.append(_text_to_z(t, self.cfg))
        return zs[-ks:]


# ======================================================================
# AGICore
# ======================================================================
class AGICore:
    """v7.3 核心: 组件装配 / 心跳循环 / 完整认知架构 / API 处理入口。"""

    def __init__(self, cfg=None, api_mode: bool = False):
        self.cfg = cfg if cfg is not None else config.get_config()
        self.cfg.ensure_dirs()
        self.log = self.cfg.setup_logging().getChild("core")
        self.api_mode = bool(api_mode)

        self.log.info("════════ 昔涟AGI v7.3 启动 (%s 模式) ════════",
                      "API" if self.api_mode else "本地")
        t0 = time.time()

        # ---- 编码/解码桥状态（必须先于 MemoryPool 构造, 其回调会调用 _encode_text） ----
        self._decoder = None          # 可选 Qwen 解码器（惰性）
        self._encoder = None          # 可选 Qwen 编码器（惰性）
        self._encoder_source = "rule"
        self._bridge_status = {"encoder": 0, "decoder": 0}   # 加载失败计数(防重载风暴)

        # ---- 记忆池（先建; 启动载入用快速规则编码, 用户输入由 Qwen 桥实时编码） ----
        self.memory = MemoryPool(self.cfg,
                                 encode_fn=lambda t: _text_to_z(t, self.cfg))
        # ---- 匹配器集群（双存储） ----
        self.cluster = MatcherCluster(self.cfg)
        # ---- 情感回环 / 路由 / 预测器 / 扩散器 ----
        self.loops = EmotionLoops(self.cfg)
        self.router = Router(self.cfg)
        self.predictor = ZPredictor(self.cfg)
        self.diffuser = Diffuser(self.cfg, memory_pool=self.memory)
        # ---- 扩散器启动: 未训练且语料在 → 直接训练（L2 转移场, 一次 ~60s） ----
        self._diffuse_codes = []              # 最近扩散候选码（认知流状态）
        self._last_in_code = protocol.encode(0, 3)   # 最近输入协议码（扩散中心）
        self._boot_diffuser()
        # ---- 工作记忆 ----
        self.wm = WorkingMemory(self.cfg)
        # ---- 认知状态 ----
        self._emotion = np.asarray(self.cfg.emotion_baseline, dtype=np.float32)
        self._z_seq = deque(maxlen=max(10, self.cfg.predictor_seq + 2))
        self._last_z = np.zeros(self.cfg.z_dim, dtype=np.float32)
        self._heartbeat = 0
        self._tick = 0
        self._last_input_ts = time.time()
        self._state = "active"            # active / idle / sleep
        self._pain = 0.0
        self._format_phase = 0
        self._throttled = False
        self._shutdown_flag = threading.Event()
        self._thread = None
        self._events = deque(maxlen=200)
        self._event_lock = threading.Lock()
        self._ongoing = threading.Lock()  # respond 串行化
        self._generating = threading.Event()  # 生成期中(心跳线程跳过 GPU 前向 = 独占)
        self._processing = threading.Event()  # 完整请求处理中(编码/匹配/生成整段, 心跳避让)
        self._pool = None                 # responses.json 模板缓存

        # ---- 任务规划 / 思维链（API 模式） ----
        if self.api_mode:
            from core.task_planner import TaskPlanner
            from core.chain_of_thought import ChainOfThought
            self.planner = TaskPlanner(self.cfg, self.cluster)
            self.cot = ChainOfThought(self.cfg, self)
        else:
            self.planner = None
            self.cot = None

        self.log.info("═══ 装配完成 (%.1fs) ═══ %s", time.time() - t0,
                      self.cfg.summarize())

    # ==================================================================
    # 编码 / 解码（规则主路径; Qwen 可选桥）
    # ==================================================================
    def _encode_text(self, text: str) -> np.ndarray:
        """文本 → 384 维 Z。

        优先 Qwen 编码头（models/qwen_encoder/head.pt, 仅在 enable_llm 时尝试）;
        否则规则编码（协议软码投影）。
        """
        if (self.cfg.enable_llm and self._encoder is None
                and self._bridge_status["encoder"] < 2):
            try:
                self._encoder = self._try_load_encoder()
                self._bridge_status["encoder"] += 1      # 成功 +1（不再重载）
            except Exception as e:
                self._bridge_status["encoder"] += 1
                self.log.warning("[CORE] Qwen 编码器加载失败(%d/2): %s",
                                 self._bridge_status["encoder"], str(e)[:120])
        if self._encoder is not None:
            try:
                z = self._encoder(text)
                if z is not None and np.linalg.norm(z) > 1e-9:
                    self._encoder_source = "qwen"
                    return z / np.linalg.norm(z)
            except Exception:
                self._encoder = None
        return _text_to_z(text, self.cfg)

    def _try_load_encoder(self):
        """加载 Qwen0.8B + 训练编码头（4-bit NF4; 失败返回 None 走规则路径）。

        编码头 head.pt（scripts/train_encoder_head.py 用昔涟语料训练）:
          {"version","project": [384, hidden], "mean": [hidden], ...}
        → z = L2(project @ (mean_pool(hidden) - mean))
        兼容旧键 {"weight": [384, hidden]}（无 mean, 直接投影）。
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, \
            BitsAndBytesConfig
        head_path = os.path.join(self.cfg.qwen_encoder_dir, "head.pt")
        model_dir = None
        for d in (self.cfg.qwen_encoder_dir, self.cfg.legacy_qwen_dir):
            if os.path.isdir(d) and os.path.isfile(os.path.join(d, "config.json")):
                model_dir = d
                break
        if model_dir is None or not os.path.isfile(head_path):
            raise FileNotFoundError("qwen_encoder/head.pt 或模型目录缺失")
        load4 = self.cfg.has_bitsandbytes()
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch.float16)
        tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=qcfg if load4 else None,
            device_map=dict(self.cfg.qwen_device_map) if load4 else None,
            torch_dtype=torch.float16, local_files_only=True)
        model.eval()
        st = torch.load(head_path, map_location="cpu", weights_only=False)
        w = st.get("project", st.get("weight"))            # [384, D] float32
        mean = st.get("mean", None)
        w = torch.as_tensor(np.asarray(w, dtype=np.float32),
                            device=model.device)
        mean_t = (torch.as_tensor(np.asarray(mean, dtype=np.float32),
                                  device=model.device)
                  if mean is not None else None)
        self.log.info("[CORE] 编码桥就绪: project %s (语料 %d 条, 方差 %.0f%%)",
                      tuple(w.shape), int(st.get("samples", 0)),
                      float(st.get("explained", 0)) * 100)

        def encode(text: str):
            ids = tok(text, return_tensors="pt", truncation=True,
                      max_length=256).to(model.device)
            with torch.no_grad():
                h = model(**ids, output_hidden_states=True) \
                    .hidden_states[-1][0].float()             # [T, D]
                h = h.mean(dim=0)                             # mean-pool（抗截断）
            if mean_t is not None:
                h = h - mean_t
            z = (w @ h.unsqueeze(1)).squeeze(1)               # [384]
            n = torch.linalg.norm(z)
            if float(n) < 1e-9:
                z = torch.zeros_like(z)
            else:
                z = z / n
            return z.cpu().numpy().astype(np.float32).reshape(-1)
        return encode

    def _try_load_decoder(self):
        """加载解码器（默认 0.8B FP32 轻量化提速; 单次生成, 不维护长 KV Cache）。

        - XILIAN_DECODER=2b 可切回 Qwen3.5-2B（models/qwen_decoder）
        - XILIAN_DECODER_QUANT=fp16 / nf4 可切换精度（nf4 为【硬性】4-bit 合规）
        - 昔涟 LoRA（train_sft_2b.py 出品, 针对 2B 基座）仅在基座匹配时自动挂载
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, \
            BitsAndBytesConfig
        model_dir = self.cfg.decoder_model_dir
        if not os.path.isdir(model_dir) or not os.path.isfile(
                os.path.join(model_dir, "config.json")):
            # 首选目录不可用 → 回退 2B 或 0.8B
            for d in (self.cfg.legacy_qwen_dir, self.cfg.qwen_decoder_dir):
                if os.path.isdir(d) and os.path.isfile(
                        os.path.join(d, "config.json")):
                    model_dir = d
                    break
            else:
                raise FileNotFoundError("解码器模型目录缺失")
        quant = getattr(self.cfg, "decoder_quant", "fp32").lower()
        load4 = self.cfg.has_bitsandbytes() and quant == "nf4"
        dtype = (torch.float32 if quant == "fp32"
                 else torch.float16 if quant == "fp16" else torch.float16)
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch.float16)
        tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=qcfg if load4 else None,
            device_map=(dict(self.cfg.qwen_device_map) if load4 else None),
            torch_dtype=dtype, local_files_only=True)
        if not load4 and model.device.type == "cpu":
            model = model.to(self.cfg.device)
        self.log.info("[CORE] 解码器加载完成: %s (%s%s)",
                      model_dir, "4-bit NF4" if load4 else quant.upper(),
                      " @ GPU" if not load4 else "")
        # 昔涟语料 SFT 适配器自动挂载（仅当适配器基座 == 当前模型目录）
        adapter_dir = self.cfg.sft_adapter_dir
        adapter_cfg = os.path.join(adapter_dir, "adapter_config.json")
        if os.path.isfile(adapter_cfg):
            try:
                import json as _json
                with open(adapter_cfg, "r", encoding="utf-8") as f:
                    base_name = _json.load(f).get("base_model_name_or_path", "")
                if os.path.normpath(base_name) == os.path.normpath(model_dir):
                    from peft import PeftModel
                    model = PeftModel.from_pretrained(model, adapter_dir,
                                                      is_trainable=False)
                    model.eval()
                    self.log.info("[CORE] 解码器已挂载昔涟 SFT 适配器 (LoRA)")
                else:
                    self.log.info(
                        "[CORE] 解码器 %s: LoRA 适配器基座为 %s, 跳过挂载（基座不匹配）",
                        model_dir, base_name)
            except Exception as e:
                self.log.warning("[CORE] LoRA 挂载失败 (%s) → 使用基座", e)
        self.log.info("[CORE] 解码器 4-bit 加载完成: %s", model_dir)
        return {"tok": tok, "model": model}

    def _generate_reply(self, text: str, z, emotion, decision, recalls) -> str:
        """回复生成: 模板优先（匹配器+情感筛选）, 可选 Qwen 单次生成。"""
        if (self.cfg.enable_llm and self._decoder is None
                and self._bridge_status["decoder"] < 2):
            try:
                self._decoder = self._try_load_decoder()
                self._bridge_status["decoder"] += 1
            except Exception as e:
                self._bridge_status["decoder"] += 1
                self.log.warning("[CORE] Qwen 解码器加载失败(%d/2): %s → 模板路径",
                                 self._bridge_status["decoder"], str(e)[:120])
        if self._decoder is not None:
            try:
                return self._llm_reply(text, recalls)
            except Exception as e:
                self.log.warning("[CORE] LLM 生成失败 (%s) → 模板路径", e)
        return self._template_reply(text, z, emotion, decision, recalls)

    def _llm_reply(self, text: str, recalls) -> str:
        """Qwen 单次生成（去KV化: 每次独立短上下文, 不维护长 KV Cache）。

        使用模型自带 chat template（Qwen3.5 之 im_start 格式, 与 SFT 训练一致）。
        API 模式: system 提示禁止舞台指示 + 输出后剥离括号动作（纯对话内容）。
        """
        tok, model = self._decoder["tok"], self._decoder["model"]
        system = (self.cfg.persona +
                  ("\n相关记忆: " + "；".join(c for _, _, c in recalls[:3])
                   if recalls else ""))
        if self.api_mode:
            system += ("\n注意：直接以对话内容开始回复，"
                       "不要使用任何括号动作描写或舞台指示"
                       "（例如「（轻轻合上书）（微微抬头）…」）。")
        ctx = self.wm.context_text(400, include_assistant=False)   # 仅用户轮次(防复读)
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user", "content": f"{ctx}\n伙伴：{text}"}],
            tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        self._generating.set()                 # 生成期 GPU 独占（心跳仅 CPU 流转）
        try:
            with torch.no_grad():
                out = model.generate(
                    **ids, max_new_tokens=self.cfg.decoder_max_new_tokens,
                    temperature=self.cfg.decoder_temperature,
                    do_sample=True, pad_token_id=tok.eos_token_id)
            gen = out[0][ids["input_ids"].shape[1]:]
            reply = tok.decode(gen, skip_special_tokens=True).strip()
        finally:
            self._generating.clear()
        if self.api_mode:
            reply = _strip_action_brackets(reply)      # API 输出净化
        return reply or "（风轻轻吹过……）"

    def _template_reply(self, text: str, z, emotion, decision, recalls) -> str:
        """模板回复: 按情感主标签 + 匹配器类别检索 responses.json。"""
        tag = self.cfg.emotion_names[int(np.argmax(emotion))]
        pool = self._response_pool()
        cands = pool.get(tag) or pool.get("calm") or []
        if not cands:
            return "嗯……人家在听哦。"
        # 优先选与当前语义相关的（top matcher 命中槽位 → 关键词重叠）
        sem, _ = _RULE_PATTERNS(text)
        sem_words = (_SEMANTIC_WORDS[sem][1] if sem < len(_SEMANTIC_WORDS) else [])
        scored = []
        for tpl in cands:
            if any(w in tpl for w in sem_words):
                scored.append(1.5)
            else:
                scored.append(1.0)
        weights = np.asarray(scored, dtype=np.float64)
        weights /= weights.sum()
        return str(np.random.choice(cands, p=weights))

    def _response_pool(self) -> dict:
        """缓存 responses.json（tag → 模板列表）。"""
        if getattr(self, "_pool", None) is not None:
            return self._pool
        pool = {}
        try:
            with open(self.cfg.responses_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for r in data.get("responses", []):
                pool.setdefault(r.get("tag", "calm"), []).append(r.get("text", ""))
        except (OSError, ValueError):
            pool = {"calm": ["嗯……人家在听哦。"]}
        self._pool = pool
        return pool

    # ==================================================================
    # 认知架构（本地模式固定走: 编码→匹配→情感→路由→记忆→回复）
    # ==================================================================
    def respond(self, text: str, meta: dict = None) -> None:
        """完整认知架构入口（用户输入 → 事件流; GUI/控制台轮询 pull_events）。"""
        with self._ongoing:
            self._processing.set()          # 请求处理期 GPU 独占（心跳仅 CPU 流转）
            try:
                self.log.debug("[CORE] respond: %s", (text or "")[:60])
                text = (text or "").strip()
                if not text:
                    return
                self._last_input_ts = time.time()
                self._state = "active"
                self._emit("input", text)

                # 任务规划（仅 API 模式）——本地模式固定完整认知架构
                if self.api_mode and self.planner is not None:
                    plan = self.planner.plan(text, None, None)
                    if plan.get("route") == "cloud":
                        self._emit("system", "（云端路由仅 API 入口生效, 本地会话按本地处理）")
                    elif plan.get("warn"):
                        self._emit("system", plan["warn"])

                # 1) 编码: 文本 → Z(384)
                z0 = self._encode_text(text)
                code = protocol.encode(1, _RULE_PATTERNS(text)[0])
                self.router.record_code(code)
                self._last_in_code = code          # 扩散中心种子（心跳扩散用）

                # 2) 深度思考（API 模式 local_deep 由 handle_chat 走 respond_structured;
                #    本地模式固定单轮认知架构）
                z, reasoning = self._cognition_cycle(text, z0)

                # 3) 记忆检索（GPU Top-30）+ 激活加分
                recalls = self.memory.recall(z, k=self.cfg.retrieval_top_k)
                self.memory.touch([m[1] for m in recalls[:8]])

                # 4) 决策（5 类）: 记忆检索/扩散 在决策驱动下补充动作
                decision = self.router.get_last()
                if decision.get("name") == "memory_recall":
                    for s, mid, content in self.memory.recall_related(z, n=3):
                        self._emit("thought", f"（想起了「{content[:40]}」）")
                elif decision.get("name") == "diffuse":
                    # 扩散决策: 语义扩散真实参与 —— 候选码入认知流, 并触发联想
                    interp = self.diffuser.diffuse([code], k=5)
                    if interp:
                        self._diffuse_codes = [int(c) for c, _ in interp][:12]
                        self._emit("thought", "（思绪沿着语义又往前走了几步）")

                # 5) 回复生成（模板/Qwen）
                reply = self._generate_reply(text, z, self._emotion, decision, recalls)
                if reasoning:
                    self._emit("thought", reasoning)

                # 6) 流式输出（GUI 逐字体验; 模板回复直接 reply 事件）
                self._emit("stream_start", "")
                self._emit("stream_chunk", reply)
                self._emit("reply", reply)

                # 7) 记忆写入 + 工作记忆
                tag = self.cfg.emotion_names[int(np.argmax(self._emotion))]
                self.memory.add(text, z, emotion_tag=tag, protocol_code=code)
                self.wm.add_user(text[:600])
                self.wm.add_assistant(reply[:600])

                # 8) 痛觉 / 格式化阶段（简化: 最近三条含「痛/疼」→ 推进阶段）
                if any(w in text for w in ("痛", "疼", "伤口", "离开我", "再见")):
                    self._pain = min(10.0, self._pain + 0.5)
                else:
                    self._pain = max(0.0, self._pain - self.cfg.pain_decay * 4)
                if self._pain >= self.cfg.pain_phase_threshold \
                        and self._format_phase < len(self.cfg.format_phases):
                    self._emit("format", self.cfg.format_phases[self._format_phase])
                    self._format_phase += 1
                    self._pain = 0.0
            finally:
                self._processing.clear()     # 请求处理完毕, 心跳恢复 GPU 流转

    # ------------------------------------------------------------------
    # 认知周期（编码后的 Z → 匹配 → 情感 → 预测器混合）
    # ------------------------------------------------------------------
    def _cognition_cycle(self, text: str, z0, deep: bool = False,
                         emit: bool = True):
        """一次完整认知周期: Z → 动态激活 → 情感回环 → 预测器混合。

        返回 (Z_final, reasoning_str)。
        deep=True 时（深度思考）: 迭代 2~3 轮, Z = 0.75·Z_match + 0.25·Z_pred,
          情感回环连续 5 心跳未变化则提前终止（【硬性】§三）。
        """
        strategy = self.cluster.on_heartbeat(idle=False)
        result = self.cluster.activate(z0, strategy)
        scores = result.get("scores")
        active = result.get("active_ids", [])
        self._z_seq.append(z0)

        z = np.asarray(z0, dtype=np.float32).copy()
        reasoning = ""
        stable = 0
        prev_emotion = self._emotion.copy()
        rounds = self.cfg.deep_think_rounds if deep else 1
        for r in range(rounds):
            summary = self.cluster.summary_vector(8)
            emotion = self.loops.step(summary, self._emotion, dt=1.0)
            self._emotion = emotion
            # 路由决策（5 类认知动作: 直接/扩散/情感调制/记忆检索/回环）
            self.router.decide(z, emotion)
            # 预测器混合（【硬性】§三: 0.75 匹配器 + 0.25 预测器）
            z_pred = self.predictor.predict(
                list(self._z_seq)[-self.cfg.predictor_seq:], self._emotion)
            w_match = self.cfg.z_match_weight if deep else (1.0 - self.cfg.z_mix_regular)
            w_pred = self.cfg.z_predict_weight if deep else self.cfg.z_mix_regular
            z = (w_match * np.asarray(z0, dtype=np.float32)
                 + w_pred * z_pred)
            z = z / (np.linalg.norm(z) + 1e-9)
            # 温和的二次激活（匹配器迭代: 以混合 Z 再激活, 修正匹配池）
            strategy2 = self.cluster.on_heartbeat(idle=False)
            self.cluster.activate(z, strategy2)
            delta = float(np.abs(emotion - prev_emotion).max())
            if delta < 0.01:
                stable += 1
                if stable >= self.cfg.deep_think_stop_beats:
                    break
            else:
                stable = 0
            prev_emotion = emotion.copy()

        if deep and self.cot is not None and self.cfg.chain_of_thought_enabled:
            reasoning = self.cot.render_reasoning({
                "text": text, "z": z, "emotion": self._emotion,
                "scores": scores, "active": active,
                "rounds": rounds, "stable": stable,
            })
        return z, reasoning

    # ==================================================================
    # API 模式入口（Cyrene-Agent 适配层调用; 契约见 adapter 文档）
    # ==================================================================
    def plan_request(self, text: str, reasoning_effort: str = None,
                     tools: list = None) -> dict:
        """任务规划: 路由决策 {route, reason, warn, forced, stripped}（仅 API 模式）。

        云端转发全局开关: cfg.cloud_forward_enabled=False（默认, 临时关闭）时,
        cloud 路由统一降级为 local（不调用 DeepSeek; XILIAN_CLOUD=1 恢复）。
        """
        if self.planner is None:
            return {"route": "local", "reason": "本地模式无任务规划层",
                    "warn": None, "forced": False, "stripped": text}
        plan = self.planner.plan(text, reasoning_effort, tools)
        if plan.get("route") == "cloud" and not self.cfg.cloud_forward_enabled:
            plan = dict(plan)
            plan["route"] = "local"
            plan["reason"] = "云端转发当前已关闭(临时), 转本地处理"
            plan["warn"] = "（云端转发已临时关闭, 本次由本地处理）"
            self.log.info("[CORE] 云端转发关闭 → 降级本地: %s", (text or "")[:40])
        return plan

    def respond_structured(self, text: str, reasoning_effort: str = None,
                           tools: list = None) -> dict:
        """非流式: 完整处理一次请求 → {id, model, route, content, reasoning,
        tool_calls, finish_reason, provider}。"""
        plan = self.plan_request(text, reasoning_effort, tools)
        route = plan["route"]
        clean = str(plan.get("stripped") or text)
        req_id = f"chatcmpl-{int(time.time() * 1000):x}"
        if route == "cloud":
            return {"id": req_id, "model": self.cfg.model_name, "route": route,
                    "content": None, "reasoning": None, "tool_calls": None,
                    "finish_reason": "cloud", "provider": "deepseek",
                    "cloud": True}
        if route == "local_deep":
            with self._ongoing:
                self._processing.set()
                try:
                    self._last_input_ts = time.time()
                    z0 = self._encode_text(clean)
                    z, reasoning = self._cognition_cycle(clean, z0, deep=True)
                    decision = self.router.get_last()
                    recalls = self.memory.recall(z, k=self.cfg.retrieval_top_k)
                    self.memory.touch([m[1] for m in recalls[:8]])
                    reply = self._generate_reply(clean, z, self._emotion, decision,
                                                 recalls)
                    self.memory.add(clean, z)
                    self.wm.add_user(clean[:600])
                    self.wm.add_assistant(reply[:600])
                finally:
                    self._processing.clear()
                return {"id": req_id, "model": self.cfg.model_name, "route": route,
                        "content": reply, "reasoning": reasoning or "（思考片刻）…",
                        "tool_calls": None, "finish_reason": "stop",
                        "provider": "cyrene-agi",
                        "warn": plan.get("warn"), "forced": plan.get("forced")}
        # local
        with self._ongoing:
            self._processing.set()
            try:
                self._last_input_ts = time.time()
                z0 = self._encode_text(clean)
                z, _ = self._cognition_cycle(clean, z0, deep=False)
                decision = self.router.get_last()
                recalls = self.memory.recall(z, k=self.cfg.retrieval_top_k)
                self.memory.touch([m[1] for m in recalls[:8]])
                reply = self._generate_reply(clean, z, self._emotion, decision,
                                             recalls)
                self.memory.add(clean, z)
                self.wm.add_user(clean[:600])
                self.wm.add_assistant(reply[:600])
            finally:
                self._processing.clear()
            return {"id": req_id, "model": self.cfg.model_name, "route": route,
                    "content": reply, "reasoning": None, "tool_calls": None,
                    "finish_reason": "stop", "provider": "cyrene-agi",
                    "warn": plan.get("warn"), "forced": plan.get("forced")}

    def respond_structured_stream(self, text: str, reasoning_effort: str = None):
        """流式: 生成器逐事件产出 → (event, data)。

        事件序列: reasoning_start → reasoning_delta* → reasoning_done
                  → reply_delta* → done
        data 均含 provider: "cyrene-agi"（【硬性】§五.4）。
        """
        result = self.respond_structured(text, reasoning_effort)
        if result.get("cloud"):
            yield "cloud_handle", result          # 交给适配层转发 DeepSeek
            return
        yield "reasoning_start", {"provider": "cyrene-agi"}
        if result.get("reasoning"):
            chunk = result["reasoning"]
            step = max(1, len(chunk) // 8)
            for i in range(0, len(chunk), step):
                yield "reasoning_delta", {"text": chunk[i:i + step],
                                          "provider": "cyrene-agi"}
        yield "reasoning_done", {"provider": "cyrene-agi"}
        content = result.get("content") or ""
        step = max(1, len(content) // 12)
        for i in range(0, len(content), step):
            yield "reply_delta", {"text": content[i:i + step],
                                  "provider": "cyrene-agi"}
        yield "done", {"finish_reason": result.get("finish_reason", "stop"),
                       "provider": "cyrene-agi",
                       "id": result.get("id")}

    # ==================================================================
    # 扩散器启动（L2 训练 + 心跳扩散参与）
    # ==================================================================
    def _boot_diffuser(self):
        """启动 L2 扩散器: v7.3 已训练则加载, 否则用昔涟语料训练（一次, ~60s）。"""
        if self.diffuser.is_trained() and not self.cfg.diffuser_force:
            self.log.info("[DIFFUSER] L2 扩散器已训练 (v7.3), 直接启用")
        else:
            self.log.info("[DIFFUSER] L2 未训练 → 用昔涟语料训练转移场…")
            if self.diffuser.train_l2():
                self.log.info("[DIFFUSER] L2 训练完成, 扩散器启动 ✓")
            else:
                self.log.warning("[DIFFUSER] 训练未完成 → L2 降级占位（L1/L3 可用）")

    def _diffuse_step(self):
        """心跳扩散（每 diffuser_beat_interval 心跳）: 以最近输入码为种子,
        在协议码空间做语义扩散, 候选写入认知流状态（供深度思考/状态面板）。"""
        try:
            interp = self.diffuser.diffuse([self._last_in_code], k=5)
            if interp:
                self._diffuse_codes = [int(c) for c, _ in interp][:12]
                self.log.debug("[DIFFUSER] 心跳扩散 → %s",
                               [hex(c) for c in self._diffuse_codes])
        except Exception as e:
            self.log.debug("[DIFFUSER] 心跳扩散跳过: %s", e)

    # ==================================================================
    # 心跳主循环（后台线程）
    # ==================================================================
    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._heartbeat_loop,
                                        name="heartbeat", daemon=True)
        self._thread.start()
        self.log.info("[CORE] 主循环已启动 (本地模式空闲 %ds 降频 / %ds 休眠微调)",
                      self.cfg.idle_after_seconds, self.cfg.sleep_after_seconds)

    def _heartbeat_loop(self):
        idle_since = time.time()
        sleep_cycles = 0
        while not self._shutdown_flag.is_set():
            t_start = time.time()
            self._tick += 1
            self._heartbeat += 1
            now = time.time()
            quiet = now - self._last_input_ts
            if quiet > self.cfg.sleep_after_seconds:
                self._state = "sleep"
            elif quiet > self.cfg.idle_after_seconds:
                self._state = "idle"
            else:
                self._state = "active"

            try:
                if self._state == "sleep":
                    sleep_cycles = self._sleep_cycle(sleep_cycles)
                else:
                    self._active_tick()
            except Exception as e:
                self.log.error("[CORE] 心跳异常: %s", e)

            # 显存监控（每 10 心跳; 超 4.5GB 告警 + 降频, 【硬性】§九）
            if self._tick % self.cfg.vram_check_ticks == 0:
                self._vram_monitor()

            # 心跳间隔: active 0.5s / idle 2.0s / sleep 5.0s; 超限降频×2
            interval = (self.cfg.heartbeat_interval_active
                        if self._state == "active"
                        else (self.cfg.heartbeat_interval_idle
                              if self._state == "idle"
                              else self.cfg.heartbeat_interval_sleep))
            if self._throttled:
                interval *= self.cfg.throttle_factor
            elapsed = time.time() - t_start
            wait = max(0.05, interval - elapsed)
            self._shutdown_flag.wait(wait)

    def _active_tick(self):
        """活跃/降频心跳: 持续流转（无输入时从记忆采样 → 激活 → 情感 → 独白）。

        生成期 GPU 独占（_generating 置位）: 跳过全部 GPU 前向
        （4000 匹配器全量激活 / 回环 / 预测器 / L2 扩散），仅维持 CPU 侧流转，
        避免 fp32 生成与心跳争抢显存与算力（服务器卡死根治）。
        """
        idle = self._state == "idle"
        if self._generating.is_set() or self._processing.is_set():
            self.memory.tick()
            return
        # 无输入: 采样记忆 → 平均 Z（持续流转的认知连续性）
        if self._last_input_ts < time.time() - 5.0:
            recalls = self.memory.recall(self._last_z, k=6)
            if recalls:
                self._last_z = _text_to_z(" ".join(c for _, _, c in recalls[:3]),
                                          self.cfg)
            else:
                self._last_z = _text_to_z("风轻轻吹过旧书页", self.cfg)
        strategy = self.cluster.on_heartbeat(idle=idle)
        self.cluster.activate(self._last_z, strategy)
        summary = self.cluster.summary_vector(8)
        self._emotion = self.loops.step(summary, self._emotion, dt=1.0)
        self._z_seq.append(self._last_z)
        if len(self._z_seq) >= self.cfg.predictor_seq:
            z_pred = self.predictor.predict(list(self._z_seq)[
                -self.cfg.predictor_seq:], self._emotion)
            self._last_z = ((1.0 - self.cfg.z_mix_regular) * self._last_z
                            + self.cfg.z_mix_regular * z_pred)
            self._last_z = self._last_z / (np.linalg.norm(self._last_z) + 1e-9)
        # 记忆自动落盘（每 50 心跳异步）
        self.memory.tick()
        # 语义扩散（每 diffuser_beat_interval 心跳; 认知流持续扩散参与）
        if self._heartbeat % self.cfg.diffuser_beat_interval == 0:
            self._diffuse_step()
        # 无输入: 每 15 心跳一句内心独白（≤30 字, 缓存 5 条）
        if idle and self._heartbeat % self.cfg.inner_thought_interval == 0 \
                and self.cot is not None:
            line = self.cot.inner_monologue({
                "emotion": self._emotion,
                "hits": self.cluster.top_matchers(3)})
            if line:
                self._emit("thought", f"（{line}）")
        # 协议映射表热更新（mtime 检查）
        if self._heartbeat % self.cfg.protocol_hot_reload_ticks == 0:
            try:
                from core import protocol as _p
                _pm = getattr(self, "_pm", None)
                if _pm is None:
                    _pm = _p.ProtocolMap(self.cfg)
                    self._pm = _pm
                _pm.check_reload()
            except Exception:
                pass

    def _sleep_cycle(self, cycles: int) -> int:
        """深度休眠: 逐周期微调 1~3 个匹配器; 每 5 周期更新预测器; 定期存档。

        唤醒检测: shutdown_flag/输入都会打断（respond 恢复 active）。
        生成期置位（_generating）: 跳过本轮微调（GPU 独占）。
        """
        if self._generating.is_set() or self._processing.is_set():
            return cycles + 1
        self.log.debug("[CORE] 休眠心跳 %d (微调窗口)", cycles)
        try:
            # 数据源: 工作记忆（用户输入 → Z / 回复锚点）
            samples_pool = []
            for role, t in list(self.wm.turns)[-12:]:
                if role == "user":
                    z = _text_to_z(t, self.cfg)
                    samples_pool.append((z, 0.9))
            if samples_pool:
                lo, hi = self.cfg.fine_tune_per_cycle
                n = random.randint(lo, hi)
                for _ in range(n):
                    idx = random.randint(0, self.cluster.count - 1)
                    got = self.cluster.fine_tune(idx, samples_pool)
                    if got:
                        self.log.info(
                            "[CORE] 休眠微调 #%d: score %.3f→%.3f (%d 样本)",
                            idx, got["before"], got["after"], got["samples"])
            # 预测器更新（每 5 睡眠周期, MSE 微调）
            if cycles % self.cfg.predictor_update_every == 4:
                zs = self.wm.sequence_zs(self.cfg.predictor_seq + 2)
                if len(zs) >= 3:
                    self.predictor.update(zs[:-1], zs[-1], self._emotion)
            # 低频存档
            if cycles % 10 == 4:
                self.memory.save_async()
        except Exception as e:
            self.log.error("[CORE] 休眠周期异常: %s", e)
        return cycles + 1

    def _vram_monitor(self):
        """显存监控: 目标 < 4.5GB; 超 4.4GB 告警, 超 4.5GB 降频（【硬性】§九）。"""
        vram = self._vram_mb()
        if vram > self.cfg.vram_budget_mb:
            self._throttled = True
            self.log.warning("[CORE] ⚠ 显存 %.0fMB 超预算 4.5GB → 降频 ×%.1f",
                             vram, self.cfg.throttle_factor)
            self._emit("system", f"（显存告警：{vram:.0f}MB，系统已自动降频）")
        elif vram > self.cfg.vram_warn_mb:
            self.log.warning("[CORE] ⚠ 显存 %.0fMB 接近告警线", vram)
        else:
            self._throttled = False
            self.log.debug("[CORE] vram=%.0fMB", vram)

    @staticmethod
    def _vram_mb() -> float:
        if torch is None or not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / (1024 ** 2)

    # ==================================================================
    # 事件（GUI / WebUI / 控制台轮询）
    # ==================================================================
    def _emit(self, kind: str, text: str, **extra):
        with self._event_lock:
            self._events.append({"type": kind, "text": text,
                                 "ts": time.time(), **extra})

    def pull_events(self) -> list:
        with self._event_lock:
            evs = list(self._events)
            self._events.clear()
        return evs

    # ==================================================================
    # 状态快照（GUI 调试窗口 / WebUI / 适配层）
    # ==================================================================
    def get_status(self) -> dict:
        ms = self.cluster.get_stats()
        return {
            "heartbeat": self._heartbeat,
            "tick": self._tick,
            "mode": "api" if self.api_mode else "local",
            "sleeping": self._state != "active",
            "state": self._state,
            "vram_mb": self._vram_mb(),
            "vram_budget": self.cfg.vram_budget_mb,
            "vram_warn": self._vram_mb() > self.cfg.vram_warn_mb,
            "throttled": self._throttled,
            "emotion": {n: float(v) for n, v in
                        zip(self.cfg.emotion_names, self._emotion.tolist())},
            "emotion_dominant": self.loops.dominant_info(),
            "memory": self.memory.stats(),
            "top_recent": self.memory.top_recent(5),
            "matcher_stats": ms,
            "predictor": self.predictor.get_stats(),
            "pain": self._pain,
            "format_phase": self._format_phase,
            "last_decision": self.router.get_last(),
            "encoder_source": self._encoder_source,
            "working_memory": {"turns": self.wm.size(),
                               "tokens_est": self.wm.tokens_est(),
                               "background": 8},
            "diffuser": self.diffuser.get_stats(),
            "diffuse_codes": len(self._diffuse_codes),
            "known_events": len(self._events),
        }

    # ==================================================================
    # 手动控制（GUI 按钮 / shutdown.py）
    # ==================================================================
    def sleep_now(self):
        if self._state == "sleep":
            return
        self._last_input_ts = time.time() - self.cfg.sleep_after_seconds - 1
        self._state = "sleep"
        self._emit("sleep", "昔涟进入了安静的梦乡")

    def wake_now(self):
        self._last_input_ts = time.time()
        self._state = "active"
        self._emit("wake", "嗯……人家醒来啦。")

    def force_save(self):
        """强制存档（记忆池阻塞落盘）。"""
        self.memory.shutdown_save()

    def respond_media(self, kind: str, media_path: str, text: str = "") -> bool:
        """媒体输入（WebUI 兼容保留）: 描述后走完整认知架构。"""
        desc = {"image": "（人家看到了一张照片）",
                "audio": "（人家听到了声音）",
                "video": "（人家看了一段影像）"}.get(kind, "（人家收到了一份文件）")
        self.respond(f"{desc} {text}".strip())
        return True

    # ==================================================================
    # 关闭
    # ==================================================================
    def shutdown(self):
        """关闭并存档（记忆池阻塞落盘 + 预测器落盘 + 信号清理）。"""
        self.log.info("[CORE] 关闭中: 存档记忆池 / 模型状态…")
        self._shutdown_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=6.0)
        try:
            self.memory.shutdown_save()
        except Exception as e:
            self.log.error("[CORE] 记忆存档失败: %s", e)
        try:
            self.predictor.save()
        except Exception:
            pass
        try:
            if os.path.isfile(self.cfg.shutdown_signal_path):
                os.remove(self.cfg.shutdown_signal_path)
        except OSError:
            pass
        if torch is not None and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
        self.log.info("[CORE] 已安全关闭")
