# -*- coding: utf-8 -*-
"""
translator.py — 语义翻译器（Qwen3.5-0.8B，真实模型，规范 §4.1 路由层 / §4.8 身份锚点）
=====================================================================================
内存/显存预估（规范 §九.1）:
  REAL 路径: Qwen3.5-0.8B ≈ 0.8B 参数 × 4B(FP32) ≈ 3.2GB 显存（GTX 1060 6GB 预算内）
            + KV Cache(2048 token) ≈ 200~300MB + 框架开销 ≈ 300MB → 峰值 ≈ 3.8GB
            视觉版 Qwen3.5-0.8B-V 按需惰性加载，强制 CPU（感知层不占 GPU 显存，规范 §二）
            AMP/4bit 降级时显存减半至 1/4（标记 [AMP_MODE] / [4BIT_MODE]）
  DEMO 路径: 无模型权重，仅 pseudo_embedding 伪嵌入（< 1MB RAM）—— 仅依赖缺失时兜底

真实模型（联网核实）:
  文本版: https://huggingface.co/Qwen/Qwen3.5-0.8B  （Qwen3.5 系列 0.8B，官方仓库）
  多模态版: https://huggingface.co/Qwen/Qwen3.5-0.8B-V（27 层 ViT 视觉编码器，规范 §3.3）
  量化参考: unsloth/Qwen3.5-0.8B-GGUF / mitslabo/Qwen3.5-0.8B-V-oQ4-fp16
  加载方式: transformers AutoModelForCausalLM + AutoTokenizer（标准 Qwen 用法）

职责:
  1. 文本 ↔ 向量（512 维）互译：encode() mean pooling（规范 §4.8），
     隐藏层 → 512 维固定随机投影（Johnson-Lindenstrauss 近似保余弦）
  2. 身份锚点向量生成 encode_anchor()，写入 cfg.anchor_vector（规范 §4.8）
  3. decode() 真实生成（chat 模板 + 温度采样）
  4. next_token_probs()：真实 logits softmax 路由信号（router.py 使用）
  5. encode_image()：Qwen3.5-0.8B-V 视觉编码（CPU 惰性加载，规范 §3.3）
  6. 路由层暂迁 CPU / 迁回 GPU 的预热与计时（规范 §4.1 / §九.6）

降级链（规范 §九.2）:
  torch/transformers 缺失 → from_pretrained 失败 → 模型文件缺失 →
  一律降级为 DEMO 伪嵌入/模板输出，并打印黄色警告（本模块保证任何环境可导入）。
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

import os
import random

import numpy as np

from config import get_config
from common_utils import (get_logger, Timer, safe_import, pseudo_embedding,
                          vram_summary, normalize)

# 模型名（与 config.qwen_model_name 一致；环境变量 PHILIA_MODEL 可覆盖）
# TODO(V3.4): 支持角色切换时替换为其它 Qwen 尺寸
MODEL_NAME = "Qwen/Qwen3.5-0.8B"

# DEMO 模式解码候选模板（昔涟风格，取材规范 §5.2 种子示例风格；仅流程验证用）
_DEMO_REPLIES = [
    "（眼睫轻垂）我在。有些话，我说了三千万世，还是想慢慢说。",
    "（指尖轻触桌面）风从翁法罗斯来，带着花海的安静。",
    "（语气极轻）我记得。安静，是因为它们在听你说话。",
    "（停顿良久）每一次告别，我都练习过很多遍，可还是会怕。",
    "（侧目望向窗外）不是为了守护世界。是为了守护那些说'明天见'的人。",
]


class Translator:
    """语义翻译器：REAL 用 Qwen3.5-0.8B（transformers，FP32，GPU 优先）；
    DEMO/失败降级为伪嵌入（最后兜底，规范 §六.1）。"""

    def __init__(self, cfg=None):
        self.cfg = cfg or get_config()
        self.logger = get_logger("translator")
        self._is_demo = bool(self.cfg.demo)
        self.model = None          # AutoModelForCausalLM（REAL）；DEMO 为 None
        self.tokenizer = None      # AutoTokenizer（REAL）；DEMO 为 None
        self.device = "cpu"        # 当前模型所在设备
        # 设备切换锁: 验证层 to_cpu/to_cuda 与主线程 decode/encode 并发时
        # 防止设备撕裂（"Expected all tensors to be on the same device"）
        self._device_lock = __import__("threading").Lock()
        self.hidden_dim = 512      # 模型隐藏维度（加载后从 config 读取）
        self._proj = None          # hidden → 512 固定随机投影矩阵
        # 视觉版（Qwen3.5-0.8B-V）按需惰性加载，强制 CPU（感知层不占显存）
        self._vision_model = None
        self._vision_processor = None

        if self._is_demo:
            self.logger.warning(
                "\033[33m[DEMO] 语义翻译器降级为伪嵌入（pseudo_embedding），仅流程验证。\033[0m")
        else:
            self._load_real()

        # 身份锚点（规范 §4.8）：初始化时生成并写入 cfg.anchor_vector
        self.encode_anchor()

    # ==================================================================
    # 模型加载 / 降级
    # ==================================================================
    def _load_real(self):
        """真实加载 Qwen3.5-0.8B（AutoModelForCausalLM，FP32，CUDA 优先）。

        显存自检（规范 §九.3②）:
          - cfg.use_4bit 且 bitsandbytes 可用 → load_in_4bit（[4BIT_MODE]）
          - cfg.use_amp → torch_dtype=float16（[AMP_MODE]）
          - 默认强制 FP32（规范 §十.1）
        任何失败 → _degrade() 降级 DEMO（规范 §九.2）。
        """
        torch = safe_import("torch")
        transformers = safe_import("transformers")
        if torch is None or transformers is None:
            self._degrade("torch 或 transformers 不可用")
            return
        try:
            model_name = self.cfg.qwen_model_name or MODEL_NAME
            # 本地模型优先（免联网）: 微调版 -ft > 原版（train_finetune_qwen.py 产物）
            ft_dir = os.path.join(self.cfg.model_cache_dir, "Qwen3.5-0.8B-ft")
            if os.path.isdir(ft_dir) and os.path.isfile(os.path.join(ft_dir, "config.json")):
                model_name = ft_dir
                self.logger.info("[BOOT] 使用微调模型目录: %s（LoRA 人设微调）", ft_dir)
            else:
                local_dir = os.path.join(self.cfg.model_cache_dir, "Qwen3.5-0.8B")
                if os.path.isdir(local_dir) and os.path.isfile(os.path.join(local_dir, "config.json")):
                    model_name = local_dir
                    self.logger.info(f"[BOOT] 使用本地模型目录: {local_dir}")
            self.logger.info(f"[BOOT] 加载真实语义翻译器 {model_name} ...")

            kwargs = {"trust_remote_code": True}
            if self.cfg.use_4bit:
                # [4BIT_MODE] 显存严重不足：bitsandbytes 4-bit 量化（规范 §二 降级）
                try:
                    kwargs.update(load_in_4bit=True, device_map="auto")
                    self.logger.warning("\033[33m[4BIT_MODE] 加载 4-bit 量化版本\033[0m")
                except Exception:
                    pass
            elif self.cfg.use_amp:
                # [AMP_MODE] 显存 < 5GB：自动混合精度（规范 §九.3②）
                kwargs["torch_dtype"] = torch.float16
                self.logger.warning("\033[33m[AMP_MODE] 启用混合精度（float16）\033[0m")
            else:
                kwargs["torch_dtype"] = torch.float32   # 强制 FP32（规范 §十.1）

            self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            self.model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            self.model.eval()

            # 隐藏维度（用于 →512 投影）
            try:
                self.hidden_dim = int(getattr(self.model.config, "hidden_size", 512))
            except Exception:
                self.hidden_dim = 512
            self._init_proj()

            # 设备选择：CUDA 优先（GTX 1060 6GB，规范 §二），无 GPU 走 CPU
            if torch.cuda.is_available() and self.device != "cuda":
                # 跨设备迁移 .to('cuda')：权重约 3.2GB，延迟 2~5s（规范 §九.5）
                self.model = self.model.to("cuda")
                self.device = "cuda"
            else:
                self.device = "cpu"
            self.logger.info(
                f"[BOOT] 真实语义翻译器就绪 device={self.device} "
                f"hidden={self.hidden_dim} {vram_summary()}")
        except Exception as e:
            self._degrade(f"真实模型加载失败: {e!r}")

    def _init_proj(self):
        """隐藏层 → 512 维固定随机投影（行单位向量，近似保持余弦相似度）。

        说明: 融合层契约要求 512 维（规范 §3.4）；0.8B 模型隐藏层通常 >512，
        使用固定种子高斯投影做降维，不参与训练，保证多次加载一致。
        """
        rng = np.random.RandomState(20240801)
        P = rng.normal(0.0, 1.0 / np.sqrt(self.hidden_dim),
                       (self.hidden_dim, self.cfg.hidden_proj_dim)).astype(np.float32)
        norms = np.linalg.norm(P, axis=1, keepdims=True) + 1e-12
        self._proj = P / norms
        self.logger.info(f"[PROJ] 隐藏层 {self.hidden_dim} → {self.cfg.hidden_proj_dim} 投影就绪")

    def _degrade(self, reason: str):
        """降级为 DEMO 伪嵌入，并打印黄色警告日志（规范 §六.1 / §九.2）。"""
        self._is_demo = True
        self.model = None
        self.tokenizer = None
        self.logger.warning(
            f"\033[33m[DEMO] 语义翻译器降级（{reason}），改用 pseudo_embedding 伪嵌入。\033[0m")

    @property
    def is_demo(self) -> bool:
        """是否处于 DEMO（伪嵌入）模式。"""
        return self._is_demo

    # ==================================================================
    # 设备一致性（验证层 to_cpu/to_cuda 与推理并发时的防撕裂）
    # ==================================================================
    def _cur_device(self):
        """模型当前实际设备（以参数所在设备为准，避免 self.device 快照过期）。"""
        if self.model is not None:
            try:
                return next(self.model.parameters()).device
            except Exception:
                pass
        return self.device

    # ==================================================================
    # 编码（文本 → 512 维）
    # ==================================================================
    def encode(self, texts) -> "np.ndarray":
        """文本列表 → (n, 512) 向量。

        REAL: AutoModelForCausalLM 前向 + output_hidden_states 取最后一层隐藏态
              → attention_mask 加权 mean pooling → 512 维投影（规范 §4.8）；
        DEMO/失败: pseudo_embedding 伪嵌入（规范 §六.1 降级边界）。
        """
        if isinstance(texts, str):
            texts = [texts]
        texts = [t or "" for t in texts]
        if not texts:
            return np.zeros((0, self.cfg.STATE_DIM), dtype=np.float32)

        if self._is_demo or self.model is None or self.tokenizer is None:
            return np.stack([pseudo_embedding(t, self.cfg.STATE_DIM) for t in texts])

        torch = safe_import("torch")
        try:
            inputs = self.tokenizer(
                texts, return_tensors="pt", padding=True, truncation=True,
                max_length=self.cfg.text_encode_max_len)
            with self._device_lock:                  # 推理与设备切换互斥（防撕裂）
                inputs = {k: v.to(self._cur_device()) for k, v in inputs.items()}
                with torch.no_grad():
                    out = self.model(**inputs, output_hidden_states=True)
                hs = out.hidden_states[-1]           # (n, L, H)
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                summed = (hs * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1.0)
                pooled = (summed / counts).detach().cpu().numpy().astype(np.float32)  # (n, H)
            vecs = pooled @ self._proj                            # (n, 512)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)   # 向量化归一（原逐行 normalize）
            vecs = np.where(norms > 1e-12, vecs / np.maximum(norms, 1e-12),
                            np.zeros_like(vecs))
            return vecs
        except Exception as e:
            self.logger.warning(f"\033[33m[DEMO] 编码失败，降级伪嵌入: {e!r}\033[0m")
            return np.stack([pseudo_embedding(t, self.cfg.STATE_DIM) for t in texts])

    # ==================================================================
    # 身份锚点（规范 §4.8）
    # ==================================================================
    def encode_anchor(self) -> "np.ndarray":
        """身份锚点向量（512 维）：anchor = Qwen.encode(anchor_generation_prompt).mean_pooling()。

        结果写入 cfg.anchor_vector，供路由层 / 验证层做余弦相似度比对
        （身份锚定阈值 anchor_similarity_threshold = 0.85，规范 §4.7）。
        """
        vec = self.encode([self.cfg.anchor_generation_prompt])[0]
        vec = vec / (float(np.linalg.norm(vec)) + 1e-12)          # 归一化为单位向量
        self.cfg.anchor_vector = vec
        self.logger.info(
            f"[ANCHOR] 身份锚点已生成 dim={vec.shape[0]} "
            f"norm={float(np.linalg.norm(vec)):.3f} "
            f"({'真实Qwen编码' if not self._is_demo else '伪嵌入'})")
        return vec

    # ==================================================================
    # 解码（生成文本）
    # ==================================================================
    def decode(self, logits_or_ids, persona: str = None, few_shots: list = None,
               temperature: float = 0.6) -> str:
        """解码为文本。

        REAL: 传入 str 提示词 → chat 模板（system 注入人设 persona + few-shot
              种子示例引导风格，人设锚定的"事前引导"）+ generate（温度采样）。
              传入 logits(1,L,V) → argmax；传入 ids → tokenizer.decode。
        DEMO: 返回昔涟风格候选模板文本（仅流程验证）。

        参数:
          persona  : system 人设文本（如 cfg.anchor_generation_prompt）；
                     提供则注入 chat 模板 system 层（防人设偏离的事前引导）
          few_shots: [{"role","content"}...] 示例对（风格 few-shot 引导）
        """
        if self._is_demo or self.model is None or self.tokenizer is None:
            return random.choice(_DEMO_REPLIES)

        torch = safe_import("torch")
        try:
            if isinstance(logits_or_ids, str):
                prompt = logits_or_ids
                # chat 模板（Qwen 系 instruct 模型标准用法）；失败则裸提示词
                try:
                    msgs = []
                    if persona:
                        msgs.append({"role": "system", "content": persona})
                    if few_shots:
                        msgs.extend(list(few_shots)[:6])   # few-shot 风格引导
                    msgs.append({"role": "user", "content": prompt})
                    text = self.tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True)
                except Exception:
                    text = prompt
                inputs = self.tokenizer(text, return_tensors="pt")
                with self._device_lock:              # 推理与设备切换互斥（防撕裂）
                    inputs = {k: v.to(self._cur_device()) for k, v in inputs.items()}
                    with torch.no_grad():
                        out_ids = self.model.generate(
                            **inputs, max_new_tokens=64, do_sample=True,
                            temperature=temperature)
                out_ids = out_ids[0][inputs["input_ids"].shape[1]:]
                reply = self.tokenizer.decode(out_ids, skip_special_tokens=True).strip()
                return self._strip_thinking(reply)

            ids = logits_or_ids
            if isinstance(ids, np.ndarray):
                if ids.ndim == 3:          # logits (1, L, V) → argmax
                    ids = ids.argmax(-1)[0]
                ids = ids.tolist()
            if isinstance(ids, torch.Tensor):
                if ids.ndim == 3:
                    ids = ids.argmax(-1)[0]
                ids = ids.tolist()
            return self._strip_thinking(
                self.tokenizer.decode(ids, skip_special_tokens=True).strip())
        except Exception as e:
            self.logger.warning(f"\033[33m[DEMO] 解码失败，降级模板: {e!r}\033[0m")
            return random.choice(_DEMO_REPLIES)

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """过滤 Qwen3.5 默认思考模式的 <think>...</think> 块（仅保留可见回复）。"""
        if not text:
            return text
        import re as _re
        cleaned = _re.sub(r"<think>.*?</think>", "", text, flags=_re.S)
        cleaned = _re.sub(r"</?think>", "", cleaned)
        return cleaned.strip()

    # ==================================================================
    # 真实路由信号：next_token_probs（router.py 使用，规范 §4.1）
    # ==================================================================
    def next_token_probs(self, prompt: str, candidates) -> dict:
        """对候选 token 做真实 softmax 路由概率（REAL）。

        参数:
          prompt    : 路由提示词（router 构造，如“请选择扩散管线：1/2/3”）
          candidates: 候选字符串列表，如 ["1", "2", "3"]

        返回:
          {"prob": {"1": 0.2, ...}, "confidence": 0.55, "argmax": "3"}
        DEMO/失败: 均匀概率 + confidence=0（router 回退启发式）。
        """
        n = len(candidates)
        if self._is_demo or self.model is None or self.tokenizer is None:
            return {"prob": {c: 1.0 / n for c in candidates},
                    "confidence": 0.0, "argmax": candidates[0] if candidates else None}
        torch = safe_import("torch")
        try:
            # 候选 token：取每个候选串的最后一个 token id（“1”/“2”/“3” 通常单 token）
            ids = []
            for c in candidates:
                t = self.tokenizer(str(c), add_special_tokens=False)["input_ids"]
                ids.append(t[-1] if t else self.tokenizer.convert_tokens_to_ids(str(c)))
            inputs = self.tokenizer(prompt, return_tensors="pt")
            with self._device_lock:              # 推理与设备切换互斥（防撕裂）
                inputs = {k: v.to(self._cur_device()) for k, v in inputs.items()}
                with torch.no_grad():
                    logits = self.model(**inputs).logits            # (1, L, V)
            last = logits[0, -1, :].float()                     # (V,)
            sel = last[torch.tensor(ids, device=last.device)]
            probs = torch.softmax(sel, dim=0).detach().cpu().numpy()
            prob_map = {str(c): float(p) for c, p in zip(candidates, probs)}
            argmax = candidates[int(np.argmax(probs))]
            return {"prob": prob_map, "confidence": float(np.max(probs)),
                    "argmax": argmax}
        except Exception as e:
            self.logger.warning(f"[ROUTE] next_token_probs 失败，返回均匀概率: {e!r}")
            return {"prob": {c: 1.0 / n for c in candidates},
                    "confidence": 0.0, "argmax": candidates[0] if candidates else None}

    # ==================================================================
    # 视觉编码（Qwen3.5-0.8B-V，CPU 惰性加载，规范 §3.3）
    # ==================================================================
    def encode_image(self, image) -> "np.ndarray":
        """图像 → 512 维视觉状态向量（规范 §3.3）。

        REAL: 惰性加载 Qwen3.5-0.8B-V（CPU，感知层不占 GPU 显存，规范 §二）
              → AutoProcessor + 视觉模型 → 最后一层隐藏态 mean pooling → 512 投影。
        DEMO/失败: 返回 512 维零向量并记日志（视觉缺失不阻塞主流程）。
        """
        if self._is_demo:
            self.logger.info("[VISUAL] DEMO 模式，视觉编码返回零向量")
            return np.zeros(self.cfg.VISUAL_DIM, dtype=np.float32)
        torch = safe_import("torch")
        transformers = safe_import("transformers")
        if torch is None or transformers is None:
            return np.zeros(self.cfg.VISUAL_DIM, dtype=np.float32)
        try:
            if self._vision_model is None:
                self._load_vision(transformers, torch)
            if self._vision_model is None:
                raise RuntimeError("Qwen3.5-0.8B-V 不可用")
            inputs = self._vision_processor(images=image, return_tensors="pt")
            inputs = {k: v.to("cpu") for k, v in inputs.items()}
            with torch.no_grad():
                out = self._vision_model(**inputs, output_hidden_states=True)
            hs = out.hidden_states[-1]                            # (1, L, H)
            pooled = hs.mean(dim=1).detach().cpu().numpy().astype(np.float32)  # (1, H)
            vec = normalize(pooled[0] @ self._proj)               # (512,)
            self.logger.info("[VISUAL] Qwen3.5-0.8B-V 编码完成 → 512 维")
            return vec
        except Exception as e:
            self.logger.warning(f"[VISUAL] 视觉编码失败，返回零向量: {e!r}")
            return np.zeros(self.cfg.VISUAL_DIM, dtype=np.float32)

    def _load_vision(self, transformers, torch):
        """惰性加载视觉模型 Qwen3.5-0.8B-V（CPU）。

        兼容多种 transformers 版本: 依次尝试 AutoModelForImageTextToText →
        AutoModel → Qwen2VLForConditionalGeneration。处理器 AutoProcessor。
        """
        name = self.cfg.qwen_vision_model_name
        # 本地视觉模型优先（ModelScope 下载到 models/ 目录）
        local_dir = os.path.join(self.cfg.model_cache_dir, "Qwen3.5-0.8B-V")
        if os.path.isdir(local_dir) and os.path.isfile(os.path.join(local_dir, "config.json")):
            name = local_dir
            self.logger.info(f"[VISUAL] 使用本地视觉模型目录: {local_dir}")
        self.logger.info(f"[VISUAL] 加载 {name}（CPU，感知层不占 GPU 显存）...")
        try:
            proc = transformers.AutoProcessor.from_pretrained(name)
        except Exception:
            proc = None
        model = None
        for cls_name in ("AutoModelForImageTextToText", "AutoModel"):
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                continue
            try:
                model = cls.from_pretrained(name, torch_dtype=torch.float32)
                break
            except Exception:
                continue
        if model is None:
            try:
                qvl = getattr(transformers, "Qwen2VLForConditionalGeneration", None)
                if qvl is not None:
                    model = qvl.from_pretrained(name, torch_dtype=torch.float32)
            except Exception:
                model = None
        if model is None:
            self.logger.warning(f"[VISUAL] {name} 加载失败（需联网下载模型文件）")
            return
        model.eval()
        self._vision_model = model
        self._vision_processor = proc
        self.logger.info(f"[VISUAL] {name} 就绪 (CPU)")

    # ==================================================================
    # 路由层暂迁 CPU / 迁回 GPU（规范 §4.1 / §九.5 / §九.6）
    # ==================================================================
    def to_cpu(self):
        """路由层暂迁 CPU：释放显存供验证层/批判层按需加载。DEMO/CPU 机器为 no-op。

        计时 > cfg.router_cpu_switch_warn_s（3 秒）触发警告日志（规范 §九.6）。
        """
        if self._is_demo or self.model is None:
            return
        torch = safe_import("torch")
        if torch is None or self.device != "cuda":
            return
        with self._device_lock:                      # 设备切换加锁（防并发撕裂）
            with Timer("router.to_cpu") as t:
                # 跨设备迁移 .to('cpu')：权重约 3.2GB，延迟约 2~3 秒（规范 §九.5）
                self.model = self.model.to("cpu")
                self.device = "cpu"
                # 迁移完成后立即释放显存（此路径无顺序约束；“先预热后清缓存”约束仅针对迁回路径）
                torch.cuda.empty_cache()
        if t.elapsed / 1000.0 > self.cfg.router_cpu_switch_warn_s:
            self.logger.warning(
                f"\033[33m[ROUTER] 路由层迁 CPU 耗时 {t.elapsed/1000.0:.1f}s "
                f"> {self.cfg.router_cpu_switch_warn_s}s 阈值（规范 §九.6）\033[0m")
        else:
            self.logger.info(f"[ROUTER] 路由层已迁 CPU（{t.elapsed/1000.0:.2f}s）")

    def to_cuda(self):
        """路由层迁回 GPU：先空推理（dummy forward）预热 CUDA 内核，再 empty_cache，
        顺序不可颠倒（规范 §4.1 / §九.6）。DEMO/CPU 机器为 no-op。"""
        if self._is_demo or self.model is None:
            return
        torch = safe_import("torch")
        if torch is None or not torch.cuda.is_available():
            return
        with self._device_lock:                      # 设备切换加锁（防并发撕裂）
            with Timer("router.to_cuda") as t:
                # 跨设备迁移 .to('cuda')：权重约 3.2GB，延迟约 2~3 秒（规范 §九.5）
                self.model = self.model.to("cuda")
                self.device = "cuda"
                # ① 先执行一次空推理，预热 CUDA 内核（dummy forward）
                pad_id = self.tokenizer.pad_token_id if self.tokenizer is not None else 0
                dummy = torch.full((1, 4), pad_id or 0, dtype=torch.long, device=self._cur_device())
                with torch.no_grad():
                    self.model(dummy)
                # ② 预热完成后再清缓存 —— 顺序不可颠倒（规范 §4.1 / §九.6）
                torch.cuda.empty_cache()
        if t.elapsed / 1000.0 > self.cfg.router_cpu_switch_warn_s:
            self.logger.warning(
                f"\033[33m[ROUTER] 路由层迁回 GPU 耗时 {t.elapsed/1000.0:.1f}s "
                f"> {self.cfg.router_cpu_switch_warn_s}s 阈值（规范 §九.6）\033[0m")
        else:
            self.logger.info(
                f"[ROUTER] 路由层已迁回 GPU（{t.elapsed/1000.0:.2f}s）{vram_summary()}")
