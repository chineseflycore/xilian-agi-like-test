# -*- coding: utf-8 -*-
"""
minimind_bridge.py — 昔涟AGI v8.0 MiniMind-O 桥接层
================================================================================
显存预估: MiniMind Thinker (~113M, hidden=768, 8 层 dense) 以 FP16 常驻约 0.26GB，
         4-bit NF4 更省；本模块仅负责编码，不维护长 KV Cache。

【职责】
  把用户文本编码成 768 维 Z 向量，供昔涟认知引擎（匹配器/记忆池）使用。
  MiniMind-O 的 Thinker（MiniMindModel, hidden=768）作为前端编码器——
  相比 v7.3 的 Qwen3.5-0.8B，体积从 0.8B 降到 0.1B，同时维度从 384 升到 768。

【双后端支持】
  - encoding_backend == "minimind_o"（v8.0 默认）：用 MiniMind Thinker 编码
  - encoding_backend == "qwen"（v7.3 回退）：用 Qwen0.8B + head.pt 编码
  统一由 build_encoder(cfg) 按配置选择，返回 callable: text -> Z 向量。

【文本编码路径】
  tokenize -> 经 Thinker forward 取 hidden_states[-1] -> mean-pool -> L2 归一
  → 768 维 Z 向量（与 MiniMind-O Thinker 隐空间同域，供 Projector 对齐）。

【加载策略】
  优先使用 transformers 目录格式（config.json + 权重），用
  AutoModelForCausalLM.from_pretrained 加载。MiniMind 权重可从 ModelScope 下载
  （gongjy/minimind-3o-pytorch 的 llm_768.pth 或 HF transformers 目录 minimind-3o）。

【注意】
  - MiniMind tokenizer 使用特殊 chat 标记（<|im_start|> 等），编码原始文本即可。
  - hidden_states 在 Transformers v5 下经 output_hidden_states=True 返回。
"""
from __future__ import annotations

import os
import logging
import numpy as np

import torch

log = logging.getLogger("xilian.minimind")


# ---------------------------------------------------------------------------
# MiniMind 编码器封装
# ---------------------------------------------------------------------------
class MiniMindEncoder:
    """MiniMind Thinker 文本编码器：text -> 768 维 Z 向量。"""

    def __init__(self, model, tokenizer, device: str = "cpu"):
        self.model = model
        # MiniMindForCausalLM 让 self.model(MiniMindModel) 返回 (hidden, pres, aux_loss) 元组，
        # 取 hidden = [B, T, D]。直接访问底层 MiniMindModel 避免 GenerationMixin 包装。
        self.thinker = getattr(model, "thinker", getattr(model, "model", model))
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def encode(self, text: str) -> np.ndarray:
        """文本 -> 768 维 Z（L2 归一化）。使用 thinker(MiniMindModel) 的 hidden。"""
        if self.thinker is None or self.tokenizer is None:
            raise RuntimeError("MiniMind 模型未加载")
        ids = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
        ).to(self.device)
        hidden = self.thinker(
            **ids,
            use_cache=False,
        )                          # MiniMindModel.forward -> (hidden[B,T,D], presents, aux)
        # hs = 第 0 个样本的 [T, D]
        if isinstance(hidden, (tuple, list)):
            hidden = hidden[0]     # [B, T, D]
        hs = hidden[0].float() if hidden.dim() == 3 else hidden.float()
        # hs 此时是 [T, D]
        mean = hs.mean(dim=0)       # mean-pool 抗截断
        norm = torch.linalg.norm(mean)
        z = torch.zeros_like(mean) if float(norm) < 1e-9 else mean / norm
        return z.cpu().numpy().astype(np.float32).reshape(-1)

    def get_hidden_dim(self) -> int:
        """返回模型 hidden 维度（通常是 768）。"""
        cfg = getattr(self.model, "config", None)
        if cfg is not None and hasattr(cfg, "hidden_size"):
            return int(cfg.hidden_size)
        return 768


# ---------------------------------------------------------------------------
# 加载 MiniMind（transformers 目录格式）
# ---------------------------------------------------------------------------
def _find_minimind_dir(cfg) -> str | None:
    """在候选目录中定位含 config.json 的 MiniMind transformers 目录。"""
    cands = []
    base = getattr(cfg, "minimind_o_dir", "")
    if base:
        cands.append(base)
    # minimind-3o 可能直接放在 models/ 下
    extra = os.path.join(getattr(cfg, "models_dir", ""), "minimind-3o")
    if os.path.normpath(extra) != os.path.normpath(base):
        cands.append(extra)
    # 也允许非本地目录（远程加载由 fetch 负责，这里只认本地）
    for d in cands:
        if d and os.path.isdir(d) and os.path.isfile(os.path.join(d, "config.json")):
            return d
    return None


def load_minimind_encoder(cfg):
    """加载 MiniMind 文本编码器，返回 MiniMindEncoder。

    关键：不用 AutoModelForCausalLM（其 remote code 会指向 MiniMindOmni，
    触发 funasr/onnxruntime 强依赖，在当前 torch 2.7 环境会 pyarrow segfault）。
    改用 MiniMindForCausalLM（纯文本 Thinker，权重以 model.* 开头，
    与下载的 pytorch_model.bin 匹配），直接 from_pretrained。
    """
    model_dir = _find_minimind_dir(cfg)
    device = getattr(cfg, "device", "cpu")
    if model_dir is None:
        model_dir = getattr(cfg, "minimind_o_dir", "")
        if not model_dir:
            raise FileNotFoundError("未找到 MiniMind 模型目录 (cfg.minimind_o_dir)")
        log.info("[MiniMind] 尝试下载 minimind-3o -> %s", model_dir)
        model_dir = _auto_download(cfg, model_dir, device)

    # 把 model_minimind.py（纯文本版）所在目录加入 sys.path，绕过 Omni remote code。
    # 优先用项目 core/ 内的自带定义（会被 git 跟踪），模型目录为兜底。
    import sys
    from .model_minimind import MiniMindForCausalLM

    from transformers import AutoTokenizer

    # tokenizer 用标准路径（trust_remote_code=False），避免触发 Omni auto_map→funasr
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, local_files_only=True, trust_remote_code=False, use_fast=False)
    model = MiniMindForCausalLM.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.float16 if str(device).startswith("cuda") else torch.float32,
    )
    model.eval()
    model.to(device)
    if str(device).startswith("cuda"):
        model.half()

    hidden = getattr(getattr(model, "config", None), "hidden_size", 768)
    layers = getattr(getattr(model, "config", None), "num_hidden_layers", 8)
    log.info("[MiniMind] 加载成功(纯文本Thinker): %s (hidden=%d, layers=%d)",
             model_dir, hidden, layers)
    return MiniMindEncoder(model, tokenizer, device)


def _auto_download(cfg, target_dir: str, device: str):
    """找不到本地目录时，尝试用 huggingface_hub / modelscope 下载 minimind-3o。

    以「对齐开源版「能跑起来」为首要目标：把远程 minimind-3o 拉成 transformers
    目录。若下载失败则抛异常，调用方降级规则路径。
    """
    repo = "jingyaogong/minimind-3o"
    os.makedirs(target_dir, exist_ok=True)
    # 优先 huggingface_hub（本地已有 .hfcache 缓存）
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(
            repo_id=repo,
            local_dir=target_dir,
            local_dir_use_symlinks=False,
        )
        log.info("[MiniMind] HF 下载完成: %s", path)
        return path
    except Exception as e:
        log.warning("[MiniMind] HF 下载失败: %s，尝试 ModelScope", str(e)[:120])
    # 降级 modelscope
    try:
        from modelscope import snapshot_download as ms_download

        path = ms_download("gongjy/minimind-3o", local_dir=target_dir)
        log.info("[MiniMind] ModelScope 下载完成: %s", path)
        return path
    except Exception as e:
        raise FileNotFoundError(
            f"MiniMind 下载失败（HF 与 ModelScope 均不可用）: {str(e)[:120]}"
        ) from e


# ---------------------------------------------------------------------------
# 加载 Qwen 编码器（v7.3 旧桥，回退用）
# ---------------------------------------------------------------------------
def load_qwen_encoder(cfg):
    """加载 v7.3 Qwen0.8B + head.pt 编码器（XILIAN_ENCODER=qwen 时用）。

    返回 callable: text -> Z 向量（保持 v7.3 行为一致）。
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    head_path = os.path.join(getattr(cfg, "qwen_encoder_dir", ""), "head.pt")
    model_dir = None
    for d in (getattr(cfg, "qwen_encoder_dir", ""), getattr(cfg, "legacy_qwen_dir", "")):
        if d and os.path.isdir(d) and os.path.isfile(os.path.join(d, "config.json")):
            model_dir = d
            break
    if model_dir is None or not os.path.isfile(head_path):
        raise FileNotFoundError("qwen_encoder/head.pt 或模型目录缺失")

    device = getattr(cfg, "device", "cpu")
    load4 = cfg.has_bitsandbytes() if hasattr(cfg, "has_bitsandbytes") else False
    qcfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=qcfg if load4 else None,
        device_map=dict(getattr(cfg, "qwen_device_map", {"": 0})) if load4 else None,
        torch_dtype=torch.float16,
        local_files_only=True,
    )
    model.eval()
    st = torch.load(head_path, map_location="cpu", weights_only=False)
    w = st.get("project", st.get("weight"))
    mean = st.get("mean", None)
    w_t = torch.as_tensor(np.asarray(w, dtype=np.float32), device=model.device)
    mean_t = (
        torch.as_tensor(np.asarray(mean, dtype=np.float32), device=model.device)
        if mean is not None else None
    )
    log.info("[Qwen] 编码桥就绪: project %s (语料 %d, 方差 %.0f%%)",
             tuple(w_t.shape), int(st.get("samples", 0)),
             float(st.get("explained", 0)) * 100)

    @torch.no_grad()
    def encode(text: str):
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=256).to(model.device)
        h = model(**ids, output_hidden_states=True).hidden_states[-1][0].float()
        h = h.mean(dim=0)
        if mean_t is not None:
            h = h - mean_t
        z = (w_t @ h.unsqueeze(1)).squeeze(1)      # [384]
        n = torch.linalg.norm(z)
        z = torch.zeros_like(z) if float(n) < 1e-9 else z / n
        return z.cpu().numpy().astype(np.float32).reshape(-1)

    return encode


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------
def build_encoder(cfg):
    """按 cfg.encoding_backend 构建编码器，返回 callable: text -> Z 向量。

    - minimind_o（v8.0 默认）：MiniMind Thinker 编码
    - qwen（v7.3 回退）：Qwen0.8B + head.pt
    """
    backend = getattr(cfg, "encoding_backend", "minimind_o").lower()
    if backend == "qwen":
        log.info("[MiniMind] 编码后端 qwen（v7.3 回退）")
        return load_qwen_encoder(cfg)
    if backend != "minimind_o":
        log.warning("[MiniMind] 未知后端 %r，回退 minimind_o", backend)
    enc = load_minimind_encoder(cfg)
    log.info("[MiniMind] 编码后端 minimind_o（v8.0）")
    return enc.encode
