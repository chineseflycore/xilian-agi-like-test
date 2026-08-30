# -*- coding: utf-8 -*-
"""
train_finetune_qwen.py — Qwen3.5-0.8B LoRA 微调（人设 SFT，GTX 1060 6GB 快速版）
==================================================================================
内存/显存预估:
  - 基础模型 Qwen3.5-0.8B FP32 ≈ 3.2GB（加载后冻结）
  - LoRA 低秩参数（r=16）×4 类投影 ≈ 数 MB + 优化器状态 ≈ 数十 MB
  - AMP fp16 激活（batch 4 × ~150 token）≈ < 0.5GB
  - 总显存 ≈ 3.8~4.2GB（GTX 1060 6GB 预算内，规范 §二）

数据: data_cache/训练资料*.txt（ChatML 对话，最多 1030 条）
方法: LoRA（peft）+ 冻结基础模型 + AMP fp16 + AdamW，1~3 epoch
产物: models/Qwen3.5-0.8B-ft/（合并后的微调模型，translator 本地优先加载）

运行:
  Python\\python.exe train_finetune_qwen.py
  PHILIA_FT_EPOCHS=2   # 轮数（默认 2，快）
  PHILIA_FT_BATCH=4    # 批大小（默认 4）
"""

import os
import sys
import re
import glob
import json
import time

_BASE = os.path.dirname(os.path.abspath(__file__))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)
try:
    import numpy  # noqa: F401
except ImportError:
    _V = os.path.join(_BASE, "vendor")
    if os.path.isdir(_V) and _V not in sys.path:
        sys.path.insert(0, _V)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from common_utils import get_logger

logger = get_logger("finetune")

DATA_DIR = os.path.join(config.BASE_DIR, "data_cache")
SRC_MODEL = os.path.join(config.BASE_DIR, "models", "Qwen3.5-0.8B")
OUT_MODEL = os.path.join(config.BASE_DIR, "models", "Qwen3.5-0.8B-ft")


def parse_chatml(path: str) -> list:
    """解析 ChatML 文件 → [(user, assistant)] 对话对。"""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()
    pat = re.compile(
        r"<\|im_start\|>user\s*(.*?)<\|im_end\|>.*?<\|im_start\|>assistant\s*(.*?)<\|im_end\|>",
        re.S)
    pairs = []
    for m in pat.finditer(raw):
        u = m.group(1).strip()
        a = m.group(2).strip()
        if u and a and len(u) > 2 and len(a) > 2:
            pairs.append((u, a))
    return pairs


def build_dataset() -> list:
    """扫描 训练资料*.txt → messages 三元组 [system, user, assistant]。"""
    system = config.get_config().anchor_generation_prompt
    files = sorted(glob.glob(os.path.join(DATA_DIR, "训练资料*.txt")))
    samples = []
    for f in files:
        pairs = parse_chatml(f)
        if pairs:
            samples.append((os.path.basename(f), len(pairs)))
            for u, a in pairs:
                samples.append({"system": system, "user": u, "assistant": a})
    # 去重（按 user 文本）
    seen, uniq = set(), []
    for s in samples:
        if isinstance(s, tuple):
            continue
        key = s["user"][:50]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(s)
    logger.info("[FT] 数据: %d 个文件, %d 条对话（去重后 %d）", len(files), len(samples) - len(files), len(uniq))
    return uniq


def tokenize_sample(tokenizer, sample, max_len: int = 512) -> dict:
    """messages → input_ids/labels（chat 模板 + 只对 assistant 计算 loss）。

    注意: assistant 内容含 <think> 块，序列较长；max_len 需足够容纳，
    截断后若 assistant 起点超出长度则丢弃该样本（防 labels 全 -100 塌缩）。
    """
    msgs = [{"role": "system", "content": sample["system"]},
            {"role": "user", "content": sample["user"]},
            {"role": "assistant", "content": sample["assistant"]}]
    enc = tokenizer.apply_chat_template(msgs, tokenize=True,
                                        return_dict=True, add_generation_prompt=False)
    input_ids = enc["input_ids"]
    # assistant 内容起点 = 只到 user 的模板（add_generation_prompt 含 <think> 块）长度
    pre = tokenizer.apply_chat_template(
        [{"role": "system", "content": sample["system"]},
         {"role": "user", "content": sample["user"]}],
        tokenize=True, add_generation_prompt=True)
    pre_len = len(pre)
    if pre_len >= len(input_ids):
        return None
    input_ids = input_ids[:max_len]
    if pre_len >= len(input_ids):
        return None                       # 截断后 assistant 起点丢失 → 丢弃
    labels = [-100] * len(input_ids)
    for i in range(pre_len, len(input_ids)):
        labels[i] = input_ids[i]
    return {"input_ids": input_ids, "labels": labels}


def main() -> int:
    t0 = time.time()
    cfg = config.get_config()
    if not os.path.isfile(os.path.join(SRC_MODEL, "config.json")):
        logger.error("[FT] 基础模型缺失: %s", SRC_MODEL)
        return 1
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        logger.error("[FT] peft 未安装，请先: Python\\python.exe -m pip install peft")
        return 1

    epochs = int(os.environ.get("PHILIA_FT_EPOCHS", "3"))
    batch = int(os.environ.get("PHILIA_FT_BATCH", "2"))
    logger.info("=" * 60)
    logger.info("[FT] Qwen3.5-0.8B LoRA 微调 开始（epochs=%d batch=%d）", epochs, batch)

    # ① 数据
    samples = build_dataset()
    if not samples:
        logger.error("[FT] 无训练数据，终止")
        return 1

    # ② 加载模型（fp16 减半显存 → GTX 1060 6GB 预算；AMP 训练）
    logger.info("[FT] 加载基础模型 %s（fp16）...", SRC_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        SRC_MODEL, torch_dtype=torch.float16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(SRC_MODEL, trust_remote_code=True)
    model.to("cuda")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False          # 全冻结（LoRA 才可训练）

    # ③ LoRA 注入
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("[FT] LoRA 可训练参数: %d (%.2f MB)", trainable, trainable * 4 / 1048576)
    model.train()

    # ④ 数据 → token（并统计 labels 覆盖，诊断用）
    data = [t for t in (tokenize_sample(tokenizer, s) for s in samples) if t]
    if not data:
        logger.error("[FT] 全部样本被截断丢弃（序列过长），调大 max_len")
        return 1
    cover = sum(1 for d in data if sum(1 for l in d["labels"] if l != -100) >= 10)
    logger.info("[FT] 有效样本 %d（labels 覆盖 ≥10 token 的 %d 个）", len(data), cover)

    # ⑤ 训练（AMP fp16，手写循环，快）
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
    total_steps = 0
    for ep in range(epochs):
        order = torch.randperm(len(data))
        ep_loss = 0.0
        for i in range(0, len(data), batch):
            idx = order[i:i + batch]
            ids = [data[j]["input_ids"] for j in idx.tolist()]
            lbs = [data[j]["labels"] for j in idx.tolist()]
            maxl = max(len(x) for x in ids)
            pad_id = tokenizer.pad_token_id or 0
            inp = torch.full((len(ids), maxl), pad_id, dtype=torch.long)
            lab = torch.full((len(ids), maxl), -100, dtype=torch.long)
            for k, (x, y) in enumerate(zip(ids, lbs)):
                inp[k, :len(x)] = torch.tensor(x, dtype=torch.long)
                lab[k, :len(y)] = torch.tensor(y, dtype=torch.long)
            attn = (inp != pad_id).long()
            inp, lab, attn = inp.cuda(), lab.cuda(), attn.cuda()
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                out = model(input_ids=inp, attention_mask=attn).logits
                loss = loss_fn(out.reshape(-1, out.size(-1)), lab.reshape(-1))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ep_loss += float(loss.item()) * len(idx)
            total_steps += 1
        logger.info("[FT] epoch %d/%d loss=%.4f（%.0fs）",
                    ep + 1, epochs, ep_loss / max(len(data), 1), time.time() - t0)

    # ⑥ 合并 LoRA 并保存（先清理旧微调目录，防文件残留）
    logger.info("[FT] 合并 LoRA → %s ...", OUT_MODEL)
    import shutil
    if os.path.isdir(OUT_MODEL):
        shutil.rmtree(OUT_MODEL, ignore_errors=True)
    model = model.merge_and_unload()
    model.save_pretrained(OUT_MODEL)
    tokenizer.save_pretrained(OUT_MODEL)

    # ⑦ 脚本内验证（生成 2 条，若空/塌缩 → 提示回退原版）
    logger.info("[FT] 验证生成 ...")
    model.eval()
    for q in ("你好，昔涟。", "昔涟，你还记得花海吗？"):
        msgs = [{"role": "system", "content": cfg.anchor_generation_prompt},
                {"role": "user", "content": q}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=48,
                                 do_sample=True, temperature=0.6)
        ids = out[0][inputs["input_ids"].shape[1]:]
        reply = tokenizer.decode(ids, skip_special_tokens=True).strip()
        logger.info("[FT] 验证: %r → %r", q[:12], reply[:60])
        if not reply or set(reply) <= {"\n", " ", "\t"}:
            logger.warning("\033[33m[FT] 生成塌缩（仅空白）—— 请回退原版: "
                           "删除 models/Qwen3.5-0.8B-ft\033[0m")

    logger.info("[FT] 微调完成，已保存 → %s（总耗时 %.0fs）", OUT_MODEL, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
