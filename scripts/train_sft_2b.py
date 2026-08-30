# -*- coding: utf-8 -*-
"""
scripts/train_sft_2b.py — Qwen3.5-2B 解码器特训微调（QLoRA SFT）
================================================================
职责: 用 knowledge/xilian_copy.json（684 条昔涟对话）对 2B 解码器做
  4-bit NF4 基座 + LoRA 适配器的对话微调（"特训"）:
  - 基座不更新（4-bit NF4 常驻显存, 峰值预算 <4.5GB）
  - LoRA r=8, alpha=16, dropout=0.05（config.sft_* 全可调）
  - 训练数据: messages=[system(人设精华), user(伙伴), assistant(昔涟)]
  - 验证集 10% 参照 loss; 训练完成后仅保存 LoRA 适配器 → models/qwen_decoder/lora_adapter
  - 运行期 QwenDecoder 检测到适配器自动挂载（PeftModel, 不训练）

用法: Python\\python.exe scripts\\train_sft_2b.py [--epochs N] [--force]
显存模型（GTX 1060 6GB）: 2B 4-bit ≈1.3GB + LoRA ≈40MB + 优化器 ≈300MB
  + 激活/梯度 ≈0.5GB ≈ 峰值 2.2GB（叠加匹配器等 0.4GB 后可运行, 建议单独运行本脚本）。
"""
import os
import sys
import time

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

import config
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig)

cfg = config.get_config()
cfg.setup_logging()
log = cfg.setup_logging().getChild("sft")


def build_dataset(cfg):
    """xilian_copy.json → 训练样本列表: [{"messages": [...]}]*N（按 sft_val_ratio 切分）。"""
    import json as _json
    with open(cfg.diffuser_train_data, "r", encoding="utf-8") as f:
        items = _json.load(f)
    persona = cfg.persona_system[:800] or "你是昔涟。"   # 训练用精简人设（运行时仍用 2800 字精华）
    samples = []
    for it in items:
        samples.append({
            "messages": [
                {"role": "system", "content": persona},
                {"role": "user", "content": str(it.get("user", ""))[:800]},
                {"role": "assistant", "content": str(it.get("xilian", ""))[:800]},
            ]})
    # 确定性切分（验证集 10%）
    import numpy as np
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(samples))
    n_val = max(1, int(len(samples) * cfg.sft_val_ratio))
    val_ids = set(perm[:n_val].tolist())
    train = [s for i, s in enumerate(samples) if i not in val_ids]
    val = [s for i, s in enumerate(samples) if i in val_ids]
    return train, val


def tokenize_sample(tokenizer, sample, max_len):
    """单样本 → (input_ids, labels); assistant 段之外的 token 置 -100（CE 掩码）。

    注意: apply_chat_template(tokenize=True) 部分版本返回 dict/Encoding（len 恒为键数），
    因此统一走「模板渲染 → 手动 tokenize(input_ids)」的稳健路径。
    """
    sys_msg, usr_msg, asst_msg = sample["messages"]
    pre_text = tokenizer.apply_chat_template([sys_msg, usr_msg], tokenize=False,
                                             add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template([sys_msg, usr_msg, asst_msg], tokenize=False,
                                              add_generation_prompt=False)
    pre = tokenizer(pre_text, add_special_tokens=False).input_ids
    full = tokenizer(full_text, add_special_tokens=False).input_ids
    full = full[:max_len]
    n_pre = len(pre)
    labels = [-100] * n_pre + full[n_pre:] if len(full) > n_pre else [-100] * len(full)
    return full, labels


def main(argv):
    force = "--force" in argv
    use_4bit = "--4bit" in argv                # 默认 FP16（Pascal 上 4-bit 反量化极慢, 3~8 倍差距）
    epochs = int(argv[argv.index("--epochs") + 1]) if "--epochs" in argv else cfg.sft_epochs
    adapter_dir = cfg.sft_adapter_dir
    if os.path.isdir(adapter_dir) and not force:
        log.info("[SFT] 适配器已存在 %s（--force 重训）", adapter_dir)
        return 0

    # 1) 数据
    train, val = build_dataset(cfg)
    log.info("[SFT] 数据: 训练 %d / 验证 %d 条", len(train), len(val))

    # 2) 基座（默认 FP16 直载; --4bit 走 NF4 量化路径）
    model_dir = os.path.join(cfg.models_dir, "qwen_decoder")
    if not os.path.isfile(os.path.join(model_dir, "config.json")):
        log.error("[SFT] 2B 权重未就绪: %s（先下载）", model_dir)
        return 1
    log.info("[SFT] 加载基座 %s（%s）…", model_dir, "4-bit NF4" if use_4bit else "FP16")
    tok = AutoTokenizer.from_pretrained(model_dir)
    if use_4bit:
        qcfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                  bnb_4bit_use_double_quant=False,
                                  bnb_4bit_compute_dtype=torch.float16)
        base = AutoModelForCausalLM.from_pretrained(
            model_dir, quantization_config=qcfg, device_map=cfg.qwen_device_map)
    else:
        base = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.float16, device_map=cfg.qwen_device_map)
    base.gradient_checkpointing_enable()       # 反向重算前向 → 显存大幅压缩（fp16 2B 必需）
    base.config.use_cache = False

    # 3) LoRA 挂载
    try:
        from peft import LoraConfig, get_peft_model
    except Exception as e:
        log.error("[SFT] peft 不可用(%s) → 安装: pip install peft", e)
        return 1
    lora_cfg = LoraConfig(r=cfg.sft_lora_r, lora_alpha=cfg.sft_lora_alpha,
                          lora_dropout=cfg.sft_lora_dropout,
                          bias="none", task_type="CAUSAL_LM",
                          target_modules="all-linear")
    model = get_peft_model(base, lora_cfg)
    if not use_4bit:
        model.enable_input_require_grads()     # checkpoint + LoRA 输入梯度要求
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("[SFT] LoRA 可训练参数 %.1fM (r=%d, 全线性层)", trainable / 1e6, cfg.sft_lora_r)

    # 4) 训练循环（单卡、梯度累积、FP16 GradScaler、验证 loss）
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=cfg.sft_lr, weight_decay=0.01)
    scaler = None if use_4bit else torch.cuda.amp.GradScaler()
    steps_per_epoch = max(1, (len(train) + cfg.sft_accum - 1) // cfg.sft_accum)
    log.info("[SFT] 每 epoch %d 步 (accum=%d) → 开始训练…", steps_per_epoch, cfg.sft_accum)
    t0 = time.time()
    best_val = float("inf")

    def _loss_fn(sample):
        ids, labels = tokenize_sample(tok, sample, cfg.sft_max_len)
        if len(ids) < 8:
            return None
        dev = next(model.parameters()).device
        ids_t = torch.as_tensor([ids], device=dev)
        lab_t = torch.as_tensor([labels], device=dev)
        return model(input_ids=ids_t, labels=lab_t).loss

    for epoch in range(epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        total, n = 0.0, 0
        for si, sample in enumerate(train):
            loss = _loss_fn(sample)
            if loss is None:
                continue
            loss = loss / cfg.sft_accum
            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()
            total += float(loss.detach().cpu()) * cfg.sft_accum
            n += 1
            if (si + 1) % cfg.sft_accum == 0:
                if scaler is None:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0)
                    opt.step()
                else:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], 1.0)
                    scaler.step(opt)
                    scaler.update()
                opt.zero_grad(set_to_none=True)
            if (si + 1) % 40 == 0:
                log.info("[SFT] epoch%d 步%d/%d loss=%.4f (%.0fs)", epoch + 1, si + 1,
                         len(train), total / max(1, n), time.time() - t0)
        if n % cfg.sft_accum != 0:
            if scaler is None:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
            else:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt)
                scaler.update()
            opt.zero_grad(set_to_none=True)
        # 验证
        model.eval()
        with torch.no_grad():
            val_loss, val_n = 0.0, 0
            for sample in val:
                loss = _loss_fn(sample)
                if loss is not None:
                    val_loss += float(loss.cpu())
                    val_n += 1
        avg_val = val_loss / max(1, val_n)
        log.info("[SFT] epoch%d 完成: 训练loss=%.4f 验证loss=%.4f (%.1fs)",
                 epoch + 1, total / max(1, n), avg_val, time.time() - t0)
        best_val = min(best_val, avg_val)

    # 5) 保存适配器（仅 LoRA, 微; 基座不动）
    os.makedirs(adapter_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tok.save_pretrained(adapter_dir)
    log.info("[SFT] 完成: 适配器 → %s (验证loss=%.4f, 耗时%.1fs) %s",
             adapter_dir, best_val, time.time() - t0,
             "（覆盖运行期将自动挂载）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
