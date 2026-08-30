# -*- coding: utf-8 -*-
"""
scripts/train_encoder_head.py — 编码头训练（昔涟语料, Qwen3.5-0.8B 冻结萃取）
=============================================================================
显存预估: 0.8B 4-bit NF4 ≈ 0.6GB + 激活 ≈ 0.5GB ≈ 峰值 1.2GB（< 4.5GB ✓）

职责: 用 knowledge/xilian_copy.json（684 条昔涟对话）训练
  models/qwen_encoder/head.pt —— 把 Qwen3.5-0.8B 的 hidden(1024 维) 投影为
  v7.3 认知架构的 384 维潜向量 Z（编码器"转译模型"的头部, 与匹配器同表示域）。

训练方法（两道工序, 全部由语料驱动）:
  工序 A【特征萃取】: 0.8B 冻结（4-bit NF4, 不更新）→ 每条文本取最后一层 hidden
    的 token 均值池化（mean-pool, 抗截断/编码稳定）→ 特征矩阵 [N, 1024]。
  工序 B【PCA 投影】: 对特征矩阵中心化 → 协方差主成分分解 → 取前 384 个主成分
    构成线性投影 W ∈ R^{384×1024}（最大化语料方差的低维语义压缩; 训练即"用
    语料估计协方差", 无随机性, 可复现）→ 报告解释方差与对规则 Z 的余弦一致性。

运行: python scripts\\train_encoder_head.py [--epochs A] [--whiten] [--force]
  输出: models/qwen_encoder/head.pt  {"version","project":W,"mean","hidden_dim",
        "z_dim","samples","explained","trained"}
  运行时: core/agi_core._try_load_encoder 读取该文件 → z = L2(W @ (h-mean))
"""
import os
import sys
import time
import json
import argparse

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)

import numpy as np
import torch

import config
from core import protocol

cfg = config.get_config()
cfg.setup_logging()
log = cfg.setup_logging().getChild("encoder-head")

HEAD_PATH = os.path.join(cfg.qwen_encoder_dir, "head.pt")


# ----------------------------------------------------------------------
# 语料装载: xilian_copy.json（user/xilian 成对）+ responses.json + memories.json
# ----------------------------------------------------------------------
def load_corpus() -> list:
    """返回文本列表（去重; 昔涟语料为主干）。"""
    texts = []
    if os.path.isfile(cfg.diffuser_train_data):
        with open(cfg.diffuser_train_data, "r", encoding="utf-8") as f:
            data = json.load(f)
        for it in data:
            if isinstance(it, dict):
                for k in ("xilian", "user"):
                    t = str(it.get(k, "")).strip()
                    if t:
                        texts.append(t)
    if os.path.isfile(cfg.responses_path):
        with open(cfg.responses_path, "r", encoding="utf-8") as f:
            for r in json.load(f).get("responses", []):
                t = str(r.get("text", "")).strip()
                if t:
                    texts.append(t)
    if os.path.isfile(cfg.memories_path):
        with open(cfg.memories_path, "r", encoding="utf-8") as f:
            for m in json.load(f).get("memories", []):
                t = str(m.get("content", "")).strip()
                if t:
                    texts.append(t)
    seen, out = set(), []
    for t in texts:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ----------------------------------------------------------------------
# 工序 A: 冻结 Qwen0.8B 萃取 mean-pooled hidden [N, 1024]
# ----------------------------------------------------------------------
def extract_features(texts, max_len: int = 128, use_4bit: bool = False,
                     batch: int = 16) -> np.ndarray:
    """加载 0.8B → 语料批量前向 → mean-pool 特征矩阵。

    默认 FP16 直载（萃取阶段仅前向, 显存 ≈1.7GB, 速度约 15× 于 4-bit dequant）;
    --use-4bit 可切换 NF4（训练显存更小, 但 Pascal 上慢数倍）。
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, \
        BitsAndBytesConfig

    model_dir = None
    for d in (cfg.qwen_encoder_dir, cfg.legacy_qwen_dir):
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "config.json")):
            model_dir = d
            break
    if model_dir is None:
        raise FileNotFoundError("0.8B 模型目录缺失 (models/Qwen3.5-0.8B)")
    log.info("[HEAD] 加载 0.8B (%s, %s)…", model_dir,
             "4-bit NF4" if use_4bit else "FP16")
    qcfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=False, bnb_4bit_compute_dtype=torch.float16)
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=qcfg if use_4bit else None,
        device_map=dict(cfg.qwen_device_map) if use_4bit else None,
        torch_dtype=torch.float16, local_files_only=True)
    if not use_4bit:
        model = model.to(cfg.device)
    model.eval()

    feats, t0 = [], time.time()
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            ids = tok(chunk, return_tensors="pt", truncation=True,
                      max_length=max_len, padding=True)
            ids = {k: v.to(model.device) for k, v in ids.items()}
            out = model(**ids, output_hidden_states=True)
            h = out.hidden_states[-1]                       # [B, T, D]
            mask = ids["attention_mask"].unsqueeze(-1).float()  # [B, T, 1]
            feat = (h.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            feats.extend(f.cpu().numpy() for f in feat)
            if (i + batch) % 240 == 0:
                log.info("[HEAD] 萃取进度 %d/%d (%.0fs)", min(i + batch,
                                                              len(texts)),
                         len(texts), time.time() - t0)
    del model
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    X = np.stack(feats).astype(np.float32)
    log.info("[HEAD] 工序A完成: 特征矩阵 %s (%.1fs, %d 条)", X.shape,
             time.time() - t0, len(texts))
    return X


# ----------------------------------------------------------------------
# 工序 B: PCA → 384 维投影（语料协方差驱动, 可复现）
# ----------------------------------------------------------------------
def train_pca(X: np.ndarray, z_dim: int, whiten: bool = False) -> dict:
    """中心化 → 协方差 SVD → 主成分投影 W ∈ R^{z_dim×D}。"""
    mean = X.mean(axis=0, keepdims=True)
    Xc = X - mean
    # A = U S Vh → 主成分方向 = Vh 的行（特征向量）, 样本域为 U
    U, S, Vh = np.linalg.svd(Xc, full_matrices=False)
    k = min(z_dim, Vh.shape[0])
    W = Vh[:k].astype(np.float32)                         # [k, D] 主成分投影
    eigenvalues = (S ** 2) / max(1, X.shape[0] - 1)       # 主成分方差
    total_var = float(eigenvalues.sum()) + 1e-12
    explained = float(eigenvalues[:k].sum()) / total_var
    if whiten:
        W = (Vh[:k] / (S[:k, None] / np.sqrt(X.shape[0] - 1) + 1e-8)) \
            .astype(np.float32)
    return {"project": W.astype(np.float32),
            "mean": mean.astype(np.float32).reshape(-1),
            "explained": explained,
            "eigen_top": eigenvalues[:k].astype(np.float32)}


def _rule_z(text: str) -> np.ndarray:
    """规则 Z（对照用, 见 core/agi_core._text_to_z）。"""
    from core.agi_core import _text_to_z
    return _text_to_z(text, cfg)


def consistency_report(X, proj, texts, z_dim: int):
    """投影 Z 与规则 Z 的余弦一致性（客观指标, 仅报告不优化）。"""
    Xc = X - proj["mean"]
    Z = (proj["project"] @ Xc.T).T                            # [N, 384]
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    sims = []
    for i, t in enumerate(texts[: len(Xc)]):
        rz = _rule_z(t)
        rz = rz / (np.linalg.norm(rz) + 1e-9)
        sims.append(float(Zn[i] @ rz))
    sims = np.asarray(sims)
    log.info("[HEAD] 一致性: 与规则Z余弦 mean=%.3f p20=%.3f p80=%.3f",
             float(sims.mean()), float(np.percentile(sims, 20)),
             float(np.percentile(sims, 80)))


# ----------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="覆盖已有 head.pt")
    ap.add_argument("--whiten", action="store_true", help="PCA 白化（默认不白化）")
    ap.add_argument("--max-len", type=int, default=128, help="萃取截断长度")
    ap.add_argument("--use-4bit", action="store_true",
                    help="NF4 加载基座（默认 FP16: 仅前向, 快约 15×）")
    args = ap.parse_args(argv)

    os.makedirs(cfg.qwen_encoder_dir, exist_ok=True)
    if os.path.isfile(HEAD_PATH) and not args.force:
        log.info("[HEAD] 编码头已存在 %s（--force 重训）", HEAD_PATH)
        return 0

    texts = load_corpus()
    log.info("[HEAD] 语料 %d 条（xilian_copy + responses + memories）", len(texts))

    X = extract_features(texts, max_len=args.max_len, use_4bit=args.use_4bit)
    proj = train_pca(X, cfg.z_dim, whiten=args.whiten)
    log.info("[HEAD] 工序B完成: 解释方差 %.1f%%（前 384 主成分）",
             proj["explained"] * 100)

    consistency_report(X, proj, texts, cfg.z_dim)

    state = {"version": "7.3",
             "project": torch.as_tensor(proj["project"], dtype=torch.float32),
             "mean": torch.as_tensor(proj["mean"], dtype=torch.float32),
             "hidden_dim": int(proj["project"].shape[1]),
             "z_dim": cfg.z_dim, "samples": int(X.shape[0]),
             "explained": float(proj["explained"]),
             "whiten": bool(args.whiten), "trained": time.time()}
    tmp = HEAD_PATH + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, HEAD_PATH)
    log.info("[HEAD] 训练完成 → %s (project %s × %d, mean %.0fKB 已保存)",
             HEAD_PATH, proj["project"].shape, X.shape[0],
             proj["mean"].nbytes / 1024)
    log.info("[HEAD] 运行时: core/agi_core 将 z = L2(W @ (h-mean)) 接 384 维 Z ✓")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
