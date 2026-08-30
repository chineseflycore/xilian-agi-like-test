# -*- coding: utf-8 -*-
"""
train_emotion_model.py — 40M 情感模型独立训练（规范 §3.2 / §八.11）
====================================================================
内存/显存预估:
  - 数据集: ~2000 条 (文本→8维情绪标签) ≈ < 1MB
  - 特征: Qwen3.5-0.8B encode（复用，≈3.2GB 显存）→ 512 维
  - 模型: 40M MLP（512→4096→4096→4096→8）FP32 ≈ 143MB（GPU/CPU）
  - 训练峰值: 显存 ≈ 3.4GB（GTX 1060 6GB 内）

流程:
  1. 数据生成: 知识库文本 + 种子示例 + 情绪关键词构造句 → 8 维情绪标签
     （标签由 emotion_model 的 DEMO 规则路径生成，训练目标=让 40M 网络拟合规则
       → 之后 REAL 路径可替代规则，规范 §十.27 经验式可成长）
  2. Qwen3.5-0.8B encode → 512 维特征
  3. 训练 40M MLP（MSE 回归 8 维，Adam，GPU）
  4. 保存 models/emotion_40m.pt（state_dict + 结构由 emotion_model.py 提供）
  5. emotion_model.py REAL 路径加载后替代 DEMO 规则

运行:
  Python\\python.exe train_emotion_model.py
  PHILIA_EMO_DATA=2000  # 数据条数（默认 2000；可调小快速验证）
  PHILIA_EMO_EPOCHS=30  # 训练轮数
"""

import os
import sys
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

import numpy as np

import config
from common_utils import get_logger, safe_import

logger = get_logger("train_emotion")

# 情绪关键词构造句（扩充规则路径样本量，标签由规则路径自动生成）
_EMO_SENTENCES = [
    # fear
    "我害怕死亡。", "黑暗让我恐惧。", "不要走，我害怕消失。", "疼痛让我害怕失去一切。",
    "梦里全是破碎的画面，我好怕。", "那场噩梦让我颤抖。",
    # happy
    "我喜欢花海，温暖又明亮。", "明天见，我很开心。", "重逢的那一刻，我笑了。",
    "阳光照进来，一切都好起来了。", "听到你的声音，我心里暖暖的。",
    # sad
    "再见，也许再也不见了。", "我怀念从前。", "孤独像无声的雨。",
    "那些记忆，我留不住。", "对不起，我只能说再见了。",
    # angry
    "这不公平！", "我讨厌欺骗。", "背叛让我愤怒。", "别这样对我。",
    # chaos
    "数据混乱，出现乱码。", "碎片错乱，记忆断裂。", "混沌中一切失序。",
    "我分不清这是梦还是现实，全部崩溃了。",
    # neutral / 平静
    "翁法罗斯的花海很安静。", "我在听风的声音。", "今天没有月亮，也没有风。",
    "黄金裔的使命，世代相传。", "泰坦的火种，仍在燃烧。",
]


def build_dataset(cfg, translator, emotion_model, n: int = 2000) -> list:
    """生成 (text, label8) 数据集。

    label8 由 emotion_model（DEMO 规则路径）生成 —— 训练目标 = 让 40M 网络
    拟合规则路径的输入输出映射（规范 §4.1 情感模型 40M / §3.2 8 维连续）。
    """
    # 1) 情绪关键词句
    pairs = [(s, emotion_model.from_text(s)) for s in _EMO_SENTENCES]
    # 2) 种子示例（规范 §5.2）
    from cloud_model_client import SEED_EXAMPLES
    for i in range(0, len(SEED_EXAMPLES), 2):
        pairs.append((SEED_EXAMPLES[i]["content"], emotion_model.from_text(
            SEED_EXAMPLES[i]["content"])))
        pairs.append((SEED_EXAMPLES[i + 1]["content"], emotion_model.from_text(
            SEED_EXAMPLES[i + 1]["content"])))
    # 3) 知识库文本块（中性/信任倾向）
    kb_path = os.path.join(config.BASE_DIR, "data_cache", "knowledge_db.json")
    if os.path.isfile(kb_path):
        try:
            with open(kb_path, "r", encoding="utf-8") as f:
                db = json.load(f)
            texts = []
            for d in db:
                for s in (d.get("text", "").split("。")):
                    s = s.strip()
                    if 8 <= len(s) <= 60:
                        texts.append(s)
            rng = np.random.RandomState(20260601)
            if len(texts) > n // 3:
                texts = list(rng.choice(texts, n // 3, replace=False))
            for t in texts:
                pairs.append((t, emotion_model.from_text(t)))
        except Exception as e:
            logger.warning("[EMO] 知识库读取失败: %s", e)
    # 4) 随机模板扩充到 n 条（关键词随机组合）
    rng = np.random.RandomState(20260601)
    kws = {"fear": ["害怕", "死亡", "黑暗", "消失", "痛"],
           "happy": ["喜欢", "温暖", "花海", "开心", "明天见"],
           "sad": ["再见", "遗忘", "孤独", "怀念", "留不住"],
           "angry": ["不公平", "欺骗", "背叛", "愤怒"],
           "chaos": ["混乱", "乱码", "崩溃", "失序"]}
    while len(pairs) < n:
        kind = rng.choice(list(kws))
        kw = rng.choice(kws[kind])
        tpl = rng.choice(["{kw}，我觉得很难受。", "说起{kw}，我心里很复杂。",
                          "关于{kw}的事，我记不太清了。", "（轻声）{kw}……"])
        text = tpl.format(kw=kw)
        pairs.append((text, emotion_model.from_text(text)))
    return pairs[:n]


def main() -> int:
    t0 = time.time()
    cfg = config.get_config()
    cfg.ensure_dirs()
    torch = safe_import("torch")
    if torch is None:
        logger.error("[EMO] torch 不可用，终止")
        return 1

    n = int(os.environ.get("PHILIA_EMO_DATA", "2000"))
    epochs = int(os.environ.get("PHILIA_EMO_EPOCHS", "30"))
    logger.info("=" * 60)
    logger.info("[EMO] 40M 情感模型训练 开始（数据=%d 条, epochs=%d）", n, epochs)

    # ① 编码器 + 规则标签生成器
    from translator import Translator
    from emotion_model import EmotionModel, build_emotion_net
    translator = Translator(cfg)
    if translator.is_demo:
        logger.error("[EMO] 编码器不可用（DEMO），终止")
        return 1
    emo_rule = EmotionModel(cfg)          # DEMO 规则路径（标签生成器）
    logger.info("[EMO] 编码器 device=%s", translator.device)

    # ② 数据集
    dataset = build_dataset(cfg, translator, emo_rule, n=n)
    logger.info("[EMO] 数据集 %d 条（含关键词句/种子/知识块/模板）", len(dataset))

    # ③ 特征化（Qwen encode → 512 维，批量编码；batch=16 防 0.8B FP32 OOM）
    X, Y = [], []
    texts = [t for t, _ in dataset]
    labels = [l for _, l in dataset]
    bs = 16
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            batch = texts[i:i + bs]
            X.extend(translator.encode(batch))      # (bs, 512) 一次前向
            Y.extend(labels[i:i + bs])
            if (i // bs + 1) % 8 == 0 or i + bs >= len(texts):
                logger.info("[EMO] 编码 %d/%d", min(i + bs, len(texts)), len(texts))
                torch.cuda.empty_cache()
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    logger.info("[EMO] 特征矩阵 %s, 标签矩阵 %s", X.shape, Y.shape)

    # ④ 训练 40M MLP（结构由 emotion_model.build_emotion_net 统一提供）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_emotion_net(in_dim=cfg.STATE_DIM, out_dim=cfg.EMOTION_DIM)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("[EMO] 模型参数: %d (%.1f MB FP32), device=%s", n_params,
                n_params * 4 / 1048576, device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()

    Xt = torch.from_numpy(X).to(device)
    Yt = torch.from_numpy(Y).to(device)
    bs = 64
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        total_loss = 0.0
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            xb, yb = Xt[idx], Yt[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        avg = total_loss / len(Xt)
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            logger.info("[EMO] epoch %d/%d loss=%.4f", ep + 1, epochs, avg)

    # ⑤ 保存 models/emotion_40m.pt
    os.makedirs(cfg.model_cache_dir, exist_ok=True)
    save_path = os.path.join(cfg.model_cache_dir, "emotion_40m.pt")
    torch.save({"state_dict": model.state_dict(),
                "in_dim": cfg.STATE_DIM, "out_dim": cfg.EMOTION_DIM,
                "params": n_params}, save_path)
    logger.info("[EMO] 模型已保存 → %s（%.1f MB）", save_path, os.path.getsize(save_path) / 1048576)
    logger.info("[EMO] 全部完成，总耗时 %.1fs", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
