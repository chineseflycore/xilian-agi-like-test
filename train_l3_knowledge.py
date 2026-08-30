# -*- coding: utf-8 -*-
"""
train_l3_knowledge.py — 专业知识库 + 1M 神经元 L3 扩散器训练
=============================================================
内存/显存预估:
  - Qwen3.5-0.8B（编码器）: ≈3.2GB 显存（FP32）/ 或 AMP 减半
  - L3 1M 神经元: 权重 CSR 50M 非零 ≈ 400MB + 投影/元数据 ≈ 40MB（CPU RAM）
  - 显存: 编码用 GPU；L3 训练纯 CPU

流程:
  1. 联网抓取专业知识库（萌娘百科多词条: 昔涟/翁法罗斯/黄金裔/白厄/阿格莱雅/
     遐蝶/泰坦/逐火之旅...）→ data_cache/knowledge_db.json
  2. Qwen3.5-0.8B encode 知识文本 → 512 维向量（规范 §4.8 同款编码）
  3. 构建 L3Diffuser（1M 神经元 × 50 连接，二维 CSR，规范 §4.2 强制规格）
  4. 逐条知识向量 hebbian_update（Hebbian 无监督固化，规范 §4.2）
  5. 保存 L3 权重 → models/l3_weights.npz（启动时可加载恢复）

运行:
  Python\\python.exe train_l3_knowledge.py
  PHILIA_L3_SCALE=1.0  # 默认全尺寸 1M；内存紧张可 0.1（10 万）验证流程
"""

import os
import sys
import re
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
from common_utils import get_logger, Timer, safe_import

logger = get_logger("train_l3")

# 专业知识库词条（萌娘百科，实测可达）
_KNOWLEDGE_ENTRIES = [
    "昔涟", "翁法罗斯", "黄金裔", "白厄", "阿格莱雅", "遐蝶",
    "泰坦", "逐火之旅", "崩坏：星穹铁道",
]
_KB_CACHE = os.path.join(config.BASE_DIR, "data_cache", "knowledge_db.json")
_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"}
_MAX_ENTRY_CHARS = 2500   # 每条词条截取字符


# ==================================================================
# ① 知识库抓取（多词条，可缓存）
# ==================================================================
def _clean_html(html: str) -> str:
    m = re.search(r'<div[^>]*id="mw-content-text"[^>]*>', html)
    body = html[m.end():] if m else html
    body = re.sub(r'<script.*?</script>', " ", body, flags=re.S)
    body = re.sub(r'<style.*?</style>', " ", body, flags=re.S)
    paras = re.findall(r'<p[^>]*>(.*?)</p>', body, re.S)
    txt = " ".join(re.sub(r'<[^>]+>', " ", p) for p in paras)
    txt = re.sub(r'&nbsp;|&amp;|&lt;|&gt;|&quot;', " ", txt)
    txt = re.sub(r'\s+', " ", txt)
    return txt.strip()


def build_knowledge_db(force: bool = False) -> list:
    """抓取多词条知识，返回 [{"entry","text"}]；命中缓存则直接读。"""
    if not force and os.path.isfile(_KB_CACHE):
        try:
            with open(_KB_CACHE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data:
                logger.info("[KB] 知识库缓存命中（%d 条，%d 字符）",
                            len(data), sum(len(d["text"]) for d in data))
                return data
        except Exception:
            pass
    requests = safe_import("requests")
    if requests is None:
        logger.warning("[KB] requests 不可用，跳过抓取")
        return []
    db = []
    for entry in _KNOWLEDGE_ENTRIES:
        try:
            url = "https://zh.moegirl.org.cn/" + requests.utils.quote(entry)
            r = requests.get(url, timeout=15, headers=_UA)
            if r.status_code != 200:
                logger.warning("[KB] %s -> HTTP %d", entry, r.status_code)
                continue
            txt = _clean_html(r.text)[:_MAX_ENTRY_CHARS]
            if len(txt) > 100:
                db.append({"entry": entry, "text": txt})
                logger.info("[KB] 抓取成功 %s: %d 字符", entry, len(txt))
        except Exception as e:
            logger.warning("[KB] 抓取异常 %s: %s", entry, e)
    if db:
        try:
            os.makedirs(os.path.dirname(_KB_CACHE), exist_ok=True)
            with open(_KB_CACHE, "w", encoding="utf-8") as f:
                json.dump(db, f, ensure_ascii=False)
            logger.info("[KB] 知识库已缓存 → %s", _KB_CACHE)
        except Exception as e:
            logger.warning("[KB] 缓存写入失败: %s", e)
    return db


# ==================================================================
# ② 文本分块（长文 → 语义块，供 encode 与 Hebbian 固化）
# ==================================================================
def chunk_text(text: str, max_len: int = 120) -> list:
    """按句切分合并为 ≤max_len 的块（中文句号/分号/换行分隔）。"""
    sentences = re.split(r"[。！？；\n]+", text)
    chunks, cur = [], ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(cur) + len(s) <= max_len:
            cur += s + "。"
        else:
            if cur:
                chunks.append(cur)
            cur = s + "。"
    if cur:
        chunks.append(cur)
    return [c for c in chunks if len(c) >= 8]


# ==================================================================
# ②b 用户提供的训练资料（data_cache/训练资料*.txt，ChatML 人格对话）
# ==================================================================
_USER_DATA_DIR = os.path.join(config.BASE_DIR, "data_cache")


def _clean_chatml(raw: str) -> str:
    """ChatML 清洗（兼容序号版: "501." 开头 + <|im_start|> 块）。

    处理: 去文件头叙述段（首条 <|im_start|> 之前）/ 序号行 / ChatML 标记 /
    分隔线 / 杂话（✅✓✔ 开头、生成说明等，整行丢弃）。
    """
    # 只保留首条对话开始之后的内容（丢弃"好的，伙伴。人家继续讲..."等文件头叙述）
    idx = raw.find("<|im_start|>")
    if idx > 0:
        raw = raw[idx:]
    text = re.sub(r"<\|im_start\|>system\s*", "", raw)
    text = re.sub(r"<\|im_start\|>user\s*", "问：", text)
    text = re.sub(r"<\|im_start\|>assistant\s*", "答：", text)
    text = re.sub(r"<\|im_end\|>\s*", "\n", text)
    text = re.sub(r"-{3,}", "\n", text)
    text = re.sub(r"(?m)^\s*\d{1,4}\.\s*$", "", text)     # 去序号行（501. / 601.）
    # 杂话过滤（整行丢弃，无论长度；含 ✅/✓/✔/完成/完毕/生成说明/总计）
    clean = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if re.match(r"^[✅✓✔]", s):
            continue
        if s.startswith(("完成", "完毕", "数据生成", "包含前", "总序号", "本次新增", "抱歉伙伴")):
            continue
        clean.append(line)
    return "\n".join(clean).strip()


def load_user_data() -> str:
    """扫描并合并 data_cache/训练资料*.txt（可多个，格式: ChatML / 序号版）。

    文件不存在/为空 → 返回 ""。
    """
    if not os.path.isdir(_USER_DATA_DIR):
        return ""
    files = sorted(f for f in os.listdir(_USER_DATA_DIR)
                   if f.startswith("训练资料") and f.endswith(".txt"))
    parts = []
    for fn in files:
        path = os.path.join(_USER_DATA_DIR, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read()
            text = _clean_chatml(raw)
            if text:
                parts.append(text)
                logger.info("[KB] 训练资料 %s: %d 字符", fn, len(text))
        except Exception as e:
            logger.warning("[KB] 训练资料 %s 读取失败: %s", fn, e)
    return "\n".join(parts)


# ==================================================================
# ③ L3 训练（Hebbian 固化）
# ==================================================================
def train_l3(cfg, translator, chunks: list) -> dict:
    """知识向量 Hebbian 固化到 L3（1M 神经元）。"""
    from diffuser_l3 import L3Diffuser
    t0 = time.time()
    logger.info("[L3] 构建 %d 神经元扩散器（二维 CSR %d×%d）...",
                cfg.l3_neuron_count_eff, cfg.l3_neuron_count_eff, cfg.l3_cols_eff)
    l3 = L3Diffuser(cfg)
    logger.info("[L3] 构建完成（%.1fs），开始 Hebbian 固化 %d 个知识块...",
                time.time() - t0, len(chunks))

    updated = 0
    # 批量编码（batch=16，0.8B FP32 下 64 会 OOM；定期释放显存碎片）→ 逐块 Hebbian 固化
    vecs = []
    bs = 16
    for i in range(0, len(chunks), bs):
        vecs.extend(translator.encode(chunks[i:i + bs]))
        if (i // bs + 1) % 8 == 0:
            try:
                import torch as _t
                _t.cuda.empty_cache()
            except Exception:
                pass
            logger.info("[L3] 已编码 %d/%d", min(i + bs, len(chunks)), len(chunks))
    logger.info("[L3] 知识块编码完成（%d 个向量）", len(vecs))
    with Timer("l3_knowledge_train") as tm:
        for i, vec in enumerate(vecs):
            l3.hebbian_update(vec, reward=1.0)           # Hebbian 无监督固化
            updated += 1
            if (i + 1) % 40 == 0:
                logger.info("[L3] 已固化 %d/%d（%.1fs）", i + 1, len(chunks),
                            tm.elapsed / 1000.0)
    logger.info("[L3] Hebbian 固化完成: %d 块, 耗时 %.1fs", updated, tm.elapsed / 1000.0)
    return l3, {"chunks": len(chunks), "updated": updated,
                "elapsed_s": round(tm.elapsed / 1000.0, 1)}


def main() -> int:
    t0 = time.time()
    cfg = config.get_config()
    cfg.ensure_dirs()
    logger.info("=" * 60)
    logger.info("[L3] 专业知识库 + 1M L3 扩散器训练 开始")

    # ① 知识库
    db = build_knowledge_db()
    if not db:
        logger.error("[L3] 知识库为空，终止")
        return 1
    total_chars = sum(len(d["text"]) for d in db)
    logger.info("[L3] 知识库: %d 词条, %d 字符", len(db), total_chars)

    # ② 编码器（真实 Qwen3.5-0.8B）
    from translator import Translator
    translator = Translator(cfg)
    if translator.is_demo:
        logger.error("[L3] 编码器不可用（DEMO），无法训练真实 L3，终止")
        return 1
    logger.info("[L3] 编码器就绪 device=%s", translator.device)

    # ③ 分块（知识库词条 + 用户训练资料）
    chunks = []
    for d in db:
        chunks.extend(chunk_text(d["text"]))
    user_data = load_user_data()
    if user_data:
        user_chunks = chunk_text(user_data)
        chunks.extend(user_chunks)
        logger.info("[L3] 用户训练资料分块: %d 块（对话语料）", len(user_chunks))
    logger.info("[L3] 知识分块总计: %d 块", len(chunks))

    # ④ 训练
    l3, stats = train_l3(cfg, translator, chunks)

    # ⑤ 保存（models/l3_weights.npz）
    os.makedirs(os.path.join(cfg.model_cache_dir), exist_ok=True)
    save_path = os.path.join(cfg.model_cache_dir, "l3_weights.npz")
    try:
        l3.save_state(save_path)
        logger.info("[L3] 权重已保存 → %s", save_path)
    except AttributeError:
        # 兼容旧接口: 用 get_weights 保存
        from scipy import sparse
        sparse.save_npz(save_path + ".csr", l3.get_weights())
        np.savez(save_path + ".meta.npz",
                 thresholds=l3.meta.thresholds, pain_marks=l3.meta.pain_marks)
        logger.info("[L3] 权重已保存（旧接口兼容）→ %s", save_path)

    # ⑥ L2 Transformer 扩散器训练（同一批知识向量，自编码式去噪，双缓冲）
    logger.info("[L2] Transformer 扩散器训练（%d 个知识块，自编码去噪）...", len(chunks))
    from diffuser_l2 import L2Diffuser
    l2 = L2Diffuser(cfg)
    t_l2 = time.time()
    # 批量编码（batch=16，0.8B FP32 下 64 会 OOM）→ 逐块 train_step
    l2_vecs = []
    for i in range(0, len(chunks), 16):
        l2_vecs.extend(translator.encode(chunks[i:i + 16]))
        if (i // 16 + 1) % 16 == 0:
            logger.info("[L2] 已编码 %d/%d", min(i + 16, len(chunks)), len(chunks))
    for i, vec in enumerate(l2_vecs):
        l2.train_step(vec, vec)                # 目标=输入（去噪自编码）
        if (i + 1) % 50 == 0:
            logger.info("[L2] 已训练 %d/%d（%.1fs）", i + 1, len(chunks), time.time() - t_l2)
    l2.swap()
    l2_path = os.path.join(cfg.model_cache_dir, "l2_weights.pt")
    l2.save_weights(l2_path)
    logger.info("[L2] 训练完成（%.1fs），权重已保存 → %s",
                time.time() - t_l2, l2_path)

    logger.info("[L3] 全部完成，总耗时 %.1fs —— 训练统计: %s",
                time.time() - t0, stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
