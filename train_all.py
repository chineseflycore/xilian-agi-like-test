# -*- coding: utf-8 -*-
"""
train_all.py — 训练数据生成与训练入口（规范 §6.1，资料增强版）
================================================================
内存/显存预估（规范 §九.1）:
  - 数据集: 200 条对话 × ≤256 token ≈ < 2MB（磁盘/内存）
  - 知识缓存: 联网抓取的真实资料（萌娘百科等）≈ < 1MB（data_cache/knowledge_xi_lian.json）
  - 训练: L2 CPU 副本（~4MB）+ L3 Hebbian（numpy/scipy）—— 均在 CPU
  - 显存: 0MB（训练不占 GPU 显存；实际推理时路由层常驻 GPU）

流程（规范 §6.1 + 资料增强）:
  0. 联网抓取"昔涟/翁法罗斯/黄金裔"真实资料（萌娘百科词条，可缓存），
     作为生成上下文 —— 让 DeepSeek"看着资料说话"，而非凭空编造
  1. 加载 5 条种子示例（规范 §5.2，cloud_model_client.SEED_EXAMPLES）
  2. 从真实资料中提取话题 → 构造 200 个场景提示（资料驱动，非固定话题表）
  3. 调用 DeepSeek API 生成对话（cloud_model_client，system 注入资料）
     · 本地缓存: 缓存键 = 种子示例哈希 + 资料哈希 + 场景提示哈希；命中跳过 API（规范 §6.1）
     · API Key 缺失降级: 种子示例 + 资料实体驱动模板生成最小数据集，黄色警告（规范 §九.9）
     · 降级数据集仅用于验证流程完整性，不用于实际训练 L2/L3（规范 §6.1 使用边界）
  4. 抽取 20% 作为验证集（规范 §6.1）
  5. 训练 L2（CPU 副本，双缓冲）与 L3（Hebbian 更新）
  6. 数据集落盘到 BASE_DIR/data_cache/train_dataset.json

运行:
  python train_all.py                # 全流程（无 Key 时抓资料+降级数据集）
  # API Key 配置优先级: hoyotool.ini [Cloud].api_key > 环境变量 DEEPSEEK_API_KEY
  # （或在 PowerShell: $env:DEEPSEEK_API_KEY="sk-..." 后 python train_all.py）
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
import re
import json
import time

import numpy as np

import config
from common_utils import get_logger, stable_hash, safe_import
from cloud_model_client import CloudModelClient, SEED_EXAMPLES

logger = get_logger("train_all")

# 场景提示模板（规范 §6.1: 构造 200 个场景提示；topic 来自真实资料提取）
_SCENARIO_TEMPLATES = [
    "和昔涟聊聊{topic}。",
    "昔涟，你还记得关于{topic}的事吗？",
    "如果今晚没有月亮，你会对{topic}说什么？",
    "三千万世的轮回里，{topic}对你意味着什么？",
    "有人问起{topic}，你会怎么回答？",
    "在翁法罗斯，{topic}是什么样子的？",
    "请用温柔的语气谈谈{topic}。",
    "昔涟，{topic}让你想起了谁？",
]

# 角色人设（生成时 system 前缀；直接复用 config.persona_system —— 完整版为
# xilian-agent 的 personality_v4.md，data_cache/reference_xilian/）
_SYSTEM_PROMPT = config.get_config().persona_system

# 联网抓取的知识源（萌娘百科可达，实测 HTTP 200；百度百科 403 反爬）
_KNOWLEDGE_SOURCES = [
    ("萌娘百科-昔涟", "https://zh.moegirl.org.cn/昔涟"),
    ("萌娘百科-翁法罗斯", "https://zh.moegirl.org.cn/翁法罗斯"),
]
_KNOWLEDGE_CACHE = os.path.join(config.BASE_DIR, "data_cache", "knowledge_xi_lian.json")
_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                     "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"}

# 中文停用词（话题提取过滤）
_STOPWORDS = set(
    "的了是在我你他她它我们你们他们这那也就都而和对从到说着过被把让会能要有没不很最"
    "以及一个一种一些这样那样因为所以但是如果虽然可能应该已经正在还是终于于是后来最后"
    "其中其中这里那里什么怎么为什么如何这个那个然后接着开始结束曾经一直非常特别"
)


# ==================================================================
# ① 联网抓取真实资料（萌娘百科，规范 §6.2 精神 + 资料增强）
# ==================================================================
def _clean_html(html: str) -> str:
    """清洗 HTML → 中文正文文本（提取 <p> 段落，鲁棒于 div 嵌套结构）。"""
    # 定位正文起点（萌娘百科: #mw-content-text），其后提取 <p> 段落
    m = re.search(r'<div[^>]*id="mw-content-text"[^>]*>', html)
    body = html[m.end():] if m else html
    body = re.sub(r'<script.*?</script>', " ", body, flags=re.S)
    body = re.sub(r'<style.*?</style>', " ", body, flags=re.S)
    paras = re.findall(r'<p[^>]*>(.*?)</p>', body, re.S)
    txt = " ".join(re.sub(r'<[^>]+>', " ", p) for p in paras)
    txt = re.sub(r'&nbsp;|&amp;|&lt;|&gt;|&quot;', " ", txt)
    txt = re.sub(r'\s+', " ", txt)
    return txt.strip()


def fetch_world_knowledge(cfg, force: bool = False) -> str:
    """联网抓取昔涟/翁法罗斯真实资料，返回清洗后的文本（可缓存）。

    返回 "" 表示抓取失败（不影响主流程，降级为固定话题表）。
    """
    # 命中本地缓存（资料变更频率低，避免每次生成都抓）
    if not force and os.path.isfile(_KNOWLEDGE_CACHE):
        try:
            with open(_KNOWLEDGE_CACHE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("text"):
                logger.info("[TRAIN] 知识缓存命中（%d 字符，%s）",
                            len(data["text"]), data.get("source", ""))
                return data["text"]
        except Exception:
            pass

    requests = safe_import("requests")
    if requests is None:
        logger.warning("[TRAIN] requests 不可用，跳过资料抓取")
        return ""

    parts = []
    for name, url in _KNOWLEDGE_SOURCES:
        try:
            r = requests.get(url, timeout=15, headers=_UA)
            if r.status_code != 200:
                logger.warning("[TRAIN] 抓取失败 %s: HTTP %d", name, r.status_code)
                continue
            txt = _clean_html(r.text)
            # 截取正文前 4000 字符（控制上下文长度）
            txt = txt[:4000]
            if len(txt) > 200:
                parts.append(f"【{name}】\n{txt}")
                logger.info("[TRAIN] 抓取成功 %s: %d 字符", name, len(txt))
        except Exception as e:
            logger.warning("[TRAIN] 抓取异常 %s: %s", name, e)

    knowledge = "\n\n".join(parts)
    if knowledge:
        try:
            os.makedirs(os.path.dirname(_KNOWLEDGE_CACHE), exist_ok=True)
            with open(_KNOWLEDGE_CACHE, "w", encoding="utf-8") as f:
                json.dump({"source": "moegirl", "text": knowledge}, f, ensure_ascii=False)
            logger.info("[TRAIN] 知识已缓存 → %s", _KNOWLEDGE_CACHE)
        except Exception as e:
            logger.warning("[TRAIN] 知识缓存写入失败: %s", e)
    else:
        logger.warning("[TRAIN] 资料抓取全部失败，将使用固定话题表")
    return knowledge


# ==================================================================
# ② 资料驱动话题提取（不再是固定 16 个话题）
# ==================================================================
def extract_topics(knowledge: str, k: int = 16) -> list:
    """从真实资料中提取高频话题（2~4 字 n-gram 词频，过滤停用词）。"""
    if not knowledge:
        return ["花海", "轮回", "死亡", "黄金裔", "翁法罗斯", "告别",
                "守护", "记忆", "害怕", "希望", "月光", "安静"]
    text = re.sub(r"[^\u4e00-\u9fff]", "", knowledge)   # 仅保留中文
    freq = {}
    for n in (4, 3, 2):      # 从长到短: 长词优先占位（"翁法罗斯" 优先于 "翁法"）
        for i in range(len(text) - n + 1):
            gram = text[i:i + n]
            if any(c in _STOPWORDS for c in gram):
                continue
            freq[gram] = freq.get(gram, 0) + 1
    # 过滤纯停用词组合与超高频通用词
    ranked = sorted(freq.items(), key=lambda kv: -kv[1])
    topics, seen = [], set()
    for gram, cnt in ranked:
        if len(topics) >= k:
            break
        if gram in seen:
            continue
        # 去子串重复（"翁法罗" 与 "翁法罗斯" 保留更长的）
        skip = False
        for t in topics:
            if gram in t or t in gram:
                skip = True
                break
        if skip:
            continue
        if cnt < 2:      # 至少出现 2 次才算话题
            continue
        seen.add(gram)
        topics.append(gram)
    logger.info("[TRAIN] 资料驱动话题提取: %s", topics[:12])
    return topics or ["翁法罗斯", "黄金裔", "轮回", "花海"]


def build_scenarios(count: int = 200, topics: list = None) -> list:
    """构造 count 个场景提示（确定性：轮转模板×话题；话题来自真实资料）。"""
    topics = topics or ["花海", "轮回", "死亡", "黄金裔", "翁法罗斯"]
    out = []
    for i in range(count):
        tpl = _SCENARIO_TEMPLATES[i % len(_SCENARIO_TEMPLATES)]
        topic = topics[(i // len(_SCENARIO_TEMPLATES)) % len(topics)]
        out.append({"prompt": tpl.format(topic=topic), "topic": topic})
    return out


# ==================================================================
# ③ 数据生成（资料增强 + 缓存；Key 缺失 → 资料驱动降级）
# ==================================================================
def generate_dataset(seed_hash: str = "", knowledge: str = "", count: int = None) -> dict:
    """生成完整数据集（Responses API + web_search；缓存；Key 缺失 → 降级数据集）。

    生成策略（训练数据"有据可依"）:
      ① Responses API（model=deepseek-v4-flash）携带 web_search 工具，
         服务端联网搜索 → 模型基于搜索结果作答（规范 §6.1 V4-Flash）；
      ② 失败回退 chat/completions（注入自抓资料 knowledge）；
      ③ API Key 缺失 → 资料驱动的降级数据集（仅流程验证）。

    返回 {"pairs": [...], "degraded": bool, "count": int, "knowledge": str,
          "search_based": int}。
    """
    client = CloudModelClient(config.get_config())
    topics = extract_topics(knowledge)
    scenarios = build_scenarios(count or config.get_config().train_scenario_count,
                                topics=topics)
    if not getattr(client, "available", False):
        # API Key 缺失 → 资料驱动的降级数据集（仅流程验证，规范 §6.1）
        return client.build_degraded_dataset(scenarios, knowledge=knowledge)

    pairs, search_based = [], 0
    for sp in scenarios:
        text = None
        # ① Responses API + web_search（模型联网搜索后作答）
        res = client.generate_with_search(
            sp["prompt"], instructions=_SYSTEM_PROMPT, knowledge=knowledge,
            force_search=True, seed_hash=seed_hash)
        if res and res.get("text"):
            text = res["text"]
            search_based += 1
        else:
            # ② 回退 chat/completions（资料注入 system）
            text = client.generate_cached(
                sp["prompt"], system=_SYSTEM_PROMPT, seed_hash=seed_hash,
                knowledge=knowledge)
        if text:
            pairs.append({"user": sp["prompt"], "assistant": text})
        else:
            logger.warning("[TRAIN] 场景生成失败: %s", sp["prompt"])
    return {"pairs": pairs, "degraded": False, "count": len(pairs),
            "knowledge": knowledge, "search_based": search_based}


def split_dataset(pairs: list, val_ratio: float = 0.2) -> dict:
    """按比例切分训练集/验证集（确定性打乱）。"""
    rng = np.random.RandomState(20240601)
    idx = rng.permutation(len(pairs))
    n_val = max(1, int(len(pairs) * val_ratio))
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return {
        "train": [pairs[i] for i in train_idx],
        "val": [pairs[i] for i in val_idx],
    }


def train_l2_l3(train_pairs: list, cfg, l2=None, l3=None) -> dict:
    """训练 L2（CPU 副本双缓冲）与 L3（Hebbian）。

    DEMO（无 torch/l2/l3）: 打印流程验证日志并返回统计，不做真实训练
    （降级数据集不用于实际训练，规范 §6.1 使用边界）。
    """
    torch = safe_import("torch")
    stats = {"l2_steps": 0, "hebbian_updates": 0, "skipped": False}
    if l2 is None and l3 is None:
        logger.warning(
            "\033[33m[TRAIN] 未提供 L2/L3 实例，跳过真实训练（流程验证模式）。\033[0m")
        stats["skipped"] = True
        return stats
    if torch is None:
        logger.warning(
            "\033[33m[TRAIN] torch 不可用，跳过真实训练（流程验证模式）。\033[0m")
        stats["skipped"] = True
        return stats
    # 简单向量化训练: 每对样本 → 伪向量 → L2 train_step / L3 hebbian_update
    from common_utils import pseudo_embedding
    for i, pair in enumerate(train_pairs[:64]):        # 训练集过大时截断保护
        u = pseudo_embedding(pair["user"], cfg.STATE_DIM)
        a = pseudo_embedding(pair["assistant"], cfg.STATE_DIM)
        if l2 is not None and hasattr(l2, "train_step"):
            try:
                l2.train_step(u, a, lr=0.01)
                stats["l2_steps"] += 1
            except TypeError:
                try:
                    l2.train_step(u, a)
                    stats["l2_steps"] += 1
                except Exception as e:
                    logger.warning("[TRAIN] L2 train_step 失败: %s", e)
        if l3 is not None and hasattr(l3, "hebbian_update"):
            try:
                l3.hebbian_update(u, reward=1.0)
                stats["hebbian_updates"] += 1
            except TypeError:
                try:
                    l3.hebbian_update(u)
                    stats["hebbian_updates"] += 1
                except Exception as e:
                    logger.warning("[TRAIN] L3 hebbian 失败: %s", e)
    if l2 is not None and hasattr(l2, "swap"):
        try:
            r = l2.swap()
            logger.info("[TRAIN] L2 原子替换结果: %s", r)
        except Exception as e:
            logger.warning("[TRAIN] L2 swap 失败: %s", e)
    return stats


def main():
    """train_all 主流程（规范 §6.1 + 资料增强 + Responses API web_search）。"""
    t0 = time.time()
    cfg = config.get_config()
    cfg.ensure_dirs()
    # 场景数可用环境变量 PHILIA_SCENARIOS 覆盖（先小样验证，再全量 200）
    n_scen = int(os.environ.get("PHILIA_SCENARIOS", str(cfg.train_scenario_count)))
    logger.info("=" * 60)
    logger.info("[TRAIN] 昔涟人格引擎 训练数据生成与训练 开始（资料增强版）")
    logger.info("[TRAIN] ① 加载 5 条种子示例（规范 §5.2）")
    seed_hash = stable_hash(json.dumps(SEED_EXAMPLES, ensure_ascii=False))
    logger.info("[TRAIN] 种子示例哈希=%s（修改种子后缓存自动失效）", seed_hash)

    logger.info("[TRAIN] ② 联网抓取真实资料（萌娘百科: 昔涟/翁法罗斯）")
    knowledge = fetch_world_knowledge(cfg)
    logger.info("[TRAIN] 知识上下文 %d 字符%s",
                len(knowledge), "（抓取失败，用固定话题）" if not knowledge else "")

    topics = extract_topics(knowledge)
    scenarios = build_scenarios(n_scen, topics=topics)
    logger.info("[TRAIN] 场景示例: %s（话题来自真实资料: %s，本次共 %d 个场景）",
                scenarios[0]["prompt"], topics[0] if topics else "?", n_scen)

    logger.info("[TRAIN] ③ 调用 DeepSeek Responses API 生成对话（web_search 联网搜索 + 本地缓存）")
    dataset = generate_dataset(seed_hash=seed_hash, knowledge=knowledge, count=n_scen)
    pairs = dataset["pairs"]
    if dataset["degraded"]:
        logger.warning(
            "\033[33m[TRAIN] 降级数据集（%d 条）：仅验证流程完整性，"
            "不用于实际训练 L2/L3（规范 §6.1）\033[0m", len(pairs))
    else:
        logger.info("[TRAIN] API 数据集 %d 条（其中 %d 条经 web_search 联网搜索生成）",
                    len(pairs), dataset.get("search_based", 0))

    # 落盘（路径基于 BASE_DIR）
    out_path = os.path.join(cfg.cache_dir, "train_dataset.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"seed_hash": seed_hash, "degraded": dataset["degraded"],
                       "knowledge_chars": len(knowledge),
                       "search_based": dataset.get("search_based", 0),
                       "count": len(pairs), "pairs": pairs[:8]},  # 仅存前 8 条示例防膨胀
                      f, ensure_ascii=False, indent=1)
        logger.info("[TRAIN] 数据集摘要已落盘: %s", out_path)
    except Exception as e:
        logger.warning("[TRAIN] 落盘失败: %s", e)

    logger.info("[TRAIN] ④ 抽取 20%% 作为验证集")
    if not pairs:
        logger.error("[TRAIN] 数据集为空，终止")
        return 1
    split = split_dataset(pairs, cfg.train_val_ratio)
    logger.info("[TRAIN] 训练集=%d 验证集=%d", len(split["train"]), len(split["val"]))

    logger.info("[TRAIN] ⑤ 训练 L2（CPU 副本）与 L3（Hebbian 更新）")
    stats = train_l2_l3(split["train"], cfg, l2=None, l3=None)
    logger.info("[TRAIN] 训练统计: %s", stats)
    logger.info("[TRAIN] 总耗时 %.1fs —— 完成（规范 §6.1）", time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
