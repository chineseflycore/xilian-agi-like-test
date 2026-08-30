# -*- coding: utf-8 -*-
"""
gen_persona_data.py — 人格训练数据生成（前台友好版，逐条实时打印）
===================================================================
内存/显存预估: 无本地模型权重；仅 API 调用 + 文本缓冲（< 10MB RAM / 0 显存）。

与 train_all.py 的区别:
  · 每生成一条，实时打印完整 user/assistant 内容（您能亲眼看到数据长什么样）
  · 缓存命中条目标记 [缓存命中]（不重复调用 API，不烧钱）
  · 落盘完整数据集 data_cache/persona_dataset.json（不截断）
  · 默认 200 条；可用 --count N 调整（如 --count 20 先试跑）

流程（规范 §6.1 + 资料增强 + Responses API web_search）:
  1. 联网抓取真实资料（萌娘百科: 昔涟/翁法罗斯，可缓存）
  2. 资料驱动话题 → 场景提示
  3. DeepSeek Responses API（deepseek-v4-flash）携带 web_search 工具:
     第一轮强制联网搜索 → 第二轮模型基于搜索结果作答（两轮自动续传）
  4. 全部落盘 + 打印统计

运行:
  Python\\python.exe gen_persona_data.py --count 200
  （或双击 gen_persona_data.bat）
"""

import os
import sys
import json
import time
import argparse

_BASE = os.path.dirname(os.path.abspath(__file__))
if _BASE not in sys.path:
    sys.path.insert(0, _BASE)
try:
    import numpy  # noqa: F401
except ImportError:
    _V = os.path.join(_BASE, "vendor")
    if os.path.isdir(_V) and _V not in sys.path:
        sys.path.insert(0, _V)

# 控制台 UTF-8（Windows GBK 控制台直接显示中文）
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import config
from common_utils import get_logger
from cloud_model_client import CloudModelClient
from train_all import fetch_world_knowledge, extract_topics, build_scenarios, _SYSTEM_PROMPT

logger = get_logger("gen_persona")


def main() -> int:
    ap = argparse.ArgumentParser(description="昔涟人格训练数据生成（前台友好）")
    ap.add_argument("--count", type=int, default=200, help="生成条数（默认 200）")
    ap.add_argument("--no-cache", action="store_true", help="忽略缓存强制重新生成")
    args = ap.parse_args()

    cfg = config.get_config()
    cfg.ensure_dirs()
    client = CloudModelClient(cfg)
    if not client.available:
        print("\n[错误] 未配置 API Key（hoyotool.ini [Cloud].api_key 或 DEEPSEEK_API_KEY）")
        print("       配置后重试。\n")
        return 1

    print("=" * 64)
    print(f"  昔涟人格训练数据生成  |  目标 {args.count} 条  |  模型 {cfg.deepseek_responses_model}")
    print("=" * 64)

    # ① 知识（缓存命中秒开）
    print("\n[1/4] 联网抓取真实资料（萌娘百科: 昔涟/翁法罗斯）...")
    knowledge = fetch_world_knowledge(cfg)
    print(f"      知识上下文 {len(knowledge)} 字符"
          + ("（抓取失败，用固定话题）" if not knowledge else ""))

    # ② 场景
    topics = extract_topics(knowledge)
    scenarios = build_scenarios(args.count, topics=topics)
    print(f"[2/4] 资料驱动话题: {topics[:8]}...")
    print(f"      场景示例: {scenarios[0]['prompt']}")

    # ③ 生成（逐条实时打印）
    print(f"[3/4] 开始生成 {len(scenarios)} 条（每条含 联网搜索→作答 两轮）...\n")
    pairs, search_based, cache_hits = [], 0, 0
    t0 = time.time()
    for i, sp in enumerate(scenarios, 1):
        print("-" * 64)
        print(f"▶ [{i}/{len(scenarios)}] 场景: {sp['prompt']}  (话题: {sp['topic']})")
        res = client.generate_with_search(
            sp["prompt"], instructions=_SYSTEM_PROMPT, knowledge=knowledge,
            force_search=True, use_cache=not args.no_cache)
        if res is None:
            print("   ✗ 生成失败（API 错误），跳过")
            continue
        text = res["text"]
        if res["searched"]:
            search_based += 1
            print("   [联网搜索] 已执行 → 基于搜索结果生成")
        else:
            cache_hits += 1
            print("   [缓存命中] 复用已生成数据（本次不调用 API，不烧钱）")
        print(f"   👤 user     : {sp['prompt']}")
        print(f"   💬 assistant: {text}")
        pairs.append({"user": sp["prompt"], "assistant": text, "topic": sp["topic"]})

    # ④ 落盘 + 统计
    print("\n" + "=" * 64)
    out_path = os.path.join(cfg.cache_dir, "persona_dataset.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"count": len(pairs), "search_based": search_based,
                   "cache_hits": cache_hits, "pairs": pairs},
                  f, ensure_ascii=False, indent=1)
    elapsed = time.time() - t0
    print(f"[4/4] 完成！")
    print(f"  · 生成 {len(pairs)} 条（目标 {len(scenarios)}）")
    print(f"  · 其中 {search_based} 条经联网搜索生成, {cache_hits} 条缓存命中")
    print(f"  · 耗时 {elapsed:.0f}s（平均 {elapsed/max(len(pairs),1):.1f}s/条）")
    print(f"  · 完整数据集 → {out_path}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
