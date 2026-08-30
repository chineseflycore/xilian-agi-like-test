# -*- coding: utf-8 -*-
"""
data_crawler.py — 爬虫模块（萌娘百科 / NGA / TAPTAP，规范 §6.2）
==========================================================================
内存/显存预估（规范 §九.1）:
  · 无模型权重常驻；仅少量字符串/列表缓冲与逐条 jsonl 记录，
    峰值占用 < 20MB RAM，GPU 显存占用 0（纯网络调用/离线流程验证）。
  · scrapy / requests 均为惰性导入（safe_import）：未安装时本模块照常
    import，不引入任何第三方依赖内存开销。

职责（规范 §6.2 爬虫模块）:
  1. 目标站点: 萌娘百科(zh.moegirl.org.cn)、NGA(nga.178.com / bbs.nga.cn)、
     TAPTAP(taptap.cn / www.taptap.cn)。
     关键词: 昔涟、翁法罗斯、黄金裔 —— 从 cfg.name / cfg.world_context 读取
     可扩展（见 _default_keywords()）。
  2. Scrapy Spider 类: MoeBaikeSpider / NgaSpider / TapTapSpider
     （继承 scrapy.Spider；start_urls 用关键词构造搜索 URL；parse 提取
     标题/正文片段）。类定义放在 build_spiders() 类工厂内（try 保护，规范
     §九.2 惰性导入）—— scrapy 未安装时模块顶层不崩，返回 {} 并打黄色降级警告。
  3. 降级路径 fetch_fallback(query): requests 可用 → 逐站点简单抓取搜索页
     文本并提取记录；requests 不可用/离线 → 返回 [] 并打黄色警告
     （说明需安装 scrapy/requests 并联网，规范 §九.2）。
  4. 统一输出格式:
       {"source": str, "url": str, "title": str, "text": str, "keywords": [str]}
     保存为 data_cache 目录下的 jsonl 文件（路径基于 BASE_DIR，禁止裸相对
     路径，规范 §九.12/§十.24）。
  5. 预演模式 preview(): 无网环境下演示完整管线与统一数据格式，生成占位
     记录（URL 为真实构造的搜索 URL；降级/预演数据仅用于流程验证，不用于
     实际训练，规范 §十.22）。

TODO(V3.4): 站点搜索端点可能随改版变化；接入反爬策略与下载限速；
            增加基于 url 哈希的增量去重与正文深度解析。
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
from urllib.parse import quote, urljoin

from config import get_config, BASE_DIR
from common_utils import get_logger, safe_import

logger = get_logger("data_crawler")

# ------------------------------------------------------------------
# 目标站点配置（规范 §6.2）
# ------------------------------------------------------------------
SITES = {
    "moegirl": {
        "name": "萌娘百科",
        "domains": ["zh.moegirl.org.cn", "moegirl.org.cn"],
        # MediaWiki 搜索端点: ?search=<关键字>
        "search_url": "https://zh.moegirl.org.cn/index.php?search={query}",
    },
    "nga": {
        "name": "NGA",
        "domains": ["nga.178.com", "bbs.nga.cn", "nga.cn"],
        # NGA 论坛搜索端点: ?search=<关键字>（端点可能随站点改版变化，仅作模板）
        "search_url": "https://bbs.nga.cn/thread.php?search={query}",
    },
    "taptap": {
        "name": "TAPTAP",
        "domains": ["taptap.cn", "www.taptap.cn"],
        # TAPTAP 搜索端点: ?kw=<关键字>
        "search_url": "https://www.taptap.cn/search?kw={query}",
    },
}

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36 PhiLia093/0.1")
TEXT_MAX = 500            # 正文片段最大长度（字符）
LINK_TEXT_MAX = 200       # 单条链接文本最大长度（字符）
DOWNLOAD_TIMEOUT_S = 15   # 降级路径 HTTP 超时（秒）


# ------------------------------------------------------------------
# 关键词与文本提取（scrapy / requests 降级路径共用，规范 §6.2）
# ------------------------------------------------------------------
def _default_keywords(cfg=None):
    """默认关键词: 昔涟（cfg.name）+ 翁法罗斯（cfg.world_context）+ 黄金裔。

    从配置读取可扩展（规范 §6.2 关键词，§8.1 配置集中管理）；去空去重。
    """
    cfg = cfg or get_config()
    out = []
    for k in (cfg.name, cfg.world_context, "黄金裔"):
        k = str(k).strip()
        if k and k not in out:
            out.append(k)
    return out


def _strip_tags(html):
    """去除 script/style 块与全部标签，折叠空白（正文片段提取兜底工具）。"""
    html = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html or "")
    html = re.sub(r"(?is)<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", html).strip()


def _extract_html(html, source, base_url, keywords):
    """从 HTML 文本提取统一格式记录（规范 §6.2 统一输出格式）。

    提取规则:
      1) 页面标题: <title> 标签内容
      2) 关键词命中的 <a> 链接: 链接文本含任一关键词 → 一条记录
         （url 经 urljoin 解析为绝对地址，正文片段取链接文本）
      3) 页面级正文片段兜底: 整页可见文本（TEXT_MAX 截断），保证至少一条
    返回: [{"source", "url", "title", "text", "keywords"}]（按 url 去重）
    """
    html = html or ""
    keywords = [str(k) for k in (keywords or []) if str(k).strip()]
    records, seen = [], set()

    def push(url, title, text, kws):
        if not url or url in seen:
            return
        seen.add(url)
        records.append({
            "source": source,
            "url": url,
            "title": (title or "").strip()[:120],
            "text": (text or "").strip()[:TEXT_MAX],
            "keywords": kws or keywords,
        })

    # 1) 页面标题
    m = re.search(r"(?is)<title[^>]*>(.*?)</title>", html)
    page_title = _strip_tags(m.group(1)) if m else source

    # 2) 关键词命中的链接（标题/正文片段取自链接文本）
    for m in re.finditer(
            r'(?is)<a\b[^>]*\bhref\s*=\s*["\']([^"\']+)["\'][^>]*>(.*?)</a>', html):
        href, inner = m.group(1), _strip_tags(m.group(2))
        if not inner:
            continue
        hit = [k for k in keywords if k in inner or k in href]
        if not hit:
            continue
        push(urljoin(base_url, href), inner[:LINK_TEXT_MAX], inner, hit)

    # 3) 页面级正文片段兜底（确保至少一条记录）
    push(base_url, page_title, _strip_tags(html), keywords)
    return records


# ------------------------------------------------------------------
# Scrapy Spider 类工厂（惰性导入，规范 §6.2 / §九.2）
# ------------------------------------------------------------------
def build_spiders(keywords=None):
    """Scrapy Spider 类工厂。

    scrapy 未安装 → 打印黄色降级警告并返回 {}（模块顶层永不触发 scrapy）。
    scrapy 已安装 → 返回:
        {"moegirl": MoeBaikeSpider, "nga": NgaSpider, "taptap": TapTapSpider}
    类定义放在本函数内（try 保护），保证 import data_crawler 永远安全。
    """
    kws = _default_keywords() if keywords is None else [str(k) for k in keywords]
    scrapy = safe_import("scrapy")   # 惰性导入，失败返回 None
    if scrapy is None:
        logger.warning(
            "\033[33m[DEGRADE] scrapy 未安装 → build_spiders() 返回 {}，无法提供 "
            "Spider 类；请安装 scrapy 并联网后重试（规范 §6.2/§九.2 降级提示）\033[0m")
        return {}
    try:
        class MoeBaikeSpider(scrapy.Spider):
            """萌娘百科爬虫: 搜索页 → 提取标题/正文片段（规范 §6.2）。"""

            name = "moegirl"
            allowed_domains = SITES["moegirl"]["domains"]
            start_urls = [SITES["moegirl"]["search_url"].format(query=quote(k)) for k in kws]

            def parse(self, response):
                # 标题/正文片段提取（统一格式，规范 §6.2）
                yield from _extract_html(response.text, self.name, response.url, kws)

        class NgaSpider(scrapy.Spider):
            """NGA 爬虫（nga.178.com / bbs.nga.cn，规范 §6.2）。"""

            name = "nga"
            allowed_domains = SITES["nga"]["domains"]
            start_urls = [SITES["nga"]["search_url"].format(query=quote(k)) for k in kws]

            def parse(self, response):
                yield from _extract_html(response.text, self.name, response.url, kws)

        class TapTapSpider(scrapy.Spider):
            """TAPTAP 爬虫（taptap.cn / www.taptap.cn，规范 §6.2）。"""

            name = "taptap"
            allowed_domains = SITES["taptap"]["domains"]
            start_urls = [SITES["taptap"]["search_url"].format(query=quote(k)) for k in kws]

            def parse(self, response):
                yield from _extract_html(response.text, self.name, response.url, kws)

        return {"moegirl": MoeBaikeSpider, "nga": NgaSpider, "taptap": TapTapSpider}
    except Exception as e:
        logger.warning(
            "\033[33m[DEGRADE] Spider 类定义失败(%s) → 返回 {}，走降级路径\033[0m", e)
        return {}


# ------------------------------------------------------------------
# 降级路径（requests 简单抓取，规范 §6.2 / §九.2）
# ------------------------------------------------------------------
def _http_get_text(requests_mod, url, timeout=DOWNLOAD_TIMEOUT_S):
    """简单 GET 搜索页文本（降级路径）。失败返回 None，不抛出。"""
    try:
        resp = requests_mod.get(url, timeout=timeout, headers={"User-Agent": UA})
        if resp.status_code != 200:
            logger.warning("[HTTP] %s 返回 HTTP %d", url, resp.status_code)
            return None
        # 中文站点编码兜底（apparent_encoding 可能为 None）
        resp.encoding = resp.apparent_encoding or resp.encoding or "utf-8"
        return resp.text
    except Exception as e:
        logger.warning("[HTTP] 抓取失败 %s: %s", url, e)
        return None


def fetch_fallback(query, keywords=None, save=False):
    """降级抓取（规范 §6.2 / §九.2）。

    参数:
        query:    单个搜索关键词（如 cfg.name "昔涟"）
        keywords: 可选扩展关键词列表（缺省用 _default_keywords()）
        save:     True 时将结果写入 data_cache jsonl

    返回:
        统一格式记录列表；requests 不可用或离线 → [] 并打印黄色警告
        （说明需安装 scrapy/requests 并联网）。
    """
    requests_mod = safe_import("requests")   # 惰性导入，失败返回 None
    if requests_mod is None:
        logger.warning(
            "\033[33m[DEGRADE] requests 未安装 → fetch_fallback() 返回空；"
            "请安装 scrapy/requests 并联网后重试（规范 §6.2 爬虫模块降级提示）\033[0m")
        return []
    kws = list(keywords) if keywords else _default_keywords()
    query = str(query).strip() or (kws[0] if kws else "")
    records, seen = [], set()
    for key, site in SITES.items():
        url = site["search_url"].format(query=quote(query))
        html = _http_get_text(requests_mod, url)
        if not html:
            continue
        for rec in _extract_html(html, key, url, [query] + kws):
            if rec["url"] not in seen:
                seen.add(rec["url"])
                records.append(rec)
    if not records:
        logger.warning(
            "\033[33m[DEGRADE] 搜索页无可提取内容或网络不可达 → 返回空；"
            "可改用 preview() 预演模式验证管线（规范 §6.2）\033[0m")
    if save and records:
        save_records(records)
    return records


# ------------------------------------------------------------------
# 统一输出: jsonl 落盘（路径基于 BASE_DIR，规范 §九.12/§十.24）
# ------------------------------------------------------------------
def _data_dir(cfg=None):
    """data_cache 目录（基于 BASE_DIR，禁止裸相对路径）。"""
    cfg = cfg or get_config()
    d = os.path.join(BASE_DIR, "data_cache")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception as e:
        logger.warning("[DEGRADE] 创建 data_cache 目录失败(%s)，回退到 cfg.cache_dir", e)
        d = cfg.cache_dir
        os.makedirs(d, exist_ok=True)
    return d


def save_records(records, filename=None, cfg=None, mode="a"):
    """统一格式记录 → jsonl 追加写（一条 JSON 一行，ensure_ascii=False）。

    参数:
        filename: 仅允许纯文件名（禁止路径穿越；实际路径 = data_cache/filename）
        mode:     "a" 追加（真实抓取日志，跨运行累积）/ "w" 覆盖（预演文件，每次重建）
    返回: 写入文件的绝对路径；写入失败 → None（降级不抛出，规范 §九.2）。
    """
    if mode not in ("a", "w"):
        raise ValueError("mode 仅支持 'a'/'w': %r" % mode)
    cfg = cfg or get_config()
    name = filename or "crawled_data.jsonl"
    if os.path.basename(name) != name:
        raise ValueError("filename 必须是纯文件名（路径基于 data_cache 目录）: %r" % name)
    path = os.path.join(_data_dir(cfg), name)
    n = 0
    try:
        with open(path, mode, encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
    except Exception as e:
        logger.warning(
            "\033[33m[DEGRADE] jsonl 写入失败(%s) → 降级返回 None（跳过保存）\033[0m", e)
        return None
    logger.info("已保存 %d 条记录 → %s", n, path)
    return path


# ------------------------------------------------------------------
# 预演模式（无网流程验证，规范 §6.2 / §十.22）
# ------------------------------------------------------------------
def preview(keywords=None, save=True, cfg=None):
    """预演模式: 不联网、不依赖 scrapy/requests，按统一输出格式生成占位记录。

    · URL 为真实构造的搜索 URL（展示实际会抓什么）
    · text 标记【预演占位数据，非真实抓取】
    · 保存到 data_cache/crawled_preview.jsonl（预演数据仅用于流程验证，
      不用于实际训练，规范 §十.22）
    """
    cfg = cfg or get_config()
    kws = _default_keywords(cfg) if keywords is None else [str(k) for k in keywords]
    records = []
    for key, site in SITES.items():
        for kw in kws:
            url = site["search_url"].format(query=quote(kw))
            records.append({
                "source": key,
                "url": url,
                "title": "【预演】%s 搜索结果: %s" % (site["name"], kw),
                "text": ("【预演占位数据，非真实抓取】关键词=%s；%s"
                         "（预演数据仅用于流程验证，不用于实际训练，规范 §十.22）"
                         % (kw, cfg.anchor_generation_prompt)),
                "keywords": [kw],
            })
    if save:
        # mode="w": 预演文件每次重建，避免跨运行累积重复（真实抓取用 mode="a"）
        save_records(records, filename="crawled_preview.jsonl", cfg=cfg, mode="w")
    return records


# ------------------------------------------------------------------
# 抓取主入口（规范 §6.2）
# ------------------------------------------------------------------
def _run_scrapy_crawl(spiders, keywords, save=True):
    """进程内实际运行 scrapy 爬虫（需联网；仅当 scrapy 已安装且可联网时使用）。

    通过 item_scraped 信号收集统一格式记录；失败由调用方 try-except 兜底
    （规范 §九.2 关键路径降级）。
    """
    from scrapy.crawler import CrawlerProcess
    from scrapy import signals
    collected = []

    def _on_item(item, spider):          # noqa: ANN001
        collected.append(dict(item))

    process = CrawlerProcess(settings={
        "USER_AGENT": UA,
        "ROBOTSTXT_OBEY": False,
        "LOG_LEVEL": "WARNING",
        "DOWNLOAD_TIMEOUT": DOWNLOAD_TIMEOUT_S,
        "RETRY_ENABLED": False,
    })
    for cls in spiders.values():
        crawler = process.create_crawler(cls)
        crawler.signals.connect(_on_item, signal=signals.item_scraped)
        process.crawl(crawler)
    process.start()   # 阻塞直到全部爬完（需联网）
    logger.info("scrapy 实跑完成，收集 %d 条记录", len(collected))
    if save and collected:
        save_records(collected)
    return collected


def crawl(keywords=None, save=True, cfg=None):
    """抓取主入口（规范 §6.2）:
      1) scrapy 可用 → 进程内实跑 CrawlerProcess（需联网）；
      2) 否则逐关键词 fetch_fallback（requests 可用时）；
      3) 均失败 → 黄色警告并返回 []（可改用 preview() 预演）。
    """
    cfg = cfg or get_config()
    kws = _default_keywords(cfg) if keywords is None else list(keywords)
    spiders = build_spiders(kws)
    if spiders:
        try:
            return _run_scrapy_crawl(spiders, kws, save=save)
        except Exception as e:
            logger.warning("[DEGRADE] scrapy 实跑失败(%s) → 回退 requests 降级路径", e)
    records = []
    for kw in kws:
        records.extend(fetch_fallback(kw, keywords=kws, save=False))
    if not records:
        logger.warning(
            "\033[33m[DEGRADE] 抓取结果为空（无 scrapy/requests 或离线）；"
            "请安装 scrapy 并联网，或调用 preview() 预演模式（规范 §6.2）\033[0m")
    if save and records:
        save_records(records)
    return records


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 自检（规范 §九.4 带时间戳启动日志）: 本机无 scrapy/requests →
    # 打印降级说明并演示预演模式；正常结束退出码 0。
    from datetime import datetime

    _cfg = get_config()
    print("[%s] [BOOT] data_crawler 自检开始 (name=%s, world_context=%s)" % (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"), _cfg.name, _cfg.world_context))

    _kws = _default_keywords(_cfg)
    print("[SELFTEST] 关键词(由 cfg.name/world_context 扩展): %s" % _kws)

    _spiders = build_spiders(_kws)
    print("[SELFTEST] build_spiders() → %s（本机无 scrapy → 黄色降级警告已打印）" %
          (list(_spiders.keys()) if _spiders else "{}"))

    _recs = fetch_fallback(_cfg.name)
    if _recs:
        print("[SELFTEST] fetch_fallback(%r) → %d 条（requests 可用，已实际抓取搜索页文本，"
              "规范 §6.2 降级路径生效）" % (_cfg.name, len(_recs)))
    else:
        print("[SELFTEST] fetch_fallback(%r) → 0 条（requests 不可用/离线 → 黄色警告已打印）"
              % _cfg.name)

    _prev = preview(save=True, cfg=_cfg)
    _prev_path = os.path.join(_data_dir(_cfg), "crawled_preview.jsonl")
    print("[SELFTEST] preview() → %d 条预演记录，字段: %s" %
          (len(_prev), list(_prev[0].keys()) if _prev else []))
    print("[SELFTEST] 预演 jsonl 已写入: %s" % _prev_path)
    if _prev:
        print("[SELFTEST] 示例记录: %s" % json.dumps(_prev[0], ensure_ascii=False))
    print("[SELFTEST] 自检通过，退出码 0")
