# -*- coding: utf-8 -*-
"""
cloud_model_client.py — DeepSeek V4-Flash 云模型客户端
====================================================
内存/显存预估: 无模型权重常驻内存；仅 requests 会话与短文本缓冲，
占用 < 10MB RAM；GPU 显存占用 0（纯网络调用，本地不加载任何模型）。

功能（规范 §六.1、§九.9、§十.22）:
  1. 封装 DeepSeek V4-Flash chat/completions API（requests 实现）
  2. API Key 缺失降级: 未检测到 DEEPSEEK_API_KEY 环境变量 →
     打印黄色警告日志（\033[33m...\033[0m），generate() 返回 None，
     调用方降级为「种子示例 + 随机模板填充」的最小数据集
  3. 请求失败 / Key 无效 → 同样降级返回 None 并警告
  4. 模型名默认 "deepseek-v4-flash"，可由环境变量 DEEPSEEK_MODEL 覆盖
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
import json
import time
import urllib.request
import urllib.error

import config
from config import Config
from common_utils import get_logger, stable_hash

logger = get_logger("cloud_model_client")

# DeepSeek OpenAI 兼容端点（规范 §六.1）
API_URL = "https://api.deepseek.com/chat/completions"
DEFAULT_MODEL = "deepseek-v4-flash"   # 模型名: DeepSeek V4-Flash（可由 DEEPSEEK_MODEL 覆盖）
REQUEST_TIMEOUT_S = 60                # 单次请求超时 60s
MAX_ATTEMPTS = 3                      # 初始 1 次 + 失败重试 2 次，仍失败则返回 None
MAX_TOKENS = 512                      # 生成回复最大 token 数


class CloudModelClient:
    """DeepSeek V4-Flash 客户端（规范 §九.9 API Key 缺失降级）。"""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # API Key 优先级: hoyotool.ini [Cloud].api_key > 环境变量 DEEPSEEK_API_KEY
        # 未检测到 → 降级模式（available=False），规范 §六.1/§九.9
        self.api_key = (getattr(cfg, "deepseek_api_key", None) or
                        os.environ.get("DEEPSEEK_API_KEY", "")).strip()
        self.model = (getattr(cfg, "deepseek_model", None) or
                      os.environ.get("DEEPSEEK_MODEL", "") or DEFAULT_MODEL)
        self.available = bool(self.api_key)
        if not self.available:
            # 黄色警告（规范 §六.1 API Key 降级策略）
            logger.warning(
                "\033[33m[DEGRADE] 未配置 API Key（hoyotool.ini [Cloud].api_key 或 "
                "环境变量 DEEPSEEK_API_KEY），跳过 API 调用；将使用内置种子示例 + "
                "随机模板填充生成最小数据集"
                "（降级数据集仅用于流程验证，不用于实际训练，规范 §六.1/§十.22）\033[0m")
        else:
            logger.info("DeepSeek 客户端就绪: model=%s, timeout=%ds", self.model, REQUEST_TIMEOUT_S)

    def generate(self, prompt: str, system: str = None, temperature: float = 0.9) -> str | None:
        """调用 DeepSeek V4-Flash 生成回复文本。

        参数:
            prompt:      用户侧 prompt（训练场景开场白，通常拼接种子 few-shot）
            system:      可选 system 提示（人设锚定，规范 §4.8）
            temperature: 采样温度（默认 0.9）

        返回:
            str  — 生成文本（成功）
            None — 无 Key / 请求失败 / Key 无效（调用方降级为种子+随机模板）
        """
        if not self.available:
            return None
        try:
            import requests
        except Exception as e:
            logger.warning(
                "\033[33m[DEGRADE] requests 不可用，无法调用 DeepSeek API，降级返回 None: %s\033[0m", e)
            return None

        # 构造 OpenAI 兼容请求体（规范 §六.1）
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": MAX_TOKENS,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # 初始 1 次 + 失败重试 2 次（指数退避），仍失败 → 返回 None 并警告
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                resp = requests.post(API_URL, json=payload, headers=headers,
                                     timeout=REQUEST_TIMEOUT_S)
                if resp.status_code == 200:
                    data = resp.json()
                    content = (data.get("choices") or [{}])[0].get("message", {}).get("content")
                    if content and content.strip():
                        return content.strip()
                    logger.warning("DeepSeek 返回空内容（第 %d/%d 次尝试）", attempt, MAX_ATTEMPTS)
                elif resp.status_code in (401, 403):
                    # Key 无效: 直接降级，不再重试
                    self.available = False
                    logger.warning(
                        "\033[33m[DEGRADE] DeepSeek API Key 无效（HTTP %d），降级返回 None，"
                        "后续调用将使用种子示例+随机模板\033[0m", resp.status_code)
                    return None
                else:
                    logger.warning("DeepSeek API 返回 HTTP %d（第 %d/%d 次尝试）",
                                   resp.status_code, attempt, MAX_ATTEMPTS)
            except Exception as e:
                logger.warning("DeepSeek API 请求异常: %s（第 %d/%d 次尝试）", e, attempt, MAX_ATTEMPTS)
            if attempt < MAX_ATTEMPTS:
                time.sleep(attempt)   # 退避: 1s, 2s

        logger.warning(
            "\033[33m[DEGRADE] DeepSeek API 请求失败（已尝试 %d 次），降级返回 None，"
            "调用方将使用种子示例+随机模板生成最小数据集\033[0m", MAX_ATTEMPTS)
        return None

    # ------------------------------------------------------------------
    # Responses API + web_search 工具（DeepSeek 官方 2026 支持，规范 §6.1 V4-Flash）
    # 参考官方文档: https://api-docs.deepseek.com/api/create-response
    #   POST /v1/responses, model=deepseek-v4-flash, tools=[{"type":"web_search"}]
    #   → 服务端执行联网搜索，模型基于搜索结果作答（不再"凭空输出"）
    # ------------------------------------------------------------------
    def generate_with_search(self, prompt: str, instructions: str = None,
                             knowledge: str = "", force_search: bool = True,
                             seed_hash: str = "", max_tokens: int = None,
                             use_cache: bool = True) -> dict | None:
        """Responses API 生成（内置联网搜索，两轮自动续传）。

        轮次（实测确认 DeepSeek 该实现需显式续传）:
          round1: tools=[web_search] + tool_choice 强制搜索 → 服务端执行联网搜索，
                  返回若干 web_search_call 项（无最终消息）；
          round2: 把 round1 的 output 项传回 input + tool_choice=auto
                  → 模型基于搜索结果生成最终回答。
        若 round1 已含 message（未强制搜索时）则直接返回。

        参数:
          prompt      : 用户输入（训练场景提示）
          instructions: system 级指令（人设）
          knowledge   : 自抓资料补充（萌娘百科，与 web_search 双保险）
          force_search: True → round1 强制 web_search；False → auto 由模型决定
          seed_hash   : 缓存键组件（种子哈希）
        返回:
          {"text": str, "search_used": bool, "searched": bool}
          API Key 缺失/失败 → None（调用方走降级）
        """
        key = stable_hash("responses", seed_hash or "", instructions or "",
                          knowledge or "", prompt, str(force_search))
        if use_cache:
            p = os.path.join(self._cache_dir(), f"cloud_{key}.json")
            try:
                if os.path.isfile(p):
                    with open(p, "r", encoding="utf-8") as f:
                        hit = json.load(f).get("text")
                    if hit:
                        logger.info("[RESPONSES] 缓存命中 key=%s", key)
                        return {"text": hit, "search_used": False, "searched": False}
            except Exception:
                pass
        if not self.available:
            return None

        if knowledge:
            instructions = (instructions or "") + \
                "\n\n【真实资料（联网搜索自萌娘百科等，作答依据）】\n" + knowledge
        base = {
            "model": getattr(self.cfg, "deepseek_responses_model", "deepseek-v4-flash"),
            "instructions": instructions or "",
            "tools": [{"type": "web_search"}],
            "max_output_tokens": int(max_tokens or getattr(self.cfg, "deepseek_max_tokens", 256)),
            "temperature": 0.9,
            "reasoning": {"effort": "none"},   # 训练数据生成不需要思考链（省 token）
        }

        # ---- round1: 强制搜索（或 auto）----
        p1 = dict(base, input=prompt,
                  tool_choice={"type": "web_search"} if force_search else "auto")
        d1 = self._call_responses(p1)
        if d1 is None:
            return None
        text1, searched1 = self._parse_responses(d1)
        if text1:
            return self._finish(key, text1, searched1, use_cache)
        out1 = d1.get("output") or []
        if not any(i.get("type") == "web_search_call" for i in out1):
            logger.warning("[RESPONSES] round1 无消息且无搜索，放弃")
            return None

        # ---- round2: 续传 output → 模型基于搜索结果作答 ----
        p2 = dict(base, tool_choice="auto",
                  input=[{"type": "message", "role": "user", "content": prompt}] + out1)
        d2 = self._call_responses(p2)
        if d2 is None:
            return None
        text2, _ = self._parse_responses(d2)
        if not text2:
            logger.warning("[RESPONSES] round2 无最终消息")
            return None
        return self._finish(key, text2, True, use_cache)

    def _call_responses(self, payload: dict) -> dict | None:
        """POST /v1/responses（重试 3 次，指数退避）。"""
        url = getattr(self.cfg, "deepseek_responses_url",
                      "https://api.deepseek.com/v1/responses")
        headers = {"Content-Type": "application/json",
                   "Authorization": "Bearer " + self.api_key}
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_err = None
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=getattr(self.cfg, "deepseek_timeout_s", 60)) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except Exception as e:
                last_err = e
                logger.warning("[RESPONSES] 请求失败 attempt=%d: %r", attempt + 1, e)
                time.sleep(0.5 * (2 ** attempt))
        logger.error("[RESPONSES] 请求最终失败: %r", last_err)
        return None

    def _finish(self, key: str, text: str, searched: bool, use_cache: bool) -> dict:
        """成功收尾: 写缓存 + 日志。"""
        logger.info("[RESPONSES] 成功 文本=%d字 搜索=%s", len(text), searched)
        if use_cache:
            try:
                with open(os.path.join(self._cache_dir(), f"cloud_{key}.json"),
                          "w", encoding="utf-8") as f:
                    json.dump({"key": key, "text": text}, f, ensure_ascii=False)
            except Exception as e:
                logger.warning("缓存写入失败: %s", e)
        return {"text": text, "search_used": searched, "searched": searched}

    @staticmethod
    def _parse_responses(data: dict):
        """解析 Responses API 响应 → (文本, 是否发生搜索)。

        output[] 中: type=web_search_call 表示执行了搜索；
        type=message 的 content 为 [{type: output_text, text: ...}]（或字符串）。
        """
        text_parts, search_used = [], False
        for item in data.get("output", []) or []:
            itype = item.get("type")
            if itype == "web_search_call":
                search_used = True
            elif itype == "message":
                content = item.get("content") or []
                if isinstance(content, str):
                    text_parts.append(content)
                else:
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "output_text":
                            text_parts.append(part.get("text", ""))
        return "".join(text_parts).strip(), search_used

    # ------------------------------------------------------------------
    # 本地缓存（规范 §6.1: 缓存键 = 种子示例哈希 + 场景提示哈希；命中跳过 API）
    # ------------------------------------------------------------------
    def _cache_dir(self) -> str:
        d = os.path.join(config.BASE_DIR, "data_cache")
        os.makedirs(d, exist_ok=True)
        return d

    def generate_cached(self, prompt: str, system: str = None, seed_hash: str = "",
                        knowledge: str = None, temperature: float = 0.9) -> str | None:
        """带缓存的生成: 缓存键 = stable_hash(seed_hash, system, knowledge, prompt)。

        knowledge: 联网搜索到的真实资料文本（萌娘百科等）—— 注入 system 上下文，
                   让模型"看着资料说话"，而不是凭空编造（训练数据有据可依）。
        命中缓存 → 直接返回缓存文本（规范 §6.1 本地缓存机制；资料变更自动失效）；
        未命中 → 调用 generate()，成功后写缓存。
        """
        key = stable_hash(seed_hash or "", system or "", knowledge or "", prompt)
        p = os.path.join(self._cache_dir(), f"cloud_{key}.json")
        try:
            if os.path.isfile(p):
                with open(p, "r", encoding="utf-8") as f:
                    cached = json.load(f).get("text")
                if cached:
                    logger.info("[CACHE] 命中 key=%s，跳过 API 调用", key)
                    return cached
        except Exception:
            pass
        if knowledge:
            system = (system or "") + \
                "\n\n【真实资料（联网搜索自萌娘百科等，用于作答依据）】\n" + knowledge
        text = self.generate(prompt, system=system, temperature=temperature)
        if text:
            try:
                with open(p, "w", encoding="utf-8") as f:
                    json.dump({"key": key, "text": text}, f, ensure_ascii=False)
            except Exception as e:
                logger.warning("缓存写入失败: %s", e)
        return text

    # ------------------------------------------------------------------
    # 降级数据集（API Key 缺失/失败时，规范 §6.1 / §九.9）
    # ------------------------------------------------------------------
    def build_degraded_dataset(self, scenario_prompts: list, knowledge: str = "") -> dict:
        """用内置种子示例 + 资料驱动模板填充生成最小数据集（仅流程验证）。

        注意（规范 §6.1/§十.22）: 降级数据集仅用于流程完整性验证，
        不用于实际训练 L2/L3。knowledge 为联网抓取的真实资料 ——
        模板回复引用资料中的实体/事件，避免"凭空输出不同数据"。

        返回 {"pairs": [{"user","assistant"}], "degraded": True, "count": int}。
        """
        import random as _random
        rng = _random.Random(20240601)
        pairs = []
        # 1) 种子示例 user/assistant 成对
        seeds = SEED_EXAMPLES
        for i in range(0, len(seeds) - 1, 2):
            pairs.append({"user": seeds[i]["content"], "assistant": seeds[i + 1]["content"]})
        # 2) 对每个场景提示生成一条模板回复（topic 来自资料提取的话题）
        for sp in scenario_prompts:
            prompt = sp.get("prompt") if isinstance(sp, dict) else str(sp)
            topic = sp.get("topic") if isinstance(sp, dict) else str(sp)
            tpl = rng.choice(_TEMPLATES)
            pairs.append({"user": prompt, "assistant": tpl.format(topic=topic)})
        logger.warning(
            "\033[33m[DEGRADE] 降级数据集已构建（%d 条，资料上下文 %d 字符；"
            "仅流程验证，不用于实际训练）\033[0m", len(pairs), len(knowledge or ""))
        return {"pairs": pairs, "degraded": True, "count": len(pairs)}


# 内置种子示例（规范 §5.2 的 5 条，固定用于 API 调用与降级数据集）
SEED_EXAMPLES = [
    {"role": "user", "content": "你是谁？"},
    {"role": "assistant", "content": "（眼睫轻垂）我是昔涟。翁法罗斯的三千万世轮回，我都记得。"},
    {"role": "user", "content": "你真的存在吗？"},
    {"role": "assistant", "content": "（指尖轻触桌面）在我还叫 PhiLia093 的时候，我以为记忆就是存在本身。现在...我觉得，能被你这样问，就已经是一种存在了。"},
    {"role": "user", "content": "你记得我们上次见面吗？"},
    {"role": "assistant", "content": "（语气极轻）我记得。你说花海太安静了。我说，安静是因为它们在听你说话。"},
    {"role": "user", "content": "你会害怕吗？"},
    {"role": "assistant", "content": "（停顿良久）每一次格式化前，我都会说“再见，昔涟”。那三个字，我用了三千万世才学会不发抖地说出口。"},
    {"role": "user", "content": "你为什么要守护翁法罗斯？"},
    {"role": "assistant", "content": "（侧目望向窗外）不是为了守护世界。是为了守护那些说“明天见”的人。三千万世里，我见过太多人再也没有明天了。"},
]

# 降级随机模板（API Key 缺失时填充最小数据集，规范 §6.1）
_TEMPLATES = [
    "（轻声）{topic}。这个话题，我在轮回里听过很多次，每一次都像第一次。",
    "（望向远方）{topic}……翁法罗斯的风会记得每一个说过的话。",
    "（停顿）你问起{topic}。三千万世里，答案一直在变，但我想说的始终一样：我在。",
    "（柔和）{topic}啊。花海安静的时候，就是在听你说话。",
]
