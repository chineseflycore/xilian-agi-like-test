# -*- coding: utf-8 -*-
"""
core/adapter/sse_handler.py — SSE 事件写入与 OpenAI 请求解析
============================================================
显存预估: < 10MB RAM

本模块仅为 Cyrene-Agent 对接适配层提供两个纯工具，仅依赖标准库 (json / typing):

  1. SSEEventWriter        封装 "event: <name>\\ndata: {json}\\n\\n" 写入并逐条 flush，
                          保证 Cyrene 端实时收到（规格 §五.4）。
  2. parse_openai_request  从 OpenAI 兼容请求体提取 messages / stream /
                          reasoning_effort / tools，返回归一化 dict 与错误标记。

本模块不持有任何模型 / 引擎实例，也不会在 import 时构造 AGICore。
"""
import json


# 合法 reasoning_effort 取值（缺省 low，非法值按 low 处理并记 warning）
_VALID_REASONING = {"none", "low", "medium", "high", "xhigh"}


class SSEEventWriter:
    """SSE 事件写入器。

    每个事件按 OpenAI 兼容流式格式写出:

        event: <event_name>\\n
        data:  <json>\\n\\n

    写入后立即 f.flush()，保证 Cyrene 端实时收到（规格 §五.4 硬性）。
    """

    def __init__(self, fp):
        """fp: 可写文件对象（通常为 BaseHTTPRequestHandler.wfile）。"""
        self.fp = fp

    def write(self, event: str, data: dict) -> None:
        """写出一行 event 与一行 data（ensure_ascii=False），并 flush。

        data 逐字段用 json.dumps 序列化，字段顺序在 Python 3.7+ 保持插入序。
        """
        # fp 为二进制流（BaseHTTPRequestHandler.wfile），须编码为 UTF-8 字节
        self.fp.write(("event: %s\n" % event).encode("utf-8"))
        self.fp.write(("data: " + json.dumps(data, ensure_ascii=False)
                       + "\n\n").encode("utf-8"))
        self.fp.flush()


def _extract_text(content):
    """从 OpenAI message content（str 或多模态 list）提取纯文本。

    - str 直接返回；
    - list 逐 part 抽取 text / 子 text 块并以换行拼接；
    - 其它类型返回 None（视为无 content）。
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                t = part.get("text")
                if isinstance(t, str):
                    parts.append(t)
                elif isinstance(t, list):
                    # 部分多模态把 text 进一步包成 list[{"type":"text","text":...}]
                    for sub in t:
                        if isinstance(sub, dict) and isinstance(sub.get("text"), str):
                            parts.append(sub["text"])
        return "\n".join(parts)
    return None


def parse_openai_request(body: dict) -> dict:
    """从 OpenAI 兼容请求体提取必要参数并归一化（规格 §5.1 硬性）。

    返回 dict 字段:
      messages        list | None      原始 messages（仅当合法时非 None）
      text            str | None       最后一条消息的文本（无 content 时 None）
      stream          bool             是否流式（缺省 False）
      reasoning_effort str            归一化后的推理强度（缺省 "low"）
      reasoning_warn  str | None       非法 reasoning_effort 的告警
      tools           list | None      工具数组（透传，缺省 None）
      error           dict | None      存在则须拒绝请求: {message,type,code}
    """
    req = {
        "messages": None,
        "text": None,
        "stream": False,
        "reasoning_effort": "low",
        "reasoning_warn": None,
        "tools": None,
        "error": None,
    }
    if not isinstance(body, dict):
        req["error"] = {
            "message": "Invalid request body: expected a JSON object.",
            "type": "invalid_request_error",
            "code": "invalid_messages",
        }
        return req

    # ---- messages 标准数组 ----
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        req["error"] = {
            "message": "Invalid request: 'messages' must be a non-empty array.",
            "type": "invalid_request_error",
            "code": "invalid_messages",
        }
        return req

    last = messages[-1]
    if not isinstance(last, dict):
        req["error"] = {
            "message": "Invalid request: last message must be an object.",
            "type": "invalid_request_error",
            "code": "invalid_messages",
        }
        return req

    # ---- 取最后一条 content 作为文本 ----
    text = _extract_text(last.get("content"))
    if text is None or text.strip() == "":
        req["error"] = {
            "message": "Invalid request: last message has no content.",
            "type": "invalid_request_error",
            "code": "invalid_messages",
        }
        return req

    req["messages"] = messages
    req["text"] = text.strip()

    # ---- stream（缺省 False）----
    req["stream"] = bool(body.get("stream", False))

    # ---- reasoning_effort（缺省 "low"，非法值回退 "low" 并记 warning）----
    re_val = body.get("reasoning_effort")
    if re_val is None:
        re_val = "low"
    elif not isinstance(re_val, str) or re_val not in _VALID_REASONING:
        req["reasoning_warn"] = (
            "invalid reasoning_effort %r; falling back to 'low'" % (re_val,))
        re_val = "low"
    req["reasoning_effort"] = re_val

    # ---- tools（数组透传给 DeepSeek）----
    tools = body.get("tools")
    req["tools"] = tools if isinstance(tools, list) else None

    return req
