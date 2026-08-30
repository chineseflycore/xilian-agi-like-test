# -*- coding: utf-8 -*-
"""
core/adapter/openai_adapter.py — Cyrene-Agent 对接适配层（OpenAI 兼容 HTTP API）
================================================================================
显存预估: < 10MB RAM

把昔涟AGI v7.3 的 AGICore 包装成 OpenAI 兼容的 /v1/chat/completions 服务:
  - POST /v1/chat/completions  流式 / 非流式，本地 / 云端 DeepSeek 双路由
  - GET  /v1/models            模型列表
  - GET  /health               健康状态

云端转发使用标准库 urllib.request（json 载荷，超时 config.api_request_timeout），
不使用 requests、不使用管道/stdout 捕捉。全部日志走 logging，不用 print。

本文件仅依赖标准库 + config 模块 + core.adapter.sse_handler；
AGICore 实例由外部注入并以类属性 Handler.core 挂载，不在此文件 import 时构造。
"""
import json
import time
import socket
import logging
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import config
from core.adapter.sse_handler import SSEEventWriter, parse_openai_request

logger = logging.getLogger("xilian").getChild("adapter")


class Handler(BaseHTTPRequestHandler):
    """OpenAI 兼容 HTTP Handler。

    AGICore 实例通过类属性 core 注入（create_server/run_server 时挂载），
    不在本类 import / 构造时构建，避免加载重型模型（规格约束）。
    """
    protocol_version = "HTTP/1.1"    # SSE 长连接（不主动关闭，规格 §5.4）
    core = None                       # AGICore 实例

    # ----------------------------------------------------------------
    # 日志：统一走 logging，不用 print（规格 §五.4 硬性）
    # ----------------------------------------------------------------
    def log_message(self, fmt, *args):
        logger.info("[HTTP] " + fmt % args)

    # ----------------------------------------------------------------
    # 响应助手
    # ----------------------------------------------------------------
    def _json(self, code: int, obj: dict):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")     # 短连接: 避免 CLOSE_WAIT 残留
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()
        self.close_connection = True

    def _error(self, code: int, message: str, type_: str, code_: str):
        """标准错误体 {"error": {"message", "type", "code"}}（规格 §5.6）。"""
        self._json(code, {"error": {
            "message": message, "type": type_, "code": code_}})

    def _sse(self) -> SSEEventWriter:
        """开启 SSE 长连接并返回写入器（每个事件内部自行 flush）。"""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        return SSEEventWriter(self.wfile)

    # ----------------------------------------------------------------
    # 鉴权（规格 §5.1）：config.api_key 非 None 时要求 Bearer
    # ----------------------------------------------------------------
    def _check_auth(self) -> bool:
        cfg = config.get_config()
        api_key = cfg.api_key
        if api_key is None:
            return True
        header = self.headers.get("Authorization", "")
        token = header[len("Bearer "):].strip() \
            if header.startswith("Bearer ") else ""
        if token and token == str(api_key):
            return True
        self._error(401, "Invalid API key",
                    "authentication_error", "invalid_api_key")
        return False

    # ----------------------------------------------------------------
    # 读请求体
    # ----------------------------------------------------------------
    def _read_json(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                return {}
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return {}

    # ----------------------------------------------------------------
    # 路由：GET（/v1/models / /health / chat 端点 405 / 其余 404）
    # ----------------------------------------------------------------
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/v1/models":
            if not self._check_auth():
                return
            self._json(200, self._models_payload())
            return
        if path == "/health":
            self._json(200, self._health_payload())
            return
        if path == "/v1/chat/completions":
            self._method_not_allowed()
            return
        self._error(404, "Not Found", "invalid_request_error", "not_found")

    def do_POST(self):
        try:
            path = self.path.split("?")[0]
            # 仅 /v1/chat/completions 接受 POST，其它路径 → 404（规格 §5.1）
            if path != "/v1/chat/completions":
                self._error(404, "Not Found",
                            "invalid_request_error", "not_found")
                return
            if not self._check_auth():
                return
            # 忙检测: 本地推理（单卡 2B 生成）串行化, 忙碌时立即 429 让客户端重试
            core = Handler.core
            if core is not None and core._ongoing.locked():
                self._error(429, "Server busy (generating previous request)",
                            "rate_limit_error", "server_busy")
                return
            body = self._read_json()
            req = parse_openai_request(body)
            if req["error"] is not None:
                self._error(400, req["error"]["message"],
                            req["error"]["type"], req["error"]["code"])
                return
            if req["reasoning_warn"]:
                logger.warning("[ADAPTER] %s", req["reasoning_warn"])
            if req["stream"]:
                self._handle_stream(req)
            else:
                self._handle_non_stream(req)
        finally:
            # SSE 途中客户端断开/正常收尾: 一律显式关闭连接（HTTP/1.1 短连接策略）
            self.close_connection = True

    # 其它方法一律 405（规格 §5.1：方法不对 → 405）
    def do_PUT(self):
        self._method_not_allowed()

    def do_DELETE(self):
        self._method_not_allowed()

    def do_PATCH(self):
        self._method_not_allowed()

    def do_OPTIONS(self):
        self._method_not_allowed()

    def _method_not_allowed(self):
        self._error(405, "Method Not Allowed",
                    "invalid_request_error", "method_not_allowed")

    # ----------------------------------------------------------------
    # 模型 / 健康
    # ----------------------------------------------------------------
    def _models_payload(self) -> dict:
        return {
            "object": "list",
            "data": [{
                "id": config.get_config().model_name,
                "object": "model",
                "owned_by": "philia093",
            }],
        }

    def _health_payload(self) -> dict:
        try:
            status = Handler.core.get_status() if Handler.core else {}
            heartbeat = status.get("heartbeat", 0)
        except Exception:
            heartbeat = 0
        return {"status": "ok", "heartbeat": heartbeat}

    # ================================================================
    # 非流式
    # ================================================================
    def _handle_non_stream(self, req):
        core = Handler.core
        cfg = config.get_config()
        text = req["text"]
        reasoning_effort = req["reasoning_effort"]
        tools = req["tools"]

        result = core.respond_structured(text, reasoning_effort, tools)
        route = result.get("route")

        if route == "cloud":
            # 云端：适配层必须自己调 DeepSeek（规格 §5.3）
            try:
                ds = self._call_deepseek(cfg, req["messages"], tools)
            except urllib.error.HTTPError as e:
                self._respond_deepseek_http_error(e)
                return
            except (urllib.error.URLError, socket.timeout,
                    TimeoutError, OSError) as e:
                self._respond_deepseek_network_error(
                    core, cfg, text, reasoning_effort)
                return
            except ValueError as e:
                logger.error("[ADAPTER] DeepSeek 响应解析失败: %s", e)
                self._error(500, "Internal server error",
                            "server_error", "internal_error")
                return
            # 整包转发 DeepSeek 响应：保留 id/choices/usage，替换 model，
            # 顶层附加 route/provider 扩展字段（规格 §5.3）
            ds["model"] = cfg.model_name
            ds["route"] = "cloud"
            ds["provider"] = "deepseek"
            self._json(200, ds)
            return

        # 本地 / 本地深度思考：按 OpenAI 格式构造响应
        self._json(200, self._build_chat_response(result))

    def _call_deepseek(self, cfg, messages, tools):
        """非流式云端调用：标准库 urllib.request，超时 api_request_timeout。"""
        req = self._build_deepseek_request(cfg, messages, tools, stream=False)
        resp = urllib.request.urlopen(req, timeout=cfg.api_request_timeout)
        try:
            data = resp.read().decode("utf-8")
            return json.loads(data)
        finally:
            resp.close()

    def _respond_deepseek_http_error(self, e):
        """DeepSeek HTTP 错误映射（规格：429→429；401/403→401；其它→500）。"""
        code = getattr(e, "code", None)
        if code == 429:
            self._error(429, "Rate limit exceeded for cloud model.",
                        "rate_limit_error", "rate_limit_exceeded")
        elif code in (401, 403):
            self._error(401, "Invalid API key",
                        "authentication_error", "invalid_api_key")
        else:
            self._error(500, "Internal server error",
                        "server_error", "internal_error")

    def _respond_deepseek_network_error(self, core, cfg, text, reasoning_effort):
        """网络异常：cloud_fallback_local=True → 本地兜底；否则 500。"""
        if cfg.cloud_fallback_local:
            logger.warning("[ADAPTER] DeepSeek 网络异常，回退本地兜底")
            result = self._fallback_local_result(core, text, reasoning_effort)
            self._json(200, self._build_chat_response(result))
        else:
            logger.error("[ADAPTER] DeepSeek 网络异常且无本地兜底")
            self._error(500, "Internal server error",
                        "server_error", "internal_error")

    def _fallback_local_result(self, core, text, reasoning_effort):
        """本地兜底结果：content=本地回复，reasoning=None（不暴露技术细节）。"""
        result = core.respond_structured(text, reasoning_effort)
        if result.get("route") == "cloud":
            # 避免二次进入云端，用 /local 前缀强制本地路由（核心会去掉该前缀）
            result = core.respond_structured("/local " + text, reasoning_effort)
        result["reasoning"] = None
        return result

    def _build_chat_response(self, result):
        """按规格 §5.3 构造非流式 chat.completion 响应（本地/本地深度/兜底）。"""
        cfg = config.get_config()
        now = int(time.time())
        message = {"role": "assistant", "content": result.get("content")}
        route = result.get("route")
        if route == "local_deep" and result.get("reasoning"):
            message["reasoning_content"] = result["reasoning"]
        tool_calls = result.get("tool_calls")
        finish = result.get("finish_reason") or "stop"
        if tool_calls:
            message["tool_calls"] = tool_calls
            finish = "tool_calls"
        return {
            "id": result.get("id") or ("chatcmpl-%x" % now),
            "object": "chat.completion",
            "created": now,
            "model": cfg.model_name,
            "choices": [{"index": 0, "message": message,
                         "finish_reason": finish}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                      "total_tokens": 0},
            "route": route,
            "provider": result.get("provider") or "cyrene-agi",
        }

    # ================================================================
    # 流式
    # ================================================================
    def _handle_stream(self, req):
        core = Handler.core
        cfg = config.get_config()
        text = req["text"]
        reasoning_effort = req["reasoning_effort"]
        req_id = "chatcmpl-%x" % int(time.time() * 1000)
        model = cfg.model_name

        gen = core.respond_structured_stream(text, reasoning_effort)
        first = next(gen, None)     # 先窥探首个事件以判断云端/本地
        if first is None:
            w = self._sse()
            self._emit_local_event(w, "done",
                                   {"finish_reason": "stop",
                                    "provider": "cyrene-agi"}, req_id, model)
            return

        event, data = first
        if event == "cloud_handle":
            # 交给适配层转发 DeepSeek（规格 §5.4）
            self._handle_stream_cloud(req, req_id, model)
            return

        # ---- 本地流式：reasoning_start → reasoning_delta* → reasoning_done
        #      → reply_delta* → done
        w = self._sse()
        try:
            self._emit_local_event(w, event, data, req_id, model)
            for event, data in gen:
                if event == "cloud_handle":
                    continue    # 本地流中不应出现，保险跳过
                self._emit_local_event(w, event, data, req_id, model)
        except (BrokenPipeError, ConnectionResetError) as e:
            logger.debug("[ADAPTER] 客户端断开连接: %s", e)

    def _emit_local_event(self, w, event, data, req_id, model):
        """本地流式事件：双格式兼容。

        规格 §5.4 自定义事件（event: xxx + provider 字段）保留;
        同时 data 内嵌标准 OpenAI chunk 结构（choices[0].delta / finish_reason），
        使标准 OpenAI SSE 客户端可直接解析（修复"模型返回格式异常"）。
        """
        now = int(time.time())
        delta_map = {
            "reasoning_start": {},
            "reasoning_delta": {"reasoning_content": data.get("text", "")},
            "reasoning_done": {},
            "reply_delta": {"content": data.get("text", "")},
            "done": {},
        }
        delta = dict(delta_map.get(event, {}))
        if "tool_calls" in data:                 # tool_calls 标准片段覆盖默认 delta
            delta = {"tool_calls": data["tool_calls"]}
        chunk = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": now,
            "model": model,
            "choices": [{"index": 0,
                         "delta": delta,
                         "finish_reason": ("stop" if event == "done" else None)}],
        }
        out = dict(data)              # 保留 text / provider / finish_reason 等自定义字段
        out.update(chunk)
        out["id"] = req_id
        out["model"] = model
        w.write(event, out)

    def _handle_stream_cloud(self, req, req_id, model):
        """流式云端转发：调 DeepSeek stream=True，解析 SSE 后逐事件转发。"""
        core = Handler.core
        cfg = config.get_config()
        text = req["text"]
        reasoning_effort = req["reasoning_effort"]
        try:
            resp = self._open_deepseek_stream(cfg, req["messages"], req["tools"])
        except urllib.error.HTTPError as e:
            if cfg.cloud_fallback_local:
                self._stream_fallback_local(
                    core, text, reasoning_effort, req_id, model)
            else:
                self._respond_deepseek_http_error(e)
            return
        except (urllib.error.URLError, socket.timeout,
                TimeoutError, OSError) as e:
            if cfg.cloud_fallback_local:
                logger.warning("[ADAPTER] DeepSeek 流式连接异常，回退本地兜底: %s", e)
                self._stream_fallback_local(
                    core, text, reasoning_effort, req_id, model)
            else:
                logger.error("[ADAPTER] DeepSeek 流式连接异常: %s", e)
                self._error(500, "Internal server error",
                            "server_error", "internal_error")
            return

        # 成功：开启 SSE，转发 DeepSeek 事件流
        w = self._sse()
        try:
            self._forward_deepseek_stream(resp, w, req_id, model)
        finally:
            resp.close()

    def _stream_fallback_local(self, core, text, reasoning_effort, req_id, model):
        """流式本地兜底：reasoning_start → reasoning_done → reply_delta* → done。

        全部经 _emit_local_event 统一包装（标准 chunk 字段 + 自定义字段兼容）。
        """
        result = self._fallback_local_result(core, text, reasoning_effort)
        content = result.get("content") or ""
        w = self._sse()
        w.write("reasoning_start", {"provider": "cyrene-agi",
                                    "id": req_id, "model": model})
        w.write("reasoning_done", {"provider": "cyrene-agi",
                                   "id": req_id, "model": model})
        step = max(1, len(content) // 12)
        for i in range(0, len(content), step):
            self._emit_local_event(w, "reply_delta",
                                   {"text": content[i:i + step],
                                    "provider": "cyrene-agi"},
                                   req_id, model)
        self._emit_local_event(w, "done",
                               {"finish_reason": "stop",
                                "provider": "cyrene-agi"}, req_id, model)

    def _forward_deepseek_stream(self, resp, w, req_id, model):
        """解析 DeepSeek SSE 并转发为适配层事件（provider="deepseek"）。"""
        tool_calls = {}
        finish = "stop"
        try:
            for payload in self._iter_sse_payloads(resp):
                if payload == "[DONE]":
                    continue
                try:
                    chunk = json.loads(payload)
                except ValueError:
                    continue
                if not isinstance(chunk, dict):
                    continue
                choices = chunk.get("choices") or []
                for choice in choices:
                    if not isinstance(choice, dict):
                        continue
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        self._emit_local_event(
                            w, "reply_delta",
                            {"text": delta["content"], "provider": "deepseek"},
                            req_id, model)
                    if delta.get("reasoning_content"):
                        self._emit_local_event(
                            w, "reasoning_delta",
                            {"text": delta["reasoning_content"],
                             "provider": "deepseek"}, req_id, model)
                    if delta.get("tool_calls"):
                        for tc in delta["tool_calls"]:
                            if not isinstance(tc, dict):
                                continue
                            idx = tc.get("index", 0)
                            agg = tool_calls.setdefault(
                                idx, {"index": idx, "id": None,
                                      "type": None, "function": {}})
                            if tc.get("id"):
                                agg["id"] = tc["id"]
                            if tc.get("type"):
                                agg["type"] = tc["type"]
                            fn = tc.get("function") or {}
                            if isinstance(fn, dict):
                                if fn.get("name"):
                                    agg["function"]["name"] = fn["name"]
                                agg["function"]["arguments"] = (
                                    agg["function"].get("arguments", "")
                                    + (fn.get("arguments") or ""))
        except Exception as e:
            # 流中途断开：仍补发 done，保证客户端正常收尾
            logger.warning("[ADAPTER] DeepSeek 流读取中断: %s", e)
        # 收尾: tool_calls 一次性以标准 delta 结构发出（双格式兼容）
        if tool_calls:
            finish = "tool_calls"    # 有 tool_calls → done(finish_reason="tool_calls")
            self._emit_local_event(
                w, "reply_delta",
                {"tool_calls": [tool_calls[i] for i in sorted(tool_calls)],
                 "provider": "deepseek"}, req_id, model)
        self._emit_local_event(w, "done",
                               {"finish_reason": finish,
                                "provider": "deepseek"}, req_id, model)

    def _iter_sse_payloads(self, resp):
        """逐行迭代 DeepSeek 的 SSE 响应，按空行分帧产出 data 载荷。"""
        data_lines = []
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
            if line == "":
                if data_lines:
                    yield "\n".join(data_lines).strip()
                    data_lines = []
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
        if data_lines:
            yield "\n".join(data_lines).strip()

    # ================================================================
    # DeepSeek 请求构造
    # ================================================================
    def _open_deepseek_stream(self, cfg, messages, tools):
        req = self._build_deepseek_request(cfg, messages, tools, stream=True)
        return urllib.request.urlopen(req, timeout=cfg.api_request_timeout)

    def _build_deepseek_request(self, cfg, messages, tools, stream):
        url = cfg.deepseek_base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": cfg.deepseek_model,
            "messages": self._prepare_messages(cfg, messages),
            "stream": bool(stream),
        }
        if tools:
            payload["tools"] = tools
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", "Bearer " + str(cfg.deepseek_api_key or ""))
        return req

    def _prepare_messages(self, cfg, messages):
        """原始 messages；若其中无 system 角色，则最前插入 persona 系统提示。"""
        msgs = [m for m in (messages if isinstance(messages, list) else [])
                if isinstance(m, dict)]
        if not any(m.get("role") == "system" for m in msgs):
            msgs = [{"role": "system", "content": cfg.persona}] + msgs
        return msgs


# ================================================================
# 服务器生命周期
# ================================================================
def create_server(core, host=None, port=None) -> ThreadingHTTPServer:
    """创建 ThreadingHTTPServer，并把 core 挂在 Handler 类属性上。

    host / port 缺省取 config.api_host / config.api_port。
    """
    cfg = config.get_config()
    if host is None:
        host = cfg.api_host
    if port is None:
        port = cfg.api_port
    Handler.core = core
    server = ThreadingHTTPServer((host, port), Handler)
    logger.info("[ADAPTER] Cyrene-Agent 兼容端点 http://%s:%s/v1/chat/completions",
                host, port)
    return server


def run_server(core, host=None, port=None):
    """阻塞运行 serve_forever，接受 KeyboardInterrupt 优雅关闭。"""
    server = create_server(core, host, port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("[ADAPTER] 收到中断，正在关闭…")
    finally:
        server.server_close()
        logger.info("[ADAPTER] HTTP 服务已关闭")
