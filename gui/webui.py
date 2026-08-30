# -*- coding: utf-8 -*-
"""
gui/webui.py — 昔涟AGI Web UI（零依赖: 仅 Python 标准库 http.server）
=====================================================================
内存预估: < 50MB（服务线程 + 页面模板常量），无显存占用。

功能（等价于 chat_window + debug_window 的 Web 版）:
  - 对话区: 用户/昔涟/思绪/系统 消息流
  - 监控面板: 心跳 / 显存(预算4.5GB) / 情感柱状图 / 记忆池大小 / Top5 激活 /
    主导情感回环 / 痛觉 / 格式化阶段 / 路由决策
  - 按钮: 手动休眠 / 手动唤醒 / 强制存档 / 退出并存档
  - 状态提示: "昔涟正在思考……"

路由（JSON API）:
  GET  /              → 单页应用（HTML/CSS/JS 全部内联）
  GET  /api/status    → get_status() 快照
  GET  /api/events    → pull_events()（客户端每秒轮询）
  POST /api/chat      → {"text": "..."}  用户输入
  POST /api/sleep|wake|save|exit          控制动作
"""
import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import config


_PAGE = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>昔涟AGI</title>
<style>
 body{margin:0;font-family:"Microsoft YaHei",system-ui,sans-serif;background:#f4f2ee;color:#333}
 header{background:#2c2a28;color:#f5ede2;padding:10px 16px;display:flex;align-items:center;gap:12px}
 header h1{font-size:18px;margin:0}
 header small{color:#bbb}
 #wrap{display:flex;height:calc(100vh - 54px)}
 #left{flex:2;display:flex;flex-direction:column;border-right:1px solid #ddd}
 #chat{flex:1;overflow-y:auto;padding:14px;background:#fff}
 .msg{margin:6px 0;max-width:80%;padding:8px 12px;border-radius:10px;line-height:1.6;white-space:pre-wrap}
 .user{background:#d8e6ff;margin-left:auto}
 .xilian{background:#fdf3e3}
 .thought{background:#f1f1f1;color:#666;font-style:italic}
 .sys{background:transparent;color:#777;text-align:center;font-size:12px;max-width:100%}
 #bar{display:flex;gap:8px;padding:10px;background:#eee}
 #inp{flex:1;padding:8px;border:1px solid #ccc;border-radius:6px}
 button{padding:8px 14px;border:0;border-radius:6px;background:#8a5a2b;color:#fff;cursor:pointer}
 button.ghost{background:#666}
 #status{padding:6px 14px;background:#fff3d6;font-size:13px}
 #right{width:380px;overflow-y:auto;background:#fafafa;padding:12px}
 h3{font-size:14px;margin:10px 0 6px;color:#555}
 .kv{font-size:13px;margin:2px 0}
 .kv b{color:#8a5a2b}
 .bars{padding:6px 0}
 .bar{display:flex;align-items:center;gap:6px;margin:3px 0}
 .bar span{width:52px;font-size:12px;color:#555}
 .track{flex:1;height:12px;background:#e4e4e4;border-radius:6px;overflow:hidden}
 .fill{height:100%;background:#5b9bd5;border-radius:6px}
 .bar i{width:34px;font-size:11px;color:#888;font-style:normal}
 .btnrow{display:flex;flex-wrap:wrap;gap:6px;margin:8px 0}
 #top5{font-size:12px;color:#444;background:#fff;border:1px solid #eee;padding:6px;border-radius:6px}
</style></head><body>
<header><h1>昔涟AGI <small>v5.1</small></h1>
 <small id="mode"></small><span style="flex:1"></span><small id="vram"></small></header>
<div id="wrap">
 <div id="left">
  <div id="chat"></div>
  <div id="status">就绪</div>
  <div id="bar">
   <input id="inp" placeholder="对昔涟说点什么……（Enter 发送）">
   <button onclick="send()">发送</button>
  </div>
  <div style="padding:0 12px 10px;font-size:12px;color:#888;display:flex;gap:8px;align-items:center">
   <label>📷 图片<input type="file" id="imgf" accept="image/*" style="display:none" onchange="pickMedia(event,'image')"></label>
   <button class="ghost" style="padding:3px 8px" onclick="document.getElementById('imgf').click()">发图片</button>
   <label>🎤 语音<input type="file" id="audf" accept="audio/*,.wav,.mp3,.m4a" style="display:none" onchange="pickMedia(event,'audio')"></label>
   <button class="ghost" style="padding:3px 8px" onclick="document.getElementById('audf').click()">发语音</button>
   <span id="mediaTip"></span>
  </div>
 </div>
 <div id="right">
  <div class="btnrow">
   <button class="ghost" onclick="ctl('sleep')">手动休眠</button>
   <button class="ghost" onclick="ctl('wake')">手动唤醒</button>
   <button class="ghost" onclick="ctl('save')">强制存档</button>
   <button onclick="ctl('exit')">退出并存档</button>
  </div>
  <div class="kv">心跳: <b id="hb">-</b> ・ 记忆池: <b id="mp">-</b> 条 ・ 状态: <b id="st">-</b></div>
  <div class="kv">主导情感: <b id="dom">-</b> ・ 痛觉: <b id="pain">-</b> ・ 格式化阶段: <b id="fp">-</b></div>
  <div class="kv">工作记忆: <b id="wm">-</b> ・ 锚点: <b id="anchor">-</b> ・ 预测MSE: <b id="mse">-</b> ・ 沉底: <b id="sunk">-</b></div>
  <div class="kv">路由决策: <b id="dec">-</b></div>
  <h3>情感向量</h3><div class="bars" id="emobars"></div>
  <h3>最近激活记忆 Top5</h3><div id="top5">（暂无）</div>
  <h3>日志提示</h3><div class="kv">聊天: POST /api/chat ・ 状态: /api/status ・ 事件: /api/events</div>
 </div>
</div>
<script>
const $=id=>document.getElementById(id);
const TAGS={user:['user','伙伴：'],xilian:['xilian','昔涟：'],thought:['thought','（思绪）'],sys:['sys','']};
function line(cls,text){const d=document.createElement('div');d.className='msg '+cls[0];
 if(cls[2]){const t=document.createElement('span');t.textContent=text;d.appendChild(t);}
 else{d.textContent=text;} $('chat').appendChild(d);$('chat').scrollTop=1e9;return d;}
let curBubble=null;                                      // 流式输出中的气泡
function streamLine(cls,text){                           // 向当前气泡追加片段
 if(!curBubble||curBubble.dataset.cls!==cls[0]){curBubble=line(cls,'');curBubble.dataset.cls=cls[0];}
 curBubble.textContent+=text;$('chat').scrollTop=1e9;}
function finishStream(cls,text){                         // 收尾: 替换为完整文本
 if(curBubble){curBubble.textContent=cls[2]+text;curBubble=null;}
 else{line(cls,cls[2]+text);} $('chat').scrollTop=1e9;}
function send(){const t=$('inp').value.trim();if(!t)return;$('inp').value='';
 line(TAGS.user,'伙伴：'+t);$('status').textContent='昔涟正在思考……';
 fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({text:t})}).then(r=>{if(!r.ok)throw new Error('HTTP '+r.status)})
 .catch(e=>{$('status').textContent='⚠ 发送失败: '+e+' （点击状态栏重试）';
  $('status').style.cursor='pointer';$('status').onclick=()=>send();});}
async function ctl(a){const r=await fetch('/api/'+a,{method:'POST'});if(!r.ok)$('status').textContent=a+' 失败';}
function pickMedia(ev,kind){
 const f=ev.target.files[0];if(!f)return;const fr=new FileReader();
 fr.onload=()=>{const t=$('inp').value.trim();
  $('mediaTip').textContent='（正在发送'+kind+'……）';
  fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({text:t,kind:kind,media_base64:fr.result})})
  .then(r=>{if(!r.ok)throw new Error('HTTP '+r.status);$('mediaTip').textContent='已发送;等待昔涟回应…';
   $('inp').value='';$('status').textContent='昔涟正在思考……';})
  .catch(e=>{$('mediaTip').textContent='⚠ 发送失败:'+e;});};
 fr.readAsDataURL(f);ev.target.value='';}
function fillbar(name,v){const d=document.createElement('div');d.className='bar';
 d.innerHTML='<span>'+name+'</span><div class="track"><div class="fill" style="width:'+
  Math.round(v*100)+'%"></div></div><i>'+v.toFixed(2)+'</i>';return d;}
async function poll(){try{const evs=await(await fetch('/api/events')).json();
 if($('status').textContent.indexOf('连接')===0)$('status').textContent='就绪';
 for(const ev of evs){
  if(ev.type==='stream_chunk'){streamLine(TAGS.xilian,ev.text);}
  else if(ev.type==='stream_start'){curBubble=null;}
  else if(ev.type==='reply'){finishStream(TAGS.xilian,ev.text);$('status').textContent='就绪';}
  else{const c=TAGS[ev.type]||TAGS.sys;line(c,ev.text);}}
 const s=await(await fetch('/api/status')).json();
 $('hb').textContent=s.heartbeat;$('mp').textContent=s.memory.size;
 $('st').textContent=s.sleeping?'休眠中':'清醒';$('mode').textContent='模式:'+s.mode;
 $('vram').textContent='显存 '+s.vram_mb.toFixed(0)+'/'+s.vram_budget.toFixed(0)+'MB';
 $('dom').textContent=(s.emotion_dominant.category||'-')+' (回环#'+s.emotion_dominant.loop+')';
 $('pain').textContent=s.pain.toFixed(2);$('fp').textContent=s.format_phase+'/3';
 $('wm').textContent=s.working_memory.turns+'轮/'+s.working_memory.tokens_est+'tok';
  $('anchor').textContent=(s.anchor_signal===undefined?0:s.anchor_signal).toFixed(3);
 $('mse').textContent=s.predictor.mse.toFixed(4);
 $('sunk').textContent=s.memory.sunk;
 $('dec').textContent=s.last_decision.name+' '+JSON.stringify(s.last_decision.probs||[]);
 const eb=$('emobars');eb.innerHTML='';
 for(const n of Object.keys(s.emotion))eb.appendChild(fillbar(n,s.emotion[n]));
 const t5=$('top5');t5.innerHTML=(s.top_recent.length?s.top_recent.map(m=>
  '['+m.id+'] '+m.content+' <i style="color:#a66">×'+m.act.toFixed(1)+'</i>').join('<br>'):'（暂无）');
}catch(e){ if($('status').textContent.indexOf('连接')!==0)
  $('status').textContent='⚠ 连接断开，自动重连中…';} setTimeout(poll,1000);}
$('inp').addEventListener('keydown',e=>{if(e.key==='Enter')send();});
setInterval(poll,1000);
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    """HTTP 处理: 静态页面 + JSON API（全部转发给 AGICore，线程安全）。"""

    core = None            # 类属性注入（弱耦合，start.py 创建时赋值）

    # ---- 基础 ------------
    def log_message(self, fmt, *args):        # 走主日志，但不刷屏（DEBUG 级）
        if config.get_config().log_level == "DEBUG":
            super().log_message(fmt, *args)

    def _write(self, raw: bytes, code: int = 200, ctype: str = "application/json; charset=utf-8"):
        """带断连保护的响应写出（浏览器刷新/断开 → WinError 10053/10054/EPIPE 静默）。"""
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError, OSError):
            pass                                # 客户端断开: 正常现象, 不刷 traceback

    def _json(self, obj, code=200):
        self._write(json.dumps(obj, ensure_ascii=False).encode("utf-8"), code)

    def _save_media(self, media_b64: str, kind: str) -> str:
        """base64 dataURL → knowledge/uploads/ 落盘, 返回路径。"""
        import base64
        import time as _t
        data = media_b64.split(",", 1)[-1]
        raw = base64.b64decode(data)
        cfg = config.get_config()
        os.makedirs(cfg.uploads_dir, exist_ok=True)
        ext = ".wav" if kind == "audio" else ".jpg"
        path = os.path.join(cfg.uploads_dir, f"{kind}_{int(_t.time()*1000)}{ext}")
        with open(path, "wb") as f:
            f.write(raw)
        return path

    def _empty(self, code=200):
        self._write(b"", code)

    def _read_body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0) or 0)
        if n <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8", errors="replace") or "{}")
        except Exception:
            return {}

    # ---- 路由 ------------
    def do_GET(self):
        core = _Handler.core
        path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            self._write(_PAGE.encode("utf-8"), 200, "text/html; charset=utf-8")
        elif path == "/api/status" and core is not None:
            self._json(core.get_status())
        elif path == "/api/events" and core is not None:
            self._json(core.pull_events())
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        core = _Handler.core
        if core is None:
            self._json({"error": "core not ready"}, 503)
            return
        path = self.path.split("?")[0]
        if path == "/api/chat":
            body = self._read_body()
            kind = str(body.get("kind", "") or "")
            media_b64 = str(body.get("media_base64", "") or "")
            text = str(body.get("text", "")).strip()
            if not text and not media_b64:
                self._json({"error": "empty"}, 400)
                return
            if media_b64:
                # 媒体输入: 解码 base64 → uploads 目录 → 多模态认知链路
                try:
                    import base64 as _b64
                    media_path = self._save_media(media_b64, kind)
                except Exception as e:
                    self._json({"error": f"media decode failed: {e}"}, 400)
                    return
                if not self.core.respond_media(kind or "image", media_path, text):
                    self._json({"error": "media not accepted"}, 500)
                    return
                self._json({"ok": True, "media": kind, "path": media_path})
            else:
                core.respond(text)
                self._json({"ok": True})
        elif path == "/api/sleep":
            core.sleep_now()
            self._json({"ok": True})
        elif path == "/api/wake":
            core.wake_now()
            self._json({"ok": True})
        elif path == "/api/save":
            core.force_save()
            self._json({"ok": True})
        elif path == "/api/exit":
            # 退出并存档: 应答后延迟关闭（给浏览器一个响应时间）
            self._json({"ok": True})
            threading.Thread(target=self._exit_worker, name="webui-exit",
                             daemon=True).start()
        else:
            self._json({"error": "not found"}, 404)

    def _exit_worker(self):
        """延迟 0.3s 关闭: core.shutdown() + 停止 HTTP 服务 + 退出进程。"""
        import time
        time.sleep(0.3)
        try:
            _Handler.core.shutdown()
        finally:
            try:
                _Handler.server.shutdown()
            finally:
                import os
                os._exit(0)


class WebUIServer:
    """昔涟AGI Web UI 服务器（后台线程 serve_forever 由 run() 持有并阻塞）。"""

    def __init__(self, core, cfg=None):
        self.cfg = cfg if cfg is not None else config.get_config()
        self.core = core
        _Handler.core = core
        self.httpd = self._make_server()
        _Handler.server = self.httpd

    def _make_server(self):
        """创建 HTTP 服务器: 优先 AF_INET6 双栈（localhost/IPv6 浏览器均可访问），
        失败回退 IPv4 127.0.0.1（Windows 通常直接双栈成功）。"""
        import socket
        host = self.cfg.webui_host
        if host == "0.0.0.0":
            host = ""                      # 全接口（局域网模式）
        try:
            srv_class = ThreadingHTTPServer
            addr_family = socket.AF_INET6
            srv = srv_class((host, self.cfg.webui_port), _Handler)
            # 双栈: 同时接受 IPv6 与 IPv4 连接（localhost → ::1 也能连）
            try:
                srv.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            except Exception:
                pass
            return srv
        except Exception:
            return ThreadingHTTPServer(("127.0.0.1", self.cfg.webui_port), _Handler)

    def run(self, open_browser: bool = True):
        """阻塞式运行（主线程）；Ctrl+C → 优雅关闭并存档。"""
        url = f"http://{self.cfg.webui_host}:{self.cfg.webui_port}"
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        import logging
        log = self.cfg.setup_logging().getChild("webui")
        log.info("Web UI 已启动: %s （Ctrl+C 退出并存档）", url)
        try:
            self.httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self):
        """停止 HTTP 服务 + 核心关闭存档。"""
        try:
            self.httpd.shutdown()
        except Exception:
            pass
        try:
            self.core.shutdown()
        except Exception:
            pass
