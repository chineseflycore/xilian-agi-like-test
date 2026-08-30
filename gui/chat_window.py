# -*- coding: utf-8 -*-
"""
gui/chat_window.py — 对话窗口（【硬性】 §十一）
================================================
内存预估: Tkinter 控件 < 20MB RAM，无显存占用。

功能: 历史消息区 / 输入框 / 发送按钮 / 状态提示（"昔涟正在思考..."）/ 退出并存档按钮。
事件驱动: 每 200ms 轮询 AGICore.pull_events()（thread-safe），
  reply → 对话气泡；thought → 内部思绪；system/format/sleep/wake → 状态行。
"""
import tkinter as tk
from tkinter import ttk, scrolledtext


class ChatWindow(tk.Tk):
    """主对话窗口（单实例，Tk root）。"""

    def __init__(self, core):
        super().__init__()
        self.core = core
        self.title("昔涟AGI · 对话")
        self.geometry("560x640")
        self.minsize(420, 480)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ---- 对话历史区（只读 Text + 滚动条） ----
        self.history = scrolledtext.ScrolledText(self, wrap="word",
                                                 state="disabled", font=("Microsoft YaHei", 10))
        self.history.pack(fill="both", expand=True, padx=8, pady=8)
        self._append_line("—— 昔涟AGI v5.1 ——", "system")

        # ---- 状态栏 + 输入区 ----
        self.status_var = tk.StringVar(value="就绪")
        self.status_label = tk.Label(self, textvariable=self.status_var, anchor="w",
                                     fg="#666666")
        self.status_label.pack(fill="x", padx=8)
        # ---- 底部系统状态条（心跳/模式/状态/显存/记忆池，经 get_status 刷新） ----
        self.sys_status_var = tk.StringVar(value="")
        self.sys_status_label = tk.Label(self, textvariable=self.sys_status_var, anchor="w",
                                         fg="#999999", font=("Microsoft YaHei", 8))
        self.sys_status_label.pack(fill="x", padx=8)
        bottom = ttk.Frame(self)
        bottom.pack(fill="x", padx=8, pady=(0, 8))
        self.entry = ttk.Entry(bottom, font=("Microsoft YaHei", 10))
        self.entry.pack(side="left", fill="x", expand=True)
        self.entry.bind("<Return>", lambda e: self._on_send())
        self.send_btn = ttk.Button(bottom, text="发送", command=self._on_send)
        self.send_btn.pack(side="left", padx=(6, 0))
        self.debug_btn = ttk.Button(bottom, text="调试窗口", command=self._open_debug)
        self.debug_btn.pack(side="left", padx=(6, 0))
        self.quit_btn = ttk.Button(bottom, text="退出并存档", command=self._on_close)
        self.quit_btn.pack(side="left", padx=(6, 0))

        self._debug_window = None
        self.after(200, self._poll_events)      # 事件轮询（不阻塞主循环）
        self.after(1000, self._refresh_status_bar)  # 系统状态条刷新

    # ------------------------------------------------------------------
    # 交互
    # ------------------------------------------------------------------
    def _on_send(self):
        """发送用户输入 → 交由核心异步处理 + 状态提示。"""
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, "end")
        self._append_line(f"伙伴：{text}", "user")
        self.status_var.set("昔涟正在思考……")
        self.core.respond(text)

    def _open_debug(self):
        """打开/激活调试窗口（懒加载）。"""
        if self._debug_window is None or not self._debug_window.winfo_exists():
            from gui.debug_window import DebugWindow
            self._debug_window = DebugWindow(self, self.core)
        else:
            self._debug_window.deiconify()
            self._debug_window.lift()

    # ------------------------------------------------------------------
    # 事件轮询
    # ------------------------------------------------------------------
    def _poll_events(self):
        """拉取核心事件并渲染到历史区（reply/thought/format/sleep/wake/system）。"""
        # 外部关闭信号（shutdown.py 写入 knowledge/shutdown.signal → 自动退出并存档）
        try:
            import os
            import config as _cfg
            sig = os.path.join(_cfg.get_config().knowledge_dir, "shutdown.signal")
            if os.path.isfile(sig):
                os.remove(sig)
                self._append_line("（收到关闭信号，正在存档并退出……）", "system")
                self._on_close()
                return
        except Exception:
            pass
        try:
            for ev in self.core.pull_events():
                kind = ev.get("type")
                text = ev.get("text", "")
                if kind == "stream_start":
                    self._begin_stream()
                elif kind == "stream_chunk":
                    self._stream_append(text)
                elif kind == "reply":
                    self._finish_stream(f"昔涟：{text}")
                    self.status_var.set("就绪")
                elif kind == "thought":
                    self._append_line(f"（思绪）{text}", "thought")
                elif kind == "input":
                    self._append_line(f"（输入）{text}", "thought")
                elif kind in ("format", "system"):
                    self._append_line(f"◆ {text} ◆", "system")
                elif kind == "sleep":
                    self._append_line(text, "system")
                    self.status_var.set("昔涟正在休息……")
                elif kind == "wake":
                    self._append_line(text, "system")
                    self.status_var.set("就绪")
        except Exception as e:
            self.status_var.set(f"事件轮询异常: {e}")
        self.after(200, self._poll_events)

    # ------------------------------------------------------------------
    # 底部系统状态条刷新（新 get_status 键）
    # ------------------------------------------------------------------
    def _refresh_status_bar(self):
        """底部状态条: 心跳/tick、模式、状态(清醒/降频/休眠)、显存、记忆池。"""
        try:
            st = self.core.get_status()
            if st.get("sleeping"):
                status = "休眠"
            elif st.get("state") == "idle":
                status = "降频"
            else:
                status = "清醒"
            vram = st.get("vram_mb", 0.0)
            budget = st.get("vram_budget", 4500.0)
            mem = st.get("memory", {})
            self.sys_status_var.set(
                f"心跳 {st.get('heartbeat', 0)} (tick {st.get('tick', 0)}) · "
                f"模式 {st.get('mode', '?')} · 状态 {status} · "
                f"显存 {vram:.0f}/{budget:.0f}MB · 记忆池 {mem.get('size', 0)} 条"
            )
        except Exception:
            pass
        self.after(1000, self._refresh_status_bar)

    # ---- 流式渲染支持（stream_chunk 追加 → reply 收尾替换为完整文本） ----
    def _begin_stream(self):
        """流式开始: 记录插入起点（供收尾替换）。"""
        self._stream_start = self.history.index("end-1c")

    def _stream_append(self, piece: str):
        """流式追加片段（不换行）。"""
        if piece:
            self.history.configure(state="normal")
            self.history.insert("end", piece)
            self.history.see("end")
            self.history.configure(state="disabled")

    def _finish_stream(self, full_text: str, who: str = "assistant"):
        """流式收尾: 用完整文本替换流式片段区（防重复显示）。"""
        start = getattr(self, "_stream_start", None)
        if start is not None:
            self.history.configure(state="normal")
            self.history.delete(start, "end-1c")
            self.history.insert("end", full_text + "\n", who)
            self.history.tag_configure(who, foreground="#8a5a00")
            self.history.see("end")
            self.history.configure(state="disabled")
            self._stream_start = None
        else:
            self._append_line(full_text, who)

    def _append_line(self, text: str, who: str):
        """历史区追加一行（配色: user/assistant/thought/system）。"""
        tags = {"user": ("#1a66ff", "bold"), "assistant": ("#8a5a00", ""),
                "thought": ("#777777", "italic"), "system": ("#555555", "")}.get(who)
        self.history.configure(state="normal")
        if tags:
            # 一次性配置标签（前景 + 字体），并对插入文本整体打标签
            self.history.tag_configure(who, foreground=tags[0],
                                       font=("Microsoft YaHei", 10, tags[1] or "normal"))
            self.history.insert("end", text + "\n", who)
        else:
            self.history.insert("end", text + "\n")
        self.history.see("end")
        self.history.configure(state="disabled")

    # ------------------------------------------------------------------
    def _on_close(self):
        """退出并存档（【硬性】 §十二）: 核心 shutdown → 释放 → 关闭窗口。"""
        try:
            self.core.shutdown()
        finally:
            self.destroy()
