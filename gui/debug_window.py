# -*- coding: utf-8 -*-
"""
gui/debug_window.py — 调试监控窗口（【硬性】 §十一）
====================================================
内存预估: Tkinter 控件 < 30MB RAM，无显存占用。

监控内容: 显存占用（预算 4.5GB）、心跳、情感向量（柱状图）、记忆池大小、
  最近激活记忆 Top5、休眠状态、主导情感回环、痛觉/格式化阶段、路由决策。
手动控制: 休眠 / 唤醒 / 强制存档 按钮。
刷新: 每 1000ms 轮询 AGICore.get_status()（线程安全快照）。
"""
import tkinter as tk
from tkinter import ttk, scrolledtext


class DebugWindow(tk.Toplevel):
    """调试监控窗口（父窗口 = ChatWindow）。"""

    def __init__(self, master, core):
        super().__init__(master)
        self.core = core
        self.title("昔涟AGI · 调试监控")
        self.geometry("620x560")
        self.attributes("-topmost", False)

        # ---- 顶部控制按钮 ----
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=8, pady=6)
        ttk.Button(bar, text="手动休眠", command=self._sleep).pack(side="left", padx=2)
        ttk.Button(bar, text="手动唤醒", command=self._wake).pack(side="left", padx=2)
        ttk.Button(bar, text="强制存档", command=self._save).pack(side="left", padx=2)
        ttk.Button(bar, text="关闭窗口", command=self.withdraw).pack(side="right", padx=2)

        # ---- 左: 状态文本（心跳/显存/记忆/休眠/主导/痛觉/决策） ----
        left = ttk.Frame(self)
        left.pack(side="left", fill="both", expand=True, padx=8, pady=4)
        self.status_text = scrolledtext.ScrolledText(left, width=34, height=20,
                                                     state="disabled", font=("Consolas", 9))
        self.status_text.pack(fill="both", expand=True)

        # ---- 右: 情感柱状图 + Top5 激活 ----
        right = ttk.Frame(self)
        right.pack(side="left", fill="both", expand=True, padx=8, pady=4)
        ttk.Label(right, text="情感向量").pack(anchor="w")
        self.canvas = tk.Canvas(right, width=250, height=140, bg="white",
                                highlightthickness=1, highlightbackground="#cccccc")
        self.canvas.pack(fill="x", pady=(2, 6))
        ttk.Label(right, text="最近激活记忆 Top5").pack(anchor="w")
        self.mem_text = scrolledtext.ScrolledText(right, width=34, height=10,
                                                  state="disabled", font=("Microsoft YaHei", 9))
        self.mem_text.pack(fill="both", expand=True)

        self.after(1000, self._refresh)

    # ------------------------------------------------------------------
    # 手动控制
    # ------------------------------------------------------------------
    def _sleep(self):
        self.core.sleep_now()
        self._refresh()

    def _wake(self):
        self.core.wake_now()
        self._refresh()

    def _save(self):
        self.core.force_save()
        self._refresh()

    # ------------------------------------------------------------------
    # 刷新（1s 轮询）
    # ------------------------------------------------------------------
    def _refresh(self):
        """拉取状态快照并刷新全部控件。"""
        try:
            st = self.core.get_status()
            self._refresh_status(st)
            self._refresh_emotion_bars(st)
            self._refresh_memories(st)
        except Exception:
            pass
        self.after(1000, self._refresh)

    def _refresh_status(self, st: dict):
        """状态文本区刷新（新 get_status 键: 心跳/显存/记忆池/工作记忆/匹配器/预测器/情感/决策/编码器/扩散器）。"""
        vram = st.get("vram_mb", 0.0)
        budget = st.get("vram_budget", 4500.0)
        dom = st.get("emotion_dominant", {})
        mem = st.get("memory", {})
        dec = st.get("last_decision", {})
        wm = st.get("working_memory", {})
        pred = st.get("predictor", {})
        ms = st.get("matcher_stats", {})
        diff = st.get("diffuser", {})

        if st.get("sleeping"):
            status = "休眠中"
        elif st.get("state") == "idle":
            status = "降频"
        else:
            status = "清醒"

        # 显存行: 超 4400 加 ⚠；vram_warn 加 "⚠超限!"
        vram_line = f"显存: {vram:.0f} / {budget:.0f} MB"
        if vram > 4400 or st.get("vram_warn"):
            vram_line += " ⚠"
            if st.get("vram_warn"):
                vram_line += "超限!"

        # 匹配器: 全域激活 / 合并状态
        burst_txt = (f"全域激活中({ms.get('full_burst_beats', 0)}心跳)"
                     if ms.get("full_burst") else "-")
        comb_txt = "待合并" if ms.get("combined_dirty") else "已同步"
        enc_txt = "协议头已训练" if st.get("encoder_source") == "qwen" else "规则编码(原型)"
        # 决策概率: 前 5 个两位小数
        probs_txt = ", ".join(f"{p:.2f}" for p in dec.get("probs", [])[:5])

        lines = [
            f"心跳: {st.get('heartbeat', 0)} (tick {st.get('tick', 0)})",
            f"模式: {st.get('mode', '?')}  状态: {status}",
            vram_line,
            f"记忆池: {mem.get('size', 0)} 条 / 沉底 {mem.get('sunk', 0)} 条",
            f"工作记忆: {wm.get('turns', 0)} 轮 / ~{wm.get('tokens_est', 0)} tok / 背景 {wm.get('background', 0)}",
            f"匹配器: 热{ms.get('hot', 0)} 暖{ms.get('warm', 0)} 冷{ms.get('cold', 0)}  {burst_txt}",
            f"  合并: {comb_txt}  更新 {ms.get('updates', 0)} 缩放 {ms.get('activation_scale', 0.0):.2f} 显存 {ms.get('vram_mb', 0.0):.0f}MB",
            f"预测器 MSE: {pred.get('mse', 0.0):.4f} (训练 {pred.get('train_count', 0)} 步)",
            f"主导情感: {dom.get('category', '-')} (回环#{dom.get('loop', '-')})",
            f"痛觉: {st.get('pain', 0.0):.2f}  格式化阶段: {st.get('format_phase', 0)}/3",
            f"决策: {dec.get('name', '-')} (id={dec.get('decision', '-')})",
            f"  编码器: {enc_txt}  扩散器: {diff.get('mode', '?')} (L2 {'已训练' if diff.get('l2_trained') else '未训练'}, L3 {diff.get('l3_edges', 0)} 边)",
            f"  probs: {probs_txt}",
        ]
        self._set_text(self.status_text, "\n".join(lines))

    @staticmethod
    def _set_text(widget, text: str):
        """只读 Text 区整体替换。"""
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _refresh_emotion_bars(self, st: dict):
        """情感向量柱状图（Canvas 矩形，8 维）。"""
        c = self.canvas
        c.delete("all")
        emo = st.get("emotion", {})
        names = list(emo.keys())
        bar_w = 22
        gap = 6
        x0 = 6
        max_h = 110
        for i, name in enumerate(names):
            x = x0 + i * (bar_w + gap)
            v = float(emo.get(name, 0.0))
            h = max(2, int(max_h * v))
            y = 130 - h
            c.create_rectangle(x, y, x + bar_w, 130, fill="#e6e6e6", outline="")
            c.create_rectangle(x, y, x + bar_w, 130, fill="#5b9bd5", outline="")
            c.create_text(x + bar_w / 2, 142, text=name[:5], font=("Microsoft YaHei", 7))
            c.create_text(x + bar_w / 2, 156, text=f"{v:.2f}", font=("Consolas", 7))

    def _refresh_memories(self, st: dict):
        """最近激活记忆 Top5 文本区。"""
        top = st.get("top_recent", [])
        lines = [f"[#{m.get('id')}] {m.get('content', '')} "
                 f"(激活 {m.get('act', 0):.2f})" for m in top] or ["（暂无）"]
        self._set_text(self.mem_text, "\n".join(lines))
