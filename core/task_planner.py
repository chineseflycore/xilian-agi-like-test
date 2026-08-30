# -*- coding: utf-8 -*-
"""
core/task_planner.py — 昔涟AGI v7.3 任务规划层（仅 API 模式启用）
================================================================
显存/内存预估: < 10MB RAM（仅路由历史与少量输入缓存等常驻，无模型权重）。

职责:
  为 Cyrene-Agent 适配层做请求路由决策 —— 判定该次请求应当
  走 本地 / 本地深度思考 / 云端 DeepSeek 之一，并附上强制本地与
  走云端比例告警等"防逃逸"保护。

  仅当 AGICore(api_mode=True) 时才使用本层；本地 GUI 会话固定走
  完整认知架构，不经过本层（见 agi_core.respond）。

依赖（均已在别处实现, 本层不改动）:
  config.get_config() 单例属性:
    complex_keywords / local_force_keywords / plan_history_len /
    cloud_ratio_warn / repeat_force_local
  core.matcher_cluster.MatcherCluster:
    adjust_activation_scale(factor) -> float（<1 抑制激活, 内部钳制 0.2~3.0）
"""
import collections
import logging

# 命名空间: 保持与其它 core 模块一致的"xilian"日志树
_LOG = logging.getLogger("xilian").getChild("planner")


class TaskPlanner:
    """四、任务规划层 —— 仅 API 模式启用。

    根据输入文本 / reasoning_effort / tools 给出路由决策，并执行
    防逃逸保护（连续相同输入强制本地 + 走云端比例告警抑制激活）。
    """

    def __init__(self, cfg, cluster):
        # cfg = config 单例; cluster = core.matcher_cluster.MatcherCluster 实例
        self.cfg = cfg
        self.cluster = cluster
        self.log = _LOG

        # 最近 plan_history_len 次路由决策（"local"/"local_deep"/"cloud"）
        self._history = collections.deque(maxlen=cfg.plan_history_len)
        # 最近 repeat_force_local 条"去空格后"的输入文本（连续相同判定）
        self._last_inputs = collections.deque(maxlen=cfg.repeat_force_local)
        # 输入 → 连续重复次数计数（用于连续判定; 一段新输入时重置）
        self._forced_map = {}
        # 最近一次计算得到的 cloud 比例缓存
        self._ratio = 0.0

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    @staticmethod
    def _compact(text: str) -> str:
        """去空格（含全角/制表等所有空白）后的输入文本。"""
        return "".join(text.split())

    def _compute_ratio(self) -> float:
        """最近 min(len, plan_history_len) 条中 "cloud" 占比; 无历史返回 0.0。"""
        n = len(self._history)
        if n == 0:
            return 0.0
        return self._history.count("cloud") / float(n)

    # ------------------------------------------------------------------
    # 路由主入口
    # ------------------------------------------------------------------
    def plan(self, text: str, reasoning_effort: str = None,
             tools: list = None) -> dict:
        """路由决策（顺序执行的硬性规则）→ 返回决策 dict。

        返回字段:
          route        : "local" | "local_deep" | "cloud"
          reason       : 命中规则的中文说明
          warn         : str|None（走云端比例过高时的中文告警）
          forced       : bool（连续相同输入 → 第 4 次强制本地）
          stripped     : 实际参与路由的清理后文本（/local 前缀被移除）
          cloud_ratio  : 本次决策后最近 cloud 占比（float）
        """
        text = text or ""
        # 步骤 1: 去掉首尾空白；作为后续 /local 前缀与清理文本的基础
        stripped = text.strip()
        route = None
        reason = ""

        # ---- 规则1: /local 前缀 → 本地（命中后直接返回, 不再判定） ----
        for kw in self.cfg.local_force_keywords:
            if stripped.startswith(kw):
                route = "local"
                reason = "命中规则1: 本地指令（%s）" % kw
                stripped = stripped[len(kw):].strip()
                break

        if route is None:
            # ---- 规则2: 深度思考 ----
            if reasoning_effort in ("high", "xhigh"):
                route = "local_deep"
                reason = "命中规则2: 深度思考（reasoning_effort=%s）" % reasoning_effort
            # ---- 规则3: 工具调用 或 复杂关键词 → 云端 ----
            elif tools or any(kw in text for kw in self.cfg.complex_keywords):
                route = "cloud"
                reason = "命中规则3: 复杂任务（工具/复杂关键词）"
            # ---- 规则4: 默认本地 ----
            else:
                route = "local"
                reason = "命中规则4: 默认本地"

        # ---- 防逃逸: 连续相同输入 → 强制本地 ----
        # 规则: 若 text（去空格比较）== 上一条 → 计数+1;
        #        计数 >= repeat_force_local 时 → 第 4 次强制本地。
        # 计数定义: 该输入在一段连续重复中"重复的次数"（首次出现计 0）。
        forced = False
        compact = self._compact(text)
        prev = self._last_inputs[-1] if self._last_inputs else None
        if prev is not None and compact == prev:
            n = self._forced_map.get(compact, 0) + 1
            self._forced_map[compact] = n
        else:
            # 不同输入 → 重置计数（重新起一段）
            n = 0
            self._forced_map = {compact: n}
        self._last_inputs.append(compact)

        if n >= self.cfg.repeat_force_local:
            forced = True
            route = "local"
            reason += "；连续相同输入 → 强制本地"
            self.log.info("[P] 连续相同输入 %d 次 → 强制本地（原文: %s）",
                          n, (text or "")[:24])

        # ---- 记录历史并计算 cloud 占比 ----
        self._history.append(route)
        ratio = self._compute_ratio()
        self._ratio = ratio

        # ---- 走云端比例过高 → 告警 + 抑制激活 ----
        warn = None
        warn_th = float(self.cfg.cloud_ratio_warn)
        if ratio > warn_th:
            pct = int(round(ratio * 100))
            th_pct = int(round(warn_th * 100))
            warn = ("（最近 %d 次决策里走云端比例已到 %d%%，超过 %d%% 的阈值了；"
                    "昔涟会把思绪收回来，多一些本地回应。）"
                    % (len(self._history), pct, th_pct))
            try:
                scale = self.cluster.adjust_activation_scale(0.9)
                self.log.warning(
                    "[P] 走云端比例过高 (%d%%>%d%%) → 抑制激活: %.3f",
                    pct, th_pct, scale)
            except Exception as e:
                self.log.warning("[P] adjust_activation_scale 调用失败: %s", e)

        return {
            "route": route,
            "reason": reason,
            "warn": warn,
            "forced": bool(forced),
            "stripped": stripped,
            "cloud_ratio": ratio,
        }

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def get_ratio(self) -> float:
        """最近 cloud 占比（无历史返回 0.0）。"""
        return self._ratio if self._history else 0.0

    def stats(self) -> dict:
        """统计快照。"""
        return {"history": len(self._history), "cloud_ratio": self._compute_ratio()}
