# -*- coding: utf-8 -*-
"""
server.py — HTTP API 服务（规范 §七，含 processing_lock / 503）
================================================================
内存/显存预估（规范 §九.1）:
  - 引擎装配: 路由层 Qwen3.5-0.8B（≈3.2GB 显存，REAL）/ 伪嵌入（DEMO）
  - 扩散器: L1 numpy（<1MB）/ L2 torch（≈4MB，GPU 主模型+CPU 副本）/ L3（~400MB CPU @1M）
  - 记忆系统: FAISS 10000×512（≈20MB CPU）
  - 服务本身: stdlib http.server 线程池，< 10MB

端点（规范 §7.1）:
  POST /chat    发送消息，返回文本 + 可选语音
  GET  /status  系统状态
  POST /switch  切换管线（L1/L2/L3）
  POST /reset   重置对话
  POST /format  手动格式化

规范 §八.23: 含 processing_lock —— 加载验证层时 /chat 返回 503。

/chat 响应结构（规范 §7.2）:
  {"response_text", "response_audio", "direct_mode", "line_used", "emotion",
   "pain_value", "alignment_score", "sentiment_override", "response_time_ms"}
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
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

import config
from common_utils import get_logger, Timer, vram_summary

logger = get_logger("server")

# 对话历史（进程内；POST /reset 清空）
HISTORY = []           # [{"role": "user"/"assistant", "content": str}]


class Engine:
    """人格引擎装配：感知 → 融合 → 路由 → 扩散 → 感性 → 输出 → 记忆 → 痛觉。

    所有组件可选；组件缺失/异常时逐级降级（规范 §九.2），保证 DEMO 可运行。
    """

    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else config.get_config()
        self.cfg.ensure_dirs()
        self.log = get_logger("engine")
        self.log.info("[ENGINE] 引擎装配开始（mode=%s）",
                      "DEMO(纯CPU流程验证)" if self.cfg.demo else "REAL")

        # ---- 感知层（CPU）----
        from perception_audio import AudioPerception
        from perception_fusion import FusionLayer
        self.audio = AudioPerception(self.cfg)
        self.fusion = FusionLayer(self.cfg)

        # ---- 语义翻译器（路由层，Qwen3.5-0.8B 真实模型 / DEMO 伪嵌入）----
        from translator import Translator
        self.translator = Translator(self.cfg)
        self.log.info("[ENGINE] 语义翻译器 mode=%s", "DEMO" if self.translator.is_demo else "REAL")

        # ---- 路由 / 情感 / 感性 ----
        from router import Router
        from emotion_model import EmotionModel
        from sentiment_module import SentimentModule
        self.router = Router(self.cfg, translator=self.translator)
        self.emotion = EmotionModel(self.cfg)
        self.sentiment = SentimentModule(self.cfg, emotion_model=self.emotion,
                                         translator=self.translator)

        # ---- 三线扩散器（L2 需 torch，缺省时内部降级）----
        self.l1 = self.l2 = self.l3 = None
        try:
            from diffuser_l1 import L1Diffuser
            self.l1 = L1Diffuser(self.cfg)
        except Exception as e:
            self.log.warning("[ENGINE] L1 不可用: %s", e)
        try:
            from diffuser_l2 import L2Diffuser
            self.l2 = L2Diffuser(self.cfg)
        except Exception as e:
            self.log.warning("[ENGINE] L2 不可用: %s", e)
        try:
            from diffuser_l3 import L3Diffuser
            self.l3 = L3Diffuser(self.cfg)
        except Exception as e:
            self.log.warning("[ENGINE] L3 不可用: %s", e)

        # ---- 加载训练好的扩散器权重（models/ 下，规范 §4.3 长期记忆恢复）----
        if self.l2 is not None:
            p2 = os.path.join(self.cfg.model_cache_dir, "l2_weights.pt")
            if os.path.isfile(p2):
                self.l2.load_weights(p2)
        if self.l3 is not None:
            p3 = os.path.join(self.cfg.model_cache_dir, "l3_weights.npz")
            if os.path.isfile(p3):
                try:
                    self.l3.load_state(p3)
                except Exception as e:
                    self.log.warning("[ENGINE] L3 权重加载失败（用初始权重）: %s", e)

        # ---- 控制器 / 记忆 / 痛觉 / 格式化 / 肢体 ----
        from line_controller import LineController
        from memory import MemorySystem
        from pain_system import PainSystem
        from formatting import Formatting
        from limb_controller import LimbController
        self.line_ctrl = LineController(self.cfg, self.l1, self.l2, self.l3, memory=None)
        self.memory = MemorySystem(self.cfg, translator=self.translator, l3=self.l3)
        self.line_ctrl.memory = self.memory
        self.pain = PainSystem(self.cfg, l3=self.l3)
        self.formatting = Formatting(self.cfg, l3=self.l3, memory=self.memory,
                                     translator=self.translator)
        self.limbs = LimbController(self.cfg)

        # ---- 好感度（借鉴 xilian-agent affection，伙伴关系成长）----
        from affection import AffectionSystem
        self.affection = AffectionSystem(self.cfg)

        # ---- 验证/批判层（默认关闭，按需加载，规范 §十.10）----
        self.validator = None
        self.critic = None
        self._validator_busy = False          # processing_lock（规范 §八.23）
        self._load_validator_critic()

        # ---- 异步训练（CPU 后台，双缓冲）----
        from trainer import AsyncTrainer
        self.trainer = AsyncTrainer(self.cfg, l2=self.l2, l3=self.l3)
        self.trainer.start()

        # ---- 静默期自发激活（随机召回，冷却 60s，规范 §4.6/§十.19/26）----
        self._last_spontaneous = 0.0
        self._last_recall_ts = 0.0
        self._silence_thread = threading.Thread(
            target=self._silence_loop, name="philia-silence", daemon=True)
        self._silence_thread.start()

        self.log.info("[ENGINE] 引擎装配完成 %s", vram_summary())

    # ------------------------------------------------------------------
    # 验证/批判层（按需加载；加载期间 processing_lock → /chat 503）
    # ------------------------------------------------------------------
    def _load_validator_critic(self):
        """默认关闭，按需加载（规范 §十.10）。DEMO 下为轻量规则实现。"""
        try:
            from validator import Validator
            from critic import Critic
            self.validator = Validator(self.cfg, translator=self.translator)
            self.critic = Critic(self.cfg, translator=self.translator)
        except Exception as e:
            self.log.warning("[ENGINE] 验证/批判层不可用（保持关闭）: %s", e)

    def _validate_with_lock(self, candidate, context):
        """加载验证层并评分（processing_lock 保护；加载期间其它 /chat 返回 503）。"""
        if self.validator is None:
            return 60.0, "验证层关闭（按需加载）"
        if self._validator_busy:
            raise RuntimeError("validator_busy")
        self._validator_busy = True
        try:
            if self.translator is not None and not self.translator.is_demo:
                self.translator.to_cpu()          # 路由层暂迁 CPU 释放显存（规范 §4.1）
            try:
                self.validator.load()
            except Exception:
                pass
            res = self.validator.validate(candidate, context)
            score = res.get("score", 60.0) if isinstance(res, dict) else float(res)
            reason = res.get("reason", "") if isinstance(res, dict) else ""
            try:
                self.validator.unload()
            except Exception:
                pass
            if self.translator is not None and not self.translator.is_demo:
                self.translator.to_cuda()         # 迁回 GPU（先预热后清缓存，规范 §九.6）
            return float(score), reason
        finally:
            self._validator_busy = False

    # ------------------------------------------------------------------
    # /chat 主流程
    # ------------------------------------------------------------------
    def chat(self, message: str, audio_path: str = None, image_path: str = None) -> dict:
        """完整对话管线（规范 §4.1 / §7.2）。"""
        t0 = time.time()
        try:
            # ① 情感（8 维，规范 §3.2）
            emotion = self.emotion.from_text(message)

            # ② 感知: 文本向量 + 听觉（可选）+ 视觉（可选）→ 融合（规范 §3.4）
            text_vec = self.translator.encode([message])[0]
            audio_vec = self.audio.process(audio_path=audio_path) if audio_path else \
                np.zeros(self.cfg.AUDIO_DIM, dtype=np.float32)
            visual_vec = self.translator.encode_image(image_path) if image_path else \
                np.zeros(self.cfg.VISUAL_DIM, dtype=np.float32)
            state = self.fusion.fuse(text_vec, audio_vec, visual_vec)

            # ③ 路由（真实 logits / 启发式降级，规范 §4.1）
            route = self.router.route(state, emotion, message)

            # ④ 扩散（按路由选择管线；L3 含 FAISS 0.3/0.7 融合，规范 §十.18）
            self.cfg.current_line = route["line_used"]
            line = self.line_ctrl.run(state, emotion, self._history_text())

            # ⑤ 解码候选（REAL: Qwen 生成；DEMO: 模板）
            # ⑤ 场景选择（借鉴 xilian 分场景提示词/温度）+ 人设事前引导
            scenario = self._pick_scenario(message)
            prompt = f"{message}\n{scenario['prompt']}"
            persona = getattr(self.cfg, "persona_system", None) or self.cfg.anchor_generation_prompt
            try:
                from cloud_model_client import SEED_EXAMPLES
                few_shots = SEED_EXAMPLES
            except Exception:
                few_shots = None
            candidate = self.translator.decode(
                prompt, persona=persona, few_shots=few_shots,
                temperature=scenario["temperature"])
            if not candidate or len(candidate) < 2:
                candidate = "（沉默）……我在这里。"

            # ⑥ 感性判定（规范 §4.1 感性模块，权重 0.6）
            summary = self._history_text()
            verdict = self.sentiment.judge(state, emotion, summary, candidate)
            if verdict["verdict"] == "reject":
                candidate = self.translator.decode(
                    prompt + "（换一种更柔和的说法）", persona=persona, few_shots=few_shots,
                    temperature=min(scenario["temperature"] + 0.1, 1.0))
                verdict = self.sentiment.judge(state, emotion, summary, candidate)
                self.sentiment.feedback(verdict["verdict"], outcome=False)
            elif verdict["verdict"] == "regenerate":
                candidate = self.translator.decode(
                    prompt + "（回到昔涟的身份）", persona=persona, few_shots=few_shots,
                    temperature=min(scenario["temperature"] - 0.1, 0.9))

            # ⑦ 验证层（按需，0.4）+ 感性仲裁（0.6，规范 §十.25）
            alignment = 60.0
            try:
                vscore, vreason = self._validate_with_lock(candidate, summary)
                blended = self.sentiment.blend_with_validator(vscore)
                alignment = round(blended["blended"] * 100.0)
                sentiment_override = bool(blended["sentiment_won"])
            except RuntimeError as e:
                if str(e) == "validator_busy":
                    raise
                alignment, sentiment_override = 60.0, False

            # ⑧ 肢体动作（输出层，规范 §4.1；好感度影响动作亲昵度）
            limb = self.limbs.react(state, emotion, candidate,
                                    affection=self.affection.value)

            # ⑨ 记忆写入 + 迁移（规范 §4.3）+ 好感度更新（借鉴 xilian affection）
            self.memory.add(state, text=message, emotion=emotion)
            self.affection.update(emotion, user_text=message,
                                  reward=0.3 if verdict["verdict"] == "pass" else -0.2)
            self.affection.save()

            # ⑩ 痛觉（低置信 → 错误路径标记，规范 §4.4）与格式化检查（§4.5）
            if route["confidence"] < self.cfg.direct_output_threshold:
                self.pain.update_from_error(error_vector=state)
            if self.pain.should_format():
                self.log.warning("[PAIN] 平均痛觉 %.1f > %.1f，触发格式化（向死而生）",
                                 self.pain.avg_pain, self.cfg.pain_format_trigger)
                fmt = self.formatting.run()
                candidate = fmt.get("phase6") or fmt.get("phase5") or "你好，世界"

            # ⑪ 异步训练（双缓冲，规范 §十.15）
            self.trainer.submit_hebbian(state, reward=0.5 if not route["direct_output"] else 0.2)
            self.trainer.submit_l2(text_vec, state)

            # 历史记录
            HISTORY.append({"role": "user", "content": message})
            HISTORY.append({"role": "assistant", "content": candidate})
            if len(HISTORY) > 20:
                del HISTORY[: len(HISTORY) - 20]

            resp = {
                "response_text": f"{limb['action']} {candidate}" if limb else candidate,
                "response_audio": None,
                "direct_mode": bool(route["direct_output"]),
                "line_used": route["line_used"],
                "emotion": self._emotion_dict(emotion),
                "dominance": round(self._dominance_of(emotion), 3),
                "affection": self.affection.value,
                "affection_level": self.affection.level(),
                "pain_value": round(float(self.pain.avg_pain), 3),
                "alignment_score": int(alignment),
                "sentiment_override": bool(alignment >= 60 and self.cfg.sentiment_weight >= 0.5),
                "response_time_ms": int((time.time() - t0) * 1000),
                "limb_action": limb.get("action", ""),
                "route_confidence": round(float(route["confidence"]), 3),
                "sentiment_verdict": verdict["verdict"],
            }
            self.log.info("[CHAT] %s → %s (line=%s direct=%s %.0fms)",
                          message[:20], candidate[:24], route["line_used"],
                          route["direct_output"], resp["response_time_ms"])
            return resp
        except Exception as e:
            self.log.error("[CHAT] 管线异常: %s", e, exc_info=True)
            return {
                "response_text": "（沉默）……我有些恍惚，让我缓一缓。",
                "response_audio": None, "direct_mode": False, "line_used": "L3",
                "emotion": self._emotion_dict(None), "pain_value": 0.0,
                "alignment_score": 40, "sentiment_override": False,
                "response_time_ms": int((time.time() - t0) * 1000),
            }

    # ------------------------------------------------------------------
    # 静默期自发激活（规范 §4.6 / §十.26）
    # ------------------------------------------------------------------
    def _silence_loop(self):
        """后台线程: 每 silence_random_recall_interval 秒随机召回一次；
        自发激活冷却 silence_cooldown_seconds（60s）。"""
        while True:
            time.sleep(10)
            now = time.time()
            interval = getattr(self.cfg, "silence_random_recall_interval", 300)
            if now - self._last_recall_ts < interval:
                continue
            self._last_recall_ts = now
            if now - self._last_spontaneous < getattr(self.cfg, "silence_cooldown_seconds", 60):
                continue
            try:
                recalls = self.memory.random_recall()
                if not recalls:
                    continue
                self._last_spontaneous = now
                state = recalls[0]["vector"]
                emotion = self.emotion.from_state(state)
                route = self.router.route(state, emotion, "", spontaneous=True)
                line = self.line_ctrl.run(state, emotion, self._history_text())
                candidate = self.translator.decode(
                    "（一段模糊的回忆浮现）" + recalls[0]["text"][:40],
                    persona=getattr(self.cfg, "persona_system", None)
                            or self.cfg.anchor_generation_prompt,
                    few_shots=None)
                verdict = self.sentiment.judge(state, emotion, self._history_text(), candidate)
                if verdict["verdict"] != "reject":
                    self.log.warning(
                        "\033[36m[SPONTANEOUS] 自发激活: %s\033[0m", candidate[:60])
                    HISTORY.append({"role": "assistant", "content": candidate})
            except Exception as e:
                self.log.warning("[SILENCE] 自发激活失败: %s", e)

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def _history_text(self) -> str:
        """最近 3 轮对话文本（验证/批判层精简上下文 ≤512 token，规范 §四.1）。"""
        recent = HISTORY[-6:] if len(HISTORY) >= 6 else HISTORY
        return "\n".join(f"{t['role']}: {t['content']}" for t in recent)[-512:]

    def _pick_scenario(self, message: str) -> dict:
        """按消息关键词选场景（借鉴 xilian 分场景提示词/温度）。

        返回 {"name", "prompt", "temperature"}。
        """
        templates = getattr(self.cfg, "scenario_templates", {}) or {}
        default = templates.get(self.cfg.scenario_default or "default", {})
        text = message or ""
        for name, sc in templates.items():
            kws = sc.get("keywords") or []
            if any(k in text for k in kws):
                return {"name": name,
                        "prompt": sc.get("prompt", ""),
                        "temperature": float(sc.get("temperature", 0.6))}
        return {"name": "default",
                "prompt": default.get("prompt", "（眼睫轻垂，斟酌了一下措辞）"),
                "temperature": float(default.get("temperature", 0.6))}

    @staticmethod
    def _dominance_of(emotion_vec) -> float:
        """PAD 第三维 dominance（借鉴 xilian PAD 引擎）: 从 8 维情绪向量估算。

        dominance = (angry×0.5 + neutral×0.3 − fear×0.4 − sad×0.3) ×0.5 + intensity×0.3
        ∈ [-1, 1]。高 = 主导/坚定；低 = 顺从/退缩。
        """
        e = np.asarray(emotion_vec, dtype=np.float32).reshape(-1)
        if e.size < 8:
            return 0.0
        p5 = np.clip(e[:5], 0.0, None)
        s = p5.sum()
        if s > 1e-9:
            p5 = p5 / s
        angry, sad, happy, neutral, fear = p5
        intensity = float(np.clip(e[5], 0.0, 1.0))
        d = (angry * 0.5 + neutral * 0.3 + happy * 0.1
             - fear * 0.4 - sad * 0.3) * 0.5 + intensity * 0.3
        return float(np.clip(d, -1.0, 1.0))

    @staticmethod
    def _emotion_dict(emotion_vec):
        """8 维情绪向量 → 可读 dict（trust/hope/sadness/...，规范 §7.2 风格）。"""
        if emotion_vec is None:
            return {"trust": 0.7, "sadness": 0.2, "pain": 0.05}
        e = np.asarray(emotion_vec, dtype=np.float32).reshape(-1)
        if e.size < 8:
            return {"trust": 0.7, "sadness": 0.2, "pain": 0.05}
        p5 = e[:5]
        return {
            "angry": round(float(p5[0]), 3), "sad": round(float(p5[1]), 3),
            "happy": round(float(p5[2]), 3), "neutral": round(float(p5[3]), 3),
            "fear": round(float(p5[4]), 3),
            "intensity": round(float(e[5]), 3), "arousal": round(float(e[6]), 3),
            "valence": round(float(e[7]), 3),
            "trust": round(float(0.7 * p5[2] + 0.3), 3),
            "pain": round(float(p5[4] * 0.5), 3),
        }

    def status(self) -> dict:
        af = self.affection.stats()
        return {
            "name": self.cfg.name,
            "mode": "DEMO(纯CPU流程验证)" if self.cfg.demo else "REAL",
            "model": self.cfg.qwen_model_name,
            "translator": "DEMO伪嵌入" if self.translator.is_demo else "REAL Qwen3.5-0.8B",
            "current_line": self.cfg.current_line,
            "memory": self.memory.stats(),
            "pain_avg": round(float(self.pain.avg_pain), 3),
            "pain_format_trigger": self.cfg.pain_format_trigger,
            "affection": af["affection"],
            "affection_level": af["level"],
            "history_len": len(HISTORY),
            "sentiment_weight": self.cfg.sentiment_weight,
            "vram": vram_summary(),
            "amp_mode": self.cfg.use_amp,
            "4bit_mode": self.cfg.use_4bit,
        }

    def shutdown(self):
        self.trainer.stop()
        try:
            self.memory.persist()
        except Exception:
            pass
        try:
            self.affection.save()
        except Exception:
            pass
        self.log.info("[ENGINE] 已停止并持久化记忆/好感度")


# ----------------------------------------------------------------------
# HTTP Handler
# ----------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    engine = None            # 由 main 注入

    # ---------------- 基础 ----------------
    def log_message(self, fmt, *args):
        logger.info("[HTTP] " + fmt % args)

    def _send(self, code: int, obj: dict):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n <= 0:
                return {}
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:
            return {}

    # ---------------- 路由 ----------------
    def do_GET(self):
        if self.path.startswith("/status"):
            self._send(200, {"ok": True, "status": self.engine.status()})
        else:
            self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        data = self._read_json()
        if self.path.startswith("/chat"):
            self._handle_chat(data)
        elif self.path.startswith("/switch"):
            line = str(data.get("line", "")).strip().upper()
            try:
                r = self.engine.line_ctrl.switch(line)
                self._send(200, {"ok": True, **r})
            except ValueError as e:
                self._send(400, {"ok": False, "error": str(e)})
        elif self.path.startswith("/reset"):
            global HISTORY
            HISTORY = []
            self._send(200, {"ok": True, "history_cleared": True})
        elif self.path.startswith("/format"):
            fmt = self.engine.formatting.run()
            self._send(200, {"ok": True, **fmt})
        else:
            self._send(404, {"ok": False, "error": "not found"})

    def _handle_chat(self, data: dict):
        message = str(data.get("message", "")).strip()
        if not message:
            self._send(400, {"ok": False, "error": "message 不能为空"})
            return
        # processing_lock（规范 §八.23）: 验证层加载中 → 503
        if self.engine._validator_busy:
            self._send(503, {"ok": False, "error": "验证层加载中，请稍后重试"})
            return
        audio_path = str(data.get("audio_path") or "").strip() or None
        image_path = str(data.get("image_path") or "").strip() or None
        resp = self.engine.chat(message, audio_path=audio_path, image_path=image_path)
        self._send(200, {"ok": True, **resp})


def create_server(engine, host: str = "127.0.0.1", port: int = 8080):
    Handler.engine = engine
    return ThreadingHTTPServer((host, port), Handler)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 自检: 装配引擎（DEMO）并跑一轮 /chat 主流程（不启动 HTTP）
    _cfg = config.get_config()
    _engine = Engine(_cfg)
    _r = _engine.chat("你好，昔涟。你还记得花海吗？")
    print("[SELFTEST] chat →", json.dumps(
        {k: _r[k] for k in ("response_text", "line_used", "direct_mode",
                            "alignment_score", "response_time_ms")},
        ensure_ascii=False))
    print("[SELFTEST] status →", json.dumps(_engine.status(), ensure_ascii=False)[:400])
    _engine.shutdown()
