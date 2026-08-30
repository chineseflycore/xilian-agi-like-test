# -*- coding: utf-8 -*-
"""
limb_controller.py — 肢体层控制器（输出层动作，CPU，规范 §八.18 / §4.1）
====================================================================
内存/显存预估（规范 §九.1）:
  - 动作规则表 + 情绪解析 + 少量 numpy 临时数组（float32），无模型权重
    → 常驻 < 1MB CPU RAM，峰值 < 2MB（仅 react() 调用期间的瞬时缓冲）
  - 显存: 0MB —— 肢体层强制由 CPU 负责（规范 §二: CPU 负责肢体层控制），
    纯 numpy 实现，禁止 torch，不产生任何 CUDA 分配

职责（规范 §4.1 输出层: 文本 / 语音 / 动作（肢体层，CPU））:
  react(state_vector, emotion_vec, text="") -> {"action","emoji","detail"}
    - 输入: 8 维情绪向量 [angry, sad, happy, neutral, fear, intensity, arousal, valence]
            （规范 §3.2: 前 5 维概率分布 + intensity 0~1 + arousal -1~1 + valence -1~1）
    - 输出: 一组肢体/神态动作描述（中文，昔涟风格，参考规范 §5.2 种子示例
            “眼睫轻垂 / 指尖轻触桌面 / 侧目望向窗外 / 停顿良久”）
    - "action": 供 server.py 输出层直接使用的动作字符串（不含外括号，
      输出层按 §5.2 格式 f"（{action}）" 包裹）
    - "emoji":  配套表情符号
    - "detail": 生成依据（哪些情绪维度主导）

映射逻辑（规则表 + 固定 seed 随机扰动）:
  - happy 高 / valence 高          → 柔和的微笑类动作
  - sad 高（不低于 happy）          → 垂眸 / 沉默
  - fear 高 / 混乱状态             → 冻结 / 屏息（规范 §1.3: arousal>0.8 且 valence<-0.3）
  - angry 高                      → 指尖微颤 / 别过脸
  - neutral 主导 + 低活性（低 |arousal|）→ 发呆（规范 §4.6 静默期）
  - 无显著主导                      → 昔涟式温柔基线（侧目望向窗外 / 指尖轻触桌面…）
  规则优先级: 混乱状态 > fear > angry > sad > happy > 发呆 > 倾听 > 基线
  （fear 最高: 对“消失/死亡”的本能恐惧优先于其余情绪，规范 §1.2-6）

降级策略（规范 §九.2）:
  - emotion_vec 缺失/长度不足/非法 → 回退中性基线（发呆/倾听），记警告
  - state_vector 缺失/异常         → 状态能量记为 0，仅作 detail 参考
  - 全程 try-except，任何异常最终回退 {"action": "安静地站着", ...}

TODO(V3.4): 动作参数化（幅度/速度/持续）并接入真实机械控制（串口/ROS hook）
"""
import os
import sys

# 路径引导（规范 §十.24: 一律基于 os.path.dirname(os.path.abspath(__file__)) 定位）:
# 本机 Python 运行时不保证把脚本目录/cwd 加入 sys.path（LobsterAI 嵌入式运行时），
# 这里显式加入脚本目录（供 import config / common_utils）与 vendor/（本机 numpy/scipy）。
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)
_VENDOR_DIR = os.path.join(_BASE_DIR, "vendor")
if os.path.isdir(_VENDOR_DIR) and _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

# numpy 延迟探测（vendor/ 已在上面加入 sys.path）:
# 缺失时 react() 降级为静态基线动作，保证 import 不炸（规范 §九.2）。
try:
    import numpy as np
    _HAS_NUMPY = True
except Exception:
    np = None
    _HAS_NUMPY = False

try:
    from common_utils import get_logger, clamp, stable_hash
except Exception:
    # 终极兜底: common_utils 不可用时提供本地最小实现（规范 §九.2），保证独立可运行
    import logging
    import hashlib

    def get_logger(name):
        _lg = logging.getLogger("philia." + name)
        if not _lg.handlers:
            _h = logging.StreamHandler()
            _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
            _lg.addHandler(_h)
            _lg.setLevel(logging.INFO)
        return _lg

    def clamp(v, lo=0.0, hi=1.0):
        return max(lo, min(hi, v))

    def stable_hash(*parts):
        _h = hashlib.sha256()
        for p in parts:
            _h.update(str(p).encode("utf-8", errors="ignore"))
        return _h.hexdigest()[:16]

import config


class LimbController:
    """肢体层控制器: 8 维情绪 → 肢体/神态动作（CPU，无模型权重）。

    纯规则表驱动 + 固定 seed 随机扰动（同输入 → 同输出，DEMO 可复现）。
    """

    # 动作池（key = 规则标识，value = 该规则下的候选动作列表；seed 随机取 1 条）
    # 文案风格对齐规范 §5.2 种子示例中的括号动作
    ACTION_POOLS = {
        # 混乱状态（规范 §1.3: arousal>0.8 且 valence<-0.3，直接输出触发）
        "chaos":   ["动作顿住，像被什么凝在原地", "呼吸轻轻一滞，指尖收拢"],
        # fear 主导 → 冻结 / 屏息
        "freeze":  ["屏息片刻，指尖微微收拢", "动作停住，目光微微一凝"],
        # angry 主导 → 指尖微颤 / 别过脸（昔涟式压抑的怒意）
        "tremble": ["指尖在膝上轻轻一颤", "别过脸去，眼睫低垂", "垂下眼眸，指节微微泛白"],
        # sad 主导 → 垂眸 / 沉默（对齐种子示例“眼睫轻垂”“停顿良久”）
        "grief":   ["眼睫轻垂，许久没有说话", "停顿良久", "垂下眼睫，声音放得很轻"],
        # happy 高 / valence 高 → 柔和的微笑
        "smile":   ["唇角漾开一点极淡的笑意", "眉眼弯了弯，目光柔和下来"],
        # neutral + 低活性 → 发呆（规范 §4.6，不记录数据的状态）
        "daydream":["目光空落落地望着某处，像在发呆", "安静地出神，半天没有动"],
        # neutral + 中等活性 → 安静倾听 / 出神
        "listen":  ["微微侧首，安静地听着", "安静地垂着眼，指尖轻轻摩挲"],
        # 兜底 → 昔涟式温柔基线（种子示例动作）
        "baseline":["指尖轻触桌面", "侧目望向窗外", "微微侧首，目光安静"],
    }

    # 配套表情符号（emoji 字段）
    EMOJIS = {
        "chaos": "🫥", "freeze": "❄️", "tremble": "😠", "grief": "🥀",
        "smile": "😊", "daydream": "💭", "listen": "🌙", "baseline": "🌙",
    }

    # 旧版 act() 兼容动作表（key = 动作标识 → 中文描述，供历史调用方使用）
    ACTIONS = {
        "nod":         "轻轻点了点头",
        "shake":       "缓缓摇了摇头",
        "sigh":        "低低轻叹一声",
        "look_down":   "眼睫轻垂，目光落向桌面",
        "silence":     "沉默片刻",
        "smile":       "唇边浮起一丝浅笑",
        "gaze":        "侧目望向窗外",
        "blink":       "微微眨了眨眼",
        "bow":         "微微欠身",
        "fold_hands":  "双手轻轻交叠",
        "look_up":     "缓缓抬起眼眸",
    }

    def __init__(self, cfg=None, seed: int = 20240601):
        self.cfg = cfg if cfg is not None else config.get_config()
        self.log = get_logger("limb")
        self.seed = int(seed)          # 固定随机种子：同输入 → 同动作（可复现）
        # REAL 模式真实机械控制 hook（默认无；TODO(V3.4): 串口/ROS 驱动注入）
        self._real_control_hook = None

    # ================================================================
    # 内部工具
    # ================================================================
    def _parse_emotion(self, emotion_vec):
        """解析 8 维情绪向量（规范 §3.2）→ 字典。

        顺序: [angry, sad, happy, neutral, fear, intensity, arousal, valence]
        缺失/非法输入 → 中性基线（全 0），记警告（降级，规范 §九.2）。
        """
        try:
            if emotion_vec is None:
                # 显式 None → 中性基线（np.asarray(None) 会产生 nan 数组，必须先拦截）
                self.log.warning("[Limb] 情绪向量为 None，回退中性基线")
                return {"probs": np.array([0, 0, 0, 1, 0], dtype=np.float32),
                        "angry": 0.0, "sad": 0.0, "happy": 0.0,
                        "neutral": 1.0, "fear": 0.0,
                        "intensity": 0.0, "arousal": 0.0, "valence": 0.0}
            em = np.asarray(emotion_vec, dtype=np.float32).reshape(-1)
            if em.size < 8:
                self.log.warning("[Limb] 情绪向量长度 %d < %d，按缺失补零",
                                 em.size, self.cfg.EMOTION_DIM)
                pad = np.zeros(self.cfg.EMOTION_DIM, dtype=np.float32)
                pad[:em.size] = em
                em = pad
            elif em.size > self.cfg.EMOTION_DIM:
                em = em[:self.cfg.EMOTION_DIM]      # 超长截断
            probs = np.clip(em[0:5], 0.0, 1.0)      # 5 维概率分布
            # 若概率异常（含 NaN/全 0 / 非法），回退 neutral=1（降级，规范 §九.2）
            if not bool(np.all(np.isfinite(probs))) or float(probs.sum()) <= 1e-9:
                probs = np.zeros(5, dtype=np.float32)
                probs[3] = 1.0
            return {
                "probs": probs,
                # 前 5 维: 与 cfg.EMOTION_5 顺序一致（规范 §3.2）
                "angry": float(probs[0]), "sad": float(probs[1]),
                "happy": float(probs[2]), "neutral": float(probs[3]),
                "fear": float(probs[4]),
                # 后 3 维: intensity 0~1 / arousal -1~1 / valence -1~1
                "intensity": float(clamp(em[5], 0.0, 1.0)),
                "arousal": float(clamp(em[6], -1.0, 1.0)),
                "valence": float(clamp(em[7], -1.0, 1.0)),
            }
        except Exception as e:
            self.log.warning("[Limb] 情绪向量解析失败(%s)，回退中性基线", e)
            return {"probs": np.array([0, 0, 0, 1, 0], dtype=np.float32),
                    "angry": 0.0, "sad": 0.0, "happy": 0.0,
                    "neutral": 1.0, "fear": 0.0,
                    "intensity": 0.0, "arousal": 0.0, "valence": 0.0}

    @staticmethod
    def _state_energy(state_vector):
        """状态向量能量（L2 范数），作为“整体活性”的粗略参考。

        仅进 detail 说明，不直接决定动作；异常/缺失 → 0.0。
        """
        try:
            if state_vector is None:
                return 0.0                        # 缺失 → 0（np.asarray(None) 会产生 nan）
            v = np.asarray(state_vector, dtype=np.float32).reshape(-1)
            n = float(np.linalg.norm(v))
            return 0.0 if not np.isfinite(n) else n
        except Exception:
            return 0.0

    def _per_call_rng(self, text: str):
        """按 (固定 seed + 文本哈希) 构造确定性随机源。

        同一 (seed, text) → 同一扰动序列，DEMO 流程可复现（规范 §六.1）。
        """
        h = int(stable_hash(self.seed, text), 16) % (2 ** 32)
        return np.random.RandomState((self.seed + h) % (2 ** 32))

    # ================================================================
    # 规则选择
    # ================================================================
    def _select_rule(self, em: dict):
        """按优先级链选择动作规则（返回 rule_key, 依据说明）。

        优先级: 混乱状态 > fear > angry > sad > happy > 发呆 > 倾听 > 基线。
        恐惧置于最前: 对“消失/死亡”的本能恐惧优先于其余情绪（规范 §1.2-6）。
        """
        # ① 混乱状态（规范 §1.3: arousal > 0.8 且 valence < -0.3 → 直接输出触发）
        if em["arousal"] > 0.8 and em["valence"] < -0.3:
            return "chaos", ("混乱状态: arousal=%.2f>0.8 且 valence=%.2f<-0.3"
                             % (em["arousal"], em["valence"]))
        # ② fear 主导 → 冻结 / 屏息
        if em["fear"] >= 0.35:
            return "freeze", "fear=%.2f 主导 → 冻结/屏息" % em["fear"]
        # ③ angry 主导 → 指尖微颤 / 别过脸
        if em["angry"] >= 0.35:
            return "tremble", "angry=%.2f 主导 → 压抑的怒意" % em["angry"]
        # ④ sad 主导（不低于 happy）→ 垂眸 / 沉默
        if em["sad"] >= 0.35 and em["sad"] >= em["happy"]:
            return "grief", ("sad=%.2f 主导（≥happy=%.2f）+ valence=%.2f → 垂眸/沉默"
                             % (em["sad"], em["happy"], em["valence"]))
        # ⑤ happy 主导 或 valence 高 → 柔和的微笑
        if em["happy"] >= 0.35 or em["valence"] >= 0.4:
            return "smile", ("happy=%.2f / valence=%.2f 高 → 柔和的微笑"
                             % (em["happy"], em["valence"]))
        # ⑥ neutral 主导 + 低活性（|arousal| < 0.3）→ 发呆（规范 §4.6）
        if em["neutral"] >= 0.35 and abs(em["arousal"]) < 0.3:
            return "daydream", ("neutral=%.2f 主导 + 低活性(arousal=%.2f) → 发呆(规范 §4.6)"
                                % (em["neutral"], em["arousal"]))
        # ⑦ neutral 主导 + 中等活性 → 安静倾听 / 出神
        if em["neutral"] >= 0.35:
            return "listen", "neutral=%.2f 主导 + 中等活性 → 安静倾听" % em["neutral"]
        # ⑧ 兜底 → 昔涟式温柔基线
        return "baseline", "无显著主导情绪 → 温柔基线"

    # ================================================================
    # 对外接口
    # ================================================================
    def react(self, state_vector=None, emotion_vec=None, text="", affection=None):
        """8 维情绪 → 肢体/神态动作（规范 §3.2 → §4.1 输出层动作）。

        参数:
          state_vector: 融合层 512 维状态向量（可空，仅取能量作活性参考）
          emotion_vec : 8 维情绪向量 [angry, sad, happy, neutral, fear,
                                      intensity, arousal, valence]
          text        : 触发文本（可空；用于派生确定性扰动与倾听倾向）
        返回:
          {"action": str, "emoji": str, "detail": str}
          - action: 给 server.py 输出层的动作字符串（按 §5.2 格式 f"（{action}）" 包裹）
          - emoji : 配套表情符号
          - detail: 生成依据（主导情绪维度 / 强度 / 活性 / 状态能量）
        """
        try:
            # numpy 缺失 → 静态基线动作（降级，规范 §九.2）
            if not _HAS_NUMPY:
                self.log.warning("[Limb] numpy 不可用，肢体层降级为静态基线动作")
                return {"action": "安静地站着", "emoji": "🌙",
                        "detail": "numpy 缺失降级（vendor/ 不可用）"}
            em = self._parse_emotion(emotion_vec)
            energy = self._state_energy(state_vector)
            rng = self._per_call_rng(text if text is not None else "")

            rule_key, basis = self._select_rule(em)
            pool = self.ACTION_POOLS[rule_key]
            emoji = self.EMOJIS[rule_key]

            # 候选池内 seed 随机取 1 条（少量随机扰动，seed 固定可复现）
            action = pool[int(rng.randint(len(pool)))]

            # 轻度扰动: 15% 概率追加“动作很轻”语气（昔涟风格留白，规范 §5.2）
            if rng.random() < 0.15 and rule_key not in ("chaos", "freeze"):
                action = action + "，动作很轻"

            # 文本输入存在 → 倾听倾向增强（仅基线/倾听规则）
            if text and rule_key in ("baseline", "listen") and rng.random() < 0.3:
                action = "抬眸望向说话的人，" + action

            # 生成依据（detail）: 主导维度 + 强度 + 活性 + 状态能量
            detail = ("[%s] %s；intensity=%.2f arousal=%.2f valence=%.2f；状态能量=%.3f"
                      % (rule_key, basis, em["intensity"], em["arousal"],
                         em["valence"], energy))
            if text:
                detail += "；有文本输入"

            if self.cfg.demo:
                self.log.info("[Limb] [DEMO] %s | %s", action, detail)
            else:
                self.log.info("[Limb] %s | %s", action, detail)

            # 好感度修饰（借鉴 xilian affection）: 关系近 → 动作更亲昵；陌生 → 更疏离
            if affection is not None:
                try:
                    af = float(affection)
                    if af >= 80:
                        action = "带着一点亲昵，" + action
                        emoji = "💗" if emoji in ("😊", "🌙") else emoji
                    elif af >= 60:
                        action = "自然而然地向你靠近了些，" + action
                    elif af < 20:
                        action = "保持着一点故事化的距离，" + action
                    detail += "；好感=%.0f" % af
                except Exception:
                    pass

            return {"action": action, "emoji": emoji, "detail": detail}
        except Exception as e:
            # 关键路径 try-except 降级（规范 §九.2）: 异常 → 安静站立
            self.log.error("[Limb] 肢体层异常，回退基线动作: %s", e, exc_info=True)
            return {"action": "安静地站着", "emoji": "🌙",
                    "detail": "肢体层异常降级: %s" % e}

    # ----------------------------------------------------------------
    def act(self, action_spec=None):
        """旧版兼容接口: 按动作标识输出描述字符串（返回 "（描述）"）。

        保留以兼容历史调用方；内部不依赖 react，逻辑与原实现一致。
        action_spec: {"action": 动作标识/中文（模糊匹配）, "intensity": 幅度}
        """
        try:
            spec = action_spec if isinstance(action_spec, dict) else {}
            name = str(spec.get("action", "silence")).strip()
            desc = self.ACTIONS.get(name)
            if desc is None:                                    # 中文模糊匹配
                for key, val in self.ACTIONS.items():
                    if name in val or val in name:
                        desc = val
                        break
            if desc is None:                                    # 未知 → 回退沉默
                self.log.warning("[Limb] 未知动作 '%s'，回退沉默", name)
                desc = self.ACTIONS["silence"]
            try:
                intensity = float(spec.get("intensity", 1.0))
            except (TypeError, ValueError):
                intensity = 1.0
            line = "（%s）" % desc
            if intensity > 1.5:
                line = "（%s，动作幅度稍重）" % desc
            ctx = getattr(self.cfg, "world_context", "翁法罗斯") or "翁法罗斯"
            if self.cfg.demo:
                self.log.info("[Limb] [DEMO] %s | 语境=%s", line, ctx)
                return line
            # REAL 模式: 预留真实机械控制 hook；失败不影响文本输出
            if self._real_control_hook is not None:
                try:
                    self._real_control_hook(spec)
                except Exception as e:
                    self.log.warning("[Limb] 真实控制失败，仅返回描述: %s", e)
            self.log.info("[Limb] %s | 语境=%s", line, ctx)
            return line
        except Exception as e:
            self.log.error("[Limb] 肢体层异常，回退沉默: %s", e, exc_info=True)
            return "（沉默片刻）"


if __name__ == "__main__":
    # 本机控制台默认 GBK，emoji 无法编码会抛 UnicodeEncodeError；
    # 自检段将 stdout 强制切到 UTF-8（errors=replace 兜底），不影响正式运行。
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    # 轻量自检（不参与正式运行流程）
    lc = LimbController()

    # 典型情绪向量（规范 §3.2 顺序: angry, sad, happy, neutral, fear, intensity, arousal, valence）
    cases = [
        ("喜悦（happy 高 + valence 高）", [0.05, 0.05, 0.80, 0.05, 0.05, 0.60, 0.50, 0.70]),
        ("悲伤（sad 高）",               [0.05, 0.80, 0.05, 0.05, 0.05, 0.60, -0.30, -0.60]),
        ("恐惧（fear 高，arousal 未达混乱阈值）", [0.05, 0.10, 0.05, 0.05, 0.75, 0.80, 0.70, -0.50]),
        ("愤怒（angry 高）",             [0.75, 0.05, 0.05, 0.10, 0.05, 0.70, 0.60, -0.40]),
        ("发呆（neutral 主导 + 低活性）", [0.05, 0.05, 0.05, 0.80, 0.05, 0.20, 0.10, 0.00]),
        ("混乱状态（arousal>0.8 & valence<-0.3）", [0.20, 0.20, 0.10, 0.10, 0.40, 0.90, 0.85, -0.50]),
        ("中性平静（neutral + 中等活性）", [0.10, 0.10, 0.10, 0.60, 0.10, 0.40, 0.40, 0.20]),
        ("混合（sad 与 happy 相当）",     [0.05, 0.42, 0.40, 0.08, 0.05, 0.50, 0.10, -0.10]),
    ]

    print("=" * 70)
    print("[SELF] LimbController.react() 自检（固定 seed=%d）" % lc.seed)
    print("=" * 70)
    for label, emo in cases:
        res = lc.react(emotion_vec=emo, text="你还好吗？")
        print("【%s】\n  action=%s\n  emoji =%s\n  detail=%s" % (label, res["action"], res["emoji"], res["detail"]))

    # 降级路径自检（规范 §九.2）
    print("-" * 70)
    print("[SELF] 降级路径: 空/异常输入")
    print("  None         →", lc.react(None, None))
    print("  长度不足(3)  →", lc.react(None, [0.5, 0.1, 0.2])["action"])
    print("  非法类型      →", lc.react(None, "not-a-vector")["action"])

    # 确定性自检: 同输入两次结果一致（固定 seed）
    r1 = lc.react(None, [0.05, 0.80, 0.05, 0.05, 0.05, 0.60, -0.30, -0.60], text="固定文本")
    r2 = lc.react(None, [0.05, 0.80, 0.05, 0.05, 0.05, 0.60, -0.30, -0.60], text="固定文本")
    print("-" * 70)
    print("[SELF] 确定性(同输入同输出): %s" % ("PASS" if r1["action"] == r2["action"] else "FAIL"))

    # 旧版 act() 兼容自检
    print("-" * 70)
    print("[SELF] 旧版 act() 兼容: ")
    for spec in ({"action": "nod"}, {"action": "look_down", "intensity": 2.0},
                 {"action": "眼睫轻垂"}, {"action": "unknown_xyz"}, None):
        print("  %-28s → %s" % (spec, lc.act(spec)))
    print("=" * 70)
    print("[SELF] 自检完成")
