# -*- coding: utf-8 -*-
"""
perception_audio.py — 听觉感知层（规范 §3.1，CPU，总延迟 < 500ms）
=================================================================
内存/显存预估（规范 §九.1）:
  - 音频缓冲: 16kHz 单声道 float32 × ≤5s ≈ 320KB（CPU RAM，超长截断）
  - silero-vad v5.0+: 约 2MB（CPU RAM，torch 权重，首次加载需 torch.hub 下载）
  - L1~L4 参数: 轻量 CNN 占位 + 事件嵌入表 + numpy GRU，合计 < 1MB（CPU RAM）
  - SenseVoiceSmall ONNX INT8: 惰性加载，加载后约 250MB（CPU RAM）；无模型文件则 0MB
  - 显存: 0MB —— 感知层强制 CPU 运行，不占用 GPU 显存（规范 §二）

层次结构（规范 §3.1）:
  L1 声学场景   64 维   轻量 CNN（numpy 1D 卷积占位）
  L2 事件流     128 维  事件嵌入表（能量分箱 → 嵌入聚合）
  L3 人声特征   128 维  SenseVoiceSmall（ONNX INT8，输出必须 .astype(np.float32)）
  L4 动态时序   192 维  GRU 编码器（numpy 实现）
  合计: 64 + 128 + 128 + 192 = 512 维

VAD 降级链（规范 §3.1 修正规格）:
  silero-vad v5.0+ 的 load_silero_vad()（必须 map_location=torch.device('cpu') 加载）
    → webrtcvad（C 扩展）
    → 无可用后端: 打印 [VAD] 警告并返回 512 维零向量
  VAD 检测无语音段 → 返回 512 维零向量并记 [VAD] 日志，禁止对空列表索引

降级策略（规范 §九.2）:
  任一依赖缺失 → 日志警告 + 返回零向量或伪向量（common_utils.pseudo_embedding 填充噪声）

TODO(V3.4): 接入真实 SenseVoiceSmall ONNX INT8 模型与训练后的 GRU 权重
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

import numpy as np

import config
from common_utils import get_logger, Timer, safe_import, normalize, pseudo_embedding


class AudioPerception:
    """听觉感知层: 四层级联 → 512 维听觉状态向量（规范 §3.1）。

    process(audio_path=None, waveform=None) -> np.ndarray (512,)
      任选其一提供音频；两者都为空（DEMO 模式）→ 直接返回零向量并记 [VAD] 日志。
    """

    SAMPLE_RATE = 16000        # silero-vad / webrtcvad 均要求 16kHz
    MAX_AUDIO_SEC = 5.0        # 超长音频截断，保证总延迟 < 500ms
    N_EVENTS = 32              # L2 事件嵌入表条目数
    L1_FEATS = 8               # 帧级声学特征维度（能量/过零率/频谱质心等）
    L1_C_OUT = 16              # 轻量 CNN 输出通道（占位）
    L1_KERNEL = 3              # 卷积核长度
    FRAME_LEN = 320            # 20ms @16k
    FRAME_HOP = 160            # 10ms @16k
    MAX_FRAMES = 64            # 特征帧上限（超出则均匀抽样）

    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else config.get_config()
        self.log = get_logger("perception.audio")
        self.rng = np.random.RandomState(20240601)
        # 模型路径一律基于 BASE_DIR（规范 §九.12/§十.24，禁止裸文件名/相对路径）
        self.sensevoice_onnx_path = os.path.join(
            config.BASE_DIR, "models", "sensevoice_small_int8.onnx")
        self.vad_backend = None      # "silero" | "webrtc" | None
        self._vad = None             # VAD 模型对象
        self._ort_session = None     # onnxruntime 推理会话（惰性加载）
        self._init_l1()
        self._init_l2()
        self._init_l4()
        self._init_vad()

    # ================================================================
    # 初始化
    # ================================================================
    def _init_l1(self):
        """L1 轻量 CNN 占位: 1D 卷积(8→16) + 时间全局平均池化 + 全连接(16→64)。"""
        self.l1_conv = self.rng.normal(
            0.0, 0.1, (self.L1_C_OUT, self.L1_FEATS, self.L1_KERNEL)).astype(np.float32)
        self.l1_fc = self.rng.normal(
            0.0, 0.1, (self.L1_C_OUT, self.cfg.audio_l1_dim)).astype(np.float32)
        self.l1_bias = np.zeros(self.cfg.audio_l1_dim, dtype=np.float32)

    def _init_l2(self):
        """L2 事件嵌入表: N_EVENTS × 128 随机嵌入（TODO(V3.4): 训练可学习嵌入）。"""
        self.event_table = self.rng.normal(
            0.0, 1.0, (self.N_EVENTS, self.cfg.audio_l2_dim)).astype(np.float32)

    def _init_l4(self):
        """L4 GRU 参数（numpy 实现）: 输入 64+128+128=320 → 隐藏 192。"""
        d = self.cfg.audio_l4_dim
        inp = self.cfg.audio_l1_dim + self.cfg.audio_l2_dim + self.cfg.audio_l3_dim
        self.gru_in = inp
        self.gru_hidden = d
        s_u = 1.0 / np.sqrt(d)       # 循环权重缩放
        s_w = 1.0 / np.sqrt(inp)     # 输入权重缩放

        def _w(shape, scale):
            return self.rng.normal(0.0, scale, shape).astype(np.float32)

        self.gru = {
            "Wz": _w((d, inp), s_w), "Uz": _w((d, d), s_u), "bz": np.zeros(d, np.float32),
            "Wr": _w((d, inp), s_w), "Ur": _w((d, d), s_u), "br": np.zeros(d, np.float32),
            "Wh": _w((d, inp), s_w), "Uh": _w((d, d), s_u), "bh": np.zeros(d, np.float32),
        }
        # 帧级特征(8) → 64 维投影，供 L4 序列输入使用
        self.frame_proj = self.rng.normal(
            0.0, 0.1, (self.L1_FEATS, self.cfg.audio_l1_dim)).astype(np.float32)

    def _init_vad(self):
        """VAD 初始化降级链: silero-vad v5.0+ → webrtcvad → None（规范 §3.1）。"""
        # ① silero-vad v5.0+（优先，轻量 PyTorch 模型，约 2MB）
        try:
            torch_mod = safe_import("torch")
            if torch_mod is None:
                raise ImportError("torch 不可用（无 CUDA 推理路径时亦可纯 CPU 使用）")
            silero = safe_import("silero_vad")
            if silero is None or not hasattr(silero, "load_silero_vad"):
                raise ImportError("silero_vad v5.0+ 未安装或无 load_silero_vad() API")
            # 规范 §3.1/§十.21: 必须使用 load_silero_vad()（禁止已废弃的 load_model()）
            self._vad = silero.load_silero_vad()
            # 必须显式 map_location=torch.device('cpu') 加载/迁移（规范 §3.1 修正规格）
            self._vad = self._vad.to(torch_mod.device("cpu"))
            self._vad.eval()
            self.vad_backend = "silero"
            self.log.info("[VAD] silero-vad v5.0+ 就绪 (CPU)")
        except Exception as e:
            # ② webrtcvad（C 扩展备选）
            try:
                webrtc = safe_import("webrtcvad")
                if webrtc is None:
                    raise ImportError("webrtcvad 未安装")
                self._vad = webrtc.Vad(2)   # 严格度 0~3，2 为平衡档
                self.vad_backend = "webrtc"
                self.log.warning("[VAD] silero-vad 不可用(%s)，降级 webrtcvad", e)
            except Exception as e2:
                # ③ 无可用后端 → 后续一律按“无语音”处理（返回零向量）
                self._vad = None
                self.vad_backend = None
                self.log.warning("[VAD] 无可用 VAD 后端(%s; %s)，音频感知将返回零向量", e, e2)

    # ================================================================
    # 对外主入口
    # ================================================================
    def process(self, audio_path=None, waveform=None):
        """主流程: 音频 → VAD → 四层特征 → 512 维向量（规范 §3.1）。

        返回: np.ndarray (512,)，float32。
        """
        timer = Timer("audio_perception")
        timer.__enter__()
        try:
            if audio_path is None and waveform is None:
                # DEMO 模式（无音频输入）: 直接返回零向量并记 [VAD] 日志
                self.log.info("[VAD] 无音频输入（DEMO 模式），返回 512 维零向量")
                return np.zeros(self.cfg.AUDIO_DIM, dtype=np.float32)
            wave = self._load_waveform(audio_path, waveform)
            if wave is None:
                return np.zeros(self.cfg.AUDIO_DIM, dtype=np.float32)
            # VAD 语音检测
            if not self._has_speech(wave):
                # 规范 §3.1: VAD 无语音段 → 零向量，禁止对空列表索引
                self.log.warning("[VAD] 无有效语音段，返回 512 维零向量")
                return np.zeros(self.cfg.AUDIO_DIM, dtype=np.float32)
            # 四层级联
            feats = self._extract_features(wave)      # (T, 8)
            l1 = self._l1_cnn(feats)                  # 64
            l2 = self._l2_events(feats)               # 128
            l3 = self._l3_sensevoice(feats)           # 128
            l4 = self._l4_gru(l2, l3, feats)          # 192
            vec = np.concatenate([l1, l2, l3, l4]).astype(np.float32)
            vec = normalize(vec)
            self.log.info("[AUDIO] 四层完成 L1=64 L2=128 L3=128 L4=192 → 512")
            return vec
        except Exception as e:
            # 关键路径 try-except 降级（规范 §九.2）: 异常 → 零向量
            self.log.error("[AUDIO] 感知异常，降级为零向量: %s", e, exc_info=True)
            return np.zeros(self.cfg.AUDIO_DIM, dtype=np.float32)
        finally:
            # 总延迟预算监控（规范 §3.1: <500ms）；finally 保证无论成败均计时
            timer.__exit__(None, None, None)
            if timer.elapsed > 500.0:
                self.log.warning("[PERF] 听觉感知总延迟 %.0fms 超出 500ms 预算", timer.elapsed)

    # ================================================================
    # 音频加载与 VAD
    # ================================================================
    def _load_waveform(self, audio_path, waveform):
        """加载/校验 16kHz float32 单声道波形；失败返回 None（调用方转零向量）。"""
        if waveform is not None:
            wave = np.asarray(waveform, dtype=np.float32).reshape(-1)
            if wave.size == 0:
                self.log.warning("[VAD] waveform 为空")
                return None
            return self._trim(wave)
        if not audio_path:
            return None
        ap = str(audio_path)
        try:
            from scipy.io import wavfile
            sr, data = wavfile.read(ap)
            wave = self._to_float(data)
            if sr != self.SAMPLE_RATE:
                wave = self._resample(wave, sr)
            return self._trim(wave)
        except Exception as e:
            # 非 wav 格式 → 尝试 soundfile（若已安装）
            try:
                sf = safe_import("soundfile")
                if sf is None:
                    raise ImportError("soundfile 未安装，仅支持 wav")
                data, sr = sf.read(ap, dtype="float32", always_2d=False)
                wave = np.asarray(data, dtype=np.float32).reshape(-1)
                if sr != self.SAMPLE_RATE:
                    wave = self._resample(wave, sr)
                return self._trim(wave)
            except Exception as e2:
                self.log.warning("[VAD] 音频加载失败(%s; %s)", e, e2)
                return None

    @staticmethod
    def _to_float(data):
        """wavfile 读取的整数/浮点 PCM → float32 [-1, 1]。"""
        data = np.asarray(data)
        if data.ndim > 1:
            data = data.mean(axis=1)  # 多声道 → 单声道（取均值）
        if data.dtype == np.float32 or data.dtype == np.float64:
            return np.asarray(data, dtype=np.float32)
        if data.dtype == np.int16:
            return (data.astype(np.float32) / 32768.0)
        if data.dtype == np.int32:
            return (data.astype(np.float32) / 2147483648.0)
        if data.dtype == np.uint8:
            return ((data.astype(np.float32) - 128.0) / 128.0)
        return np.asarray(data, dtype=np.float32)

    @staticmethod
    def _resample(x, sr):
        """线性插值重采样到 16kHz（numpy 实现，无 librosa 依赖）。"""
        n_out = int(round(len(x) * AudioPerception.SAMPLE_RATE / float(sr)))
        if n_out <= 0:
            return np.zeros(0, dtype=np.float32)
        idx = np.linspace(0.0, len(x) - 1.0, n_out)
        return np.interp(idx, np.arange(len(x)), x).astype(np.float32)

    def _trim(self, wave):
        """截断超长音频（≤ MAX_AUDIO_SEC 秒），控制处理延迟。"""
        max_n = int(self.SAMPLE_RATE * self.MAX_AUDIO_SEC)
        if len(wave) > max_n:
            self.log.info("[VAD] 音频超长，截断至 %.1fs", self.MAX_AUDIO_SEC)
            wave = wave[:max_n]
        return wave

    def _has_speech(self, wave):
        """VAD 语音检测；无后端/异常/无语音 → False（调用方返回零向量）。"""
        if self.vad_backend is None:
            self.log.warning("[VAD] 无 VAD 后端，按无语音处理")
            return False
        try:
            if self.vad_backend == "silero":
                return self._has_speech_silero(wave)
            if self.vad_backend == "webrtc":
                return self._has_speech_webrtc(wave)
        except Exception as e:
            self.log.warning("[VAD] 语音检测异常(%s)，按无语音处理", e)
        return False

    def _has_speech_silero(self, wave):
        """silero-vad 推理（CPU，逐 512 样本块打分，禁止对空列表索引）。"""
        torch_mod = safe_import("torch")
        if torch_mod is None:
            return False
        t = torch_mod.from_numpy(wave).float()
        step = 512
        probs = []
        with torch_mod.no_grad():
            for i in range(0, max(1, t.numel()), step):
                chunk = t[i:i + step]
                if chunk.numel() == 0:
                    continue
                try:
                    p = self._vad(chunk, self.SAMPLE_RATE)
                    probs.append(float(p[0]))
                except Exception:
                    continue
        if not probs:  # 规范 §3.1: 禁止对空列表索引
            return False
        return any(p > 0.5 for p in probs)

    def _has_speech_webrtc(self, wave):
        """webrtcvad 推理（20ms 帧，语音帧占比 > 30% 视为有语音）。"""
        pcm = (np.clip(wave, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        fl = self.FRAME_LEN  # 20ms @16k = 320 样本
        n = len(wave) // fl
        if n == 0:
            return False
        speech = 0
        for i in range(n):
            frame = pcm[i * fl:(i + 1) * fl]
            try:
                if self._vad.is_speech(frame, self.SAMPLE_RATE):
                    speech += 1
            except Exception:
                continue
        return (speech / n) > 0.3

    # ================================================================
    # 特征提取
    # ================================================================
    def _extract_features(self, wave):
        """分帧提取 8 维声学特征: [对数能量, 过零率, 频谱质心, 频谱平坦度, 4 频带能量比]。"""
        n = len(wave)
        if n < self.FRAME_LEN:
            # 音频过短 → 补零到 1 帧（避免产生空帧序列）
            wave = np.pad(wave, (0, self.FRAME_LEN - n))
            n = len(wave)
        frames = [wave[s:s + self.FRAME_LEN]
                  for s in range(0, n - self.FRAME_LEN + 1, self.FRAME_HOP)]
        if not frames:  # 规范 §3.1: 禁止对空列表索引
            return np.zeros((1, self.L1_FEATS), dtype=np.float32)
        feats = np.stack(frames).astype(np.float32)  # (T, 320)
        # 帧数超上限 → 均匀抽样（确定性）
        T = feats.shape[0]
        if T > self.MAX_FRAMES:
            pick = np.linspace(0, T - 1, self.MAX_FRAMES).astype(np.int64)
            feats = feats[pick]
        # 频谱
        spec = np.abs(np.fft.rfft(feats, axis=1))                       # (T, 161)
        freqs = np.fft.rfftfreq(self.FRAME_LEN, 1.0 / self.SAMPLE_RATE)  # (161,)
        # 对数能量
        rms = np.sqrt((feats ** 2).mean(axis=1) + 1e-12)
        log_e = np.log(rms + 1e-12)
        # 过零率
        zcr = (np.abs(np.diff(feats, axis=1)) > 1e-4).sum(axis=1) / (self.FRAME_LEN - 1)
        # 频谱质心
        spec_sum = spec.sum(axis=1) + 1e-12
        centroid = (spec * freqs[None, :]).sum(axis=1) / spec_sum
        # 频谱平坦度（近似）
        flat = (np.exp((np.log(spec + 1e-12) * spec).sum(axis=1) / spec_sum)
                / (spec_sum / spec.shape[1] + 1e-12))
        # 4 频带能量比（0-1k / 1k-2k / 2k-4k / 4k-8k）
        def _band(a, b):
            m = (freqs >= a) & (freqs < b)
            return (spec[:, m] ** 2).sum(axis=1)
        b1, b2, b3, b4 = _band(0, 1000), _band(1000, 2000), _band(2000, 4000), _band(4000, 8000)
        btot = b1 + b2 + b3 + b4 + 1e-12
        out = np.stack([log_e, zcr, centroid / 8000.0, flat,
                        b1 / btot, b2 / btot, b3 / btot, b4 / btot], axis=1)
        return out.astype(np.float32)

    # ================================================================
    # L1 ~ L4
    # ================================================================
    def _l1_cnn(self, feats):
        """L1 声学场景(64): numpy 1D 卷积(8→16) + 时间全局平均池化 + FC(16→64) + tanh。"""
        T, F = feats.shape
        if T < self.L1_KERNEL:
            # 帧数不足卷积核 → 尾部补零到可卷积长度（避免索引越界）
            pad = np.zeros((self.L1_KERNEL - T, F), dtype=np.float32)
            feats = np.concatenate([feats, pad], axis=0)
            T = feats.shape[0]
        # 滑窗 (T-K+1, K, F)
        windows = np.stack([feats[t:t + self.L1_KERNEL] for t in range(T - self.L1_KERNEL + 1)])
        # einsum: (tw, k, f) × (o, f, k) → (tw, o)  对 (f, k) 求和
        conv = np.einsum("tfk,ofk->to", windows, self.l1_conv).astype(np.float32)
        pooled = conv.mean(axis=0)  # 时间全局平均池化 → (16,)
        out = np.tanh(pooled @ self.l1_fc + self.l1_bias).astype(np.float32)  # (64,)
        return normalize(out)

    def _l2_events(self, feats):
        """L2 事件流(128): 帧能量分箱 → 事件嵌入表聚合 → 128 维。"""
        if feats.shape[0] == 0:
            return np.zeros(self.cfg.audio_l2_dim, dtype=np.float32)
        energy = feats[:, 0]  # 第一维为对数能量
        lo, hi = float(energy.min()), float(energy.max())
        if hi - lo < 1e-9:
            idx = np.zeros(energy.shape[0], dtype=np.int64)
        else:
            idx = np.clip(((energy - lo) / (hi - lo) * (self.N_EVENTS - 1)).astype(np.int64),
                          0, self.N_EVENTS - 1)
        if len(idx) == 0:  # 规范 §3.1: 禁止对空列表索引
            return np.zeros(self.cfg.audio_l2_dim, dtype=np.float32)
        agg = self.event_table[idx].mean(axis=0).astype(np.float32)
        return normalize(agg)

    def _load_ort_session(self):
        """惰性加载 SenseVoiceSmall ONNX INT8（CPU）。"""
        try:
            ort = safe_import("onnxruntime")
            if ort is None:
                raise ImportError("onnxruntime 未安装")
            if not os.path.isfile(self.sensevoice_onnx_path):
                raise FileNotFoundError("模型不存在: %s" % self.sensevoice_onnx_path)
            self._ort_session = ort.InferenceSession(
                self.sensevoice_onnx_path, providers=["CPUExecutionProvider"])
            self.log.info("[L3] SenseVoiceSmall ONNX INT8 就绪 (CPU)")
        except Exception as e:
            self.log.warning("[L3] ONNX 加载失败: %s", e)
            self._ort_session = None

    def _l3_sensevoice(self, feats):
        """L3 人声特征(128): SenseVoiceSmall ONNX INT8；不可用 → 伪向量降级。

        规范 §3.1: ONNX 输出必须通过 .astype(np.float32) 显式转换为 float32。
        """
        try:
            if self._ort_session is None:
                self._load_ort_session()
            if self._ort_session is None:
                raise RuntimeError("onnxruntime 会话不可用")
            x = feats.astype(np.float32)
            input_name = self._ort_session.get_inputs()[0].name
            out = self._ort_session.run(None, {input_name: x[None, ...]})
            # 必须显式 .astype(np.float32)（规范 §3.1）
            vec = np.asarray(out[0]).reshape(-1).astype(np.float32)
            if vec.size >= self.cfg.audio_l3_dim:
                return normalize(vec[:self.cfg.audio_l3_dim])
            pad = np.zeros(self.cfg.audio_l3_dim, dtype=np.float32)
            pad[:vec.size] = vec
            return normalize(pad)
        except Exception as e:
            self.log.warning("[L3] SenseVoiceSmall 不可用(%s)，降级伪人声特征", e)
            # 用 common_utils.pseudo_embedding 填充噪声（规范 §九.2 降级边界）
            base = pseudo_embedding("sensevoice_fallback", self.cfg.audio_l3_dim, seed=7)
            noise = self.rng.normal(0.0, 0.05, self.cfg.audio_l3_dim).astype(np.float32)
            return normalize(base + noise)

    def _l4_gru(self, l2, l3, feats):
        """L4 动态时序(192): numpy GRU 编码。

        序列 = [帧级特征投影(64) ‖ L2 广播(128) ‖ L3 广播(128)] → (T, 320)
        取最后时刻隐藏状态 → 192 维。
        """
        T = feats.shape[0]
        if T == 0:
            return np.zeros(self.cfg.audio_l4_dim, dtype=np.float32)
        proj = feats @ self.frame_proj  # (T, 64)
        seq = np.concatenate([
            proj,
            np.tile(l2[None, :], (T, 1)),
            np.tile(l3[None, :], (T, 1)),
        ], axis=1).astype(np.float32)
        h = np.zeros(self.gru_hidden, dtype=np.float32)
        g = self.gru
        for t in range(T):
            x = seq[t]
            z = self._sigmoid(g["Wz"] @ x + g["Uz"] @ h + g["bz"])
            r = self._sigmoid(g["Wr"] @ x + g["Ur"] @ h + g["br"])
            hh = np.tanh(g["Wh"] @ x + g["Uh"] @ (r * h) + g["bh"])
            h = (1.0 - z) * h + z * hh
        return normalize(h.astype(np.float32))

    @staticmethod
    def _sigmoid(x):
        """数值稳定 sigmoid。"""
        x = np.asarray(x, dtype=np.float32)
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


if __name__ == "__main__":
    # 轻量自检（不参与正式运行流程）
    ap = AudioPerception()
    sr = ap.SAMPLE_RATE
    tt = np.arange(sr) / sr
    wave = (0.3 * np.sin(2 * np.pi * 220.0 * tt)
            + 0.1 * np.random.RandomState(0).normal(0.0, 1.0, sr)).astype(np.float32)
    v = ap.process(waveform=wave)
    print("[SELF] 语音波形 →", v.shape, "范数=", round(float(np.linalg.norm(v)), 3))
    z = ap.process()
    print("[SELF] 无输入(DEMO) →", z.shape, "范数=", round(float(np.linalg.norm(z)), 3))
