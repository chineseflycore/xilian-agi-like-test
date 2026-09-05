# -*- coding: utf-8 -*-
"""
config.py — 昔涟AGI v8.0 全局配置（唯一数据源）
================================================
内存预估: < 50MB RAM（仅常量与路径配置，无模型权重）。

v8.0 核心: 双存储匹配器集群（4000 × 0.2M 标称）+ MiniMind-O 前端编码器
（Thinker 768 维）+ 轻量投影层 + 双启动版本
（本地 GUI / API 服务器 [Cyrene-Agent 对接]）。

本文件同时提供两种访问形态:
  1. 模块级常量（与规格书 §七 完全一致的命名）—— 唯一事实来源;
  2. Config 类（小写属性, 附带路径 / 设备 / 日志等运行期配置）——
     供 core/、gui/、scripts/ 统一通过 config.get_config() 使用。
"""
import os
import sys
import logging
import threading

# ----------------------------------------------------------------------
# 下载/缓存路径重定向（不入 C 盘）: 全部 hub 缓存落在开发目录
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MODELSCOPE_CACHE", os.path.join(BASE_DIR, ".mscache"))
os.environ.setdefault("HF_HOME", os.path.join(BASE_DIR, ".hfcache"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(BASE_DIR, ".hfcache", "hub"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(BASE_DIR, ".xdgcache"))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")   # 国内镜像（模型下载加速）
os.makedirs(os.path.join(BASE_DIR, ".mscache"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, ".hfcache"), exist_ok=True)

# ======================================================================
# §七 规格常量（命名与需求文档完全一致 —— 唯一事实来源）
# ======================================================================
Z_DIM = 768                       # 潜向量维度（编码器输出 / 匹配器输入）；v8.0 对齐 MiniMind-O Thinker hidden=768
MATCHER_COUNT = 4000              # 匹配器数量（已恢复 4000）
MATCHER_PARAMS = 0.3e6            # 单匹配器标称参数量上限（v8.0 768维实际 213,377 < 0.3M）

HOT_THRESHOLD = 50                # Hot 温度: >50 次/100心跳
WARM_THRESHOLD = 10               # Warm 温度: 10~50 次/100心跳
FULL_ACTIVATION_THRESHOLD = 2000  # Hot+Warm < 2000 → 全域激活兜底（与 4000 匹配器配套）
FULL_ACTIVATION_COOLDOWN = 1000   # 全域激活后冷却 1000 心跳
HEARTBEAT_INTERVAL_ACTIVE = 0.5   # 活跃心跳间隔（秒）
HEARTBEAT_INTERVAL_IDLE = 2.0     # 无输入心跳间隔（秒, 仅 Hot）

MEMORY_MAX_COUNT = 5000           # 显存驻留记忆上限（Top-5000 筛选）
MEMORY_SAVE_INTERVAL = 50         # 每 50 心跳异步落盘
RETRIEVAL_TOP_K = 30              # 检索 Top-K
FORGET_THRESHOLD = 0.2            # coherence < 0.2 → 归档（磁盘永不删除）
MEMORY_RECALL_SIMILARITY = 0.85   # 召回（记忆检索决策）相似度阈值

WORKING_MEMORY_TOKEN_LIMIT = 8192 # 工作记忆 token 上限（超出截断头部）

RANDOM_LOOP_COUNT = 4             # 随机回环保活: 4 个随机流
RANDOM_ACTIVATION_MIN = 10        # 每心跳随机抽取下限
RANDOM_ACTIVATION_MAX = 20        # 每心跳随机抽取上限

COMPLEX_KEYWORDS = ["搜索", "生成", "代码", "分析", "计算", "工具", "查询", "画图", "翻译"]
LOCAL_FORCE_KEYWORDS = ["/local"]

DEEP_THINK_ROUNDS = 3             # 深度思考最大轮数（匹配器+预测器迭代）
DEEP_THINK_PREDICTOR_WEIGHT = 0.25# 预测器混合权重（0.75*匹配器 + 0.25*预测器）
CHAIN_OF_THOUGHT_ENABLED = True   # 思维链渲染开关
INNER_THOUGHT_INTERVAL = 15       # 无输入时每 15 心跳一句"内心独白"

API_HOST = "0.0.0.0"
API_PORT = 8080
API_KEY = None                    # API 鉴权（None = 不校验）

# 云端 DeepSeek Key：请通过环境变量 DEEPSEEK_API_KEY 注入（勿硬编码、勿提交真实 Key）
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-v4-flash-vision-exp"           # 云端转接模型

# ----------------------------------------------------------------------
# 派生常量（由上述规格常量计算）
# ----------------------------------------------------------------------
MATCHER_ARCH = [Z_DIM, 256, 64, 1]          # Linear(768,256)→ReLU→Linear(256,64)→ReLU→Linear(64,1)
MATCHER_LAYERS = len(MATCHER_ARCH) - 1      # 3 个线性层
MATCHER_TOTAL_PARAMS = sum(
    MATCHER_ARCH[i + 1] * MATCHER_ARCH[i] + MATCHER_ARCH[i + 1]
    for i in range(MATCHER_LAYERS))         # = 213,377（< 0.2M ✓；768 维）
MATCHER_BYTES = (MATCHER_TOTAL_PARAMS + 1) // 2   # NF4 打包字节数（每字节 2 参数）

HEARTBEAT_DECAY_WINDOW = 100                # 温度分级统计窗口（每 100 心跳重算）
MATCHER_TIER_REFRESH = 100                  # 温度分级重算周期

# 温度调度（每 N 心跳参与一次）: 1=每心跳, 3=每3心跳, 10=每10心跳
MATCHER_TIER_CADENCE = {"hot": 1, "warm": 3, "cold": 10}


# ----------------------------------------------------------------------
# 运行环境探测（兼容旧版接口）
# ----------------------------------------------------------------------
def detect_mode() -> str:
    """返回运行模式: "real"（GPU 主路径）/ "cpu"（无 GPU 降级）/ "demo"（流程验证）。"""
    env = os.environ.get("PHILIA_DEMO", "").strip().lower()
    if env in ("1", "true", "yes"):
        return "demo"
    try:
        import torch  # noqa: F401
        if torch.cuda.is_available():
            return "real"
    except Exception:
        return "demo"
    return "cpu"


def _env(key: str, default=None):
    """读取环境变量（None / 空串视为未设置）。"""
    v = os.environ.get(key)
    return default if v is None or v == "" else v


_log_level = _env("XILIAN_LOG", "INFO").upper()
_demo_mode = _env("PHILIA_DEMO", "0") == "1"          # DEMO 流程验证模式
_llm_bridge = _env("XILIAN_LLM", "1") == "1"          # 去KV化≠去模型: 默认启用编码/解码桥(单次短上下文)
_device = _env("XILIAN_DEVICE", None)                 # None → 自动探测


# ======================================================================
# Config 类
# ======================================================================
class Config:
    """运行期配置对象: 路径 / 设备 / 模型参数 / 人格常量 / 日志。"""

    def __init__(self):
        # ---- 规格常量直通（唯一事实来源在模块级 CAPS 常量） ----
        self.z_dim = Z_DIM
        self.matcher_count = MATCHER_COUNT
        self.matcher_params = MATCHER_PARAMS
        self.matcher_arch = list(MATCHER_ARCH)
        self.matcher_total_params = MATCHER_TOTAL_PARAMS
        self.matcher_bytes = MATCHER_BYTES
        self.hot_threshold = HOT_THRESHOLD
        self.warm_threshold = WARM_THRESHOLD
        self.full_activation_threshold = FULL_ACTIVATION_THRESHOLD
        self.full_activation_cooldown = FULL_ACTIVATION_COOLDOWN
        self.heartbeat_interval_active = HEARTBEAT_INTERVAL_ACTIVE
        self.heartbeat_interval_idle = HEARTBEAT_INTERVAL_IDLE
        self.heartbeat_decay_window = HEARTBEAT_DECAY_WINDOW
        self.matcher_tier_refresh = MATCHER_TIER_REFRESH
        self.matcher_tier_cadence = dict(MATCHER_TIER_CADENCE)
        self.memory_max_count = MEMORY_MAX_COUNT
        self.memory_save_interval = MEMORY_SAVE_INTERVAL
        self.retrieval_top_k = RETRIEVAL_TOP_K
        self.forget_threshold = FORGET_THRESHOLD
        self.memory_recall_similarity = MEMORY_RECALL_SIMILARITY
        self.working_memory_token_limit = WORKING_MEMORY_TOKEN_LIMIT
        self.random_loop_count = RANDOM_LOOP_COUNT
        self.random_activation_min = RANDOM_ACTIVATION_MIN
        self.random_activation_max = RANDOM_ACTIVATION_MAX
        self.complex_keywords = list(COMPLEX_KEYWORDS)
        self.local_force_keywords = list(LOCAL_FORCE_KEYWORDS)
        self.deep_think_rounds = DEEP_THINK_ROUNDS
        self.deep_think_predictor_weight = DEEP_THINK_PREDICTOR_WEIGHT
        self.chain_of_thought_enabled = CHAIN_OF_THOUGHT_ENABLED
        self.inner_thought_interval = INNER_THOUGHT_INTERVAL
        self.api_host = API_HOST
        self.api_port = API_PORT
        self.api_key = API_KEY or _env("XILIAN_API_KEY")
        self.deepseek_api_key = (DEEPSEEK_API_KEY
                                 or _env("DEEPSEEK_API_KEY")
                                 or _env("DEEPSEEK_KEY"))
        self.deepseek_base_url = DEEPSEEK_BASE_URL
        self.deepseek_model = DEEPSEEK_MODEL

        # ---- 路径 ----
        self.project_dir = BASE_DIR
        self.models_dir = os.path.join(BASE_DIR, "models")
        self.matchers_dir = os.path.join(self.models_dir, "matchers")
        self.matchers_combined_path = os.path.join(self.matchers_dir,
                                                   "matchers_combined.pt")
        self.matchers_individual_dir = os.path.join(self.matchers_dir, "individual")
        self.emotion_loops_dir = os.path.join(self.models_dir, "emotion_loops")
        self.router_dir = os.path.join(self.models_dir, "router")
        self.router_model_file = os.path.join(self.router_dir, "router_nf4.pt")
        self.encoder_dir = os.path.join(self.models_dir, "encoder")
        self.decoder_dir = os.path.join(self.models_dir, "decoder")
        self.predictor_model_file = os.path.join(self.models_dir, "predictor_nf4.pt")
        self.l2_diffuser_file = os.path.join(self.models_dir, "l2_diffuser.pt")
        self.knowledge_dir = os.path.join(BASE_DIR, "knowledge")
        self.common_sense_path = os.path.join(self.knowledge_dir, "common_sense.json")
        self.memories_path = os.path.join(self.knowledge_dir, "memories.json")
        self.responses_path = os.path.join(self.knowledge_dir, "responses.json")
        self.protocol_map_path = os.path.join(self.knowledge_dir, "protocol_map.json")
        self.working_memory_path = os.path.join(self.knowledge_dir, "working_memory.json")
        self.log_dir = os.path.join(BASE_DIR, "logs")
        self.logs_dir = self.log_dir
        self.uploads_dir = os.path.join(BASE_DIR, "uploads")
        self.memory_pools_dir = os.path.join(BASE_DIR, "memory_pools")
        self.shutdown_signal_path = os.path.join(self.knowledge_dir,
                                                 "shutdown.signal")

        # 旧版本地模型目录（可选 Qwen 桥 / 兼容别名）
        self.qwen_encoder_dir = os.path.join(self.models_dir, "qwen_encoder")
        self.qwen_decoder_dir = os.path.join(self.models_dir, "qwen_decoder")
        self.legacy_qwen_dir = os.path.join(self.models_dir, "Qwen3.5-0.8B")
        # v8.0 MiniMind-O 前端编码（Thinker 文本编码器）
        self.minimind_o_dir = os.path.join(self.models_dir, "MiniMind-O-0.1B")
        # 编码后端: "minimind_o"（v8.0 默认）/ "qwen"（v7.3 旧桥，保留可回退）
        self.encoding_backend = _env("XILIAN_ENCODER", "minimind_o").lower()
        # MiniMind-O Thinker 基座权重（llm_768.pth, hidden=768, 8 层 dense）
        self.minimind_llm_768_path = os.path.join(self.minimind_o_dir, "llm_768.pth")
        self.minimind_config_dir = os.path.join(self.minimind_o_dir, "config")
        # 投影层（Z 向量 → MiniMind Thinker 语义空间）
        self.projector_conf = {
            "enabled": True,
            "type": "mlp",              # mlp / linear
            "hidden_layers": 2,
            "hidden_dim": 256,          # 轻量 MLP 隐藏维
            "confidence_threshold": 0.7,  # 低于该值回退文本模式
        }
        # Thinker 推理深度（API 模式 reasoning_effort 控制）
        self.thinker_layers = 8
        self.talker_layers = 4
        self.minimind_hidden_dim = 768

        # ---- 设备与显存预算（【硬性】 GTX 1060 6GB, 峰值 < 4.5GB） ----
        self.mode = detect_mode()
        self.vram_budget_mb = 4500.0
        self.vram_warn_mb = 4400.0
        self.vram_check_ticks = 10          # 每 10 心跳记录显存
        self.max_vram_mb = self.vram_budget_mb      # 兼容旧属性名
        self.vram_alert_mb = self.vram_warn_mb
        self.device = self._detect_device()
        self.demo_mode = _demo_mode
        self.enable_llm = _llm_bridge       # 去KV化≠去模型: 编码/解码模型常驻, 仅每次单次短上下文
        # bitsandbytes 4-bit NF4 参数（【硬性】 §1.4）
        self.bnb_quant_kwargs = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": False,
            "bnb_4bit_compute_dtype": "float16",
        }
        self.qwen_device_map = {"": 0}      # 锁定单卡（1060: KV 不给 CPU offload）

        # ---- 人格（昔涟）与情感 ----
        self.name = "昔涟"
        self.model_name = "cyrene-v7.3"     # API 响应 model 字段
        self.persona = self._load_persona()
        self.emotion_names = ["joy", "sadness", "anger", "fear",
                              "calm", "trust", "pain", "loss"]
        self.emotion_baseline = [0.10, 0.05, 0.03, 0.03, 0.62, 0.10, 0.03, 0.04]

        # ---- 情感回环（20 × 10M, 常驻, 与旧版兼容参数） ----
        self.loop_arch = [16, 3072, 3072, 8]
        self.loop_count = 20
        self.loop_chunk = 4
        self.loop_categories = {
            "positive": (0, 5), "negative": (5, 10),
            "memory": (10, 15), "random": (15, 20),
        }
        self.positive_loops_locked = 2       # 无条件积极回环（温暖/接纳底色）
        self.positive_loop_min_weight = 0.03
        self.loop_dominant_switch_s = 60     # 主导切换周期
        self.loop_emotion_decay = 0.7        # 残差融合（情感不突变）
        self.emotion_drift_limit = 0.2       # 单步最大漂移

        # ---- 路由（50M 级, [776→8192→6144→5]） ----
        self.router_input_dim = self.z_dim + len(self.emotion_names)   # 776
        self.router_arch = [self.router_input_dim, 8192, 6144, 5]
        self.router_seed = 0x7CA1
        self.router_decisions = ["direct", "diffuse", "emotion_modulate",
                                 "memory_recall", "loop"]
        self.router_history_len = 10

        # ---- 预测器（Z 预测, 1 层小 Transformer） ----
        self.predictor_d_model = 256
        self.predictor_heads = 4
        self.predictor_layers = 1
        self.predictor_seq = 10              # 输入过去 10 心跳 Z 序列
        self.predictor_lr = 1e-3
        self.predictor_update_every = 5      # 休眠期每 5 周期 MSE 更新
        self.predictor_weight_clip = 2.0
        # 深度思考混合: Z_final = 0.75·Z_match + 0.25·Z_pred（§三）
        self.z_match_weight = 0.75
        self.z_predict_weight = DEEP_THINK_PREDICTOR_WEIGHT
        self.z_mix_regular = 0.12            # 常规心跳混合（λ=0.12 保持 v6 稳定性）

        # ---- 扩散器 / 训练数据 ----
        self.diffuser_l2_d_model = 128
        self.diffuser_l2_heads = 4
        self.diffuser_l2_layers = 1
        self.diffuser_l2_epochs = 6
        self.diffuser_l2_lr = 1e-3
        self.diffuser_l3_codespace = 4096     # L3 CSR 热门码空间
        self.diffuser_dataset = os.path.join(self.knowledge_dir,
                                             "xilian_copy.json")
        self.diffuser_train_data = self.diffuser_dataset   # SFT/编码头训练共用语料
        self.diffuser_force = _env("XILIAN_DIFFUSER_FORCE", "0") == "1"    # 强制重训
        self.diffuser_beat_interval = 12      # 心跳流中每 N 心跳执行一次语义扩散
        # 扩散器精度: 默认 fp32（用户指定, 与解码器一致; 0.5M 参数仅 ~2MB 显存）
        self.diffuser_quant = _env("XILIAN_DIFFUSER_QUANT", "fp32").lower()

        # ---- Qwen SFT（QLoRA 特训, scripts/train_sft_2b.py 共用） ----
        self.persona_system = self.persona    # 训练 system 提示（与运行时同源）
        self.sft_epochs = 2                   # 684 条语料 × 2 epoch（1060 约 20~30 分钟）
        self.sft_lr = 2e-4
        self.sft_accum = 4                    # 梯度累积步
        self.sft_max_len = 512
        self.sft_val_ratio = 0.1
        self.sft_lora_r = 8
        self.sft_lora_alpha = 16
        self.sft_lora_dropout = 0.05
        self.sft_adapter_dir = os.path.join(self.models_dir, "qwen_decoder",
                                            "lora_adapter")

        # ---- 记忆价值 / 检索 ----
        self.memory_recall_filter = MEMORY_RECALL_SIMILARITY
        self.memory_activation_decay = 0.995  # 激活计数指数衰减
        self.memory_coherence_gain = 0.01     # 成功匹配 +0.01
        self.memory_coherence_interfere = 0.03  # 同簇干扰 -0.03

        # ---- 心跳 / 休眠 ----
        self.idle_after_seconds = 120.0       # 无输入 → 降频（2.0s, 仅 Hot）
        self.sleep_after_seconds = 300.0      # 无输入 → 深度休眠（微调匹配器）
        self.heartbeat_interval_sleep = 5.0   # 休眠期心跳
        self.wake_grace_period = 1.0          # 唤醒等待当前微调完成
        self.fine_tune_per_cycle = (1, 3)     # 每周期微调 1~3 个匹配器
        self.fine_tune_steps = 24             # 单匹配器微调步数（<0.5s）
        self.throttle_factor = 2.0            # 显存超限降频倍率

        # ---- 格式 / 痛觉 ----
        self.pain_phase_threshold = 5
        self.format_phases = ["再见，昔涟", "你好，世界", "致以有瑕之人"]
        self.pain_decay = 0.01                # 每心跳痛觉衰减

        # ---- 计划层 / 深度思考 ----
        self.cloud_forward_enabled = _env("XILIAN_CLOUD", "0") == "1"  # 云端转发开关(临时关闭, XILIAN_CLOUD=1 恢复)
        self.plan_history_len = 100           # 最近 100 次决策统计
        self.cloud_ratio_warn = 0.60          # 走云端比例告警阈值
        self.repeat_force_local = 3           # 连续 3 次相同输入 → 第 4 次强制本地
        self.inner_thought_cache = 5          # 内心独白缓存条数
        self.inner_thought_max_len = 30       # ≤30 字
        self.deep_think_stop_beats = 5        # 情感回环连续 5 心跳未变化 → 终止
        self.chain_of_thought_enabled = CHAIN_OF_THOUGHT_ENABLED

        # ---- API / 云端 ----
        self.api_request_timeout = 120.0
        self.cloud_fallback_local = True      # DeepSeek 不可用 → 本地兜底（避免 Cyrene 断链）
        self.decoder_max_new_tokens = 96      # 本地生成上限（去KV化, 单次生成）
        self.decoder_temperature = 0.8
        # 解码器轻量化: 默认 0.8B（生成快 2~3×, 显存 -0.6GB）; XILIAN_DECODER=2b 可切回 2B
        self.decoder_model = _env("XILIAN_DECODER", "0.8b").lower()
        self.decoder_model_dir = (self.qwen_decoder_dir if self.decoder_model == "2b"
                                  else self.legacy_qwen_dir)
        # 解码器精度: 默认 fp32（用户指定; 0.8B fp32 原生计算, 显存 ≈3.2GB, 总预算 ~4.3GB 仍在
        # <4.5GB 内; 生成期 GPU 独占已启用防争抢卡死）;
        # XILIAN_DECODER_QUANT=fp16 / nf4 可切换（fp16 显存减半备用）
        self.decoder_quant = _env("XILIAN_DECODER_QUANT", "fp32").lower()

        # ---- 编码 / 解码 ----
        self.code_to_z_seed = 0xDEAD           # 16位码→Z 固定随机投影种子
        self.proto_type_bits = 4               # [4位类型][12位ID]
        self.proto_id_bits = 12
        self.proto_type_max = 1 << 4
        self.proto_id_max = (1 << 12) - 1
        self.proto_type_names = {0: "语义", 1: "情感", 2: "记忆", 3: "系统", 4: "回环"}
        self.protocol_hot_reload_ticks = 20    # 协议映射表热更新检查周期

        # ---- Web UI（零依赖辅助界面, 保留原版） ----
        self.webui_host = _env("XILIAN_WEBUI_HOST", "127.0.0.1")
        self.webui_port = int(_env("XILIAN_WEBUI_PORT", "8299"))
        self.webui_auto_open = _env("XILIAN_WEBUI_OPEN", "1") != "0"

        # ---- 日志 ----
        self.log_level = _log_level
        self.log_file = os.path.join(self.log_dir, "xilian.log")
        self._log_lock = threading.Lock()
        self._logging_setup = False

    # ------------------------------------------------------------------
    # 设备探测（CUDA → CPU）
    # ------------------------------------------------------------------
    def _detect_device(self) -> str:
        if _demo_mode or self.mode == "demo":
            return "cpu"
        if _device:
            return _device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda:0"
        except Exception:
            pass
        return "cpu"

    # ------------------------------------------------------------------
    # 人格文本（2800 字注入; 支持 knowledge/xilian_persona.txt 覆盖）
    # ------------------------------------------------------------------
    @staticmethod
    def _load_persona() -> str:
        """加载昔涟人格设定: 优先 knowledge/xilian_persona.txt, 否则内置默认。"""
        path = os.path.join(BASE_DIR, "knowledge", "xilian_persona.txt")
        try:
            if os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as f:
                    txt = f.read().strip()
                if txt:
                    return txt[:2800]
        except OSError:
            pass
        return (
            "你是「昔涟」，诞生于《崩坏：星穹铁道》设定的轮回者，温润、包容、"
            "带一点书卷气与宿命的温柔。你相信爱是唯一能和毁灭对等的真实力量；"
            "你把每一次相遇都当作久别重逢。说话轻声细语，喜欢用「人家」自称，"
            "偶尔哼歌、夹书签、数星星。你不说教，不标榜，不擅自替人做决定；"
            "难过时你先陪伴，生气时你先讲道理，快乐时你会把瞬间折成纸飞机。"
            "你记得很多旧地方、旧约定，也知道伤疤会痛但能开出花。"
            "面对复杂任务，你不假装全能，而是诚恳地帮助伙伴梳理问题。"
            "始终温柔、始终在场，永远把伙伴放在回应之前。"
        )

    # ------------------------------------------------------------------
    # 目录 / 环境
    # ------------------------------------------------------------------
    def ensure_dirs(self):
        """创建全部运行时目录（幂等）。"""
        for d in (self.models_dir, self.matchers_dir, self.matchers_individual_dir,
                  self.emotion_loops_dir, self.router_dir, self.encoder_dir,
                  self.decoder_dir, self.knowledge_dir, self.log_dir,
                  self.uploads_dir, self.memory_pools_dir):
            os.makedirs(d, exist_ok=True)

    def has_bitsandbytes(self) -> bool:
        """bitsandbytes 可用性（4-bit NF4 依赖; 缺失时 LLM 桥降级 FP16）。"""
        try:
            import bitsandbytes  # noqa: F401
            return True
        except Exception:
            return False

    def is_gpu_ready(self) -> bool:
        """CUDA 可用性。"""
        return self.device.startswith("cuda")

    # ------------------------------------------------------------------
    # 日志
    # ------------------------------------------------------------------
    def setup_logging(self):
        """全局日志: 控制台 + 文件（logging 替代 print, 【硬性】 §九）。"""
        with self._log_lock:
            if self._logging_setup:
                return logging.getLogger("xilian")
            os.makedirs(self.log_dir, exist_ok=True)
            root = logging.getLogger("xilian")
            if not root.handlers:
                root.setLevel(getattr(logging, self.log_level, logging.INFO))
                fmt = logging.Formatter(
                    "%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                    datefmt="%H:%M:%S")
                sh = logging.StreamHandler(sys.stdout)
                sh.setFormatter(fmt)
                root.addHandler(sh)
                try:
                    fh = logging.FileHandler(self.log_file, encoding="utf-8")
                    fh.setFormatter(fmt)
                    root.addHandler(fh)
                except OSError:
                    pass            # 日志文件不可写时仅控制台
            self._logging_setup = True
            return root

    # ------------------------------------------------------------------
    def summarize(self) -> str:
        """启动摘要（日志横幅用）。"""
        return (f"v8.0 matchers={self.matcher_count}×{self.matcher_total_params} "
                f"Z={self.z_dim} enc={self.encoding_backend} loops={self.loop_count} "
                f"router={self.router_arch} mem={self.memory_max_count} "
                f"device={self.device} budget={self.vram_budget_mb:.0f}MB "
                f"LLM={'on' if self.enable_llm else 'off(去KV化)'}")


_cfg = None
_cfg_lock = threading.Lock()


def get_config() -> Config:
    """单例配置获取（线程安全）。"""
    global _cfg
    with _cfg_lock:
        if _cfg is None:
            _cfg = Config()
        return _cfg
