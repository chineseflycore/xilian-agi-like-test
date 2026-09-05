# xilian-agi-like-test

以《崩坏：星穹铁道》·昔涟为人格内核的低资源 AGI 项目（开源/可复现实验版）。

> **⚠️ 该项目由 AI 生成，作者只能保证在 1066 移动版 + i5-6300HQ + 16GB DDR4 2133 环境成功运行。**
>
> 本仓库为**非官方粉丝项目**，与米哈游（MiHoYo）无关；角色形象、故事与语料版权归米哈游所有。

## 项目简介

本项目是一个面向 **GTX 1060 6GB / 16GB RAM** 的低资源人格引擎实验：
以「昔涟」为人格内核，通过 **4000 路匹配器集群 + 动态激活 + 记忆池 + 扩散器** 实现轻量级对话与认知连续性，并支持 **Cyrene-Agent 兼容 API**（OpenAI 格式 `/v1/chat/completions`）。

> **v8.0 核心**：接入 **MiniMind-O 前端编码器**（Thinker 768 维潜向量 Z）+ **轻量投影层**（置信度回退），将 Z 维度从 384 对齐到 768，并与 `core/projector.py`、`model_minimind.py`、`minimind_bridge.py` 联动。

- 代码：`core/`（认知核心）、`gui/`（聊天界面）、`scripts/`（启动/训练入口）
- 配置：`config.py`（全局常量）、`hoyotool.ini`（示例配置，密钥留空）
- 依赖：见 `requirements.txt`

## 快速开始

```bat
:: 1) 安装依赖（torch / transformers / bitsandbytes 等）
pip install -r requirements.txt

:: 2) 配置 API Key（可选，仅训练数据生成 / 云端路由需要）
::    Windows:  set DEEPSEEK_API_KEY=<your-api-key>   （PowerShell: $env:DEEPSEEK_API_KEY="<your-api-key>"）
::    或者在 hoyotool.ini [Cloud] 段填 api_key（不建议，容易被误提交）

:: 3) 启动（本地 GUI 模式 / API 模式）
python scripts\start.bat          :: 本地模式
python scripts\start_api.bat      :: API 模式（端口 8080）

:: 4) 自检
python main.py --selftest
```

> 未配置 API Key 时程序自动降级（跳过云端调用，使用内置种子示例），不影响引擎启动与本地对话。
> 有必要可以配置python虚拟环境。
> 50系显卡可能需要修改部分代码。

## 目录结构

```
xilian-agi-like-test/
├── core/                 # 认知核心（匹配器集群 / 记忆池 / 扩散器 / 路由）
│   ├── agi_core.py       # 引擎主线
│   ├── matcher_cluster.py
│   ├── memory_pool.py
│   ├── diffuser.py
│   ├── projector.py      # v8.0 轻量投影层 + 置信度回退
│   ├── model_minimind.py # v8.0 MiniMind Thinker（hidden=768）
│   ├── minimind_bridge.py# v8.0 MiniMind-O 编码桥（Thinker hidden → 768 Z）
│   └── adapter/          # OpenAI 兼容 API / SSE
├── gui/                  # 聊天窗口 / Web UI
├── scripts/              # 启动 / 训练入口
├── config.py             # 全局配置（密钥走环境变量）
├── hoyotool.ini          # 示例配置（api_key 留空）
├── requirements.txt
└── README.md
```

## 角色语录 / 人设数据怎么生成

本项目的「昔涟」语料与人格数据，由以下脚本链路生成（从真实资料抓取 → 大模型生成 → 微调 → 知识源）：

### 1. 抓取参考人设（`fetch_cyrene_ref.py`）

从 GitHub `HeartEase1/cyrene.skill` 抓取人设资料（`SKILL.md` / `personality.md` / `profile.md` / `background_story.md` / `interaction.md` 等），输出到 `data_cache/reference_cyrene/`。

```bat
python fetch_cyrene_ref.py
```

> 引用第三方仓库资料仅作参考，版权归原作者与米哈游所有。

### 2. 生成人格训练数据（`gen_persona_data.py`）

用 DeepSeek API（`deepseek-v4-flash`，Responses API + `web_search` 联网搜索）生成 `N` 条昔涟人格对话：

```bat
:: 先生成 20 条试跑，再生成 200 条
python gen_persona_data.py --count 20
python gen_persona_data.py --count 200
```

流程：
1. `fetch_world_knowledge()` 联网抓取萌娘百科「昔涟 / 翁法罗斯」资料（可缓存）；
2. 资料 → 话题 → 场景提示；
3. 第一轮强制联网搜索，第二轮基于搜索结果作答（两轮自动续传）；
4. 落盘 `data_cache/persona_dataset.json`（完整不截断），支持缓存命中复用。

> 种子示例在 `train_all.py` 的 `SEED_EXAMPLES` 中可自行修改；修改后缓存自动失效。

### 3. 生成 ChatML 语料（`train_all.py` → `data_cache/训练资料*.txt`）

`train_all.py` 将生成数据整理为 ChatML 格式（`<|im_start|>user ... <|im_end|> <|im_start|>assistant ...`）的训练语料，供微调使用。

```bat
python train_all.py
```

### 4. LoRA 微调（`train_finetune_qwen.py`）

在 **GTX 1060 6GB** 预算内对 `Qwen3.5-0.8B` 做人设 SFT（LoRA r=16、冻结基础模型、AMP fp16、1~3 epoch）：

```bat
python train_finetune_qwen.py
:: 可选: PHILIA_FT_EPOCHS=2  PHILIA_FT_BATCH=4
```

产物：`models/Qwen3.5-0.8B-ft/`（合并后的微调模型，本地推理优先加载）。

### 5. 知识源训练（`train_l3_knowledge.py` / `train_all.py`）

知识层（L3）使用抓取到的资料训练知识源，供扩散器 / 检索使用。详细参数见各脚本 docstring。

---

### 💡 提醒（版权边界）

- 抓取的萌娘百科 / 第三方仓库资料**仅供参考研究**，请勿将完整语料直接发布；
- 角色「昔涟」及其故事版权归 **米哈游（MiHoYo）** 所有；
- 本项目代码以 MIT 许可发布，**不包括**任何角色原作文本、图片与官方语料。

## 环境变量

| 变量 | 说明 |
|---|---|
| `DEEPSEEK_API_KEY` | DeepSeek API Key（训练数据生成 / 云端路由） |
| `XILIAN_ENCODER` | 输入编码后端（默认 `minimind_o`，可选 `qwen`=v7.3 旧桥 / `rule`=规则编码） |
| `XILIAN_LLM` | 本地 LLM 桥开关（默认 1，可选 0） |
| `XILIAN_DECODER` | 解码器规模（`8b` / `2b`） |
| `PHILIA_DEMO` | 全流程演示模式（纯 CPU 验证） |
| `PHILIA_L3_SCALE` | L3 扩散器规模缩放（低内存机） |

## License

本项目代码以 **MIT License** 发布（见 `LICENSE`）。角色版权归米哈游，请遵守版权边界。
