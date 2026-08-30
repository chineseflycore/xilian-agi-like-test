#!/usr/bin/env bash
# ============================================================
# 昔涟AGI · 一键启动（Linux/macOS）
# 环境检查 → 加载最新模型 → 启动 Web UI → 打开浏览器
# ============================================================
set -e
cd "$(dirname "$0")/.."
PY=python3
PORT=${PORT:-8080}

echo "============================================"
echo "  昔涟AGI · 一键启动"
echo "============================================"

# 1) Python
command -v "$PY" >/dev/null || { echo "[错误] 未找到 python3"; exit 1; }
echo "[OK] Python: $($PY --version 2>&1)"

# 2) torch + CUDA
if $PY -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "[OK] torch: $($PY -c 'import torch; print(torch.__version__)') (CUDA)"
else
    echo "[警告] CUDA 不可用，尝试 CPU 运行"
fi

# 3) 模型
if [ -f "models/Qwen3.5-0.8B-ft/config.json" ]; then
    echo "[OK] 模型: Qwen3.5-0.8B-ft（微调版）"
elif [ -f "models/Qwen3.5-0.8B/config.json" ]; then
    echo "[OK] 模型: Qwen3.5-0.8B（原版）"
else
    echo "[警告] 模型未下载"
fi

# 4) flask
$PY -c "import flask" 2>/dev/null || { echo "[提示] 安装 flask..."; $PY -m pip install flask -q; }

echo "[启动] http://127.0.0.1:$PORT  （/train 训练面板）"
( sleep 2; xdg-open "http://127.0.0.1:$PORT" 2>/dev/null || open "http://127.0.0.1:$PORT" 2>/dev/null || true ) &
exec $PY -m ui.webui --host 127.0.0.1 --port "$PORT"
