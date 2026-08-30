#!/usr/bin/env bash
# ============================================================
# 昔涟AGI · 一键训练（Linux/macOS）
# 用法: ./train.sh [l3|all]
# ============================================================
set -e
cd "$(dirname "$0")/.."
PY=python3
MODE=${1:-l3}

echo "============================================"
echo "  昔涟AGI · 一键训练（mode=$MODE）"
echo "============================================"

$PY -c "import torch" 2>/dev/null || echo "[警告] torch 缺失，训练将受限"

if [ "$MODE" = "all" ]; then
    echo "[训练] 阶段1: L3/L2 扩散器（知识库 + 训练资料）..."
    $PY train_l3_knowledge.py
    echo "[训练] 阶段2: 40M 情感模型..."
    $PY train_emotion_model.py
else
    $PY train_l3_knowledge.py
fi

echo "[完成] 模型已保存至 models/"
