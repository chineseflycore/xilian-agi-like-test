@echo off
rem ============================================================
rem 昔涟AGI · 一键训练（Windows）
rem 基于配置文件自动训练 → 显示进度（轮次/Loss）→ 断点续训 → 自动保存
rem 用法: train.bat [l3|all]   l3=扩散器训练(默认)  all=扩散器+情感模型
rem ============================================================
chcp 65001 >nul
title 昔涟AGI - 训练
cd /d "%~dp0\.."

set "PY=%~dp0..\Python\python.exe"
set "MODE=%1"
if "%MODE%"=="" set "MODE=l3"

echo.
echo ============================================
echo   昔涟AGI · 一键训练（mode=%MODE%）
echo ============================================

rem ---- 环境 ----
if not exist "%PY%" ( echo [错误] 未找到便携版 Python & pause & exit /b 1 )
"%PY%" -c "import torch" >nul 2>&1 || ( echo [警告] torch 缺失，训练将受限 & pause )

rem ---- 断点续训说明 ----
echo [提示] L3 权重 models\l3_weights.npz 存在将作为起点续训；
echo        知识库 data_cache\knowledge_db.json 命中则跳过重复抓取。

rem ---- 执行 ----
if /i "%MODE%"=="all" (
    echo [训练] 阶段1: L3/L2 扩散器（知识库 + 训练资料）...
    "%PY%" train_l3_knowledge.py
    echo [训练] 阶段2: 40M 情感模型...
    "%PY%" train_emotion_model.py
) else (
    "%PY%" train_l3_knowledge.py
)

echo.
echo [完成] 模型已保存至 models\ （l3_weights.npz / l2_weights.pt）
pause
