@echo off
chcp 65001 >nul
title 昔涟人格训练数据生成
cd /d "%~dp0"

set "PY=%~dp0Python\python.exe"
set "COUNT=200"

if not "%1"=="" set "COUNT=%1"

echo.
echo ============================================
echo   昔涟人格训练数据生成（前台，逐条可见）
echo   条数: %COUNT%   （用法: gen_persona_data.bat [条数]）
echo ============================================
echo.

"%PY%" "%~dp0gen_persona_data.py" --count %COUNT%

echo.
pause
