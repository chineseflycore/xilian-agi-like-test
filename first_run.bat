@echo off
chcp 65001 >nul
title 昔涟人格引擎 - 首次配置向导
cd /d "%~dp0"

set "PY=%~dp0Python\python.exe"

echo.
echo ============================================
echo   昔涟人格引擎 - 首次预启动配置向导
echo ============================================
echo.

REM ---- 1) 运行检查脚本 ----
if not exist "%PY%" (
    echo [错误] 未找到便携版 Python: "%PY%"
    echo        请将 Python 3.11.9 embeddable 版解压到项目 Python\ 目录。
    pause
    exit /b 1
)
"%PY%" "%~dp0first_run.py"
set "CHECK_EXIT=%ERRORLEVEL%"

REM ---- 2) 若 API Key 未配置，打开 hoyotool.ini 供填写 ----
"%PY%" -c "import configparser; c=configparser.ConfigParser(); c.read(r'hoyotool.ini', encoding='utf-8'); raise SystemExit(0 if c.get('Cloud','api_key',fallback='').strip() else 1)" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [提示] 未检测到 API Key。正在打开 hoyotool.ini 供您填写...
    echo        请在 [Cloud] 段的 api_key = 后面粘贴您的 DeepSeek Key，
    echo        保存后重新运行本向导确认，或直接运行 start.bat 启动引擎。
    echo.
    notepad "%~dp0hoyotool.ini"
)

echo.
if "%CHECK_EXIT%"=="0" (
    echo 检查通过，可运行 start.bat 启动引擎。
) else (
    echo 请按上方提示修复缺项后，重新运行本向导。
)
pause
