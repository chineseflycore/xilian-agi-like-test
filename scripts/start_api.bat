@echo off
rem ============================================================
rem start_api.bat - Xilian AGI v7.3 API mode
rem python start.py --api -> task planner + HTTP server :8080
rem OpenAI-compatible endpoint for Cyrene-Agent:
rem   POST http://127.0.0.1:8080/v1/chat/completions
rem ============================================================
setlocal
chcp 65001 >nul 2>&1
cd /d "%~dp0"

set "PY=%~dp0..\Python\python.exe"
if not exist "%PY%" set "PY=python"

echo ============================================
echo   Xilian AGI v7.3 - API mode (task planner)
echo   endpoint: http://127.0.0.1:8080/v1/chat/completions
echo ============================================
"%PY%" "%~dp0start.py" --api %*
if errorlevel 1 (
    echo.
    echo [ERROR] startup failed, see log above.
    pause
)
endlocal
