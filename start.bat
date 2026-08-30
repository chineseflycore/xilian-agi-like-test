@echo off
rem ============================================================
rem start.bat - Xilian AGI v7.3 launcher (LOCAL mode, default)
rem same as scripts\start_local.bat; use scripts\start_api.bat
rem for API mode (Cyrene-Agent endpoint on :8080)
rem ============================================================
setlocal
chcp 65001 >nul 2>&1
cd /d "%~dp0"

set "PY=%~dp0Python\python.exe"
if not exist "%PY%" set "PY=python"

echo ============================================
echo   Xilian AGI v7.3 - starting (local mode)
echo   (API mode: use scripts\start_api.bat)
echo ============================================
"%PY%" "%~dp0scripts\start.py" %*
if errorlevel 1 (
    echo.
    echo [ERROR] startup failed, see log above.
    pause
)
endlocal
