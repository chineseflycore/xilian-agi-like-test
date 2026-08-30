@echo off
rem ============================================================
rem start_local.bat - Xilian AGI v7.3 LOCAL mode (explicit)
rem python start.py  -> local GUI, full cognition, no planner
rem ============================================================
setlocal
chcp 65001 >nul 2>&1
cd /d "%~dp0"

set "PY=%~dp0..\Python\python.exe"
if not exist "%PY%" set "PY=python"

echo ============================================
echo   Xilian AGI v7.3 - LOCAL mode (GUI)
echo   [no task planner / no API server]
echo ============================================
"%PY%" "%~dp0start.py" %*
if errorlevel 1 (
    echo.
    echo [ERROR] startup failed, see log above.
    pause
)
endlocal
