@echo off
rem ============================================================
rem start.bat - Xilian AGI v7.3 local-mode launcher (default)
rem Delegates to scripts\start.py -> start.py (local GUI)
rem ============================================================
setlocal
chcp 65001 >nul 2>&1
cd /d "%~dp0"

set "PY=%~dp0..\Python\python.exe"
if not exist "%PY%" set "PY=python"

echo ============================================
echo   Xilian AGI v7.3 - local mode (GUI)
echo ============================================
"%PY%" "%~dp0start.py" %*
if errorlevel 1 (
    echo.
    echo [ERROR] startup failed, see log above.
    pause
)
endlocal
