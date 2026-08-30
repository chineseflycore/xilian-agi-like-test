# ============================================================
# start.ps1 — 昔涟人格引擎 启动脚本（PowerShell 版）
# 用法:  powershell -ExecutionPolicy Bypass -File .\start.ps1
# 也可直接:  .\start.bat（双击）
# ============================================================
$ErrorActionPreference = "Stop"
$Base = Split-Path -Parent $MyInvocation.MyCommand.Path
$PY = Join-Path $Base "Python\python.exe"
$ModelDir = Join-Path $Base "models\Qwen3.5-0.8B"

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host "   昔涟人格引擎（PhiLia093）启动器" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# ---- 1) 便携版 Python ----
if (-not (Test-Path $PY)) {
    Write-Host "[错误] 未找到便携版 Python 3.11.9: $PY" -ForegroundColor Red
    exit 1
}
Write-Host "[OK] Python: $PY" -ForegroundColor Green

# ---- 2) torch + CUDA ----
try {
    & $PY -c "import torch; assert torch.cuda.is_available()" 2>$null
    if ($LASTEXITCODE -ne 0) { throw "cuda not available" }
    $TorchV = & $PY -c "import torch; print(torch.__version__)" 2>$null
    Write-Host "[OK] torch: $TorchV (CUDA 可用)" -ForegroundColor Green
} catch {
    Write-Host "[警告] torch 未安装或 CUDA 不可用，请先执行:" -ForegroundColor Yellow
    Write-Host "  & `"$PY`" -m pip install torch==2.0.1+cu117 --index-url https://download.pytorch.org/whl/cu117" -ForegroundColor Yellow
    Write-Host "  & `"$PY`" -m pip install transformers modelscope -i https://mirrors.aliyun.com/pypi/simple/" -ForegroundColor Yellow
    exit 1
}

# ---- 3) 模型 ----
if (-not (Test-Path (Join-Path $ModelDir "config.json"))) {
    Write-Host "[警告] 模型未下载（models\Qwen3.5-0.8B\config.json 不存在），请先执行:" -ForegroundColor Yellow
    Write-Host "  & `"$PY`" -c `"from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3.5-0.8B', local_dir=r'$ModelDir')`"" -ForegroundColor Yellow
    exit 1
}
Write-Host "[OK] 模型: $ModelDir" -ForegroundColor Green

# ---- 4) L3 规模（默认 1.0 = 100 万神经元；内存紧张可设 0.1）----
if (-not $env:PHILIA_L3_SCALE) { $env:PHILIA_L3_SCALE = "1.0" }
Write-Host "[OK] L3 规模: $env:PHILIA_L3_SCALE (PHILIA_L3_SCALE 可调: 0.1=10万 / 1.0=100万神经元)" -ForegroundColor Green

Write-Host ""
Write-Host "[启动] 引擎启动中... 首次加载 Qwen3.5-0.8B 约需 1-2 分钟" -ForegroundColor Cyan
Write-Host "[访问] http://127.0.0.1:8080  (Ctrl+C 停止)" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

Push-Location $Base
try {
    & $PY "main.py"
} finally {
    Pop-Location
}
