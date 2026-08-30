# ============================================================
# setup_env.ps1 — 昔涟人格引擎 环境一键准备（便携版 Python 3.11.9）
# 用法:
#   powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
#
# 做什么:
#   1. 卸载旧 torch（若存在）
#   2. 安装 torch 2.6.0+cu126（阿里云直链，约 2.4GB，GTX 1060 驱动 582.53 支持 CUDA 12.6）
#   3. 升级 transformers 到最新（Qwen3.5-0.8B 加载需要；新版要求 torch>=2.5）
#   4. 安装 modelscope（模型下载用）
#   5. 验证 torch/CUDA/transformers
# ============================================================
$ErrorActionPreference = "Stop"
$Base = Split-Path -Parent $MyInvocation.MyCommand.Path
$PY = Join-Path $Base "Python\python.exe"

if (-not (Test-Path $PY)) {
    Write-Host "[错误] 未找到便携版 Python 3.11.9: $PY" -ForegroundColor Red
    exit 1
}

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "  昔涟人格引擎 - 环境一键准备" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
& $PY --version

# ---- 1) 卸载旧 torch（避免 CPU 版 / 旧版残留）----
Write-Host "`n[1/4] 卸载旧 torch ..." -ForegroundColor Cyan
& $PY -m pip uninstall -y torch 2>$null | Out-Null
Write-Host "      完成（无则跳过）"

# ---- 2) 安装 torch 2.6.0+cu126（阿里云直链，约 2.4GB，请耐心等待）----
Write-Host "`n[2/4] 安装 torch 2.6.0+cu126（约 2.4GB，阿里云直链）..." -ForegroundColor Cyan
& $PY -m pip install --disable-pip-version-check --timeout 60 --retries 5 `
    "https://mirrors.aliyun.com/pytorch-wheels/cu126/torch-2.6.0%2Bcu126-cp311-cp311-win_amd64.whl" `
    -i https://mirrors.aliyun.com/pypi/simple/
if ($LASTEXITCODE -ne 0) { Write-Host "[错误] torch 安装失败" -ForegroundColor Red; exit 1 }

# ---- 3) 升级 transformers + 安装 modelscope ----
Write-Host "`n[3/4] 升级 transformers + 安装 modelscope ..." -ForegroundColor Cyan
& $PY -m pip install --disable-pip-version-check --timeout 30 --retries 3 -U `
    transformers modelscope -i https://mirrors.aliyun.com/pypi/simple/
if ($LASTEXITCODE -ne 0) { Write-Host "[错误] transformers/modelscope 安装失败" -ForegroundColor Red; exit 1 }

# ---- 4) 验证 ----
Write-Host "`n[4/4] 验证 ..." -ForegroundColor Cyan
& $PY -c "import torch; print('torch      :', torch.__version__); print('cuda_available =', torch.cuda.is_available()); print('gpu        :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
& $PY -c "import transformers; print('transformers :', transformers.__version__)"
& $PY -c "import modelscope; print('modelscope : OK')"

Write-Host "`n=== 环境准备完成！===" -ForegroundColor Green
Write-Host "下一步下载模型（如未下载）:" -ForegroundColor Yellow
Write-Host "  & `"$PY`" -c `"from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3.5-0.8B', local_dir=r'$Base\models\Qwen3.5-0.8B')`"" -ForegroundColor Yellow
Write-Host "然后启动:  .\start.bat" -ForegroundColor Yellow
