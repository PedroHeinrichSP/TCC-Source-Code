param(
    [string]$Method = "nerf_static",
    [string]$Dataset = "blender_synthetic",
    [string]$Root = "./data/blender_synthetic/nerf_synthetic/lego",
    [string]$Preset = "quick"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "../../..")).Path
Set-Location $ProjectRoot
$env:PYTHONPATH = "$ProjectRoot\src"

# Venv detection: prefer active, fallback to .venv-mx330-311, then venv
if ($env:VIRTUAL_ENV) {
    Write-Host "Using active virtual environment: $env:VIRTUAL_ENV" -ForegroundColor Green
} elseif (Test-Path ".venv-mx330-311\Scripts\activate.ps1") {
    & ".venv-mx330-311\Scripts\Activate.ps1"
} elseif (Test-Path "venv\Scripts\activate.ps1") {
    & "venv\Scripts\Activate.ps1"
} else {
    Write-Host "ERROR: Virtual environment not found." -ForegroundColor Red
    Write-Host "Expected one of:" -ForegroundColor Red
    Write-Host "  - active shell venv (recommended)" -ForegroundColor Red
    Write-Host "  - ./.venv-mx330-311" -ForegroundColor Red
    Write-Host "  - ./venv" -ForegroundColor Red
    Write-Host "Activate/create one and try again." -ForegroundColor Red
    exit 1
}

$PythonExe = python -c "import sys; sys.stdout.write(sys.executable)" 2>$null
Write-Host "Python in use: $PythonExe" -ForegroundColor Green

Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "   Quick Benchmark (Single Method)" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Method:  $Method" -ForegroundColor Yellow
Write-Host "Dataset: $Dataset" -ForegroundColor Yellow
Write-Host "Scene:   $Root" -ForegroundColor Yellow
Write-Host ""

# CPU-only mode detection
if ($env:NVS_FORCE_CPU -eq "1") {
    $env:CUDA_VISIBLE_DEVICES = ""
    Write-Host "CPU-only mode enabled (NVS_FORCE_CPU=1)." -ForegroundColor Yellow
}

# Resolve dataset path
if (-not (Test-Path $Root)) {
    $AltRoot = ".\data\blender_synthetic\nerf_synthetic\lego"
    if (Test-Path $AltRoot) {
        $Root = $AltRoot
        Write-Host "Using alternate path: $Root" -ForegroundColor Yellow
    } else {
        Write-Host "Dataset not found at '$Root'. Run '.\scripts\windows\powershell\download_dataset.ps1' first." -ForegroundColor Red
        exit 1
    }
}

# torchsearchsorted is optional: D-NeRF falls back to torch.searchsorted.
$TorchSearchsortedSetup = Join-Path $ProjectRoot "third_party\d_nerf\torchsearchsorted\setup.py"
if (Test-Path $TorchSearchsortedSetup) {
    $importOk = $false
    try {
        & python -c "from torchsearchsorted import searchsorted" *> $null
        $importOk = ($LASTEXITCODE -eq 0)
    } catch {
        $importOk = $false
    }
    if (-not $importOk) {
        Write-Host "torchsearchsorted not available. Using torch.searchsorted fallback." -ForegroundColor Yellow
    }
}

Write-Host "Starting benchmark..." -ForegroundColor Green
Write-Host ""

$cmd = @(
    "python", "-m", "nvs_benchmark.cli", "method-run",
    "--method", $Method,
    "--dataset", $Dataset,
    "--root", $Root,
    "--split", "train",
    "--output-dir", "./artifacts",
    "--log-dir", "./logs",
    "--compute-metrics",
    "--snapshot-file", "./artifacts/metrics/latest.json",
    "--append-snapshot",
    "--preset", $Preset,
    "--adaptive-preset"
)

if ($env:NVS_EXTRA_JSON) {
    $cmd += @("--extra-json", $env:NVS_EXTRA_JSON)
}

& $cmd[0] $cmd[1..($cmd.Length-1)]

if ($LASTEXITCODE -ne 0) {
    Write-Host "Benchmark failed (exit=$LASTEXITCODE)" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "====================================================" -ForegroundColor Green
Write-Host "Benchmark complete!" -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Results saved to:" -ForegroundColor Cyan
Write-Host "  - Metrics:  ./artifacts/metrics/latest.json" -ForegroundColor Cyan
Write-Host "  - Logs:     ./logs/runs/" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  - View results:       .\scripts\windows\powershell\preview.ps1" -ForegroundColor Cyan
Write-Host "  - Generate report:    .\scripts\windows\powershell\generate_report.ps1" -ForegroundColor Cyan
Write-Host ""
