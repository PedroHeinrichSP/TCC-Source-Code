param(
    [string]$Method = "nerf_static",
    [string]$Dataset = "blender_synthetic",
    [string]$Root = "./data/blender_synthetic/nerf_synthetic/lego",
    [string]$ExtraFile = "",
    [switch]$ComputeMetrics
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..\..\..")).Path
Set-Location $ProjectRoot
$env:PYTHONPATH = "$ProjectRoot\src"

function Get-VenvPython {
    param([string]$RootPath)

    $VenvPython = Join-Path $RootPath "venv\Scripts\python.exe"
    if (-not (Test-Path $VenvPython)) {
        throw "Virtual environment not found at '$VenvPython'. Run './scripts/windows/powershell/setup.ps1' first."
    }
    return $VenvPython
}

$PythonExe = Get-VenvPython -RootPath $ProjectRoot

function Test-CudaAvailable {
    param([string]$PythonPath)
    try {
        $out = & $PythonPath -c "import torch; print('1' if torch.cuda.is_available() else '0')" 2>$null
        return (($out -join "") -match "1")
    } catch {
        return $false
    }
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "              Quick Benchmark (Single Method)               " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Method:  $Method" -ForegroundColor Yellow
Write-Host "Dataset: $Dataset" -ForegroundColor Yellow
Write-Host "Scene:   $Root" -ForegroundColor Yellow
Write-Host "Python:  $PythonExe" -ForegroundColor Yellow
Write-Host ""

# Resolver a raiz do dataset
if (-not (Test-Path $Root)) {
    $AltRoot = "./data/blender_synthetic/nerf_synthetic/lego"
    if (Test-Path $AltRoot) {
        $Root = $AltRoot
        Write-Host "Using alternate path: $Root" -ForegroundColor Yellow
    } else {
        throw "Dataset not found at '$Root' or '$AltRoot'. Run './scripts/windows/powershell/download_dataset.ps1' first."
    }
}

$MethodToRun = $Method
$ExtraFileToUse = $ExtraFile
$HasCuda = Test-CudaAvailable -PythonPath $PythonExe

if (-not $HasCuda) {
    if ($Method -in @("nerf_static", "nerf_dynamic")) {
        Write-Host "No CUDA detected. Running '$Method' on CPU (this may be slow)." -ForegroundColor Yellow
    } elseif ($Method -eq "gs_static" -and -not $ExtraFileToUse) {
        $CpuGsConfig = "./configs/gs_static_cpu.json"
        if (Test-Path $CpuGsConfig) {
            Write-Host "No CUDA detected. Using CPU config: $CpuGsConfig" -ForegroundColor Yellow
            $ExtraFileToUse = $CpuGsConfig
        }
    }
}

Write-Host "Starting benchmark..." -ForegroundColor Green
Write-Host ""

$methodArgs = @(
    "-m", "nvs_benchmark.cli", "method-run",
    "--method", $MethodToRun,
    "--dataset", $Dataset,
    "--root", $Root,
    "--split", "train",
    "--output-dir", "./artifacts",
    "--log-dir", "./logs",
    "--compute-metrics",
    "--snapshot-file", "./artifacts/metrics/latest.json",
    "--append-snapshot"
)

if ($ExtraFileToUse) {
    $methodArgs += @("--extra-file", $ExtraFileToUse)
}

& $PythonExe @methodArgs
$runExit = $LASTEXITCODE

if ($runExit -ne 0) {
    throw "Benchmark failed for method '$MethodToRun' (exit=$runExit)."
}

if ($ComputeMetrics) {
    Write-Host ""
    Write-Host "Calculating reference metrics..." -ForegroundColor Yellow
    $ReferenceDir = Join-Path (Split-Path $Root -Parent) "lego\test" 2>$null
    if ((Test-Path $ReferenceDir)) {
        $refArgs = @(
            "-m", "nvs_benchmark.cli", "method-run",
            "--method", $MethodToRun,
            "--dataset", $Dataset,
            "--root", $Root,
            "--split", "train",
            "--output-dir", "./artifacts",
            "--log-dir", "./logs",
            "--compute-metrics",
            "--reference-dir", $ReferenceDir,
            "--snapshot-file", "./artifacts/metrics/latest.json",
            "--append-snapshot"
        )
        if ($ExtraFileToUse) {
            $refArgs += @("--extra-file", $ExtraFileToUse)
        }
        & $PythonExe @refArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Reference metrics run failed for method '$MethodToRun' (exit=$LASTEXITCODE)."
        }
    }
}

Write-Host ""
Write-Host "===========================================================" -ForegroundColor Green
Write-Host "Benchmark complete!" -ForegroundColor Green
Write-Host "===========================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Results saved to:" -ForegroundColor Cyan
Write-Host "  - Metrics:  ./artifacts/metrics/latest.json" -ForegroundColor Cyan
Write-Host "  - Logs:     ./logs/runs/" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  - View results:       ./scripts/windows/powershell/preview.ps1" -ForegroundColor Cyan
Write-Host "  - Generate report:    ./scripts/windows/powershell/generate_report.ps1" -ForegroundColor Cyan
Write-Host ""
