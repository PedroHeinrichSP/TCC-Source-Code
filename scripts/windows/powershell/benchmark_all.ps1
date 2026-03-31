param(
    [string]$Dataset = "blender_synthetic",
    [string]$Root = "./data/blender_synthetic/nerf_synthetic/lego",
    [string]$OutputDir = "./artifacts",
    [string]$ReportName = "benchmark_all"
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

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Full Benchmark Suite (All Methods)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Dataset: $Dataset" -ForegroundColor Yellow
Write-Host "Scene:   $Root" -ForegroundColor Yellow
Write-Host "Python:  $PythonExe" -ForegroundColor Yellow
Write-Host ""

if (-not (Test-Path $Root)) {
    throw "Dataset not found at '$Root'. Run './scripts/windows/powershell/download_dataset.ps1' first."
}

$Methods = @("nerf_static", "nerf_dynamic", "gs_static", "gs_dynamic")
$SnapshotFile = Join-Path $OutputDir "metrics\$ReportName.json"
New-Item (Split-Path $SnapshotFile -Parent) -ItemType Directory -Force -ErrorAction SilentlyContinue | Out-Null

Write-Host "Starting benchmark suite..." -ForegroundColor Green
Write-Host "This may take a while." -ForegroundColor Yellow
Write-Host ""

foreach ($Method in $Methods) {
    Write-Host "------------------------------------------------------------" -ForegroundColor Cyan
    Write-Host "Running $Method..." -ForegroundColor Yellow
    Write-Host "------------------------------------------------------------" -ForegroundColor Cyan

    & $PythonExe -m nvs_benchmark.cli method-run `
        --method $Method `
        --dataset $Dataset `
        --root $Root `
        --split train `
        --output-dir $OutputDir `
        --log-dir ./logs `
        --compute-metrics `
        --snapshot-file $SnapshotFile `
        --append-snapshot

    if ($LASTEXITCODE -eq 0) {
        Write-Host "[ok] $Method completed" -ForegroundColor Green
    } else {
        Write-Host "[warn] $Method failed or skipped (possible missing GPU support)." -ForegroundColor Yellow
    }
    Write-Host ""
}

Write-Host "============================================================" -ForegroundColor Green
Write-Host "Benchmark suite complete." -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Results: $SnapshotFile" -ForegroundColor Cyan
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  - ./scripts/windows/powershell/preview.ps1" -ForegroundColor Cyan
Write-Host "  - ./scripts/windows/powershell/generate_report.ps1 -ReportName $ReportName" -ForegroundColor Cyan
