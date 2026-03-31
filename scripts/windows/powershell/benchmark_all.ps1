param(
    [string]$Dataset = "blender_synthetic",
    [string]$Root = "./data/blender_synthetic/nerf_synthetic/lego",
    [string]$OutputDir = "./artifacts",
    [string]$ReportName = "benchmark_all"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "../../..")).Path
Set-Location $ProjectRoot
$env:PYTHONPATH = "$ProjectRoot\src"

# Venv detection
if ($env:VIRTUAL_ENV) {
    Write-Host "Using active virtual environment: $env:VIRTUAL_ENV" -ForegroundColor Green
} elseif (Test-Path "venv\Scripts\activate.ps1") {
    & "venv\Scripts\Activate.ps1"
} else {
    Write-Host "❌ Virtual environment not found at ./venv" -ForegroundColor Red
    Write-Host "Run '.\scripts\windows\powershell\setup.ps1' first to create the environment." -ForegroundColor Red
    exit 1
}

Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║            Full Benchmark Suite (All Methods)              ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "Dataset: $Dataset" -ForegroundColor Yellow
Write-Host "Scene:   $Root" -ForegroundColor Yellow
Write-Host ""

if (-not (Test-Path $Root)) {
    Write-Host "❌ Dataset not found at '$Root'" -ForegroundColor Red
    Write-Host "Run '.\scripts\windows\powershell\download_dataset.ps1' first." -ForegroundColor Red
    exit 1
}

$Methods = @("nerf_static", "nerf_dynamic", "gs_static", "gs_dynamic")
$SnapshotFile = Join-Path $OutputDir "metrics\$ReportName.json"

New-Item (Split-Path $SnapshotFile -Parent) -ItemType Directory -Force -ErrorAction SilentlyContinue | Out-Null

Write-Host "⏱️  Starting benchmark suite (this may take a while)..." -ForegroundColor Green
Write-Host ""

foreach ($Method in $Methods) {
    Write-Host "────────────────────────────────────────────────────────────" -ForegroundColor Cyan
    Write-Host "Running $Method..." -ForegroundColor Yellow
    Write-Host "────────────────────────────────────────────────────────────" -ForegroundColor Cyan
    
    python -m nvs_benchmark.cli method-run `
        --method $Method `
        --dataset $Dataset `
        --root $Root `
        --split train `
        --output-dir $OutputDir `
        --log-dir ./logs `
        --compute-metrics `
        --snapshot-file $SnapshotFile `
        --append-snapshot 2>&1 | Out-Null
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[✓] $Method completed" -ForegroundColor Green
    } else {
        Write-Host "[⚠] $Method failed or skipped (possible missing GPU/dependencies)" -ForegroundColor Yellow
    }
    Write-Host ""
}

Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host "✅ Benchmark suite complete." -ForegroundColor Green
Write-Host "════════════════════════════════════════════════════════════" -ForegroundColor Green
Write-Host ""
Write-Host "Results: $SnapshotFile" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  - View results:       .\scripts\windows\powershell\preview.ps1" -ForegroundColor Cyan
Write-Host "  - Generate report:    .\scripts\windows\powershell\generate_report.ps1 -ReportName $ReportName" -ForegroundColor Cyan
Write-Host ""
