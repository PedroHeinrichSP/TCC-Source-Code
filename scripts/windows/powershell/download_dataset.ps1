param(
    [string]$Dataset = "blender_synthetic",
    [string]$CatalogFile = "./configs/install_catalog.json"
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
Write-Host "║              Download / Setup Dataset                      ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

Write-Host "Available datasets:" -ForegroundColor Cyan
Write-Host "  • blender_synthetic : Synthetic scenes (Lego, Chair, etc.) - ~500MB" -ForegroundColor Yellow
Write-Host "  • d_nerf : Dynamic scenes (human motion) - ~1GB" -ForegroundColor Yellow
Write-Host ""

if ($Dataset -notmatch "^(blender_synthetic|d_nerf)$") {
    Write-Host "⚠️  Invalid dataset: $Dataset" -ForegroundColor Yellow
    exit 1
}

Write-Host "📥 Downloading $Dataset..." -ForegroundColor Green
Write-Host ""

python -m nvs_benchmark.cli install `
    --catalog-file $CatalogFile `
    --only datasets `
    --execute

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host "✅ Dataset downloaded successfully!" -ForegroundColor Green
    Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host ""
    Write-Host "✓ Dataset location: ./data/$Dataset" -ForegroundColor Cyan
    Write-Host "✓ Ready for benchmarking!" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Next step: Run a benchmark" -ForegroundColor Cyan
    Write-Host "  .\scripts\windows\powershell\benchmark_quick.ps1" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host "❌ Dataset download failed" -ForegroundColor Red
    exit 1
}
