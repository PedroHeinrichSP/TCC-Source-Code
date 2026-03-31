param(
    [string]$Dataset = "blender_synthetic",
    [string]$CatalogFile = "./configs/install_catalog.json"
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
Write-Host "Download or Setup Dataset" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Python: $PythonExe" -ForegroundColor Yellow
Write-Host ""

$DatasetInfo = @{
    "blender_synthetic" = "Synthetic scenes (Lego, Chair, etc.) - ~500MB"
    "d_nerf" = "Dynamic scenes (human motion) - ~1GB"
}

Write-Host "Available datasets:" -ForegroundColor Cyan
foreach ($key in $DatasetInfo.Keys) {
    Write-Host "  - $key : $($DatasetInfo[$key])" -ForegroundColor Yellow
}
Write-Host ""

if ($Dataset -notmatch "^(blender_synthetic|d_nerf)$") {
    throw "Invalid dataset: $Dataset"
}

Write-Host "Downloading $Dataset..." -ForegroundColor Green
Write-Host ""

& $PythonExe -m nvs_benchmark.cli install `
    --catalog-file $CatalogFile `
    --only datasets `
    --execute

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "Dataset downloaded successfully." -ForegroundColor Green
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "Dataset location: ./data/$Dataset" -ForegroundColor Cyan
    Write-Host "Next step: ./scripts/windows/powershell/benchmark_quick.ps1" -ForegroundColor Cyan
} else {
    throw "Dataset download failed"
}
