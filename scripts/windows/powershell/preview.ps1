param(
    [string]$Host = "127.0.0.1",
    [int]$Port = 8765,
    [string]$MetricsFile = "./artifacts/metrics/latest.json"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "../../..")).Path
Set-Location $ProjectRoot
$env:PYTHONPATH = "$ProjectRoot\src"

$RunUiPreviewScript = Join-Path $ScriptDir "run_ui_preview.ps1"
if (-not (Test-Path $RunUiPreviewScript)) {
    Write-Host "Required script not found: $RunUiPreviewScript" -ForegroundColor Red
    exit 1
}

Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "   3D Interactive Viewer (Viser)" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting 3D viewer..." -ForegroundColor Green
Write-Host ""
Write-Host "Opening browser at: http://$Host`:$Port" -ForegroundColor Yellow
Write-Host ""
Write-Host "Features:" -ForegroundColor Cyan
Write-Host "  - Compare method results side-by-side" -ForegroundColor Cyan
Write-Host "  - Rotate, zoom, pan the 3D scene" -ForegroundColor Cyan
Write-Host "  - View rendered frames from each method" -ForegroundColor Cyan
Write-Host "  - Check performance metrics (FPS, VRAM)" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Reuse robust script with full scene detection
$childArgs = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $RunUiPreviewScript,
    "-Host", $Host,
    "-Port", $Port
)

if ($MetricsFile -ne "./artifacts/metrics/latest.json") {
    $childArgs += @("-MetricsFile", $MetricsFile)
}

& powershell @childArgs
