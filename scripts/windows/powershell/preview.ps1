param(
    [Alias("Host")]
    [string]$UiHost = "127.0.0.1",
    [int]$Port = 8765,
    [string]$MetricsFile = "./artifacts/metrics/latest_preview.json",
    [string]$SceneTransformsFile = "./data/blender_synthetic/nerf_synthetic/lego/transforms_train.json",
    [string]$InstallCatalogFile = "./configs/install_catalog.json",
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..\..\..")).Path
Set-Location $ProjectRoot
$env:PYTHONPATH = "$ProjectRoot\src"

$RunUiPreviewScript = Join-Path $ScriptDir "run_ui_preview.ps1"
if (-not (Test-Path $RunUiPreviewScript)) {
    throw "Required script not found: $RunUiPreviewScript"
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "          3D Interactive Viewer (Viser)                     " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Starting dashboard + spatial preview..." -ForegroundColor Green
Write-Host ""
Write-Host ("Opening dashboard at: http://{0}:8780" -f $UiHost) -ForegroundColor Yellow
Write-Host ""
Write-Host "Features:" -ForegroundColor Cyan
Write-Host "  - Compare method results side-by-side" -ForegroundColor Cyan
Write-Host "  - Rotate, zoom, pan the 3D scene" -ForegroundColor Cyan
Write-Host "  - View rendered frames from each method" -ForegroundColor Cyan
Write-Host "  - Check performance metrics (FPS, VRAM)" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Reusa o script robusto com detecção de cena real e parâmetros completos.
$childArgs = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", $RunUiPreviewScript,
    "-UiHost", $UiHost,
    "-Port", $Port,
    "-SceneTransformsFile", $SceneTransformsFile,
    "-InstallCatalogFile", $InstallCatalogFile
)
if ($MetricsFile -ne "./artifacts/metrics/latest_preview.json") {
    $childArgs += @("-MetricsFile", $MetricsFile)
}
if ($NoBrowser) {
    $childArgs += "-NoBrowser"
}

& powershell @childArgs
