param(
    [string]$UiHost = "127.0.0.1",
    [int]$Port = 8765,
    [string]$MetricsFile = "./artifacts/metrics/latest_preview.json",
    [string]$SceneTransformsFile = "./data/blender_synthetic/nerf_synthetic/lego/transforms_train.json",
    [string]$InstallCatalogFile = "./configs/install_catalog.json",
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$WindowsDir = Split-Path -Parent $ScriptDir
$ScriptsDir = Split-Path -Parent $WindowsDir
$ProjectRoot = Split-Path -Parent $ScriptsDir
Set-Location $ProjectRoot
$env:PYTHONPATH = "$ProjectRoot\src"

$VenvPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$PythonExe = if (Test-Path $VenvPython) { $VenvPython } else { "python" }

# Se estiver no default latest_preview, prioriza snapshots nao-smoke quando disponiveis.
if ($MetricsFile -eq "./artifacts/metrics/latest_preview.json") {
    $metricsCandidates = @(
        "./artifacts/metrics/dev_method_run_snapshot.json",
        "./artifacts/metrics/lego_preliminar.json",
        "./artifacts/metrics/lego_preliminar_cpu.json",
        "./artifacts/metrics/latest_preview.json"
    )
    foreach ($candidate in $metricsCandidates) {
        if (Test-Path $candidate) {
            $MetricsFile = $candidate
            break
        }
    }
}

$sceneCandidates = @(
    $SceneTransformsFile,
    "./data/blender_synthetic/nerf_synthetic/lego/transforms_train.json",
    "./data/blender_synthetic/lego/lego/transforms_train.json"
)
$ResolvedSceneTransformsFile = $null
foreach ($candidate in $sceneCandidates) {
    if (Test-Path $candidate) {
        $ResolvedSceneTransformsFile = $candidate
        break
    }
}

if (-not $ResolvedSceneTransformsFile) {
    throw "Cena real nao encontrada. Execute: powershell -ExecutionPolicy Bypass -File ./scripts/windows/powershell/setup_lego.ps1"
}

# Converter caminhos para absolutos para evitar problemas com diretório de trabalho
$AbsMetricsFile = (Resolve-Path $MetricsFile).Path
$AbsSceneTransformsFile = (Resolve-Path $ResolvedSceneTransformsFile).Path
$AbsInstallCatalogFile = (Resolve-Path $InstallCatalogFile).Path

$args = @(
    "-m", "nvs_benchmark.cli", "ui-preview",
    "--host", $UiHost,
    "--port", $Port,
    "--metrics-file", $AbsMetricsFile,
    "--scene-transforms-file", $AbsSceneTransformsFile,
    "--install-catalog-file", $AbsInstallCatalogFile
)

if ($NoBrowser) {
    $args += "--no-browser"
}

& $PythonExe @args
