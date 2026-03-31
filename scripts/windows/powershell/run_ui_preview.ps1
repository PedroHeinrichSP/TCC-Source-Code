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

function Get-VenvPython {
    param([string]$RootPath)

    $VenvPython = Join-Path $RootPath "venv\Scripts\python.exe"
    if (-not (Test-Path $VenvPython)) {
        throw "Virtual environment not found at '$VenvPython'. Run './scripts/windows/powershell/setup.ps1' first."
    }
    return $VenvPython
}

$PythonExe = Get-VenvPython -RootPath $ProjectRoot

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
