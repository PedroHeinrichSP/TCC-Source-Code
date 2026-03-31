param(
    [string]$Host = "127.0.0.1",
    [int]$Port = 8765,
    [string]$MetricsFile = "./artifacts/metrics/latest.json",
    [string]$SceneFile = "./data/blender_synthetic/nerf_synthetic/lego/transforms_train.json"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "../../..")).Path
Set-Location $ProjectRoot
$env:PYTHONPATH = "$ProjectRoot\src"

# Venv detection
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
    exit 1
}

$PythonExe = python -c "import sys; sys.stdout.write(sys.executable)" 2>$null
Write-Host "Python in use: $PythonExe" -ForegroundColor Green

# Prefer non-smoke snapshots if default metrics file
if ($MetricsFile -eq "./artifacts/metrics/latest.json") {
    $metricsCandidates = @(
        "./artifacts/metrics/dev_method_run_snapshot.json",
        "./artifacts/metrics/lego_preliminar.json",
        "./artifacts/metrics/lego_preliminar_cpu.json",
        "./artifacts/metrics/latest.json"
    )
    foreach ($candidate in $metricsCandidates) {
        if (Test-Path $candidate) {
            $MetricsFile = $candidate
            break
        }
    }
}

$sceneCandidates = @(
    $SceneFile,
    "./data/blender_synthetic/nerf_synthetic/lego/transforms_train.json",
    "./data/blender_synthetic/lego/lego/transforms_train.json"
)

$ResolvedSceneFile = $null
foreach ($candidate in $sceneCandidates) {
    if (Test-Path $candidate) {
        $ResolvedSceneFile = $candidate
        break
    }
}

if (-not $ResolvedSceneFile) {
    Write-Host "Scene not found. Execute setup or install Blender Synthetic in ./data/blender_synthetic." -ForegroundColor Red
    exit 1
}

# Convert paths to absolute to avoid working directory issues
$AbsMetricsFile = (Resolve-Path $MetricsFile).Path
$AbsSceneFile = (Resolve-Path $ResolvedSceneFile).Path
$AbsCatalogFile = (Resolve-Path "./configs/install_catalog.json").Path

Write-Host "UI Preview Configuration" -ForegroundColor Cyan
Write-Host "  Metrics File: $AbsMetricsFile" -ForegroundColor Yellow
Write-Host "  Scene File:   $AbsSceneFile" -ForegroundColor Yellow
Write-Host ""

$cmdArgs = @(
    "-m", "nvs_benchmark.cli", "ui-preview",
    "--host", $Host,
    "--port", $Port,
    "--metrics-file", $AbsMetricsFile,
    "--scene-transforms-file", $AbsSceneFile,
    "--install-catalog-file", $AbsCatalogFile
)

python @cmdArgs
