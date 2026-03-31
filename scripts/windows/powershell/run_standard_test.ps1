param(
    [string]$OutputDir = "./artifacts",
    [string]$LogDir = "./logs",
    [string]$SnapshotFile = "./artifacts/metrics/latest_preview.json",
    [string]$ReportName = "benchmark_report",
    [switch]$RunUnitTests
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..\..\..")).Path
Set-Location $ProjectRoot
$env:PYTHONPATH = "$ProjectRoot\src"

$VenvPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$PythonExe = if (Test-Path $VenvPython) { $VenvPython } else { "python" }

$args = @(
    "-m", "nvs_benchmark.cli", "standard-test",
    "--output-dir", $OutputDir,
    "--log-dir", $LogDir,
    "--snapshot-file", $SnapshotFile,
    "--report-name", $ReportName
)

if ($RunUnitTests) {
    $args += "--run-unit-tests"
}

& $PythonExe @args
