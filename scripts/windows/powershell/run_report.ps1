param(
    [string]$SnapshotFile = "./artifacts/metrics/latest_preview.json",
    [string]$OutputDir = "./artifacts/reports",
    [string]$ReportName = "benchmark_report",
    [string]$LogDir = "./logs",
    [switch]$NoPdf
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..\..\..")).Path
Set-Location $ProjectRoot
$env:PYTHONPATH = "$ProjectRoot\src"

$VenvPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$PythonExe = if (Test-Path $VenvPython) { $VenvPython } else { "python" }

$reportArgs = @(
    "-m", "nvs_benchmark.cli", "report-generate",
    "--snapshot-file", $SnapshotFile,
    "--output-dir", $OutputDir,
    "--report-name", $ReportName,
    "--log-dir", $LogDir
)

if ($NoPdf) {
    $reportArgs += "--no-pdf"
}

& $PythonExe @reportArgs
