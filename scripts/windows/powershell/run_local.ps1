$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..\..\..")).Path
Set-Location $ProjectRoot
$env:PYTHONPATH = "$ProjectRoot\src"

$VenvPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$PythonExe = if (Test-Path $VenvPython) { $VenvPython } else { "python" }

& $PythonExe -m pip install --upgrade pip
& $PythonExe -m pip install -e .

& $PythonExe -m nvs_benchmark.cli status
& $PythonExe -m nvs_benchmark.cli methods-check --output-dir ./artifacts --log-dir ./logs
& $PythonExe -m nvs_benchmark.cli metrics-check --output-dir ./artifacts --snapshot-file ./artifacts/metrics/latest_preview.json --log-dir ./logs
& $PythonExe -m nvs_benchmark.cli report-generate --snapshot-file ./artifacts/metrics/latest_preview.json --output-dir ./artifacts/reports --report-name benchmark_report --log-dir ./logs
