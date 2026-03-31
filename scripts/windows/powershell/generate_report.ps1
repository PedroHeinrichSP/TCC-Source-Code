param(
    [string]$SnapshotFile = "artifacts/metrics/latest.json",
    [string]$ReportName = "benchmark_report",
    [string]$OutputDir = "./artifacts/reports"
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

Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "   Generate HTML Report" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host ""

if (-not (Test-Path $SnapshotFile)) {
    Write-Host "Metrics file not found: $SnapshotFile" -ForegroundColor Red
    Write-Host "Run a benchmark first with: .\scripts\windows\powershell\benchmark_quick.ps1" -ForegroundColor Red
    exit 1
}

Write-Host "Generating report from metrics..." -ForegroundColor Green
Write-Host ""
Write-Host "Input:  $SnapshotFile" -ForegroundColor Yellow
Write-Host "Output: $OutputDir/$ReportName.html" -ForegroundColor Yellow
Write-Host ""

python -m nvs_benchmark.cli report-generate `
    --snapshot-file $SnapshotFile `
    --output-dir $OutputDir `
    --report-name $ReportName `
    --log-dir ./logs `
    --no-pdf

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "====================================================" -ForegroundColor Green
    Write-Host "Report generated successfully!" -ForegroundColor Green
    Write-Host "====================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "📄 HTML Report:" -ForegroundColor Cyan
    Write-Host "   $OutputDir/$ReportName.html" -ForegroundColor Cyan
    Write-Host ""
    
    # Try to open in browser
    try {
        $ReportPath = (Get-Item "$OutputDir/$ReportName.html").FullName
        Start-Process $ReportPath
        Write-Host "Opening report in browser..." -ForegroundColor Yellow
    } catch {
        Write-Host "💡 Open manually: $OutputDir/$ReportName.html" -ForegroundColor Cyan
    }
    Write-Host ""
} else {
    Write-Host "ERROR: Report generation failed" -ForegroundColor Red
    exit 1
}
