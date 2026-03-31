param(
    [string]$SnapshotFile = "./artifacts/metrics/latest.json",
    [string]$ReportName = "benchmark_report",
    [string]$OutputDir = "./artifacts/reports",
    [switch]$NoPdf
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

Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                Generate HTML Report                        ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""
Write-Host "Python: $PythonExe" -ForegroundColor Yellow
Write-Host ""

if (-not (Test-Path $SnapshotFile)) {
    throw "Metrics file not found: $SnapshotFile`nRun a benchmark first with: ./scripts/windows/powershell/benchmark_quick.ps1"
}

Write-Host "📊 Generating report from metrics..." -ForegroundColor Green
Write-Host ""
Write-Host "Input:  $SnapshotFile" -ForegroundColor Yellow
Write-Host "Output: $OutputDir/$ReportName.html" -ForegroundColor Yellow
Write-Host ""

$reportArgs = @(
    "-m", "nvs_benchmark.cli", "report-generate",
    "--snapshot-file", $SnapshotFile,
    "--output-dir", $OutputDir,
    "--report-name", $ReportName,
    "--log-dir", "./logs"
)

if ($NoPdf) {
    $reportArgs += "--no-pdf"
}

& $PythonExe @reportArgs

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host "✅ Report generated successfully!" -ForegroundColor Green
    Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Green
    Write-Host ""
    Write-Host "📄 HTML Report:" -ForegroundColor Cyan
    Write-Host "   $OutputDir/$ReportName.html" -ForegroundColor Cyan
    Write-Host ""
    
    # Tentar abrir no navegador padrão
    try {
        $ReportPath = (Get-Item "$OutputDir/$ReportName.html").FullName
        & powershell -NoProfile -Command "Start-Process '$ReportPath'"
        Write-Host "🌐 Opening report in browser..." -ForegroundColor Yellow
    } catch {
        Write-Host "💡 Open manually: $OutputDir/$ReportName.html" -ForegroundColor Cyan
    }
} else {
    throw "Report generation failed"
}
