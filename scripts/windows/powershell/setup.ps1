param(
    [switch]$SkipDataset
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..\..\..")).Path
Set-Location $ProjectRoot
$env:PYTHONPATH = "$ProjectRoot\src"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "         NVS Benchmark - Environment Setup                  " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Passo 1: Criar ambiente virtual
Write-Host "[1/3] Creating virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path "venv")) {
    python -m venv venv
    Write-Host "[OK] Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "[OK] Virtual environment already exists" -ForegroundColor Green
}

# Passo 2: Instalar dependências
Write-Host ""
Write-Host "[2/3] Installing dependencies..." -ForegroundColor Yellow
$VenvPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$PythonExe = if (Test-Path $VenvPython) { $VenvPython } else { "python" }

& $PythonExe -m pip install --upgrade pip 2>&1 | Out-Null
& $PythonExe -m pip install -e . 2>&1 | Out-Null
Write-Host "[OK] Dependencies installed" -ForegroundColor Green

# Passo 3: Validar a instalação
Write-Host ""
Write-Host "[3/3] Validating installation..." -ForegroundColor Yellow
& $PythonExe -m nvs_benchmark.cli status
Write-Host "[OK] Setup validated" -ForegroundColor Green

# Opcional: Baixar dataset
if (-not $SkipDataset) {
    Write-Host ""
    Write-Host "Would you like to download a dataset for benchmarking?" -ForegroundColor Cyan
    $response = Read-Host "Download blender_synthetic (500MB) now? (yes/no)"
    
    if ($response -eq "yes" -or $response -eq "y") {
        Write-Host ""
        & $PythonExe -m nvs_benchmark.cli install `
            --catalog-file ./configs/install_catalog.json `
            --only datasets `
            --execute
        Write-Host "[OK] Dataset downloaded successfully" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "===========================================================" -ForegroundColor Green
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "===========================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Run a quick benchmark:    ./scripts/windows/powershell/benchmark_quick.ps1" -ForegroundColor Cyan
Write-Host "  2. View 3D results:          ./scripts/windows/powershell/preview.ps1" -ForegroundColor Cyan
Write-Host "  3. Generate HTML report:     ./scripts/windows/powershell/generate_report.ps1" -ForegroundColor Cyan
Write-Host ""
