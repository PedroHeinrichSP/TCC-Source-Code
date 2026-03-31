param(
    [switch]$SkipDataset
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..\..\..")).Path
Set-Location $ProjectRoot
$env:PYTHONPATH = "$ProjectRoot\src"

function Invoke-Checked {
    param(
        [string]$Program,
        [string[]]$CommandArgs,
        [string]$ErrorMessage
    )

    & $Program @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "$ErrorMessage (exit=$LASTEXITCODE)"
    }
}

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

Invoke-Checked -Program $PythonExe -CommandArgs @("-m", "pip", "install", "--upgrade", "pip") -ErrorMessage "Failed to upgrade pip"
Invoke-Checked -Program $PythonExe -CommandArgs @("-m", "pip", "install", "-e", ".") -ErrorMessage "Failed to install project in editable mode"
Write-Host "[OK] Dependencies installed" -ForegroundColor Green

# Passo 3: Validar a instalação
Write-Host ""
Write-Host "[3/3] Validating installation..." -ForegroundColor Yellow

$ExpectedPackagePaths = @(
    "src\nvs_benchmark\__init__.py",
    "src\nvs_benchmark\data\__init__.py",
    "src\nvs_benchmark\core\__init__.py"
)

foreach ($relPath in $ExpectedPackagePaths) {
    if (-not (Test-Path (Join-Path $ProjectRoot $relPath))) {
        throw "Repository appears incomplete: missing '$relPath'. Re-clone or sync this workspace before running setup."
    }
}

Invoke-Checked -Program $PythonExe -CommandArgs @("-m", "nvs_benchmark.cli", "status") -ErrorMessage "Setup validation failed"
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
