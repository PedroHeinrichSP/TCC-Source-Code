param(
    [switch]$SkipDataset
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..\..\..")).Path
Set-Location $ProjectRoot
$env:PYTHONPATH = "$ProjectRoot\src"

Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "   NVS Benchmark - Environment Setup" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Create virtual environment
Write-Host "Step 1/3: Creating virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path "venv")) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3 -m venv venv
    } elseif (Get-Command python3 -ErrorAction SilentlyContinue) {
        & python3 -m venv venv
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -m venv venv
    } else {
        throw "Python launcher not found. Install Python 3.x and re-run setup."
    }
    Write-Host "Virtual environment created" -ForegroundColor Green
} else {
    Write-Host "Virtual environment already exists" -ForegroundColor Green
}

# Activate venv
& ".\venv\Scripts\Activate.ps1"

# Step 2: Install dependencies
Write-Host ""
Write-Host "Step 2/3: Installing dependencies..." -ForegroundColor Yellow

if (-not (& python -m pip install --upgrade pip)) {
    throw "Failed to upgrade pip"
}

if (-not (& python -m pip install -e .)) {
    throw "Failed to install nvs_benchmark package. Make sure pyproject.toml is in the project root."
}

# torchsearchsorted is optional: this repo has fallback to torch.searchsorted in D-NeRF.
$TorchSearchsortedSetup = Join-Path $ProjectRoot "third_party\d_nerf\torchsearchsorted\setup.py"
if (Test-Path $TorchSearchsortedSetup) {
    $importOk = $false
    try {
        & python -c "from torchsearchsorted import searchsorted" *> $null
        $importOk = ($LASTEXITCODE -eq 0)
    } catch {
        $importOk = $false
    }
    if (-not $importOk) {
        Write-Host "torchsearchsorted not available. Continuing with torch.searchsorted fallback." -ForegroundColor Yellow
    }
}

Write-Host "Dependencies installed" -ForegroundColor Green

# Step 3: Validate installation
Write-Host ""
Write-Host "Step 3/3: Validating installation..." -ForegroundColor Yellow

if (-not (& python -m nvs_benchmark.cli status)) {
    throw "Failed to validate nvs_benchmark installation"
}

Write-Host "Setup validated" -ForegroundColor Green

# Optional: Download dataset
if (-not $SkipDataset) {
    Write-Host ""
    Write-Host "Download a dataset for benchmarking?" -ForegroundColor Cyan
    $response = Read-Host "Download blender_synthetic (500MB) now? (yes/no)"
    
    if ($response -eq "yes" -or $response -eq "y") {
        Write-Host ""
        & python -m nvs_benchmark.cli install `
            --catalog-file ./configs/install_catalog.json `
            --only datasets `
            --execute
        
        if ($LASTEXITCODE -ne 0) {
            throw "Dataset download failed"
        }
        Write-Host "Dataset downloaded successfully" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "====================================================" -ForegroundColor Green
Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "====================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Cyan
Write-Host "  1. Run a quick benchmark:    .\scripts\windows\powershell\benchmark_quick.ps1" -ForegroundColor Cyan
Write-Host "  2. View 3D results:          .\scripts\windows\powershell\preview.ps1" -ForegroundColor Cyan
Write-Host "  3. Generate HTML report:     .\scripts\windows\powershell\generate_report.ps1" -ForegroundColor Cyan
Write-Host ""
