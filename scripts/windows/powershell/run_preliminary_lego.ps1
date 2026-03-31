param(
    [string]$DatasetRoot = "./data/blender_synthetic/lego",
    [string]$OutputDir = "./artifacts",
    [string]$LogDir = "./logs",
    [string]$SnapshotFile = "./artifacts/metrics/lego_preliminar.json",
    [string]$ReportName = "lego_preliminar",
    [string]$GsConfigFile = "./configs/gs_static_real.json",
    [switch]$SetupLego,
    [switch]$SetupGsRepo,
    [switch]$NoPdf
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..\..\..")).Path
Set-Location $ProjectRoot
$env:PYTHONPATH = "$ProjectRoot\src"

$VenvPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$PythonExe = if (Test-Path $VenvPython) { $VenvPython } else { "python" }

function Invoke-Checked {
    param(
        [string]$Program,
        [string[]]$Args,
        [string]$ErrorMessage
    )
    & $Program @Args
    if ($LASTEXITCODE -ne 0) {
        throw "$ErrorMessage (exit=$LASTEXITCODE)"
    }
}

function Test-CudaAvailable {
    # Checar se o CUDA está disponível criando um script Python temporário
    try {
        $venv = Join-Path $ProjectRoot "venv\Scripts\python.exe"
        $py = if (Test-Path $venv) { $venv } else { "python" }
        
        # Criar um arquivo Python temporário para o teste do CUDA
        $tempDir = [System.IO.Path]::GetTempPath()
        $testScript = Join-Path $tempDir "cuda_test_$([System.Guid]::NewGuid()).py"
        
        $pythonCode = "import torch`nprint(`"CUDA`" if torch.cuda.is_available() else `"CPU`")"
        Set-Content -Path $testScript -Value $pythonCode -Encoding UTF8
        
        $output = & $py $testScript 2>$null
        Remove-Item $testScript -Force -ErrorAction SilentlyContinue
        
        return ($output -match "CUDA")
    } catch {
        return $false
    }
}

function Invoke-MethodRunWithFallback {
    param(
        [string]$Method = "gs_static",
        [string]$ConfigFile = $GsConfigFile,
        [string[]]$ExtraArgs = @()
    )
    
    $methodToUse = $Method
    $configToUse = $ConfigFile
    
    if ($Method -eq "gs_static" -and -not (Test-CudaAvailable)) {
        Write-Host "[warn] CUDA nao disponivel. Alterando para nerf_static (CPU)..."
        $methodToUse = "nerf_static"
        $configToUse = ""  # nerf_static nao precisa de config extra neste stage
    }
    
    $args = @(
        "-m", "nvs_benchmark.cli", "method-run",
        "--method", $methodToUse,
        "--dataset", "blender_synthetic",
        "--root", $ResolvedDatasetRoot,
        "--split", "train",
        "--output-dir", $OutputDir,
        "--log-dir", $LogDir,
        "--compute-metrics",
        "--reference-dir", "$ResolvedDatasetRoot/test",
        "--snapshot-file", $SnapshotFile,
        "--append-snapshot"
    )
    
    if ($configToUse -and (Test-Path $configToUse)) {
        $args += "--extra-file", $configToUse
    }
    
    if ($ExtraArgs) {
        $args += $ExtraArgs
    }
    
    Write-Host "[run] Executando $methodToUse com metricas..."
    & $PythonExe @args
    
    if ($LASTEXITCODE -ne 0) {
        throw "Falha em method-run $methodToUse (exit=$LASTEXITCODE)"
    }
    
    Write-Host "[ok] $methodToUse completou com sucesso. Snapshot: $SnapshotFile"
}

if ($SetupGsRepo) {
    Write-Host "[setup] Configurando repositorio 3DGS..."
    powershell -ExecutionPolicy Bypass -File ./scripts/windows/powershell/setup_gs_static.ps1
}

if ($SetupLego) {
    Write-Host "[setup] Configurando dataset Lego..."
    powershell -ExecutionPolicy Bypass -File ./scripts/windows/powershell/setup_lego.ps1
    if ($LASTEXITCODE -ne 0) {
        throw "Falha no setup_lego.ps1 (exit=$LASTEXITCODE)"
    }
}

$ResolvedDatasetRoot = $DatasetRoot
$CandidateNested = "./data/blender_synthetic/nerf_synthetic/lego"
if (-not (Test-Path $ResolvedDatasetRoot) -and (Test-Path $CandidateNested)) {
    $ResolvedDatasetRoot = $CandidateNested
}

if (-not (Test-Path $ResolvedDatasetRoot)) {
    throw "Dataset Lego nao encontrado em '$DatasetRoot' nem em '$CandidateNested'."
}

Write-Host "[check] Validando dataset..."
Invoke-Checked -Program $PythonExe -Args @(
    "-m", "nvs_benchmark.cli", "dataset-check",
    "--dataset", "blender_synthetic",
    "--root", $ResolvedDatasetRoot,
    "--split", "train"
) -ErrorMessage "Falha em dataset-check"

Invoke-MethodRunWithFallback -Method "gs_static" -ConfigFile $GsConfigFile

$reportArgs = @(
    "-m", "nvs_benchmark.cli", "report-generate",
    "--snapshot-file", $SnapshotFile,
    "--output-dir", "$OutputDir/reports",
    "--report-name", $ReportName,
    "--log-dir", $LogDir
)
if ($NoPdf) {
    $reportArgs += "--no-pdf"
}

Write-Host "[report] Gerando relatorio..."
Invoke-Checked -Program $PythonExe -Args $reportArgs -ErrorMessage "Falha em report-generate"

Write-Host "[ok] Fluxo preliminar concluido."
Write-Host "Snapshot: $SnapshotFile"
Write-Host "Relatorio HTML: $OutputDir/reports/$ReportName.html"
Write-Host "Dataset root usado: $ResolvedDatasetRoot"
