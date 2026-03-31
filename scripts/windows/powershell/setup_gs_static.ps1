param(
    [string]$RepoUrl = "https://github.com/graphdeco-inria/gaussian-splatting",
    [string]$TargetDir = "./third_party/gaussian_splatting"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..\..\..")).Path
Set-Location $ProjectRoot

$TargetPath = Join-Path $ProjectRoot $TargetDir
$TargetParent = Split-Path -Parent $TargetPath
if (-not (Test-Path $TargetParent)) {
    New-Item -ItemType Directory -Path $TargetParent -Force | Out-Null
}

if (Test-Path $TargetPath) {
    Write-Host "[skip] Repositorio ja existe: $TargetPath"
} else {
    $gitCmd = Get-Command git -ErrorAction SilentlyContinue
    if ($null -ne $gitCmd) {
        Write-Host "[plan] Clonando com git..."
        & git clone --depth 1 $RepoUrl $TargetPath
        if ($LASTEXITCODE -ne 0) {
            throw "Falha no git clone de $RepoUrl"
        }
    } else {
        Write-Host "[warn] Git nao encontrado. Usando fallback ZIP..."
        $zipUrl = "$RepoUrl/archive/refs/heads/main.zip"
        $zipFile = Join-Path $TargetParent "gaussian_splatting_main.zip"
        $extractRoot = Join-Path $TargetParent "gaussian_splatting_extract"

        $ProgressPreference = "SilentlyContinue"
        Invoke-WebRequest -Uri $zipUrl -OutFile $zipFile -ErrorAction Stop

        if (Test-Path $extractRoot) {
            Remove-Item -Recurse -Force $extractRoot
        }
        Expand-Archive -Path $zipFile -DestinationPath $extractRoot -Force

        $inner = Get-ChildItem -Path $extractRoot -Directory | Select-Object -First 1
        if ($null -eq $inner) {
            throw "Falha ao extrair ZIP do repositorio 3DGS."
        }

        Move-Item -Path $inner.FullName -Destination $TargetPath -Force
        Remove-Item -Path $zipFile -Force
        Remove-Item -Path $extractRoot -Recurse -Force
    }
}

$trainFile = Join-Path $TargetPath "train.py"
$renderFile = Join-Path $TargetPath "render.py"
if (-not (Test-Path $trainFile) -or -not (Test-Path $renderFile)) {
    throw "Repositorio 3DGS incompleto em $TargetPath (train.py/render.py ausentes)."
}

Write-Host "[ok] Repositorio 3DGS pronto em: $TargetPath"
Write-Host "[next] Ajuste dependencias do repo oficial antes do treino (CUDA/PyTorch/submodulos)."
