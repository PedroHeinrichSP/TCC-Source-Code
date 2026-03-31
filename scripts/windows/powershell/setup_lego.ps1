param(
    [string]$CatalogFile = "./configs/install_catalog.json",
    [string]$LegoRoot = "./data/blender_synthetic/nerf_synthetic/lego"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = (Resolve-Path (Join-Path $ScriptDir "..\..\..")).Path
Set-Location $ProjectRoot
$env:PYTHONPATH = "$ProjectRoot\src"

$VenvPython = Join-Path $ProjectRoot "venv\Scripts\python.exe"
$PythonExe = if (Test-Path $VenvPython) { $VenvPython } else { "python" }

Write-Host "[1/3] Instalando dataset Blender synthetic via catalogo..."
& $PythonExe -m nvs_benchmark.cli install --catalog-file $CatalogFile --only datasets --execute

$LegoPath = Join-Path $ProjectRoot $LegoRoot
$LegacyLegoPath = Join-Path $ProjectRoot "data\blender_synthetic\lego"
if (-not (Test-Path $LegoPath) -and (Test-Path $LegacyLegoPath)) {
    $LegoPath = $LegacyLegoPath
}

if (-not (Test-Path $LegoPath)) {
    throw "Cena Lego nao encontrada. Verifique extracao em ./data/blender_synthetic."
}

Write-Host "[2/3] Validando estrutura do dataset Lego em: $LegoPath"
& $PythonExe -m nvs_benchmark.cli dataset-check --dataset blender_synthetic --root $LegoPath --split train

Write-Host "[3/3] Setup concluido com sucesso."
Write-Host "Use este root nos proximos comandos: $LegoPath"
