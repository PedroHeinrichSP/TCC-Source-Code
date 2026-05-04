# Auto-generated script from notebook: notebooks\nvs_benchmark_colab.ipynb

# ---- cell ----
# Parametros globais de repositorio e caminhos
REPO_URL = "https://github.com/PedroHeinrichSP/TCC-Source-Code.git"  # URL git do projeto
REPO_DIR = "/content/TCC"  # Caminho de clone no Colab
BRANCH = "update"  # Ex.: "main", "update"
RUN_ID = "colab_quick"  # Identificador da execucao (nome dos artefatos)
USE_GOOGLE_DRIVE = True  # Opcoes: True (persistente no Drive) | False (somente runtime)
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/NVS_Benchmark"  # Pasta base no Drive
DRIVE_DATA_DIR = f"{DRIVE_OUTPUT_DIR}/data"  # Cache de datasets
DRIVE_ARTIFACTS_DIR = f"{DRIVE_OUTPUT_DIR}/artifacts"  # Saidas/relatorios

# ============================================================================
# SELECAO + PERSISTENCIA
# ============================================================================
import json
import os
from pathlib import Path

# Controle de precedencia da selecao
LOAD_SAVED_SELECTION = False  # True: carrega do JSON salvo | False: usa valores editados abaixo

# Defaults editaveis (se LOAD_SAVED_SELECTION=False, estes valores vencem)
SELECTED_METHOD = "nerf_static"  # Opcoes comuns: "nerf_static", "nerf_dynamic", "gs_static", "gs_dynamic"
SELECTED_DATASET = "blender_synthetic"  # Opcoes comuns: "blender_synthetic", "d_nerf", "mipnerf360", "tanks_and_temples", "custom"
SELECTED_PRESET = "quick"  # Opcoes: "smoke", "quick", "preview", "standard", "full"
# ⚠ NOTA: No Colab, use "smoke" ou "quick". "full" causa OOM (exit -9).
RUN_MODE = "full"  # Opcoes: "full" | "quick_check"
APPLY_COMPATIBILITY_FILTER = True  # Opcoes: True | False
STRICT_RESULTS = True  # Opcoes: True | False
GENERATE_PDF = False  # Opcoes: True | False
MIN_REQUIRED_PAIRS = 1  # Inteiro >= 1
SELECTED_DATASET_ROOT = ""  # Vazio para auto-resolver na celula de descoberta

LOCAL_SELECTION_STATE_FILE = "/content/nvs_benchmark_selection.json"  # Estado local da sessao
DRIVE_SELECTION_STATE_FILE = f"{DRIVE_OUTPUT_DIR}/selection_state.json"  # Estado persistente no Drive


def _selection_state_file() -> Path:
    """Usa Drive quando estiver montado; senao usa arquivo local do runtime."""
    drive_ready = USE_GOOGLE_DRIVE and Path("/content/drive/MyDrive").exists()
    return Path(DRIVE_SELECTION_STATE_FILE if drive_ready else LOCAL_SELECTION_STATE_FILE)


def load_selection_state() -> bool:
    """Carrega selecao salva, se existir."""
    global SELECTED_METHOD, SELECTED_DATASET, SELECTED_PRESET
    global RUN_MODE, APPLY_COMPATIBILITY_FILTER, STRICT_RESULTS, GENERATE_PDF, MIN_REQUIRED_PAIRS

    state_file = _selection_state_file()
    if not state_file.exists():
        return False

    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[warn] Falha ao ler estado salvo em {state_file}: {exc}")
        return False

    SELECTED_METHOD = str(state.get("selected_method", SELECTED_METHOD))
    SELECTED_DATASET = str(state.get("selected_dataset", SELECTED_DATASET))
    SELECTED_PRESET = str(state.get("selected_preset", SELECTED_PRESET))
    RUN_MODE = str(state.get("run_mode", RUN_MODE))
    APPLY_COMPATIBILITY_FILTER = bool(state.get("apply_compatibility_filter", APPLY_COMPATIBILITY_FILTER))
    STRICT_RESULTS = bool(state.get("strict_results", STRICT_RESULTS))
    GENERATE_PDF = bool(state.get("generate_pdf", GENERATE_PDF))
    MIN_REQUIRED_PAIRS = int(state.get("min_required_pairs", MIN_REQUIRED_PAIRS))
    return True


def save_selection_state(reason: str = "manual") -> None:
    """Persiste selecao atual para reutilizacao nas proximas execucoes."""
    state_file = _selection_state_file()
    state_file.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "selected_method": SELECTED_METHOD,
        "selected_dataset": SELECTED_DATASET,
        "selected_preset": SELECTED_PRESET,
        "run_mode": RUN_MODE,
        "apply_compatibility_filter": APPLY_COMPATIBILITY_FILTER,
        "strict_results": STRICT_RESULTS,
        "generate_pdf": GENERATE_PDF,
        "min_required_pairs": MIN_REQUIRED_PAIRS,
        "saved_reason": reason,
    }
    state_file.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


loaded_from_state = False
if LOAD_SAVED_SELECTION:
    loaded_from_state = load_selection_state()

print("=" * 70)
print("SELECAO INICIAL")
print("=" * 70)
print(f"Metodo:           {SELECTED_METHOD}")
print(f"Dataset:          {SELECTED_DATASET}")
print(f"Preset:           {SELECTED_PRESET}")
print(f"Modo:             {RUN_MODE}")
print(f"Estrito:          {STRICT_RESULTS}")
print(f"Gerar PDF:        {GENERATE_PDF}")
print(f"Min pairs:        {MIN_REQUIRED_PAIRS}")
if LOAD_SAVED_SELECTION:
    print(f"Estado carregado: {'SIM' if loaded_from_state else 'NAO (arquivo inexistente/invalido)'}")
else:
    print("Estado carregado: NAO (priorizando valores editados nesta celula)")
print(f"Arquivo estado:   {_selection_state_file()}")
print("=" * 70)

# Sempre salva o estado vigente (editado ou carregado)
save_selection_state(reason="cell2-applied")

# ---- cell ----
# Clone do repositório
# Configurações estão na Célula 2: REPO_URL, BRANCH, REPO_DIR
print("=" * 70)
print(f"Clonando repositório: {REPO_URL} (branch: {BRANCH})")
print("=" * 70)

import os
import shutil
import subprocess
import sys

try:
    __import__("google.colab")
except Exception as exc:
    raise RuntimeError("Este notebook foi desenhado para Google Colab.") from exc

# Mude para um diretório seguro antes de remover REPO_DIR (evita erros no Colab)
os.chdir("/content") if os.path.exists("/content") else os.chdir(os.path.expanduser("~"))

if os.path.exists(REPO_DIR):
    print(f"Removendo pasta existente: {REPO_DIR}")
    shutil.rmtree(REPO_DIR)

print("Clonando...")
clone_result = subprocess.run([
    "git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, REPO_DIR
], text=True, capture_output=True)
if clone_result.stdout:
    print(clone_result.stdout)
if clone_result.stderr:
    print(clone_result.stderr)
if clone_result.returncode != 0:
    raise RuntimeError(f"Falha ao clonar o repositório (code={clone_result.returncode}).")

os.chdir(REPO_DIR)
print(f"\n✓ Projeto clonado em: {os.getcwd()}")

# ---- cell ----
# Setup do ambiente
print("=" * 70)
print("Instalando dependências...")
print("=" * 70)

pip_upgrade = subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], text=True, capture_output=True)
print(pip_upgrade.stdout)
if pip_upgrade.stderr:
    print(pip_upgrade.stderr)
if pip_upgrade.returncode != 0:
    raise RuntimeError(f"Falha ao atualizar pip (code={pip_upgrade.returncode}).")

pip_install = subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], text=True, capture_output=True)
print(pip_install.stdout)
if pip_install.stderr:
    print(pip_install.stderr)
if pip_install.returncode != 0:
    raise RuntimeError(f"Falha ao instalar o projeto em modo editável (code={pip_install.returncode}).")

print("\nVerificando instalação...")
status_result = subprocess.run([sys.executable, "-m", "nvs_benchmark.cli", "status"], text=True, capture_output=True)
print(status_result.stdout)
if status_result.stderr:
    print(status_result.stderr)
if status_result.returncode != 0:
    raise RuntimeError(f"Falha na verificação de status (code={status_result.returncode}).")

# ============================================================================
# Clone dos métodos necessários (D-NeRF é requerido mesmo para nerf_static)
# ============================================================================
print("\n" + "=" * 70)
print("Clonando repositórios de métodos...")
print("=" * 70)

from pathlib import Path

# Garantir que third_party existe
third_party_dir = Path("./third_party")
third_party_dir.mkdir(parents=True, exist_ok=True)
print(f"✓ Diretório third_party pronto: {third_party_dir.absolute()}")

methods_to_clone = [
    {
        "id": "d_nerf",
        "path": "./third_party/d_nerf",
        "url": "https://github.com/albertpumarola/D-NeRF.git",
        "description": "D-NeRF (suporte temporal para NeRF)",
    },
    {
        "id": "gaussian_splatting",
        "path": "./third_party/gaussian_splatting",
        "url": "https://github.com/graphdeco-inria/gaussian-splatting.git",
        "description": "3D Gaussian Splatting",
    },
]

cloned_methods = []
for method in methods_to_clone:
    method_path = Path(method["path"])
    if method_path.exists() and list(method_path.iterdir()):
        print(f"✓ {method['id']} já existe em {method['path']}")
        cloned_methods.append(method['id'])
    else:
        print(f"\nClonando {method['description']}...")
        print(f"  URL: {method['url']}")
        print(f"  Destino: {method['path']}")
        
        # Garantir que o diretório pai existe
        method_path.parent.mkdir(parents=True, exist_ok=True)
        
        clone_cmd = ["git", "clone", "--depth", "1", method["url"], method["path"]]
        print(f"$ {' '.join(clone_cmd)}")
        
        clone_result = subprocess.run(clone_cmd, text=True, capture_output=True)
        
        if clone_result.stdout:
            print(clone_result.stdout)
        
        if clone_result.returncode == 0:
            # Validar que o clone funcionou
            if method_path.exists() and list(method_path.iterdir()):
                print(f"✓ {method['id']} clonado com sucesso")
                cloned_methods.append(method['id'])
            else:
                print(f"⚠ Clone falhou (diretório vazio ou não acessível): {method['path']}")
                if clone_result.stderr:
                    print(f"  Erro: {clone_result.stderr}")
        else:
            print(f"⚠ Falha ao clonar {method['id']} (returncode={clone_result.returncode})")
            if clone_result.stderr:
                print(f"  Erro: {clone_result.stderr}")
            print(f"  (Você pode clonar manualmente depois se necessário)")

print("\n" + "=" * 70)
print(f"Métodos prontos: {', '.join(cloned_methods) if cloned_methods else '[nenhum clonado com sucesso]'}")
print("=" * 70)
print("\n✓ Ambiente pronto.")

# ---- cell ----
# Montagem opcional do Google Drive
# Mude USE_GOOGLE_DRIVE para False se quiser rodar sem persistência no Drive
print("=" * 70)
print("Configuração: Google Drive")
print("=" * 70)
print(f"Backup no Drive: {'SIM' if USE_GOOGLE_DRIVE else 'NÃO'}")

if USE_GOOGLE_DRIVE:
    drive_mod = __import__("google.colab", fromlist=["drive"])
    drive = getattr(drive_mod, "drive")
    drive.mount("/content/drive")

    from pathlib import Path

    Path(DRIVE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(DRIVE_DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(DRIVE_ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)

    # Apos montar o Drive, recarrega selecao (se existir) e persiste no arquivo do Drive
    if "load_selection_state" in globals() and "save_selection_state" in globals():
        loaded = load_selection_state()
        save_selection_state(reason="cell5-drive-sync")
        print(f"✓ Estado de seleção sincronizado com Drive ({'carregado' if loaded else 'criado'})")

    print("✓ Drive montado com sucesso.")
    print(f"✓ Cache de dados: {DRIVE_DATA_DIR}")
    print(f"✓ Cache de artefatos: {DRIVE_ARTIFACTS_DIR}")
else:
    print("  (Para ativar, mude USE_GOOGLE_DRIVE = True na Célula 2)")
print()

# ---- cell ----
# Verificação de GPU no Colab
print("=" * 70)
print("Verificação de Hardware")
print("=" * 70)

import torch
cuda_available = torch.cuda.is_available()
print(f"CUDA disponível: {'SIM ✓' if cuda_available else 'NÃO ✗'}")
if cuda_available:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memória: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠ GPU não detectada. O benchmark rodará em CPU (mais lento).")
print()

# ---- cell ----
# Download de datasets via catálogo com cache no Drive
# Se USE_GOOGLE_DRIVE=True, tenta restaurar datasets já baixados antes de instalar de novo.
print("=" * 70)
print("Preparando datasets...")
print("=" * 70)
print(f"Dataset selecionado para execução: {SELECTED_DATASET}")
print(f"Root antes do restore: {SELECTED_DATASET_ROOT}")
print()

from pathlib import Path

local_data_dir = Path("./data")
drive_data_dir = Path(DRIVE_DATA_DIR)

cache_restored = False
if USE_GOOGLE_DRIVE:
    if sync_dir_if_exists(drive_data_dir, local_data_dir):
        print(f"✓ Cache de datasets restaurado de: {drive_data_dir}")
        cache_restored = True
    else:
        print("[info] Nenhum cache de datasets encontrado no Drive. Será feita instalação local.")

# Verificar quais datasets estão disponíveis após cache restore
available_for_run = discover_available_datasets()

# DIAGNÓSTICO detalhado
print("\n[diagnóstico] Datasets descobertos:")
if available_for_run:
    for dataset_name, path in available_for_run.items():
        path_obj = Path(path)
        exists = path_obj.exists()
        has_train = (path_obj / "transforms_train.json").exists()
        has_test = (path_obj / "transforms_test.json").exists()
        print(f"  - {dataset_name}: {path}")
        print(f"      exists={exists} | train={has_train} | test={has_test}")
else:
    print("  [nenhum dataset encontrado]")

if SELECTED_DATASET in available_for_run:
    SELECTED_DATASET_ROOT = available_for_run[SELECTED_DATASET]
    # Force-normalize para eliminar qualquer duplicação do cache restaurado
    SELECTED_DATASET_ROOT = str(_normalize_dataset_path(Path(SELECTED_DATASET_ROOT)))
    print(f"\n✓ Dataset '{SELECTED_DATASET}' já disponível em: {SELECTED_DATASET_ROOT}")
    print("  (Pulando download automático)")
else:
    print(f"\n⚠ Dataset '{SELECTED_DATASET}' não encontrado. Tentando instalar...")

    install_cmd = [
        sys.executable,
        "-m",
        "nvs_benchmark.cli",
        "install",
        "--catalog-file",
        "./configs/install_catalog.json",
        "--only",
        "datasets",
        "--execute",
    ]

    install_result = run_logged(install_cmd, label="datasets-install", check=False)

    # Validar novamente após install
    available_for_run = discover_available_datasets()
    if SELECTED_DATASET in available_for_run:
        SELECTED_DATASET_ROOT = available_for_run[SELECTED_DATASET]
        print(f"✓ Root selecionado atualizado para: {SELECTED_DATASET_ROOT}")
    else:
        print(f"\n⚠ AVISO: Dataset '{SELECTED_DATASET}' ainda não disponível após install.")
        print("  Você pode:")
        print("  1. Tentar novamente (alguns downloads são intermitentes)")
        print("  2. Baixar manualmente (ver URLs em ./configs/install_catalog.json)")
        print("  3. Selecionar um dataset diferente (veja lista acima)")

# Persistir root escolhido para as próximas células
if "save_selection_state" in globals():
    save_selection_state(reason="cell8-dataset-root-refresh")

if USE_GOOGLE_DRIVE and local_data_dir.exists():
    sync_dir_if_exists(local_data_dir, drive_data_dir)
    print(f"✓ Cache de datasets sincronizado para: {drive_data_dir}")

print(f"\n[resumo] Root final selecionado: {SELECTED_DATASET_ROOT}")
print("✓ Datasets preparados.")

# ---- cell ----
# Benchmark (usa seleções da Célula 7)
print("=" * 70)
print(f"Executando benchmark: {SELECTED_METHOD} x {SELECTED_DATASET}")
print("=" * 70)

from pathlib import Path

snapshot_file = f"./artifacts/metrics/{RUN_ID}.json"

# Validação rápida do dataset antes do benchmark pesado
root_path = Path(SELECTED_DATASET_ROOT)
print("[preflight-diagnóstico]")
print(f"  cwd:                   {Path.cwd()}")
print(f"  selected_dataset:      {SELECTED_DATASET}")
print(f"  selected_dataset_root: {SELECTED_DATASET_ROOT}")
print(f"  root_exists:           {root_path.exists()}")
print(f"  has_transforms_train:  {(root_path / 'transforms_train.json').exists()}")
print(f"  has_transforms_test:   {(root_path / 'transforms_test.json').exists()}")

if not root_path.exists() and "discover_available_datasets" in globals():
    print("  [recover] tentando redescobrir root do dataset...")
    recovered = discover_available_datasets()
    print(f"  [recover] datasets encontrados: {list(recovered.keys())}")
    if SELECTED_DATASET in recovered:
        SELECTED_DATASET_ROOT = recovered[SELECTED_DATASET]
        root_path = Path(SELECTED_DATASET_ROOT)
        print(f"  [recover] root atualizado para: {SELECTED_DATASET_ROOT}")
        if "save_selection_state" in globals():
            save_selection_state(reason="cell9-preflight-recover-root")

preflight_cmd = [
    sys.executable,
    "-m",
    "nvs_benchmark.cli",
    "dataset-check",
    "--dataset",
    SELECTED_DATASET,
    "--root",
    SELECTED_DATASET_ROOT,
    "--split",
    "train",
]
preflight = run_logged(preflight_cmd, label="dataset-preflight", check=False)
if preflight.returncode != 0:
    raise RuntimeError(
        f"dataset-check falhou para {SELECTED_DATASET} em {SELECTED_DATASET_ROOT}. "
        "Corrija a raiz do dataset na Célula 7 antes de continuar."
    )

cmd = [
    sys.executable,
    "-m",
    "nvs_benchmark.cli",
    "method-run",
    "--method",
    SELECTED_METHOD,
    "--dataset",
    SELECTED_DATASET,
    "--root",
    SELECTED_DATASET_ROOT,
    "--split",
    "train",
    "--preset",
    SELECTED_PRESET,
    "--output-dir",
    "./artifacts",
    "--log-dir",
    "./logs",
    "--compute-metrics",
    "--snapshot-file",
    snapshot_file,
    "--append-snapshot",
]

if STRICT_RESULTS:
    cmd.extend(["--strict-results", "--min-required-pairs", str(MIN_REQUIRED_PAIRS)])

result = run_logged(cmd, label="method-run", check=False)
if result.returncode != 0:
    raise RuntimeError(
        f"method-run falhou para {SELECTED_METHOD} x {SELECTED_DATASET} (code={result.returncode}). "
        "Veja os logs acima para a causa exata."
    )

print(f"\n✓ Snapshot gerado em: {snapshot_file}")

# ---- cell ----
# Gerar relatorio HTML (usa seleções da Célula 7)
print("=" * 70)
print("Gerando relatório HTML...")
print("=" * 70)

report_name = f"{RUN_ID}_{SELECTED_METHOD}_{SELECTED_DATASET}_report"

cmd = [
    sys.executable, "-m", "nvs_benchmark.cli", "report-generate",
    "--snapshot-file", snapshot_file,
    "--output-dir", "./artifacts/reports",
    "--report-name", report_name,
    "--log-dir", "./logs",
]

if not GENERATE_PDF:
    cmd.append("--no-pdf")

if STRICT_RESULTS:
    cmd.extend(["--strict-snapshot", "--min-methods", "1", "--require-finite-metrics"])

report_result = run_logged(cmd, label="report-generate", check=False)
if report_result.returncode != 0:
    raise RuntimeError("Falha ao gerar o relatório consolidado.")

report_html = f"./artifacts/reports/{report_name}.html"

if USE_GOOGLE_DRIVE:
    sync_dir_if_exists(Path("./artifacts"), Path(DRIVE_ARTIFACTS_DIR))
    print(f"✓ Artefatos sincronizados para: {DRIVE_ARTIFACTS_DIR}")

print(f"\n✓ Relatório HTML gerado: {report_html}")

# ---- cell ----
# Exibir o relatório no notebook
print("=" * 70)
print("Carregando relatório...")
print("=" * 70)

from IPython.display import IFrame, display
import os

if not os.path.exists(report_html):
    print(f"✗ Arquivo não encontrado: {report_html}")
    print("  Verifique se a célula anterior executou sem erros.")
else:
    print(f"✓ Abrindo: {report_html}\n")
    display(IFrame(src=report_html, width=1200, height=700))

# ---- cell ----
# Compactar e baixar artefatos
print("=" * 70)
print("Preparando download de artefatos...")
print("=" * 70)

import pathlib
colab_files_mod = __import__("google.colab", fromlist=["files"])
files = getattr(colab_files_mod, "files")

zip_path = "/content/nvs_benchmark_artifacts"
print(f"Compactando {REPO_DIR}/artifacts...")
archive_file = shutil.make_archive(zip_path, "zip", REPO_DIR, "artifacts")
print(f"Arquivo gerado: {archive_file}")

if USE_GOOGLE_DRIVE:
    drive_archive_dir = Path(DRIVE_ARTIFACTS_DIR) / "archives"
    drive_archive_dir.mkdir(parents=True, exist_ok=True)
    drive_archive_file = drive_archive_dir / Path(archive_file).name
    shutil.copy2(archive_file, drive_archive_file)
    print(f"✓ ZIP salvo no Drive em: {drive_archive_file}")

if pathlib.Path(archive_file).exists():
    print(f"Tamanho: {pathlib.Path(archive_file).stat().st_size / (1024*1024):.1f} MB")
    print("\n↓ Iniciando download...")
    files.download(archive_file)
else:
    print("✗ Arquivo não encontrado!")

# ---- cell ----
# Backup opcional dos artefatos no Google Drive
print("=" * 70)
print("Backup de resultados")
print("=" * 70)

if USE_GOOGLE_DRIVE:
    print(f"Backup: SIM → {DRIVE_OUTPUT_DIR}")
    os.makedirs(DRIVE_OUTPUT_DIR, exist_ok=True)
    metrics_dst = os.path.join(DRIVE_OUTPUT_DIR, "metrics")
    reports_dst = os.path.join(DRIVE_OUTPUT_DIR, "reports")

    if os.path.exists(metrics_dst):
        shutil.rmtree(metrics_dst)
    if os.path.exists(reports_dst):
        shutil.rmtree(reports_dst)

    print(f"Copiando métricas...")
    shutil.copytree("./artifacts/metrics", metrics_dst)
    print(f"Copiando relatórios...")
    shutil.copytree("./artifacts/reports", reports_dst)

    drive_data_dst = Path(DRIVE_DATA_DIR)
    if Path("./data").exists():
        if drive_data_dst.exists():
            shutil.rmtree(drive_data_dst)
        print("Sincronizando datasets para o Drive...")
        shutil.copytree("./data", drive_data_dst)

    print(f"✓ Backup concluido em: {DRIVE_OUTPUT_DIR}")
else:
    print("Backup: NÃO")
    print("  Para fazer backup no Drive, mude USE_GOOGLE_DRIVE = True na Célula 2")
