# Auto-generated script from notebook: notebooks\nvs_benchmark_local_blender_synthetic.ipynb

# ---- cell ----
# Configuracao local e helpers
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Repositorio e caminhos locais
# ---------------------------------------------------------------------------
REPO_URL = "https://github.com/PedroHeinrichSP/TCC-Source-Code.git"
BRANCH = "update"
REPO_DIR_NAME = "TCC"
AUTO_CLONE_REPO = True

def looks_like_project_root(path: Path) -> bool:
    return (path / "pyproject.toml").exists() and (path / "configs" / "install_catalog.json").exists()

def find_project_root(start: Path) -> Path | None:
    candidates = []
    current = start.resolve()
    candidates.append(current.parent if current.name == "notebooks" else current)
    candidates.extend([current / REPO_DIR_NAME, current / "TCC-Source-Code"])
    candidates.extend(current.parents)
    for candidate in candidates:
        if looks_like_project_root(candidate):
            return candidate.resolve()
    return None

START_DIR = Path.cwd().resolve()
PROJECT_ROOT = find_project_root(START_DIR)

if PROJECT_ROOT is None:
    if not AUTO_CLONE_REPO:
        raise RuntimeError("Repositorio nao encontrado. Ative AUTO_CLONE_REPO=True ou execute este notebook dentro da raiz do projeto.")
    clone_target = (START_DIR / REPO_DIR_NAME).resolve()
    if clone_target.exists() and any(clone_target.iterdir()):
        raise RuntimeError(f"Destino de clone ja existe e nao esta vazio: {clone_target}")
    print("=" * 70)
    print("Repositorio nao encontrado. Clonando projeto...")
    print("=" * 70)
    clone_cmd = ["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(clone_target)]
    print(f"$ {' '.join(clone_cmd)}")
    clone_result = subprocess.run(clone_cmd, text=True, capture_output=True)
    if clone_result.stdout:
        print(clone_result.stdout)
    if clone_result.stderr:
        print(clone_result.stderr)
    if clone_result.returncode != 0:
        raise RuntimeError(f"Falha ao clonar repositorio (code={clone_result.returncode}).")
    PROJECT_ROOT = clone_target

os.chdir(PROJECT_ROOT)

ENVIRONMENT = "local"
DATASET_ID = "blender_synthetic"
DEFAULT_DATASET_ROOT = Path("./data/blender_synthetic/nerf_synthetic/lego")
CATALOG_FILE = Path("./configs/install_catalog.json")
BLENDER_ONLY_CATALOG_FILE = Path("./notebooks/artifacts/install_catalog_blender_only.json")

# ---------------------------------------------------------------------------
# Selecao do benchmark
# ---------------------------------------------------------------------------
SELECTED_METHOD = "nerf_static"  # Opcoes comuns: nerf_static, nerf_dynamic, gs_static, gs_dynamic
SELECTED_PRESET = "quick"        # Opcoes: smoke, quick, preview, standard, full
RUN_MODE = "full"                # full | quick_check
STRICT_RESULTS = True
GENERATE_PDF = False
MIN_REQUIRED_PAIRS = 1

# Fluxo padrao: clonar/localizar repo -> instalar projeto -> preparar dataset -> rodar benchmark.
INSTALL_PROJECT = True            # True: executa pip install -e . depois de entrar no repo
INSTALL_DATASET_IF_MISSING = True # True: baixa Blender Synthetic via catalogo se nao existir
CLONE_THIRD_PARTY_IF_MISSING = False

STATE_FILE = Path("./notebooks/artifacts/local_blender_synthetic_selection.json")

def generate_run_id(environment: str, preset: str, method: str, dataset: str) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return f"{environment}_{preset}_{method}_{dataset}_{stamp}"

def run_logged(cmd, label: str, check: bool = True, cwd: Path | str | None = None):
    print("\n" + "=" * 70)
    print(f"[{label}] $ {' '.join(map(str, cmd))}")
    print("=" * 70)
    result = subprocess.run(
        [str(part) for part in cmd],
        text=True,
        capture_output=True,
        cwd=str(cwd) if cwd else None,
    )
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if check and result.returncode != 0:
        raise RuntimeError(f"Comando '{label}' falhou com code={result.returncode}")
    return result

def normalize_blender_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "transforms_train.json").exists():
        return path
    lego = path / "nerf_synthetic" / "lego"
    if (lego / "transforms_train.json").exists():
        return lego.resolve()
    nested = list(path.glob("**/transforms_train.json"))
    if nested:
        return nested[0].parent.resolve()
    return path

def discover_blender_synthetic_root() -> Path | None:
    candidates = [
        DEFAULT_DATASET_ROOT,
        Path("./data/blender_synthetic/lego"),
        Path("./data/blender_synthetic"),
    ]
    for candidate in candidates:
        normalized = normalize_blender_root(candidate)
        if (normalized / "transforms_train.json").exists():
            return normalized
    return None

def write_blender_only_catalog() -> Path:
    catalog = json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
    datasets = [item for item in catalog.get("datasets", []) if item.get("id") == DATASET_ID]
    if not datasets:
        raise RuntimeError(f"Dataset {DATASET_ID} nao encontrado em {CATALOG_FILE}")
    BLENDER_ONLY_CATALOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    filtered = {"datasets": datasets, "methods": [], "notes": ["Catalogo temporario gerado pelo notebook local."]}
    BLENDER_ONLY_CATALOG_FILE.write_text(json.dumps(filtered, indent=2, ensure_ascii=True), encoding="utf-8")
    return BLENDER_ONLY_CATALOG_FILE

def save_state(run_id: str, dataset_root: Path) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "environment": ENVIRONMENT,
        "selected_method": SELECTED_METHOD,
        "selected_dataset": DATASET_ID,
        "selected_preset": SELECTED_PRESET,
        "run_mode": RUN_MODE,
        "strict_results": STRICT_RESULTS,
        "generate_pdf": GENERATE_PDF,
        "min_required_pairs": MIN_REQUIRED_PAIRS,
        "dataset_root": str(dataset_root),
        "run_id": run_id,
    }
    STATE_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")

RUN_ID = generate_run_id(ENVIRONMENT, SELECTED_PRESET, SELECTED_METHOD, DATASET_ID)
SELECTED_DATASET_ROOT = discover_blender_synthetic_root() or DEFAULT_DATASET_ROOT.resolve()

print("=" * 70)
print("CONFIGURACAO LOCAL")
print("=" * 70)
print(f"Inicio:         {START_DIR}")
print(f"Projeto:        {PROJECT_ROOT}")
print(f"Repo URL:       {REPO_URL}")
print(f"Branch:         {BRANCH}")
print(f"Python:         {sys.executable}")
print(f"Ambiente:       {ENVIRONMENT}")
print(f"Metodo:         {SELECTED_METHOD}")
print(f"Dataset:        {DATASET_ID} (fixo)")
print(f"Dataset root:   {SELECTED_DATASET_ROOT}")
print(f"Preset:         {SELECTED_PRESET}")
print(f"RUN_ID:         {RUN_ID}")
print(f"Estado:         {STATE_FILE}")
print("=" * 70)
save_state(RUN_ID, Path(SELECTED_DATASET_ROOT))

# ---- cell ----
# Setup do ambiente local
print("=" * 70)
print("Setup local")
print("=" * 70)

print(f"Diretorio atual: {Path.cwd()}")
if not looks_like_project_root(Path.cwd()):
    raise RuntimeError("Diretorio atual nao parece ser a raiz do projeto. Execute a celula de configuracao novamente.")

if INSTALL_PROJECT:
    run_logged([sys.executable, "-m", "pip", "install", "-e", "."], label="pip-install-editable")
else:
    print("Pulando pip install -e . (INSTALL_PROJECT=False)")

# Status do pacote/CLI. Se falhar por pacote nao instalado, ative INSTALL_PROJECT=True ou execute no venv do projeto.
status = run_logged([sys.executable, "-m", "nvs_benchmark.cli", "status"], label="cli-status", check=False)
if status.returncode != 0:
    raise RuntimeError("Falha ao executar nvs_benchmark.cli status. Ative INSTALL_PROJECT=True ou selecione o kernel/venv correto.")

third_party_dir = Path("./third_party")
third_party_dir.mkdir(parents=True, exist_ok=True)
methods_to_clone = [
    {"id": "d_nerf", "path": third_party_dir / "d_nerf", "url": "https://github.com/albertpumarola/D-NeRF.git"},
    {"id": "gaussian_splatting", "path": third_party_dir / "gaussian_splatting", "url": "https://github.com/graphdeco-inria/gaussian-splatting.git"},
]

if CLONE_THIRD_PARTY_IF_MISSING:
    for method in methods_to_clone:
        if method["path"].exists() and any(method["path"].iterdir()):
            print(f"OK: {method['id']} ja existe em {method['path']}")
            continue
        clone_cmd = ["git", "clone", "--depth", "1", "--recursive", method["url"], str(method["path"])]
        run_logged(clone_cmd, label=f"clone-{method['id']}", check=False)
else:
    print("Pulando clone de third_party (CLONE_THIRD_PARTY_IF_MISSING=False)")

print("\nAmbiente pronto para as proximas celulas.")

# ---- cell ----
# Dependencias compiladas para Gaussian Splatting no Colab
print("=" * 70)
print("Instalando dependencias compiladas dos metodos")
print("=" * 70)

compiled_method_status: dict[str, bool] = {}


def print_tail(text: str, *, lines: int = 40, prefix: str = "") -> None:
    if not text:
        return
    for line in text.splitlines()[-lines:]:
        if line.strip():
            print(f"{prefix}{line}" if prefix else line)


def install_cuda_submodule(label: str, package_path: Path, *, required: bool = True) -> bool:
    if not package_path.exists():
        level = "warn" if required else "info"
        print(f"[{level}] {label}: diretorio nao encontrado em {package_path}")
        return not required

    if label == "simple-knn":
        package_dir = package_path / "simple_knn"
        package_dir.mkdir(parents=True, exist_ok=True)
        init_file = package_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text("", encoding="utf-8")
            print(f"[info] {label}: criado arquivo de pacote em {init_file}")

    install_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-build-isolation",
        "-e",
        str(package_path),
    ]
    print(f"\nInstalando {label} em: {package_path}")
    print(f"$ {' '.join(install_cmd)}")
    result = subprocess.run(install_cmd, text=True, capture_output=True)
    print_tail(result.stdout, lines=40)
    if result.returncode == 0:
        print(f"OK: {label} instalado com sucesso")
        return True

    print(f"AVISO: falha ao instalar {label} (code={result.returncode})")
    print_tail(result.stderr, lines=30, prefix="  ")
    return False


gs_path = Path("./third_party/gaussian_splatting")
if gs_path.exists():
    print(f"\nGaussian Splatting encontrado em: {gs_path}")
    try:
        submod = subprocess.run(["git", "-C", str(gs_path), "submodule", "status"], text=True, capture_output=True)
        if submod.stdout:
            print("[git submodule status]\n" + submod.stdout)
    except Exception:
        pass

    if (gs_path / "train.py").exists() and (gs_path / "render.py").exists():
        print("[info] Repositorio base do Gaussian Splatting presente. Ele nao e um pacote pip instalavel.")
        compiled_method_status["gaussian_splatting_repo"] = True
    else:
        print("[warn] Repositorio Gaussian Splatting incompleto. train.py/render.py nao encontrados.")
        compiled_method_status["gaussian_splatting_repo"] = False

    required_submodules = [
        ("diff-gaussian-rasterization", gs_path / "submodules" / "diff-gaussian-rasterization"),
        ("simple-knn", gs_path / "submodules" / "simple-knn"),
    ]
    optional_submodules = [
        ("fused-ssim", gs_path / "submodules" / "fused-ssim"),
    ]

    required_ok = True
    for label, package_path in required_submodules:
        required_ok = install_cuda_submodule(label, package_path, required=True) and required_ok

    for label, package_path in optional_submodules:
        install_cuda_submodule(label, package_path, required=False)

    probe_cmd = [
        sys.executable,
        "-c",
        "import diff_gaussian_rasterization, simple_knn._C; print('ok')",
    ]
    probe = subprocess.run(probe_cmd, text=True, capture_output=True, cwd=str(gs_path))
    probe_ok = probe.returncode == 0 and "ok" in probe.stdout
    if not probe_ok:
        print("[warn] Probe final das extensoes do gs_static falhou.")
        print_tail(probe.stderr, lines=20, prefix="  ")
    compiled_method_status["gs_static_extensions"] = required_ok and probe_ok
else:
    print("[info] Repositorio gaussian_splatting ausente; pulando compilacao de extensoes.")
    compiled_method_status["gaussian_splatting_repo"] = False
    compiled_method_status["gs_static_extensions"] = False

print("\n" + "=" * 70)
print(f"Repositorios prontos: {', '.join([m['id'] for m in methods_to_clone if Path(m['path']).exists()]) or '[nenhum]'}")
print(
    "Extensoes do gs_static: "
    + ("prontas" if compiled_method_status.get("gs_static_extensions") else "pendentes")
)
print("=" * 70)
if compiled_method_status.get("gs_static_extensions"):
    print("\nOK: ambiente pronto.")
else:
    print("\nAVISO: ambiente base pronto, mas gs_static ainda nao esta compilado corretamente.")

# ---- cell ----
# Verificacao de hardware
print("=" * 70)
print("Verificacao de hardware")
print("=" * 70)

try:
    import torch
    cuda_available = torch.cuda.is_available()
    print(f"CUDA disponivel: {'SIM' if cuda_available else 'NAO'}")
    if cuda_available:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memoria: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("GPU nao detectada. O benchmark pode rodar em CPU, mas sera mais lento.")
except Exception as exc:
    print(f"Nao foi possivel consultar torch/CUDA: {exc}")

# ---- cell ----
# Preparar apenas o dataset Blender Synthetic
print("=" * 70)
print("Preparando dataset Blender Synthetic")
print("=" * 70)

import shutil
import ssl
import urllib.request
import zipfile

SELECTED_DATASET_ROOT = discover_blender_synthetic_root()

if SELECTED_DATASET_ROOT is None:
    print(f"Dataset nao encontrado em {DEFAULT_DATASET_ROOT}")
    if INSTALL_DATASET_IF_MISSING:
        if not CATALOG_FILE.exists():
            raise FileNotFoundError(f"Catalogo nao encontrado: {CATALOG_FILE}")

        blender_root = Path("./data/blender_synthetic")
        download_dir = Path("./artifacts/downloads")
        archive_path = download_dir / "nerf_example_data.zip"
        download_dir.mkdir(parents=True, exist_ok=True)
        blender_root.mkdir(parents=True, exist_ok=True)

        def download_with_progress(url: str, destination: Path) -> Path:
            print(f"Baixando dataset de {url}")
            context = ssl.create_default_context()
            if destination.exists():
                destination.unlink()
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, context=context) as response, destination.open("wb") as output_file:
                total_size_header = response.headers.get("Content-Length")
                total_size = int(total_size_header) if total_size_header and total_size_header.isdigit() else None
                downloaded_bytes = 0
                last_report = time.time()
                started_at = last_report
                chunk_size = 1024 * 1024
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    output_file.write(chunk)
                    downloaded_bytes += len(chunk)
                    now = time.time()
                    if now - last_report >= 5 or (total_size is not None and downloaded_bytes >= total_size):
                        elapsed = max(now - started_at, 0.001)
                        speed_mb_s = downloaded_bytes / elapsed / (1024 * 1024)
                        if total_size is not None:
                            pct = downloaded_bytes * 100 / total_size
                            print(f"  {downloaded_bytes / (1024 * 1024):.1f} MB / {total_size / (1024 * 1024):.1f} MB ({pct:.1f}%) - {speed_mb_s:.1f} MB/s")
                        else:
                            print(f"  {downloaded_bytes / (1024 * 1024):.1f} MB baixados - {speed_mb_s:.1f} MB/s")
                        last_report = now
            return destination

        print("Iniciando download manual com progresso. Esta celula pode demorar alguns minutos, mas nao deve ficar silenciosa.")
        try:
            download_with_progress("https://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip", archive_path)
            print(f"Extraindo {archive_path} em {blender_root}")
            with zipfile.ZipFile(archive_path) as zip_file:
                zip_file.extractall(blender_root)
        finally:
            if archive_path.exists():
                archive_path.unlink()

        SELECTED_DATASET_ROOT = discover_blender_synthetic_root()
    else:
        raise FileNotFoundError("Dataset Blender Synthetic ausente. Ative INSTALL_DATASET_IF_MISSING=True ou informe os arquivos localmente.")

if SELECTED_DATASET_ROOT is None:
    raise FileNotFoundError("Dataset instalado, mas transforms_train.json nao foi encontrado.")

SELECTED_DATASET_ROOT = normalize_blender_root(Path(SELECTED_DATASET_ROOT))
print(f"Dataset root final: {SELECTED_DATASET_ROOT}")
print(f"transforms_train.json: {(SELECTED_DATASET_ROOT / 'transforms_train.json').exists()}")
print(f"transforms_test.json:  {(SELECTED_DATASET_ROOT / 'transforms_test.json').exists()}")

preflight_cmd = [
    sys.executable,
    "-m",
    "nvs_benchmark.cli",
    "dataset-check",
    "--dataset",
    DATASET_ID,
    "--root",
    str(SELECTED_DATASET_ROOT),
    "--split",
    "train",
]
preflight = run_logged(preflight_cmd, label="dataset-check", check=False)
if preflight.returncode != 0:
    raise RuntimeError(f"dataset-check falhou para {DATASET_ID} em {SELECTED_DATASET_ROOT}")

save_state(RUN_ID, SELECTED_DATASET_ROOT)
print("Dataset pronto.")

# ---- cell ----
# Executar benchmark
print("=" * 70)
print(f"Executando benchmark: {SELECTED_METHOD} x {DATASET_ID}")
print("=" * 70)

Path("./artifacts/metrics").mkdir(parents=True, exist_ok=True)
Path("./logs").mkdir(parents=True, exist_ok=True)
snapshot_file = Path("./artifacts/metrics") / f"{RUN_ID}.json"

if RUN_MODE == "quick_check":
    print("RUN_MODE=quick_check: pulando treinamento; dataset-check ja foi executado.")
else:
    cmd = [
        sys.executable,
        "-m",
        "nvs_benchmark.cli",
        "method-run",
        "--method",
        SELECTED_METHOD,
        "--dataset",
        DATASET_ID,
        "--root",
        str(SELECTED_DATASET_ROOT),
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
        str(snapshot_file),
        "--append-snapshot",
    ]
    if STRICT_RESULTS:
        cmd.extend(["--strict-results", "--min-required-pairs", str(MIN_REQUIRED_PAIRS)])

    result = run_logged(cmd, label="method-run", check=False)
    if result.returncode != 0:
        raise RuntimeError(f"method-run falhou para {SELECTED_METHOD} x {DATASET_ID} (code={result.returncode})")

print(f"Snapshot: {snapshot_file}")

# ---- cell ----
# Gerar relatorio HTML
print("=" * 70)
print("Gerando relatorio HTML")
print("=" * 70)

if RUN_MODE == "quick_check":
    raise RuntimeError("RUN_MODE=quick_check nao gera snapshot de benchmark. Mude RUN_MODE para 'full' para gerar relatorio.")

report_name = f"{RUN_ID}_{SELECTED_METHOD}_{DATASET_ID}_report"
cmd = [
    sys.executable,
    "-m",
    "nvs_benchmark.cli",
    "report-generate",
    "--snapshot-file",
    str(snapshot_file),
    "--output-dir",
    "./artifacts/reports",
    "--report-name",
    report_name,
    "--log-dir",
    "./logs",
]
if not GENERATE_PDF:
    cmd.append("--no-pdf")
if STRICT_RESULTS:
    cmd.extend(["--strict-snapshot", "--min-methods", "1", "--require-finite-metrics"])

report_result = run_logged(cmd, label="report-generate", check=False)
if report_result.returncode != 0:
    raise RuntimeError("Falha ao gerar o relatorio consolidado.")

report_html = Path("./artifacts/reports") / f"{report_name}.html"
print(f"Relatorio HTML: {report_html.resolve()}")

# ---- cell ----
# Exibir relatorio no notebook
from IPython.display import IFrame, display

if not Path(report_html).exists():
    print(f"Arquivo nao encontrado: {report_html}")
else:
    display(IFrame(src=str(report_html), width=1200, height=700))

# ---- cell ----
# Compactar artefatos localmente
import shutil

archive_base = Path("./artifacts") / f"{RUN_ID}_artifacts"
archive_file = shutil.make_archive(str(archive_base), "zip", "./artifacts")
print(f"ZIP gerado: {Path(archive_file).resolve()}")
print(f"Tamanho: {Path(archive_file).stat().st_size / (1024 * 1024):.1f} MB")
