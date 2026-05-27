# Auto-generated script from notebook: notebooks\nvs_benchmark_local_pc.ipynb

# ---- cell ----
# Configuracao local e helpers
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional

REPO_URL = "https://github.com/PedroHeinrichSP/TCC-Source-Code.git"
BRANCH = "update"
REPO_DIR_NAME = "TCC"
AUTO_CLONE_REPO = True

ENVIRONMENT = "local_pc"
INSTALL_PROJECT = True
UPGRADE_PIP = True
CLONE_THIRD_PARTY_IF_MISSING = True
INSTALL_COMPILED_METHOD_DEPS = True
INSTALL_DATASET_IF_MISSING = True

SELECTED_METHOD = "nerf_static"
SELECTED_DATASET = "blender_synthetic"
SELECTED_PRESET = "quick"
RUN_MODE = "full"  # full | quick_check
STRICT_RESULTS = True
GENERATE_PDF = False
MIN_REQUIRED_PAIRS = 1
LOAD_SAVED_SELECTION = False
ENABLE_NERF_MEMORY_TUNING = True
FALLBACK_TO_SMOKE_ON_OOM = True

SKIP_TRAINING = False
LOCAL_ARTIFACTS_ZIP = ""  # Ex.: r"C:\\Users\\voce\\Downloads\\artifacts.zip"
SELECTED_DATASET_ROOT = ""  # Ex.: r"D:\\datasets\\blender_synthetic\\nerf_synthetic\\lego"
SELECTED_SCENE_NAME = "garden"  # Cena padrao para datasets multi-cena como mipnerf360
SELECTED_TT_SCENE_NAME = "train"  # Cena padrao para Tanks and Temples
EXTRA_JSON = ""  # JSON inline opcional para --extra-json
USE_LOCAL_BACKUP = False
BACKUP_DATASETS = False
LOCAL_BACKUP_DIR = Path.home() / "NVS_Benchmark_Backup"

CATALOG_FILE = Path("./configs/install_catalog.json")
TEMP_CATALOG_FILE = Path("./notebooks/artifacts/local_selected_dataset_catalog.json")
STATE_FILE = Path("./notebooks/artifacts/local_pc_selection.json")
DATA_SEARCH_ROOTS = [
    Path("./data"),
    Path.home() / "datasets" / "nvs_benchmark",
]
MANUAL_DATASET_IDS = {"tanks_and_temples"}
DISCOVERY_DATASET_IDS = ["blender_synthetic", "d_nerf", "mipnerf360", "tanks_and_temples"]

UPLOADED_CHECKPOINT = None
UPLOADED_RENDERS_DIR = None
UPLOADED_TRAIN_SECONDS = 0.0
UPLOADED_INFERENCE_SECONDS = 0.0
snapshot_file = None
report_html = None


def looks_like_project_root(path: Path) -> bool:
    return (path / "pyproject.toml").exists() and (path / "configs" / "install_catalog.json").exists()



def find_project_root(start: Path) -> Path | None:
    current = start.resolve()
    candidates = [current.parent if current.name == "notebooks" else current]
    candidates.extend([current / REPO_DIR_NAME, current / "TCC-Source-Code"])
    candidates.extend(current.parents)
    for candidate in candidates:
        if looks_like_project_root(candidate):
            return candidate.resolve()
    return None



def generate_run_id(environment: str, preset: str, method: str, dataset: str) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dataset_clean = dataset.split("/")[-1].replace(" ", "_").replace("-", "_").lower()
    return f"{environment}_{preset}_{method}_{dataset_clean}_{stamp}"



def run_logged(cmd, label: str, check: bool = True, cwd: Optional[Path | str] = None):
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



def extract_zip_artifacts(zip_path: str, extract_to: str) -> tuple[str, str]:
    extract_path = Path(extract_to)
    if extract_path.exists():
        shutil.rmtree(extract_path)
    extract_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    checkpoint_patterns = [
        "checkpoint.pth",
        "checkpoint.pkl",
        "checkpoint.ckpt",
        "checkpoint.pt",
        "model.pth",
        "final.pth",
    ]
    checkpoint_path = None
    for pattern in checkpoint_patterns:
        matches = list(extract_path.glob(f"**/{pattern}"))
        if matches:
            checkpoint_path = str(matches[0].resolve())
            break

    if checkpoint_path is None:
        for ext in ["*.pth", "*.pkl", "*.ckpt", "*.pt"]:
            matches = list(extract_path.glob(f"**/{ext}"))
            if matches:
                checkpoint_path = str(matches[0].resolve())
                break

    if checkpoint_path is None:
        raise FileNotFoundError("Nenhum arquivo de checkpoint foi encontrado no ZIP local.")

    renders_dir = None
    for candidate in ["renders", "output", "images", "images_val"]:
        candidate_path = extract_path / candidate
        if candidate_path.exists() and candidate_path.is_dir() and list(candidate_path.glob("*.png")):
            renders_dir = str(candidate_path.resolve())
            break

    if renders_dir is None:
        raise FileNotFoundError("Nenhum diretorio com PNGs foi encontrado no ZIP local.")

    return checkpoint_path, renders_dir



def _collapse_duplicate_segments(path: Path) -> Path:
    parts = list(path.parts)
    if not parts:
        return path
    collapsed = [parts[0]]
    for part in parts[1:]:
        if part != collapsed[-1]:
            collapsed.append(part)
    return Path(*collapsed)



def _dir_has_files(path: Path) -> bool:
    try:
        return path.exists() and path.is_dir() and any(path.iterdir())
    except OSError:
        return False



def _path_has_dataset_markers(path: Path) -> bool:
    return (
        (path / "transforms_train.json").exists()
        or (path / "poses_bounds.npy").exists()
        or _dir_has_files(path / "images")
        or _dir_has_files(path / "sparse" / "0")
    )



def normalize_dataset_path(path: Path) -> Path:
    path = Path(path).expanduser().resolve()
    if _path_has_dataset_markers(path):
        return path
    collapsed_path = _collapse_duplicate_segments(path)
    if _path_has_dataset_markers(collapsed_path):
        return collapsed_path.resolve()

    nested_candidates = []
    nested_candidates.extend(candidate.parent.resolve() for candidate in collapsed_path.glob("**/transforms_train.json"))
    nested_candidates.extend(candidate.parent.resolve() for candidate in collapsed_path.glob("**/poses_bounds.npy"))
    nested_candidates.extend(candidate.parent.resolve() for candidate in collapsed_path.glob("**/images") if _dir_has_files(candidate))
    nested_candidates.extend(
        candidate.parent.parent.resolve() for candidate in collapsed_path.glob("**/sparse/0") if _dir_has_files(candidate)
    )
    unique_candidates = sorted(
        {candidate for candidate in nested_candidates},
        key=lambda candidate: (len(candidate.parts), str(candidate).lower()),
    )
    if len(unique_candidates) == 1:
        return unique_candidates[0]
    return collapsed_path.resolve()



def _search_candidates_for_marker(base: Path, marker: str) -> list[Path]:
    if marker == "transforms_train.json":
        return [candidate.parent.resolve() for candidate in base.glob("**/transforms_train.json")]
    if marker == "poses_bounds.npy":
        return [candidate.parent.resolve() for candidate in base.glob("**/poses_bounds.npy")]
    if marker == "images":
        return [candidate.parent.resolve() for candidate in base.glob("**/images") if _dir_has_files(candidate)]
    if marker == "sparse/0":
        return [candidate.parent.parent.resolve() for candidate in base.glob("**/sparse/0") if _dir_has_files(candidate)]
    if marker == "image_sets":
        return [candidate.resolve() for candidate in base.glob("**/image_sets/*") if candidate.is_dir() and _dir_has_files(candidate)]
    return []



def discover_dataset_candidates(search_roots: list[Path], dataset_id: str) -> list[Path]:
    markers_by_dataset = {
        "blender_synthetic": ("transforms_train.json",),
        "d_nerf": ("transforms_train.json",),
        "mipnerf360": ("transforms_train.json", "poses_bounds.npy", "sparse/0"),
        "tanks_and_temples": ("image_sets", "transforms_train.json", "poses_bounds.npy", "images", "sparse/0"),
    }
    markers = markers_by_dataset.get(dataset_id, ("transforms_train.json", "poses_bounds.npy", "images", "sparse/0"))
    candidates: list[Path] = []
    for base in search_roots:
        base = Path(base).expanduser().resolve()
        if not base.exists():
            continue
        for marker in markers:
            candidates.extend(_search_candidates_for_marker(base, marker))
    unique_candidates = sorted(
        {candidate for candidate in candidates},
        key=lambda candidate: (len(candidate.parts), str(candidate).lower()),
    )
    return unique_candidates



def choose_preferred_candidate(dataset_id: str, candidates: list[Path]) -> Path:
    if not candidates:
        raise ValueError(f"Nenhum candidato encontrado para {dataset_id}")
    if dataset_id == "mipnerf360":
        preferred_scene = SELECTED_SCENE_NAME.strip().lower()
        if preferred_scene:
            for candidate in candidates:
                candidate_name = candidate.name.lower()
                candidate_text = str(candidate).lower()
                if candidate_name == preferred_scene or preferred_scene in candidate_text:
                    return candidate
    if dataset_id == "tanks_and_temples":
        preferred_scene = SELECTED_TT_SCENE_NAME.strip().lower()
        if preferred_scene:
            for candidate in candidates:
                candidate_name = candidate.name.lower()
                candidate_text = str(candidate).lower()
                if candidate_name == preferred_scene or preferred_scene in candidate_text:
                    return candidate
    return candidates[0]


def resolve_saved_scene_root(dataset_id: str, root_value: str) -> str:
    root_path = Path(root_value).expanduser().resolve()
    if _path_has_dataset_markers(root_path):
        return str(root_path)
    search_roots = [root_path]
    if dataset_id == "tanks_and_temples":
        search_roots.append(root_path / "image_sets")
    candidates = discover_dataset_candidates(search_roots, dataset_id)
    if candidates:
        return str(normalize_dataset_path(choose_preferred_candidate(dataset_id, candidates)))
    return str(root_path)



def discover_available_datasets(search_roots: list[Path]) -> dict[str, str]:
    discovered: dict[str, str] = {}
    for dataset_id in DISCOVERY_DATASET_IDS:
        candidates = discover_dataset_candidates(search_roots, dataset_id)
        if len(candidates) == 1:
            discovered[dataset_id] = str(candidates[0])
        elif candidates and dataset_id in {"mipnerf360", "tanks_and_temples"}:
            discovered[dataset_id] = str(normalize_dataset_path(choose_preferred_candidate(dataset_id, candidates)))
    return discovered



def write_single_dataset_catalog(dataset_id: str) -> Path:
    catalog_payload = json.loads(CATALOG_FILE.read_text(encoding="utf-8"))
    datasets = [item for item in catalog_payload.get("datasets", []) if item.get("id") == dataset_id]
    if not datasets:
        raise RuntimeError(f"Dataset {dataset_id} nao encontrado em {CATALOG_FILE}")
    if dataset_id == "tanks_and_temples":
        selected_scene = SELECTED_TT_SCENE_NAME.strip().lower()
        if selected_scene:
            dataset_item = dict(datasets[0])
            dataset_item["command"] = (
                "python ./scripts/download_tanks_and_temples.py "
                "--pathname ./data/tanks_and_temples "
                "--source inria "
                f"--scene {selected_scene}"
            )
            datasets = [dataset_item]
    TEMP_CATALOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    filtered = {
        "datasets": datasets,
        "methods": [],
        "notes": [f"Catalogo temporario gerado pelo notebook local para {dataset_id}."],
    }
    TEMP_CATALOG_FILE.write_text(json.dumps(filtered, indent=2, ensure_ascii=True), encoding="utf-8")
    return TEMP_CATALOG_FILE



def save_state(dataset_root: str) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "environment": ENVIRONMENT,
        "selected_method": SELECTED_METHOD,
        "selected_dataset": SELECTED_DATASET,
        "selected_preset": SELECTED_PRESET,
        "run_mode": RUN_MODE,
        "strict_results": STRICT_RESULTS,
        "generate_pdf": GENERATE_PDF,
        "min_required_pairs": MIN_REQUIRED_PAIRS,
        "skip_training": SKIP_TRAINING,
        "dataset_root": dataset_root,
        "local_artifacts_zip": LOCAL_ARTIFACTS_ZIP,
    }
    STATE_FILE.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")



def load_state() -> None:
    global SELECTED_METHOD, SELECTED_DATASET, SELECTED_PRESET, RUN_MODE
    global STRICT_RESULTS, GENERATE_PDF, MIN_REQUIRED_PAIRS, SKIP_TRAINING
    global SELECTED_DATASET_ROOT, LOCAL_ARTIFACTS_ZIP
    if not LOAD_SAVED_SELECTION or not STATE_FILE.exists():
        return
    payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
    SELECTED_METHOD = payload.get("selected_method", SELECTED_METHOD)
    SELECTED_DATASET = payload.get("selected_dataset", SELECTED_DATASET)
    SELECTED_PRESET = payload.get("selected_preset", SELECTED_PRESET)
    RUN_MODE = payload.get("run_mode", RUN_MODE)
    STRICT_RESULTS = payload.get("strict_results", STRICT_RESULTS)
    GENERATE_PDF = payload.get("generate_pdf", GENERATE_PDF)
    MIN_REQUIRED_PAIRS = payload.get("min_required_pairs", MIN_REQUIRED_PAIRS)
    SKIP_TRAINING = payload.get("skip_training", SKIP_TRAINING)
    SELECTED_DATASET_ROOT = payload.get("dataset_root", SELECTED_DATASET_ROOT)
    LOCAL_ARTIFACTS_ZIP = payload.get("local_artifacts_zip", LOCAL_ARTIFACTS_ZIP)
    print(f"Selecao restaurada de: {STATE_FILE}")


START_DIR = Path.cwd().resolve()
PROJECT_ROOT = find_project_root(START_DIR)

if PROJECT_ROOT is None:
    if not AUTO_CLONE_REPO:
        raise RuntimeError("Repositorio nao encontrado. Ative AUTO_CLONE_REPO=True ou abra o notebook dentro da raiz do projeto.")
    clone_target = (START_DIR / REPO_DIR_NAME).resolve()
    if clone_target.exists() and any(clone_target.iterdir()):
        raise RuntimeError(f"Destino de clone ja existe e nao esta vazio: {clone_target}")
    clone_cmd = ["git", "clone", "--depth", "1", "--branch", BRANCH, REPO_URL, str(clone_target)]
    print("=" * 70)
    print("Repositorio nao encontrado. Clonando projeto...")
    print("=" * 70)
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


# ---- cell ----
# Consolidar selecao do notebook
# A celula inicial acima eh a fonte principal de verdade.
DEFAULT_SELECTION = {
    "SELECTED_METHOD": SELECTED_METHOD,
    "SELECTED_DATASET": SELECTED_DATASET,
    "SELECTED_PRESET": SELECTED_PRESET,
    "RUN_MODE": RUN_MODE,
    "STRICT_RESULTS": STRICT_RESULTS,
    "GENERATE_PDF": GENERATE_PDF,
    "MIN_REQUIRED_PAIRS": MIN_REQUIRED_PAIRS,
    "LOAD_SAVED_SELECTION": LOAD_SAVED_SELECTION,
    "ENABLE_NERF_MEMORY_TUNING": ENABLE_NERF_MEMORY_TUNING,
    "FALLBACK_TO_SMOKE_ON_OOM": FALLBACK_TO_SMOKE_ON_OOM,
    "SELECTED_SCENE_NAME": SELECTED_SCENE_NAME,
    "SELECTED_TT_SCENE_NAME": SELECTED_TT_SCENE_NAME,
}


def _resolve_default(name: str, default):
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value == "":
        return default
    if isinstance(default, bool):
        return raw_value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(default, int):
        try:
            return int(raw_value)
        except ValueError:
            return default
    return raw_value


if LOAD_SAVED_SELECTION:
    load_state()

SELECTED_METHOD = _resolve_default("NVS_SELECTED_METHOD", DEFAULT_SELECTION["SELECTED_METHOD"])
SELECTED_DATASET = _resolve_default("NVS_SELECTED_DATASET", DEFAULT_SELECTION["SELECTED_DATASET"])
SELECTED_PRESET = _resolve_default("NVS_SELECTED_PRESET", DEFAULT_SELECTION["SELECTED_PRESET"])
RUN_MODE = _resolve_default("NVS_RUN_MODE", DEFAULT_SELECTION["RUN_MODE"])
STRICT_RESULTS = _resolve_default("NVS_STRICT_RESULTS", DEFAULT_SELECTION["STRICT_RESULTS"])
GENERATE_PDF = _resolve_default("NVS_GENERATE_PDF", DEFAULT_SELECTION["GENERATE_PDF"])
MIN_REQUIRED_PAIRS = _resolve_default("NVS_MIN_REQUIRED_PAIRS", DEFAULT_SELECTION["MIN_REQUIRED_PAIRS"])
LOAD_SAVED_SELECTION = _resolve_default("NVS_LOAD_SAVED_SELECTION", DEFAULT_SELECTION["LOAD_SAVED_SELECTION"])
ENABLE_NERF_MEMORY_TUNING = _resolve_default(
    "NVS_ENABLE_NERF_MEMORY_TUNING",
    DEFAULT_SELECTION["ENABLE_NERF_MEMORY_TUNING"],
)
FALLBACK_TO_SMOKE_ON_OOM = _resolve_default(
    "NVS_FALLBACK_TO_SMOKE_ON_OOM",
    DEFAULT_SELECTION["FALLBACK_TO_SMOKE_ON_OOM"],
)
SELECTED_SCENE_NAME = _resolve_default("NVS_SELECTED_SCENE_NAME", DEFAULT_SELECTION["SELECTED_SCENE_NAME"])
SELECTED_TT_SCENE_NAME = _resolve_default(
    "NVS_SELECTED_TT_SCENE_NAME",
    DEFAULT_SELECTION["SELECTED_TT_SCENE_NAME"],
)

if SELECTED_DATASET_ROOT:
    SELECTED_DATASET_ROOT = resolve_saved_scene_root(SELECTED_DATASET, SELECTED_DATASET_ROOT)
RUN_ID = generate_run_id(ENVIRONMENT, SELECTED_PRESET, SELECTED_METHOD, SELECTED_DATASET)
AVAILABLE_DATASETS = discover_available_datasets(DATA_SEARCH_ROOTS)
if SELECTED_DATASET_ROOT:
    resolved_dataset_root = str(normalize_dataset_path(Path(SELECTED_DATASET_ROOT)))
else:
    resolved_dataset_root = AVAILABLE_DATASETS.get(SELECTED_DATASET, "")

print("=" * 70)
print("CONFIGURACAO LOCAL")
print("=" * 70)
print(f"Inicio:             {START_DIR}")
print(f"Projeto:            {PROJECT_ROOT}")
print(f"Repo URL:           {REPO_URL}")
print(f"Branch:             {BRANCH}")
print(f"Python:             {sys.executable}")
print(f"Metodo:             {SELECTED_METHOD}")
print(f"Dataset:            {SELECTED_DATASET}")
print(f"Dataset root atual: {resolved_dataset_root or '[auto] ainda nao resolvido'}")
print(f"Preset:             {SELECTED_PRESET}")
print(f"RUN_ID:             {RUN_ID}")
print(f"State file:         {STATE_FILE}")
if AVAILABLE_DATASETS:
    print("Datasets detectados:")
    for dataset_name, dataset_path in sorted(AVAILABLE_DATASETS.items()):
        print(f"  - {dataset_name}: {dataset_path}")
else:
    print("Datasets detectados: nenhum")
if SELECTED_DATASET in MANUAL_DATASET_IDS:
    print(
        "Observacao: Tanks and Temples pode expor varias cenas em subpastas; se o notebook nao resolver uma raiz unica,"
        " informe SELECTED_DATASET_ROOT para a cena desejada."
    )
print("=" * 70)


# ---- cell ----
import os
from pathlib import Path


def _extra_dataset_search_roots(dataset_id: str) -> list[Path]:
    roots: list[Path] = []
    if os.name != "nt":
        return roots
    for drive_letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        drive_root = Path(f"{drive_letter}:/")
        if not drive_root.exists():
            continue
        for candidate in (
            drive_root / dataset_id,
            drive_root / "data" / dataset_id,
            drive_root / "datasets" / dataset_id,
            drive_root / "datasets" / "nvs_benchmark" / dataset_id,
        ):
            if candidate.exists():
                roots.append(candidate.resolve())
    unique_roots: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        resolved = root.resolve()
        resolved_key = str(resolved).lower()
        if resolved_key in seen:
            continue
        seen.add(resolved_key)
        unique_roots.append(resolved)
    return unique_roots


if SELECTED_DATASET in {"mipnerf360", "tanks_and_temples"}:
    extra_roots = _extra_dataset_search_roots(SELECTED_DATASET)
    if extra_roots:
        for candidate_root in extra_roots:
            if candidate_root not in DATA_SEARCH_ROOTS:
                DATA_SEARCH_ROOTS.append(candidate_root)
        AVAILABLE_DATASETS = discover_available_datasets(DATA_SEARCH_ROOTS)
        if not SELECTED_DATASET_ROOT:
            if SELECTED_DATASET in AVAILABLE_DATASETS:
                SELECTED_DATASET_ROOT = AVAILABLE_DATASETS[SELECTED_DATASET]
            else:
                candidates = discover_dataset_candidates(DATA_SEARCH_ROOTS, SELECTED_DATASET)
                if candidates:
                    SELECTED_DATASET_ROOT = str(
                        normalize_dataset_path(
                            choose_preferred_candidate(SELECTED_DATASET, candidates)
                        )
                    )
        if SELECTED_DATASET_ROOT:
            print(f"Dataset root autodetectado no Windows: {SELECTED_DATASET_ROOT}")

# ---- cell ----
# Ajustes de descoberta para cenas image-based
_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".PNG", ".JPG", ".JPEG", ".WEBP")


def _path_has_dataset_markers(path: Path) -> bool:
    return (
        (path / "transforms_train.json").exists()
        or (path / "poses_bounds.npy").exists()
        or _dir_has_files(path / "images")
        or _dir_has_files(path / "images_2")
        or _dir_has_files(path / "images_4")
        or _dir_has_files(path / "images_8")
        or _dir_has_files(path / "sparse" / "0")
        or any(candidate.is_file() and candidate.suffix in _IMAGE_SUFFIXES for candidate in path.rglob("*"))
    )



def _search_candidates_for_marker(base: Path, marker: str) -> list[Path]:
    if marker == "transforms_train.json":
        return [candidate.parent.resolve() for candidate in base.glob("**/transforms_train.json")]
    if marker == "poses_bounds.npy":
        return [candidate.parent.resolve() for candidate in base.glob("**/poses_bounds.npy")]
    if marker in {"images", "images_2", "images_4", "images_8"}:
        return [candidate.parent.resolve() for candidate in base.glob(f"**/{marker}") if _dir_has_files(candidate)]
    if marker == "sparse/0":
        return [candidate.parent.parent.resolve() for candidate in base.glob("**/sparse/0") if _dir_has_files(candidate)]
    if marker == "image_sets":
        return [candidate.resolve() for candidate in base.glob("**/image_sets/*") if candidate.is_dir() and _dir_has_files(candidate)]
    if marker == "direct_images":
        candidates = {
            candidate.parent.resolve()
            for pattern in _IMAGE_SUFFIXES
            for candidate in base.glob(f"**/*{pattern}")
            if candidate.is_file()
        }
        return sorted(candidates, key=lambda candidate: (len(candidate.parts), str(candidate).lower()))
    return []



def discover_dataset_candidates(search_roots: list[Path], dataset_id: str) -> list[Path]:
    markers_by_dataset = {
        "blender_synthetic": ("transforms_train.json",),
        "d_nerf": ("transforms_train.json",),
        "mipnerf360": ("transforms_train.json", "poses_bounds.npy", "images", "images_2", "images_4", "images_8", "sparse/0", "direct_images"),
        "tanks_and_temples": ("image_sets", "transforms_train.json", "poses_bounds.npy", "images", "images_2", "images_4", "images_8", "sparse/0", "direct_images"),
    }
    markers = markers_by_dataset.get(dataset_id, ("transforms_train.json", "poses_bounds.npy", "images", "images_2", "images_4", "images_8", "sparse/0", "direct_images"))
    candidates: list[Path] = []
    for base in search_roots:
        base = Path(base).expanduser().resolve()
        if not base.exists():
            continue
        for marker in markers:
            candidates.extend(_search_candidates_for_marker(base, marker))
    unique_candidates = sorted(
        {candidate for candidate in candidates},
        key=lambda candidate: (len(candidate.parts), str(candidate).lower()),
    )
    return unique_candidates



def resolve_saved_scene_root(dataset_id: str, root_value: str) -> str:
    root_path = Path(root_value).expanduser().resolve()
    if _path_has_dataset_markers(root_path):
        return str(root_path)
    search_roots = [root_path]
    if dataset_id == "tanks_and_temples":
        search_roots.append(root_path / "image_sets")
    candidates = discover_dataset_candidates(search_roots, dataset_id)
    if candidates:
        return str(normalize_dataset_path(choose_preferred_candidate(dataset_id, candidates)))
    if dataset_id == "tanks_and_temples":
        return ""
    return str(root_path)


# ---- cell ----
def _is_tanks_and_temples_scene_root(candidate: Path) -> bool:
    if not candidate.exists() or not candidate.is_dir():
        return False
    ancestor_names = {parent.name.lower() for parent in candidate.parents}
    if not ancestor_names.intersection({"image_sets", "videos"}):
        return False
    has_direct_images = any(
        file_candidate.is_file() and file_candidate.suffix.lower() in {suffix.lower() for suffix in _IMAGE_SUFFIXES}
        for file_candidate in candidate.rglob("*")
    )
    return (
        _existing_splits(candidate)
        or (candidate / "images").exists()
        or (candidate / "poses_bounds.npy").exists()
        or has_direct_images
    )



def discover_dataset_candidates(search_roots: list[Path], dataset_id: str) -> list[Path]:
    if dataset_id == "tanks_and_temples":
        candidates: list[Path] = []
        for base in search_roots:
            base = Path(base).expanduser().resolve()
            if not base.exists():
                continue
            if _is_tanks_and_temples_scene_root(base):
                candidates.append(base)
            for marker in ("image_sets", "videos"):
                for candidate in base.glob(f"**/{marker}/*"):
                    if candidate.is_dir() and _is_tanks_and_temples_scene_root(candidate):
                        candidates.append(candidate.resolve())
        unique_candidates = sorted(
            {candidate for candidate in candidates},
            key=lambda candidate: (len(candidate.parts), str(candidate).lower()),
        )
        return unique_candidates

    markers_by_dataset = {
        "blender_synthetic": ("transforms_train.json",),
        "d_nerf": ("transforms_train.json",),
        "mipnerf360": ("transforms_train.json", "poses_bounds.npy", "sparse/0"),
    }
    markers = markers_by_dataset.get(dataset_id, ("transforms_train.json", "poses_bounds.npy", "images", "sparse/0"))
    candidates: list[Path] = []
    for base in search_roots:
        base = Path(base).expanduser().resolve()
        if not base.exists():
            continue
        for marker in markers:
            candidates.extend(_search_candidates_for_marker(base, marker))
    unique_candidates = sorted(
        {candidate for candidate in candidates},
        key=lambda candidate: (len(candidate.parts), str(candidate).lower()),
    )
    return unique_candidates



def resolve_saved_scene_root(dataset_id: str, root_value: str) -> str:
    root_path = Path(root_value).expanduser().resolve()
    if dataset_id == "tanks_and_temples":
        if _is_tanks_and_temples_scene_root(root_path):
            return str(root_path)
        search_roots = [root_path, root_path / "image_sets", root_path / "videos"]
        candidates = discover_dataset_candidates(search_roots, dataset_id)
        if candidates:
            return str(normalize_dataset_path(choose_preferred_candidate(dataset_id, candidates)))
        return ""
    if _path_has_dataset_markers(root_path):
        return str(root_path)
    search_roots = [root_path]
    candidates = discover_dataset_candidates(search_roots, dataset_id)
    if candidates:
        return str(normalize_dataset_path(choose_preferred_candidate(dataset_id, candidates)))
    return str(root_path)

# ---- cell ----
# Setup do ambiente local
print("=" * 70)
print("Setup local")
print("=" * 70)

print(f"Diretorio atual: {Path.cwd()}")
if not looks_like_project_root(Path.cwd()):
    raise RuntimeError("Diretorio atual nao parece ser a raiz do projeto. Execute a celula de configuracao novamente.")

if UPGRADE_PIP:
    run_logged([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], label="pip-upgrade", check=False)
else:
    print("Pulando upgrade do pip (UPGRADE_PIP=False)")

if INSTALL_PROJECT:
    run_logged([sys.executable, "-m", "pip", "install", "-e", "."], label="pip-install-editable")
else:
    print("Pulando pip install -e . (INSTALL_PROJECT=False)")

status = run_logged([sys.executable, "-m", "nvs_benchmark.cli", "status"], label="cli-status", check=False)
if status.returncode != 0:
    raise RuntimeError("Falha ao executar nvs_benchmark.cli status. Ajuste o kernel/venv ou ative INSTALL_PROJECT=True.")

if CLONE_THIRD_PARTY_IF_MISSING:
    third_party_dir = Path("./third_party")
    third_party_dir.mkdir(parents=True, exist_ok=True)
    methods_to_clone = [
        {"id": "d_nerf", "path": third_party_dir / "d_nerf", "url": "https://github.com/albertpumarola/D-NeRF.git"},
        {"id": "gaussian_splatting", "path": third_party_dir / "gaussian_splatting", "url": "https://github.com/graphdeco-inria/gaussian-splatting.git"},
        {"id": "4d_gaussians", "path": third_party_dir / "4d_gaussians", "url": "https://github.com/hustvl/4DGaussians.git"},
    ]
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
# Verificacao de hardware local
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
        print("GPU nao detectada. O benchmark pode rodar em CPU, mas ficara mais lento.")
except Exception as exc:
    print(f"Nao foi possivel consultar torch/CUDA: {exc}")


# ---- cell ----
# Dependencias compiladas dos metodos Gaussian (quando selecionados)
print("=" * 70)
print("Dependencias do metodo selecionado")
print("=" * 70)

def install_extension(label: str, path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Submodulo ausente para {label}: {path}. Reexecute o clone com --recursive.")
    if "simple-knn" in label:
        package_dir = path / "simple_knn"
        package_dir.mkdir(parents=True, exist_ok=True)
        init_file = package_dir / "__init__.py"
        if not init_file.exists():
            init_file.write_text("", encoding="utf-8")
    run_logged(
        [sys.executable, "-m", "pip", "install", "--no-build-isolation", "-e", str(path)],
        label=f"install-{label}",
    )


if SELECTED_METHOD not in {"gs_static", "gs_dynamic"}:
    print("Metodo NeRF selecionado: nenhuma extensao CUDA adicional sera compilada nesta celula.")
elif not INSTALL_COMPILED_METHOD_DEPS:
    print("Compilacao automatica desativada (INSTALL_COMPILED_METHOD_DEPS=False).")
elif SELECTED_METHOD == "gs_static":
    gs_path = Path("./third_party/gaussian_splatting")
    if not gs_path.exists():
        raise FileNotFoundError("Repositorio gaussian_splatting ausente. Execute a celula de setup com CLONE_THIRD_PARTY_IF_MISSING=True.")
    run_logged([sys.executable, "-m", "pip", "install", "plyfile>=1.0.3", "joblib>=1.4"], label="install-gs-runtime")
    install_extension("diff-gaussian-rasterization", gs_path / "submodules" / "diff-gaussian-rasterization")
    install_extension("simple-knn", gs_path / "submodules" / "simple-knn")
    fused_ssim_path = gs_path / "submodules" / "fused-ssim"
    if fused_ssim_path.exists():
        install_extension("fused-ssim", fused_ssim_path)
    probe = run_logged(
        [sys.executable, "-c", "import diff_gaussian_rasterization, simple_knn._C, plyfile; print('ok')"],
        label="probe-gs-static",
        check=False,
        cwd=gs_path,
    )
    if probe.returncode != 0:
        raise RuntimeError("Extensoes de gs_static nao ficaram disponiveis no kernel atual.")
else:
    g4d_path = Path("./third_party/4d_gaussians")
    if not g4d_path.exists():
        raise FileNotFoundError("Repositorio 4d_gaussians ausente. Execute a celula de setup com CLONE_THIRD_PARTY_IF_MISSING=True.")
    run_logged([sys.executable, "-m", "pip", "install", "plyfile>=1.0.3", "joblib>=1.4"], label="install-4dgs-runtime")
    mmcv_probe = run_logged([sys.executable, "-c", "import mmcv; print(mmcv.__version__)"], label="probe-mmcv", check=False)
    if mmcv_probe.returncode != 0:
        mmcv_install = run_logged([sys.executable, "-m", "pip", "install", "mmcv==1.6.0"], label="install-mmcv", check=False)
        if mmcv_install.returncode != 0:
            run_logged([sys.executable, "-m", "pip", "install", "mmcv-lite"], label="install-mmcv-lite")
    install_extension("depth-diff-gaussian-rasterization", g4d_path / "submodules" / "depth-diff-gaussian-rasterization")
    install_extension("simple-knn-4dgs", g4d_path / "submodules" / "simple-knn")
    probe = run_logged(
        [sys.executable, "-c", "import mmcv, simple_knn._C, plyfile; print('ok')"],
        label="probe-gs-dynamic",
        check=False,
        cwd=g4d_path,
    )
    if probe.returncode != 0:
        raise RuntimeError("Extensoes de gs_dynamic nao ficaram disponiveis no kernel atual; revise CUDA, PyTorch e mmcv.")


# ---- cell ----
# Preparar artifacts locais para metrics-only (opcional)
print("=" * 70)
print("Artifacts locais")
print("=" * 70)
print(f"SKIP_TRAINING: {SKIP_TRAINING}")

if not SKIP_TRAINING:
    print("Modo full training ativo. Esta celula e opcional.")
elif not LOCAL_ARTIFACTS_ZIP:
    raise RuntimeError("SKIP_TRAINING=True, mas LOCAL_ARTIFACTS_ZIP esta vazio. Informe um ZIP local com checkpoint e renders.")
else:
    zip_path = Path(LOCAL_ARTIFACTS_ZIP).expanduser().resolve()
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP local nao encontrado: {zip_path}")

    extract_dir = Path("./notebooks/artifacts/uploaded_artifacts")
    UPLOADED_CHECKPOINT, UPLOADED_RENDERS_DIR = extract_zip_artifacts(str(zip_path), str(extract_dir))

    print(f"ZIP local:    {zip_path}")
    print(f"Checkpoint:   {UPLOADED_CHECKPOINT}")
    print(f"Renders dir:  {UPLOADED_RENDERS_DIR}")
    print("Artifacts prontos para a celula de benchmark.")


# ---- cell ----
# Preparar dataset local
print("=" * 70)
print("Preparando dataset")
print("=" * 70)

AVAILABLE_DATASETS = discover_available_datasets(DATA_SEARCH_ROOTS)
dataset_root = None
candidate_roots: list[Path] = []

if SELECTED_DATASET_ROOT:
    dataset_root = normalize_dataset_path(Path(SELECTED_DATASET_ROOT))
    print(f"Usando dataset root informado manualmente: {dataset_root}")
elif SELECTED_DATASET in AVAILABLE_DATASETS:
    dataset_root = normalize_dataset_path(Path(AVAILABLE_DATASETS[SELECTED_DATASET]))
    print(f"Dataset detectado automaticamente: {dataset_root}")
else:
    candidate_roots = discover_dataset_candidates(DATA_SEARCH_ROOTS, SELECTED_DATASET)
    if len(candidate_roots) == 1:
        dataset_root = normalize_dataset_path(candidate_roots[0])
        print(f"Dataset detectado automaticamente: {dataset_root}")
    elif candidate_roots:
        print(f"Foram encontrados {len(candidate_roots)} candidatos para {SELECTED_DATASET}:")
        for candidate in candidate_roots:
            print(f"  - {candidate}")
        if SELECTED_DATASET in {"mipnerf360", "tanks_and_temples"}:
            dataset_root = normalize_dataset_path(choose_preferred_candidate(SELECTED_DATASET, candidate_roots))
            print(f"Dataset {SELECTED_DATASET} selecionado automaticamente: {dataset_root}")

if dataset_root is None and INSTALL_DATASET_IF_MISSING:
    if not CATALOG_FILE.exists():
        raise FileNotFoundError(f"Catalogo nao encontrado: {CATALOG_FILE}")
    print(f"Gerando catalogo temporario para dataset: {SELECTED_DATASET}")
    filtered_catalog = write_single_dataset_catalog(SELECTED_DATASET)
    install_cmd = [
        sys.executable,
        "-m",
        "nvs_benchmark.cli",
        "install",
        "--catalog-file",
        str(filtered_catalog),
        "--only",
        "datasets",
        "--execute",
    ]
    install_result = run_logged(install_cmd, label="datasets-install", check=False)
    if install_result.returncode != 0:
        raise RuntimeError(f"Falha ao instalar dataset {SELECTED_DATASET} via catalogo filtrado.")

    AVAILABLE_DATASETS = discover_available_datasets(DATA_SEARCH_ROOTS)
    if SELECTED_DATASET in AVAILABLE_DATASETS:
        dataset_root = normalize_dataset_path(Path(AVAILABLE_DATASETS[SELECTED_DATASET]))
        print(f"Dataset detectado automaticamente apos instalacao: {dataset_root}")
    else:
        candidate_roots = discover_dataset_candidates(DATA_SEARCH_ROOTS, SELECTED_DATASET)
        if candidate_roots:
            print(f"Foram encontrados {len(candidate_roots)} candidatos para {SELECTED_DATASET} apos a instalacao:")
            for candidate in candidate_roots:
                print(f"  - {candidate}")
        if len(candidate_roots) == 1:
            dataset_root = normalize_dataset_path(candidate_roots[0])
            print(f"Dataset detectado automaticamente apos instalacao: {dataset_root}")
        elif candidate_roots and SELECTED_DATASET in {"mipnerf360", "tanks_and_temples"}:
            dataset_root = normalize_dataset_path(choose_preferred_candidate(SELECTED_DATASET, candidate_roots))
            print(f"Dataset {SELECTED_DATASET} selecionado automaticamente apos a instalacao: {dataset_root}")
        elif candidate_roots:
            print(f"Nenhuma raiz valida unica foi encontrada para {SELECTED_DATASET} apos a instalacao.")

if dataset_root is None or not dataset_root.exists():
    if SELECTED_DATASET in MANUAL_DATASET_IDS:
        prepared_root = (PROJECT_ROOT / "data" / SELECTED_DATASET).resolve()
        image_sets_dir = prepared_root / "image_sets"
        video_sets_dir = prepared_root / "videos"
        scene_candidates = discover_dataset_candidates(
            [prepared_root, image_sets_dir, video_sets_dir],
            SELECTED_DATASET,
        )
        if scene_candidates:
            dataset_root = normalize_dataset_path(choose_preferred_candidate(SELECTED_DATASET, scene_candidates))
            SELECTED_DATASET_ROOT = str(dataset_root)
            save_state(SELECTED_DATASET_ROOT)
            print(f"Cena Tanks and Temples encontrada: {SELECTED_DATASET_ROOT}")
            print("Dataset preparado.")
        else:
            print("Tanks and Temples nao foi encontrado em uma cena extraida valida.")
            print(f"Diretorio esperado: {prepared_root}")
            print("Baixe/extrai manualmente em image_sets/<cena> ou videos/<cena> e defina SELECTED_DATASET_ROOT para a cena desejada.")
            raise FileNotFoundError(
                "Nenhuma cena valida de Tanks and Temples foi encontrada. O download automatico pode exigir autenticacao no site oficial."
            )
    else:
        raise FileNotFoundError(
            f"Nao foi possivel resolver a raiz do dataset {SELECTED_DATASET}. Informe SELECTED_DATASET_ROOT ou ative INSTALL_DATASET_IF_MISSING=True."
        )
else:
    SELECTED_DATASET_ROOT = str(dataset_root)
    print(f"Dataset root final: {SELECTED_DATASET_ROOT}")
    print(f"transforms_train.json: {(dataset_root / 'transforms_train.json').exists()}")
    print(f"transforms_test.json:  {(dataset_root / 'transforms_test.json').exists()}")

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
    preflight = run_logged(preflight_cmd, label="dataset-check", check=False)
    if preflight.returncode != 0:
        raise RuntimeError(f"dataset-check falhou para {SELECTED_DATASET} em {SELECTED_DATASET_ROOT}")

    save_state(SELECTED_DATASET_ROOT)
    print("Dataset pronto.")


# ---- cell ----
# Executar benchmark
print("=" * 70)
print(f"Executando benchmark: {SELECTED_METHOD} x {SELECTED_DATASET}")
print("=" * 70)

Path("./artifacts/metrics").mkdir(parents=True, exist_ok=True)
Path("./logs").mkdir(parents=True, exist_ok=True)
snapshot_file = Path("./artifacts/metrics") / f"{RUN_ID}.json"

metrics_only_mode = bool(SKIP_TRAINING and UPLOADED_CHECKPOINT and UPLOADED_RENDERS_DIR)
if SKIP_TRAINING and not metrics_only_mode:
    raise RuntimeError("SKIP_TRAINING=True, mas os artifacts locais ainda nao foram preparados. Execute a celula anterior.")
if metrics_only_mode:
    print("Modo metrics-only ativo.")
    cmd = [
        sys.executable,
        "-m",
        "nvs_benchmark.cli",
        "metrics-compute",
        "--method",
        SELECTED_METHOD,
        "--checkpoint",
        UPLOADED_CHECKPOINT,
        "--rendered-dir",
        UPLOADED_RENDERS_DIR,
        "--dataset",
        SELECTED_DATASET,
        "--root",
        SELECTED_DATASET_ROOT,
        "--split",
        "train",
        "--snapshot-file",
        str(snapshot_file),
        "--append-snapshot",
        "--train-seconds",
        str(UPLOADED_TRAIN_SECONDS),
        "--inference-seconds",
        str(UPLOADED_INFERENCE_SECONDS),
        "--log-dir",
        "./logs",
    ]
    if STRICT_RESULTS:
        cmd.extend(["--strict-results", "--min-required-pairs", str(MIN_REQUIRED_PAIRS)])
    result = run_logged(cmd, label="metrics-compute", check=False)
    if result.returncode != 0:
        raise RuntimeError(f"metrics-compute falhou para {SELECTED_METHOD} (code={result.returncode})")
elif RUN_MODE == "quick_check":
    print("RUN_MODE=quick_check: dataset-check ja foi executado; nenhum treino sera iniciado.")
else:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")
    run_logged(
        [sys.executable, "-c", "import torch; print('CUDA_SUBPROCESS=', torch.cuda.is_available()); print('GPU_NAME=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"],
        label="cuda-subprocess-check",
        check=False,
    )

    method_extra = json.loads(EXTRA_JSON) if EXTRA_JSON else {}
    if SELECTED_METHOD.startswith("nerf_") and ENABLE_NERF_MEMORY_TUNING:
        nerf_extra = {
            "nerf_half_res": True,
            "nerf_n_rand": 256,
            "nerf_n_samples": 32,
            "nerf_n_importance": 0,
            "nerf_chunk": 1024,
            "nerf_netchunk": 4096,
        }
        nerf_extra.update(method_extra)
        method_extra = nerf_extra

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
        str(snapshot_file),
        "--append-snapshot",
    ]
    if method_extra:
        cmd.extend(["--extra-json", json.dumps(method_extra)])
    if STRICT_RESULTS:
        cmd.extend(["--strict-results", "--min-required-pairs", str(MIN_REQUIRED_PAIRS)])
    result = run_logged(cmd, label="method-run", check=False)
    if result.returncode != 0:
        combined_output = f"{result.stdout}\n{result.stderr}".lower()
        oom_like = "exit=-9" in combined_output or "out of memory" in combined_output
        if FALLBACK_TO_SMOKE_ON_OOM and oom_like and SELECTED_METHOD.startswith("nerf_") and SELECTED_PRESET != "smoke":
            print("Falha de memoria detectada; repetindo o metodo NeRF com preset smoke.")
            retry_cmd = list(cmd)
            retry_cmd[retry_cmd.index("--preset") + 1] = "smoke"
            retry_result = run_logged(retry_cmd, label="method-run-smoke-fallback", check=False)
            if retry_result.returncode != 0:
                raise RuntimeError("method-run e fallback smoke falharam. Revise GPU, memoria e logs da execucao.")
        else:
            raise RuntimeError(f"method-run falhou para {SELECTED_METHOD} x {SELECTED_DATASET} (code={result.returncode})")

print(f"Snapshot alvo: {snapshot_file}")


# ---- cell ----
# Gerar relatorio HTML
print("=" * 70)
print("Gerando relatorio HTML")
print("=" * 70)

if RUN_MODE == "quick_check" and not SKIP_TRAINING:
    raise RuntimeError("RUN_MODE=quick_check nao gera snapshot de benchmark. Use RUN_MODE='full' ou SKIP_TRAINING=True com artifacts locais.")
if snapshot_file is None:
    raise RuntimeError("snapshot_file nao foi definido. Execute a celula de benchmark antes desta.")

report_name = f"{RUN_ID}_{SELECTED_METHOD}_{SELECTED_DATASET}_report"
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

if report_html is None:
    raise RuntimeError("report_html ainda nao foi definido. Execute a celula de relatorio antes desta.")
if not Path(report_html).exists():
    print(f"Arquivo nao encontrado: {report_html}")
else:
    display(IFrame(src=Path(report_html).resolve().as_uri(), width=1200, height=700))


# ---- cell ----
# Compactar artifacts e executar backup local opcional
import shutil

archive_dir = Path("./notebooks/artifacts/archives")
archive_dir.mkdir(parents=True, exist_ok=True)
archive_base = archive_dir / f"{RUN_ID}_artifacts"
archive_file = shutil.make_archive(str(archive_base), "zip", "./artifacts")
print(f"ZIP gerado: {Path(archive_file).resolve()}")
print(f"Tamanho: {Path(archive_file).stat().st_size / (1024 * 1024):.1f} MB")

if USE_LOCAL_BACKUP:
    backup_root = Path(LOCAL_BACKUP_DIR).expanduser().resolve()
    backup_artifacts = backup_root / "artifacts"
    backup_artifacts.mkdir(parents=True, exist_ok=True)
    for directory in ["metrics", "reports"]:
        source = Path("./artifacts") / directory
        if source.exists():
            shutil.copytree(source, backup_artifacts / directory, dirs_exist_ok=True)
    archive_backup_dir = backup_artifacts / "archives"
    archive_backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(archive_file, archive_backup_dir / Path(archive_file).name)
    if BACKUP_DATASETS and Path("./data").exists():
        shutil.copytree("./data", backup_root / "data", dirs_exist_ok=True)
    print(f"Backup local concluido em: {backup_root}")
else:
    print("Backup local desativado. Ative USE_LOCAL_BACKUP=True na celula de configuracao se necessario.")

