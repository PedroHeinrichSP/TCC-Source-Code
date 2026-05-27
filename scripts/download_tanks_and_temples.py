from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


INRIA_TANDT_ZIP_URL = "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip"
OFFICIAL_DOWNLOADER_URL = (
    "https://raw.githubusercontent.com/IntelVCL/TanksAndTemples/master/python_toolbox/download_t2_dataset.py"
)
KNOWN_TANDT_SCENES = {
    "auditorium",
    "ballroom",
    "barn",
    "caterpillar",
    "church",
    "courthouse",
    "family",
    "francis",
    "horse",
    "ignatius",
    "lighthouse",
    "m60",
    "museum",
    "panther",
    "playground",
    "temple",
    "train",
    "truck",
}


def _dir_has_scene_markers(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    return (
        (path / "images").exists()
        or (path / "sparse" / "0").exists()
        or (path / "poses_bounds.npy").exists()
        or any(candidate.is_file() and candidate.suffix.lower() in {".png", ".jpg", ".jpeg"} for candidate in path.rglob("*"))
    )


def _download_with_progress(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"[skip] arquivo ja existe: {destination}")
        return destination

    print(f"[download] {url}")
    try:
        with urllib.request.urlopen(url) as response, destination.open("wb") as out:
            total = int(response.headers.get("Content-Length", "0") or "0")
            downloaded = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    print(
                        f"\r    {downloaded / (1024 * 1024):.0f} / {total / (1024 * 1024):.0f} MB",
                        end="",
                        flush=True,
                    )
                else:
                    print(f"\r    {downloaded / (1024 * 1024):.0f} MB", end="", flush=True)
        print()
        return destination
    except Exception as exc:
        print()
        print(f"[warn] urllib falhou ao baixar {url}: {type(exc).__name__}: {exc}")
        _safe_unlink(destination)

    fallback_errors: list[str] = []

    if sys.platform.startswith("win"):
        ps_command = (
            "$ProgressPreference='SilentlyContinue'; "
            f"Invoke-WebRequest -Uri '{url}' -OutFile '{destination}' -UseBasicParsing"
        )
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_command],
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode == 0 and destination.exists() and destination.stat().st_size > 0:
            print("[ok] download concluido via PowerShell.")
            return destination
        fallback_errors.append(
            f"powershell exit={completed.returncode}: {(completed.stderr or completed.stdout or '').strip()[:400]}"
        )

    completed = subprocess.run(
        ["curl", "-L", url, "-o", str(destination)],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.returncode == 0 and destination.exists() and destination.stat().st_size > 0:
        print("[ok] download concluido via curl.")
        return destination
    fallback_errors.append(
        f"curl exit={completed.returncode}: {(completed.stderr or completed.stdout or '').strip()[:400]}"
    )
    _safe_unlink(destination)

    details = " | ".join(error for error in fallback_errors if error)
    raise RuntimeError(
        "Falha ao baixar Tanks and Temples por HTTPS. "
        "O ambiente pode estar bloqueando urllib/TLS. "
        f"Tentativas de fallback: {details}"
    )


def _safe_rmtree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except TypeError:
        if path.exists():
            path.unlink()


def _copy_scene(scene_root: Path, destination_root: Path) -> Path:
    destination = destination_root / scene_root.name
    if destination.exists():
        _safe_rmtree(destination)
    shutil.copytree(scene_root, destination)
    return destination


def _normalize_inria_tandt_layout(extracted_root: Path, target_path: Path, scene_filter: str | None) -> list[Path]:
    image_sets_root = target_path / "image_sets"
    image_sets_root.mkdir(parents=True, exist_ok=True)

    requested_scene = (scene_filter or "").strip().lower()
    copied: list[Path] = []
    seen: set[str] = set()

    for candidate in extracted_root.rglob("*"):
        if not candidate.is_dir():
            continue
        scene_name = candidate.name.strip().lower()
        if scene_name not in KNOWN_TANDT_SCENES:
            continue
        if requested_scene and scene_name != requested_scene:
            continue
        if scene_name in seen:
            continue
        if not _dir_has_scene_markers(candidate):
            continue
        copied.append(_copy_scene(candidate, image_sets_root))
        seen.add(scene_name)

    return copied


def _download_from_inria(pathname: Path, scene: str | None) -> int:
    downloads_dir = (Path("./artifacts/downloads").resolve())
    archive_path = downloads_dir / "tandt_db.zip"
    extract_root = downloads_dir / "tandt_db_extracted"

    _download_with_progress(INRIA_TANDT_ZIP_URL, archive_path)
    _safe_rmtree(extract_root)
    extract_root.mkdir(parents=True, exist_ok=True)

    print(f"[extract] {archive_path} -> {extract_root}")
    with zipfile.ZipFile(archive_path, "r") as zip_ref:
        zip_ref.extractall(extract_root)

    copied = _normalize_inria_tandt_layout(extract_root, pathname, scene)
    if not copied and scene:
        print(f"[warn] cena '{scene}' nao encontrada em tandt_db.zip; usando qualquer cena disponivel.")
        copied = _normalize_inria_tandt_layout(extract_root, pathname, None)
    if not copied:
        raise FileNotFoundError(
            "Nao foi possivel localizar cenas de Tanks and Temples dentro de tandt_db.zip "
            f"(origem extraida: {extract_root})."
        )

    print("[ok] cenas prontas:")
    for scene_root in copied:
        print(f"  - {scene_root}")
    return 0


def download_official_downloader(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(OFFICIAL_DOWNLOADER_URL, destination)
    patch_official_downloader(destination)
    return destination


def patch_official_downloader(script_path: Path) -> None:
    source = script_path.read_text(encoding="utf-8-sig")
    patched_source = source.replace(
        "with open(fname) as f:",
        'with open(fname, encoding="utf-8-sig", errors="replace") as f:',
    )

    if patched_source != source:
        script_path.write_text(patched_source, encoding="utf-8")


def build_official_command(script_path: Path, pathname: Path, group: str, modality: str) -> list[str]:
    return [
        sys.executable,
        str(script_path),
        "--modality",
        modality,
        "--group",
        group,
        "--pathname",
        str(pathname),
        "--calc_md5_off",
    ]


def extract_downloaded_archives(target_path: Path, modality: str) -> None:
    roots = []
    if modality in {"image", "both"}:
        roots.append(target_path / "image_sets")
    if modality in {"video", "both"}:
        roots.append(target_path / "videos")

    invalid_archives: list[str] = []
    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for zip_file in sorted(root.glob("*.zip")):
            scene_name = zip_file.stem
            extract_dir = root / scene_name
            if extract_dir.exists():
                _safe_rmtree(extract_dir)
            extract_dir.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(zip_file, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)
            except zipfile.BadZipFile:
                _safe_rmtree(extract_dir)
                _safe_unlink(zip_file)
                invalid_archives.append(str(zip_file))
                continue

            _safe_unlink(zip_file)

    if invalid_archives:
        invalid_list = ", ".join(invalid_archives)
        raise RuntimeError(
            "Alguns arquivos ZIP baixados para Tanks and Temples eram invalidos e foram removidos: "
            f"{invalid_list}. O downloader oficial retornou HTML/login page em vez do ZIP real. "
            "Use --source inria ou forneca o dataset manualmente."
        )


def _download_from_official(pathname: Path, group: str, modality: str) -> int:
    script_path = download_official_downloader(Path("./artifacts/downloads/download_t2_dataset.py").resolve())
    command = build_official_command(script_path, pathname, group, modality)

    print(f"[info] downloader oficial: {OFFICIAL_DOWNLOADER_URL}")
    print(f"[info] destino: {pathname}")
    print(f"[info] comando: {' '.join(command)}")

    completed = subprocess.run(command, check=False)
    extract_downloaded_archives(pathname, modality)
    return completed.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Tanks and Temples data for the benchmark.")
    parser.add_argument("--pathname", default="./data/tanks_and_temples", help="Diretorio de destino do dataset")
    parser.add_argument(
        "--group",
        default="all",
        choices=["intermediate", "advanced", "both", "training", "all"],
        help="Subconjunto de cenas para o downloader oficial",
    )
    parser.add_argument(
        "--modality",
        default="image",
        choices=["image", "video", "both"],
        help="Tipo de conteudo para o downloader oficial",
    )
    parser.add_argument(
        "--source",
        default="inria",
        choices=["inria", "official"],
        help="Origem do dataset: zip publico da Inria ou downloader oficial do Tanks and Temples.",
    )
    parser.add_argument(
        "--scene",
        default="",
        help="Cena preferida para manter/preparar (ex: truck, train, family).",
    )
    args = parser.parse_args()

    target_path = Path(args.pathname).expanduser().resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    if args.source == "inria":
        return _download_from_inria(target_path, args.scene or None)
    return _download_from_official(target_path, args.group, args.modality)


if __name__ == "__main__":
    raise SystemExit(main())
