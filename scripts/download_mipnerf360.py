from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


MIPNERF360_URL = "https://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".PNG", ".JPG", ".JPEG", ".WEBP")


def _format_size(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def _has_mipnerf360_scene_content(path: Path) -> bool:
    try:
        scene_dirs = [candidate for candidate in path.iterdir() if candidate.is_dir()]
    except OSError:
        return True

    for scene_dir in scene_dirs:
        if (scene_dir / "poses_bounds.npy").exists() or (scene_dir / "sparse" / "0").exists():
            return True
        try:
            if any(candidate.is_file() and candidate.suffix in IMAGE_SUFFIXES for candidate in scene_dir.rglob("*")):
                return True
        except OSError:
            continue
    return False


def download_with_progress(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    print(f"[info] conectando a: {url}", flush=True)
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as output_file:
        total_size_header = response.headers.get("Content-Length")
        total_size = int(total_size_header) if total_size_header and total_size_header.isdigit() else None
        downloaded_bytes = 0
        last_report = 0.0
        started_at = 0.0
        import time

        started_at = time.time()
        last_report = started_at
        chunk_size = 512 * 1024
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            output_file.write(chunk)
            downloaded_bytes += len(chunk)
            now = time.time()
            if now - last_report >= 1 or (total_size is not None and downloaded_bytes >= total_size):
                elapsed = max(now - started_at, 0.001)
                speed_mb_s = downloaded_bytes / elapsed / (1024 * 1024)
                if total_size is not None:
                    pct = downloaded_bytes * 100 / total_size
                    print(
                        f"  {downloaded_bytes / (1024 * 1024):.1f} MB / {_format_size(total_size)} "
                        f"({pct:.1f}%) - {speed_mb_s:.1f} MB/s",
                        flush=True,
                    )
                else:
                    print(
                        f"  {downloaded_bytes / (1024 * 1024):.1f} MB baixados - {speed_mb_s:.1f} MB/s",
                        flush=True,
                    )
                last_report = now
    return destination


def extract_archive(zip_path: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(destination)


def _resolve_archive_path(target_path: Path, archive_name: str, archive_path_arg: str) -> Path:
    if archive_path_arg.strip():
        return Path(archive_path_arg).expanduser().resolve()

    candidate_paths = [
        target_path / archive_name,
        target_path.parent / archive_name,
        (Path("./artifacts/downloads") / archive_name).expanduser().resolve(),
    ]
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path.resolve()
    return candidate_paths[1].resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the Mip-NeRF 360 dataset with progress output.")
    parser.add_argument(
        "--pathname",
        default="./data/mipnerf360",
        help="Diretorio de destino do dataset",
    )
    parser.add_argument(
        "--url",
        default=MIPNERF360_URL,
        help="URL do arquivo zip do dataset",
    )
    parser.add_argument(
        "--archive-name",
        default="360_v2.zip",
        help="Nome do arquivo zip local",
    )
    parser.add_argument(
        "--archive-path",
        default="",
        help="Caminho local opcional para reutilizar/salvar o ZIP baixado",
    )
    args = parser.parse_args()

    target_path = Path(args.pathname).expanduser().resolve()
    target_path.mkdir(parents=True, exist_ok=True)
    archive_path = _resolve_archive_path(target_path, args.archive_name, args.archive_path)

    if _has_mipnerf360_scene_content(target_path):
        print(f"[skip] Mip-NeRF 360 ja esta disponivel em: {target_path}")
        return 0

    if archive_path.exists() and not zipfile.is_zipfile(archive_path):
        print(f"[warn] arquivo existente nao e um ZIP valido, removendo: {archive_path}", flush=True)
        archive_path.unlink()

    print(f"[info] dataset url: {args.url}")
    print(f"[info] destino: {target_path}")
    print(f"[info] arquivo local: {archive_path}")

    if not archive_path.exists():
        print(f"[info] baixando em: {archive_path}")
        download_with_progress(args.url, archive_path)
    else:
        print(f"[info] reutilizando ZIP existente: {archive_path}")

    print(f"[info] extraindo em: {target_path}")
    extract_archive(archive_path, target_path)
    print("[ok] Mip-NeRF 360 pronto")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
