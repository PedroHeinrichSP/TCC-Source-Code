from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


MIPNERF360_URL = "https://storage.googleapis.com/gresearch/refraw360/360_v2.zip"


def _format_size(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def download_with_progress(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as output_file:
        total_size_header = response.headers.get("Content-Length")
        total_size = int(total_size_header) if total_size_header and total_size_header.isdigit() else None
        downloaded_bytes = 0
        last_report = 0.0
        started_at = 0.0
        import time

        started_at = time.time()
        last_report = started_at
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
                    print(
                        f"  {downloaded_bytes / (1024 * 1024):.1f} MB / {_format_size(total_size)} "
                        f"({pct:.1f}%) - {speed_mb_s:.1f} MB/s"
                    )
                else:
                    print(f"  {downloaded_bytes / (1024 * 1024):.1f} MB baixados - {speed_mb_s:.1f} MB/s")
                last_report = now
    return destination


def extract_archive(zip_path: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(destination)


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
        help="Nome do arquivo zip temporario",
    )
    args = parser.parse_args()

    target_path = Path(args.pathname).expanduser().resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="mipnerf360_download_") as temp_dir:
        archive_path = Path(temp_dir) / args.archive_name
        print(f"[info] dataset url: {args.url}")
        print(f"[info] destino: {target_path}")
        print(f"[info] baixando em: {archive_path}")
        download_with_progress(args.url, archive_path)
        print(f"[info] extraindo em: {target_path}")
        extract_archive(archive_path, target_path)

    print("[ok] Mip-NeRF 360 pronto")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())