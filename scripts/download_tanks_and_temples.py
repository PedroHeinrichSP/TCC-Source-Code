from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path


OFFICIAL_DOWNLOADER_URL = (
    "https://raw.githubusercontent.com/IntelVCL/TanksAndTemples/master/python_toolbox/download_t2_dataset.py"
)


def download_official_downloader(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(OFFICIAL_DOWNLOADER_URL, destination)
    return destination


def build_command(script_path: Path, pathname: Path, group: str, modality: str) -> list[str]:
    return [
        sys.executable,
        str(script_path),
        "--modality",
        modality,
        "--group",
        group,
        "--pathname",
        str(pathname),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download Tanks and Temples data by running the official Python toolbox downloader."
    )
    parser.add_argument(
        "--pathname",
        default="./data/tanks_and_temples",
        help="Diretorio de destino do dataset",
    )
    parser.add_argument(
        "--group",
        default="all",
        choices=["intermediate", "advanced", "both", "training", "all"],
        help="Subconjunto de cenas a baixar",
    )
    parser.add_argument(
        "--modality",
        default="image",
        choices=["image", "video", "both"],
        help="Tipo de conteudo a baixar",
    )
    parser.add_argument(
        "--script-path",
        default="./artifacts/downloads/download_t2_dataset.py",
        help="Caminho local para o downloader oficial baixado temporariamente",
    )
    args = parser.parse_args()

    target_path = Path(args.pathname).expanduser().resolve()
    target_path.mkdir(parents=True, exist_ok=True)

    script_path = download_official_downloader(Path(args.script_path).expanduser().resolve())
    command = build_command(script_path, target_path, args.group, args.modality)

    print(f"[info] downloader oficial: {OFFICIAL_DOWNLOADER_URL}")
    print(f"[info] destino: {target_path}")
    print(f"[info] comando: {' '.join(command)}")

    completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())