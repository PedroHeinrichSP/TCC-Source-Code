from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


OFFICIAL_DOWNLOADER_URL = (
    "https://raw.githubusercontent.com/IntelVCL/TanksAndTemples/master/python_toolbox/download_t2_dataset.py"
)


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

    image_pattern = re.compile(
        r"(?ms)        print\('\\nmd5 original:   ' \+ image_md5_dict\[scene\]\)\n"
        r"        md5_check = h_md5 == image_md5_dict\[scene\]\n\n"
        r"        if \(md5_check\):"
    )
    image_replacement = (
        "        expected_md5 = image_md5_dict.get(scene)\n"
        "        if expected_md5 is None:\n"
        "            print('\\nWarning: MD5 reference missing, extracting without verification.\\n')\n"
        "        else:\n"
        "            print('\\nmd5 original:   ' + expected_md5)\n"
        "            md5_check = h_md5 == expected_md5\n\n"
        "        if (expected_md5 is None or md5_check):"
    )
    patched_source = image_pattern.sub(image_replacement, patched_source, count=1)

    video_pattern = re.compile(
        r"(?ms)        print\('\\nmd5 original:   ' \+ video_md5_dict\[scene\]\)\n"
        r"        md5_check = h_md5 == video_md5_dict\[scene\]\n\n"
        r"        if \(md5_check\):"
    )
    video_replacement = (
        "        expected_md5 = video_md5_dict.get(scene)\n"
        "        if expected_md5 is None:\n"
        "            print('\\nWarning: MD5 reference missing, extracting without verification.\\n')\n"
        "        else:\n"
        "            print('\\nmd5 original:   ' + expected_md5)\n"
        "            md5_check = h_md5 == expected_md5\n\n"
        "        if (expected_md5 is None or md5_check):"
    )
    patched_source = video_pattern.sub(video_replacement, patched_source, count=1)

    if patched_source != source:
        script_path.write_text(patched_source, encoding="utf-8")


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
        "--calc_md5_off",
    ]


def extract_downloaded_archives(target_path: Path, modality: str) -> None:
    roots = []
    if modality in {"image", "both"}:
        roots.append(target_path / "image_sets")
    if modality in {"video", "both"}:
        roots.append(target_path / "videos")

    for root in roots:
        if not root.exists() or not root.is_dir():
            continue
        for zip_file in root.glob("*.zip"):
            scene_name = zip_file.stem
            extract_dir = root / scene_name
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            extract_dir.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(zip_file, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)
            except zipfile.BadZipFile as exc:
                try:
                    zip_file.unlink(missing_ok=True)
                except TypeError:
                    if zip_file.exists():
                        zip_file.unlink()
                raise RuntimeError(
                    f"Arquivo invalido baixado para {scene_name}: {zip_file}. "
                    "O downloader oficial retornou HTML/login page em vez do ZIP real. "
                    "Neste ambiente, o Tanks and Temples precisa de download manual/autenticado."
                ) from exc


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
    if completed.returncode == 0:
        extract_downloaded_archives(target_path, args.modality)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())