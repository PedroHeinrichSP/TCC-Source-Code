"""Utilitários para adaptadores reutilizarem frames reais em renders stub."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from matplotlib import image as mpimg


def _candidate_splits(preferred: str | None) -> list[str]:
    """Retorna lista priorizada de splits para procurar transforms."""
    splits = []
    if preferred:
        splits.append(preferred)
    for fallback in ("train", "test", "val"):
        if fallback not in splits:
            splits.append(fallback)
    return splits


def _load_transforms(root: Path, split: str) -> dict | None:
    """Lê transforms JSON de um split, retornando None se ausente/inválido."""
    transforms_path = root / f"transforms_{split}.json"
    if not transforms_path.exists():
        return None
    try:
        return json.loads(transforms_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _resolve_frame_path(root: Path, file_path: str) -> Path | None:
    """Resolve caminho de frame a partir de uma entrada de transforms."""
    if not file_path:
        return None

    candidate_paths = []
    raw = Path(file_path)
    if raw.is_absolute():
        candidate_paths.append(raw)
    else:
        candidate_paths.append(root / raw)

    if raw.suffix == "":
        candidate_paths.append(Path(str(candidate_paths[0]) + ".png"))
        candidate_paths.append(Path(str(candidate_paths[0]) + ".jpg"))
        candidate_paths.append(Path(str(candidate_paths[0]) + ".jpeg"))

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate
    return None


def find_first_frame(root: str | Path, split: str | None = None) -> Path | None:
    """Busca o primeiro caminho de frame referenciado no transforms JSON."""
    root_path = Path(root)
    if not root_path.exists():
        return None

    for candidate_split in _candidate_splits(split):
        payload = _load_transforms(root_path, candidate_split)
        if not payload or not isinstance(payload, dict):
            continue
        frames = payload.get("frames", [])
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            file_path = frame.get("file_path")
            if not isinstance(file_path, str):
                continue
            resolved = _resolve_frame_path(root_path, file_path)
            if resolved is not None:
                return resolved
    return None


def copy_frame_to_render_dir(frame_path: Path, render_dir: Path, target_name: str = "frame_0000.png") -> Path:
    """Copia um frame para o diretório de render com nome padronizado."""
    render_dir.mkdir(parents=True, exist_ok=True)
    output_path = render_dir / target_name
    shutil.copyfile(frame_path, output_path)
    return output_path


def render_stub_from_dataset(root: str | Path, split: str | None, render_dir: Path) -> bool:
    """Copia frame real do dataset para render e melhora realismo do stub."""
    frame_path = find_first_frame(root=root, split=split)
    if frame_path is None:
        return False
    copy_frame_to_render_dir(frame_path, render_dir)
    return True


def _save_image_as_png(source_path: Path, target_path: Path) -> None:
    """Re-salva uma imagem em PNG para padronizar nomes usados nas métricas."""
    image = mpimg.imread(source_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    mpimg.imsave(target_path, image)


def _candidate_reference_images(root_path: Path) -> list[Path]:
    """Coleta imagens candidatas diretamente na cena ou no diretório images/."""
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.PNG")
    candidates: list[Path] = []
    for folder in (root_path, root_path / "images", root_path / "image"):
        if not folder.exists() or not folder.is_dir():
            continue
        for pattern in patterns:
            candidates.extend(folder.glob(pattern))
    unique_candidates = sorted({candidate.resolve() for candidate in candidates}, key=lambda path: path.name.lower())
    return unique_candidates


def export_reference_frames_from_dataset(
    root: str | Path,
    split: str | None,
    reference_dir: Path,
) -> int:
    """Exporta frames de referência reais do dataset para um diretório temporário.

    As imagens são gravadas como ``frame_XXXX.png`` para manter o mesmo esquema
    de nomes esperado pelos renders produzidos pelos adaptadores do benchmark.
    """
    root_path = Path(root)
    if not root_path.exists() or not root_path.is_dir():
        return 0

    reference_dir.mkdir(parents=True, exist_ok=True)
    for old_file in reference_dir.glob("*.png"):
        old_file.unlink(missing_ok=True)

    copied = 0
    for candidate_split in _candidate_splits(split):
        payload = _load_transforms(root_path, candidate_split)
        if not payload or not isinstance(payload, dict):
            continue

        frames = payload.get("frames", [])
        if not isinstance(frames, list) or not frames:
            continue

        for index, frame in enumerate(frames):
            if not isinstance(frame, dict):
                continue
            file_path = frame.get("file_path")
            if not isinstance(file_path, str) or not file_path.strip():
                continue
            resolved = _resolve_frame_path(root_path, file_path)
            if resolved is None:
                continue

            target_path = reference_dir / f"frame_{index:04d}.png"
            _save_image_as_png(resolved, target_path)
            copied += 1

        if copied > 0:
            return copied

    image_candidates = _candidate_reference_images(root_path)
    if image_candidates:
        for index, resolved in enumerate(image_candidates):
            target_path = reference_dir / f"frame_{index:04d}.png"
            _save_image_as_png(resolved, target_path)
        return len(image_candidates)

    return copied
