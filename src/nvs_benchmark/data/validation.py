"""Dataset integrity validation helpers for preflight checks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")


@dataclass(frozen=True)
class DataValidationReport:
    """Structured report for dataset integrity preflight."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]
    checked_frames: int
    total_frames: int
    corrupted_images: list[str]
    missing_images: list[str]
    split: str


def validate_dataset_integrity(
    root: str | Path,
    *,
    split: str = "train",
    full_scan: bool = False,
    sample_size: int = 16,
) -> DataValidationReport:
    """Validate transforms schema and image integrity for a dataset split.

    The validator scans all frames when ``full_scan`` is True, otherwise a
    deterministic sample is used for faster preflight checks.
    """
    dataset_root = Path(root)
    errors: list[str] = []
    warnings: list[str] = []
    corrupted_images: list[str] = []
    missing_images: list[str] = []

    transforms_path = dataset_root / f"transforms_{split}.json"
    if not transforms_path.exists():
        fallback = dataset_root / "transforms_train.json"
        if fallback.exists():
            transforms_path = fallback
            warnings.append(f"split '{split}' nao encontrado; usando transforms_train.json")
        else:
            errors.append(f"arquivo de transforms nao encontrado: {transforms_path}")
            return DataValidationReport(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                checked_frames=0,
                total_frames=0,
                corrupted_images=corrupted_images,
                missing_images=missing_images,
                split=split,
            )

    try:
        transforms = json.loads(transforms_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        errors.append(f"transforms JSON invalido: {exc}")
        return DataValidationReport(
            is_valid=False,
            errors=errors,
            warnings=warnings,
            checked_frames=0,
            total_frames=0,
            corrupted_images=corrupted_images,
            missing_images=missing_images,
            split=split,
        )

    frames = transforms.get("frames")
    if not isinstance(frames, list):
        errors.append("transforms.json invalido: campo 'frames' ausente ou nao-lista")
        return DataValidationReport(
            is_valid=False,
            errors=errors,
            warnings=warnings,
            checked_frames=0,
            total_frames=0,
            corrupted_images=corrupted_images,
            missing_images=missing_images,
            split=split,
        )

    if "camera_angle_x" in transforms and not isinstance(transforms.get("camera_angle_x"), (int, float)):
        errors.append("transforms.json invalido: 'camera_angle_x' deve ser numerico")

    total_frames = len(frames)
    indices = _sample_indices(total_frames, sample_size) if not full_scan else list(range(total_frames))

    for index in indices:
        frame = frames[index]
        if not isinstance(frame, dict):
            errors.append(f"frame[{index}] invalido: esperado objeto")
            continue

        transform_matrix = frame.get("transform_matrix")
        if transform_matrix is not None and not _is_valid_transform_matrix(transform_matrix):
            errors.append(f"frame[{index}] transform_matrix invalida (esperado 4x4 numerico)")

        file_path = frame.get("file_path")
        if not isinstance(file_path, str) or not file_path.strip():
            errors.append(f"frame[{index}] invalido: file_path ausente")
            continue

        image_path = _resolve_frame_image(dataset_root, file_path)
        if image_path is None:
            missing_images.append(file_path)
            continue

        if not _is_image_readable(image_path):
            corrupted_images.append(image_path.relative_to(dataset_root).as_posix())

    if missing_images:
        errors.append(f"{len(missing_images)} imagem(ns) referenciada(s) nao encontrada(s)")
    if corrupted_images:
        errors.append(f"{len(corrupted_images)} imagem(ns) corrompida(s) detectada(s)")

    return DataValidationReport(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        checked_frames=len(indices),
        total_frames=total_frames,
        corrupted_images=corrupted_images,
        missing_images=missing_images,
        split=split,
    )


def _sample_indices(total: int, sample_size: int) -> list[int]:
    if total <= 0:
        return []
    if sample_size <= 0 or sample_size >= total:
        return list(range(total))
    step = max(total // sample_size, 1)
    return list(range(0, total, step))[:sample_size]


def _is_valid_transform_matrix(value: Any) -> bool:
    if not isinstance(value, list) or len(value) != 4:
        return False
    for row in value:
        if not isinstance(row, list) or len(row) != 4:
            return False
        for cell in row:
            if not isinstance(cell, (int, float)):
                return False
    return True


def _resolve_frame_image(dataset_root: Path, frame_path: str) -> Path | None:
    normalized = frame_path.replace("\\", "/").lstrip("./")
    candidate = dataset_root / normalized
    if candidate.exists() and candidate.is_file():
        return candidate

    if candidate.suffix:
        return None

    for suffix in _IMAGE_SUFFIXES:
        with_suffix = candidate.with_suffix(suffix)
        if with_suffix.exists() and with_suffix.is_file():
            return with_suffix
    return None


def _is_image_readable(path: Path) -> bool:
    if path.stat().st_size <= 0:
        return False
    if not _has_supported_image_signature(path):
        return False

    try:
        from skimage import io as skio

        image = skio.imread(str(path))
        return image is not None and getattr(image, "size", 0) > 0
    except Exception:
        # Header-level check is kept as fallback when skimage backend is unavailable.
        return True


def _has_supported_image_signature(path: Path) -> bool:
    try:
        header = path.read_bytes()[:16]
    except Exception:
        return False

    is_png = header.startswith(b"\x89PNG\r\n\x1a\n")
    is_jpeg = header.startswith(b"\xff\xd8\xff")
    is_webp = header.startswith(b"RIFF") and len(header) >= 12 and header[8:12] == b"WEBP"
    return is_png or is_jpeg or is_webp
