"""Dataset fingerprint utilities for cache invalidation and dedup detection."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass(frozen=True)
class DatasetFingerprint:
    """Compact dataset fingerprint used as cache key component."""

    key: str
    transforms_hash: str
    image_count: int
    sampled_paths: list[str]


def build_dataset_fingerprint(root: str | Path, split: str = "train", sample_size: int = 8) -> DatasetFingerprint:
    """Build a deterministic fingerprint from transforms + sampled image metadata."""
    dataset_root = Path(root)
    transforms_path = dataset_root / f"transforms_{split}.json"
    if not transforms_path.exists():
        fallback = dataset_root / "transforms_train.json"
        transforms_path = fallback if fallback.exists() else transforms_path

    transforms_hash = "missing"
    if transforms_path.exists():
        payload = json.loads(transforms_path.read_text(encoding="utf-8-sig"))
        normalized = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        transforms_hash = hashlib.sha256(normalized).hexdigest()

    images = sorted(
        [item for item in dataset_root.rglob("*") if item.is_file() and item.suffix.lower() in _IMAGE_SUFFIXES],
        key=lambda path: path.as_posix().lower(),
    )

    sampled = _sample_images(images, sample_size)
    sample_tokens: list[str] = []
    for path in sampled:
        stat = path.stat()
        relative = path.relative_to(dataset_root).as_posix()
        sample_tokens.append(f"{relative}|{stat.st_size}|{int(stat.st_mtime)}")

    sample_signature = ";".join(sample_tokens)
    digest_source = f"{transforms_hash}|{len(images)}|{sample_signature}".encode("utf-8")
    key = hashlib.sha256(digest_source).hexdigest()

    return DatasetFingerprint(
        key=key,
        transforms_hash=transforms_hash,
        image_count=len(images),
        sampled_paths=[path.relative_to(dataset_root).as_posix() for path in sampled],
    )


def _sample_images(paths: list[Path], sample_size: int) -> list[Path]:
    if sample_size <= 0 or len(paths) <= sample_size:
        return paths

    step = max(len(paths) // sample_size, 1)
    sampled = [paths[index] for index in range(0, len(paths), step)]
    return sampled[:sample_size]
