"""Geração de dados sintéticos de referência para smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import image as mpimg


def _gradient(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    y = np.linspace(0.0, 1.0, 64, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    noise = rng.normal(0, 0.01, size=(64, 64)).astype(np.float32)
    img = np.stack([
        np.clip(xx + noise, 0.0, 1.0),
        np.clip(yy + noise, 0.0, 1.0),
        np.clip(0.5 * (xx + yy) + noise, 0.0, 1.0),
    ], axis=-1)
    return img


def write_reference_image(reference_dir: str | Path, file_name: str = "frame_0000.png") -> Path:
    """Gera imagem de referência sintética para smoke tests de métricas."""
    root = Path(reference_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / file_name
    mpimg.imsave(path, _gradient(seed=42))
    return path
