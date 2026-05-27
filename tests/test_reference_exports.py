"""Testes para exportação de referências usadas nas métricas."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib import image as mpimg

from nvs_benchmark.methods.utils import export_reference_frames_from_dataset


def test_export_reference_frames_from_dataset(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    train_dir = dataset_root / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    image_a = np.zeros((8, 8, 3), dtype=np.float32)
    image_a[..., 0] = 1.0
    image_b = np.zeros((8, 8, 3), dtype=np.float32)
    image_b[..., 1] = 1.0

    mpimg.imsave(train_dir / "r_0.png", image_a)
    mpimg.imsave(train_dir / "r_1.png", image_b)

    transforms = {
        "camera_angle_x": 0.6911112070083618,
        "frames": [
            {"file_path": "train/r_0"},
            {"file_path": "train/r_1"},
        ],
    }
    (dataset_root / "transforms_train.json").write_text(json.dumps(transforms), encoding="utf-8")

    reference_dir = tmp_path / "references"
    copied = export_reference_frames_from_dataset(dataset_root, "train", reference_dir)

    assert copied == 2
    assert (reference_dir / "frame_0000.png").exists()
    assert (reference_dir / "frame_0001.png").exists()