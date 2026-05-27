from __future__ import annotations

import json
from pathlib import Path

from nvs_benchmark.cli_extensions import estimate_execution_time, validate_dataset_integrity_preflight, validate_dataset_path
from nvs_benchmark.core import HardwareProfile
from nvs_benchmark.core.presets import resolve_iterations
from nvs_benchmark.data.validation import validate_dataset_integrity


def _write_minimal_png(path: Path) -> None:
    path.write_bytes(
        bytes.fromhex(
            "89504E470D0A1A0A"
            "0000000D4948445200000001000000010802000000907753DE"
            "0000000C49444154789C6360F8CF0000020201004F94CEBE"
            "0000000049454E44AE426082"
        )
    )


def test_validate_dataset_integrity_ok(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    train = root / "train"
    train.mkdir(parents=True, exist_ok=True)

    _write_minimal_png(train / "frame_000.png")
    transforms = {
        "camera_angle_x": 0.7,
        "frames": [
            {
                "file_path": "./train/frame_000.png",
                "transform_matrix": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        ],
    }
    (root / "transforms_train.json").write_text(json.dumps(transforms), encoding="utf-8")

    report = validate_dataset_integrity(root, split="train", full_scan=True)
    assert report.is_valid
    assert report.checked_frames == 1


def test_validate_dataset_integrity_detects_missing_and_corrupt(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    train = root / "train"
    train.mkdir(parents=True, exist_ok=True)

    (train / "bad.png").write_bytes(b"not-a-real-png")
    transforms = {
        "camera_angle_x": 0.7,
        "frames": [
            {"file_path": "./train/missing.png", "transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
            {"file_path": "./train/bad.png", "transform_matrix": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]},
        ],
    }
    (root / "transforms_train.json").write_text(json.dumps(transforms), encoding="utf-8")

    report = validate_dataset_integrity(root, split="train", full_scan=True)
    assert not report.is_valid
    assert len(report.missing_images) == 1
    assert len(report.corrupted_images) == 1


def test_tanks_and_temples_validation_rejects_mipnerf360_root(tmp_path: Path) -> None:
    root = tmp_path / "mipnerf360" / "garden"
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "sparse" / "0").mkdir(parents=True, exist_ok=True)
    _write_minimal_png(root / "images" / "frame_000.png")

    result = validate_dataset_path("tanks_and_temples", str(root))

    assert not result.is_valid


def test_tanks_and_temples_validation_accepts_scene_under_image_sets(tmp_path: Path) -> None:
    root = tmp_path / "data" / "tanks_and_temples" / "image_sets" / "Family"
    root.mkdir(parents=True, exist_ok=True)
    _write_minimal_png(root / "frame_000.png")

    path_result = validate_dataset_path("tanks_and_temples", str(root))
    preflight_result = validate_dataset_integrity_preflight(
        dataset_name="tanks_and_temples",
        root=str(root),
        split="train",
        full_scan=False,
    )

    assert path_result.is_valid
    assert preflight_result.is_valid


def test_resolve_iterations_adaptive_low_profile() -> None:
    params = resolve_iterations(
        method_id="nerf_static",
        preset_name="quick",
        extra={"adaptive_preset": True, "detected_hardware": {"has_gpu": False}},
        hardware_profile=HardwareProfile.ADAPTIVE,
    )
    assert params["N_iter"] == 500


def test_estimate_time_adaptive_never_increases_iterations() -> None:
    estimate = estimate_execution_time(
        method_id="nerf_static",
        preset="quick",
        adaptive_preset=True,
    )
    assert estimate["adjusted_iterations"] <= estimate["iterations"]
