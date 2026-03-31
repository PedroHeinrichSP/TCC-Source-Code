from __future__ import annotations

from pathlib import Path

from nvs_benchmark.core.cache_registry import CacheEntry, CacheRegistry
from nvs_benchmark.core.experiments import ExperimentManager, ExperimentRecord
from nvs_benchmark.data.fingerprint import build_dataset_fingerprint


def test_cache_registry_put_get_payload(tmp_path: Path) -> None:
    registry = CacheRegistry(output_dir=tmp_path)
    entry = CacheEntry(
        key="metrics:test:1",
        artifact_type="benchmark_metrics",
        payload={"method": "nerf_static", "psnr": 10.0},
        metadata={"dataset": "blender_synthetic"},
    )
    registry.put(entry)

    loaded = registry.get("metrics:test:1")
    assert loaded is not None
    assert loaded.payload is not None
    assert loaded.payload["psnr"] == 10.0


def test_experiment_manager_record_and_filter(tmp_path: Path) -> None:
    manager = ExperimentManager(output_dir=tmp_path)
    manager.record(
        ExperimentRecord(
            run_id="run-1",
            timestamp="2026-03-30T10:00:00Z",
            command="method-run",
            status="success",
            method="nerf_static",
            dataset="blender_synthetic",
            preset="quick",
            config_digest="abc",
            checkpoint_path="/tmp/checkpoint",
            rendered_dir="/tmp/renders",
            metrics_path="/tmp/metrics.json",
            report_paths=[],
            metrics_summary={"psnr": 12.0, "ssim": 0.7, "lpips": 0.2, "fps": 1.1},
            metadata={"cache_enabled": True},
        )
    )

    rows = manager.list(method="nerf_static", status="success")
    assert len(rows) == 1
    assert rows[0]["dataset"] == "blender_synthetic"


def test_dataset_fingerprint_changes_when_transforms_change(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir(parents=True, exist_ok=True)

    transforms = root / "transforms_train.json"
    transforms.write_text('{"camera_angle_x": 0.7, "frames": []}', encoding="utf-8")

    image = root / "frame_000.png"
    image.write_bytes(b"not-a-real-image")

    first = build_dataset_fingerprint(root, split="train")

    transforms.write_text('{"camera_angle_x": 0.9, "frames": []}', encoding="utf-8")
    second = build_dataset_fingerprint(root, split="train")

    assert first.key != second.key
