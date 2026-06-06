from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import image as mpimg

from nvs_benchmark.evaluation.quality import evaluate_quality_metrics


def test_evaluate_quality_metrics_composites_rgba_references_on_white(tmp_path: Path) -> None:
    pred_dir = tmp_path / "pred"
    ref_dir = tmp_path / "ref"
    pred_dir.mkdir()
    ref_dir.mkdir()

    pred = np.ones((8, 8, 3), dtype=np.float32)
    ref = np.zeros((8, 8, 4), dtype=np.float32)
    ref[..., :3] = 0.0
    ref[..., 3] = 0.0

    mpimg.imsave(pred_dir / "frame_0000.png", pred)
    mpimg.imsave(ref_dir / "frame_0000.png", ref)

    metrics = evaluate_quality_metrics(pred_dir=pred_dir, ref_dir=ref_dir)

    assert metrics["pairs"] == 1.0
    assert metrics["psnr"] > 90.0
    assert metrics["ssim"] > 0.99
    assert metrics["lpips"] < 0.01


def test_evaluate_quality_metrics_can_limit_pairs(tmp_path: Path, monkeypatch) -> None:
    pred_dir = tmp_path / "pred"
    ref_dir = tmp_path / "ref"
    pred_dir.mkdir()
    ref_dir.mkdir()

    for index in range(5):
        image = np.full((8, 8, 3), index / 5.0, dtype=np.float32)
        mpimg.imsave(pred_dir / f"frame_{index:04d}.png", image)
        mpimg.imsave(ref_dir / f"frame_{index:04d}.png", image)

    monkeypatch.setattr("nvs_benchmark.evaluation.quality._compute_lpips", lambda pred, ref: 0.0)

    metrics = evaluate_quality_metrics(pred_dir=pred_dir, ref_dir=ref_dir, max_pairs=2)

    assert metrics["pairs"] == 2.0
    assert metrics["psnr"] > 90.0


def test_evaluate_quality_metrics_can_downscale_before_lpips(tmp_path: Path, monkeypatch) -> None:
    pred_dir = tmp_path / "pred"
    ref_dir = tmp_path / "ref"
    pred_dir.mkdir()
    ref_dir.mkdir()

    pred = np.ones((96, 64, 3), dtype=np.float32)
    ref = np.ones((96, 64, 3), dtype=np.float32)
    mpimg.imsave(pred_dir / "frame_0000.png", pred)
    mpimg.imsave(ref_dir / "frame_0000.png", ref)

    seen_shapes: list[tuple[int, int]] = []

    def _fake_lpips(pred_image, ref_image) -> float:
        seen_shapes.append(pred_image.shape[:2])
        return 0.0

    monkeypatch.setattr("nvs_benchmark.evaluation.quality._compute_lpips", _fake_lpips)

    metrics = evaluate_quality_metrics(pred_dir=pred_dir, ref_dir=ref_dir, max_image_dim=48)

    assert metrics["pairs"] == 1.0
    assert seen_shapes == [(48, 32)]


def test_evaluate_quality_metrics_logs_ascii_progress_bar(tmp_path: Path, monkeypatch, capsys) -> None:
    pred_dir = tmp_path / "pred"
    ref_dir = tmp_path / "ref"
    pred_dir.mkdir()
    ref_dir.mkdir()

    for index in range(2):
        image = np.full((8, 8, 3), index / 2.0, dtype=np.float32)
        mpimg.imsave(pred_dir / f"frame_{index:04d}.png", image)
        mpimg.imsave(ref_dir / f"frame_{index:04d}.png", image)

    monkeypatch.setattr("nvs_benchmark.evaluation.quality._compute_lpips", lambda pred, ref: 0.0)

    evaluate_quality_metrics(pred_dir=pred_dir, ref_dir=ref_dir, log_every=1)
    output = capsys.readouterr().out

    assert "[metrics] pair=1/2 (50%) |##########----------|" in output
    assert "[metrics] pair=2/2 (100%) |####################|" in output
