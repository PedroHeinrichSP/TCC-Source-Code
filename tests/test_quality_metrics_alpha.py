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
