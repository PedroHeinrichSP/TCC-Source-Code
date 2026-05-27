import unittest

from nvs_benchmark.cli import _validate_real_metrics
from nvs_benchmark.cli_extensions import validate_matrix_completeness
from nvs_benchmark.evaluation import BenchmarkMetrics


def _metric(**overrides):
    base = {
        "method": "nerf_static",
        "pairs": 4.0,
        "psnr": 22.0,
        "ssim": 0.8,
        "lpips": 0.2,
        "fps": 8.0,
        "vram_gb": 6.0,
        "train_seconds": 10.0,
        "inference_seconds": 2.0,
        "frame_time_ms": 12.0,
        "latency_p50_ms": 10.0,
        "latency_p90_ms": 14.0,
        "latency_p99_ms": 18.0,
    }
    base.update(overrides)
    return BenchmarkMetrics(**base)


class StrictResultsValidationTests(unittest.TestCase):
    def test_accepts_valid_metrics(self):
        metric = _metric()
        error = _validate_real_metrics(metric, min_required_pairs=1)
        self.assertIsNone(error)

    def test_rejects_insufficient_pairs(self):
        metric = _metric(pairs=0.0)
        error = _validate_real_metrics(metric, min_required_pairs=1)
        self.assertIsNotNone(error)
        self.assertIn("Pairs insuficiente", error)

    def test_rejects_non_finite_value(self):
        metric = _metric(psnr=float("nan"))
        error = _validate_real_metrics(metric, min_required_pairs=1)
        self.assertIsNotNone(error)
        self.assertIn("nao finitos", error)

    def test_validate_matrix_completeness_passes(self):
        payload = {
            "blender_synthetic|nerf_static": {"psnr": 1.0},
            "blender_synthetic|gs_static": {"psnr": 2.0},
        }
        result = validate_matrix_completeness(
            payload,
            expected_combos=[
                "blender_synthetic|nerf_static",
                "blender_synthetic|gs_static",
            ],
            strict=True,
        )
        self.assertTrue(result.is_valid)

    def test_validate_matrix_completeness_rejects_missing_combo(self):
        payload = {
            "blender_synthetic|nerf_static": {"psnr": 1.0},
        }
        result = validate_matrix_completeness(
            payload,
            expected_combos=[
                "blender_synthetic|nerf_static",
                "blender_synthetic|gs_static",
            ],
            strict=True,
        )
        self.assertFalse(result.is_valid)
        self.assertIn("faltando", result.message)


if __name__ == "__main__":
    unittest.main()
