from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from nvs_benchmark.reporting.report import generate_comparison_reports, generate_experiments_comparison_report


class ReportingTests(unittest.TestCase):
    def test_generate_comparison_reports_html(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot.json"
            snapshot.write_text(
                json.dumps(
                    {
                        "method_a": {
                            "psnr": 20.0,
                            "ssim": 0.8,
                            "lpips": 0.4,
                            "fps": 10.0,
                            "vram_gb": 8.0,
                            "train_seconds": 100.0,
                            "inference_seconds": 10.0,
                        },
                        "method_b": {
                            "psnr": 30.0,
                            "ssim": 0.9,
                            "lpips": 0.2,
                            "fps": 20.0,
                            "vram_gb": 6.0,
                            "train_seconds": 70.0,
                            "inference_seconds": 8.0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = generate_comparison_reports(
                snapshot_file=snapshot,
                output_dir=root / "reports",
                report_name="unit_report",
                generate_pdf=False,
            )

            html_path = Path(result["html_path"])
            self.assertTrue(html_path.exists())
            self.assertEqual(result["winner"], "method_b")

    def test_generate_experiments_comparison_report(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = root / "artifacts"
            reports = artifacts / "reports"
            metrics = artifacts / "metrics"
            reports.mkdir(parents=True, exist_ok=True)
            metrics.mkdir(parents=True, exist_ok=True)

            metric_a = metrics / "run_a.json"
            metric_b = metrics / "run_b.json"
            metric_a.write_text(
                json.dumps({"method_a": {"psnr": 22.0, "ssim": 0.82, "lpips": 0.35, "fps": 9.0}}),
                encoding="utf-8",
            )
            metric_b.write_text(
                json.dumps({"method_b": {"psnr": 29.0, "ssim": 0.9, "lpips": 0.21, "fps": 15.0}}),
                encoding="utf-8",
            )

            history = artifacts / "experiments_history.json"
            history.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "experiments": [
                            {
                                "run_id": "run-a",
                                "method": "method_a",
                                "dataset": "blender_synthetic",
                                "metrics_path": str(metric_a),
                                "metrics_summary": {"psnr": 22.0, "ssim": 0.82, "lpips": 0.35, "fps": 9.0},
                            },
                            {
                                "run_id": "run-b",
                                "method": "method_b",
                                "dataset": "blender_synthetic",
                                "metrics_path": str(metric_b),
                                "metrics_summary": {"psnr": 29.0, "ssim": 0.9, "lpips": 0.21, "fps": 15.0},
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            result = generate_experiments_comparison_report(
                artifacts_dir=artifacts,
                run_ids=["run-a", "run-b"],
                output_dir=reports,
                report_name="history_compare",
                generate_pdf=False,
            )

            html_path = Path(result["html_path"])
            self.assertTrue(html_path.exists())
            self.assertIn("method_b", result["winner"])

    def test_generate_comparison_reports_strict_snapshot_requires_expected_methods(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot_strict_missing.json"
            snapshot.write_text(
                json.dumps(
                    {
                        "method_a": {
                            "psnr": 24.0,
                            "ssim": 0.85,
                            "lpips": 0.30,
                            "fps": 12.0,
                            "vram_gb": 7.0,
                            "train_seconds": 90.0,
                            "inference_seconds": 9.0,
                            "frame_time_ms": 8.0,
                            "latency_p50_ms": 7.0,
                            "latency_p90_ms": 10.0,
                            "latency_p99_ms": 14.0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                generate_comparison_reports(
                    snapshot_file=snapshot,
                    output_dir=root / "reports",
                    report_name="strict_missing",
                    generate_pdf=False,
                    strict_snapshot=True,
                    expected_methods=["method_a", "method_b"],
                )

    def test_generate_comparison_reports_pdf_e2e(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot = root / "snapshot_pdf.json"
            snapshot.write_text(
                json.dumps(
                    {
                        "method_a": {
                            "psnr": 24.0,
                            "ssim": 0.85,
                            "lpips": 0.30,
                            "fps": 12.0,
                            "vram_gb": 7.0,
                            "train_seconds": 90.0,
                            "inference_seconds": 9.0,
                        },
                        "method_b": {
                            "psnr": 28.0,
                            "ssim": 0.90,
                            "lpips": 0.22,
                            "fps": 16.0,
                            "vram_gb": 6.5,
                            "train_seconds": 80.0,
                            "inference_seconds": 8.0,
                        },
                    }
                ),
                encoding="utf-8",
            )

            result = generate_comparison_reports(
                snapshot_file=snapshot,
                output_dir=root / "reports",
                report_name="pdf_e2e",
                generate_pdf=True,
            )

            html_path = Path(result["html_path"])
            self.assertTrue(html_path.exists())

            if "pdf_error" in result:
                self.skipTest(f"WeasyPrint indisponivel neste ambiente: {result['pdf_error']}")

            self.assertIn("pdf_path", result)
            pdf_path = Path(result["pdf_path"])
            self.assertTrue(pdf_path.exists())
            self.assertGreater(pdf_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
