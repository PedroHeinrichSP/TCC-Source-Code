from pathlib import Path

from nvs_benchmark.core import (
    DatasetSpec,
    HardwareProfile,
    InferenceRequest,
    InferenceResult,
    MethodCapabilities,
    PerformanceStats,
    RunConfig,
)
from nvs_benchmark.core.orchestrator import Orchestrator
from nvs_benchmark.evaluation.benchmark import BenchmarkMetrics


def test_resolve_reference_split_defaults_to_dataset_split_without_override():
    config = RunConfig(
        run_id="test",
        dataset=DatasetSpec(name="blender_synthetic", root="/tmp/dataset", split="train"),
        method="gs_static",
        hardware_profile=HardwareProfile.ADAPTIVE,
        extra={},
    )

    assert Orchestrator._resolve_reference_split(config) == "train"


def test_resolve_reference_split_respects_explicit_override():
    config = RunConfig(
        run_id="test",
        dataset=DatasetSpec(name="blender_synthetic", root="/tmp/dataset", split="train"),
        method="gs_static",
        hardware_profile=HardwareProfile.ADAPTIVE,
        extra={"reference_split": "val"},
    )

    assert Orchestrator._resolve_reference_split(config) == "val"


def test_resolve_reference_split_uses_method_eval_split_when_available():
    config = RunConfig(
        run_id="test",
        dataset=DatasetSpec(name="blender_synthetic", root="/tmp/dataset", split="train"),
        method="gs_static",
        hardware_profile=HardwareProfile.ADAPTIVE,
        extra={"gs_eval_split": "test"},
    )

    assert Orchestrator._resolve_reference_split(config) == "test"


def test_resolve_reference_split_uses_dynamic_method_eval_split_when_available():
    config = RunConfig(
        run_id="test",
        dataset=DatasetSpec(name="d_nerf", root="/tmp/dataset", split="train"),
        method="gs_dynamic",
        hardware_profile=HardwareProfile.ADAPTIVE,
        extra={"gs_dynamic_eval_split": "test"},
    )

    assert Orchestrator._resolve_reference_split(config) == "test"


class _InferenceReferenceMethod:
    method_id = "gs_static"
    capabilities = MethodCapabilities(
        supports_train=False,
        supports_inference=True,
        supports_dynamic_scene=False,
        supports_limited_gpu=True,
    )

    def __init__(self, pred_dir: Path, ref_dir: Path):
        self._pred_dir = pred_dir
        self._ref_dir = ref_dir

    def validate_config(self, config: RunConfig) -> None:
        return None

    def infer(self, request: InferenceRequest) -> InferenceResult:
        return InferenceResult(
            method=self.method_id,
            rendered_dir=str(self._pred_dir),
            frames=1,
            inference_seconds=0.1,
            logs={"reference_dir": str(self._ref_dir)},
        )

    def collect_performance(self) -> PerformanceStats:
        return PerformanceStats(
            fps=10.0,
            vram_gb_peak=1.0,
            train_seconds=0.0,
            inference_seconds=0.1,
        )


def test_orchestrator_prefers_backend_reference_dir(monkeypatch, tmp_path):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    backend_ref_dir = tmp_path / "backend_refs"
    backend_ref_dir.mkdir()
    (pred_dir / "frame_0000.png").write_bytes(b"png")
    (backend_ref_dir / "frame_0000.png").write_bytes(b"png")

    method = _InferenceReferenceMethod(pred_dir=pred_dir, ref_dir=backend_ref_dir)
    orchestrator = Orchestrator()
    orchestrator.registry.register(method)

    captured: dict[str, str] = {}

    def fake_evaluate_benchmark_metrics(**kwargs):
        captured["ref_dir"] = str(kwargs["ref_dir"])
        return BenchmarkMetrics(
            method="gs_static",
            pairs=1.0,
            psnr=20.0,
            ssim=0.9,
            lpips=0.1,
            fps=10.0,
            vram_gb=1.0,
            train_seconds=0.0,
            inference_seconds=0.1,
            frame_time_ms=100.0,
            latency_p50_ms=100.0,
            latency_p90_ms=100.0,
            latency_p99_ms=100.0,
        )

    monkeypatch.setattr("nvs_benchmark.core.orchestrator.evaluate_benchmark_metrics", fake_evaluate_benchmark_metrics)
    monkeypatch.setattr(
        "nvs_benchmark.core.orchestrator.export_reference_frames_from_dataset",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("dataset references should not be exported")),
    )

    config = RunConfig(
        run_id="test",
        dataset=DatasetSpec(name="custom", root=str(dataset_root), split="train"),
        method="gs_static",
        output_dir=str(tmp_path / "artifacts"),
        hardware_profile=HardwareProfile.ADAPTIVE,
        report_formats=[],
        extra={
            "checkpoint_path": str(tmp_path / "checkpoint"),
            "skip_snapshot": True,
            "skip_reports": True,
        },
    )

    result = orchestrator.run(config)

    assert captured["ref_dir"] == str(backend_ref_dir)
    assert result.metrics is not None
    assert result.metrics.pairs == 1.0
