from types import SimpleNamespace

from nvs_benchmark.cli import run_install, run_methods_check, run_metrics_check
from nvs_benchmark.core import DatasetSpec, InferenceResult, TrainResult
from nvs_benchmark.evaluation import BenchmarkMetrics
from nvs_benchmark.methods.gs_static.adapter import GSStaticHardwareError


class _Registry:
    def __init__(self, methods):
        self._methods = methods

    def list_ids(self):
        return list(self._methods.keys())

    def get(self, method_id):
        return self._methods[method_id]


class _WorkingMethod:
    method_id = "nerf_static"

    def validate_config(self, config):
        return None

    def train(self, request):
        return TrainResult(
            method=self.method_id,
            checkpoint_path="./artifacts/checkpoint",
            train_seconds=1.0,
            output_dir="./artifacts",
        )

    def infer(self, request):
        return InferenceResult(
            method=self.method_id,
            rendered_dir="./artifacts/renders",
            frames=2,
            inference_seconds=0.5,
        )


class _CudaOnlyMethod:
    method_id = "gs_static"

    def validate_config(self, config):
        raise GSStaticHardwareError("gs_static requer CUDA para execucao real.")


def _metric(method):
    return BenchmarkMetrics(
        method=method,
        pairs=4.0,
        psnr=20.0,
        ssim=0.8,
        lpips=0.2,
        fps=10.0,
        vram_gb=0.0,
        train_seconds=1.0,
        inference_seconds=0.5,
        frame_time_ms=4.0,
        latency_p50_ms=3.0,
        latency_p90_ms=5.0,
        latency_p99_ms=7.0,
    )


def test_methods_check_skips_gs_static_without_cuda(monkeypatch, capsys):
    dataset = DatasetSpec(name="blender_synthetic", root="./data/blender_synthetic/nerf_synthetic/lego")
    registry = _Registry({"nerf_static": _WorkingMethod(), "gs_static": _CudaOnlyMethod()})

    monkeypatch.setattr("nvs_benchmark.cli._resolve_smoke_dataset_spec", lambda: dataset)
    monkeypatch.setattr("nvs_benchmark.cli.build_registry_with_all_methods", lambda: registry)

    exit_code = run_methods_check(output_dir="./artifacts", log_dir="./logs")
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "[gs_static] skipped:" in captured
    assert "Metodos pulados por hardware: ['gs_static']" in captured
    assert "[nerf_static] checkpoint:" in captured


def test_metrics_check_skips_gs_static_without_cuda(monkeypatch, capsys):
    dataset = DatasetSpec(name="blender_synthetic", root="./data/blender_synthetic/nerf_synthetic/lego")
    registry = _Registry({"nerf_static": object(), "gs_static": object()})
    saved = {}

    class _FakeOrchestrator:
        def __init__(self, registry):
            self.registry = registry

        def run(self, config):
            if config.method == "gs_static":
                raise GSStaticHardwareError("gs_static requer CUDA para execucao real.")
            return SimpleNamespace(metrics=_metric(config.method))

    monkeypatch.setattr("nvs_benchmark.cli._resolve_smoke_dataset_spec", lambda: dataset)
    monkeypatch.setattr("nvs_benchmark.cli.build_registry_with_all_methods", lambda: registry)
    monkeypatch.setattr("nvs_benchmark.cli.Orchestrator", _FakeOrchestrator)
    monkeypatch.setattr(
        "nvs_benchmark.cli.save_metrics_snapshot",
        lambda metrics, snapshot_file: saved.update(
            {"metrics": list(metrics), "snapshot_file": snapshot_file}
        ),
    )

    exit_code = run_metrics_check(
        output_dir="./artifacts",
        snapshot_file="./artifacts/metrics/latest_preview.json",
        log_dir="./logs",
    )
    captured = capsys.readouterr().out

    assert exit_code == 0
    assert "[gs_static] skipped:" in captured
    assert len(saved["metrics"]) == 1
    assert saved["metrics"][0].method == "nerf_static"


def test_run_install_reports_missing_catalog(tmp_path, capsys):
    missing_catalog = tmp_path / "missing_catalog.json"

    exit_code = run_install(str(missing_catalog), only="datasets", execute=False)
    captured = capsys.readouterr().out

    assert exit_code == 1
    assert "Catalog not found:" in captured
    assert "Nenhum item encontrado para instalacao" in captured
