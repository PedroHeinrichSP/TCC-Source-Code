"""Orquestração de pipeline para execuções completas do benchmark."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from nvs_benchmark.evaluation import BenchmarkMetrics, evaluate_benchmark_metrics, save_metrics_snapshot, write_reference_image
from nvs_benchmark.reporting import generate_comparison_reports
from nvs_benchmark.data.fingerprint import build_dataset_fingerprint

from .cache_registry import CacheEntry, CacheRegistry
from .contracts import InferenceRequest, InferenceResult, ReportFormat, RunArtifacts, RunConfig, TrainRequest
from .experiments import ExperimentManager, ExperimentRecord
from .registry import MethodRegistry


@dataclass(frozen=True)
class OrchestratorResult:
    """Saída de uma única execução orquestrada."""

    artifacts: RunArtifacts
    metrics: BenchmarkMetrics | None


@dataclass
class Orchestrator:
    """Coordenador de alto nível para as etapas de treino/inferência/métricas/relatório."""

    name: str = "nvs-benchmark-orchestrator"
    registry: MethodRegistry = field(default_factory=MethodRegistry)

    @staticmethod
    def _config_digest(config: RunConfig) -> str:
        payload = {
            "run_id": config.run_id,
            "dataset": {
                "name": config.dataset.name,
                "root": config.dataset.root,
                "split": config.dataset.split,
                "metadata": config.dataset.metadata,
            },
            "method": config.method,
            "seed": config.seed,
            "output_dir": config.output_dir,
            "hardware_profile": config.hardware_profile.value,
            "extra": config.extra,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

    @staticmethod
    def _metrics_from_cache_payload(payload: dict[str, Any]) -> BenchmarkMetrics:
        return BenchmarkMetrics(
            method=str(payload["method"]),
            pairs=float(payload["pairs"]),
            psnr=float(payload["psnr"]),
            ssim=float(payload["ssim"]),
            lpips=float(payload["lpips"]),
            fps=float(payload["fps"]),
            vram_gb=float(payload["vram_gb"]),
            train_seconds=float(payload["train_seconds"]),
            inference_seconds=float(payload["inference_seconds"]),
            frame_time_ms=float(payload["frame_time_ms"]),
            latency_p50_ms=float(payload["latency_p50_ms"]),
            latency_p90_ms=float(payload["latency_p90_ms"]),
            latency_p99_ms=float(payload["latency_p99_ms"]),
        )

    def run(self, config: RunConfig) -> OrchestratorResult:
        """Executa uma rodada completa de benchmark para um único método.

        Extras esperados em config.extra (opcional):
        - checkpoint_path: quando o método não suportar treinamento
        - reference_dir: pasta com imagens de referência
        - snapshot_file: caminho de saída para o snapshot JSON
        - report_output_dir: diretório de saída para os relatórios
        - report_name: nome-base para os arquivos de relatório
        - skip_metrics: pular validação de métricas de qualidade e desempenho
        - skip_snapshot: pular o salvamento de snapshots
        - skip_reports: pular a construção de relatórios indiferente de report_formats
        """
        method = self.registry.get(config.method)
        method.validate_config(config)
        cache_enabled = config.extra.get("cache_enabled", True)
        reuse_renders = config.extra.get("reuse_renders", True)
        reuse_metrics = config.extra.get("reuse_metrics", True)
        cache_max_size_gb = float(config.extra.get("cache_max_size_gb", 50.0))

        cache_registry = CacheRegistry(output_dir=config.output_dir)
        experiment_manager = ExperimentManager(output_dir=config.output_dir)

        dataset_fingerprint = build_dataset_fingerprint(config.dataset.root, split=config.dataset.split)
        config_digest = self._config_digest(config)

        if cache_enabled:
            dataset_cache_key = f"dataset:{config.dataset.name}:{dataset_fingerprint.key}"
            cache_registry.put(
                CacheEntry(
                    key=dataset_cache_key,
                    artifact_type="dataset_fingerprint",
                    payload={
                        "dataset": config.dataset.name,
                        "split": config.dataset.split,
                        "transforms_hash": dataset_fingerprint.transforms_hash,
                        "image_count": dataset_fingerprint.image_count,
                        "sampled_paths": dataset_fingerprint.sampled_paths,
                    },
                    metadata={
                        "dataset": config.dataset.name,
                        "split": config.dataset.split,
                        "dataset_key": dataset_fingerprint.key,
                    },
                )
            )

        checkpoint_path = None
        train_seconds = 0.0
        if method.capabilities.supports_train:
            train_result = method.train(TrainRequest(config=config))
            checkpoint_path = train_result.checkpoint_path
            train_seconds = train_result.train_seconds
        else:
            checkpoint_path = config.extra.get("checkpoint_path")
            if not checkpoint_path:
                raise ValueError(
                    f"Method '{config.method}' does not support training. "
                    "Provide 'checkpoint_path' in RunConfig.extra."
                )

        if not method.capabilities.supports_inference:
            raise ValueError(f"Method '{config.method}' does not support inference.")

        infer_result: InferenceResult | None = None
        if cache_enabled and reuse_renders:
            render_key = hashlib.sha256(
                (
                    f"render:{config.method}:{config.dataset.name}:{dataset_fingerprint.key}:"
                    f"{config.dataset.split}:{checkpoint_path}:{config_digest}"
                ).encode("utf-8")
            ).hexdigest()
            cached_render_entry = cache_registry.get(render_key)
            if cached_render_entry and cached_render_entry.path:
                cached_frames = int(cached_render_entry.metadata.get("frames", 0))
                infer_result = InferenceResult(
                    method=config.method,
                    rendered_dir=cached_render_entry.path,
                    frames=cached_frames,
                    inference_seconds=0.0,
                )

        if infer_result is None:
            infer_result = method.infer(
                InferenceRequest(
                    config=config,
                    checkpoint_path=str(checkpoint_path),
                    split=config.dataset.split,
                )
            )
            if cache_enabled and reuse_renders:
                render_key = hashlib.sha256(
                    (
                        f"render:{config.method}:{config.dataset.name}:{dataset_fingerprint.key}:"
                        f"{config.dataset.split}:{checkpoint_path}:{config_digest}"
                    ).encode("utf-8")
                ).hexdigest()
                cache_registry.put(
                    CacheEntry(
                        key=render_key,
                        artifact_type="rendered_images",
                        path=str(infer_result.rendered_dir),
                        metadata={
                            "method": config.method,
                            "dataset": config.dataset.name,
                            "dataset_key": dataset_fingerprint.key,
                            "split": config.dataset.split,
                            "frames": infer_result.frames,
                            "checkpoint_path": str(checkpoint_path),
                            "config_digest": config_digest,
                        },
                    )
                )

        metrics: BenchmarkMetrics | None = None
        metrics_path: str | None = None
        snapshot_path: str | None = None
        report_paths: list[str] = []

        if not config.extra.get("skip_metrics"):
            reference_dir = config.extra.get("reference_dir")
            if reference_dir:
                ref_dir = Path(reference_dir)
            else:
                ref_dir = Path(config.output_dir) / config.run_id / config.method / "references"
                write_reference_image(ref_dir)

            metrics_cache_key = hashlib.sha256(
                (
                    f"metrics:{config.method}:{config.dataset.name}:{dataset_fingerprint.key}:"
                    f"{infer_result.rendered_dir}:{ref_dir}:{config_digest}"
                ).encode("utf-8")
            ).hexdigest()
            cached_metrics = cache_registry.get(metrics_cache_key) if cache_enabled and reuse_metrics else None

            if cached_metrics and cached_metrics.payload:
                metrics = self._metrics_from_cache_payload(cached_metrics.payload)
            else:
                metrics = evaluate_benchmark_metrics(
                    method=config.method,
                    pred_dir=infer_result.rendered_dir,
                    ref_dir=ref_dir,
                    frames=infer_result.frames,
                    train_seconds=train_seconds,
                    inference_seconds=infer_result.inference_seconds,
                )
                if cache_enabled and reuse_metrics:
                    cache_registry.put(
                        CacheEntry(
                            key=metrics_cache_key,
                            artifact_type="benchmark_metrics",
                            payload={
                                "method": metrics.method,
                                "pairs": metrics.pairs,
                                "psnr": metrics.psnr,
                                "ssim": metrics.ssim,
                                "lpips": metrics.lpips,
                                "fps": metrics.fps,
                                "vram_gb": metrics.vram_gb,
                                "train_seconds": metrics.train_seconds,
                                "inference_seconds": metrics.inference_seconds,
                                "frame_time_ms": metrics.frame_time_ms,
                                "latency_p50_ms": metrics.latency_p50_ms,
                                "latency_p90_ms": metrics.latency_p90_ms,
                                "latency_p99_ms": metrics.latency_p99_ms,
                            },
                            metadata={
                                "method": config.method,
                                "dataset": config.dataset.name,
                                "dataset_key": dataset_fingerprint.key,
                                "config_digest": config_digest,
                            },
                        )
                    )

            if not config.extra.get("skip_snapshot"):
                snapshot_path = config.extra.get("snapshot_file") or str(
                    Path(config.output_dir) / "metrics" / f"{config.run_id}.json"
                )
                save_metrics_snapshot([metrics], snapshot_path)
                metrics_path = str(snapshot_path)

            want_html = ReportFormat.HTML in config.report_formats
            want_pdf = ReportFormat.PDF in config.report_formats
            if (want_html or want_pdf) and not config.extra.get("skip_reports"):
                report_output_dir = config.extra.get("report_output_dir") or str(Path(config.output_dir) / "reports")
                report_name = config.extra.get("report_name") or "benchmark_report"
                snapshot_for_report = metrics_path or snapshot_path
                if not snapshot_for_report:
                    raise RuntimeError("Snapshot nao disponivel para gerar relatorio.")
                report_result = generate_comparison_reports(
                    snapshot_file=snapshot_for_report,
                    output_dir=report_output_dir,
                    report_name=report_name,
                    generate_pdf=want_pdf,
                )
                report_paths.append(report_result["html_path"])
                if "pdf_path" in report_result:
                    report_paths.append(report_result["pdf_path"])

        if cache_enabled:
            cache_registry.evict_lru(max_size_gb=cache_max_size_gb)

        artifacts = RunArtifacts(
            run_id=config.run_id,
            method=config.method,
            dataset=config.dataset.name,
            checkpoint_path=str(checkpoint_path),
            rendered_dir=infer_result.rendered_dir,
            metrics_path=metrics_path,
            report_paths=report_paths,
        )

        experiment_manager.record(
            ExperimentRecord(
                run_id=config.run_id,
                timestamp=self._utc_now(),
                command="orchestrator-run",
                status="success",
                method=config.method,
                dataset=config.dataset.name,
                preset=config.extra.get("preset"),
                config_digest=config_digest,
                checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
                rendered_dir=str(infer_result.rendered_dir),
                metrics_path=metrics_path,
                report_paths=report_paths,
                metrics_summary=(
                    {
                        "psnr": metrics.psnr,
                        "ssim": metrics.ssim,
                        "lpips": metrics.lpips,
                        "fps": metrics.fps,
                    }
                    if metrics is not None
                    else None
                ),
                metadata={
                    "dataset_key": dataset_fingerprint.key,
                    "cache_enabled": cache_enabled,
                    "reuse_renders": reuse_renders,
                    "reuse_metrics": reuse_metrics,
                },
            )
        )
        return OrchestratorResult(artifacts=artifacts, metrics=metrics)
