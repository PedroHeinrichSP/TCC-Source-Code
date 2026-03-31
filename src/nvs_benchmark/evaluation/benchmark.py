"""Aggregation of quality and performance metrics per method."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from .performance import RuntimePerformance, collect_runtime_performance
from .quality import evaluate_quality_metrics


@dataclass(frozen=True)
class BenchmarkMetrics:
    """Consolidated quality and performance results for a method."""

    method: str
    pairs: float
    psnr: float
    ssim: float
    lpips: float
    fps: float
    vram_gb: float
    train_seconds: float
    inference_seconds: float
    frame_time_ms: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p99_ms: float


def evaluate_benchmark_metrics(
    *,
    method: str,
    pred_dir: str | Path,
    ref_dir: str | Path,
    frames: int,
    train_seconds: float,
    inference_seconds: float,
    frame_times_ms: list[float] | None = None,
) -> BenchmarkMetrics:
    """Evaluate quality and performance metrics for a method."""
    quality = evaluate_quality_metrics(pred_dir=pred_dir, ref_dir=ref_dir)
    perf: RuntimePerformance = collect_runtime_performance(
        frames=frames,
        train_seconds=train_seconds,
        inference_seconds=inference_seconds,
        frame_times_ms=frame_times_ms,
    )

    return BenchmarkMetrics(
        method=method,
        pairs=quality["pairs"],
        psnr=quality["psnr"],
        ssim=quality["ssim"],
        lpips=quality["lpips"],
        fps=perf.fps,
        vram_gb=perf.vram_gb_peak,
        train_seconds=perf.train_seconds,
        inference_seconds=perf.inference_seconds,
        frame_time_ms=perf.frame_time_ms,
        latency_p50_ms=perf.latency_p50_ms,
        latency_p90_ms=perf.latency_p90_ms,
        latency_p99_ms=perf.latency_p99_ms,
    )


def save_metrics_snapshot(metrics: list[BenchmarkMetrics], output_file: str | Path) -> None:
    """Save JSON snapshot of metrics for UI/report consumption."""

    def sanitize(value):
        if isinstance(value, dict):
            return {key: sanitize(inner) for key, inner in value.items()}
        if isinstance(value, list):
            return [sanitize(inner) for inner in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {entry.method: sanitize(asdict(entry)) for entry in metrics}
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
