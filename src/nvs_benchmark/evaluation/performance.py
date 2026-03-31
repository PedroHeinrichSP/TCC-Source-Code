"""Performance metrics for training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RuntimePerformance:
    """Summary of runtime performance for a run."""

    fps: float
    vram_gb_peak: float
    train_seconds: float
    inference_seconds: float
    frame_time_ms: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p99_ms: float


def _safe_vram_peak_gb() -> float:
    """Read peak VRAM (GB) when CUDA is available; otherwise return 0."""
    try:
        import torch
    except Exception:
        return 0.0

    if not torch.cuda.is_available():
        return 0.0
    peak_bytes = torch.cuda.max_memory_allocated()
    return float(peak_bytes / (1024**3))


def _percentiles(values_ms: Iterable[float]) -> tuple[float, float, float]:
    data = np.array(list(values_ms), dtype=np.float32)
    if data.size == 0:
        return 0.0, 0.0, 0.0
    p50, p90, p99 = np.percentile(data, [50, 90, 99]).astype(float)
    return float(p50), float(p90), float(p99)


def collect_runtime_performance(
    *,
    frames: int,
    train_seconds: float,
    inference_seconds: float,
    frame_times_ms: Iterable[float] | None = None,
) -> RuntimePerformance:
    """Compute FPS, VRAM peak, and latency percentiles for the run."""
    fps = float(frames / inference_seconds) if inference_seconds > 0 else 0.0
    if frame_times_ms is not None:
        p50, p90, p99 = _percentiles(frame_times_ms)
        frame_time_ms = p50
    else:
        frame_time_ms = float(inference_seconds / frames * 1000.0) if frames > 0 else 0.0
        p50 = frame_time_ms
        p90 = frame_time_ms
        p99 = frame_time_ms

    return RuntimePerformance(
        fps=fps,
        vram_gb_peak=_safe_vram_peak_gb(),
        train_seconds=float(train_seconds),
        inference_seconds=float(inference_seconds),
        frame_time_ms=frame_time_ms,
        latency_p50_ms=p50,
        latency_p90_ms=p90,
        latency_p99_ms=p99,
    )
