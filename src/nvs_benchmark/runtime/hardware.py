"""Hardware detection helpers for adaptive presets and estimate-time."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareSnapshot:
    """Detected runtime hardware characteristics."""

    has_gpu: bool
    gpu_name: str | None
    vram_gb: float | None
    recommended_profile: str


def detect_hardware_snapshot() -> HardwareSnapshot:
    """Detect GPU/VRAM and return a conservative hardware profile recommendation."""
    has_gpu = False
    gpu_name = None
    vram_gb = None

    try:
        import torch

        if torch.cuda.is_available():
            has_gpu = True
            props = torch.cuda.get_device_properties(0)
            gpu_name = str(props.name)
            vram_gb = float(props.total_memory / (1024**3))
    except Exception:
        has_gpu = False
        gpu_name = None
        vram_gb = None

    if not has_gpu:
        recommended = "low"
    elif vram_gb is None:
        recommended = "medium"
    elif vram_gb < 6.0:
        recommended = "low"
    elif vram_gb < 10.0:
        recommended = "medium"
    else:
        recommended = "high"

    return HardwareSnapshot(
        has_gpu=has_gpu,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        recommended_profile=recommended,
    )
