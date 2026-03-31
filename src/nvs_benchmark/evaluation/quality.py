"""Quality metrics (PSNR, SSIM, LPIPS) for NVS."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from matplotlib import image as mpimg
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

_LPIPS_MODEL = None
_LPIPS_DEVICE = None


def _to_float32_rgb(image: np.ndarray) -> np.ndarray:
    """Normalize image to float32 in [0, 1] with 3 channels."""
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def _load_image(path: Path) -> np.ndarray:
    """Load image file as normalized RGB ndarray."""
    return _to_float32_rgb(mpimg.imread(path))


def _supported_images(root: Path) -> list[Path]:
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(root.glob(pattern))
    return sorted(files)


def _build_pairs(pred_dir: Path, ref_dir: Path) -> list[tuple[Path, Path]]:
    """Build filename-aligned pairs between predicted and reference folders."""
    pred_files = {p.name: p for p in _supported_images(pred_dir)}
    ref_files = {p.name: p for p in _supported_images(ref_dir)}
    common = sorted(set(pred_files.keys()) & set(ref_files.keys()))
    return [(pred_files[name], ref_files[name]) for name in common]


def _get_lpips_model():
    """Load LPIPS model lazily and reuse across calls."""
    global _LPIPS_MODEL, _LPIPS_DEVICE
    if _LPIPS_MODEL is None:
        try:
            import lpips
            import torch
        except Exception as exc:
            raise RuntimeError(
                "LPIPS is mandatory for the benchmark. "
                "Install the 'lpips' and 'torch' dependencies before running."
            ) from exc
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _LPIPS_DEVICE = device
        _LPIPS_MODEL = lpips.LPIPS(net="alex").to(device)
    return _LPIPS_MODEL, _LPIPS_DEVICE


def _compute_lpips(pred: np.ndarray, ref: np.ndarray) -> float:
    """Compute LPIPS (mandatory) between two normalized RGB images."""
    import torch

    loss_fn, device = _get_lpips_model()
    pred_t = torch.from_numpy(pred.transpose(2, 0, 1)).unsqueeze(0).to(device)
    ref_t = torch.from_numpy(ref.transpose(2, 0, 1)).unsqueeze(0).to(device)

    pred_t = pred_t * 2.0 - 1.0
    ref_t = ref_t * 2.0 - 1.0

    with torch.no_grad():
        value = loss_fn(pred_t, ref_t)
    return float(value.item())


def _nanmean(values: Iterable[float]) -> float:
    arr = np.array(list(values), dtype=np.float32)
    if arr.size == 0:
        return float("nan")
    if np.isnan(arr).all():
        return float("nan")
    return float(np.nanmean(arr))


def evaluate_quality_metrics(
    pred_dir: str | Path,
    ref_dir: str | Path,
) -> dict[str, float]:
    """Evaluate PSNR, SSIM, and LPIPS between renders and references.

    Returns global means and the number of evaluated pairs.
    """
    pred_root = Path(pred_dir)
    ref_root = Path(ref_dir)
    pairs = _build_pairs(pred_root, ref_root)

    psnr_values: list[float] = []
    ssim_values: list[float] = []
    lpips_values: list[float] = []

    for pred_path, ref_path in pairs:
        pred = _load_image(pred_path)
        ref = _load_image(ref_path)

        if pred.shape != ref.shape:
            min_h = min(pred.shape[0], ref.shape[0])
            min_w = min(pred.shape[1], ref.shape[1])
            pred = pred[:min_h, :min_w, :]
            ref = ref[:min_h, :min_w, :]

        psnr_values.append(float(peak_signal_noise_ratio(ref, pred, data_range=1.0)))
        ssim_values.append(float(structural_similarity(ref, pred, data_range=1.0, channel_axis=2)))
        lpips_values.append(_compute_lpips(pred, ref))

    return {
        "pairs": float(len(pairs)),
        "psnr": _nanmean(psnr_values),
        "ssim": _nanmean(ssim_values),
        "lpips": _nanmean(lpips_values),
    }
