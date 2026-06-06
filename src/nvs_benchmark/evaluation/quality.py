"""Quality metrics (PSNR, SSIM, LPIPS) for NVS."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
from matplotlib import image as mpimg
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

_LPIPS_MODEL = None
_LPIPS_DEVICE = None
_LPIPS_MIN_IMAGE_DIM = 32


def _to_float32_rgb(image: np.ndarray) -> np.ndarray:
    """Normalize image to float32 in [0, 1] with 3 channels."""
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    if image.ndim == 3 and image.shape[2] == 4:
        rgb = image[:, :, :3]
        alpha = image[:, :, 3:4]
        image = rgb * alpha + (1.0 - alpha)
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


def _limit_pairs_evenly(
    pairs: list[tuple[Path, Path]],
    max_pairs: int | None,
) -> list[tuple[Path, Path]]:
    """Seleciona pares distribuidos ao longo da sequencia para reduzir custo."""
    if max_pairs is None or max_pairs <= 0 or len(pairs) <= max_pairs:
        return pairs
    selected_indices = np.linspace(0, len(pairs) - 1, num=max_pairs, dtype=int)
    return [pairs[int(index)] for index in selected_indices]


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


def _downscale_if_needed(image: np.ndarray, max_image_dim: int | None) -> np.ndarray:
    """Reduz imagem preservando aspecto quando a maior dimensao excede o limite."""
    if max_image_dim is None or max_image_dim <= 0:
        return image

    height, width = image.shape[:2]
    longest_edge = max(height, width)
    if longest_edge <= max_image_dim:
        return image

    scale = float(max_image_dim) / float(longest_edge)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    if min(resized_width, resized_height) < _LPIPS_MIN_IMAGE_DIM:
        return image
    image_uint8 = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
    resampling_module = getattr(Image, "Resampling", Image)
    resized = Image.fromarray(image_uint8).resize((resized_width, resized_height), resampling_module.BILINEAR)
    return np.asarray(resized).astype(np.float32) / 255.0


def _ssim_win_size(pred: np.ndarray, ref: np.ndarray) -> int | None:
    """Resolve janela valida do SSIM para imagens pequenas."""
    min_dim = min(pred.shape[0], pred.shape[1], ref.shape[0], ref.shape[1])
    if min_dim < 3:
        return None
    win_size = min(7, int(min_dim))
    if win_size % 2 == 0:
        win_size -= 1
    return win_size if win_size >= 3 else None


def _resolve_log_every(total_pairs: int, log_every: int | None) -> int | None:
    """Resolve cadence de logs de progresso para o loop de metricas."""
    if total_pairs <= 0:
        return None
    if log_every is not None:
        return max(1, int(log_every))
    if total_pairs <= 20:
        return 1
    return max(1, math.ceil(total_pairs / 20))


def _format_progress_bar(done: int, total: int, width: int = 20) -> str:
    """Renderiza uma barra ASCII deterministica para progresso em logs."""
    if total <= 0:
        return "-" * width
    clamped_done = max(0, min(done, total))
    filled = int(round((clamped_done / total) * width))
    filled = max(0, min(filled, width))
    return f"{'#' * filled}{'-' * (width - filled)}"


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
    *,
    max_pairs: int | None = None,
    max_image_dim: int | None = None,
    log_every: int | None = None,
) -> dict[str, float]:
    """Evaluate PSNR, SSIM, and LPIPS between renders and references.

    Returns global means and the number of evaluated pairs.
    """
    pred_root = Path(pred_dir)
    ref_root = Path(ref_dir)
    pairs = _limit_pairs_evenly(_build_pairs(pred_root, ref_root), max_pairs)

    psnr_values: list[float] = []
    ssim_values: list[float] = []
    lpips_values: list[float] = []

    total_pairs = len(pairs)
    effective_log_every = _resolve_log_every(total_pairs, log_every)
    if total_pairs > 0:
        start_bar = _format_progress_bar(0, total_pairs)
        print(
            f"[metrics] pair=0/{total_pairs} (0%) |{start_bar}| starting "
            f"max_dim={max_image_dim or 'full'} log_every={effective_log_every or 'off'}",
            flush=True,
        )
    for index, (pred_path, ref_path) in enumerate(pairs, start=1):
        if effective_log_every and (index == 1 or index == total_pairs or index % effective_log_every == 0):
            pct = int(round((index / max(total_pairs, 1)) * 100))
            bar = _format_progress_bar(index, total_pairs)
            print(
                f"[metrics] pair={index}/{total_pairs} ({pct}%) |{bar}| "
                f"file={pred_path.name} max_dim={max_image_dim or 'full'}",
                flush=True,
            )
        pred = _load_image(pred_path)
        ref = _load_image(ref_path)

        if pred.shape != ref.shape:
            min_h = min(pred.shape[0], ref.shape[0])
            min_w = min(pred.shape[1], ref.shape[1])
            pred = pred[:min_h, :min_w, :]
            ref = ref[:min_h, :min_w, :]

        pred = _downscale_if_needed(pred, max_image_dim)
        ref = _downscale_if_needed(ref, max_image_dim)

        psnr_values.append(float(peak_signal_noise_ratio(ref, pred, data_range=1.0)))
        win_size = _ssim_win_size(pred, ref)
        if win_size is None:
            ssim_values.append(float("nan"))
        else:
            ssim_values.append(float(structural_similarity(ref, pred, data_range=1.0, channel_axis=2, win_size=win_size)))
        lpips_values.append(_compute_lpips(pred, ref))

    return {
        "pairs": float(len(pairs)),
        "psnr": _nanmean(psnr_values),
        "ssim": _nanmean(ssim_values),
        "lpips": _nanmean(lpips_values),
    }
