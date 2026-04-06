"""UI de pre-visualizacao em Viser para metricas e frustums de camera."""

from __future__ import annotations

import json
import math
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import viser
from skimage import io as skio

DEFAULT_METHOD_IDS = ["nerf_static", "nerf_dynamic", "gs_static", "gs_dynamic"]

PREVIEW_METRICS = {
    "nerf_static": {"psnr": 26.4, "ssim": 0.91, "lpips": 0.19, "fps": 1.2, "vram_gb": 6.8},
    "nerf_dynamic": {"psnr": 24.8, "ssim": 0.89, "lpips": 0.23, "fps": 0.9, "vram_gb": 8.5},
    "gs_static": {"psnr": 27.1, "ssim": 0.92, "lpips": 0.17, "fps": 38.0, "vram_gb": 7.1},
    "gs_dynamic": {"psnr": 25.6, "ssim": 0.90, "lpips": 0.20, "fps": 26.0, "vram_gb": 9.4},
}

DEFAULT_INSTALL_CATALOG = {
    "datasets": [
        {
            "id": "blender_synthetic_smoke",
            "label": "Blender Synthetic (smoke)",
            "path": "./data/_smoke/blender",
            "url": "",
        },
        {
            "id": "d_nerf",
            "label": "D-NeRF",
            "path": "./data/d_nerf",
            "url": "",
        },
    ],
    "methods": [
        {
            "id": "nerf_static",
            "label": "NeRF Static (bmild/nerf)",
            "path": "./third_party/nerf",
            "url": "https://github.com/bmild/nerf",
        },
        {
            "id": "nerf_dynamic",
            "label": "D-NeRF (albertpumarola/D-NeRF)",
            "path": "./third_party/d_nerf",
            "url": "https://github.com/albertpumarola/D-NeRF",
        },
        {
            "id": "gs_static",
            "label": "3D Gaussian Splatting",
            "path": "./third_party/gaussian_splatting",
            "url": "https://github.com/graphdeco-inria/gaussian-splatting",
        },
        {
            "id": "gs_dynamic",
            "label": "4D Gaussian Splatting",
            "path": "./third_party/4d_gaussians",
            "url": "https://github.com/hustvl/4DGaussians",
        },
    ],
    "notes": [
        "Preencha URLs de datasets conforme sua origem/licenca.",
        "Caminhos sao sugestoes; ajuste conforme sua estrutura local.",
    ],
}

FALLBACK_FOV_X = 0.691
DEFAULT_ASPECT = 4.0 / 3.0
MAX_SCENE_FRUSTUM_IMAGES = 24
VISUAL_HULL_GRID_RES = 28
VISUAL_HULL_MAX_VIEWS = 20


@dataclass
class _PreviewRuntimeState:
    server: viser.ViserServer
    artifacts_root: Path
    scene_transforms_file: str | None
    scene_payload: dict
    latest_camera: dict = field(default_factory=dict)
    latest_client_id: int | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


_PREVIEW_RUNTIMES: dict[int, _PreviewRuntimeState] = {}
_PREVIEW_RUNTIMES_LOCK = threading.Lock()


def _format_metric(value: float | None, digits: int = 2) -> str:
    """Formata metrica numerica para exibicao segura em tabela."""
    if value is None:
        return "--"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "--"
    if math.isnan(numeric) or math.isinf(numeric):
        return "--"
    return f"{numeric:.{digits}f}"


def _load_metrics_snapshot(metrics_file: str | None) -> dict[str, dict]:
    """Carrega snapshot de metricas de arquivo JSON, com fallback vazio."""
    if not metrics_file:
        return {}
    path = Path(metrics_file)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _load_install_catalog(catalog_file: str | None) -> dict:
    """Carrega catalogo de instalacao para datasets e modelos."""
    if not catalog_file:
        return dict(DEFAULT_INSTALL_CATALOG)
    path = Path(catalog_file)
    if not path.exists():
        return dict(DEFAULT_INSTALL_CATALOG)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else dict(DEFAULT_INSTALL_CATALOG)
    except Exception:
        return dict(DEFAULT_INSTALL_CATALOG)


def _is_ready(path_value: str | None) -> bool:
    if not path_value:
        return False
    return Path(path_value).exists()


def _status_badge(installed: bool, has_url: bool) -> str:
    if installed:
        return "[PRONTO]"
    if not has_url:
        return "[FALTA URL]"
    return "[BAIXAR]"


def _format_size(size_mb: float | None) -> str:
    if size_mb is None:
        return "--"
    try:
        value = float(size_mb)
    except (TypeError, ValueError):
        return "--"
    if value >= 1024.0:
        return f"{value / 1024.0:.2f} GB"
    return f"{value:.0f} MB"


def _copy_to_clipboard(text_value: str) -> bool:
    try:
        import tkinter as tk
    except Exception:
        return False
    try:
        root = tk.Tk()
        root.withdraw()
        root.clipboard_clear()
        root.clipboard_append(text_value)
        root.update()
        root.destroy()
        return True
    except Exception:
        return False


def _camera_angle_x_from_scene(scene_transforms_file: str | None) -> float | None:
    """Le camera_angle_x do arquivo de transforms da cena, quando disponivel."""
    if not scene_transforms_file:
        return None
    path = Path(scene_transforms_file)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            value = payload.get("camera_angle_x")
            return float(value) if value is not None else None
    except Exception:
        return None
    return None


def _scene_coordinate_convention(scene_transforms_file: str | None) -> str:
    """Resolve convention from transforms metadata: opencv (default) or opengl."""
    if not scene_transforms_file:
        return "opengl"
    path = Path(scene_transforms_file)
    if not path.exists():
        return "opengl"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "opengl"
    if not isinstance(payload, dict):
        return "opengl"
    raw = str(payload.get("coordinate_convention") or "").strip().lower()
    if raw in {"opencv", "opengl"}:
        return raw
    return "opengl"


def _matrix_pose_to_viser(matrix: np.ndarray, convention: str) -> tuple[np.ndarray, np.ndarray]:
    """Convert matrix pose to the same frame used for frustums in Viser."""
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    if convention == "opengl":
        flip = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
        return rotation @ flip, translation
    return rotation, translation


def _safe_matrices_from_scene(scene_transforms_file: str | None) -> list[np.ndarray]:
    """Carrega matrizes de camera 4x4 de transforms quando disponiveis."""
    if not scene_transforms_file:
        return []
    path = Path(scene_transforms_file)
    if not path.exists():
        return []

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        frames = payload.get("frames", []) if isinstance(payload, dict) else []
        matrices: list[np.ndarray] = []
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            matrix = frame.get("transform_matrix")
            if not isinstance(matrix, list):
                continue
            arr = np.array(matrix, dtype=np.float32)
            if arr.shape == (4, 4):
                matrices.append(arr)
        return matrices
    except Exception:
        return []


def _normalize_rgb_uint8(image: np.ndarray | None) -> np.ndarray | None:
    """Normaliza imagem para RGB uint8 para uso em frustums."""
    if image is None:
        return None
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[-1] != 3:
        return None
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _resolve_frame_image_path(scene_transforms_file: str, frame: dict) -> Path | None:
    """Resolve caminho de imagem de frame em formatos comuns de transforms NeRF."""
    file_path = frame.get("file_path")
    if not isinstance(file_path, str) or not file_path.strip():
        return None

    transforms_path = Path(scene_transforms_file)
    base_dir = transforms_path.parent
    normalized = file_path.replace("\\", "/")
    candidate = Path(normalized)

    if not candidate.is_absolute():
        candidate = base_dir / candidate

    variants = [candidate]
    if candidate.suffix == "":
        variants.extend([candidate.with_suffix(".png"), candidate.with_suffix(".jpg"), candidate.with_suffix(".jpeg")])

    for option in variants:
        if option.exists():
            return option
    return None


def _scene_frame_images(scene_transforms_file: str | None, frame_count: int) -> list[np.ndarray | None]:
    """Carrega imagens de frame para enriquecer frustums com preview visual."""
    if not scene_transforms_file:
        return [None] * frame_count
    path = Path(scene_transforms_file)
    if not path.exists():
        return [None] * frame_count

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return [None] * frame_count

    frames = payload.get("frames", []) if isinstance(payload, dict) else []
    if not isinstance(frames, list) or not frames:
        return [None] * frame_count

    # Evita sobrecarga visual/performance em cenas grandes.
    stride = max(1, len(frames) // max(1, MAX_SCENE_FRUSTUM_IMAGES))
    selected_indices = set(range(0, len(frames), stride))

    images: list[np.ndarray | None] = []
    for index in range(min(frame_count, len(frames))):
        frame = frames[index]
        if index not in selected_indices or not isinstance(frame, dict):
            images.append(None)
            continue

        image_path = _resolve_frame_image_path(scene_transforms_file, frame)
        if image_path is None:
            images.append(None)
            continue

        try:
            raw = skio.imread(image_path)
            images.append(_normalize_rgb_uint8(raw))
        except Exception:
            images.append(None)

    if frame_count > len(images):
        images.extend([None] * (frame_count - len(images)))
    return images[:frame_count]


def _scene_frame_rgba_images(scene_transforms_file: str | None, frame_count: int) -> list[np.ndarray | None]:
    """Carrega imagens RGBA dos frames para reconstrução leve por silhueta."""
    if not scene_transforms_file:
        return [None] * frame_count
    path = Path(scene_transforms_file)
    if not path.exists():
        return [None] * frame_count

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return [None] * frame_count

    frames = payload.get("frames", []) if isinstance(payload, dict) else []
    if not isinstance(frames, list) or not frames:
        return [None] * frame_count

    images: list[np.ndarray | None] = []
    for index in range(min(frame_count, len(frames))):
        frame = frames[index]
        if not isinstance(frame, dict):
            images.append(None)
            continue
        image_path = _resolve_frame_image_path(scene_transforms_file, frame)
        if image_path is None:
            images.append(None)
            continue
        try:
            rgba = skio.imread(image_path)
            if rgba is None or rgba.ndim < 3:
                images.append(None)
                continue
            if rgba.dtype != np.uint8:
                rgba = np.clip(rgba, 0, 255).astype(np.uint8)
            if rgba.shape[-1] == 3:
                alpha = np.full((rgba.shape[0], rgba.shape[1], 1), 255, dtype=np.uint8)
                rgba = np.concatenate([rgba, alpha], axis=-1)
            elif rgba.shape[-1] > 4:
                rgba = rgba[..., :4]
            images.append(rgba)
        except Exception:
            images.append(None)

    if frame_count > len(images):
        images.extend([None] * (frame_count - len(images)))
    return images[:frame_count]


def _project_world_points_to_image(
    points_world: np.ndarray,
    c2w: np.ndarray,
    fov_x: float,
    width: int,
    height: int,
    convention: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Projeta pontos do mundo para coordenadas de pixel no frame."""
    w2c = np.linalg.inv(c2w)
    points_h = np.concatenate([points_world, np.ones((points_world.shape[0], 1), dtype=np.float32)], axis=1)
    cam = (w2c @ points_h.T).T[:, :3]

    if convention == "opengl":
        z_forward = -cam[:, 2]
        y_cam = -cam[:, 1]
    else:
        z_forward = cam[:, 2]
        y_cam = cam[:, 1]
    valid_z = z_forward > 1e-4

    focal = 0.5 * float(width) / math.tan(0.5 * float(fov_x))
    x = (cam[:, 0] / np.maximum(z_forward, 1e-6)) * focal + (float(width) * 0.5)
    y = (y_cam / np.maximum(z_forward, 1e-6)) * focal + (float(height) * 0.5)

    inside = (
        valid_z
        & (x >= 0.0)
        & (x <= (width - 1))
        & (y >= 0.0)
        & (y <= (height - 1))
    )
    return x, y, inside


def _build_visual_hull_point_cloud(scene_payload: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Reconstrói nuvem de pontos aproximada por consistência de silhueta."""
    matrices: list[np.ndarray] = scene_payload.get("matrices", [])
    rgba_frames: list[np.ndarray | None] = scene_payload.get("frame_rgba_images", [])
    fov_x = float(scene_payload.get("fov_x", FALLBACK_FOV_X))
    convention = str(scene_payload.get("coordinate_convention") or "opencv").strip().lower()

    valid_views: list[tuple[np.ndarray, np.ndarray]] = []
    for index, matrix in enumerate(matrices[:VISUAL_HULL_MAX_VIEWS]):
        if index >= len(rgba_frames):
            break
        rgba = rgba_frames[index]
        if rgba is None or rgba.ndim != 3 or rgba.shape[-1] < 4:
            continue
        alpha_channel = rgba[..., 3]
        threshold = int(np.percentile(alpha_channel, 60))
        threshold = max(14, min(threshold, 220))
        alpha = alpha_channel >= threshold
        if int(alpha.sum()) < 50:
            continue
        valid_views.append((matrix.astype(np.float32), rgba.astype(np.uint8)))

    if len(valid_views) < 3:
        return None, None

    centers = np.stack([view[0][:3, 3] for view in valid_views], axis=0)
    radius = float(np.median(np.linalg.norm(centers, axis=1)))
    extent = max(0.5, min(1.8, radius * 0.55))

    axis = np.linspace(-extent, extent, VISUAL_HULL_GRID_RES, dtype=np.float32)
    gx, gy, gz = np.meshgrid(axis, axis, axis, indexing="xy")
    points = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)

    visible_counts = np.zeros(points.shape[0], dtype=np.int32)
    hit_counts = np.zeros(points.shape[0], dtype=np.int32)
    color_acc = np.zeros((points.shape[0], 3), dtype=np.float32)

    for c2w, rgba in valid_views:
        h, w = rgba.shape[:2]
        x, y, inside = _project_world_points_to_image(points, c2w, fov_x, w, h, convention)
        visible_counts += inside.astype(np.int32)
        if not np.any(inside):
            continue

        xi = np.clip(np.round(x[inside]).astype(np.int32), 0, w - 1)
        yi = np.clip(np.round(y[inside]).astype(np.int32), 0, h - 1)
        sampled = rgba[yi, xi]
        sample_threshold = max(14, int(np.percentile(sampled[:, 3], 55)))
        fg = sampled[:, 3] >= sample_threshold
        inside_indices = np.where(inside)[0]
        fg_indices = inside_indices[fg]
        if fg_indices.size == 0:
            continue

        hit_counts[fg_indices] += 1
        color_acc[fg_indices] += sampled[fg, :3].astype(np.float32)

    enough_visible = visible_counts >= 3
    consistent = hit_counts >= np.maximum(2, (visible_counts * 0.45).astype(np.int32))
    keep = enough_visible & consistent
    if not np.any(keep):
        return None, None

    cloud = points[keep]
    counts = np.maximum(hit_counts[keep][:, None], 1)
    colors = np.clip(color_acc[keep] / counts, 0.0, 255.0).astype(np.float32) / 255.0

    if cloud.shape[0] > 12000:
        stride = int(math.ceil(cloud.shape[0] / 12000.0))
        cloud = cloud[::stride]
        colors = colors[::stride]

    return cloud.astype(np.float32), colors.astype(np.float32)


def _synthetic_camera_matrices(count: int = 8) -> list[np.ndarray]:
    """Gera cameras sinteticas em orbita para pre-visualizacoes smoke."""
    matrices: list[np.ndarray] = []
    radius = 2.2
    for index in range(max(count, 3)):
        angle = 2.0 * np.pi * index / float(max(count, 3))
        x = radius * np.cos(angle)
        y = 0.8
        z = radius * np.sin(angle)

        forward = np.array([-x, -y, -z], dtype=np.float32)
        forward = forward / (np.linalg.norm(forward) + 1e-8)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(up, forward)
        right = right / (np.linalg.norm(right) + 1e-8)
        true_up = np.cross(forward, right)

        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, 0] = right
        c2w[:3, 1] = true_up
        c2w[:3, 2] = -forward
        c2w[:3, 3] = np.array([x, y, z], dtype=np.float32)
        matrices.append(c2w)
    return matrices


def _scene_payload(scene_transforms_file: str | None) -> dict:
    """Monta estrutura padronizada de cena com cameras reais ou contingencia."""
    matrices = _safe_matrices_from_scene(scene_transforms_file)
    coordinate_convention = _scene_coordinate_convention(scene_transforms_file)
    used_fallback = False
    if not matrices:
        matrices = _synthetic_camera_matrices(count=8)
        used_fallback = True

    fov_x = _camera_angle_x_from_scene(scene_transforms_file) or FALLBACK_FOV_X
    frame_images = _scene_frame_images(scene_transforms_file, len(matrices))
    frame_rgba_images = _scene_frame_rgba_images(scene_transforms_file, len(matrices))
    cloud_points, cloud_colors = _build_visual_hull_point_cloud(
        {
            "matrices": matrices,
            "frame_rgba_images": frame_rgba_images,
            "fov_x": float(fov_x),
            "coordinate_convention": coordinate_convention,
        }
    )
    if cloud_points is None or cloud_colors is None:
        alternate = "opencv" if coordinate_convention == "opengl" else "opengl"
        alt_points, alt_colors = _build_visual_hull_point_cloud(
            {
                "matrices": matrices,
                "frame_rgba_images": frame_rgba_images,
                "fov_x": float(fov_x),
                "coordinate_convention": alternate,
            }
        )
        if alt_points is not None and alt_colors is not None:
            cloud_points, cloud_colors = alt_points, alt_colors
            coordinate_convention = alternate
    return {
        "matrices": matrices,
        "frame_images": frame_images,
        "frame_rgba_images": frame_rgba_images,
        "scene_points": cloud_points,
        "scene_colors": cloud_colors,
        "camera_count": len(matrices),
        "used_fallback": used_fallback,
        "scene_source": scene_transforms_file or "(not provided)",
        "fov_x": float(fov_x),
        "aspect": DEFAULT_ASPECT,
        "coordinate_convention": coordinate_convention,
    }


def _build_metrics_markdown(selected_methods: list[str], metrics_snapshot: dict[str, dict]) -> str:
    """Gera tabela Markdown de metricas para os metodos selecionados."""
    lines = [
        "| Metodo | PSNR | SSIM | LPIPS | FPS | VRAM (GB) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in selected_methods:
        metrics = metrics_snapshot.get(method) or PREVIEW_METRICS.get(method)
        if not metrics:
            lines.append(
                "| {method} | -- | -- | -- | -- | -- |".format(method=method)
            )
            continue
        lines.append(
            "| {method} | {psnr} | {ssim} | {lpips} | {fps} | {vram} |".format(
                method=method,
                psnr=_format_metric(metrics.get("psnr")),
                ssim=_format_metric(metrics.get("ssim"), digits=3),
                lpips=_format_metric(metrics.get("lpips"), digits=3),
                fps=_format_metric(metrics.get("fps")),
                vram=_format_metric(metrics.get("vram_gb")),
            )
        )

    if len(lines) == 2:
        lines.append("| -- | -- | -- | -- | -- | -- |")
        lines.append("| Selecione ao menos um metodo para exibir metricas. | | | | | |")
    return "\n".join(lines)


def _resolve_method_ids(metrics_snapshot: dict[str, dict]) -> list[str]:
    """Resolve IDs de metodos a partir dos padroes e chaves do snapshot."""
    method_ids = list(DEFAULT_METHOD_IDS)
    for key in metrics_snapshot.keys():
        if key not in method_ids:
            method_ids.append(key)
    return method_ids


def _resolve_artifacts_root(metrics_file: str | None) -> Path:
    """Resolve diretorio raiz de artefatos com base no arquivo de metricas."""
    if metrics_file:
        metrics_path = Path(metrics_file).resolve()
        if metrics_path.exists():
            return metrics_path.parent.parent
    return Path("./artifacts").resolve()


def _find_method_image(artifacts_root: Path, method_id: str, kind: str) -> Path | None:
    """Localiza imagem de render/referencia de um metodo nos artefatos."""
    candidates = [
        artifacts_root / f"metrics-{method_id}" / method_id / kind / "frame_0000.png",
        artifacts_root / f"smoke-{method_id}" / method_id / kind / "frame_0000.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_image(path: Path) -> np.ndarray | None:
    """Carrega imagem do disco e normaliza para RGB uint8 no Viser."""
    try:
        image = skio.imread(path)
    except Exception:
        return None
    if image is None:
        return None
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.shape[-1] > 3:
        image = image[..., :3]
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _placeholder_image(width: int = 320, height: int = 240) -> np.ndarray:
    """Retorna imagem placeholder neutra para renders ausentes."""
    canvas = np.full((height, width, 3), 235, dtype=np.uint8)
    canvas[:, ::20, :] = 220
    canvas[::20, :, :] = 220
    return canvas


def _load_method_images(artifacts_root: Path, method_id: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Resolve imagens de render/referencia de um metodo nos artefatos."""
    render_path = _find_method_image(artifacts_root, method_id, "renders")
    ref_path = _find_method_image(artifacts_root, method_id, "references")
    render_image = _load_image(render_path) if render_path else None
    ref_image = _load_image(ref_path) if ref_path else None
    return render_image, ref_image


def _rotmat_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
    """Converte matriz de rotacao 3x3 para quaternion no formato wxyz."""
    m00, m01, m02 = rot[0]
    m10, m11, m12 = rot[1]
    m20, m21, m22 = rot[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float32)


def _add_scene_frustums(server: viser.ViserServer, scene_payload: dict) -> None:
    """Adiciona frustums de camera e centros na cena do Viser."""
    matrices: list[np.ndarray] = scene_payload["matrices"]
    frame_images: list[np.ndarray | None] = scene_payload.get("frame_images", [])
    fov_x = float(scene_payload["fov_x"])
    aspect = float(scene_payload["aspect"])
    convention = str(scene_payload.get("coordinate_convention") or "opencv").strip().lower()
    fov_y = 2.0 * math.atan(math.tan(fov_x * 0.5) / aspect)

    centers = []
    for index, matrix in enumerate(matrices):
        rotation_cv, translation_cv = _matrix_pose_to_viser(matrix, convention)
        centers.append(translation_cv)
        frustum_image = frame_images[index] if index < len(frame_images) else None
        server.scene.add_camera_frustum(
            name=f"/frustums/cam_{index:03d}",
            fov=fov_y,
            aspect=aspect,
            scale=0.35,
            line_width=2.0,
            color=(20, 20, 20),
            image=frustum_image,
            variant="filled" if frustum_image is not None else "wireframe",
            wxyz=_rotmat_to_quat_wxyz(rotation_cv),
            position=translation_cv,
        )

    if centers:
        points = np.stack(centers, axis=0)
        colors = np.tile(np.array([[0.12, 0.45, 0.85]], dtype=np.float32), (points.shape[0], 1))
        server.scene.add_point_cloud(
            name="/camera_centers",
            points=points,
            colors=colors,
            point_size=0.02,
        )

    scene_points = scene_payload.get("scene_points")
    scene_colors = scene_payload.get("scene_colors")
    if isinstance(scene_points, np.ndarray) and isinstance(scene_colors, np.ndarray) and scene_points.size > 0:
        server.scene.add_point_cloud(
            name="/scene_proxy_cloud",
            points=scene_points,
            colors=scene_colors,
            point_size=0.012,
        )


def run_preview_ui(
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    metrics_file: str | None = "./artifacts/metrics/latest_preview.json",
    scene_transforms_file: str | None = "./data/blender_synthetic/transforms_train.json",
    install_catalog_file: str | None = "./configs/install_catalog.json",
    minimal: bool = False,
) -> None:
    """Inicia UI Viser para metodos, metricas e frustums da cena."""
    metrics_snapshot = _load_metrics_snapshot(metrics_file)
    method_ids = _resolve_method_ids(metrics_snapshot)
    scene = _scene_payload(scene_transforms_file)
    artifacts_root = _resolve_artifacts_root(metrics_file)
    install_catalog = _load_install_catalog(install_catalog_file)

    server = viser.ViserServer(host=host, port=port, label="NVS Benchmark")
    server.scene.world_axes.visible = True
    server.gui.set_panel_label("Pre-visualizacao NVS Benchmark")
    server.gui.configure_theme(
        control_layout="fixed",
        control_width="large",
        dark_mode=False,
        show_logo=False,
        brand_color=(19, 99, 135),
    )

    _add_scene_frustums(server, scene)

    runtime_state = _PreviewRuntimeState(
        server=server,
        artifacts_root=artifacts_root,
        scene_transforms_file=scene_transforms_file,
        scene_payload=scene,
    )
    with _PREVIEW_RUNTIMES_LOCK:
        _PREVIEW_RUNTIMES[port] = runtime_state

    @server.on_client_connect
    def _on_client_connect(client: viser.ClientHandle) -> None:
        with runtime_state.lock:
            runtime_state.latest_client_id = int(client.client_id)

        @client.camera.on_update
        def _on_camera_update(camera: viser.CameraHandle) -> None:
            with runtime_state.lock:
                runtime_state.latest_client_id = int(client.client_id)
                runtime_state.latest_camera = {
                    "client_id": int(client.client_id),
                    "timestamp": float(camera.update_timestamp),
                    "position": np.asarray(camera.position, dtype=np.float64).tolist(),
                    "look_at": np.asarray(camera.look_at, dtype=np.float64).tolist(),
                    "up_direction": np.asarray(camera.up_direction, dtype=np.float64).tolist(),
                    "wxyz": np.asarray(camera.wxyz, dtype=np.float64).tolist(),
                    "fov": float(camera.fov),
                    "image_width": int(camera.image_width),
                    "image_height": int(camera.image_height),
                }

    if minimal:
        url = f"http://{server.get_host()}:{server.get_port()}"
        print(f"UI Viser (widget) disponivel em: {url}")
        if open_browser:
            webbrowser.open(url)
        while True:
            time.sleep(0.1)

    tabs = server.gui.add_tab_group()
    selected_methods = list(method_ids)
    method_checks: dict[str, viser.GuiCheckboxHandle] = {}
    render_handles: dict[str, viser.GuiImageHandle] = {}
    reference_handles: dict[str, viser.GuiImageHandle] = {}
    missing_handles: dict[str, viser.GuiMarkdownHandle] = {}
    placeholder = _placeholder_image()

    with tabs.add_tab("Visao Geral"):
        server.gui.add_markdown(
            "### Pre-visualizacao do Benchmark\n"
            "Selecione metodos para filtrar metricas e renders. A cena 3D exibe frustums de camera "
            "na convencao OpenCV (+Z para frente)."
        )
        with server.gui.add_folder("Metodos"):
            for method in method_ids:
                method_checks[method] = server.gui.add_checkbox(method, initial_value=True)

        metrics_markdown = server.gui.add_markdown(
            _build_metrics_markdown(selected_methods, metrics_snapshot)
        )
        server.gui.add_markdown(
            f"Fonte da cena: `{scene['scene_source']}`  \n"
            f"Numero de cameras: `{scene['camera_count']}`  \n"
            f"Cameras de contingencia: `{scene['used_fallback']}`"
        )
        reload_button = server.gui.add_button("Recarregar metricas")

    with tabs.add_tab("Renders"):
        server.gui.add_markdown("### Renders de exemplo")
        for method in method_ids:
            with server.gui.add_folder(method.upper()):
                render_image, ref_image = _load_method_images(artifacts_root, method)

                if render_image is not None:
                    render_handles[method] = server.gui.add_image(render_image, label="Render")
                if ref_image is not None:
                    reference_handles[method] = server.gui.add_image(ref_image, label="Reference")
                if render_image is None and ref_image is None:
                    missing_handles[method] = server.gui.add_markdown(
                        "Nenhuma imagem de render encontrada para este metodo nos artefatos."
                    )

    with tabs.add_tab("Comparar"):
        server.gui.add_markdown("### Comparacao lado a lado")
        server.gui.add_markdown(
            "Selecione dois metodos para comparar renders e referencias lado a lado."
        )
        left_default = method_ids[0]
        right_default = method_ids[1] if len(method_ids) > 1 else method_ids[0]
        left_selector = server.gui.add_dropdown("Metodo esquerdo", method_ids, initial_value=left_default)
        right_selector = server.gui.add_dropdown("Metodo direito", method_ids, initial_value=right_default)

        with server.gui.add_folder("Vista esquerda"):
            left_render = server.gui.add_image(placeholder, label="Render")
            left_reference = server.gui.add_image(placeholder, label="Referencia")
            left_note = server.gui.add_markdown("Nenhuma imagem encontrada para o metodo selecionado.")

        with server.gui.add_folder("Vista direita"):
            right_render = server.gui.add_image(placeholder, label="Render")
            right_reference = server.gui.add_image(placeholder, label="Referencia")
            right_note = server.gui.add_markdown("Nenhuma imagem encontrada para o metodo selecionado.")

    with tabs.add_tab("Instalacao"):
        server.gui.add_markdown("### Instalar datasets e modelos")
        server.gui.add_markdown(
            "Configure o catalogo em `configs/install_catalog.json`. "
            "Botoes de download ficam desativados quando o caminho ja existe."
        )

        datasets = install_catalog.get("datasets", [])
        methods = install_catalog.get("methods", [])
        notes = install_catalog.get("notes", [])

        ready_datasets = sum(1 for item in datasets if _is_ready(item.get("path")))
        ready_methods = sum(1 for item in methods if _is_ready(item.get("path")))
        server.gui.add_markdown(
            f"**Status geral**  \nDatasets prontos: `{ready_datasets}/{len(datasets)}`  \nModelos prontos: `{ready_methods}/{len(methods)}`"
        )

        if notes:
            server.gui.add_markdown("\n".join(f"- {note}" for note in notes))

        with server.gui.add_folder("Datasets"):
            for item in datasets:
                label = str(item.get("label") or item.get("id") or "dataset")
                path = str(item.get("path") or "")
                url = str(item.get("url") or "")
                command = str(item.get("command") or "")
                size_mb = item.get("size_mb")
                installed = _is_ready(path)
                badge = _status_badge(installed, bool(url))

                server.gui.add_markdown(
                    f"**{label}** {badge}  \nPath: `{path or '--'}`  \nTamanho estimado: `{_format_size(size_mb)}`"
                )

                open_button = server.gui.add_button("Abrir pasta")
                open_button.disabled = not installed
                if installed:
                    open_button.on_click(
                        lambda _event, p=path: webbrowser.open(Path(p).resolve().as_uri())
                    )

                download_button = server.gui.add_button("Baixar dataset")
                download_button.disabled = installed or not url
                if installed:
                    server.gui.add_markdown("Status: ja disponivel.")
                elif not url:
                    server.gui.add_markdown("Status: defina uma URL no catalogo para habilitar download.")
                else:
                    server.gui.add_markdown("Status: pronto para download.")
                    download_button.on_click(lambda _event, link=url: webbrowser.open(link))

                if command:
                    server.gui.add_markdown(f"Comando sugerido: `{command}`")
                    copy_button = server.gui.add_button("Copiar comando")
                    copy_button.on_click(
                        lambda _event, cmd=command: _copy_to_clipboard(cmd)
                    )

        with server.gui.add_folder("Modelos"):
            for item in methods:
                label = str(item.get("label") or item.get("id") or "modelo")
                path = str(item.get("path") or "")
                url = str(item.get("url") or "")
                command = str(item.get("command") or "")
                size_mb = item.get("size_mb")
                installed = _is_ready(path)
                badge = _status_badge(installed, bool(url))

                server.gui.add_markdown(
                    f"**{label}** {badge}  \nPath: `{path or '--'}`  \nTamanho estimado: `{_format_size(size_mb)}`"
                )

                open_button = server.gui.add_button("Abrir pasta")
                open_button.disabled = not installed
                if installed:
                    open_button.on_click(
                        lambda _event, p=path: webbrowser.open(Path(p).resolve().as_uri())
                    )

                button = server.gui.add_button("Abrir repositorio")
                button.disabled = installed or not url
                if installed:
                    server.gui.add_markdown("Status: ja disponivel.")
                elif not url:
                    server.gui.add_markdown("Status: defina uma URL no catalogo para habilitar link.")
                else:
                    server.gui.add_markdown("Status: pronto para download/clone.")
                    button.on_click(lambda _event, link=url: webbrowser.open(link))

                if not command and url and path:
                    command = f"git clone {url} {path}"
                if command:
                    server.gui.add_markdown(f"Comando sugerido: `{command}`")
                    copy_button = server.gui.add_button("Copiar comando")
                    copy_button.on_click(
                        lambda _event, cmd=command: _copy_to_clipboard(cmd)
                    )


    def refresh_compare() -> None:
        """Atualiza imagens da comparacao lado a lado."""
        for selector, render_handle, ref_handle, note_handle in (
            (left_selector, left_render, left_reference, left_note),
            (right_selector, right_render, right_reference, right_note),
        ):
            method_id = selector.value
            render_image, ref_image = _load_method_images(artifacts_root, method_id)
            if render_image is None:
                render_handle.image = placeholder
            else:
                render_handle.image = render_image
            if ref_image is None:
                ref_handle.image = placeholder
            else:
                ref_handle.image = ref_image
            note_handle.visible = render_image is None and ref_image is None

    def refresh_selected() -> None:
        """Atualiza metodos selecionados e visibilidade de metricas/renders."""
        selected = [method for method, handle in method_checks.items() if handle.value]
        if not selected:
            selected = list(method_ids)
        metrics_markdown.content = _build_metrics_markdown(selected, metrics_snapshot)
        for method in method_ids:
            visible = method in selected
            if method in render_handles:
                render_handles[method].visible = visible
            if method in reference_handles:
                reference_handles[method].visible = visible
            if method in missing_handles:
                missing_handles[method].visible = visible

    def reload_metrics() -> None:
        """Recarrega snapshot de metricas e reflete na interface."""
        nonlocal metrics_snapshot
        metrics_snapshot = _load_metrics_snapshot(metrics_file)
        refresh_selected()

    for handle in method_checks.values():
        handle.on_update(lambda _event: refresh_selected())
    reload_button.on_click(lambda _event: reload_metrics())
    left_selector.on_update(lambda _event: refresh_compare())
    right_selector.on_update(lambda _event: refresh_compare())
    refresh_selected()
    refresh_compare()

    url = f"http://{server.get_host()}:{server.get_port()}"
    print(f"UI de pre-visualizacao disponivel em: {url}")
    if open_browser:
        webbrowser.open(url)

    while True:
        time.sleep(0.1)


def get_preview_runtime_state(port: int) -> dict:
    """Return latest camera/training-scene state from active preview runtime."""
    with _PREVIEW_RUNTIMES_LOCK:
        runtime = _PREVIEW_RUNTIMES.get(int(port))
    if runtime is None:
        return {
            "available": False,
            "reason": "preview-runtime-not-found",
        }

    with runtime.lock:
        camera = dict(runtime.latest_camera)
        scene = runtime.scene_payload
    return {
        "available": True,
        "camera": camera,
        "scene": {
            "source": scene.get("scene_source"),
            "camera_count": int(scene.get("camera_count") or 0),
            "used_fallback": bool(scene.get("used_fallback")),
            "coordinate_convention": scene.get("coordinate_convention") or "opencv",
            "proxy_point_count": int(scene["scene_points"].shape[0]) if isinstance(scene.get("scene_points"), np.ndarray) else 0,
        },
    }


def capture_preview_camera_render(
    *,
    port: int,
    width: int,
    height: int,
    output_name: str = "live_camera_render.jpg",
) -> dict:
    """Capture current Viser camera viewport from the latest connected client."""
    with _PREVIEW_RUNTIMES_LOCK:
        runtime = _PREVIEW_RUNTIMES.get(int(port))
    if runtime is None:
        return {"ok": False, "error": "preview-runtime-not-found"}

    clients = runtime.server.get_clients()
    if not clients:
        return {"ok": False, "error": "no-connected-clients"}

    with runtime.lock:
        preferred = runtime.latest_client_id
    client = clients.get(preferred) if preferred is not None else None
    if client is None:
        client = next(iter(clients.values()))

    safe_width = int(max(160, min(width, 1920)))
    safe_height = int(max(120, min(height, 1080)))
    try:
        image = client.camera.get_render(height=safe_height, width=safe_width, transport_format="jpeg")
    except Exception as exc:
        return {"ok": False, "error": f"capture-failed: {exc}"}

    output_dir = runtime.artifacts_root / "preview"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name
    try:
        skio.imsave(output_path, image, check_contrast=False)
    except Exception as exc:
        return {"ok": False, "error": f"save-failed: {exc}"}

    with runtime.lock:
        camera = dict(runtime.latest_camera)

    return {
        "ok": True,
        "path": str(output_path),
        "camera": camera,
        "width": safe_width,
        "height": safe_height,
    }


def refresh_preview_scene(port: int) -> dict:
    """Recompute scene payload and update frustums/proxy cloud in the active preview."""
    with _PREVIEW_RUNTIMES_LOCK:
        runtime = _PREVIEW_RUNTIMES.get(int(port))
    if runtime is None:
        return {"ok": False, "error": "preview-runtime-not-found"}

    scene_file = runtime.scene_transforms_file
    new_scene = _scene_payload(scene_file)
    try:
        _add_scene_frustums(runtime.server, new_scene)
    except Exception as exc:
        return {"ok": False, "error": f"scene-refresh-failed: {exc}"}

    with runtime.lock:
        runtime.scene_payload = new_scene

    return {
        "ok": True,
        "camera_count": int(new_scene.get("camera_count") or 0),
        "proxy_point_count": int(new_scene["scene_points"].shape[0]) if isinstance(new_scene.get("scene_points"), np.ndarray) else 0,
        "used_fallback": bool(new_scene.get("used_fallback")),
    }


def nearest_scene_frame_index(port: int) -> int | None:
    """Return nearest scene frame index to current camera in Viser coordinates."""
    with _PREVIEW_RUNTIMES_LOCK:
        runtime = _PREVIEW_RUNTIMES.get(int(port))
    if runtime is None:
        return None

    with runtime.lock:
        camera = dict(runtime.latest_camera)
        scene = runtime.scene_payload

    position = camera.get("position")
    matrices = scene.get("matrices")
    convention = str(scene.get("coordinate_convention") or "opengl").strip().lower()
    if not isinstance(position, list) or len(position) != 3:
        return None
    if not isinstance(matrices, list) or not matrices:
        return None

    cam = np.asarray(position, dtype=np.float32)
    best_idx = None
    best_dist = None
    for idx, matrix in enumerate(matrices):
        if not isinstance(matrix, np.ndarray) or matrix.shape != (4, 4):
            continue
        _, center = _matrix_pose_to_viser(matrix, convention)
        dist = float(np.linalg.norm(center.astype(np.float32) - cam))
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx
