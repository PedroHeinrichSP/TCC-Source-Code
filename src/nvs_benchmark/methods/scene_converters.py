"""Helpers compartilhados para adaptar cenas reais ao formato esperado por backends."""

from __future__ import annotations

import importlib.util
import json
import shutil
from functools import lru_cache
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

REQUIRED_BLENDER_SPLITS = ("transforms_train.json", "transforms_val.json", "transforms_test.json")


def find_images(directory: Path) -> list[Path]:
    patterns = ("*.png", "*.jpg", "*.jpeg")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(directory.rglob(pattern))
    return sorted(files)


def preferred_real_image_subdirs(
    *,
    dataset_name: str,
    preset_name: str | None,
) -> tuple[str, ...]:
    preset = (preset_name or "").strip().lower()
    if dataset_name == "mipnerf360":
        if preset in {"smoke", "quick", "preview"}:
            return ("images_8", "images_4", "images_2", "images")
        return ("images_4", "images_2", "images_8", "images")
    return ("images",)


def preferred_real_max_image_dim(
    *,
    dataset_name: str,
    preset_name: str | None,
) -> int | None:
    preset = (preset_name or "").strip().lower()
    if dataset_name == "mipnerf360":
        return None
    if dataset_name == "tanks_and_temples":
        if preset in {"smoke", "quick", "preview"}:
            return 960
        return 1280
    return None


def has_required_blender_splits(root: Path) -> bool:
    return all((root / split_file).exists() for split_file in REQUIRED_BLENDER_SPLITS)


def has_real_scene_layout(root: Path) -> bool:
    return (
        root.exists()
        and root.is_dir()
        and (
            (root / "sparse" / "0").exists()
            or (root / "poses_bounds.npy").exists()
            or (root / "images").exists()
            or bool(find_images(root))
        )
    )


def has_pose_priors(root: Path) -> bool:
    return has_required_blender_splits(root) or (root / "sparse" / "0").exists()


def _resolve_colmap_loader_path() -> Path:
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        candidate = parent / "third_party" / "gaussian_splatting" / "scene" / "colmap_loader.py"
        if candidate.exists():
            return candidate
    raise RuntimeError(
        "Nao foi possivel localizar third_party/gaussian_splatting/scene/colmap_loader.py "
        f"a partir de {current_file}."
    )


@lru_cache(maxsize=1)
def _load_colmap_loader_module():
    module_path = _resolve_colmap_loader_path()
    spec = importlib.util.spec_from_file_location("nvs_benchmark_colmap_loader", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Nao foi possivel carregar helpers COLMAP em: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_colmap_intrinsics_text(path: Path) -> dict[int, dict[str, object]]:
    cameras: dict[int, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as fp:
        for raw_line in fp:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            camera_id = int(elems[0])
            model = elems[1]
            width = int(elems[2])
            height = int(elems[3])
            params = np.array(tuple(map(float, elems[4:])), dtype=np.float64)
            cameras[camera_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras


def _resolve_colmap_image_path(
    scene_root: Path,
    image_name: str,
    preferred_subdirs: tuple[str, ...] | None = None,
) -> Path | None:
    search_roots: list[Path] = []
    for subdir in preferred_subdirs or ("images",):
        candidate_root = scene_root / subdir
        if candidate_root.exists():
            search_roots.append(candidate_root)
    search_roots.extend([scene_root / "images", scene_root])

    candidates: list[Path] = []
    for root in search_roots:
        candidates.extend(
            [
                root / image_name,
                root / Path(image_name).name,
                root / f"{Path(image_name).stem}.png",
                root / f"{Path(image_name).stem}.jpg",
                root / f"{Path(image_name).stem}.jpeg",
            ]
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _save_rgba_png(source_path: Path, target_path: Path, *, max_image_dim: int | None = None) -> None:
    image = imageio.imread(source_path)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim == 3 and image.shape[-1] == 3:
        alpha = np.full((*image.shape[:2], 1), 255, dtype=image.dtype)
        image = np.concatenate([image, alpha], axis=-1)
    elif image.ndim == 3 and image.shape[-1] > 4:
        image = image[:, :, :4]

    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            if float(np.nanmax(image)) <= 1.0:
                image = np.clip(image * 255.0, 0, 255)
            else:
                image = np.clip(image, 0, 255)
        else:
            image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)

    if max_image_dim is not None:
        height, width = image.shape[:2]
        longest_edge = max(height, width)
        if longest_edge > max_image_dim:
            import cv2

            scale = float(max_image_dim) / float(longest_edge)
            resized_width = max(1, int(round(width * scale)))
            resized_height = max(1, int(round(height * scale)))
            image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(target_path, image)


def prepare_colmap_scene_to_blender(
    *,
    source_root: Path,
    prepared_root: Path,
    holdout_stride: int,
    include_time_metadata: bool = False,
    preferred_image_subdirs: tuple[str, ...] | None = None,
    max_image_dim: int | None = None,
) -> Path:
    """Converte uma cena COLMAP para o layout Blender esperado pelos backends NeRF."""
    meta_path = prepared_root / "conversion_meta.json"
    expected_meta = {
        "source_root": str(source_root.resolve()),
        "holdout_stride": int(holdout_stride),
        "include_time_metadata": bool(include_time_metadata),
        "preferred_image_subdirs": list(preferred_image_subdirs or []),
        "max_image_dim": int(max_image_dim) if max_image_dim is not None else None,
    }

    if has_required_blender_splits(prepared_root) and meta_path.exists():
        try:
            current_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            current_meta = None
        if current_meta == expected_meta:
            return prepared_root

    if prepared_root.exists():
        shutil.rmtree(prepared_root)
    prepared_root.mkdir(parents=True, exist_ok=True)

    sparse_root = source_root / "sparse" / "0"
    if not sparse_root.exists():
        raise ValueError(
            f"Cena real requer diretorio COLMAP em {sparse_root} para conversao ao backend NeRF."
        )

    loader = _load_colmap_loader_module()
    images_bin = sparse_root / "images.bin"
    cameras_bin = sparse_root / "cameras.bin"
    images_txt = sparse_root / "images.txt"
    cameras_txt = sparse_root / "cameras.txt"

    if images_bin.exists() and cameras_bin.exists():
        extrinsics = loader.read_extrinsics_binary(images_bin)
        intrinsics = loader.read_intrinsics_binary(cameras_bin)
    elif images_txt.exists() and cameras_txt.exists():
        extrinsics = loader.read_extrinsics_text(images_txt)
        intrinsics = _read_colmap_intrinsics_text(cameras_txt)
    else:
        raise ValueError(
            f"Nao foi possivel localizar arquivos COLMAP em {sparse_root}: images.bin/text e cameras.bin/text."
        )

    camera_entries = sorted(extrinsics.items(), key=lambda item: item[1].name.lower())
    if not camera_entries:
        raise ValueError(f"Nenhuma camera COLMAP encontrada em {sparse_root}.")

    train_frames: list[dict[str, object]] = []
    test_frames: list[dict[str, object]] = []
    camera_angle_x: float | None = None

    for index, (_, extrinsic) in enumerate(camera_entries):
        intrinsic = intrinsics[extrinsic.camera_id]
        if hasattr(intrinsic, "model"):
            model = intrinsic.model
            width = int(intrinsic.width)
            params = np.asarray(intrinsic.params, dtype=np.float64)
        else:
            model = intrinsic["model"]
            width = int(intrinsic["width"])
            params = np.asarray(intrinsic["params"], dtype=np.float64)

        if model not in {"PINHOLE", "SIMPLE_PINHOLE"}:
            raise ValueError(f"Modelo de camera COLMAP nao suportado para conversao NeRF: {model}.")

        source_image = _resolve_colmap_image_path(
            source_root,
            extrinsic.name,
            preferred_subdirs=preferred_image_subdirs,
        )
        if source_image is None:
            raise ValueError(f"Imagem nao encontrada para COLMAP frame: {extrinsic.name}")

        target_stem = Path(extrinsic.name).stem
        target_image = prepared_root / "images" / f"{target_stem}.png"
        _save_rgba_png(source_image, target_image, max_image_dim=max_image_dim)

        w2c = np.eye(4, dtype=np.float64)
        w2c[:3, :3] = loader.qvec2rotmat(extrinsic.qvec)
        w2c[:3, 3] = np.asarray(extrinsic.tvec, dtype=np.float64)
        c2w = np.linalg.inv(w2c)
        blender_transform = c2w.copy()
        blender_transform[:3, 1:3] *= -1.0

        fx = float(params[0])
        if camera_angle_x is None:
            camera_angle_x = float(2.0 * np.arctan(width / (2.0 * fx)))

        frame: dict[str, object] = {
            "file_path": f"images/{target_stem}",
            "transform_matrix": blender_transform.tolist(),
        }
        if include_time_metadata:
            frame["time"] = 0.0

        if len(camera_entries) == 1:
            train_frames.append(frame)
            test_frames.append(frame)
        elif index % holdout_stride == 0:
            test_frames.append(frame)
        else:
            train_frames.append(frame)

    if not train_frames and test_frames:
        train_frames.append(test_frames[0])
    if not test_frames and train_frames:
        test_frames.append(train_frames[-1])

    payload_base = {"camera_angle_x": camera_angle_x or 0.0}
    (prepared_root / "transforms_train.json").write_text(
        json.dumps({**payload_base, "frames": train_frames}, indent=2),
        encoding="utf-8",
    )
    (prepared_root / "transforms_val.json").write_text(
        json.dumps({**payload_base, "frames": []}, indent=2),
        encoding="utf-8",
    )
    (prepared_root / "transforms_test.json").write_text(
        json.dumps({**payload_base, "frames": test_frames}, indent=2),
        encoding="utf-8",
    )
    meta_path.write_text(json.dumps(expected_meta, indent=2), encoding="utf-8")
    return prepared_root
