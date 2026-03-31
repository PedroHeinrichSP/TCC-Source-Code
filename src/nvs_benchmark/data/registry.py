"""Carregadores de dataset e validação estrutural para o benchmark."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from nvs_benchmark.core import DatasetSpec

SUPPORTED_DATASETS = ["blender_synthetic", "d_nerf", "custom"]


class DatasetValidationError(ValueError):
    """Erro de validação estrutural de dataset."""


def _count_image_files(root: Path) -> int:
    """Conta arquivos de imagem em uma árvore de diretórios."""
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.PNG")
    total = 0
    for pattern in patterns:
        total += len(list(root.rglob(pattern)))
    return total


def _existing_splits(root: Path) -> list[str]:
    """Lista splits disponíveis com base em arquivos transforms_<split>.json."""
    splits: list[str] = []
    for split_name in ("train", "val", "test"):
        transforms_path = root / f"transforms_{split_name}.json"
        if transforms_path.exists():
            splits.append(split_name)
    return splits


def _read_transforms(root: Path, split: str) -> dict:
    """Lê e valida o JSON de transforms de um split."""
    transforms_path = root / f"transforms_{split}.json"
    if not transforms_path.exists():
        raise DatasetValidationError(f"Arquivo nao encontrado: {transforms_path}")
    try:
        with transforms_path.open("r", encoding="utf-8-sig") as fp:
            return json.load(fp)
    except json.JSONDecodeError as exc:
        raise DatasetValidationError(f"JSON invalido em {transforms_path}: {exc}") from exc


@dataclass
class BlenderSyntheticLoader:
    """Carregador para o formato Blender Synthetic."""

    dataset_name: str = "blender_synthetic"

    def validate(self, root: str | Path) -> None:
        """Valida estrutura mínima esperada para Blender Synthetic."""
        root_path = Path(root)
        if not root_path.exists() or not root_path.is_dir():
            raise DatasetValidationError(f"Diretorio invalido para dataset: {root_path}")

        splits = _existing_splits(root_path)
        if "train" not in splits or "test" not in splits:
            raise DatasetValidationError(
                "Blender Synthetic requer pelo menos transforms_train.json e transforms_test.json"
            )

    def load(self, root: str | Path, split: str = "train") -> DatasetSpec:
        """Carrega metadados do split solicitado para Blender Synthetic."""
        root_path = Path(root)
        self.validate(root_path)
        transforms = _read_transforms(root_path, split)
        frames = transforms.get("frames", [])

        metadata = {
            "available_splits": _existing_splits(root_path),
            "frame_count": len(frames),
            "image_file_count": _count_image_files(root_path),
            "camera_angle_x": transforms.get("camera_angle_x"),
        }
        return DatasetSpec(name=self.dataset_name, root=str(root_path), split=split, metadata=metadata)


@dataclass
class DNeRFLoader:
    """Carregador para formato D-NeRF com metadados temporais."""

    dataset_name: str = "d_nerf"

    def validate(self, root: str | Path) -> None:
        """Valida estrutura mínima esperada para D-NeRF."""
        root_path = Path(root)
        if not root_path.exists() or not root_path.is_dir():
            raise DatasetValidationError(f"Diretorio invalido para dataset: {root_path}")

        splits = _existing_splits(root_path)
        if "train" not in splits or "test" not in splits:
            raise DatasetValidationError("D-NeRF requer pelo menos transforms_train.json e transforms_test.json")

    def load(self, root: str | Path, split: str = "train") -> DatasetSpec:
        """Carrega metadados do split solicitado para D-NeRF."""
        root_path = Path(root)
        self.validate(root_path)
        transforms = _read_transforms(root_path, split)
        frames = transforms.get("frames", [])
        has_time_metadata = any("time" in frame for frame in frames)

        metadata = {
            "available_splits": _existing_splits(root_path),
            "frame_count": len(frames),
            "image_file_count": _count_image_files(root_path),
            "has_time_metadata": has_time_metadata,
        }
        return DatasetSpec(name=self.dataset_name, root=str(root_path), split=split, metadata=metadata)


@dataclass
class CustomDatasetLoader:
    """Carregador para datasets do usuário com convenção transforms_*.json."""

    dataset_name: str = "custom"

    def validate(self, root: str | Path) -> None:
        """Valida a estrutura mínima de um dataset customizado."""
        root_path = Path(root)
        if not root_path.exists() or not root_path.is_dir():
            raise DatasetValidationError(f"Diretorio invalido para dataset: {root_path}")

        splits = _existing_splits(root_path)
        if not splits:
            raise DatasetValidationError(
                "Dataset custom requer ao menos um transforms_<split>.json (ex: transforms_train.json)."
            )

    def load(self, root: str | Path, split: str = "train") -> DatasetSpec:
        """Carrega metadados com fallback para o primeiro split disponível."""
        root_path = Path(root)
        self.validate(root_path)
        available = _existing_splits(root_path)
        split_to_use = split if split in available else available[0]
        transforms = _read_transforms(root_path, split_to_use)
        frames = transforms.get("frames", [])
        has_time_metadata = any("time" in frame for frame in frames)

        metadata = {
            "available_splits": available,
            "frame_count": len(frames),
            "image_file_count": _count_image_files(root_path),
            "has_time_metadata": has_time_metadata,
            "camera_angle_x": transforms.get("camera_angle_x"),
        }
        return DatasetSpec(name=self.dataset_name, root=str(root_path), split=split_to_use, metadata=metadata)


def get_loader(dataset_name: str) -> BlenderSyntheticLoader | DNeRFLoader | CustomDatasetLoader:
    """Resolve o carregador adequado para um dataset suportado."""
    normalized = dataset_name.strip().lower()
    if normalized == "blender_synthetic":
        return BlenderSyntheticLoader()
    if normalized == "d_nerf":
        return DNeRFLoader()
    if normalized == "custom":
        return CustomDatasetLoader()
    raise DatasetValidationError(
        f"Dataset nao suportado: {dataset_name}. Suportados: {', '.join(SUPPORTED_DATASETS)}"
    )


def validate_dataset(dataset_name: str, root: str | Path) -> tuple[bool, str]:
    """Valida dataset e retorna uma tupla (ok, mensagem)."""
    try:
        loader = get_loader(dataset_name)
        loader.validate(root)
        return True, "Dataset valido"
    except DatasetValidationError as exc:
        return False, str(exc)


def load_dataset(dataset_name: str, root: str | Path, split: str = "train") -> DatasetSpec:
    """Carrega metadados de dataset para um split."""
    loader = get_loader(dataset_name)
    return loader.load(root=root, split=split)
