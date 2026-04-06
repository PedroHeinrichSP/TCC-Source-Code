"""Dados: carregadores e validação de datasets."""

from .registry import (
    SUPPORTED_DATASETS,
    BlenderSyntheticLoader,
    CustomDatasetLoader,
    DNeRFLoader,
    MipNeRF360Loader,
    TanksAndTemplesLoader,
    DatasetValidationError,
    get_loader,
    load_dataset,
    validate_dataset,
)
from .fingerprint import DatasetFingerprint, build_dataset_fingerprint
from .validation import DataValidationReport, validate_dataset_integrity

__all__ = [
    "SUPPORTED_DATASETS",
    "BlenderSyntheticLoader",
    "CustomDatasetLoader",
    "DNeRFLoader",
    "MipNeRF360Loader",
    "TanksAndTemplesLoader",
    "DatasetValidationError",
    "get_loader",
    "load_dataset",
    "validate_dataset",
    "DatasetFingerprint",
    "build_dataset_fingerprint",
    "DataValidationReport",
    "validate_dataset_integrity",
]
