"""Contratos e tipos compartilhados entre os módulos do benchmark NVS."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class MethodKind(str, Enum):
    """Famílias de métodos suportadas pelo benchmark."""

    NERF_STATIC = "nerf_static"
    NERF_DYNAMIC = "nerf_dynamic"
    GS_STATIC = "gs_static"
    GS_DYNAMIC = "gs_dynamic"
    EXTERNAL = "external"


class HardwareProfile(str, Enum):
    """Perfis de hardware para orientar configurações de execução."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ADAPTIVE = "adaptive"


class ReportFormat(str, Enum):
    """Formatos de exportação de relatório."""

    HTML = "html"
    PDF = "pdf"


@dataclass(frozen=True)
class DatasetSpec:
    """Especificação canônica de dataset usada em treino/inferência."""

    name: str
    root: str
    split: str = "train"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RunConfig:
    """Configuração completa e reproduzível de uma execução."""

    run_id: str
    dataset: DatasetSpec
    method: str
    seed: int = 42
    output_dir: str = "./artifacts"
    log_dir: str = "./logs"
    hardware_profile: HardwareProfile = HardwareProfile.ADAPTIVE
    report_formats: list[ReportFormat] = field(default_factory=lambda: [ReportFormat.HTML, ReportFormat.PDF])
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainRequest:
    """Entrada da etapa de treinamento de um método."""

    config: RunConfig
    checkpoint_path: str | None = None


@dataclass(frozen=True)
class InferenceRequest:
    """Entrada da etapa de inferência de um método."""

    config: RunConfig
    checkpoint_path: str
    split: str = "test"


@dataclass(frozen=True)
class TrainResult:
    """Saída padronizada da etapa de treinamento."""

    method: str
    checkpoint_path: str
    train_seconds: float
    output_dir: str
    logs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InferenceResult:
    """Saída padronizada da etapa de inferência."""

    method: str
    rendered_dir: str
    frames: int
    inference_seconds: float
    logs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PerformanceStats:
    """Resumo de desempenho coletado por método."""

    fps: float
    vram_gb_peak: float
    train_seconds: float
    inference_seconds: float


@dataclass(frozen=True)
class RunArtifacts:
    """Referências aos artefatos finais gerados em uma execução."""

    run_id: str
    method: str
    dataset: str
    checkpoint_path: str
    rendered_dir: str
    metrics_path: str | None = None
    report_paths: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MethodCapabilities:
    """Capacidades declaradas por um adaptador de método."""

    supports_train: bool = True
    supports_inference: bool = True
    supports_dynamic_scene: bool = False
    supports_limited_gpu: bool = True


class BenchmarkMethod(Protocol):
    """Interface mínima que todo adaptador de método deve implementar."""

    method_id: str
    display_name: str
    capabilities: MethodCapabilities

    def validate_config(self, config: RunConfig) -> None:
        ...

    def train(self, request: TrainRequest) -> TrainResult:
        ...

    def infer(self, request: InferenceRequest) -> InferenceResult:
        ...

    def collect_performance(self) -> PerformanceStats:
        ...
