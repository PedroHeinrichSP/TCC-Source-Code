"""Core: contratos, registro de métodos e orquestração do benchmark."""

from .contracts import (
	BenchmarkMethod,
	DatasetSpec,
	HardwareProfile,
	InferenceRequest,
	InferenceResult,
	MethodCapabilities,
	MethodKind,
	PerformanceStats,
	ReportFormat,
	RunArtifacts,
	RunConfig,
	TrainRequest,
	TrainResult,
)
from .registry import MethodRegistry
from .orchestrator import Orchestrator
from .cache_registry import CacheEntry, CacheRegistry
from .experiments import ExperimentManager, ExperimentRecord
from .presets import (
	TrainingPreset,
	list_preset_summaries,
	load_presets,
	resolve_iterations,
)

__all__ = [
	"BenchmarkMethod",
	"DatasetSpec",
	"HardwareProfile",
	"InferenceRequest",
	"InferenceResult",
	"MethodCapabilities",
	"MethodKind",
	"MethodRegistry",
	"Orchestrator",
	"CacheEntry",
	"CacheRegistry",
	"ExperimentManager",
	"ExperimentRecord",
	"PerformanceStats",
	"ReportFormat",
	"RunArtifacts",
	"RunConfig",
	"TrainRequest",
	"TrainResult",
	"TrainingPreset",
	"list_preset_summaries",
	"load_presets",
	"resolve_iterations",
]
