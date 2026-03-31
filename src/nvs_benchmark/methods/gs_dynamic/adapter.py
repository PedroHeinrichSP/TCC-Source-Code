"""Adaptador stub de 4D Gaussian Splatting dinâmico no contrato unificado."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
from matplotlib import image as mpimg

from nvs_benchmark.core import (
    InferenceRequest,
    InferenceResult,
    MethodCapabilities,
    PerformanceStats,
    RunConfig,
    TrainRequest,
    TrainResult,
)
from nvs_benchmark.methods.utils import render_stub_from_dataset


@dataclass
class GSDynamicAdapter:
    """Adaptador para integração de 4D Gaussian Splatting dinâmico.

    Implementação stub que reutiliza frames reais quando disponíveis,
    mantendo os renders de preview mais próximos de cenas reais.
    """

    method_id: str = "gs_dynamic"
    display_name: str = "4DGS Dynamic (hustvl/4DGaussians)"
    capabilities: MethodCapabilities = MethodCapabilities(
        supports_train=True,
        supports_inference=True,
        supports_dynamic_scene=True,
        supports_limited_gpu=True,
    )
    base_repo_url: str = "https://github.com/hustvl/4DGaussians"

    def validate_config(self, config: RunConfig) -> None:
        """Valida configuração para cenários de 4DGS dinâmico."""
        if config.method != self.method_id:
            raise ValueError(f"Metodo incompat�vel. Esperado '{self.method_id}', recebido '{config.method}'")
        if config.dataset.name not in {"d_nerf", "blender_synthetic", "custom"}:
            raise ValueError("Dataset nao suportado para GS dinamico neste estagio.")

    def train(self, request: TrainRequest) -> TrainResult:
        """Executa treino stub e cria checkpoint placeholder."""
        self.validate_config(request.config)
        start = perf_counter()

        checkpoint_dir = Path(request.config.output_dir) / request.config.run_id / self.method_id / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "model_stub.ckpt"
        checkpoint_path.write_text("stub checkpoint for gs dynamic\n", encoding="utf-8")

        elapsed = perf_counter() - start
        return TrainResult(
            method=self.method_id,
            checkpoint_path=str(checkpoint_path),
            train_seconds=elapsed,
            output_dir=str(checkpoint_dir.parent),
            logs={"mode": "stub", "repo": self.base_repo_url},
        )

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Executa inferência stub e gera frame placeholder com tempo."""
        self.validate_config(request.config)
        start = perf_counter()

        render_dir = Path(request.config.output_dir) / request.config.run_id / self.method_id / "renders"
        render_dir.mkdir(parents=True, exist_ok=True)
        if not render_stub_from_dataset(request.config.dataset.root, request.split, render_dir):
            xx, yy = np.meshgrid(np.linspace(0, 1, 64), np.linspace(0, 1, 64))
            image = np.stack([yy, 0.5 * xx, xx], axis=-1).astype(np.float32)
            mpimg.imsave(render_dir / "frame_0000.png", image)

        elapsed = perf_counter() - start
        return InferenceResult(
            method=self.method_id,
            rendered_dir=str(render_dir),
            frames=1,
            inference_seconds=elapsed,
            logs={"mode": "stub", "checkpoint_used": request.checkpoint_path},
        )

    def collect_performance(self) -> PerformanceStats:
        """Retorna métricas sintéticas de desempenho para integração stub."""
        return PerformanceStats(
            fps=0.0,
            vram_gb_peak=0.0,
            train_seconds=0.0,
            inference_seconds=0.0,
        )
