"""Adaptador externo para conectar modelos e scripts fornecidos pelo usuário."""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from nvs_benchmark.core import (
    InferenceRequest,
    InferenceResult,
    MethodCapabilities,
    PerformanceStats,
    RunConfig,
    TrainRequest,
    TrainResult,
)


@dataclass
class ExternalMethodAdapter:
    """Adaptador para modelos e pipelines definidos pelo usuário.

    O adaptador espera comandos ou artefatos pré-computados em RunConfig.extra,
    permitindo plugar qualquer modelo externo sem alterar o núcleo do benchmark.

    Chaves suportadas em config.extra:
    - train_command: list[str] ou string de shell para treino.
    - infer_command: list[str] ou string de shell para inferência.
    - checkpoint_path: caminho para checkpoint produzido externamente.
    - rendered_dir: caminho para diretório com frames renderizados.
    - frames: inteiro opcional com número de frames renderizados.
    """

    method_id: str = "external"
    display_name: str = "External Method (user-defined)"
    capabilities: MethodCapabilities = MethodCapabilities(
        supports_train=True,
        supports_inference=True,
        supports_dynamic_scene=True,
        supports_limited_gpu=True,
    )

    def validate_config(self, config: RunConfig) -> None:
        """Valida se a configuração é compatível com o adaptador externo."""
        if config.method != self.method_id:
            raise ValueError(f"Metodo incompat�vel. Esperado '{self.method_id}', recebido '{config.method}'")

    def _normalize_command(self, command: object | None) -> list[str] | None:
        """Normaliza definição de comando para lista de argumentos."""
        if command is None:
            return None
        if isinstance(command, list):
            return [str(item) for item in command]
        if isinstance(command, str):
            return shlex.split(command)
        raise ValueError("train_command/infer_command must be a list[str] or string")

    def _build_env(self, config: RunConfig, checkpoint_path: str | None = None, rendered_dir: str | None = None) -> dict:
        """Monta variáveis de ambiente para comandos externos."""
        env = os.environ.copy()
        env.update(
            {
                "NVS_DATASET_ROOT": config.dataset.root,
                "NVS_DATASET_SPLIT": config.dataset.split,
                "NVS_OUTPUT_DIR": str(config.output_dir),
                "NVS_RUN_ID": config.run_id,
                "NVS_METHOD_ID": config.method,
            }
        )
        if checkpoint_path:
            env["NVS_CHECKPOINT_PATH"] = checkpoint_path
        if rendered_dir:
            env["NVS_RENDERED_DIR"] = rendered_dir
        return env

    def train(self, request: TrainRequest) -> TrainResult:
        """Executa treino externo ou usa caminho de checkpoint já informado."""
        self.validate_config(request.config)
        start = perf_counter()

        extra = request.config.extra
        train_command = self._normalize_command(extra.get("train_command"))
        checkpoint_path = extra.get("checkpoint_path")

        checkpoint_dir = Path(request.config.output_dir) / request.config.run_id / self.method_id / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        fallback_checkpoint = checkpoint_dir / "external.ckpt"

        if train_command:
            env = self._build_env(request.config, checkpoint_path=str(fallback_checkpoint))
            subprocess.run(train_command, check=True, env=env)

        resolved_checkpoint = Path(checkpoint_path) if checkpoint_path else fallback_checkpoint
        if not resolved_checkpoint.exists():
            raise RuntimeError(
                "Checkpoint externo nao encontrado apos treino. "
                "Forneca 'checkpoint_path' valido em RunConfig.extra ou garanta que train_command produza o arquivo."
            )

        elapsed = perf_counter() - start
        return TrainResult(
            method=self.method_id,
            checkpoint_path=str(resolved_checkpoint),
            train_seconds=elapsed,
            output_dir=str(checkpoint_dir.parent),
            logs={"mode": "external", "train_command": train_command},
        )

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Executa inferência externa ou usa diretório de renders pré-computado."""
        self.validate_config(request.config)
        start = perf_counter()

        extra = request.config.extra
        infer_command = self._normalize_command(extra.get("infer_command"))
        rendered_dir = extra.get("rendered_dir")

        output_render_dir = Path(request.config.output_dir) / request.config.run_id / self.method_id / "renders"
        output_render_dir.mkdir(parents=True, exist_ok=True)

        resolved_render_dir = Path(rendered_dir) if rendered_dir else output_render_dir
        if infer_command:
            env = self._build_env(
                request.config,
                checkpoint_path=request.checkpoint_path,
                rendered_dir=str(resolved_render_dir),
            )
            subprocess.run(infer_command, check=True, env=env)

        frames = int(extra.get("frames", 0))
        if frames <= 0 and resolved_render_dir.exists():
            frames = len(list(resolved_render_dir.glob("*.png")))
        if frames <= 0:
            raise RuntimeError(
                "Nenhum frame PNG encontrado no rendered_dir externo. "
                "Forneca 'rendered_dir' valido em RunConfig.extra ou garanta que infer_command gere imagens PNG."
            )

        elapsed = perf_counter() - start
        return InferenceResult(
            method=self.method_id,
            rendered_dir=str(resolved_render_dir),
            frames=frames,
            inference_seconds=elapsed,
            logs={"mode": "external", "infer_command": infer_command},
        )

    def collect_performance(self) -> PerformanceStats:
        """Retorna métricas vazias quando não houver coleta externa."""
        return PerformanceStats(
            fps=0.0,
            vram_gb_peak=0.0,
            train_seconds=0.0,
            inference_seconds=0.0,
        )
