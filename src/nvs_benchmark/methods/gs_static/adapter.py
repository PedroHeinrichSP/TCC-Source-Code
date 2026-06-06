"""Adaptador de 3D Gaussian Splatting estático no contrato unificado."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
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
from nvs_benchmark.core.presets import resolve_iterations
from nvs_benchmark.methods.scene_converters import has_pose_priors, has_real_scene_layout
from nvs_benchmark.methods.subprocess_utils import format_subprocess_error, run_subprocess_streaming


class GSStaticRuntimeError(RuntimeError):
    """Erro base de ambiente para a integracao do 3D Gaussian Splatting."""


class GSStaticHardwareError(GSStaticRuntimeError):
    """Indica host incompatível com a execucao real do 3DGS."""


class GSStaticDependencyError(GSStaticRuntimeError):
    """Indica dependencias do runtime 3DGS ausentes ou incompletas."""


@dataclass
class GSStaticAdapter:
    """Adaptador para integração de 3D Gaussian Splatting estático.

    A integração chama os scripts oficiais do repositório 3DGS via subprocess,
    mantendo os contratos de treino/inferência do benchmark.
    """

    method_id: str = "gs_static"
    display_name: str = "3DGS Static (graphdeco-inria/gaussian-splatting)"
    capabilities: MethodCapabilities = MethodCapabilities(
        supports_train=True,
        supports_inference=True,
        supports_dynamic_scene=False,
        supports_limited_gpu=True,
    )
    base_repo_url: str = "https://github.com/graphdeco-inria/gaussian-splatting"
    repo_path: str = "./third_party/gaussian_splatting"
    python_executable: str = sys.executable

    def validate_config(self, config: RunConfig) -> None:
        """Valida configuração mínima para 3DGS estático."""
        if config.method != self.method_id:
            raise ValueError(f"Metodo incompativel. Esperado '{self.method_id}', recebido '{config.method}'")
        if config.dataset.name not in {"blender_synthetic", "d_nerf", "custom", "mipnerf360", "tanks_and_temples"}:
            raise ValueError("Dataset nao suportado para GS estatico neste estagio.")

        dataset_root = Path(config.dataset.root)
        if not dataset_root.exists():
            raise ValueError(f"Dataset root nao encontrado: {dataset_root}")
        if config.dataset.name in {"mipnerf360", "tanks_and_temples"}:
            if not has_real_scene_layout(dataset_root):
                raise ValueError(
                    f"{config.dataset.name} requer uma cena extraida com images/, sparse/0, poses_bounds.npy ou imagens diretas."
                )
            if not has_pose_priors(dataset_root):
                raise ValueError(
                    f"{config.dataset.name} requer poses/cameras em sparse/0 ou transforms_*.json para treino GS estatico."
                )

        repo_path = self._resolve_repo_path(config)
        if not (repo_path / "train.py").exists() or not (repo_path / "render.py").exists():
            raise ValueError(
                "Repositorio 3DGS nao encontrado ou incompleto. "
                f"Esperado em: {repo_path} (com train.py e render.py)."
            )
        runtime = self._probe_runtime(config)
        torch_error = runtime.get("torch_error")
        if torch_error:
            raise GSStaticDependencyError(
                "Runtime do gs_static nao possui torch disponivel no interpretador configurado. "
                f"Detalhe: {torch_error}"
            )
        if not bool(runtime.get("cuda_available")):
            raise GSStaticHardwareError(
                "gs_static requer CUDA para execucao real. "
                f"Interpretador: {runtime.get('python_executable', self._resolve_python(config))}"
            )
        module_errors = runtime.get("module_errors", {})
        missing_modules = [name for name, error in module_errors.items() if error]
        if missing_modules:
            details = "; ".join(f"{name}: {module_errors[name]}" for name in missing_modules)
            raise GSStaticDependencyError(
                "Dependencias nativas do Gaussian Splatting nao estao disponiveis. "
                f"Modulos com erro: {details}"
            )

    def train(self, request: TrainRequest) -> TrainResult:
        """Executa treino real via script oficial train.py do 3DGS."""
        self.validate_config(request.config)
        start = perf_counter()

        checkpoint_dir = Path(request.config.output_dir) / request.config.run_id / self.method_id / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model_dir = checkpoint_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)

        command = self._build_train_command(request.config, model_dir)
        env = self._build_env(request.config)
        self._run_command(
            command=command,
            cwd=self._resolve_repo_path(request.config),
            env=env,
            stage="train",
        )

        checkpoint_path = str(model_dir)
        explicit_checkpoint = request.config.extra.get("gs_checkpoint_path")
        if explicit_checkpoint:
            checkpoint_path = str(Path(explicit_checkpoint))

        elapsed = perf_counter() - start
        return TrainResult(
            method=self.method_id,
            checkpoint_path=checkpoint_path,
            train_seconds=elapsed,
            output_dir=str(checkpoint_dir.parent),
            logs={
                "mode": "official_subprocess",
                "repo": self.base_repo_url,
                "repo_path": str(self._resolve_repo_path(request.config)),
                "command": command,
            },
        )

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Executa inferência real via script oficial render.py do 3DGS."""
        self.validate_config(request.config)
        start = perf_counter()
        eval_split = self._resolve_eval_split(request)

        render_dir = Path(request.config.output_dir) / request.config.run_id / self.method_id / "renders"
        render_dir.mkdir(parents=True, exist_ok=True)

        command = self._build_render_command(request.config, request.checkpoint_path, split=eval_split)
        env = self._build_env(request.config, checkpoint_path=request.checkpoint_path, rendered_dir=render_dir)
        self._run_command(
            command=command,
            cwd=self._resolve_repo_path(request.config),
            env=env,
            stage="infer",
        )

        source_images = self._resolve_render_sources(
            request.config,
            request.checkpoint_path,
            split=eval_split,
        )
        reference_images = self._resolve_reference_sources(source_images)
        frames = self._copy_renders_to_output(
            source_images=source_images,
            render_dir=render_dir,
            dataset_root=Path(request.config.dataset.root),
            split=eval_split,
        )
        reference_dir = Path(request.config.output_dir) / request.config.run_id / self.method_id / "references_backend"
        reference_frames = self._copy_reference_frames_to_output(
            source_images=reference_images,
            reference_dir=reference_dir,
        )

        if frames == 0:
            raise RuntimeError(
                "Inferencia 3DGS finalizou sem imagens renderizadas. "
                "Verifique caminho do modelo, parametros de render e formato do dataset."
            )

        elapsed = perf_counter() - start
        return InferenceResult(
            method=self.method_id,
            rendered_dir=str(render_dir),
            frames=frames,
            inference_seconds=elapsed,
            logs={
                "mode": "official_subprocess",
                "checkpoint_used": request.checkpoint_path,
                "eval_split": eval_split,
                "command": command,
                "reference_dir": str(reference_dir) if reference_frames > 0 else None,
                "reference_frames": reference_frames,
            },
        )

    def collect_performance(self) -> PerformanceStats:
        """Retorna métricas sintéticas de desempenho.

        A coleta real de VRAM/latencias pode ser adicionada em etapa futura.
        """
        return PerformanceStats(
            fps=0.0,
            vram_gb_peak=0.0,
            train_seconds=0.0,
            inference_seconds=0.0,
        )

    def _resolve_repo_path(self, config: RunConfig) -> Path:
        configured = config.extra.get("gs_repo_path", self.repo_path)
        return Path(str(configured)).resolve()

    def _resolve_python(self, config: RunConfig) -> str:
        configured = config.extra.get("python_executable", self.python_executable)
        resolved = str(configured).strip() if configured is not None else ""
        return resolved or sys.executable

    def _probe_runtime(self, config: RunConfig) -> dict[str, object]:
        python_executable = self._resolve_python(config)
        repo_path = self._resolve_repo_path(config)
        probe_script = """
import importlib
import json
import sys

result = {
    "python_executable": sys.executable,
    "cuda_available": False,
    "module_errors": {},
}

try:
    import torch
except Exception as exc:
    result["torch_error"] = f"{type(exc).__name__}: {exc}"
else:
    result["torch_version"] = getattr(torch, "__version__", "unknown")
    result["cuda_available"] = bool(torch.cuda.is_available())
    for module_name in ("diff_gaussian_rasterization", "simple_knn._C", "plyfile"):
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            result["module_errors"][module_name] = f"{type(exc).__name__}: {exc}"
        else:
            result["module_errors"][module_name] = None

print(json.dumps(result))
""".strip()
        completed = subprocess.run(
            [python_executable, "-c", probe_script],
            cwd=str(repo_path),
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            stderr = self._decode_output(completed.stderr)[-4000:]
            raise GSStaticDependencyError(
                "Nao foi possivel inspecionar o runtime do Gaussian Splatting. "
                f"Interpretador: {python_executable}. STDERR: {stderr}"
            )
        stdout = self._decode_output(completed.stdout).strip()
        if not stdout:
            raise GSStaticDependencyError("Probe do runtime do Gaussian Splatting nao retornou dados.")
        try:
            return json.loads(stdout.splitlines()[-1])
        except json.JSONDecodeError as exc:
            raise GSStaticDependencyError(
                "Probe do runtime do Gaussian Splatting retornou saida invalida."
            ) from exc

    def _normalize_command(self, value: object | None) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, str):
            return shlex.split(value)
        raise ValueError("Comando deve ser list[str] ou string.")

    def _build_train_command(self, config: RunConfig, model_dir: Path) -> list[str]:
        custom = self._normalize_command(config.extra.get("gs_train_command"))
        if custom is not None:
            return custom

        command = [
            self._resolve_python(config),
            "train.py",
            "-s",
            str(Path(config.dataset.root).resolve()),
            "-m",
            str(model_dir.resolve()),
        ]

        iter_params = resolve_iterations(
            method_id=self.method_id,
            preset_name=config.extra.get("preset"),
            iterations=config.extra.get("iterations"),
            extra=config.extra,
            hardware_profile=config.hardware_profile,
        )
        gs_iterations = iter_params.get("iterations")
        if gs_iterations is not None:
            command.extend(["--iterations", str(gs_iterations)])
        gs_resolution = config.extra.get("gs_resolution", iter_params.get("gs_resolution"))
        if gs_resolution is not None:
            command.extend(["--resolution", str(int(gs_resolution))])

        if self._uses_blender_synthetic_defaults(config):
            command.extend(["--eval", "--white_background"])

        extra_args = self._normalize_command(config.extra.get("gs_train_args"))
        if extra_args:
            command.extend(extra_args)
        return command

    def _build_render_command(self, config: RunConfig, checkpoint_path: str, split: str = "test") -> list[str]:
        custom = self._normalize_command(config.extra.get("gs_render_command"))
        if custom is not None:
            return custom

        iter_params = resolve_iterations(
            method_id=self.method_id,
            preset_name=config.extra.get("preset"),
            iterations=config.extra.get("iterations"),
            extra=config.extra,
            hardware_profile=config.hardware_profile,
        )
        command = [
            self._resolve_python(config),
            "render.py",
            "-m",
            str(Path(checkpoint_path).resolve()),
        ]
        gs_resolution = config.extra.get("gs_resolution", iter_params.get("gs_resolution"))
        if gs_resolution is not None:
            command.extend(["--resolution", str(int(gs_resolution))])
        if self._uses_blender_synthetic_defaults(config):
            command.extend(
                [
                    "-s",
                    str(Path(config.dataset.root).resolve()),
                    "--eval",
                    "--white_background",
                ]
            )
        if split == "test":
            command.append("--skip_train")
        elif split == "train":
            command.append("--skip_test")

        extra_args = self._normalize_command(config.extra.get("gs_render_args"))
        if extra_args:
            command.extend(extra_args)
        return command

    def _resolve_eval_split(self, request: InferenceRequest) -> str:
        explicit = str(request.config.extra.get("gs_eval_split", "")).strip().lower()
        if explicit:
            return explicit

        requested = str(request.split).strip().lower()
        if requested:
            return requested

        dataset_split = str(request.config.dataset.split).strip().lower()
        if dataset_split:
            return dataset_split

        return "test"

    def _uses_blender_synthetic_defaults(self, config: RunConfig) -> bool:
        return config.dataset.name == "blender_synthetic"

    def _build_env(
        self,
        config: RunConfig,
        checkpoint_path: str | None = None,
        rendered_dir: Path | None = None,
    ) -> dict[str, str]:
        env = os.environ.copy()
        
        # Adicionar raiz do projeto ao PYTHONPATH para que submódulos locais 
        # sejam descobertos quando o subprocess roda. Usa a raiz do projeto (não gs_splatting directory).
        project_root = str(self._resolve_repo_path(config).parent.parent.resolve())
        pythonpath = env.get("PYTHONPATH", "")
        if project_root not in pythonpath:
            if pythonpath:
                pythonpath = f"{project_root}{os.pathsep}{pythonpath}"
            else:
                pythonpath = project_root
            env["PYTHONPATH"] = pythonpath
        
        env.update(
            {
                "NVS_DATASET_ROOT": str(Path(config.dataset.root).resolve()),
                "NVS_DATASET_SPLIT": config.dataset.split,
                "NVS_OUTPUT_DIR": str(Path(config.output_dir).resolve()),
                "NVS_RUN_ID": config.run_id,
                "NVS_METHOD_ID": config.method,
            }
        )
        if checkpoint_path:
            env["NVS_CHECKPOINT_PATH"] = str(Path(checkpoint_path).resolve())
        if rendered_dir is not None:
            env["NVS_RENDERED_DIR"] = str(rendered_dir.resolve())
        return env

    def _run_command(self, command: list[str], cwd: Path, env: dict[str, str], stage: str) -> None:
        print(f"[{self.method_id}] Executando {stage}: {' '.join(command[:4])}...", flush=True)
        completed = run_subprocess_streaming(
            command,
            cwd=cwd,
            env=env,
        )
        if not completed.success:
            raise RuntimeError(
                f"Falha em gs_static::{stage} (exit={completed.returncode}).\n"
                f"Comando: {' '.join(command)}\n"
                f"CWD: {cwd}\n"
                f"{format_subprocess_error(completed)}"
            )

    @staticmethod
    def _decode_output(payload: bytes | str | None) -> str:
        if payload is None:
            return ""
        if isinstance(payload, bytes):
            return payload.decode("utf-8", errors="replace")
        return payload

    def _resolve_render_sources(self, config: RunConfig, checkpoint_path: str, split: str = "test") -> list[Path]:
        explicit_render_dir = config.extra.get("gs_render_output_dir")
        roots: list[Path] = []
        if explicit_render_dir:
            roots.append(Path(str(explicit_render_dir)).resolve())
        roots.append(Path(checkpoint_path).resolve())

        images: list[Path] = []
        for root in roots:
            if not root.exists():
                continue

            preferred_splits = [split]
            if "test" not in preferred_splits:
                preferred_splits.append("test")
            if "train" not in preferred_splits:
                preferred_splits.append("train")

            for split_name in preferred_splits:
                split_renders = sorted(root.glob(f"{split_name}/ours*/renders/*.png"))
                if split_renders:
                    return split_renders

            renders = sorted(root.rglob("*.png"))
            if renders:
                images.extend(renders)
                break
        return images

    def _resolve_reference_sources(self, render_sources: list[Path]) -> list[Path]:
        if not render_sources:
            return []

        render_dir = render_sources[0].parent
        if render_dir.name != "renders":
            return []

        gt_dir = render_dir.parent / "gt"
        if not gt_dir.exists():
            return []
        return sorted(gt_dir.glob("*.png"))

    def _copy_renders_to_output(
        self,
        source_images: list[Path],
        render_dir: Path,
        dataset_root: Path,
        split: str,
    ) -> int:
        if not source_images:
            return 0

        render_dir.mkdir(parents=True, exist_ok=True)
        for old in render_dir.glob("*.png"):
            old.unlink(missing_ok=True)

        copied = 0

        for index, source in enumerate(source_images):
            # Normalize filenames to the benchmark convention used by
            # exported references so quality metrics can match pairs.
            target_name = f"frame_{index:04d}.png"
            shutil.copyfile(source, render_dir / target_name)
            copied += 1
        return copied

    def _copy_reference_frames_to_output(self, source_images: list[Path], reference_dir: Path) -> int:
        if not source_images:
            return 0

        reference_dir.mkdir(parents=True, exist_ok=True)
        for old in reference_dir.glob("*.png"):
            old.unlink(missing_ok=True)

        copied = 0
        for index, source in enumerate(source_images):
            target_name = f"frame_{index:04d}.png"
            shutil.copyfile(source, reference_dir / target_name)
            copied += 1
        return copied
