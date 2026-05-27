"""Adaptador real de 4D Gaussian Splatting dinamico no contrato unificado."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
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


class GSDynamicRuntimeError(RuntimeError):
    """Erro base de ambiente para a integracao do 4D Gaussian Splatting."""


class GSDynamicHardwareError(GSDynamicRuntimeError):
    """Indica host incompatível com a execucao real do 4DGS."""


class GSDynamicDependencyError(GSDynamicRuntimeError):
    """Indica dependencias do runtime 4DGS ausentes ou incompletas."""


def _find_images(directory: Path) -> list[Path]:
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(directory.rglob(pattern))
    return sorted(files)


@dataclass
class GSDynamicAdapter:
    """Adaptador real para 4D Gaussian Splatting.

    Integra o repositório hustvl/4DGaussians via subprocess. O suporte nativo
    automatizado deste benchmark é focado em cenas dinamicas do dataset D-NeRF.
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
    repo_path: str = "./third_party/4d_gaussians"
    python_executable: str = field(default_factory=lambda: sys.executable)

    _last_train_seconds: float = field(default=0.0, init=False, repr=False)
    _last_infer_seconds: float = field(default=0.0, init=False, repr=False)
    _last_frames: int = field(default=0, init=False, repr=False)

    def validate_config(self, config: RunConfig) -> None:
        """Valida configuração mínima para 4DGS dinamico."""
        if config.method != self.method_id:
            raise ValueError(f"Metodo incompativel. Esperado '{self.method_id}', recebido '{config.method}'")

        if config.dataset.name == "blender_synthetic":
            raise ValueError(
                "4DGaussians nao deve ser usado com blender_synthetic neste benchmark. "
                "Use um dataset realmente dinamico, como d_nerf."
            )
        if config.dataset.name not in {"d_nerf", "custom", "mipnerf360", "tanks_and_temples"}:
            raise ValueError("Dataset nao suportado para GS dinamico neste estagio.")

        dataset_root = Path(config.dataset.root)
        if not dataset_root.exists():
            raise ValueError(f"Dataset root nao encontrado: {dataset_root}")

        repo_path = self._resolve_repo_path(config)
        if not (repo_path / "train.py").exists() or not (repo_path / "render.py").exists():
            raise ValueError(
                "Repositorio 4DGaussians nao encontrado ou incompleto. "
                f"Esperado em: {repo_path} (com train.py e render.py)."
            )

        if config.dataset.name == "d_nerf":
            has_time_metadata = bool(config.dataset.metadata.get("has_time_metadata", False))
            if not has_time_metadata:
                raise ValueError(
                    "Dataset d_nerf sem metadados temporais detectados. "
                    "As entradas de transforms devem conter campo 'time'."
                )
        elif config.dataset.name in {"mipnerf360", "tanks_and_temples"}:
            if not has_real_scene_layout(dataset_root):
                raise ValueError(
                    f"{config.dataset.name} requer uma cena extraida com images/, sparse/0, poses_bounds.npy ou imagens diretas."
                )
            if not has_pose_priors(dataset_root):
                raise ValueError(
                    f"{config.dataset.name} requer poses/cameras em sparse/0 ou transforms_*.json para treino GS dinamico."
                )
        elif config.dataset.name == "custom":
            custom_cfg = config.extra.get("gs_dynamic_config_file")
            if not custom_cfg:
                raise ValueError(
                    "Dataset custom para gs_dynamic requer 'gs_dynamic_config_file' apontando para "
                    "um config compativel com 4DGaussians."
                )

        runtime = self._probe_runtime(config)
        torch_error = runtime.get("torch_error")
        if torch_error:
            raise GSDynamicDependencyError(
                "Runtime do gs_dynamic nao possui torch disponivel no interpretador configurado. "
                f"Detalhe: {torch_error}"
            )
        if not bool(runtime.get("cuda_available")):
            raise GSDynamicHardwareError(
                "gs_dynamic requer CUDA para execucao real. "
                f"Interpretador: {runtime.get('python_executable', self._resolve_python(config))}"
            )
        module_errors = runtime.get("module_errors", {})
        missing_modules = [name for name, error in module_errors.items() if error]
        if missing_modules:
            details = "; ".join(f"{name}: {module_errors[name]}" for name in missing_modules)
            raise GSDynamicDependencyError(
                "Dependencias do 4DGaussians nao estao disponiveis. "
                f"Modulos com erro: {details}"
            )

    def train(self, request: TrainRequest) -> TrainResult:
        """Executa treino real via train.py do 4DGaussians."""
        self.validate_config(request.config)
        start = perf_counter()

        output_base = Path(request.config.output_dir) / request.config.run_id / self.method_id
        model_dir = output_base / "checkpoints" / "model"
        model_dir.mkdir(parents=True, exist_ok=True)

        config_file = self._resolve_config_file(request.config, output_base=output_base)
        command = self._build_train_command(request.config, model_dir, config_file)
        env = self._build_env(request.config)

        self._run_command(
            command=command,
            cwd=self._resolve_repo_path(request.config),
            env=env,
            stage="train",
        )

        elapsed = perf_counter() - start
        self._last_train_seconds = elapsed

        return TrainResult(
            method=self.method_id,
            checkpoint_path=str(model_dir),
            train_seconds=elapsed,
            output_dir=str(output_base),
            logs={
                "mode": "official_subprocess",
                "repo": self.base_repo_url,
                "repo_path": str(self._resolve_repo_path(request.config)),
                "command": command,
                "config_file": str(config_file),
            },
        )

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Executa renderização real via render.py do 4DGaussians."""
        self.validate_config(request.config)
        start = perf_counter()
        eval_split = str(request.config.extra.get("gs_dynamic_eval_split", "test")).strip().lower() or "test"

        output_base = Path(request.config.output_dir) / request.config.run_id / self.method_id
        render_dir = output_base / "renders"
        render_dir.mkdir(parents=True, exist_ok=True)

        model_dir = Path(request.checkpoint_path)
        config_file = self._resolve_config_file(request.config, output_base=output_base)
        command = self._build_render_command(request.config, model_dir, config_file, split=eval_split)
        env = self._build_env(request.config, checkpoint_path=request.checkpoint_path, rendered_dir=render_dir)

        self._run_command(
            command=command,
            cwd=self._resolve_repo_path(request.config),
            env=env,
            stage="infer",
        )

        source_images = self._resolve_render_sources(model_dir, split=eval_split)
        frames = self._copy_renders_to_output(source_images=source_images, render_dir=render_dir)
        if frames == 0:
            raise RuntimeError(
                "Inferencia 4DGS finalizou sem imagens renderizadas. "
                "Verifique checkpoint, config e estrutura do dataset."
            )

        elapsed = perf_counter() - start
        self._last_infer_seconds = elapsed
        self._last_frames = frames

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
                "config_file": str(config_file),
            },
        )

    def collect_performance(self) -> PerformanceStats:
        """Retorna métricas de desempenho da última execução real."""
        fps = 0.0
        if self._last_infer_seconds > 0 and self._last_frames > 0:
            fps = float(self._last_frames / self._last_infer_seconds)

        vram_gb = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                vram_gb = float(torch.cuda.max_memory_allocated() / (1024**3))
        except Exception:
            pass

        return PerformanceStats(
            fps=fps,
            vram_gb_peak=vram_gb,
            train_seconds=self._last_train_seconds,
            inference_seconds=self._last_infer_seconds,
        )

    def _resolve_repo_path(self, config: RunConfig) -> Path:
        configured = config.extra.get("gs_dynamic_repo_path", self.repo_path)
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
    for module_name in ("mmcv", "simple_knn._C", "plyfile"):
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
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr[-4000:] if completed.stderr else ""
            raise GSDynamicDependencyError(
                "Nao foi possivel inspecionar o runtime do 4DGaussians. "
                f"Interpretador: {python_executable}. STDERR: {stderr}"
            )
        stdout = completed.stdout.strip()
        if not stdout:
            raise GSDynamicDependencyError("Probe do runtime do 4DGaussians nao retornou dados.")
        try:
            return json.loads(stdout.splitlines()[-1])
        except json.JSONDecodeError as exc:
            raise GSDynamicDependencyError(
                "Probe do runtime do 4DGaussians retornou saida invalida."
            ) from exc

    def _normalize_command(self, value: object | None) -> list[str] | None:
        if value is None:
            return None
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, str):
            return shlex.split(value)
        raise ValueError("Comando deve ser list[str] ou string.")

    def _resolve_config_file(self, config: RunConfig, *, output_base: Path) -> Path:
        explicit = config.extra.get("gs_dynamic_config_file")
        if explicit:
            return Path(str(explicit)).resolve()

        config_dir = output_base / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        scene_name = Path(config.dataset.root).name
        config_path = config_dir / f"{scene_name}_4dgs_benchmark.py"
        config_path.write_text(self._build_dnerf_config_text(config), encoding="utf-8")
        return config_path

    def _build_dnerf_config_text(self, config: RunConfig) -> str:
        if config.dataset.name not in {"d_nerf", "mipnerf360", "tanks_and_temples"}:
            raise ValueError(
                "Geracao automatica de config 4DGS so esta disponivel para d_nerf, mipnerf360 ou tanks_and_temples."
            )

        iter_params = resolve_iterations(
            method_id=self.method_id,
            preset_name=config.extra.get("preset"),
            iterations=config.extra.get("iterations"),
            extra=config.extra,
            hardware_profile=config.hardware_profile,
        )
        fine_iterations = int(iter_params.get("iterations", 20000))
        coarse_iterations = int(
            config.extra.get(
                "gs_dynamic_coarse_iterations",
                min(3000, max(100, fine_iterations // 4)),
            )
        )
        time_resolution = int(config.extra.get("gs_dynamic_time_resolution", 25))
        spatial_resolution = list(config.extra.get("gs_dynamic_spatial_resolution", [64, 64, 64]))
        multires = list(config.extra.get("gs_dynamic_multires", [1, 2]))
        net_width = int(config.extra.get("gs_dynamic_net_width", 64))
        defor_depth = int(config.extra.get("gs_dynamic_defor_depth", 0))
        bounds = float(config.extra.get("gs_dynamic_bounds", 1.6))
        render_process = bool(config.extra.get("gs_dynamic_render_process", False))

        return (
            "OptimizationParams = dict(\n"
            f"    coarse_iterations = {coarse_iterations},\n"
            f"    iterations = {fine_iterations},\n"
            "    deformation_lr_init = 0.00016,\n"
            "    deformation_lr_final = 0.0000016,\n"
            "    deformation_lr_delay_mult = 0.01,\n"
            "    grid_lr_init = 0.0016,\n"
            "    grid_lr_final = 0.000016,\n"
            "    pruning_interval = 8000,\n"
            "    percent_dense = 0.01,\n"
            f"    render_process = {str(render_process)},\n"
            "    weight_decay_iteration = 0,\n"
            ")\n\n"
            "ModelHiddenParams = dict(\n"
            "    kplanes_config = {\n"
            "        'grid_dimensions': 2,\n"
            "        'input_coordinate_dim': 4,\n"
            "        'output_coordinate_dim': 32,\n"
            f"        'resolution': [{spatial_resolution[0]}, {spatial_resolution[1]}, {spatial_resolution[2]}, {time_resolution}],\n"
            "    },\n"
            f"    multires = {multires},\n"
            f"    defor_depth = {defor_depth},\n"
            f"    net_width = {net_width},\n"
            "    plane_tv_weight = 0.0001,\n"
            "    time_smoothness_weight = 0.01,\n"
            "    l1_time_planes = 0.0001,\n"
            "    weight_decay_iteration = 0,\n"
            f"    bounds = {bounds},\n"
            ")\n"
        )

    def _build_train_command(self, config: RunConfig, model_dir: Path, config_file: Path) -> list[str]:
        custom = self._normalize_command(config.extra.get("gs_dynamic_train_command"))
        if custom is not None:
            return custom

        command = [
            self._resolve_python(config),
            "train.py",
            "-s",
            str(Path(config.dataset.root).resolve()),
            "--model_path",
            str(model_dir.resolve()),
            "--expname",
            f"benchmark/{config.dataset.name}/{Path(config.dataset.root).name}",
            "--configs",
            str(config_file.resolve()),
            "--port",
            str(int(config.extra.get("gs_dynamic_port", 6017))),
        ]

        extra_args = self._normalize_command(config.extra.get("gs_dynamic_train_args"))
        if extra_args:
            command.extend(extra_args)
        return command

    def _build_render_command(
        self,
        config: RunConfig,
        model_dir: Path,
        config_file: Path,
        *,
        split: str = "test",
    ) -> list[str]:
        custom = self._normalize_command(config.extra.get("gs_dynamic_render_command"))
        if custom is not None:
            return custom

        command = [
            self._resolve_python(config),
            "render.py",
            "--model_path",
            str(model_dir.resolve()),
            "--configs",
            str(config_file.resolve()),
            "--skip_video",
        ]

        iteration = config.extra.get("gs_dynamic_render_iteration")
        if iteration is not None:
            command.extend(["--iteration", str(int(iteration))])

        if split == "test":
            command.append("--skip_train")
        elif split == "train":
            command.append("--skip_test")

        extra_args = self._normalize_command(config.extra.get("gs_dynamic_render_args"))
        if extra_args:
            command.extend(extra_args)
        return command

    def _build_env(
        self,
        config: RunConfig,
        checkpoint_path: str | None = None,
        rendered_dir: Path | None = None,
    ) -> dict[str, str]:
        env = os.environ.copy()
        project_root = str(self._resolve_repo_path(config).parent.parent.resolve())
        pythonpath = env.get("PYTHONPATH", "")
        if project_root not in pythonpath:
            pythonpath = f"{project_root}{os.pathsep}{pythonpath}" if pythonpath else project_root
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
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            stdout = completed.stdout[-4000:] if completed.stdout else ""
            stderr = completed.stderr[-4000:] if completed.stderr else ""
            raise RuntimeError(
                f"Falha em gs_dynamic::{stage} (exit={completed.returncode}).\n"
                f"Comando: {' '.join(command)}\n"
                f"CWD: {cwd}\n"
                f"STDOUT:\n{stdout}\n"
                f"STDERR:\n{stderr}"
            )

    def _resolve_render_sources(self, model_dir: Path, *, split: str = "test") -> list[Path]:
        if not model_dir.exists():
            return []

        preferred = [split]
        if "test" not in preferred:
            preferred.append("test")
        if "train" not in preferred:
            preferred.append("train")

        for split_name in preferred:
            split_root = model_dir / split_name
            if split_root.exists():
                candidates = sorted(split_root.glob("ours_*/renders"))
                if candidates:
                    latest = candidates[-1]
                    renders = _find_images(latest)
                    if renders:
                        return renders

        fallback_dirs = sorted(model_dir.rglob("renders"))
        for directory in fallback_dirs:
            renders = _find_images(directory)
            if renders:
                return renders

        fallback = _find_images(model_dir)
        return fallback

    def _copy_renders_to_output(self, source_images: list[Path], render_dir: Path) -> int:
        if not source_images:
            return 0

        render_dir.mkdir(parents=True, exist_ok=True)
        for old in render_dir.glob("*.png"):
            old.unlink(missing_ok=True)

        copied = 0
        for index, source in enumerate(source_images):
            target_name = f"frame_{index:04d}.png"
            shutil.copyfile(source, render_dir / target_name)
            copied += 1
        return copied
