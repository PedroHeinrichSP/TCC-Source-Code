"""Adaptador real de NeRF estático usando D-NeRF (PyTorch) como engine.

Como o NeRF original (bmild/nerf) utiliza TensorFlow 1.15 + Python 3.7,
incompatível com o ambiente atual, este adaptador usa o repositório D-NeRF
(albertpumarola/D-NeRF) em PyTorch como backend. O D-NeRF é um superset
do NeRF original e suporta cenas estáticas nativamente.
"""

from __future__ import annotations

import os
import shlex
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
from nvs_benchmark.methods.scene_converters import (
    find_images,
    has_pose_priors,
    has_real_scene_layout,
    has_required_blender_splits,
    prepare_colmap_scene_to_blender,
    preferred_real_image_subdirs,
    preferred_real_max_image_dim,
)


def _get_hardware_info(config: RunConfig) -> dict:
    """Retorna informacoes de hardware detectadas, se existirem."""
    detected = config.extra.get("detected_hardware", {})
    return detected if isinstance(detected, dict) else {}


@dataclass
class NeRFStaticAdapter:
    """Adaptador para NeRF estático usando D-NeRF (PyTorch) como engine.

    Chama run_dnerf.py do repositório third_party/d_nerf via subprocess.
    Para cenas estáticas, usa nerf_type=original com fundo branco.
    O treino e a renderização são executados como processos separados.
    """

    method_id: str = "nerf_static"
    display_name: str = "NeRF Static (via D-NeRF PyTorch)"
    capabilities: MethodCapabilities = MethodCapabilities(
        supports_train=True,
        supports_inference=True,
        supports_dynamic_scene=False,
        supports_limited_gpu=True,
    )
    base_repo_url: str = "https://github.com/albertpumarola/D-NeRF"
    repo_path: str = "./third_party/d_nerf"
    python_executable: str = field(default_factory=lambda: sys.executable)

    _last_train_seconds: float = field(default=0.0, init=False, repr=False)
    _last_infer_seconds: float = field(default=0.0, init=False, repr=False)
    _last_frames: int = field(default=0, init=False, repr=False)

    def validate_config(self, config: RunConfig) -> None:
        """Valida configuração mínima para NeRF estático."""
        if config.method != self.method_id:
            raise ValueError(f"Metodo incompativel. Esperado '{self.method_id}', recebido '{config.method}'")
        if config.dataset.name not in {"blender_synthetic", "d_nerf", "custom", "mipnerf360", "tanks_and_temples"}:
            raise ValueError(
                "Dataset nao suportado para NeRF estatico neste estagio. "
                "Use 'blender_synthetic', 'd_nerf', 'mipnerf360', 'tanks_and_temples' ou 'custom'."
            )

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
                    f"{config.dataset.name} requer poses/cameras em sparse/0 ou transforms_*.json para treino NeRF."
                )

        repo = self._resolve_repo_path(config)
        if not (repo / "run_dnerf.py").exists():
            raise ValueError(
                f"Repositorio D-NeRF nao encontrado em: {repo}. "
                "Execute: git clone https://github.com/albertpumarola/D-NeRF ./third_party/d_nerf"
            )

    def train(self, request: TrainRequest) -> TrainResult:
        """Executa treino real via run_dnerf.py do repositório D-NeRF."""
        self.validate_config(request.config)
        start = perf_counter()

        output_base = Path(request.config.output_dir) / request.config.run_id / self.method_id
        logs_dir = output_base / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        dataset_root = self._resolve_dataset_root(request.config, output_base)

        iter_params = resolve_iterations(
            method_id=self.method_id,
            preset_name=request.config.extra.get("preset"),
            iterations=request.config.extra.get("iterations"),
            extra=request.config.extra,
            hardware_profile=request.config.hardware_profile,
        )

        config_file = self._generate_config_file(
            request.config,
            logs_dir=logs_dir,
            iter_params=iter_params,
            dataset_root=dataset_root,
        )

        command = self._build_train_command(request.config, config_file)
        env = self._build_env(request.config, dataset_root=dataset_root)

        self._run_command(
            command=command,
            cwd=self._resolve_repo_path(request.config),
            env=env,
            stage="train",
        )

        checkpoint_path = self._find_latest_checkpoint(logs_dir)
        elapsed = perf_counter() - start
        self._last_train_seconds = elapsed

        return TrainResult(
            method=self.method_id,
            checkpoint_path=str(checkpoint_path or logs_dir),
            train_seconds=elapsed,
            output_dir=str(output_base),
            logs={
                "mode": "real_dnerf_subprocess",
                "repo": self.base_repo_url,
                "repo_path": str(self._resolve_repo_path(request.config)),
                "command": command,
                "iter_params": iter_params,
                "dataset_root": str(dataset_root),
            },
        )

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Executa renderização das vistas de teste via run_dnerf.py."""
        self.validate_config(request.config)
        start = perf_counter()

        output_base = Path(request.config.output_dir) / request.config.run_id / self.method_id
        render_dir = output_base / "renders"
        render_dir.mkdir(parents=True, exist_ok=True)
        dataset_root = self._resolve_dataset_root(request.config, output_base)

        checkpoint_dir = Path(request.checkpoint_path)

        iter_params = resolve_iterations(
            method_id=self.method_id,
            preset_name=request.config.extra.get("preset"),
            iterations=request.config.extra.get("iterations"),
            extra=request.config.extra,
            hardware_profile=request.config.hardware_profile,
        )

        config_file = self._generate_config_file(
            request.config,
            # Keep the same basedir used during training so D-NeRF can find ckpts.
            logs_dir=output_base / "logs",
            iter_params=iter_params,
            dataset_root=dataset_root,
            render_only=True,
            render_test=True,
        )

        command = self._build_render_command(request.config, config_file)
        env = self._build_env(request.config, dataset_root=dataset_root)

        self._run_command(
            command=command,
            cwd=self._resolve_repo_path(request.config),
            env=env,
            stage="infer",
        )

        frames = self._collect_renders(
            checkpoint_dir=checkpoint_dir if checkpoint_dir.is_dir() else checkpoint_dir.parent,
            render_dir=render_dir,
        )

        if frames == 0:
            raise RuntimeError(
                "Inferencia NeRF finalizou sem imagens renderizadas. "
                "Verifique se o treino completou com sucesso."
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
                "mode": "real_dnerf_subprocess",
                "checkpoint_used": request.checkpoint_path,
                "command": command,
                "dataset_root": str(dataset_root),
            },
        )

    def collect_performance(self) -> PerformanceStats:
        """Retorna métricas de desempenho da última execução."""
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
        """Resolve caminho do repositório D-NeRF."""
        configured = config.extra.get("nerf_repo_path", self.repo_path)
        return Path(str(configured)).resolve()

    def _resolve_python(self, config: RunConfig) -> str:
        """Resolve executável Python."""
        configured = config.extra.get("python_executable", self.python_executable)
        resolved = str(configured).strip() if configured is not None else ""
        return resolved or sys.executable

    def _resolve_dataset_root(self, config: RunConfig, output_base: Path) -> Path:
        """Resolve a raiz efetiva usada pelo backend do NeRF estático."""
        source_root = Path(config.dataset.root).resolve()
        if has_required_blender_splits(source_root):
            return source_root

        if config.dataset.name not in {"mipnerf360", "tanks_and_temples"}:
            return source_root

        prepared_root = output_base.resolve() / "prepared_dataset" / f"{config.dataset.name}_{source_root.name}"
        return prepare_colmap_scene_to_blender(
            source_root=source_root,
            prepared_root=prepared_root,
            holdout_stride=max(2, int(config.extra.get("nerf_llffhold", 8))),
            preferred_image_subdirs=preferred_real_image_subdirs(
                dataset_name=config.dataset.name,
                preset_name=str(config.extra.get("preset", "")),
            ),
            max_image_dim=(
                int(config.extra["nerf_real_scene_max_dim"])
                if config.extra.get("nerf_real_scene_max_dim") is not None
                else preferred_real_max_image_dim(
                    dataset_name=config.dataset.name,
                    preset_name=str(config.extra.get("preset", "")),
                )
            ),
        )

    def _generate_config_file(
        self,
        config: RunConfig,
        logs_dir: Path,
        iter_params: dict,
        dataset_root: Path,
        render_only: bool = False,
        render_test: bool = False,
    ) -> Path:
        """Gera arquivo de configuração para run_dnerf.py.

        Para cenas estáticas, usa nerf_type=original e define parâmetros
        compatíveis com o NeRF original (sem componente temporal ativo).
        """
        exp_name = f"{config.run_id}_{self.method_id}"

        n_iter = iter_params.get("N_iter", 1000)

        hardware = _get_hardware_info(config)
        has_gpu = bool(hardware.get("has_gpu", False))
        vram_gb = hardware.get("vram_gb")
        high_vram = has_gpu and isinstance(vram_gb, (int, float)) and float(vram_gb) >= 8.0

        n_samples = int(config.extra.get("nerf_n_samples", 64 if high_vram else 32))
        n_importance = int(config.extra.get("nerf_n_importance", 128 if high_vram else 0))
        n_rand = int(config.extra.get("nerf_n_rand", 1024 if high_vram else 128))
        chunk = int(config.extra.get("nerf_chunk", 4096 if high_vram else 1024))
        netchunk = int(config.extra.get("nerf_netchunk", 16384 if high_vram else 4096))
        precrop_iters = int(config.extra.get("nerf_precrop_iters", 0))
        precrop_frac = float(config.extra.get("nerf_precrop_frac", 0.5))
        half_res = bool(config.extra.get("nerf_half_res", not high_vram))
        if config.dataset.name in {"mipnerf360", "tanks_and_temples"}:
            half_res = bool(config.extra.get("nerf_force_half_res_after_prepare", False))

        lines = [
            f"expname = {exp_name}",
            f"basedir = {logs_dir.resolve()}",
            f"datadir = {dataset_root.resolve()}",
            "dataset_type = blender",
            "",
            "nerf_type = original",
            "no_batching = True",
            "not_zero_canonical = False",
            "use_viewdirs = True",
            "white_bkgd = True",
            f"lrate_decay = {max(250, n_iter // 4)}",
            "",
            f"N_iter = {n_iter}",
            f"N_samples = {n_samples}",
            f"N_importance = {n_importance}",
            f"N_rand = {n_rand}",
            f"chunk = {chunk}",
            f"netchunk = {netchunk}",
            "testskip = 1",
            "",
            f"precrop_iters = {precrop_iters}",
            "precrop_iters_time = 0",
            f"precrop_frac = {precrop_frac}",
            "",
            f"half_res = {str(half_res)}",
            "do_half_precision = False",
            "",
            f"i_print = {iter_params.get('i_print', 500)}",
            f"i_img = {min(n_iter, iter_params.get('i_img', 5000))}",
            f"i_weights = {iter_params.get('i_weights', n_iter)}",
            f"i_testset = {iter_params.get('i_testset', n_iter)}",
            f"i_video = {iter_params.get('i_video', n_iter)}",
        ]

        if render_only:
            lines.append("render_only = True")
        if render_test:
            lines.append("render_test = True")

        config_path = logs_dir / f"config_{self.method_id}.txt"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return config_path

    def _build_train_command(self, config: RunConfig, config_file: Path) -> list[str]:
        """Monta comando de treino."""
        custom = config.extra.get("nerf_train_command")
        if custom:
            return shlex.split(custom) if isinstance(custom, str) else list(custom)

        run_script = self._resolve_repo_path(config) / "run_dnerf.py"
        return [
            self._resolve_python(config),
            "-m",
            "nvs_benchmark.methods.dnerf_runner",
            "--run-script",
            str(run_script.resolve()),
            "--config", str(config_file.resolve()),
        ]

    def _build_render_command(self, config: RunConfig, config_file: Path) -> list[str]:
        """Monta comando de renderização (render_only + render_test)."""
        custom = config.extra.get("nerf_render_command")
        if custom:
            return shlex.split(custom) if isinstance(custom, str) else list(custom)

        run_script = self._resolve_repo_path(config) / "run_dnerf.py"
        return [
            self._resolve_python(config),
            "-m",
            "nvs_benchmark.methods.dnerf_runner",
            "--run-script",
            str(run_script.resolve()),
            "--config", str(config_file.resolve()),
        ]

    def _build_env(self, config: RunConfig, dataset_root: Path | None = None) -> dict[str, str]:
        """Monta variáveis de ambiente para o subprocess."""
        env = os.environ.copy()
        
        # Adicionar raiz do projeto ao PYTHONPATH para que submódulos locais (ex: torchsearchsorted) 
        # sejam descobertos quando o subprocess roda. Usa a raiz do projeto (não d_nerf directory).
        project_root = str(self._resolve_repo_path(config).parent.parent.resolve())
        pythonpath = env.get("PYTHONPATH", "")
        if project_root not in pythonpath:
            if pythonpath:
                pythonpath = f"{project_root}{os.pathsep}{pythonpath}"
            else:
                pythonpath = project_root
            env["PYTHONPATH"] = pythonpath
        
        dataset_root = (dataset_root or Path(config.dataset.root)).resolve()
        env.update({
            "NVS_DATASET_ROOT": str(dataset_root),
            "NVS_OUTPUT_DIR": str(Path(config.output_dir).resolve()),
            "NVS_RUN_ID": config.run_id,
            "NVS_METHOD_ID": config.method,
        })
        return env

    def _run_command(self, command: list[str], cwd: Path, env: dict[str, str], stage: str) -> None:
        """Executa comando subprocess com tratamento de erro."""
        print(f"[{self.method_id}] Executando {stage}: {' '.join(command[:4])}...")
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
                f"Falha em {self.method_id}::{stage} (exit={completed.returncode}).\n"
                f"Comando: {' '.join(command)}\n"
                f"CWD: {cwd}\n"
                f"STDOUT:\n{stdout}\n"
                f"STDERR:\n{stderr}"
            )

    def _find_latest_checkpoint(self, logs_dir: Path) -> Path | None:
        """Encontra o checkpoint mais recente no diretório de logs."""
        ckpts: list[Path] = []
        for subdir in logs_dir.rglob("*"):
            if subdir.is_file() and subdir.suffix == ".tar":
                ckpts.append(subdir)
        if ckpts:
            return sorted(ckpts, key=lambda p: p.stat().st_mtime)[-1]

        for subdir in logs_dir.iterdir():
            if subdir.is_dir():
                return subdir
        return logs_dir

    def _collect_renders(self, checkpoint_dir: Path, render_dir: Path) -> int:
        """Copia renders do D-NeRF para o diretório padrão do benchmark."""
        for old in render_dir.glob("*.png"):
            old.unlink(missing_ok=True)

        render_sources: list[Path] = []

        for sub in checkpoint_dir.rglob("renderonly_test_*"):
            if sub.is_dir():
                estim_dir = sub / "estim"
                if estim_dir.exists():
                    render_sources = sorted(estim_dir.glob("*.png"))
                    break
                direct_pngs = sorted(sub.glob("*.png"))
                if direct_pngs:
                    render_sources = direct_pngs
                    break

        if not render_sources:
            render_sources = _find_images(checkpoint_dir)

        copied = 0
        for idx, src in enumerate(render_sources):
            if src.name.endswith(".mp4") or "video" in src.name.lower():
                continue
            target_name = f"frame_{idx:04d}.png"
            shutil.copyfile(src, render_dir / target_name)
            copied += 1

        return copied
