"""Adaptador real de D-NeRF dinâmico usando repositório oficial.

Integra diretamente com o repositório albertpumarola/D-NeRF via subprocess,
executando treino e renderização de cenas dinâmicas com componente temporal.
"""

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


def _find_images(directory: Path) -> list[Path]:
    """Busca imagens PNG/JPG em um diretório recursivamente."""
    patterns = ("*.png", "*.jpg", "*.jpeg")
    files: list[Path] = []
    for pattern in patterns:
        files.extend(directory.rglob(pattern))
    return sorted(files)


@dataclass
class NeRFDynamicAdapter:
    """Adaptador para D-NeRF dinâmico (albertpumarola/D-NeRF).

    Chama run_dnerf.py do repositório third_party/d_nerf via subprocess.
    Para cenas dinâmicas, usa nerf_type=direct_temporal com suporte temporal.
    """

    method_id: str = "nerf_dynamic"
    display_name: str = "D-NeRF Dynamic (albertpumarola/D-NeRF)"
    capabilities: MethodCapabilities = MethodCapabilities(
        supports_train=True,
        supports_inference=True,
        supports_dynamic_scene=True,
        supports_limited_gpu=True,
    )
    base_repo_url: str = "https://github.com/albertpumarola/D-NeRF"
    repo_path: str = "./third_party/d_nerf"
    python_executable: str = field(default_factory=lambda: sys.executable)

    _last_train_seconds: float = field(default=0.0, init=False, repr=False)
    _last_infer_seconds: float = field(default=0.0, init=False, repr=False)
    _last_frames: int = field(default=0, init=False, repr=False)

    def validate_config(self, config: RunConfig) -> None:
        """Valida configuração para cenários de NeRF dinâmico."""
        if config.method != self.method_id:
            raise ValueError(f"Metodo incompativel. Esperado '{self.method_id}', recebido '{config.method}'")
        if config.dataset.name not in {"d_nerf", "blender_synthetic", "custom"}:
            raise ValueError("Dataset nao suportado para NeRF dinamico neste estagio.")

        dataset_root = Path(config.dataset.root)
        if not dataset_root.exists():
            raise ValueError(f"Dataset root nao encontrado: {dataset_root}")

        repo = self._resolve_repo_path(config)
        if not (repo / "run_dnerf.py").exists():
            raise ValueError(
                f"Repositorio D-NeRF nao encontrado em: {repo}. "
                "Execute: git clone https://github.com/albertpumarola/D-NeRF ./third_party/d_nerf"
            )

    def train(self, request: TrainRequest) -> TrainResult:
        """Executa treino real via run_dnerf.py com suporte temporal."""
        self.validate_config(request.config)
        start = perf_counter()

        output_base = Path(request.config.output_dir) / request.config.run_id / self.method_id
        logs_dir = output_base / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

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
        )

        command = self._build_train_command(request.config, config_file)
        env = self._build_env(request.config)

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
            },
        )

    def infer(self, request: InferenceRequest) -> InferenceResult:
        """Executa renderização das vistas de teste via run_dnerf.py."""
        self.validate_config(request.config)
        start = perf_counter()

        output_base = Path(request.config.output_dir) / request.config.run_id / self.method_id
        render_dir = output_base / "renders"
        render_dir.mkdir(parents=True, exist_ok=True)

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
            logs_dir=checkpoint_dir if checkpoint_dir.is_dir() else checkpoint_dir.parent,
            iter_params=iter_params,
            render_only=True,
            render_test=True,
        )

        command = self._build_render_command(request.config, config_file)
        env = self._build_env(request.config)

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
                "Inferencia D-NeRF finalizou sem imagens renderizadas. "
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
        configured = config.extra.get("dnerf_repo_path", self.repo_path)
        return Path(str(configured)).resolve()

    def _resolve_python(self, config: RunConfig) -> str:
        """Resolve executável Python."""
        configured = config.extra.get("python_executable", self.python_executable)
        resolved = str(configured).strip() if configured is not None else ""
        return resolved or sys.executable

    def _generate_config_file(
        self,
        config: RunConfig,
        logs_dir: Path,
        iter_params: dict,
        render_only: bool = False,
        render_test: bool = False,
    ) -> Path:
        """Gera arquivo de configuração para run_dnerf.py.

        Para cenas dinâmicas, usa nerf_type=direct_temporal com
        suporte completo a campos de deformação temporal.
        """
        dataset_root = Path(config.dataset.root).resolve()
        exp_name = f"{config.run_id}_{self.method_id}"

        n_iter = iter_params.get("N_iter", 1000)

        lines = [
            f"expname = {exp_name}",
            f"basedir = {logs_dir.resolve()}",
            f"datadir = {dataset_root}",
            "dataset_type = blender",
            "",
            "nerf_type = direct_temporal",
            "no_batching = True",
            "not_zero_canonical = False",
            "use_viewdirs = True",
            "white_bkgd = True",
            f"lrate_decay = {max(250, n_iter // 2)}",
            "",
            f"N_iter = {n_iter}",
            "N_samples = 64",
            "N_importance = 128",
            "N_rand = 500",
            "testskip = 1",
            "",
            "precrop_iters = 500",
            f"precrop_iters_time = {min(n_iter // 50, 10000)}",
            "precrop_frac = 0.5",
            "",
            "half_res = True",
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
        custom = config.extra.get("dnerf_train_command")
        if custom:
            return shlex.split(custom) if isinstance(custom, str) else list(custom)

        return [
            self._resolve_python(config),
            "run_dnerf.py",
            "--config", str(config_file.resolve()),
        ]

    def _build_render_command(self, config: RunConfig, config_file: Path) -> list[str]:
        """Monta comando de renderização."""
        custom = config.extra.get("dnerf_render_command")
        if custom:
            return shlex.split(custom) if isinstance(custom, str) else list(custom)

        return [
            self._resolve_python(config),
            "run_dnerf.py",
            "--config", str(config_file.resolve()),
        ]

    def _build_env(self, config: RunConfig) -> dict[str, str]:
        """Monta variáveis de ambiente para o subprocess."""
        env = os.environ.copy()
        env.update({
            "NVS_DATASET_ROOT": str(Path(config.dataset.root).resolve()),
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
