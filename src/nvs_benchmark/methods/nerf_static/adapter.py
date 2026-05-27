"""Adaptador real de NeRF estático usando D-NeRF (PyTorch) como engine.

Como o NeRF original (bmild/nerf) utiliza TensorFlow 1.15 + Python 3.7,
incompatível com o ambiente atual, este adaptador usa o repositório D-NeRF
(albertpumarola/D-NeRF) em PyTorch como backend. O D-NeRF é um superset
do NeRF original e suporta cenas estáticas nativamente.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
from functools import lru_cache
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

import imageio.v2 as imageio
import numpy as np

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


def _get_hardware_info(config: RunConfig) -> dict:
    """Retorna informacoes de hardware detectadas, se existirem."""
    detected = config.extra.get("detected_hardware", {})
    return detected if isinstance(detected, dict) else {}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


@lru_cache(maxsize=1)
def _load_colmap_loader_module():
    module_path = _project_root() / "third_party" / "gaussian_splatting" / "scene" / "colmap_loader.py"
    spec = importlib.util.spec_from_file_location("nvs_benchmark_colmap_loader", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Nao foi possivel carregar helpers COLMAP em: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_colmap_intrinsics_text(path: Path) -> dict[int, dict[str, object]]:
    cameras: dict[int, dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as fp:
        for raw_line in fp:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            elems = line.split()
            camera_id = int(elems[0])
            model = elems[1]
            width = int(elems[2])
            height = int(elems[3])
            params = np.array(tuple(map(float, elems[4:])), dtype=np.float64)
            cameras[camera_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras


def _resolve_colmap_image_path(scene_root: Path, image_name: str) -> Path | None:
    candidates = [
        scene_root / "images" / image_name,
        scene_root / "images" / Path(image_name).name,
        scene_root / "images" / f"{Path(image_name).stem}.png",
        scene_root / "images" / f"{Path(image_name).stem}.jpg",
        scene_root / "images" / f"{Path(image_name).stem}.jpeg",
        scene_root / image_name,
        scene_root / Path(image_name).name,
        scene_root / f"{Path(image_name).stem}.png",
        scene_root / f"{Path(image_name).stem}.jpg",
        scene_root / f"{Path(image_name).stem}.jpeg",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _save_rgba_png(source_path: Path, target_path: Path) -> None:
    image = imageio.imread(source_path)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim == 3 and image.shape[-1] == 3:
        alpha = np.full((*image.shape[:2], 1), 255, dtype=image.dtype)
        image = np.concatenate([image, alpha], axis=-1)
    elif image.ndim == 3 and image.shape[-1] > 4:
        image = image[:, :, :4]

    if image.dtype != np.uint8:
        if np.issubdtype(image.dtype, np.floating):
            if float(np.nanmax(image)) <= 1.0:
                image = np.clip(image * 255.0, 0, 255)
            else:
                image = np.clip(image, 0, 255)
        else:
            image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(target_path, image)


def _has_mipnerf360_scene_layout(root: Path) -> bool:
    return (
        root.exists()
        and root.is_dir()
        and (
            (root / "sparse" / "0").exists()
            or (root / "poses_bounds.npy").exists()
            or (root / "images").exists()
            or bool(_find_images(root))
        )
    )


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
        if config.dataset.name not in {"blender_synthetic", "d_nerf", "custom", "mipnerf360"}:
            raise ValueError(
                "Dataset nao suportado para NeRF estatico neste estagio. "
                "Use 'blender_synthetic', 'd_nerf', 'mipnerf360' ou 'custom'."
            )

        dataset_root = Path(config.dataset.root)
        if not dataset_root.exists():
            raise ValueError(f"Dataset root nao encontrado: {dataset_root}")
        if config.dataset.name == "mipnerf360" and not _has_mipnerf360_scene_layout(dataset_root):
            raise ValueError(
                "Mip-NeRF 360 requer uma cena extraida com images/, sparse/0, poses_bounds.npy ou imagens diretas."
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
        required_splits = ("transforms_train.json", "transforms_val.json", "transforms_test.json")
        if all((source_root / split_file).exists() for split_file in required_splits):
            return source_root

        if config.dataset.name != "mipnerf360":
            return source_root

        prepared_root = output_base / "prepared_dataset" / source_root.name
        return self._prepare_mipnerf360_dataset(
            source_root=source_root,
            prepared_root=prepared_root,
            holdout_stride=max(2, int(config.extra.get("nerf_llffhold", 8))),
        )

    def _prepare_mipnerf360_dataset(self, *, source_root: Path, prepared_root: Path, holdout_stride: int) -> Path:
        """Converte uma cena COLMAP do Mip-NeRF 360 para o formato Blender usado pelo backend."""
        required_splits = ("transforms_train.json", "transforms_val.json", "transforms_test.json")
        if prepared_root.exists() and all((prepared_root / split_file).exists() for split_file in required_splits):
            return prepared_root

        if prepared_root.exists():
            shutil.rmtree(prepared_root)
        prepared_root.mkdir(parents=True, exist_ok=True)

        sparse_root = source_root / "sparse" / "0"
        if not sparse_root.exists():
            raise ValueError(
                f"Mip-NeRF 360 requer diretorio COLMAP em {sparse_root} para conversao ao backend NeRF."
            )

        loader = _load_colmap_loader_module()
        images_bin = sparse_root / "images.bin"
        cameras_bin = sparse_root / "cameras.bin"
        images_txt = sparse_root / "images.txt"
        cameras_txt = sparse_root / "cameras.txt"

        if images_bin.exists() and cameras_bin.exists():
            extrinsics = loader.read_extrinsics_binary(images_bin)
            intrinsics = loader.read_intrinsics_binary(cameras_bin)
        elif images_txt.exists() and cameras_txt.exists():
            extrinsics = loader.read_extrinsics_text(images_txt)
            intrinsics = _read_colmap_intrinsics_text(cameras_txt)
        else:
            raise ValueError(
                f"Nao foi possivel localizar arquivos COLMAP em {sparse_root}: images.bin/text e cameras.bin/text."
            )

        camera_entries = sorted(extrinsics.items(), key=lambda item: item[1].name.lower())
        if not camera_entries:
            raise ValueError(f"Nenhuma camera COLMAP encontrada em {sparse_root}.")

        train_frames: list[dict[str, object]] = []
        test_frames: list[dict[str, object]] = []
        camera_angle_x: float | None = None

        for index, (_, extrinsic) in enumerate(camera_entries):
            intrinsic = intrinsics[extrinsic.camera_id]
            if hasattr(intrinsic, "model"):
                model = intrinsic.model
                width = int(intrinsic.width)
                params = np.asarray(intrinsic.params, dtype=np.float64)
            else:
                model = intrinsic["model"]
                width = int(intrinsic["width"])
                params = np.asarray(intrinsic["params"], dtype=np.float64)

            if model not in {"PINHOLE", "SIMPLE_PINHOLE"}:
                raise ValueError(
                    f"Modelo de camera COLMAP nao suportado para conversao NeRF: {model}."
                )

            source_image = _resolve_colmap_image_path(source_root, extrinsic.name)
            if source_image is None:
                raise ValueError(f"Imagem nao encontrada para COLMAP frame: {extrinsic.name}")

            target_stem = Path(extrinsic.name).stem
            target_image = prepared_root / "images" / f"{target_stem}.png"
            _save_rgba_png(source_image, target_image)

            w2c = np.eye(4, dtype=np.float64)
            w2c[:3, :3] = loader.qvec2rotmat(extrinsic.qvec)
            w2c[:3, 3] = np.asarray(extrinsic.tvec, dtype=np.float64)
            c2w = np.linalg.inv(w2c)
            blender_transform = c2w.copy()
            blender_transform[:3, 1:3] *= -1.0

            fx = float(params[0])
            if camera_angle_x is None:
                camera_angle_x = float(2.0 * np.arctan(width / (2.0 * fx)))

            frame = {
                "file_path": f"images/{target_stem}",
                "transform_matrix": blender_transform.tolist(),
            }
            if len(camera_entries) == 1:
                train_frames.append(frame)
                test_frames.append(frame)
            elif index % holdout_stride == 0:
                test_frames.append(frame)
            else:
                train_frames.append(frame)

        if not train_frames and test_frames:
            train_frames.append(test_frames[0])
        if not test_frames and train_frames:
            test_frames.append(train_frames[-1])

        payload_base = {"camera_angle_x": camera_angle_x or 0.0}
        (prepared_root / "transforms_train.json").write_text(
            json.dumps({**payload_base, "frames": train_frames}, indent=2),
            encoding="utf-8",
        )
        (prepared_root / "transforms_val.json").write_text(
            json.dumps({**payload_base, "frames": []}, indent=2),
            encoding="utf-8",
        )
        (prepared_root / "transforms_test.json").write_text(
            json.dumps({**payload_base, "frames": test_frames}, indent=2),
            encoding="utf-8",
        )

        return prepared_root

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

        lines = [
            f"expname = {exp_name}",
            f"basedir = {logs_dir.resolve()}",
            f"datadir = {dataset_root}",
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
