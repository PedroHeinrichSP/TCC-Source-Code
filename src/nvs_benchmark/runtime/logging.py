"""Logging estruturado e metadados de execução para reprodutibilidade.

Fornece:
- Contexto de ambiente (OS, Python, GPU, dependências)
- Trilha de eventos JSONL com timestamps e durações
- Context manager para cronometrar etapas (log_step)
- Utilitário log_tail() para depuração rápida
"""

from __future__ import annotations

import json
import platform
import socket
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Generator


DEFAULT_DEPENDENCIES = [
    "numpy",
    "pandas",
    "torch",
    "torchvision",
    "scikit-image",
    "matplotlib",
    "weasyprint",
]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _utc_now_ms() -> str:
    """Timestamp com precisão de milissegundos para eventos internos."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _safe_dependency_versions(packages: list[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    try:
        from importlib.metadata import PackageNotFoundError, version
    except Exception:
        return versions

    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
        except Exception:
            versions[package] = "unknown"
    return versions


def _collect_gpu_info() -> dict:
    """Coleta informações de GPU disponível (CUDA/NVIDIA) sem dependência obrigatória."""
    gpu_info: dict = {"available": False, "devices": []}
    try:
        import torch

        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["cuda_version"] = torch.version.cuda or "unknown"
            gpu_info["device_count"] = torch.cuda.device_count()
            gpu_info["devices"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "vram_gb": round(torch.cuda.get_device_properties(i).total_memory / 1e9, 2),
                }
                for i in range(torch.cuda.device_count())
            ]
    except Exception:
        pass
    return gpu_info


@dataclass(frozen=True)
class RunPaths:
    """Caminhos dos arquivos de contexto e eventos de uma execução."""

    run_dir: Path
    context_file: Path
    events_file: Path


class RunLogger:
    """Logger de execução com contexto de ambiente e trilha de eventos JSONL.

    Recursos:
    - Contexto completo (OS, Python, GPU, hostname, dependências)
    - Método event() para registrar eventos atomicamente em JSONL
    - context_manager log_step() para cronometrar etapas automaticamente
    - log_tail() para inspecionar os últimos eventos rapidamente
    """

    def __init__(
        self,
        *,
        command: str,
        parameters: dict | None = None,
        log_dir: str | Path = "./logs",
        run_id: str | None = None,
        dependencies: list[str] | None = None,
        collect_gpu: bool = True,
    ) -> None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        normalized_command = command.replace(" ", "_").lower()
        resolved_run_id = run_id or f"{timestamp}-{normalized_command}-{uuid.uuid4().hex[:8]}"

        root = Path(log_dir)
        run_dir = root / "runs" / resolved_run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        self._run_id = resolved_run_id
        self._paths = RunPaths(
            run_dir=run_dir,
            context_file=run_dir / "run_context.json",
            events_file=run_dir / "events.jsonl",
        )

        gpu_info = _collect_gpu_info() if collect_gpu else {"available": False, "note": "skipped"}

        payload = {
            "run_id": self._run_id,
            "created_at": _utc_now(),
            "command": command,
            "parameters": parameters or {},
            "host": {
                "hostname": socket.gethostname(),
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            "python": {
                "version": sys.version,
                "executable": sys.executable,
            },
            "gpu": gpu_info,
            "dependencies": _safe_dependency_versions(dependencies or DEFAULT_DEPENDENCIES),
        }
        self._paths.context_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def run_dir(self) -> Path:
        return self._paths.run_dir

    @property
    def events_file(self) -> Path:
        return self._paths.events_file

    def event(self, event_type: str, payload: dict | None = None) -> None:
        """Registra um evento atômico no arquivo JSONL de trilha."""
        row = {
            "timestamp": _utc_now_ms(),
            "event": event_type,
            "payload": payload or {},
        }
        with self._paths.events_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    @contextmanager
    def log_step(self, step_name: str, payload: dict | None = None) -> Generator[None, None, None]:
        """Context manager que registra início/fim de uma etapa com duração.

        Uso::

            with logger.log_step("train"):
                model.train()
        """
        import time

        self.event(f"{step_name}.started", payload)
        t0 = time.perf_counter()
        status = "success"
        try:
            yield
        except Exception as exc:
            status = "failed"
            self.event(f"{step_name}.failed", {"error": str(exc), "type": type(exc).__name__})
            raise
        finally:
            elapsed = time.perf_counter() - t0
            self.event(
                f"{step_name}.finished",
                {"status": status, "elapsed_seconds": round(elapsed, 3)},
            )

    def finish(self, status: str, payload: dict | None = None) -> None:
        """Registra evento de conclusão da execução."""
        self.event("run_finished", {"status": status, **(payload or {})})

    def log_tail(self, n: int = 10) -> list[dict]:
        """Retorna os últimos *n* eventos do arquivo JSONL (para debug rápido).

        Útil para inspecionar o estado da execução sem abrir o arquivo completo.
        """
        if not self._paths.events_file.exists():
            return []
        lines = self._paths.events_file.read_text(encoding="utf-8").splitlines()
        tail_lines = lines[-n:] if len(lines) > n else lines
        events = []
        for line in tail_lines:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return events

    def print_tail(self, n: int = 10) -> None:
        """Imprime os últimos *n* eventos formatados no console."""
        events = self.log_tail(n)
        if not events:
            print(f"[{self._run_id}] Nenhum evento registrado ainda.")
            return
        print(f"[{self._run_id}] Ultimos {len(events)} eventos:")
        for ev in events:
            ts = ev.get("timestamp", "?")
            name = ev.get("event", "?")
            payload_items = ev.get("payload", {})
            # Mostrar elapsed se disponível
            elapsed = payload_items.get("elapsed_seconds")
            suffix = f" ({elapsed:.3f}s)" if elapsed is not None else ""
            print(f"  {ts}  {name}{suffix}")

