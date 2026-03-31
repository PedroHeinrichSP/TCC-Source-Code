"""Logging estruturado e metadados de execução para reprodutibilidade."""

from __future__ import annotations

import json
import platform
import sys
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


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


@dataclass(frozen=True)
class RunPaths:
    """Caminhos dos arquivos de contexto e eventos de uma execução."""

    run_dir: Path
    context_file: Path
    events_file: Path


class RunLogger:
    """Logger de execução com contexto de ambiente e trilha de eventos JSONL."""

    def __init__(
        self,
        *,
        command: str,
        parameters: dict | None = None,
        log_dir: str | Path = "./logs",
        run_id: str | None = None,
        dependencies: list[str] | None = None,
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

        payload = {
            "run_id": self._run_id,
            "created_at": _utc_now(),
            "command": command,
            "parameters": parameters or {},
            "python": {
                "version": sys.version,
                "executable": sys.executable,
            },
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
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
        row = {
            "timestamp": _utc_now(),
            "event": event_type,
            "payload": payload or {},
        }
        with self._paths.events_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def finish(self, status: str, payload: dict | None = None) -> None:
        self.event("run_finished", {"status": status, **(payload or {})})
