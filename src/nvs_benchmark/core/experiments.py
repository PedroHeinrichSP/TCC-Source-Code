"""Experiment history manager persisted in artifacts/experiments_history.json."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentRecord:
    """Canonical experiment record."""

    run_id: str
    timestamp: str
    command: str
    status: str
    method: str
    dataset: str
    preset: str | None
    config_digest: str
    checkpoint_path: str | None
    rendered_dir: str | None
    metrics_path: str | None
    report_paths: list[str]
    metrics_summary: dict[str, float] | None
    metadata: dict[str, Any]


class ExperimentManager:
    """Append-only JSON history manager for benchmark runs."""

    schema_version = 1

    def __init__(self, *, output_dir: str | Path = "./artifacts") -> None:
        self._history_file = Path(output_dir) / "experiments_history.json"

    @property
    def history_file(self) -> Path:
        return self._history_file

    def record(self, item: ExperimentRecord) -> None:
        payload = self._load()
        payload["experiments"].append(item.__dict__)
        self._save(payload)

    def list(
        self,
        *,
        method: str | None = None,
        dataset: str | None = None,
        status: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        payload = self._load()
        rows = payload.get("experiments", [])
        filtered = []
        for row in rows:
            if method and row.get("method") != method:
                continue
            if dataset and row.get("dataset") != dataset:
                continue
            if status and row.get("status") != status:
                continue
            filtered.append(row)

        if limit is not None and limit > 0:
            return filtered[-limit:]
        return filtered

    def _load(self) -> dict[str, Any]:
        if not self._history_file.exists():
            payload = {
                "schema_version": self.schema_version,
                "created_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
                "experiments": [],
            }
            self._save(payload)
            return payload

        payload = json.loads(self._history_file.read_text(encoding="utf-8-sig"))
        if payload.get("schema_version") != self.schema_version:
            payload = self._migrate(payload)
            self._save(payload)
        payload.setdefault("experiments", [])
        return payload

    def _save(self, payload: dict[str, Any]) -> None:
        payload["schema_version"] = self.schema_version
        self._history_file.parent.mkdir(parents=True, exist_ok=True)
        temp = self._history_file.with_suffix(self._history_file.suffix + ".tmp")
        temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, self._history_file)

    def _migrate(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload["schema_version"] = self.schema_version
        payload.setdefault("experiments", [])
        return payload
