"""JSON-based cache registry for benchmark artifacts and computed metrics."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _stable_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CacheEntry:
    """Single cached artifact entry."""

    key: str
    artifact_type: str
    path: str | None = None
    payload: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    checksum: str | None = None
    size_bytes: int = 0
    created_at: str = field(default_factory=_utc_now)
    accessed_at: str = field(default_factory=_utc_now)


class CacheRegistry:
    """Lightweight JSON cache registry with optional LRU eviction."""

    schema_version = 1

    def __init__(self, *, output_dir: str | Path = "./artifacts") -> None:
        self._root = Path(output_dir)
        self._registry_file = self._root / "cache_registry.json"
        self._stats_file = self._root / "cache_stats.json"

    @property
    def registry_file(self) -> Path:
        return self._registry_file

    def get(self, key: str) -> CacheEntry | None:
        data = self._load_registry()
        raw = data["entries"].get(key)
        if raw is None:
            self.record_miss(key)
            return None

        entry = CacheEntry(**raw)
        if not self.validate(entry):
            self.delete(key)
            self.record_miss(key)
            return None

        self.touch(key)
        self.record_hit(key)
        return CacheEntry(**self._load_registry()["entries"][key])

    def find_first(self, *, artifact_type: str, filters: dict[str, Any] | None = None) -> CacheEntry | None:
        filters = filters or {}
        data = self._load_registry()
        for raw in data["entries"].values():
            entry = CacheEntry(**raw)
            if entry.artifact_type != artifact_type:
                continue
            if any(entry.metadata.get(name) != value for name, value in filters.items()):
                continue
            if not self.validate(entry):
                continue
            self.touch(entry.key)
            return CacheEntry(**self._load_registry()["entries"][entry.key])
        return None

    def put(self, entry: CacheEntry) -> None:
        data = self._load_registry()
        now = _utc_now()

        normalized = asdict(entry)
        normalized["created_at"] = normalized.get("created_at") or now
        normalized["accessed_at"] = now

        if normalized.get("path"):
            path = Path(str(normalized["path"]))
            if path.exists() and path.is_file():
                normalized["size_bytes"] = path.stat().st_size
                normalized["checksum"] = normalized.get("checksum") or self._file_checksum(path)
            elif path.exists() and path.is_dir():
                normalized["size_bytes"] = self._directory_size(path)

        if normalized.get("payload") and not normalized.get("checksum"):
            normalized["checksum"] = _stable_digest(normalized["payload"])

        data["entries"][entry.key] = normalized
        self._save_registry(data)

    def delete(self, key: str) -> None:
        data = self._load_registry()
        if key in data["entries"]:
            del data["entries"][key]
            self._save_registry(data)

    def touch(self, key: str) -> None:
        data = self._load_registry()
        raw = data["entries"].get(key)
        if raw is None:
            return
        raw["accessed_at"] = _utc_now()
        data["entries"][key] = raw
        self._save_registry(data)

    def validate(self, entry: CacheEntry) -> bool:
        if entry.path:
            path = Path(entry.path)
            if not path.exists():
                return False
            if entry.checksum and path.is_file():
                return self._file_checksum(path) == entry.checksum
            return True

        if entry.payload is not None and entry.checksum:
            return _stable_digest(entry.payload) == entry.checksum

        return True

    def evict_lru(self, *, max_size_gb: float) -> int:
        max_bytes = int(max_size_gb * (1024**3))
        data = self._load_registry()
        entries = [CacheEntry(**raw) for raw in data["entries"].values()]
        total_size = sum(max(entry.size_bytes, 0) for entry in entries)
        if total_size <= max_bytes:
            return 0

        evicted = 0
        sorted_entries = sorted(entries, key=lambda item: item.accessed_at)
        for entry in sorted_entries:
            if total_size <= max_bytes:
                break
            total_size -= max(entry.size_bytes, 0)
            data["entries"].pop(entry.key, None)
            evicted += 1

        self._save_registry(data)
        self._update_stats({"evictions": evicted, "last_eviction_at": _utc_now()})
        return evicted

    def record_hit(self, key: str) -> None:
        stats = self._load_stats()
        stats["hits"] = int(stats.get("hits", 0)) + 1
        stats["last_hit_key"] = key
        stats["last_hit_at"] = _utc_now()
        self._save_stats(stats)

    def record_miss(self, key: str) -> None:
        stats = self._load_stats()
        stats["misses"] = int(stats.get("misses", 0)) + 1
        stats["last_miss_key"] = key
        stats["last_miss_at"] = _utc_now()
        self._save_stats(stats)

    def _load_registry(self) -> dict[str, Any]:
        if not self._registry_file.exists():
            data = {
                "schema_version": self.schema_version,
                "created_at": _utc_now(),
                "entries": {},
            }
            self._save_registry(data)
            return data

        payload = json.loads(self._registry_file.read_text(encoding="utf-8-sig"))
        if payload.get("schema_version") != self.schema_version:
            payload = self._migrate_registry(payload)
            self._save_registry(payload)
        payload.setdefault("entries", {})
        return payload

    def _save_registry(self, payload: dict[str, Any]) -> None:
        payload["schema_version"] = self.schema_version
        self._write_json_atomic(self._registry_file, payload)

    def _load_stats(self) -> dict[str, Any]:
        if not self._stats_file.exists():
            stats = {
                "schema_version": self.schema_version,
                "hits": 0,
                "misses": 0,
                "evictions": 0,
                "created_at": _utc_now(),
            }
            self._save_stats(stats)
            return stats
        payload = json.loads(self._stats_file.read_text(encoding="utf-8-sig"))
        payload.setdefault("hits", 0)
        payload.setdefault("misses", 0)
        payload.setdefault("evictions", 0)
        return payload

    def _save_stats(self, payload: dict[str, Any]) -> None:
        payload["schema_version"] = self.schema_version
        self._write_json_atomic(self._stats_file, payload)

    def _migrate_registry(self, payload: dict[str, Any]) -> dict[str, Any]:
        payload["schema_version"] = self.schema_version
        payload.setdefault("entries", {})
        return payload

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, path)

    @staticmethod
    def _directory_size(path: Path) -> int:
        return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())

    @staticmethod
    def _file_checksum(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()
