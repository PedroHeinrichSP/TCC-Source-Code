"""Utilitários para instalar datasets e repósitorios de métodos de um catálogo."""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InstallItem:
    """Entrada única do catálogo para conteúdo instalável."""

    item_id: str
    label: str
    path: str
    url: str
    command: str
    size_mb: float | None


@dataclass(frozen=True)
class InstallCatalog:
    """Catálogo de datasets e métodos."""

    datasets: list[InstallItem]
    methods: list[InstallItem]
    notes: list[str]


def _to_float(value: object | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_install_catalog(catalog_file: str | Path) -> InstallCatalog:
    """Carregar o catálogo de instalação a partir do arquivo JSON."""
    path = Path(catalog_file)
    if not path.exists():
        return InstallCatalog(datasets=[], methods=[], notes=[f"Catalog not found: {path}"])

    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        return InstallCatalog(
            datasets=[],
            methods=[],
            notes=[f"Invalid catalog JSON: {exc}"],
        )
    datasets = []
    methods = []
    for item in payload.get("datasets", []):
        datasets.append(
            InstallItem(
                item_id=str(item.get("id") or "dataset"),
                label=str(item.get("label") or item.get("id") or "dataset"),
                path=str(item.get("path") or ""),
                url=str(item.get("url") or ""),
                command=str(item.get("command") or ""),
                size_mb=_to_float(item.get("size_mb")),
            )
        )
    for item in payload.get("methods", []):
        methods.append(
            InstallItem(
                item_id=str(item.get("id") or "method"),
                label=str(item.get("label") or item.get("id") or "method"),
                path=str(item.get("path") or ""),
                url=str(item.get("url") or ""),
                command=str(item.get("command") or ""),
                size_mb=_to_float(item.get("size_mb")),
            )
        )
    notes = [str(note) for note in payload.get("notes", []) if note]
    return InstallCatalog(datasets=datasets, methods=methods, notes=notes)


def _is_installed(path_value: str) -> bool:
    if not path_value:
        return False
    return Path(path_value).exists()


def _suggest_command(item: InstallItem) -> str:
    if item.command:
        return item.command
    if item.url and item.path:
        return f"git clone {item.url} {item.path}"
    return ""


def install_items(
    *,
    catalog: InstallCatalog,
    only: str = "all",
    execute: bool = False,
) -> list[str]:
    """Instalar itens usando comandos fornecidos no catálogo.

    Retorna uma lista de mensagens descrevendo as ações tomadas.
    """
    messages: list[str] = []
    targets: list[InstallItem] = []
    if only in {"datasets", "all"}:
        targets.extend(catalog.datasets)
    if only in {"methods", "all"}:
        targets.extend(catalog.methods)

    for item in targets:
        if _is_installed(item.path):
            messages.append(f"[skip] {item.label}: already exists at {item.path}")
            continue
        command = _suggest_command(item)
        if not command:
            messages.append(f"[warn] {item.label}: no command or URL configured")
            continue
        messages.append(f"[plan] {item.label}: {command}")
        if execute:
            args = shlex.split(command)
            subprocess.run(args, check=True)
            messages.append(f"[ok] {item.label}: command executed")

    return messages


def install_item_by_id(
    *,
    catalog: InstallCatalog,
    item_id: str,
    execute: bool = False,
) -> list[str]:
    """Instalar um único item pelo seu ID."""
    messages: list[str] = []
    targets = catalog.datasets + catalog.methods
    matches = [item for item in targets if item.item_id == item_id]
    if not matches:
        return [f"[error] item not found: {item_id}"]

    item = matches[0]
    if _is_installed(item.path):
        return [f"[skip] {item.label}: already exists at {item.path}"]

    command = _suggest_command(item)
    if not command:
        return [f"[warn] {item.label}: no command or URL configured"]

    messages.append(f"[plan] {item.label}: {command}")
    if execute:
        args = shlex.split(command)
        subprocess.run(args, check=True)
        messages.append(f"[ok] {item.label}: command executed")
    return messages
