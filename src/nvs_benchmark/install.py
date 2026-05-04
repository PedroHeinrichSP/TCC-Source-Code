"""Utilitários para instalar datasets e repósitorios de métodos de um catálogo."""

from __future__ import annotations

import json
import platform
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


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


def _exec_cross_platform(command: str) -> subprocess.CompletedProcess:
    """Executa comando shell de forma cross-platform.
    
    No Windows: usa PowerShell
    No Linux/macOS: usa sh
    """
    os_name = platform.system()
    if os_name == "Windows":
        # Windows: use PowerShell natively
        return subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            text=True,
            capture_output=False,
        )
    else:
        # Linux/macOS: use sh
        return subprocess.run(
            ["sh", "-c", command],
            text=True,
            capture_output=False,
        )


def _ensure_git_submodules(path_value: str) -> Tuple[int, str]:
    """Tenta inicializar/atualizar submódulos Git no diretório fornecido.

    Retorna tupla (returncode, output). returncode 0 indica sucesso.
    """
    try:
        repo_path = Path(path_value)
        if not repo_path.exists():
            return 1, f"path not found: {repo_path}"

        gitmodules = repo_path / ".gitmodules"
        if not gitmodules.exists():
            return 0, "no .gitmodules present"

        proc = subprocess.run(
            ["git", "-C", str(repo_path), "submodule", "update", "--init", "--recursive"],
            text=True,
            capture_output=True,
        )
        out = ""
        if proc.stdout:
            out += proc.stdout
        if proc.stderr:
            out += "\n" + proc.stderr
        return proc.returncode, out
    except Exception as exc:
        return 1, str(exc)


def _install_dataset_wget_curl(dataset_name: str, url: str, target_path: str) -> int:
    """Baixa dataset usando wget ou curl (fallback para Linux/Colab).
    
    Retorna 0 se sucesso, non-zero se falha.
    """
    target = Path(target_path).parent
    target.mkdir(parents=True, exist_ok=True)
    
    # Tenta wget primeiro
    cmd_wget = f'wget -q "{url}" -O /tmp/{dataset_name}.zip && unzip -q /tmp/{dataset_name}.zip -d "{target}" && rm /tmp/{dataset_name}.zip'
    ret = subprocess.run(["sh", "-c", cmd_wget], capture_output=True).returncode
    if ret == 0:
        return 0
    
    # Fallback para curl
    cmd_curl = f'curl -s -L "{url}" -o /tmp/{dataset_name}.zip && unzip -q /tmp/{dataset_name}.zip -d "{target}" && rm /tmp/{dataset_name}.zip'
    return subprocess.run(["sh", "-c", cmd_curl], capture_output=True).returncode


def install_items(
    *,
    catalog: InstallCatalog,
    only: str = "all",
    execute: bool = False,
) -> list[str]:
    """Instalar itens usando comandos fornecidos no catálogo.

    Retorna uma lista de mensagens descrevendo as ações tomadas.
    Funciona cross-platform: PowerShell no Windows, sh no Linux/macOS/Colab.
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
            # Detecta se é comando PowerShell Windows e adapta para Linux
            os_name = platform.system()
            if ("powershell" in command.lower() or "invoke-webrequest" in command.lower()) and os_name != "Windows":
                # Comando PowerShell em ambiente Linux/Colab: usa wget/curl
                messages.append(f"[info] {item.label}: detectado comando Windows, usando wget/curl em {os_name}")
                ret = _install_dataset_wget_curl(item.item_id, item.url, item.path)
                if ret == 0:
                    messages.append(f"[ok] {item.label}: command executed (wget/curl)")
                else:
                    messages.append(f"[error] {item.label}: command failed (wget/curl returned {ret})")
            else:
                # Comando generico ou nativo: usa _exec_cross_platform
                try:
                    proc = _exec_cross_platform(command)
                    if proc.returncode == 0:
                        messages.append(f"[ok] {item.label}: command executed")
                        # After a successful clone/install, attempt to init/update git submodules if present
                        try:
                            sub_ret, sub_out = _ensure_git_submodules(item.path)
                            if sub_ret == 0:
                                if sub_out and "no .gitmodules" not in sub_out.lower():
                                    messages.append(f"[ok] {item.label}: submodules initialized")
                                else:
                                    messages.append(f"[info] {item.label}: no git submodules to init")
                            else:
                                messages.append(f"[warn] {item.label}: submodule init failed (exit {sub_ret})")
                                if sub_out:
                                    messages.append(f"[debug] submodule output: {sub_out}")
                        except Exception as exc:
                            messages.append(f"[warn] {item.label}: submodule init raised: {exc}")
                    else:
                        messages.append(f"[error] {item.label}: command failed (exit {proc.returncode})")
                except Exception as exc:
                    messages.append(f"[error] {item.label}: {exc}")
                # If the command failed but the path exists, still try to init submodules
                if Path(item.path).exists():
                    try:
                        sub_ret, sub_out = _ensure_git_submodules(item.path)
                        if sub_ret == 0:
                            messages.append(f"[ok] {item.label}: submodules initialized (post-failure)")
                        else:
                            messages.append(f"[warn] {item.label}: submodule init failed after failure (exit {sub_ret})")
                            if sub_out:
                                messages.append(f"[debug] submodule output: {sub_out}")
                    except Exception as exc:
                        messages.append(f"[warn] {item.label}: submodule init post-failure raised: {exc}")

    return messages


def install_item_by_id(
    *,
    catalog: InstallCatalog,
    item_id: str,
    execute: bool = False,
) -> list[str]:
    """Instalar um único item pelo seu ID (cross-platform)."""
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
        # Mesma lógica cross-platform que install_items
        os_name = platform.system()
        if ("powershell" in command.lower() or "invoke-webrequest" in command.lower()) and os_name != "Windows":
            messages.append(f"[info] {item.label}: detectado comando Windows, usando wget/curl em {os_name}")
            ret = _install_dataset_wget_curl(item.item_id, item.url, item.path)
            if ret == 0:
                messages.append(f"[ok] {item.label}: command executed (wget/curl)")
            else:
                messages.append(f"[error] {item.label}: command failed (wget/curl returned {ret})")
        else:
            try:
                proc = _exec_cross_platform(command)
                if proc.returncode == 0:
                    messages.append(f"[ok] {item.label}: command executed")
                    # After successful install, try to initialize git submodules if any
                    try:
                        sub_ret, sub_out = _ensure_git_submodules(item.path)
                        if sub_ret == 0:
                            if sub_out and "no .gitmodules" not in sub_out.lower():
                                messages.append(f"[ok] {item.label}: submodules initialized")
                            else:
                                messages.append(f"[info] {item.label}: no git submodules to init")
                        else:
                            messages.append(f"[warn] {item.label}: submodule init failed (exit {sub_ret})")
                            if sub_out:
                                messages.append(f"[debug] submodule output: {sub_out}")
                    except Exception as exc:
                        messages.append(f"[warn] {item.label}: submodule init raised: {exc}")
                else:
                    messages.append(f"[error] {item.label}: command failed (exit {proc.returncode})")
            except Exception as exc:
                messages.append(f"[error] {item.label}: {exc}")
            # If the command failed but produced the target path, still try submodule init
        if Path(item.path).exists():
            try:
                sub_ret, sub_out = _ensure_git_submodules(item.path)
                if sub_ret == 0:
                    messages.append(f"[ok] {item.label}: submodules initialized (post-failure)")
                else:
                    messages.append(f"[warn] {item.label}: submodule init failed after failure (exit {sub_ret})")
                    if sub_out:
                        messages.append(f"[debug] submodule output: {sub_out}")
            except Exception as exc:
                messages.append(f"[warn] {item.label}: submodule init post-failure raised: {exc}")
    return messages
