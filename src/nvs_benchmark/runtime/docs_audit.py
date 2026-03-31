"""Auditoria automatizada de docstrings públicas no código-fonte."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MissingDocItem:
    """Representa um símbolo público sem docstring."""

    file: str
    symbol: str


def audit_docstrings(source_root: str | Path) -> list[MissingDocItem]:
    """Retorna símbolos públicos sem docstring em um diretório-fonte Python."""
    root = Path(source_root)
    missing: list[MissingDocItem] = []

    for path in sorted(root.rglob("*.py")):
        # `utf-8-sig` evita SyntaxError em arquivos com BOM (U+FEFF).
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))

        if ast.get_docstring(tree) is None:
            missing.append(MissingDocItem(file=str(path), symbol="module"))

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name.startswith("_"):
                    continue
                if ast.get_docstring(node) is None:
                    missing.append(MissingDocItem(file=str(path), symbol=node.name))

    return missing


def save_doc_audit_report(missing: list[MissingDocItem], output_file: str | Path) -> None:
    """Salva relatório JSON de auditoria de docstrings."""
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "missing_count": len(missing),
        "missing": [{"file": item.file, "symbol": item.symbol} for item in missing],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
