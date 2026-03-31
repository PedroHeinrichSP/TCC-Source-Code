"""Registro central de métodos para descoberta e recuperação por ID."""

from __future__ import annotations

from dataclasses import dataclass, field

from .contracts import BenchmarkMethod


@dataclass
class MethodRegistry:
    """Coleção tipada de métodos de benchmark registrados em memória."""

    _methods: dict[str, BenchmarkMethod] = field(default_factory=dict)

    def register(self, method: BenchmarkMethod) -> None:
        """Registra um método único no catálogo."""
        if method.method_id in self._methods:
            raise ValueError(f"Método já registrado: {method.method_id}")
        self._methods[method.method_id] = method

    def get(self, method_id: str) -> BenchmarkMethod:
        """Retorna um método pelo ID, com erro amigável se ausente."""
        if method_id not in self._methods:
            known = ", ".join(sorted(self._methods)) or "nenhum"
            raise KeyError(f"Método não encontrado: {method_id}. Registrados: {known}")
        return self._methods[method_id]

    def list_ids(self) -> list[str]:
        """Lista IDs de métodos registrados em ordem determinística."""
        return sorted(self._methods.keys())
