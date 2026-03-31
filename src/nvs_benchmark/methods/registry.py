"""Construtores de registros de métodos com integrações suportadas."""

from __future__ import annotations

import os

from nvs_benchmark.core import MethodRegistry

from .external.adapter import ExternalMethodAdapter
from .gs_dynamic.adapter import GSDynamicAdapter
from .gs_static.adapter import GSStaticAdapter
from .nerf_dynamic.adapter import NeRFDynamicAdapter
from .nerf_static.adapter import NeRFStaticAdapter


def _should_enable_external() -> bool:
    """Habilita adaptador externo quando NVS_ENABLE_EXTERNAL estiver definido."""
    return os.environ.get("NVS_ENABLE_EXTERNAL", "").lower() in {"1", "true", "yes"}


def build_registry_with_nerf_methods() -> MethodRegistry:
    """Cria MethodRegistry com adaptadores NeRF estático e dinâmico."""
    registry = MethodRegistry()
    registry.register(NeRFStaticAdapter())
    registry.register(NeRFDynamicAdapter())
    return registry


def build_registry_with_all_methods() -> MethodRegistry:
    """Cria MethodRegistry com os quatro métodos da V1.

    Quando NVS_ENABLE_EXTERNAL=1, também registra ExternalMethodAdapter.
    """
    registry = MethodRegistry()
    registry.register(NeRFStaticAdapter())
    registry.register(NeRFDynamicAdapter())
    registry.register(GSStaticAdapter())
    registry.register(GSDynamicAdapter())
    if _should_enable_external():
        registry.register(ExternalMethodAdapter())
    return registry
