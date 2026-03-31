"""Métodos: integrações com repositórios base e adaptadores externos."""

from .external.adapter import ExternalMethodAdapter
from .gs_dynamic.adapter import GSDynamicAdapter
from .gs_static.adapter import GSStaticAdapter
from .nerf_dynamic.adapter import NeRFDynamicAdapter
from .nerf_static.adapter import NeRFStaticAdapter
from .registry import build_registry_with_all_methods, build_registry_with_nerf_methods

__all__ = [
    "ExternalMethodAdapter",
    "GSDynamicAdapter",
    "GSStaticAdapter",
    "NeRFDynamicAdapter",
    "NeRFStaticAdapter",
    "build_registry_with_all_methods",
    "build_registry_with_nerf_methods",
]
