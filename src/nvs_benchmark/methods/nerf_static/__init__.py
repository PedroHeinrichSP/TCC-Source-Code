"""Integração do método NeRF estático."""

BASE_REPO = "https://github.com/bmild/nerf"

from .adapter import NeRFStaticAdapter

__all__ = ["BASE_REPO", "NeRFStaticAdapter"]
