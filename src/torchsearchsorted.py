"""Compatibility shim for D-NeRF expecting torchsearchsorted.searchsorted.

Some environments provide an incompatible torchsearchsorted package (or none at all).
D-NeRF imports `searchsorted` from this module, so we proxy to `torch.searchsorted`.
"""

from __future__ import annotations

import torch


def searchsorted(
    sorted_sequence: torch.Tensor,
    values: torch.Tensor,
    out_int32: bool = False,
    right: bool = False,
    side: str | None = None,
    out: torch.Tensor | None = None,
    sorter: torch.Tensor | None = None,
) -> torch.Tensor:
    """Drop-in replacement for torchsearchsorted.searchsorted.

    Supports legacy arguments used by NeRF/D-NeRF code while delegating to
    torch.searchsorted available in modern PyTorch versions.
    """

    return torch.searchsorted(
        sorted_sequence,
        values,
        out_int32=out_int32,
        right=right,
        side=side,
        out=out,
        sorter=sorter,
    )
