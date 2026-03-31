"""Wrapper para executar run_dnerf.py com fallback robusto de torchsearchsorted.

Nao altera codigo em third_party. Este modulo injeta um shim em tempo de
execucao quando o modulo torchsearchsorted nao expoe o simbolo searchsorted.
"""

from __future__ import annotations

import argparse
import runpy
import sys
import types
from pathlib import Path


def _install_torchsearchsorted_shim() -> None:
    """Garante que `from torchsearchsorted import searchsorted` funcione."""
    try:
        import torchsearchsorted as tss  # type: ignore

        if hasattr(tss, "searchsorted"):
            return
    except Exception:
        pass

    import torch

    shim = types.ModuleType("torchsearchsorted")

    def _searchsorted(a, v, side: str = "left", out=None):
        right = side.lower() == "right"
        if out is None:
            return torch.searchsorted(a, v, right=right)
        return torch.searchsorted(a, v, right=right, out=out)

    shim.searchsorted = _searchsorted  # type: ignore[attr-defined]
    sys.modules["torchsearchsorted"] = shim


def _install_torch_cuda_fallback() -> None:
    """Permite que set_default_tensor_type aceiteur tipos CUDA quando nao disponivel."""
    import torch

    _original_set_default_tensor_type = torch.set_default_tensor_type

    def _set_default_tensor_type_safe(type_name: str) -> None:
        """Wrapper que ignora tipos CUDA se nao disponivel, fall back para CPU."""
        if not torch.cuda.is_available() and "cuda" in str(type_name).lower():
            cpu_type = str(type_name).lower().replace("cuda.", "").replace(".cuda", "")
            cpu_type = f"torch.{cpu_type}" if not cpu_type.startswith("torch.") else cpu_type
            try:
                _original_set_default_tensor_type(cpu_type)
                return
            except Exception:
                pass

        try:
            _original_set_default_tensor_type(type_name)
        except TypeError as e:
            if "not available" in str(e):
                pass
            else:
                raise

    torch.set_default_tensor_type = _set_default_tensor_type_safe  # type: ignore[attr-defined]


def main(argv: list[str] | None = None) -> int:
    """Executa run_dnerf.py preservando os argumentos originais."""
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--run-script", required=True, help="Caminho para run_dnerf.py")
    args, passthrough = parser.parse_known_args(argv)

    run_script = Path(args.run_script).resolve()
    if not run_script.exists():
        raise FileNotFoundError(f"Arquivo run_dnerf.py nao encontrado: {run_script}")

    _install_torchsearchsorted_shim()
    _install_torch_cuda_fallback()

    sys.argv = [str(run_script), *passthrough]
    runpy.run_path(str(run_script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
