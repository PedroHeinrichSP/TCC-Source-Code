"""Sistema de presets de treinamento para configurar iterações por método."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contracts import HardwareProfile

_DEFAULT_PRESETS_PATH = Path(__file__).resolve().parents[3] / "configs" / "training_presets.json"

PRESET_NAMES = ("smoke", "quick", "sweep", "preview", "standard", "full")


@dataclass(frozen=True)
class TrainingPreset:
    """Configuração de iterações para um método em um preset."""

    name: str
    label: str
    description: str
    params: dict[str, Any] = field(default_factory=dict)


def load_presets(path: str | Path | None = None) -> dict[str, dict[str, TrainingPreset]]:
    """Carrega presets do JSON e retorna {preset_name: {method_id: TrainingPreset}}.

    Args:
        path: caminho do JSON de presets. Se None, usa o padrão em configs/.

    Returns:
        Dicionário aninhado: preset_name -> method_id -> TrainingPreset.
    """
    presets_path = Path(path) if path else _DEFAULT_PRESETS_PATH
    if not presets_path.exists():
        return {}
    data = json.loads(presets_path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, TrainingPreset]] = {}

    for preset_name, preset_data in data.get("presets", {}).items():
        label = preset_data.get("label", preset_name)
        description = preset_data.get("description", "")
        methods: dict[str, TrainingPreset] = {}
        for method_id in ("nerf_static", "nerf_dynamic", "gs_static", "gs_dynamic"):
            if method_id in preset_data:
                methods[method_id] = TrainingPreset(
                    name=preset_name,
                    label=label,
                    description=description,
                    params=preset_data[method_id],
                )
        result[preset_name] = methods
    return result


def get_preset_for_method(
    method_id: str,
    preset_name: str,
    presets_path: str | Path | None = None,
) -> TrainingPreset | None:
    """Retorna o preset de iterações para um método específico.

    Args:
        method_id: ID do método (ex: nerf_static).
        preset_name: nome do preset (ex: quick, standard).
        presets_path: caminho alternativo para o JSON de presets.

    Returns:
        TrainingPreset ou None se não encontrado.
    """
    all_presets = load_presets(presets_path)
    preset_group = all_presets.get(preset_name)
    if not preset_group:
        return None
    return preset_group.get(method_id)


def resolve_iterations(
    *,
    method_id: str,
    preset_name: str | None = None,
    iterations: int | None = None,
    extra: dict[str, Any] | None = None,
    hardware_profile: HardwareProfile = HardwareProfile.ADAPTIVE,
    presets_path: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve parâmetros de iteração a partir de preset, override direto, ou extra.

    Prioridade: iterations direto > extra > preset > padrão (quick).

    Args:
        method_id: ID do método.
        preset_name: nome do preset (ex: quick, preview).
        iterations: override direto de número de iterações.
        extra: dict com configurações extras do RunConfig.
        presets_path: caminho alternativo para o JSON de presets.

    Returns:
        Dict com parâmetros de iteração resolvidos.
    """
    params: dict[str, Any] = {}

    effective_preset = preset_name or "quick"
    preset = get_preset_for_method(method_id, effective_preset, presets_path)
    if preset:
        params.update(preset.params)

    if extra:
        for key in ("N_iter", "iterations", "i_weights", "i_testset", "i_video", "i_print"):
            if key in extra:
                params[key] = extra[key]

    if iterations is not None:
        if method_id in ("nerf_static", "nerf_dynamic"):
            params["N_iter"] = iterations
            if "i_weights" not in params or params.get("i_weights", 0) > iterations:
                params["i_weights"] = iterations
            if "i_testset" not in params or params.get("i_testset", 0) > iterations:
                params["i_testset"] = iterations
        elif method_id in ("gs_static", "gs_dynamic"):
            params["iterations"] = iterations

    apply_adaptive = bool(extra.get("adaptive_preset", False)) if extra else False
    if apply_adaptive and iterations is None:
        params = _apply_conservative_scaling(
            method_id=method_id,
            params=params,
            hardware_profile=hardware_profile,
            detected_hardware=(extra or {}).get("detected_hardware", {}),
        )

    return params


def _apply_conservative_scaling(
    *,
    method_id: str,
    params: dict[str, Any],
    hardware_profile: HardwareProfile,
    detected_hardware: dict[str, Any],
) -> dict[str, Any]:
    factor = _resolve_scaling_factor(hardware_profile=hardware_profile, detected_hardware=detected_hardware)
    if factor >= 1.0:
        return params

    scaled = dict(params)
    if method_id in ("nerf_static", "nerf_dynamic"):
        if "N_iter" in scaled:
            scaled["N_iter"] = max(1, int(float(scaled["N_iter"]) * factor))
        if "i_weights" in scaled:
            scaled["i_weights"] = max(1, int(float(scaled["i_weights"]) * factor))
        if "i_testset" in scaled:
            scaled["i_testset"] = max(1, int(float(scaled["i_testset"]) * factor))
    elif method_id in ("gs_static", "gs_dynamic"):
        if "iterations" in scaled:
            scaled["iterations"] = max(1, int(float(scaled["iterations"]) * factor))
    return scaled


def _resolve_scaling_factor(*, hardware_profile: HardwareProfile, detected_hardware: dict[str, Any]) -> float:
    if hardware_profile == HardwareProfile.HIGH:
        return 1.0
    if hardware_profile == HardwareProfile.MEDIUM:
        return 0.75
    if hardware_profile == HardwareProfile.LOW:
        return 0.5

    has_gpu = bool(detected_hardware.get("has_gpu", False))
    vram = detected_hardware.get("vram_gb")
    if not has_gpu:
        return 0.5
    if isinstance(vram, (float, int)):
        if vram < 6.0:
            return 0.5
        if vram < 10.0:
            return 0.75
    return 1.0


def list_preset_summaries(presets_path: str | Path | None = None) -> list[dict[str, str]]:
    """Retorna lista de resumos de presets para exibição na UI.

    Returns:
        Lista de dicts com name, label, description.
    """
    presets_path_resolved = Path(presets_path) if presets_path else _DEFAULT_PRESETS_PATH
    if not presets_path_resolved.exists():
        return []
    data = json.loads(presets_path_resolved.read_text(encoding="utf-8"))
    summaries: list[dict[str, str]] = []
    for name, preset_data in data.get("presets", {}).items():
        summaries.append({
            "name": name,
            "label": preset_data.get("label", name),
            "description": preset_data.get("description", ""),
        })
    return summaries
