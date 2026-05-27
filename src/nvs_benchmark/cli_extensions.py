"""Extensões e utilitários para a interface CLI.

Fornece:
- Validações robustas
- Coleta de informações do sistema
- Feedback ao usuário
- Funcionalidades auxiliares
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import json
import sys
from dataclasses import dataclass

from nvs_benchmark.data.validation import validate_dataset_integrity
from nvs_benchmark.data.registry import _is_tanks_and_temples_scene_root
from nvs_benchmark.runtime import detect_hardware_snapshot


_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".PNG", ".JPG", ".JPEG", ".WEBP")


@dataclass
class ValidationResult:
    """Resultado de uma validação."""
    is_valid: bool
    message: str
    details: Optional[Dict[str, Any]] = None


def _normalize_matrix_key(key: str) -> tuple[str, str] | None:
    """Normaliza chave de snapshot no formato dataset|method."""
    if "|" not in key:
        return None
    dataset, method = key.split("|", 1)
    dataset = dataset.strip()
    method = method.strip()
    if not dataset or not method:
        return None
    return dataset, method


def validate_matrix_completeness(
    snapshot_payload: dict[str, Any] | str | Path,
    *,
    expected_combos: list[str],
    strict: bool = True,
) -> ValidationResult:
    """Valida se um snapshot consolidado cobre todas as combinacoes esperadas."""
    try:
        if isinstance(snapshot_payload, (str, Path)):
            payload = json.loads(Path(snapshot_payload).read_text(encoding="utf-8-sig"))
        else:
            payload = snapshot_payload
    except Exception as exc:
        return ValidationResult(
            is_valid=False,
            message=f"Snapshot de matriz invalido: {exc}",
        )

    if not isinstance(payload, dict):
        return ValidationResult(
            is_valid=False,
            message="Snapshot de matriz invalido: esperado objeto JSON por combinacao",
        )

    present = set()
    invalid_keys: list[str] = []
    for raw_key in payload.keys():
        normalized = _normalize_matrix_key(str(raw_key))
        if normalized is None:
            invalid_keys.append(str(raw_key))
            continue
        present.add(f"{normalized[0]}|{normalized[1]}")

    expected = [item.strip() for item in expected_combos if item and item.strip()]
    missing = [combo for combo in expected if combo not in present]
    unexpected = sorted(item for item in present if item not in expected)

    is_valid = not invalid_keys and not missing and (not strict or not unexpected)
    details = {
        "expected_count": len(expected),
        "present_count": len(present),
        "missing_combos": missing,
        "unexpected_combos": unexpected,
        "invalid_keys": invalid_keys,
    }

    if is_valid:
        return ValidationResult(
            is_valid=True,
            message="Snapshot consolidado cobre todas as combinacoes esperadas",
            details=details,
        )

    problems = []
    if invalid_keys:
        problems.append(f"chaves_invalidas={invalid_keys}")
    if missing:
        problems.append(f"faltando={missing}")
    if strict and unexpected:
        problems.append(f"inesperadas={unexpected}")

    return ValidationResult(
        is_valid=False,
        message="Snapshot consolidado incompleto: " + "; ".join(problems),
        details=details,
    )


def _collect_image_files(root_path: Path) -> list[Path]:
    try:
        image_files = [
            candidate
            for candidate in root_path.rglob("*")
            if candidate.is_file() and candidate.suffix in _IMAGE_SUFFIXES
        ]
    except OSError:
        return []
    return sorted(image_files, key=lambda candidate: (len(candidate.parts), str(candidate).lower()))


def _validate_direct_image_dataset_integrity(
    root: str | Path,
    *,
    full_scan: bool = False,
    sample_size: int = 16,
) -> ValidationResult:
    root_path = Path(root)
    image_files = _collect_image_files(root_path)
    if not image_files:
        return ValidationResult(
            is_valid=False,
            message=f"Dataset sem imagens validas em {root}",
            details={"expected_path": str(root_path.absolute())},
        )

    checked_files = image_files if full_scan else image_files[: max(sample_size, 1)]
    corrupted_images = [
        str(candidate.relative_to(root_path))
        for candidate in checked_files
        if candidate.stat().st_size <= 0
    ]

    details = {
        "checked_frames": len(checked_files),
        "total_frames": len(image_files),
        "warnings": [],
        "errors": [],
        "missing_images": [],
        "corrupted_images": corrupted_images,
    }
    if corrupted_images:
        details["errors"] = [f"{len(corrupted_images)} imagem(ns) corrompida(s) detectada(s)"]
        return ValidationResult(
            is_valid=False,
            message="Falha na validacao de integridade do dataset",
            details=details,
        )

    return ValidationResult(
        is_valid=True,
        message=(
            f"Integridade do dataset valida (imagens verificadas: "
            f"{len(checked_files)}/{len(image_files)})"
        ),
        details=details,
    )


def validate_dataset_path(dataset_name: str, root: str) -> ValidationResult:
    """Valida se um caminho de dataset existe e contém arquivos esperados."""
    root_path = Path(root)
    
    if not root_path.exists():
        return ValidationResult(
            is_valid=False,
            message=f"Caminho do dataset não existe: {root}",
            details={"expected_path": str(root_path.absolute())}
        )
    
    if not root_path.is_dir():
        return ValidationResult(
            is_valid=False,
            message=f"Caminho do dataset não é um diretório: {root}",
        )
    
    # Verificações específicas por tipo de dataset
    if dataset_name == "blender_synthetic":
        required_dirs = ["train", "test", "val"]
        required_files = ["transforms_train.json"]
        
        missing_dirs = [d for d in required_dirs if not (root_path / d).exists()]
        missing_files = [f for f in required_files if not (root_path / f).exists()]
        
        if missing_dirs or missing_files:
            return ValidationResult(
                is_valid=False,
                message=f"Dataset {dataset_name} incompleto em {root}",
                details={
                    "missing_directories": missing_dirs,
                    "missing_files": missing_files,
                }
            )

    if dataset_name == "mipnerf360":
        has_image_files = bool(_collect_image_files(root_path))
        has_sparse = (root_path / "sparse" / "0").exists() and (root_path / "sparse" / "0").is_dir()
        has_numpy = (root_path / "poses_bounds.npy").exists()

        if not has_image_files and not has_sparse and not has_numpy:
            return ValidationResult(
                is_valid=False,
                message=(
                    f"Dataset {dataset_name} incompleto em {root}: esperado imagens validas, sparse/0 "
                    "ou poses_bounds.npy na base."
                ),
            )

    if dataset_name == "tanks_and_temples":
        if not _is_tanks_and_temples_scene_root(root_path):
            return ValidationResult(
                is_valid=False,
                message=(
                    f"Dataset {dataset_name} invalido em {root}: use uma cena extraida em image_sets/<cena> "
                    "ou videos/<cena> com imagens, images/, poses_bounds.npy ou transforms_*.json."
                ),
            )
    
    return ValidationResult(
        is_valid=True,
        message=f"Dataset {dataset_name} válido em {root}",
        details={"path": str(root_path.absolute())}
    )


def validate_dataset_integrity_preflight(
    *,
    dataset_name: str,
    root: str,
    split: str,
    full_scan: bool = False,
    sample_size: int = 16,
) -> ValidationResult:
    """Executa validação estrutural e de integridade de imagens/transforms."""
    path_result = validate_dataset_path(dataset_name, root)
    if not path_result.is_valid:
        return path_result

    root_path = Path(root)
    transforms_path = root_path / f"transforms_{split}.json"
    if dataset_name in {"mipnerf360", "tanks_and_temples"} and not transforms_path.exists():
        return _validate_direct_image_dataset_integrity(
            root_path,
            full_scan=full_scan,
            sample_size=sample_size,
        )

    report = validate_dataset_integrity(
        root=root,
        split=split,
        full_scan=full_scan,
        sample_size=sample_size,
    )
    details = {
        "split": split,
        "checked_frames": report.checked_frames,
        "total_frames": report.total_frames,
        "warnings": report.warnings,
        "errors": report.errors,
        "missing_images": report.missing_images,
        "corrupted_images": report.corrupted_images,
    }
    if report.is_valid:
        return ValidationResult(
            is_valid=True,
            message=(
                f"Integridade do dataset valida (frames verificados: "
                f"{report.checked_frames}/{report.total_frames})"
            ),
            details=details,
        )

    return ValidationResult(
        is_valid=False,
        message="Falha na validacao de integridade do dataset",
        details=details,
    )


def validate_output_directory(output_dir: str) -> ValidationResult:
    """Valida se o diretório de saída é acessível."""
    output_path = Path(output_dir)
    
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Tenta criar arquivo de teste
        test_file = output_path / ".write_test"
        test_file.write_text("test")
        test_file.unlink()
        
        return ValidationResult(
            is_valid=True,
            message=f"Diretório de saída acessível: {output_dir}",
        )
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            message=f"Não é possível escrever em {output_dir}: {str(e)}",
        )


def validate_method_available(method_id: str) -> ValidationResult:
    """Valida se um método está disponível."""
    from nvs_benchmark.methods import build_registry_with_all_methods
    
    try:
        registry = build_registry_with_all_methods()
        available_methods = registry.list_ids()
        
        if method_id not in available_methods:
            return ValidationResult(
                is_valid=False,
                message=f"Método não encontrado: {method_id}",
                details={"available_methods": available_methods}
            )
        
        method = registry.get(method_id)
        return ValidationResult(
            is_valid=True,
            message=f"Método disponível: {method_id}",
            details={
                "method_id": method.method_id,
                "display_name": method.display_name,
                "supports_train": method.capabilities.supports_train,
                "supports_inference": method.capabilities.supports_inference,
                "supports_dynamic": method.capabilities.supports_dynamic_scene,
            }
        )
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            message=f"Erro ao validar método {method_id}: {str(e)}",
        )


def validate_snapshot_file(snapshot_path: str, must_exist: bool = False) -> ValidationResult:
    """Valida um arquivo de snapshot de métricas."""
    path = Path(snapshot_path)
    
    if must_exist and not path.exists():
        return ValidationResult(
            is_valid=False,
            message=f"Arquivo de snapshot não encontrado: {snapshot_path}",
        )
    
    if path.exists():
        try:
            with open(path, encoding="utf-8-sig") as f:
                data = json.load(f)
            
            if not isinstance(data, (dict, list)):
                return ValidationResult(
                    is_valid=False,
                    message=f"Arquivo de snapshot tem formato inválido (esperado JSON dict ou list)",
                )
            
            return ValidationResult(
                is_valid=True,
                message=f"Arquivo de snapshot válido: {snapshot_path}",
                details={"size_entries": len(data) if isinstance(data, dict) else len(data)}
            )
        except json.JSONDecodeError as e:
            return ValidationResult(
                is_valid=False,
                message=f"Arquivo de snapshot tem JSON inválido: {str(e)}",
            )
    else:
        # Arquivo não existe ainda, verificar se o diretório é acessível
        parent_dir = path.parent
        if not parent_dir.exists():
            try:
                parent_dir.mkdir(parents=True, exist_ok=True)
                return ValidationResult(
                    is_valid=True,
                    message=f"Snap shot será criado em: {snapshot_path}",
                )
            except Exception as e:
                return ValidationResult(
                    is_valid=False,
                    message=f"Não é possível criar diretório para snapshot: {str(e)}",
                )


def estimate_execution_time(
    method_id: str,
    preset: Optional[str] = None,
    iterations: Optional[int] = None,
    adaptive_preset: bool = False,
) -> Dict[str, Any]:
    """Estima o tempo de execução baseado no método e preset."""
    
    # Estimativas de tempo por método e iterações (em CPU)
    # Estes são valores aproximados baseados em execuções anteriores
    time_estimates = {
        "nerf_static": {
            "smoke": (100, 1, 2),  # (iterações, min, max)
            "quick": (1000, 5, 15),
            "preview": (10000, 30, 60),
            "standard": (50000, 120, 360),
            "full": (200000, 1800, 3600),
        },
        "nerf_dynamic": {
            "smoke": (100, 2, 3),
            "quick": (1000, 7, 20),
            "preview": (10000, 40, 90),
            "standard": (50000, 180, 480),
            "full": (200000, 2400, 4800),
        },
        "gs_static": {
            "smoke": (100, 30, 60),  # GS é mais lento no setup
            "quick": (1000, 60, 180),
            "preview": (10000, 300, 600),
            "standard": (50000, 1800, 3600),
            "full": (200000, 7200, 14400),
        },
        "gs_dynamic": {
            "smoke": (100, 40, 80),
            "quick": (1000, 120, 300),
            "preview": (10000, 600, 1200),
            "standard": (50000, 3600, 7200),
            "full": (200000, 14400, 28800),
        },
    }
    
    if method_id not in time_estimates:
        return {
            "method": method_id,
            "estimate_available": False,
            "message": "Estimativa de tempo não disponível para este método",
        }
    
    method_estimates = time_estimates[method_id]
    
    # Determinar iterações e preset
    if iterations is not None:
        # Usar iterações diretas
        iters = iterations
        preset_name = "custom"
    elif preset is not None:
        if preset not in method_estimates:
            return {
                "method": method_id,
                "preset": preset,
                "estimate_available": False,
                "message": f"Preset '{preset}' não encontrado para {method_id}",
            }
        iters, min_time, max_time = method_estimates[preset]
        preset_name = preset
    else:
        # Usar preset padrão "quick"
        iters, min_time, max_time = method_estimates.get("quick", (1000, 5, 15))
        preset_name = "quick"
    
    # Se iterações foram customizadas, estimar proporcionalmente
    if iterations is not None and preset is not None:
        base_iters, base_min, base_max = method_estimates[preset]
        ratio = iterations / base_iters
        min_time = int(base_min * ratio)
        max_time = int(base_max * ratio)
    elif iterations is not None:
        # Estimar baseado em "quick"
        base_iters, base_min, base_max = method_estimates.get("quick", (1000, 5, 15))
        ratio = iterations / base_iters
        min_time = int(base_min * ratio)
        max_time = int(base_max * ratio)
    
    hardware = detect_hardware_snapshot()
    speed_factor = _hardware_speed_factor(hardware)

    adjusted_iterations = iters
    adaptive_factor = 1.0
    if adaptive_preset and iterations is None:
        adaptive_factor = _adaptive_iterations_factor(hardware)
        adjusted_iterations = max(1, int(iters * adaptive_factor))

    ratio = adjusted_iterations / max(iters, 1)
    min_time = int(min_time * ratio * speed_factor)
    max_time = int(max_time * ratio * speed_factor)

    return {
        "method": method_id,
        "preset": preset_name,
        "iterations": iters,
        "adjusted_iterations": adjusted_iterations,
        "estimate_minutes": {"min": min_time, "max": max_time},
        "estimate_hours": {"min": round(min_time / 60, 2), "max": round(max_time / 60, 2)},
        "hardware": {
            "has_gpu": hardware.has_gpu,
            "gpu_name": hardware.gpu_name,
            "vram_gb": hardware.vram_gb,
            "recommended_profile": hardware.recommended_profile,
        },
        "adaptive_preset": adaptive_preset,
        "adaptive_factor": adaptive_factor,
        "speed_factor": speed_factor,
        "message": f"Tempo estimado: {min_time}-{max_time} minutos (~{round(min_time/60, 1)}-{round(max_time/60, 1)} horas)",
    }


def _hardware_speed_factor(hardware: Any) -> float:
    if not hardware.has_gpu:
        return 1.0
    if hardware.vram_gb is None:
        return 0.6
    if hardware.vram_gb < 6.0:
        return 0.8
    if hardware.vram_gb < 10.0:
        return 0.5
    return 0.35


def _adaptive_iterations_factor(hardware: Any) -> float:
    if not hardware.has_gpu:
        return 0.5
    if hardware.vram_gb is None:
        return 0.75
    if hardware.vram_gb < 6.0:
        return 0.5
    if hardware.vram_gb < 10.0:
        return 0.75
    return 1.0


def print_validation_result(result: ValidationResult, verbose: bool = False) -> None:
    """Imprime resultado de validação de forma formatada."""
    status = "✓" if result.is_valid else "✗"
    print(f"{status} {result.message}")
    
    if verbose and result.details:
        for key, value in result.details.items():
            if isinstance(value, list):
                print(f"  {key}:")
                for item in value:
                    print(f"    - {item}")
            else:
                print(f"  {key}: {value}")


def print_separator(char: str = "=", width: Optional[int] = None) -> None:
    """Imprime separador visual."""
    if width is None:
        width = 70
    print(char * width)


def format_time_estimate(estimate: Dict[str, Any]) -> str:
    """Formata estimativa de tempo para exibição."""
    if not estimate.get("estimate_available", True):
        return estimate.get("message", "Estimativa não disponível")
    
    msg = estimate.get("message", "")
    if "hours" in estimate.get("estimate_hours", {}):
        hours_min = estimate["estimate_hours"]["min"]
        hours_max = estimate["estimate_hours"]["max"]
        return f"{msg}\n⏱️  {hours_min}-{hours_max} horas"
    return msg


def get_disk_usage(path: str) -> Dict[str, Any]:
    """Obtém informações de uso de disco para um caminho."""
    import os
    import shutil
    
    path_obj = Path(path)
    
    if not path_obj.exists():
        return {"error": f"Caminho não existe: {path}"}
    
    try:
        if path_obj.is_file():
            size_bytes = path_obj.stat().st_size
        else:
            size_bytes = sum(
                f.stat().st_size
                for f in path_obj.rglob("*")
                if f.is_file()
            )
        
        size_gb = size_bytes / (1024 ** 3)
        total, used, free = shutil.disk_usage(path_obj)
        
        return {
            "path_size_gb": round(size_gb, 2),
            "total_disk_gb": round(total / (1024 ** 3), 2),
            "used_disk_gb": round(used / (1024 ** 3), 2),
            "free_disk_gb": round(free / (1024 ** 3), 2),
        }
    except Exception as e:
        return {"error": str(e)}
