"""Testes de validação dos adapters real (NeRF estático e D-NeRF dinâmico)."""

import pytest
from pathlib import Path
from nvs_benchmark.core import (
    RunConfig,
    DatasetSpec,
    HardwareProfile,
    TrainRequest,
    InferenceRequest,
)
from nvs_benchmark.methods.nerf_static.adapter import NeRFStaticAdapter
from nvs_benchmark.methods.nerf_dynamic.adapter import NeRFDynamicAdapter
from nvs_benchmark.methods.gs_static.adapter import GSStaticAdapter, GSStaticHardwareError
from nvs_benchmark.methods.gs_dynamic.adapter import GSDynamicAdapter
from nvs_benchmark.core.presets import resolve_iterations


class TestAdapterInstantiation:
    """Testes de instanciação dos adapters."""

    def test_nerf_static_adapter(self):
        """Testa instantiação do adapter NeRF estático."""
        adapter = NeRFStaticAdapter()
        assert adapter.method_id == "nerf_static"
        assert adapter.capabilities.supports_train
        assert adapter.capabilities.supports_inference
        assert not adapter.capabilities.supports_dynamic_scene

    def test_nerf_dynamic_adapter(self):
        """Testa instantiação do adapter D-NeRF dinâmico."""
        adapter = NeRFDynamicAdapter()
        assert adapter.method_id == "nerf_dynamic"
        assert adapter.capabilities.supports_train
        assert adapter.capabilities.supports_inference
        assert adapter.capabilities.supports_dynamic_scene

    def test_gs_static_adapter(self):
        """Testa instantiação do adapter 3DGS estático."""
        adapter = GSStaticAdapter()
        assert adapter.method_id == "gs_static"
        assert adapter.capabilities.supports_train
        assert adapter.capabilities.supports_inference
        assert not adapter.capabilities.supports_dynamic_scene

    def test_gs_dynamic_adapter(self):
        """Testa instantiação do adapter 4DGS dinâmico."""
        adapter = GSDynamicAdapter()
        assert adapter.method_id == "gs_dynamic"
        assert adapter.capabilities.supports_train
        assert adapter.capabilities.supports_inference
        assert adapter.capabilities.supports_dynamic_scene


class TestAdapterValidation:
    """Testes de validação de configuração dos adapters."""

    def test_nerf_static_validate_config_invalid_method(self):
        """Testa validação quando método não coincide."""
        adapter = NeRFStaticAdapter()
        config = RunConfig(
            run_id="test",
            dataset=DatasetSpec(name="blender_synthetic", root="./data/dummy"),
            method="wrong_method",
            hardware_profile=HardwareProfile.ADAPTIVE,
        )
        with pytest.raises(ValueError, match="Metodo incompativel"):
            adapter.validate_config(config)

    def test_nerf_static_validate_config_invalid_dataset_root(self):
        """Testa validação com dataset root inexistente."""
        adapter = NeRFStaticAdapter()
        config = RunConfig(
            run_id="test",
            dataset=DatasetSpec(name="blender_synthetic", root="/nonexistent/path"),
            method="nerf_static",
            hardware_profile=HardwareProfile.ADAPTIVE,
        )
        with pytest.raises(ValueError, match="Dataset root nao encontrado"):
            adapter.validate_config(config)


    def test_gs_static_validate_config_requires_cuda(self, tmp_path, monkeypatch):
        """Testa que gs_static falha cedo em host sem CUDA."""
        adapter = GSStaticAdapter()
        repo_dir = tmp_path / "gaussian_splatting"
        repo_dir.mkdir()
        (repo_dir / "train.py").write_text("# train\n", encoding="utf-8")
        (repo_dir / "render.py").write_text("# render\n", encoding="utf-8")
        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir()

        monkeypatch.setattr(
            adapter,
            "_probe_runtime",
            lambda config: {
                "python_executable": "python",
                "torch_version": "2.2.0",
                "cuda_available": False,
                "module_errors": {
                    "diff_gaussian_rasterization": None,
                    "simple_knn._C": None,
                },
            },
        )

        config = RunConfig(
            run_id="test",
            dataset=DatasetSpec(name="blender_synthetic", root=str(dataset_dir)),
            method="gs_static",
            hardware_profile=HardwareProfile.ADAPTIVE,
            extra={"gs_repo_path": str(repo_dir)},
        )
        with pytest.raises(GSStaticHardwareError, match="requer CUDA"):
            adapter.validate_config(config)


class TestIterationResolution:
    """Testes de resolução de iterações (presets)."""

    def test_preset_smoke(self):
        """Testa preset smoke para NeRF estático."""
        result = resolve_iterations(
            method_id="nerf_static",
            preset_name="smoke",
        )
        assert result["N_iter"] == 100
        assert result["i_weights"] == 100

    def test_preset_quick(self):
        """Testa preset quick para NeRF estático."""
        result = resolve_iterations(
            method_id="nerf_static",
            preset_name="quick",
        )
        assert result["N_iter"] == 1000

    def test_preset_standard(self):
        """Testa preset standard para NeRF estático."""
        result = resolve_iterations(
            method_id="nerf_static",
            preset_name="standard",
        )
        assert result["N_iter"] == 50000

    def test_preset_full(self):
        """Testa preset full para NeRF estático."""
        result = resolve_iterations(
            method_id="nerf_static",
            preset_name="full",
        )
        assert result["N_iter"] == 200000

    def test_iterations_override(self):
        """Testa override direto de iterações."""
        result = resolve_iterations(
            method_id="nerf_static",
            preset_name="quick",
            iterations=5000,
        )
        assert result["N_iter"] == 5000

    def test_iterations_override_priority(self):
        """Testa que iterations override tem prioridade sobre preset."""
        result = resolve_iterations(
            method_id="nerf_static",
            preset_name="standard",
            iterations=2000,
        )
        assert result["N_iter"] == 2000

    def test_gs_static_preset(self):
        """Testa resolução de iterações para GS estático."""
        result = resolve_iterations(
            method_id="gs_static",
            preset_name="quick",
        )
        assert result["iterations"] == 1000

    def test_nerf_dynamic_preset(self):
        """Testa resolução de iterações para D-NeRF dinâmico."""
        result = resolve_iterations(
            method_id="nerf_dynamic",
            preset_name="preview",
        )
        assert result["N_iter"] == 10000


class TestIntegration:
    """Testes de integração de alto nível."""

    def test_cli_preset_vs_iterations(self):
        """Verifica que CLI pode passar --preset e --iterations."""
        # Simula resolução como feita no CLI
        extra = {}
        extra["preset"] = "quick"
        extra["iterations"] = 5000

        result = resolve_iterations(
            method_id="nerf_static",
            preset_name=extra.get("preset"),
            iterations=extra.get("iterations"),
            extra=extra,
        )
        # iterations deve ter prioridade
        assert result["N_iter"] == 5000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
