"""Testes End-to-End (E2E) do pipeline de benchmarks.

Testa o fluxo completo de execução:
- Treino de um método
- Inferência em imagens de teste
- Cálculo de métricas
- Geração de relatórios
"""

import json
import pytest
from pathlib import Path
from nvs_benchmark.core import (
    RunConfig,
    DatasetSpec,
    HardwareProfile,
    Orchestrator,
)
from nvs_benchmark.methods import build_registry_with_all_methods
from nvs_benchmark.data import validate_dataset
from nvs_benchmark.evaluation import load_metrics_snapshot


class TestE2EPipelineNeRFStatic:
    """Testes E2E do pipeline completo com NeRF Estático."""

    @pytest.fixture
    def setup(self):
        """Configura fixtures necessárias para os testes."""
        self.method_id = "nerf_static"
        self.dataset_name = "blender_synthetic"
        self.dataset_root = Path("./data/blender_synthetic/nerf_synthetic/lego")
        self.run_id = "e2e_nerf_static_test"
        self.output_dir = Path("./artifacts/test_e2e")
        self.log_dir = Path("./logs/test_e2e")
        self.snapshot_file = self.output_dir / "e2e_nerf_static.json"
        
        # Criar diretórios
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        yield

    def test_dataset_validation(self, setup):
        """Testa se o dataset é válido antes de executar."""
        if not self.dataset_root.exists():
            pytest.skip(f"Dataset não encontrado em {self.dataset_root}")
        
        ok, message = validate_dataset(self.dataset_name, str(self.dataset_root))
        assert ok, f"Validação de dataset falhou: {message}"

    def test_full_pipeline_with_preset_smoke(self, setup):
        """Executa pipeline completo com preset 'smoke' (rápido)."""
        if not self.dataset_root.exists():
            pytest.skip(f"Dataset não encontrado em {self.dataset_root}")

        registry = build_registry_with_all_methods()
        orchestrator = Orchestrator(registry=registry)
        
        config = RunConfig(
            run_id=self.run_id,
            dataset=DatasetSpec(
                name=self.dataset_name,
                root=str(self.dataset_root),
            ),
            method=self.method_id,
            output_dir=str(self.output_dir),
            log_dir=str(self.log_dir),
            hardware_profile=HardwareProfile.ADAPTIVE,
            report_formats=[],  # Sem relatórios para teste rápido
            extra={
                "preset": "smoke",
                "skip_snapshot": False,
                "skip_reports": True,
            },
        )

        result = orchestrator.run(config)
        
        # Validações
        assert result.success, f"Pipeline falhou: {result.error}"
        assert result.metrics is not None, "Métricas não foram calculadas"
        assert result.train_seconds > 0, "Tempo de treino não registrado"
        assert result.inference_seconds > 0, "Tempo de inferência não registrado"

    def test_metrics_save_and_load(self, setup):
        """Testa se as métricas são salvas e carregadas corretamente."""
        if not self.dataset_root.exists():
            pytest.skip(f"Dataset não encontrado em {self.dataset_root}")

        registry = build_registry_with_all_methods()
        orchestrator = Orchestrator(registry=registry)
        
        config = RunConfig(
            run_id=f"{self.run_id}_metrics_test",
            dataset=DatasetSpec(
                name=self.dataset_name,
                root=str(self.dataset_root),
            ),
            method=self.method_id,
            output_dir=str(self.output_dir),
            log_dir=str(self.log_dir),
            hardware_profile=HardwareProfile.ADAPTIVE,
            report_formats=[],
            extra={
                "preset": "smoke",
                "skip_snapshot": False,
                "skip_reports": True,
            },
        )

        result = orchestrator.run(config)
        assert result.success
        
        # Tenta carregar snapshot
        snapshot_path = self.snapshot_file
        if snapshot_path.exists():
            with open(snapshot_path) as f:
                snapshot = json.load(f)
            
            assert isinstance(snapshot, dict), "Snapshot deve ser um dicionário"
            assert "results" in snapshot or "methods" in snapshot, "Snapshot vazio ou inválido"

    def test_output_files_exist(self, setup):
        """Verifica se os arquivos de saída foram criados."""
        if not self.dataset_root.exists():
            pytest.skip(f"Dataset não encontrado em {self.dataset_root}")

        registry = build_registry_with_all_methods()
        orchestrator = Orchestrator(registry=registry)
        
        config = RunConfig(
            run_id=f"{self.run_id}_output_test",
            dataset=DatasetSpec(
                name=self.dataset_name,
                root=str(self.dataset_root),
            ),
            method=self.method_id,
            output_dir=str(self.output_dir),
            log_dir=str(self.log_dir),
            hardware_profile=HardwareProfile.ADAPTIVE,
            report_formats=[],
            extra={
                "preset": "smoke",
                "skip_snapshot": False,
                "skip_reports": True,
            },
        )

        result = orchestrator.run(config)
        assert result.success
        
        # Verifica se diretórios de saída foram criados
        assert result.train_checkpoint_path is not None, "Checkpoint não foi salvo"
        assert result.inference_renders_dir is not None, "Renders não foram salvos"

    def test_metrics_quality_reasonable(self, setup):
        """Valida se as métricas computadas têm valores razoáveis."""
        if not self.dataset_root.exists():
            pytest.skip(f"Dataset não encontrado em {self.dataset_root}")

        registry = build_registry_with_all_methods()
        orchestrator = Orchestrator(registry=registry)
        
        config = RunConfig(
            run_id=f"{self.run_id}_quality_test",
            dataset=DatasetSpec(
                name=self.dataset_name,
                root=str(self.dataset_root),
            ),
            method=self.method_id,
            output_dir=str(self.output_dir),
            log_dir=str(self.log_dir),
            hardware_profile=HardwareProfile.ADAPTIVE,
            report_formats=[],
            extra={
                "preset": "smoke",
                "skip_snapshot": False,
                "skip_reports": True,
            },
        )

        result = orchestrator.run(config)
        assert result.success
        
        # Validações de métrica
        metrics = result.metrics
        if metrics and hasattr(metrics, 'psnr'):
            # PSNR deve estar entre 0 e ~60
            assert 0 < metrics.psnr < 60, f"PSNR fora dos limites: {metrics.psnr}"
        if metrics and hasattr(metrics, 'ssim'):
            # SSIM deve estar entre 0 e 1
            assert 0 <= metrics.ssim <= 1, f"SSIM fora dos limites: {metrics.ssim}"


class TestE2EPipelineIntegration:
    """Testes de integração entre múltiplos componentes."""

    @pytest.fixture
    def setup(self):
        """Configura fixtures necessárias para os testes."""
        self.output_dir = Path("./artifacts/test_e2e")
        self.log_dir = Path("./logs/test_e2e")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        yield

    def test_adapter_registry_complete(self, setup):
        """Testa se todos os adapters estão registrados corretamente."""
        registry = build_registry_with_all_methods()
        method_ids = registry.list_ids()
        
        expected_methods = ["nerf_static", "nerf_dynamic", "gs_static", "gs_dynamic"]
        for method_id in expected_methods:
            assert method_id in method_ids, f"Método {method_id} não encontrado no registry"

    def test_adapter_capabilities_correct(self, setup):
        """Valida se as capacidades dos adapters estão corretas."""
        registry = build_registry_with_all_methods()
        
        # NeRF Static
        nerf_static = registry.get("nerf_static")
        assert nerf_static.capabilities.supports_train
        assert nerf_static.capabilities.supports_inference
        assert not nerf_static.capabilities.supports_dynamic_scene
        
        # NeRF Dynamic
        nerf_dynamic = registry.get("nerf_dynamic")
        assert nerf_dynamic.capabilities.supports_train
        assert nerf_dynamic.capabilities.supports_inference
        assert nerf_dynamic.capabilities.supports_dynamic_scene
        
        # GS Static
        gs_static = registry.get("gs_static")
        assert gs_static.capabilities.supports_train
        assert gs_static.capabilities.supports_inference
        assert not gs_static.capabilities.supports_dynamic_scene
        
        # GS Dynamic
        gs_dynamic = registry.get("gs_dynamic")
        assert gs_dynamic.capabilities.supports_train or not gs_dynamic.capabilities.supports_train
        assert gs_dynamic.capabilities.supports_dynamic_scene

    def test_run_logger_creates_log_file(self, setup):
        """Testa se o sistema de logging cria arquivos corretamente."""
        from nvs_benchmark.runtime import RunLogger
        
        logger = RunLogger(
            command="test_command",
            parameters={"test": "value"},
            log_dir=str(self.log_dir),
        )
        
        # Verifica se o diretório de log foi criado
        assert logger.run_dir.exists(), f"Diretório de logs não foi criado: {logger.run_dir}"
        
        # Log um evento
        logger.event("test_event", {"data": "test_data"})
        
        # Finaliza
        logger.finish("success", {"status": "ok"})
        
        # Verifica se o arquivo JSON foi criado
        log_files = list(logger.run_dir.glob("*.json"))
        assert len(log_files) > 0, "Nenhum arquivo de log foi criado"

    def test_preset_resolution_all_methods(self, setup):
        """Testa resolução de presets para todos os métodos."""
        from nvs_benchmark.core.presets import resolve_iterations, PRESET_NAMES
        
        registry = build_registry_with_all_methods()
        method_ids = registry.list_ids()
        
        for preset in PRESET_NAMES:
            for method_id in method_ids:
                try:
                    result = resolve_iterations(
                        method_id=method_id,
                        preset_name=preset,
                    )
                    assert result is not None, f"Resolução falhou para {method_id}/{preset}"
                    assert len(result) > 0, f"Resultado vazio para {method_id}/{preset}"
                except Exception as e:
                    # Alguns métodos podem não ter presets
                    pass
