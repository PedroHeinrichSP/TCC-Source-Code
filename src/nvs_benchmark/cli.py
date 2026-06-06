"""Interface de linha de comando para operações do benchmark NVS."""

import argparse
import json
import math
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from nvs_benchmark.core import DatasetSpec, HardwareProfile, RunConfig
from nvs_benchmark.core import Orchestrator
from nvs_benchmark.core import InferenceRequest, TrainRequest
from nvs_benchmark.core.presets import PRESET_NAMES, list_preset_summaries
from nvs_benchmark.data import SUPPORTED_DATASETS, load_dataset, validate_dataset
from nvs_benchmark.evaluation import BenchmarkMetrics, save_metrics_snapshot, evaluate_benchmark_metrics, write_reference_image
from nvs_benchmark.methods import ExternalMethodAdapter, build_registry_with_all_methods
from nvs_benchmark.methods.utils import export_reference_frames_from_dataset
from nvs_benchmark.methods.gs_static.adapter import GSStaticHardwareError
from nvs_benchmark.reporting import generate_comparison_reports
from nvs_benchmark.runtime import RunLogger, audit_docstrings, save_doc_audit_report
from nvs_benchmark.install import load_install_catalog, install_items
from nvs_benchmark.cli_extensions import (
    validate_dataset_path,
    validate_dataset_integrity_preflight,
    validate_output_directory,
    validate_method_available,
    validate_snapshot_file,
    validate_matrix_completeness,
    estimate_execution_time,
    print_validation_result,
    print_separator,
    format_time_estimate,
    get_disk_usage,
)


def _resolve_smoke_dataset_spec() -> DatasetSpec:
    """Resolve dataset de smoke com fallback para fixture local conhecida."""
    candidates = [
        Path("./data/_smoke/blender"),
        Path("./data/blender_synthetic/nerf_synthetic/lego"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return DatasetSpec(name="blender_synthetic", root=str(candidate))
    expected = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Nenhum dataset de smoke encontrado. Caminhos verificados: {expected}")


def _is_hardware_skip(method_id: str, exc: Exception) -> bool:
    """Retorna True quando a falha deve virar skip por hardware incompatível."""
    return method_id == "gs_static" and isinstance(exc, GSStaticHardwareError)


def _sanitize_run_token(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    normalized = normalized.strip("_-")
    return normalized or "run"


def _build_cli_run_id(*, method_id: str, dataset: str, root: str) -> str:
    scene_name = Path(root).name or dataset
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return "-".join(
        [
            "custom",
            _sanitize_run_token(method_id),
            _sanitize_run_token(dataset),
            _sanitize_run_token(scene_name),
            stamp,
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    """Monta parser de argumentos com todos os subcomandos suportados."""
    parser = argparse.ArgumentParser(prog="nvs-benchmark")
    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser("init", help="Validar estrutura do projeto")
    init_parser.add_argument("--config", default="configs/base.yaml", help="Caminho da configuração base")

    subparsers.add_parser("status", help="Exibir status do benchmark")
    subparsers.add_parser("contracts", help="Validar contratos compartilhados")

    dataset_parser = subparsers.add_parser("dataset-check", help="Validar e carregar metadados de dataset")
    dataset_parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS, help="Nome do dataset")
    dataset_parser.add_argument("--root", required=True, help="Diretório raiz do dataset")
    dataset_parser.add_argument("--split", default="train", help="Split para carregar metadados")

    methods_parser = subparsers.add_parser("methods-check", help="Validar integração dos métodos")
    methods_parser.add_argument("--output-dir", default="./artifacts", help="Diretório de saída para smoke test")
    methods_parser.add_argument("--log-dir", default="./logs", help="Diretório de logs estruturados")

    metrics_parser = subparsers.add_parser("metrics-check", help="Executar inferência smoke e calcular métricas")
    metrics_parser.add_argument("--output-dir", default="./artifacts", help="Diretório de saída da execução")
    metrics_parser.add_argument(
        "--snapshot-file",
        default="./artifacts/metrics/latest_preview.json",
        help="Arquivo JSON de snapshot",
    )
    metrics_parser.add_argument("--log-dir", default="./logs", help="Diretório de logs estruturados")

    metrics_compute_parser = subparsers.add_parser(
        "metrics-compute",
        help="Recomputar métricas a partir de checkpoint e imagens renderizadas pré-existentes",
    )
    metrics_compute_parser.add_argument("--method", required=True, help="Identificador do método (ex: nerf_static)")
    metrics_compute_parser.add_argument("--checkpoint", required=True, help="Caminho para checkpoint treinado")
    metrics_compute_parser.add_argument("--rendered-dir", required=True, help="Diretório com imagens renderizadas (PNG)")
    metrics_compute_parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS, help="Nome do dataset")
    metrics_compute_parser.add_argument("--root", required=True, help="Diretório raiz do dataset")
    metrics_compute_parser.add_argument("--reference-dir", default=None, help="Diretório de imagens de referência (override)")
    metrics_compute_parser.add_argument("--split", default="train", help="Split do dataset")
    metrics_compute_parser.add_argument(
        "--snapshot-file",
        required=True,
        help="Arquivo JSON para salvar métricas",
    )
    metrics_compute_parser.add_argument(
        "--append-snapshot",
        action="store_true",
        help="Anexar/atualizar método no snapshot existente",
    )
    metrics_compute_parser.add_argument(
        "--train-seconds",
        type=float,
        default=0.0,
        help="Tempo de treinamento em segundos (para referência)",
    )
    metrics_compute_parser.add_argument(
        "--inference-seconds",
        type=float,
        default=0.0,
        help="Tempo de inferência em segundos (para referência)",
    )
    metrics_compute_parser.add_argument("--log-dir", default="./logs", help="Diretório de logs estruturados")
    metrics_compute_parser.add_argument(
        "--strict-results",
        action="store_true",
        help="Falha se metricas invalidas forem detectadas",
    )
    metrics_compute_parser.add_argument(
        "--min-required-pairs",
        type=int,
        default=1,
        help="Numero minimo de pares validos exigidos",
    )
    metrics_compute_parser.add_argument(
        "--metrics-max-pairs",
        type=int,
        default=None,
        help="Limita a quantidade de pares avaliados durante a etapa de metricas",
    )
    metrics_compute_parser.add_argument(
        "--metrics-max-image-dim",
        type=int,
        default=None,
        help="Reduz a maior dimensao das imagens antes de calcular metricas",
    )
    metrics_compute_parser.add_argument(
        "--metrics-log-every",
        type=int,
        default=None,
        help="Emite progresso de metricas a cada N pares",
    )

    report_parser = subparsers.add_parser(
        "report-generate",
        help="Gerar relatório comparativo a partir do snapshot de métricas",
    )
    report_parser.add_argument(
        "--snapshot-file",
        default="./artifacts/metrics/latest_preview.json",
        help="Arquivo JSON de snapshot",
    )
    report_parser.add_argument(
        "--output-dir",
        default="./artifacts/reports",
        help="Diretório de saída do relatório",
    )
    report_parser.add_argument(
        "--report-name",
        default="benchmark_report",
        help="Nome-base do arquivo de relatório",
    )
    report_parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Não gerar relatório em PDF",
    )
    report_parser.add_argument("--log-dir", default="./logs", help="Diretório de logs estruturados")
    report_parser.add_argument(
        "--strict-snapshot",
        action="store_true",
        help="Falhar se o snapshot nao atender criterios de completude/sanidade",
    )
    report_parser.add_argument(
        "--expected-methods",
        default=None,
        help="Lista separada por virgula de metodos esperados no snapshot",
    )
    report_parser.add_argument(
        "--min-methods",
        type=int,
        default=None,
        help="Quantidade minima de metodos exigida no snapshot",
    )
    report_parser.add_argument(
        "--require-finite-metrics",
        action="store_true",
        help="Exigir metricas finitas em todos os metodos do snapshot",
    )
    report_parser.add_argument(
        "--expected-matrix",
        default=None,
        help="Lista separada por virgula de combinacoes esperadas no formato dataset|method",
    )

    docs_parser = subparsers.add_parser("docs-check", help="Auditar docstrings públicas")
    docs_parser.add_argument("--source-dir", default="./src/nvs_benchmark", help="Diretório de código-fonte")
    docs_parser.add_argument(
        "--report-file",
        default="./artifacts/docs/latest_doc_audit.json",
        help="Arquivo JSON de relatório de saída",
    )
    docs_parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Falhar se docstrings ausentes forem encontradas",
    )

    standard_parser = subparsers.add_parser("standard-test", help="Executar bateria padrão de validação")
    standard_parser.add_argument("--output-dir", default="./artifacts", help="Diretório base de artefatos")
    standard_parser.add_argument("--log-dir", default="./logs", help="Diretório de logs estruturados")
    standard_parser.add_argument(
        "--snapshot-file",
        default="./artifacts/metrics/latest_preview.json",
        help="Arquivo de snapshot de métricas",
    )
    standard_parser.add_argument(
        "--report-name",
        default="benchmark_report",
        help="Nome-base do arquivo de relatório",
    )
    standard_parser.add_argument(
        "--run-unit-tests",
        action="store_true",
        help="Executar testes unitários ao final da suíte",
    )

    install_parser = subparsers.add_parser("install", help="Instalar datasets/modelos via catalogo")
    install_parser.add_argument(
        "--catalog-file",
        default="./configs/install_catalog.json",
        help="Catalogo JSON de instalacao",
    )
    install_parser.add_argument(
        "--only",
        choices=["datasets", "methods", "all"],
        default="all",
        help="Seleciona quais itens instalar",
    )
    install_parser.add_argument(
        "--execute",
        action="store_true",
        help="Executa comandos definidos no catalogo",
    )

    method_parser = subparsers.add_parser(
        "method-run",
        help="Executar um unico metodo (incluindo external) em um dataset",
    )

    method_parser.add_argument("--method", required=True, help="ID do método (ex: nerf_static, external)")
    method_parser.add_argument(
        "--run-id",
        default=None,
        help="Identificador da execucao para organizar artifacts/run_id/metodo",
    )
    method_parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS, help="Nome do dataset")
    method_parser.add_argument("--root", required=True, help="Diretório raiz do dataset")
    method_parser.add_argument("--split", default="train", help="Split para treino/inferência")
    method_parser.add_argument("--output-dir", default="./artifacts", help="Diretório de saída")
    method_parser.add_argument("--log-dir", default="./logs", help="Diretório de logs estruturados")
    method_parser.add_argument("--compute-metrics", action="store_true", help="Calcular métricas ao final")
    method_parser.add_argument("--reference-dir", default=None, help="Diretório de imagens de referência")
    method_parser.add_argument(
        "--snapshot-file",
        default=None,
        help="Arquivo JSON para salvar métricas do method-run",
    )
    method_parser.add_argument(
        "--append-snapshot",
        action="store_true",
        help="Anexar/atualizar método no snapshot existente em --snapshot-file",
    )
    method_parser.add_argument(
        "--extra-json",
        default=None,
        help="JSON inline com parâmetros extras (ex: comandos externos)",
    )
    method_parser.add_argument(
        "--extra-file",
        default=None,
        help="Caminho para arquivo JSON com parâmetros extras",
    )
    method_parser.add_argument(
        "--preset",
        default=None,
        choices=list(PRESET_NAMES),
        help="Preset de iterações: smoke, quick, preview, standard, full",
    )
    method_parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override direto do número de iterações de treino",
    )
    method_parser.add_argument(
        "--estimate-time",
        action="store_true",
        help="Exibir estimativa de tempo antes de executar",
    )
    method_parser.add_argument(
        "--cache-disable",
        action="store_true",
        help="Desabilitar cache inteligente nesta execução",
    )
    method_parser.add_argument(
        "--cache-max-size-gb",
        type=float,
        default=50.0,
        help="Limite máximo de cache em GB para limpeza LRU",
    )
    method_parser.add_argument(
        "--reuse-renders",
        action="store_true",
        help="Permitir reutilização de renders quando houver cache válido",
    )
    method_parser.add_argument(
        "--reuse-metrics",
        action="store_true",
        help="Permitir reutilização de métricas quando houver cache válido",
    )
    method_parser.add_argument(
        "--adaptive-preset",
        action="store_true",
        help="Aplicar ajuste conservador de iterações com base no hardware disponível",
    )
    method_parser.add_argument(
        "--validation-full",
        action="store_true",
        help="Validar 100%% dos frames do dataset antes de executar",
    )
    method_parser.add_argument(
        "--validation-sample-size",
        type=int,
        default=16,
        help="Quantidade de frames amostrados no preflight quando --validation-full nao for usado",
    )
    method_parser.add_argument(
        "--strict-results",
        action="store_true",
        help="Falha se metricas invalidas forem detectadas (NaN/inf, pairs insuficiente, fps<=0)",
    )
    method_parser.add_argument(
        "--min-required-pairs",
        type=int,
        default=1,
        help="Numero minimo de pares validos exigidos quando --strict-results estiver ativo",
    )

    subparsers.add_parser("presets-list", help="Listar presets de iterações disponíveis")

    list_datasets_parser = subparsers.add_parser("list-datasets", help="Listar datasets disponíveis")
    list_datasets_parser.add_argument(
        "--check-files",
        action="store_true",
        help="Verificar se os arquivos de dataset existem",
    )

    list_methods_parser = subparsers.add_parser("list-methods", help="Listar métodos disponíveis")
    list_methods_parser.add_argument(
        "--detailed",
        action="store_true",
        help="Mostrar informações detalhadas de cada método",
    )

    estimate_parser = subparsers.add_parser("estimate-time", help="Estimar tempo de execução")
    estimate_parser.add_argument("--method", required=True, help="ID do método")
    estimate_parser.add_argument(
        "--preset",
        default=None,
        choices=list(PRESET_NAMES),
        help="Preset de iterações",
    )
    estimate_parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Número de iterações customizado",
    )
    estimate_parser.add_argument(
        "--adaptive-preset",
        action="store_true",
        help="Aplicar ajuste conservador por hardware na estimativa",
    )

    dry_run_parser = subparsers.add_parser("validate-run", help="Validar configuração antes de executar")
    dry_run_parser.add_argument("--method", required=True, help="ID do método")
    dry_run_parser.add_argument("--dataset", required=True, choices=SUPPORTED_DATASETS, help="Nome do dataset")
    dry_run_parser.add_argument("--root", required=True, help="Diretório raiz do dataset")
    dry_run_parser.add_argument("--output-dir", default="./artifacts", help="Diretório de saída")
    dry_run_parser.add_argument("--preset", default=None, choices=list(PRESET_NAMES), help="Preset de iterações")
    dry_run_parser.add_argument("--iterations", type=int, default=None, help="Número de iterações")
    dry_run_parser.add_argument(
        "--validation-full",
        action="store_true",
        help="Validar 100%% dos frames do dataset",
    )
    dry_run_parser.add_argument(
        "--validation-sample-size",
        type=int,
        default=16,
        help="Frames amostrados na validacao quando --validation-full nao for usado",
    )
    dry_run_parser.add_argument(
        "--adaptive-preset",
        action="store_true",
        help="Simular ajuste conservador de preset no estimate-time",
    )

    create_adapter_parser = subparsers.add_parser(
        "create-adapter",
        help="Gerar scaffold de adaptador de metodo para estudantes",
    )
    create_adapter_parser.add_argument(
        "name",
        help="Nome do adaptador (ex: meu_metodo). Sera usado como ID e nome da pasta.",
    )
    create_adapter_parser.add_argument(
        "--output-dir",
        default="./src/nvs_benchmark/methods",
        help="Diretório onde a pasta do adaptador sera criada",
    )
    create_adapter_parser.add_argument(
        "--kind",
        default="nerf_static",
        choices=["nerf_static", "nerf_dynamic", "gs_static", "gs_dynamic", "external"],
        help="Familia do metodo (define capabilities padrao)",
    )

    return parser


def run_create_adapter(name: str, output_dir: str, kind: str) -> int:
    """Gera scaffold completo de adaptador de metodo para o estudante."""
    import re
    import textwrap

    # Validar nome
    if not re.match(r"^[a-z][a-z0-9_]{1,31}$", name):
        print("Erro: nome deve comecar com letra minuscula e conter apenas letras, digitos e '_' (max 32 chars).")
        return 1

    adapter_dir = Path(output_dir) / name
    if adapter_dir.exists():
        print(f"Pasta ja existe: {adapter_dir}")
        print("Remova-a manualmente ou escolha outro nome.")
        return 1

    is_dynamic = kind in {"nerf_dynamic", "gs_dynamic"}
    is_gs = kind in {"gs_static", "gs_dynamic"}
    display_name = name.replace("_", " ").title()

    # --- adapter/__init__.py ---
    init_content = textwrap.dedent(f"""\
        \"\"\"Adaptador de metodo: {display_name}.\"\"\"  # noqa: D100

        from .adapter import {name.title().replace('_','')}Adapter

        __all__ = ["{name.title().replace('_','')}Adapter"]
        """)

    # --- adapter/adapter.py ---
    supports_dynamic = str(is_dynamic).lower()
    adapter_content = textwrap.dedent(f"""\
        \"\"\"Implementacao do adaptador {display_name}.

        Seguir os passos marcados com TODO para completar a integracao.
        Consulte os adapters existentes e a documentacao principal do projeto para orientacoes.
        \"\"\"

        from __future__ import annotations

        import subprocess
        from pathlib import Path

        from nvs_benchmark.core.contracts import (
            InferenceRequest,
            InferenceResult,
            MethodCapabilities,
            PerformanceStats,
            RunConfig,
            TrainRequest,
            TrainResult,
        )
        from nvs_benchmark.methods.subprocess_utils import format_subprocess_error, run_subprocess


        class {name.title().replace('_', '')}Adapter:
            \"\"\"Adaptador para o metodo {display_name}.\"\"\"

            method_id: str = "{name}"
            display_name: str = "{display_name}"
            capabilities: MethodCapabilities = MethodCapabilities(
                supports_train=True,
                supports_inference=True,
                supports_dynamic_scene={supports_dynamic},
                supports_limited_gpu=True,
            )

            def validate_config(self, config: RunConfig) -> None:
                \"\"\"Valida a configuracao antes de executar.\"\"\"
                # TODO: valide pre-condicoes necessarias, ex: GPU disponivel, dataset correto, etc.
                pass

            def train(self, request: TrainRequest) -> TrainResult:
                \"\"\"Executa o treinamento do metodo.\"\"\"
                config = request.config
                output_dir = Path(config.output_dir) / config.run_id / self.method_id
                output_dir.mkdir(parents=True, exist_ok=True)
                checkpoint = output_dir / "checkpoint_final.pth"

                # TODO: construa o comando real para treinar seu metodo.
                # Exemplo:
                #   cmd = [
                #       "python", "./third_party/{name}/train.py",
                #       "--data", config.dataset.root,
                #       "--output", str(output_dir),
                #   ]
                #   result = run_subprocess(cmd, timeout=config.timeout_seconds)
                #   if not result.success:
                #       raise RuntimeError(format_subprocess_error(result))

                # Remova este bloco stub quando implementar o treino real:
                import time
                start = time.perf_counter()
                checkpoint.write_text("# stub checkpoint", encoding="utf-8")
                elapsed = time.perf_counter() - start

                return TrainResult(
                    method=self.method_id,
                    checkpoint_path=str(checkpoint),
                    train_seconds=elapsed,
                    output_dir=str(output_dir),
                )

            def infer(self, request: InferenceRequest) -> InferenceResult:
                \"\"\"Executa a inferencia (render de imagens de novos pontos de vista).\"\"\"
                config = request.config
                render_dir = Path(config.output_dir) / config.run_id / self.method_id / "renders"
                render_dir.mkdir(parents=True, exist_ok=True)

                # TODO: construa o comando real para inferencia do seu metodo.
                # Exemplo:
                #   cmd = [
                #       "python", "./third_party/{name}/render.py",
                #       "--checkpoint", request.checkpoint_path,
                #       "--output", str(render_dir),
                #   ]
                #   result = run_subprocess(cmd, timeout=config.timeout_seconds)
                #   if not result.success:
                #       raise RuntimeError(format_subprocess_error(result))

                # Stub: copia primeiro frame do dataset como render simulado
                import time
                from nvs_benchmark.methods.utils import render_stub_from_dataset
                start = time.perf_counter()
                render_stub_from_dataset(root=config.dataset.root, split=request.split, render_dir=render_dir)
                elapsed = time.perf_counter() - start

                rendered_frames = list(render_dir.glob("*.png")) + list(render_dir.glob("*.jpg"))
                return InferenceResult(
                    method=self.method_id,
                    rendered_dir=str(render_dir),
                    frames=len(rendered_frames),
                    inference_seconds=elapsed,
                )

            def collect_performance(self) -> PerformanceStats:
                \"\"\"Retorna estatísticas de desempenho medidas durante a execucao.\"\"\"
                # TODO: preencha com valores reais coletados durante train/infer.
                return PerformanceStats(
                    fps=0.0,
                    vram_gb_peak=0.0,
                    train_seconds=0.0,
                    inference_seconds=0.0,
                )
        """)

    # --- README.md do adaptador ---
    readme_content = textwrap.dedent(f"""\
        # Adaptador: {display_name}

        Este diretorio contem o adaptador NVS Benchmark para o metodo **{display_name}**.

        ## Como implementar

        1. Edite `adapter.py` e substitua os blocos `# TODO` pelo codigo real do seu metodo.
        2. O metodo deve:
           - `train()`: executar treinamento e salvar checkpoint
           - `infer()`: renderizar imagens de novos pontos de vista
        3. Use `run_subprocess()` de `methods/subprocess_utils.py` para chamar scripts externos.
        4. Registre o adaptador em `methods/__init__.py` ou `methods/registry.py`.

        ## Testando

        ```bash
        nvs-benchmark method-run \\\\
            --method {name} \\\\
            --dataset blender_synthetic \\\\
            --root ./data/blender_synthetic/nerf_synthetic/lego \\\\
            --preset smoke
        ```

        ## Estrutura

        ```
        {name}/
          __init__.py     # exporta o adaptador
          adapter.py      # implementacao principal (edite aqui)
          README.md       # este arquivo
        ```
        """)

    # Criar arquivos
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "__init__.py").write_text(init_content, encoding="utf-8")
    (adapter_dir / "adapter.py").write_text(adapter_content, encoding="utf-8")
    (adapter_dir / "README.md").write_text(readme_content, encoding="utf-8")

    print(f"Scaffold criado em: {adapter_dir}")
    print()
    print("Proximos passos:")
    print(f"  1. Edite {adapter_dir / 'adapter.py'} e substitua os blocos TODO")
    print(f"  2. Registre o adaptador no registry (methods/__init__.py ou methods/registry.py)")
    print(f"  3. Teste com:")
    print(f"     nvs-benchmark method-run --method {name} --dataset blender_synthetic ")
    print(f"       --root ./data/minha_cena --preset smoke")
    print()
    return 0


def run_init(config_path: str) -> int:
    """Valida caminho do arquivo de configuração base."""
    path = Path(config_path)
    if not path.exists():
        print(f"Configuracao nao encontrada: {path}")
        return 1
    print("Estrutura base inicializada com sucesso.")
    print(f"Configuracao: {path}")
    return 0


def run_status() -> int:
    """Exibe status atual do projeto."""
    print("nvs-benchmark v0.1.0")
    print("Status: estrutura base criada")
    return 0


def run_contracts_check() -> int:
    """Executa verificação rápida dos contratos compartilhados."""
    sample = RunConfig(
        run_id="sample-run",
        dataset=DatasetSpec(name="blender_synthetic", root="./data/blender"),
        method="nerf_static",
        hardware_profile=HardwareProfile.ADAPTIVE,
    )
    print("Contratos carregados com sucesso.")
    print(f"Run ID: {sample.run_id}")
    print(f"Metodo: {sample.method}")
    print(f"Dataset: {sample.dataset.name}")
    print(f"Perfil de hardware: {sample.hardware_profile.value}")
    return 0


def run_dataset_check(dataset: str, root: str, split: str) -> int:
    """Valida e carrega um split de dataset."""
    ok, message = validate_dataset(dataset, root)
    if not ok:
        print(f"Validacao falhou: {message}")
        return 1

    spec = load_dataset(dataset_name=dataset, root=root, split=split)
    print("Dataset validado e carregado com sucesso.")
    print(f"Dataset: {spec.name}")
    print(f"Split: {spec.split}")
    print(f"Root: {spec.root}")
    print(f"Metadados: {spec.metadata}")
    return 0


def run_methods_check(output_dir: str, log_dir: str) -> int:
    """Executa treino/inferência smoke para todos os métodos registrados."""
    logger = RunLogger(
        command="methods-check",
        parameters={"output_dir": output_dir, "log_dir": log_dir},
        log_dir=log_dir,
    )
    try:
        registry = build_registry_with_all_methods()
        method_ids = registry.list_ids()
        print(f"Metodos registrados: {method_ids}")
        logger.event("run_started", {"method_ids": method_ids})

        sample_dataset = _resolve_smoke_dataset_spec()
        completed_methods: list[str] = []
        skipped_methods: list[str] = []
        for method_id in method_ids:
            config = RunConfig(
                run_id=f"smoke-{method_id}",
                dataset=sample_dataset,
                method=method_id,
                output_dir=output_dir,
                log_dir=log_dir,
                hardware_profile=HardwareProfile.ADAPTIVE,
            )
            method = registry.get(method_id)
            try:
                method.validate_config(config)
                train_result = method.train(TrainRequest(config=config))
                infer_result = method.infer(
                    InferenceRequest(config=config, checkpoint_path=train_result.checkpoint_path, split="test")
                )
            except Exception as exc:
                if _is_hardware_skip(method_id, exc):
                    skipped_methods.append(method_id)
                    logger.event("method_skipped", {"method": method_id, "reason": str(exc)})
                    print(f"[{method_id}] skipped: {exc}")
                    continue
                raise

            logger.event(
                "method_completed",
                {
                    "method": method_id,
                    "checkpoint": train_result.checkpoint_path,
                    "renders": infer_result.rendered_dir,
                    "train_seconds": train_result.train_seconds,
                    "inference_seconds": infer_result.inference_seconds,
                },
            )
            completed_methods.append(method_id)
            print(f"[{method_id}] checkpoint: {train_result.checkpoint_path}")
            print(f"[{method_id}] renders: {infer_result.rendered_dir}")

        logger.finish(
            "success",
            {
                "methods_count": len(completed_methods),
                "skipped_methods": skipped_methods,
                "dataset_root": sample_dataset.root,
            },
        )
        print(f"Dataset de smoke: {sample_dataset.root}")
        print(f"Metodos validados com sucesso: {completed_methods}")
        if skipped_methods:
            print(f"Metodos pulados por hardware: {skipped_methods}")
        print(f"Logs da execucao: {logger.run_dir}")
        return 0
    except Exception as exc:
        logger.finish("failed", {"error": str(exc)})
        print(f"Falha em methods-check: {exc}")
        print(f"Logs da execucao: {logger.run_dir}")
        return 1


def run_metrics_check(output_dir: str, snapshot_file: str, log_dir: str) -> int:
    """Executa benchmark smoke com PSNR/SSIM/LPIPS e métricas de desempenho."""
    logger = RunLogger(
        command="metrics-check",
        parameters={
            "output_dir": output_dir,
            "snapshot_file": snapshot_file,
            "log_dir": log_dir,
        },
        log_dir=log_dir,
    )
    try:
        registry = build_registry_with_all_methods()
        orchestrator = Orchestrator(registry=registry)
        method_ids = registry.list_ids()
        sample_dataset = _resolve_smoke_dataset_spec()
        metrics = []
        skipped_methods: list[str] = []
        logger.event("run_started", {"method_ids": method_ids})

        for method_id in method_ids:
            run_id = f"metrics-{method_id}"
            config = RunConfig(
                run_id=run_id,
                dataset=sample_dataset,
                method=method_id,
                output_dir=output_dir,
                log_dir=log_dir,
                hardware_profile=HardwareProfile.ADAPTIVE,
                report_formats=[],
                extra={"skip_snapshot": True, "skip_reports": True},
            )

            try:
                result = orchestrator.run(config)
            except Exception as exc:
                if _is_hardware_skip(method_id, exc):
                    skipped_methods.append(method_id)
                    logger.event("method_skipped", {"method": method_id, "reason": str(exc)})
                    print(f"[{method_id}] skipped: {exc}")
                    continue
                raise
            if result.metrics is None:
                raise RuntimeError(f"Metricas nao foram calculadas para {method_id}.")
            metric = result.metrics
            metrics.append(metric)
            logger.event(
                "metrics_computed",
                {
                    "method": method_id,
                    "psnr": metric.psnr,
                    "ssim": metric.ssim,
                    "lpips": metric.lpips,
                    "fps": metric.fps,
                    "vram_gb": metric.vram_gb,
                    "frame_time_ms": metric.frame_time_ms,
                    "latency_p50_ms": metric.latency_p50_ms,
                    "latency_p90_ms": metric.latency_p90_ms,
                    "latency_p99_ms": metric.latency_p99_ms,
                },
            )
            print(
                f"[{method_id}] PSNR={metric.psnr:.2f} SSIM={metric.ssim:.3f} "
                f"LPIPS={metric.lpips:.3f} FPS={metric.fps:.2f} VRAM={metric.vram_gb:.2f}"
            )

        save_metrics_snapshot(metrics, snapshot_file)
        logger.finish(
            "success",
            {
                "snapshot_file": snapshot_file,
                "methods_count": len(metrics),
                "skipped_methods": skipped_methods,
                "dataset_root": sample_dataset.root,
            },
        )
        print(f"Dataset de smoke: {sample_dataset.root}")
        if skipped_methods:
            print(f"Metodos pulados por hardware: {skipped_methods}")
        print(f"Snapshot salvo em: {snapshot_file}")
        print(f"Logs da execucao: {logger.run_dir}")
        return 0
    except Exception as exc:
        logger.finish("failed", {"error": str(exc)})
        print(f"Falha em metrics-check: {exc}")
        print(f"Logs da execucao: {logger.run_dir}")
        return 1


def run_metrics_compute(
    *,
    method: str,
    checkpoint: str,
    rendered_dir: str,
    dataset: str,
    root: str,
    reference_dir: str | None,
    split: str,
    snapshot_file: str,
    append_snapshot: bool,
    train_seconds: float,
    inference_seconds: float,
    log_dir: str,
    strict_results: bool,
    min_required_pairs: int,
    metrics_max_pairs: int | None,
    metrics_max_image_dim: int | None,
    metrics_log_every: int | None,
) -> int:
    """Recomputa métricas a partir de checkpoint e imagens renderizadas pré-existentes."""
    logger = RunLogger(
        command="metrics-compute",
        parameters={
            "method": method,
            "checkpoint": checkpoint,
            "rendered_dir": rendered_dir,
            "dataset": dataset,
            "root": root,
            "reference_dir": reference_dir,
            "split": split,
            "snapshot_file": snapshot_file,
            "append_snapshot": append_snapshot,
            "train_seconds": train_seconds,
            "inference_seconds": inference_seconds,
            "strict_results": strict_results,
            "min_required_pairs": min_required_pairs,
            "metrics_max_pairs": metrics_max_pairs,
            "metrics_max_image_dim": metrics_max_image_dim,
            "metrics_log_every": metrics_log_every,
        },
        log_dir=log_dir,
    )
    try:
        # Validar arquivos de entrada
        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint não encontrado: {checkpoint}")

        rendered_path = Path(rendered_dir)
        if not rendered_path.exists() or not rendered_path.is_dir():
            raise FileNotFoundError(f"Diretório de renders não encontrado: {rendered_dir}")

        # Contar frames PNG
        frames = len(list(rendered_path.glob("*.png")))
        if frames == 0:
            raise ValueError(f"Nenhum arquivo PNG encontrado em: {rendered_dir}")

        print(f"[{method}] Carregado:")
        print(f"  Checkpoint: {checkpoint}")
        print(f"  Renders: {rendered_dir} ({frames} frames)")
        print(f"  Train time: {train_seconds:.1f}s | Inference time: {inference_seconds:.1f}s")

        # Resolver referências
        if reference_dir:
            ref_path = Path(reference_dir)
        else:
            dataset_spec = load_dataset(dataset_name=dataset, root=root, split=split)
            ref_path = Path("./artifacts") / "metrics" / f"_references_{method}_{dataset_spec.name}_{split}"
            copied = export_reference_frames_from_dataset(
                root=dataset_spec.root,
                split=split,
                reference_dir=ref_path,
            )
            if copied == 0:
                write_reference_image(ref_path)

        if not ref_path.exists():
            raise FileNotFoundError(f"Diretório de referência não encontrado: {ref_path}")

        # Computar métricas
        print(
            f"\nComputando métricas... pairs<={metrics_max_pairs or 'all'} "
            f"max_dim={metrics_max_image_dim or 'full'} log_every={metrics_log_every or 'auto'}"
        )
        metric = evaluate_benchmark_metrics(
            method=method,
            pred_dir=rendered_path,
            ref_dir=ref_path,
            frames=frames,
            train_seconds=train_seconds,
            inference_seconds=inference_seconds,
            max_pairs=metrics_max_pairs,
            max_image_dim=metrics_max_image_dim,
            log_every=metrics_log_every,
        )

        # Validação de métricas
        if strict_results:
            validation_error = _validate_real_metrics(metric, min_required_pairs=min_required_pairs)
            if validation_error:
                raise RuntimeError(f"Falha na validação de métricas: {validation_error}")

        # Salvar snapshot
        _save_method_metric_snapshot(
            metric=metric,
            snapshot_file=snapshot_file,
            append=append_snapshot,
        )

        if strict_results:
            snapshot_validation = validate_snapshot_file(snapshot_file, must_exist=True)
            if not snapshot_validation.is_valid:
                raise RuntimeError(f"Falha ao validar snapshot: {snapshot_validation.message}")

        logger.event(
            "metrics_computed",
            {
                "method": method,
                "psnr": metric.psnr,
                "ssim": metric.ssim,
                "lpips": metric.lpips,
                "fps": metric.fps,
                "vram_gb": metric.vram_gb,
                "frame_time_ms": metric.frame_time_ms,
                "latency_p50_ms": metric.latency_p50_ms,
                "latency_p90_ms": metric.latency_p90_ms,
                "latency_p99_ms": metric.latency_p99_ms,
                "snapshot_file": snapshot_file,
            },
        )

        print(
            f"[{method}] PSNR={metric.psnr:.2f} SSIM={metric.ssim:.3f} "
            f"LPIPS={metric.lpips:.3f} FPS={metric.fps:.2f} VRAM={metric.vram_gb:.2f}"
        )
        print(f"Snapshot salvo em: {snapshot_file}")

        logger.finish("success")
        print("Métricas computadas com sucesso.")
        print(f"Logs da execução: {logger.run_dir}")
        return 0
    except Exception as exc:
        logger.finish("failed", {"error": str(exc)})
        print(f"Falha em metrics-compute: {exc}")
        print(f"Logs da execução: {logger.run_dir}")
        return 1


def _load_extra(extra_json: str | None, extra_file: str | None) -> dict:
    """Carrega configuração extra a partir de string JSON ou arquivo."""
    if extra_json and extra_file:
        raise ValueError("Use apenas --extra-json ou --extra-file, nao ambos.")
    if extra_file:
        return json.loads(Path(extra_file).read_text(encoding="utf-8"))
    if extra_json:
        return json.loads(extra_json)
    return {}


def _validate_real_metrics(metric: BenchmarkMetrics, min_required_pairs: int) -> str | None:
    """Valida se as métricas representam execução real e utilizável."""
    required_fields = {
        "pairs": metric.pairs,
        "psnr": metric.psnr,
        "ssim": metric.ssim,
        "lpips": metric.lpips,
        "fps": metric.fps,
        "vram_gb": metric.vram_gb,
        "train_seconds": metric.train_seconds,
        "inference_seconds": metric.inference_seconds,
        "frame_time_ms": metric.frame_time_ms,
        "latency_p50_ms": metric.latency_p50_ms,
        "latency_p90_ms": metric.latency_p90_ms,
        "latency_p99_ms": metric.latency_p99_ms,
    }
    non_finite = [name for name, value in required_fields.items() if not math.isfinite(float(value))]
    if non_finite:
        return f"Campos com valores nao finitos: {', '.join(non_finite)}"

    if metric.pairs < float(min_required_pairs):
        return f"Pairs insuficiente: {metric.pairs} < {min_required_pairs}"

    if metric.fps <= 0.0:
        return f"FPS invalido para resultado real: {metric.fps}"

    return None


def run_method_run(
    *,
    method_id: str,
    run_id: str | None,
    dataset: str,
    root: str,
    split: str,
    output_dir: str,
    log_dir: str,
    compute_metrics: bool,
    reference_dir: str | None,
    snapshot_file: str | None,
    append_snapshot: bool,
    extra_json: str | None,
    extra_file: str | None,
    preset: str | None = None,
    iterations: int | None = None,
    estimate_time_before_run: bool = False,
    cache_disable: bool = False,
    cache_max_size_gb: float = 50.0,
    reuse_renders: bool = False,
    reuse_metrics: bool = False,
    adaptive_preset: bool = False,
    validation_full: bool = False,
    validation_sample_size: int = 16,
    strict_results: bool = False,
    min_required_pairs: int = 1,
) -> int:
    """Executa um único método (incluindo external) em um dataset."""
    logger = RunLogger(
        command="method-run",
        parameters={
            "method": method_id,
            "run_id": run_id,
            "dataset": dataset,
            "root": root,
            "split": split,
            "output_dir": output_dir,
            "log_dir": log_dir,
            "compute_metrics": compute_metrics,
            "reference_dir": reference_dir,
            "snapshot_file": snapshot_file,
            "append_snapshot": append_snapshot,
            "estimate_time_before_run": estimate_time_before_run,
            "cache_disable": cache_disable,
            "cache_max_size_gb": cache_max_size_gb,
            "reuse_renders": reuse_renders,
            "reuse_metrics": reuse_metrics,
            "adaptive_preset": adaptive_preset,
            "validation_full": validation_full,
            "validation_sample_size": validation_sample_size,
            "strict_results": strict_results,
            "min_required_pairs": min_required_pairs,
        },
        log_dir=log_dir,
    )
    try:
        extra = _load_extra(extra_json, extra_file)
        if preset:
            extra["preset"] = preset
        if iterations is not None:
            extra["iterations"] = iterations
        extra["cache_enabled"] = not cache_disable
        extra["cache_max_size_gb"] = cache_max_size_gb
        extra["reuse_renders"] = reuse_renders
        extra["reuse_metrics"] = reuse_metrics
        extra["adaptive_preset"] = adaptive_preset

        dataset_validation = validate_dataset_integrity_preflight(
            dataset_name=dataset,
            root=root,
            split=split,
            full_scan=validation_full,
            sample_size=validation_sample_size,
        )
        if not dataset_validation.is_valid:
            print(dataset_validation.message)
            print_validation_result(dataset_validation, verbose=True)
            return 1

        hardware_info = estimate_execution_time(
            method_id=method_id,
            preset=preset,
            iterations=iterations,
            adaptive_preset=adaptive_preset,
        ).get("hardware", {})
        extra["detected_hardware"] = hardware_info

        if estimate_time_before_run:
            estimate = estimate_execution_time(
                method_id=method_id,
                preset=preset,
                iterations=iterations,
                adaptive_preset=adaptive_preset,
            )
            if estimate.get("estimate_available", True):
                print("Estimativa antes da execução:")
                print(format_time_estimate(estimate))
                if estimate.get("adaptive_preset"):
                    print(f"Iteracoes ajustadas (conservador): {estimate.get('adjusted_iterations')}")
                print()
        dataset_spec = load_dataset(dataset_name=dataset, root=root, split=split)
        run_id = _build_cli_run_id(method_id=method_id, dataset=dataset, root=root)
        print(f"Run ID resolvido: {run_id}")
        config = RunConfig(
            run_id=run_id,
            dataset=dataset_spec,
            method=method_id,
            output_dir=output_dir,
            log_dir=log_dir,
            hardware_profile=HardwareProfile.ADAPTIVE,
            extra=extra,
        )

        registry = build_registry_with_all_methods()
        orchestrator = Orchestrator(registry=registry)
        if method_id in registry.list_ids():
            pass
        elif method_id == "external":
            registry.register(ExternalMethodAdapter())
        else:
            raise ValueError(f"Metodo nao registrado: {method_id}")

        extra = dict(config.extra)
        extra["skip_reports"] = True
        extra["skip_snapshot"] = True
        if not compute_metrics:
            extra["skip_metrics"] = True
        if reference_dir:
            extra["reference_dir"] = reference_dir
        config = RunConfig(
            run_id=config.run_id,
            dataset=config.dataset,
            method=config.method,
            seed=config.seed,
            output_dir=config.output_dir,
            log_dir=config.log_dir,
            hardware_profile=config.hardware_profile,
            report_formats=[],
            extra=extra,
        )

        result = orchestrator.run(config)
        logger.event(
            "method_completed",
            {
                "method": method_id,
                "checkpoint": result.artifacts.checkpoint_path,
                "renders": result.artifacts.rendered_dir,
            },
        )

        if compute_metrics and result.metrics is not None:
            metric = result.metrics
            if strict_results:
                validation_error = _validate_real_metrics(metric, min_required_pairs=min_required_pairs)
                if validation_error:
                    raise RuntimeError(f"Falha na validacao de metricas reais: {validation_error}")
            if snapshot_file:
                _save_method_metric_snapshot(
                    metric=metric,
                    snapshot_file=snapshot_file,
                    append=append_snapshot,
                )
                if strict_results:
                    snapshot_validation = validate_snapshot_file(snapshot_file, must_exist=True)
                    if not snapshot_validation.is_valid:
                        raise RuntimeError(
                            f"Falha ao validar snapshot apos method-run: {snapshot_validation.message}"
                        )
            logger.event(
                "metrics_computed",
                {
                    "method": method_id,
                    "psnr": metric.psnr,
                    "ssim": metric.ssim,
                    "lpips": metric.lpips,
                    "fps": metric.fps,
                    "vram_gb": metric.vram_gb,
                    "frame_time_ms": metric.frame_time_ms,
                    "latency_p50_ms": metric.latency_p50_ms,
                    "latency_p90_ms": metric.latency_p90_ms,
                    "latency_p99_ms": metric.latency_p99_ms,
                    "snapshot_file": snapshot_file,
                },
            )
            print(
                f"[{method_id}] PSNR={metric.psnr:.2f} SSIM={metric.ssim:.3f} "
                f"LPIPS={metric.lpips:.3f} FPS={metric.fps:.2f} VRAM={metric.vram_gb:.2f}"
            )
            if snapshot_file:
                print(f"Snapshot atualizado em: {snapshot_file}")

        if compute_metrics and strict_results and result.metrics is None:
            raise RuntimeError("Falha na validacao de metricas reais: metodo nao retornou metricas.")

        logger.finish("success")
        print("Execucao concluida com sucesso.")
        print(f"Logs da execucao: {logger.run_dir}")
        return 0
    except Exception as exc:
        logger.finish("failed", {"error": str(exc)})
        print(f"Falha em method-run: {exc}")
        print(f"Logs da execucao: {logger.run_dir}")
        return 1


def run_install(catalog_file: str, only: str, execute: bool) -> int:
    """Instala datasets/modelos a partir do catalogo."""
    catalog = load_install_catalog(catalog_file)
    fatal_catalog_problem = any(
        note.startswith("Catalog not found:") or note.startswith("Invalid catalog JSON:")
        for note in catalog.notes
    )
    for note in catalog.notes:
        print(f"[note] {note}")
    messages = install_items(catalog=catalog, only=only, execute=execute)
    if not messages:
        print(f"[info] Nenhum item encontrado para instalacao com --only {only}.")
    for line in messages:
        print(line)
    return 1 if fatal_catalog_problem else 0


def _metric_from_snapshot_entry(method: str, payload: dict) -> BenchmarkMetrics:
    """Converte entrada de snapshot para BenchmarkMetrics."""
    return BenchmarkMetrics(
        method=method,
        pairs=float(payload.get("pairs", 0.0)),
        psnr=float(payload.get("psnr", 0.0)),
        ssim=float(payload.get("ssim", 0.0)),
        lpips=float(payload.get("lpips", 0.0)),
        fps=float(payload.get("fps", 0.0)),
        vram_gb=float(payload.get("vram_gb", 0.0)),
        train_seconds=float(payload.get("train_seconds", 0.0)),
        inference_seconds=float(payload.get("inference_seconds", 0.0)),
        frame_time_ms=float(payload.get("frame_time_ms", 0.0)),
        latency_p50_ms=float(payload.get("latency_p50_ms", 0.0)),
        latency_p90_ms=float(payload.get("latency_p90_ms", 0.0)),
        latency_p99_ms=float(payload.get("latency_p99_ms", 0.0)),
    )


def _save_method_metric_snapshot(metric: BenchmarkMetrics, snapshot_file: str, append: bool) -> None:
    """Salva uma métrica de método em snapshot, com merge opcional."""
    path = Path(snapshot_file)
    if not append or not path.exists():
        save_metrics_snapshot([metric], snapshot_file)
        return

    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    existing: list[BenchmarkMetrics] = []
    for method_name, method_payload in payload.items():
        if not isinstance(method_payload, dict):
            continue
        existing.append(_metric_from_snapshot_entry(method_name, method_payload))

    merged = {entry.method: entry for entry in existing}
    merged[metric.method] = metric
    save_metrics_snapshot(list(merged.values()), snapshot_file)


def run_report_generate(
    snapshot_file: str,
    output_dir: str,
    report_name: str,
    no_pdf: bool,
    log_dir: str,
    strict_snapshot: bool = False,
    expected_methods: str | None = None,
    min_methods: int | None = None,
    require_finite_metrics: bool = False,
    expected_matrix: str | None = None,
) -> int:
    """Gera relatório comparativo a partir de um snapshot."""
    logger = RunLogger(
        command="report-generate",
        parameters={
            "snapshot_file": snapshot_file,
            "output_dir": output_dir,
            "report_name": report_name,
            "no_pdf": no_pdf,
            "log_dir": log_dir,
            "strict_snapshot": strict_snapshot,
            "expected_methods": expected_methods,
            "min_methods": min_methods,
            "require_finite_metrics": require_finite_metrics,
            "expected_matrix": expected_matrix,
        },
        log_dir=log_dir,
    )
    try:
        logger.event("run_started")
        parsed_expected_methods: list[str] | None = None
        if expected_methods:
            parsed_expected_methods = [item.strip() for item in expected_methods.split(",") if item.strip()]

        if strict_snapshot and expected_matrix:
            matrix_result = validate_matrix_completeness(
                snapshot_file,
                expected_combos=[item.strip() for item in expected_matrix.split(",") if item.strip()],
                strict=True,
            )
            if not matrix_result.is_valid:
                print(matrix_result.message)
                print_validation_result(matrix_result, verbose=True)
                logger.finish("failed", {"error": matrix_result.message, "details": matrix_result.details})
                return 1

        result = generate_comparison_reports(
            snapshot_file=snapshot_file,
            output_dir=output_dir,
            report_name=report_name,
            generate_pdf=not no_pdf,
            strict_snapshot=strict_snapshot,
            expected_methods=parsed_expected_methods,
            min_methods=min_methods,
            require_finite_metrics=require_finite_metrics,
        )
        logger.finish("success", result)
        print(f"Vencedor geral (rank medio): {result['winner']}")
        print(f"Relatorio HTML: {result['html_path']}")
        if "pdf_path" in result:
            print(f"Relatorio PDF: {result['pdf_path']}")
        if "pdf_error" in result:
            print(f"Aviso ao gerar PDF: {result['pdf_error']}")
        print(f"Logs da execucao: {logger.run_dir}")
        return 0
    except Exception as exc:
        logger.finish("failed", {"error": str(exc)})
        print(f"Falha em report-generate: {exc}")
        print(f"Logs da execucao: {logger.run_dir}")
        return 1


def run_docs_check(source_dir: str, report_file: str, fail_on_missing: bool) -> int:
    """Audita docstrings públicas e salva relatório JSON."""
    missing = audit_docstrings(source_dir)
    save_doc_audit_report(missing, report_file)
    print(f"Relatorio de documentacao: {report_file}")
    print(f"Simbolos publicos sem docstring: {len(missing)}")
    if missing:
        preview = missing[:10]
        for item in preview:
            print(f"- {item.file} :: {item.symbol}")
        if len(missing) > len(preview):
            print(f"... e mais {len(missing) - len(preview)} item(ns)")
    if fail_on_missing and missing:
        return 1
    return 0


def run_standard_test(
    output_dir: str,
    log_dir: str,
    snapshot_file: str,
    report_name: str,
    run_unit_tests: bool,
) -> int:
    """Executa a suíte padrão de validação do benchmark."""
    logger = RunLogger(
        command="standard-test",
        parameters={
            "output_dir": output_dir,
            "log_dir": log_dir,
            "snapshot_file": snapshot_file,
            "report_name": report_name,
            "run_unit_tests": run_unit_tests,
        },
        log_dir=log_dir,
    )
    try:
        logger.event("run_started")

        steps = [
            ("contracts", lambda: run_contracts_check()),
            (
                "dataset-check",
                lambda: run_dataset_check(
                    dataset="blender_synthetic",
                    root=_resolve_smoke_dataset_spec().root,
                    split="train",
                ),
            ),
            ("methods-check", lambda: run_methods_check(output_dir=output_dir, log_dir=log_dir)),
            (
                "metrics-check",
                lambda: run_metrics_check(
                    output_dir=output_dir,
                    snapshot_file=snapshot_file,
                    log_dir=log_dir,
                ),
            ),
            (
                "report-generate",
                lambda: run_report_generate(
                    snapshot_file=snapshot_file,
                    output_dir=str(Path(output_dir) / "reports"),
                    report_name=report_name,
                    no_pdf=True,
                    log_dir=log_dir,
                ),
            ),
            (
                "docs-check",
                lambda: run_docs_check(
                    source_dir="./src/nvs_benchmark",
                    report_file="./artifacts/docs/latest_doc_audit.json",
                    fail_on_missing=False,
                ),
            ),
        ]

        for step_name, step in steps:
            logger.event("step_started", {"step": step_name})
            exit_code = step()
            logger.event("step_finished", {"step": step_name, "exit_code": exit_code})
            if exit_code != 0:
                logger.finish("failed", {"step": step_name, "exit_code": exit_code})
                print(f"Bateria padrao falhou na etapa: {step_name}")
                print(f"Logs da execucao: {logger.run_dir}")
                return exit_code

        if run_unit_tests:
            logger.event("step_started", {"step": "unit-tests"})
            completed = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], check=False)
            logger.event("step_finished", {"step": "unit-tests", "exit_code": completed.returncode})
            if completed.returncode != 0:
                logger.finish("failed", {"step": "unit-tests", "exit_code": completed.returncode})
                print("Bateria padrao falhou nos testes automaticos.")
                print(f"Logs da execucao: {logger.run_dir}")
                return completed.returncode

        logger.finish("success")
        print("Bateria padrao concluida com sucesso.")
        print(f"Logs da execucao: {logger.run_dir}")
        return 0
    except Exception as exc:
        logger.finish("failed", {"error": str(exc)})
        print(f"Falha em standard-test: {exc}")
        print(f"Logs da execucao: {logger.run_dir}")
        return 1


def run_list_datasets(check_files: bool) -> int:
    """Lista datasets disponíveis no projeto."""
    print_separator("=")
    print("📊 Datasets Disponíveis")
    print_separator("=")
    print()
    
    for dataset_name in SUPPORTED_DATASETS:
        print(f"• {dataset_name}")
    
    print()
    print(f"Total: {len(SUPPORTED_DATASETS)} datasets")
    print()
    
    if check_files:
        print("Verificando arquivos dos datasets...")
        print_separator("-")
        
        dataset_paths = {
            "blender_synthetic": "./data/blender_synthetic",
            "d_nerf": "./data/d_nerf",
            "custom": "./data/custom",
        }
        
        for dataset_name, path in dataset_paths.items():
            result = validate_dataset_path(dataset_name, path)
            print()
            print(f"{dataset_name}:")
            print_validation_result(result, verbose=True)
    
    return 0


def run_list_methods(detailed: bool) -> int:
    """Lista métodos disponíveis no projeto."""
    print_separator("=")
    print("🧠 Métodos Disponíveis")
    print_separator("=")
    print()
    
    registry = build_registry_with_all_methods()
    method_ids = registry.list_ids()
    
    for method_id in method_ids:
        method = registry.get(method_id)
        print(f"• {method_id}")
        
        if detailed:
            print(f"  Nome: {method.display_name}")
            print(f"  Treino: {'✓' if method.capabilities.supports_train else '✗'}")
            print(f"  Inferência: {'✓' if method.capabilities.supports_inference else '✗'}")
            print(f"  Dinâmico: {'✓' if method.capabilities.supports_dynamic_scene else '✗'}")
            print()
    
    print(f"Total: {len(method_ids)} métodos")
    return 0


def run_estimate_time(method_id: str, preset: str | None, iterations: int | None, adaptive_preset: bool) -> int:
    """Estima o tempo de execução para uma configuração."""
    print_separator("=")
    print("⏱️  Estimativa de Tempo de Execução")
    print_separator("=")
    print()
    
    estimate = estimate_execution_time(
        method_id=method_id,
        preset=preset,
        iterations=iterations,
        adaptive_preset=adaptive_preset,
    )
    
    if not estimate.get("estimate_available", True):
        print(f"❌ {estimate.get('message', 'Estimativa não disponível')}")
        return 1
    
    print(f"Método: {estimate['method']}")
    if "preset" in estimate:
        print(f"Preset: {estimate['preset']}")
    print(f"Iterações: {estimate['iterations']}")
    if adaptive_preset:
        print(f"Iterações ajustadas (conservador): {estimate.get('adjusted_iterations')}")
    hardware = estimate.get("hardware", {})
    if hardware:
        print(
            f"Hardware: gpu={hardware.get('has_gpu')} "
            f"vram_gb={hardware.get('vram_gb')} perfil={hardware.get('recommended_profile')}"
        )
    print()
    
    time_estimate = estimate.get("estimate_hours", {})
    if time_estimate:
        print(f"⏱️  {time_estimate['min']}-{time_estimate['max']} horas")
        print(f"   ({estimate['estimate_minutes']['min']}-{estimate['estimate_minutes']['max']} minutos)")
    print()
    
    print("ℹ️  Estas são estimativas para CPU. Com GPU, podem ser significativamente mais rápidas.")
    return 0


def run_validate_run(
    method_id: str,
    dataset: str,
    root: str,
    output_dir: str,
    preset: str | None,
    iterations: int | None,
    validation_full: bool,
    validation_sample_size: int,
    adaptive_preset: bool,
) -> int:
    """Valida configuração de execução sem rodar o método."""
    print_separator("=")
    print("🔍 Validação de Configuração")
    print_separator("=")
    print()
    
    all_valid = True
    
    # Validar método
    print("1. Validando método...")
    method_result = validate_method_available(method_id)
    print_validation_result(method_result, verbose=True)
    all_valid = all_valid and method_result.is_valid
    print()
    
    # Validar dataset
    print("2. Validando dataset...")
    dataset_result = validate_dataset_integrity_preflight(
        dataset_name=dataset,
        root=root,
        split="train",
        full_scan=validation_full,
        sample_size=validation_sample_size,
    )
    print_validation_result(dataset_result, verbose=True)
    all_valid = all_valid and dataset_result.is_valid
    print()
    
    # Validar output dir
    print("3. Validando diretório de saída...")
    output_result = validate_output_directory(output_dir)
    print_validation_result(output_result, verbose=True)
    all_valid = all_valid and output_result.is_valid
    print()
    
    # Estimativa de tempo
    if all_valid:
        print("4. Estimativa de tempo...")
        estimate = estimate_execution_time(
            method_id=method_id,
            preset=preset,
            iterations=iterations,
            adaptive_preset=adaptive_preset,
        )
        if estimate.get("estimate_available", True):
            print(f"⏱️  {estimate['estimate_hours']['min']}-{estimate['estimate_hours']['max']} horas")
        print()
    
    # Uso de disco
    if all_valid:
        print("5. Informações de disco...")
        disk_info = get_disk_usage(output_dir)
        if "error" not in disk_info:
            print(f"  Tamanho total do disco: {disk_info['total_disk_gb']} GB")
            print(f"  Espaço disponível: {disk_info['free_disk_gb']} GB")
        print()
    
    # Resumo
    print_separator("=")
    if all_valid:
        print("✓ Configuração válida! Pronto para executar.")
        print()
        print("Para rodar o benchmark, use:")
        print()
        cmd = f"  python -m nvs_benchmark.cli method-run \\"
        print(cmd)
        print(f"    --method {method_id} \\")
        print(f"    --dataset {dataset} \\")
        print(f"    --root {root} \\")
        if preset:
            print(f"    --preset {preset} \\")
        if iterations:
            print(f"    --iterations {iterations} \\")
        print(f"    --output-dir {output_dir} \\")
        print("    --compute-metrics")
        print()
        return 0
    else:
        print("✗ Configuração inválida. Corrija os erros acima e tente novamente.")
        return 1


def main() -> int:
    """Ponto de entrada principal da CLI."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "init":
        return run_init(args.config)
    if args.command == "status":
        return run_status()
    if args.command == "contracts":
        return run_contracts_check()
    if args.command == "dataset-check":
        return run_dataset_check(dataset=args.dataset, root=args.root, split=args.split)
    if args.command == "methods-check":
        return run_methods_check(output_dir=args.output_dir, log_dir=args.log_dir)
    if args.command == "metrics-check":
        return run_metrics_check(
            output_dir=args.output_dir,
            snapshot_file=args.snapshot_file,
            log_dir=args.log_dir,
        )
    if args.command == "metrics-compute":
        return run_metrics_compute(
            method=args.method,
            checkpoint=args.checkpoint,
            rendered_dir=args.rendered_dir,
            dataset=args.dataset,
            root=args.root,
            reference_dir=args.reference_dir,
            split=args.split,
            snapshot_file=args.snapshot_file,
            append_snapshot=args.append_snapshot,
            train_seconds=args.train_seconds,
            inference_seconds=args.inference_seconds,
            log_dir=args.log_dir,
            strict_results=args.strict_results,
            min_required_pairs=args.min_required_pairs,
            metrics_max_pairs=args.metrics_max_pairs,
            metrics_max_image_dim=args.metrics_max_image_dim,
            metrics_log_every=args.metrics_log_every,
        )
    if args.command == "install":
        return run_install(
            catalog_file=args.catalog_file,
            only=args.only,
            execute=args.execute,
        )
    if args.command == "report-generate":
        return run_report_generate(
            snapshot_file=args.snapshot_file,
            output_dir=args.output_dir,
            report_name=args.report_name,
            no_pdf=args.no_pdf,
            log_dir=args.log_dir,
            strict_snapshot=args.strict_snapshot,
            expected_methods=args.expected_methods,
            min_methods=args.min_methods,
            require_finite_metrics=args.require_finite_metrics,
            expected_matrix=args.expected_matrix,
        )
    if args.command == "docs-check":
        return run_docs_check(
            source_dir=args.source_dir,
            report_file=args.report_file,
            fail_on_missing=args.fail_on_missing,
        )
    if args.command == "standard-test":
        return run_standard_test(
            output_dir=args.output_dir,
            log_dir=args.log_dir,
            snapshot_file=args.snapshot_file,
            report_name=args.report_name,
            run_unit_tests=args.run_unit_tests,
        )
    if args.command == "method-run":
        return run_method_run(
            method_id=args.method,
            run_id=args.run_id,
            dataset=args.dataset,
            root=args.root,
            split=args.split,
            output_dir=args.output_dir,
            log_dir=args.log_dir,
            compute_metrics=args.compute_metrics,
            reference_dir=args.reference_dir,
            snapshot_file=args.snapshot_file,
            append_snapshot=args.append_snapshot,
            extra_json=args.extra_json,
            extra_file=args.extra_file,
            preset=args.preset,
            iterations=args.iterations,
            estimate_time_before_run=args.estimate_time,
            cache_disable=args.cache_disable,
            cache_max_size_gb=args.cache_max_size_gb,
            reuse_renders=args.reuse_renders,
            reuse_metrics=args.reuse_metrics,
            adaptive_preset=args.adaptive_preset,
            validation_full=args.validation_full,
            validation_sample_size=args.validation_sample_size,
            strict_results=args.strict_results,
            min_required_pairs=args.min_required_pairs,
        )
    if args.command == "presets-list":
        summaries = list_preset_summaries()
        if not summaries:
            print("Nenhum preset encontrado.")
            return 0
        print(f"{'Nome':<12} {'Label':<20} {'Descrição'}")
        print("-" * 70)
        for s in summaries:
            print(f"{s['name']:<12} {s['label']:<20} {s['description']}")
        return 0
    if args.command == "list-datasets":
        return run_list_datasets(check_files=args.check_files)
    if args.command == "list-methods":
        return run_list_methods(detailed=args.detailed)
    if args.command == "estimate-time":
        return run_estimate_time(
            method_id=args.method,
            preset=args.preset,
            iterations=args.iterations,
            adaptive_preset=args.adaptive_preset,
        )
    if args.command == "validate-run":
        return run_validate_run(
            method_id=args.method,
            dataset=args.dataset,
            root=args.root,
            output_dir=args.output_dir,
            preset=args.preset,
            iterations=args.iterations,
            validation_full=args.validation_full,
            validation_sample_size=args.validation_sample_size,
            adaptive_preset=args.adaptive_preset,
        )
    if args.command == "create-adapter":
        return run_create_adapter(
            name=args.name,
            output_dir=args.output_dir,
            kind=args.kind,
        )

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
