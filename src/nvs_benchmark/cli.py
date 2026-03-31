"""Interface de linha de comando para operações do benchmark NVS."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from nvs_benchmark.core import DatasetSpec, HardwareProfile, RunConfig
from nvs_benchmark.core import Orchestrator
from nvs_benchmark.core import InferenceRequest, TrainRequest
from nvs_benchmark.core.presets import PRESET_NAMES, list_preset_summaries
from nvs_benchmark.data import SUPPORTED_DATASETS, load_dataset, validate_dataset
from nvs_benchmark.evaluation import BenchmarkMetrics, save_metrics_snapshot
from nvs_benchmark.methods import ExternalMethodAdapter, build_registry_with_all_methods
from nvs_benchmark.reporting import generate_comparison_reports
from nvs_benchmark.runtime import RunLogger, audit_docstrings, save_doc_audit_report
from nvs_benchmark.ui import run_preview_ui, run_dashboard_ui
from nvs_benchmark.install import load_install_catalog, install_items
from nvs_benchmark.cli_extensions import (
    validate_dataset_path,
    validate_dataset_integrity_preflight,
    validate_output_directory,
    validate_method_available,
    validate_snapshot_file,
    estimate_execution_time,
    print_validation_result,
    print_separator,
    format_time_estimate,
    get_disk_usage,
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

    ui_parser = subparsers.add_parser("ui-preview", help="Abrir a UI de preview em Viser")
    ui_parser.add_argument("--host", default="127.0.0.1", help="Host")
    ui_parser.add_argument("--port", type=int, default=8765, help="Porta")
    ui_parser.add_argument("--no-browser", action="store_true", help="Não abrir navegador")
    ui_parser.add_argument(
        "--metrics-file",
        default="./artifacts/metrics/latest_preview.json",
        help="Arquivo JSON de métricas",
    )
    ui_parser.add_argument(
        "--scene-transforms-file",
        default="./data/blender_synthetic/transforms_train.json",
        help="Arquivo de transforms para frustums da cena",
    )
    ui_parser.add_argument(
        "--install-catalog-file",
        default="./configs/install_catalog.json",
        help="Catalogo JSON de downloads para datasets e modelos",
    )
    ui_parser.add_argument(
        "--dashboard-port",
        type=int,
        default=8780,
        help="Porta do dashboard web",
    )
    ui_parser.add_argument(
        "--viser-port",
        type=int,
        default=8765,
        help="Porta do Viser (iframe)",
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

    return parser


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

        sample_dataset = DatasetSpec(name="blender_synthetic", root="./data/_smoke/blender")
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
            method.validate_config(config)

            train_result = method.train(TrainRequest(config=config))
            infer_result = method.infer(
                InferenceRequest(config=config, checkpoint_path=train_result.checkpoint_path, split="test")
            )

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
            print(f"[{method_id}] checkpoint: {train_result.checkpoint_path}")
            print(f"[{method_id}] renders: {infer_result.rendered_dir}")

        logger.finish("success", {"methods_count": len(method_ids)})
        print("Integracao dos metodos validada com sucesso.")
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
        sample_dataset = DatasetSpec(name="blender_synthetic", root="./data/_smoke/blender")
        metrics = []
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

            result = orchestrator.run(config)
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
        logger.finish("success", {"snapshot_file": snapshot_file, "methods_count": len(metrics)})
        print(f"Snapshot salvo em: {snapshot_file}")
        print(f"Logs da execucao: {logger.run_dir}")
        return 0
    except Exception as exc:
        logger.finish("failed", {"error": str(exc)})
        print(f"Falha em metrics-check: {exc}")
        print(f"Logs da execucao: {logger.run_dir}")
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


def run_method_run(
    *,
    method_id: str,
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
) -> int:
    """Executa um único método (incluindo external) em um dataset."""
    logger = RunLogger(
        command="method-run",
        parameters={
            "method": method_id,
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
        config = RunConfig(
            run_id=f"custom-{method_id}",
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
            if snapshot_file:
                _save_method_metric_snapshot(
                    metric=metric,
                    snapshot_file=snapshot_file,
                    append=append_snapshot,
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
    messages = install_items(catalog=catalog, only=only, execute=execute)
    for line in messages:
        print(line)
    return 0


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


def run_ui_preview(host: str, port: int, no_browser: bool, metrics_file: str, scene_transforms_file: str, install_catalog_file: str, dashboard_port: int, viser_port: int) -> int:
    """Inicia UI Viser para métricas e pré-visualização 3D da cena."""
    effective_dashboard_port = dashboard_port
    if port != 8765 and dashboard_port == 8780:
        effective_dashboard_port = port

    run_dashboard_ui(
        host=host,
        dashboard_port=effective_dashboard_port,
        viser_port=viser_port,
        open_browser=not no_browser,
        metrics_file=metrics_file,
        scene_transforms_file=scene_transforms_file,
        install_catalog_file=install_catalog_file,
    )
    return 0


def run_report_generate(snapshot_file: str, output_dir: str, report_name: str, no_pdf: bool, log_dir: str) -> int:
    """Gera relatório comparativo a partir de um snapshot."""
    logger = RunLogger(
        command="report-generate",
        parameters={
            "snapshot_file": snapshot_file,
            "output_dir": output_dir,
            "report_name": report_name,
            "no_pdf": no_pdf,
            "log_dir": log_dir,
        },
        log_dir=log_dir,
    )
    try:
        logger.event("run_started")
        result = generate_comparison_reports(
            snapshot_file=snapshot_file,
            output_dir=output_dir,
            report_name=report_name,
            generate_pdf=not no_pdf,
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
                lambda: run_dataset_check(dataset="blender_synthetic", root="./data/_smoke/blender", split="train"),
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
    if args.command == "install":
        return run_install(
            catalog_file=args.catalog_file,
            only=args.only,
            execute=args.execute,
        )
    if args.command == "ui-preview":
        return run_ui_preview(
            host=args.host,
            port=args.port,
            no_browser=args.no_browser,
            metrics_file=args.metrics_file,
            scene_transforms_file=args.scene_transforms_file,
            install_catalog_file=args.install_catalog_file,
            dashboard_port=args.dashboard_port,
            viser_port=args.viser_port,
        )
    if args.command == "report-generate":
        return run_report_generate(
            snapshot_file=args.snapshot_file,
            output_dir=args.output_dir,
            report_name=args.report_name,
            no_pdf=args.no_pdf,
            log_dir=args.log_dir,
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

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
