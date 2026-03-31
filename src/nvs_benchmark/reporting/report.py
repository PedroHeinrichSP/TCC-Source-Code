"""Geração de relatório comparativo para resultados do benchmark NVS."""

from __future__ import annotations

import base64
import io
import json
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class MethodMetrics:
    """Métricas agregadas de um método a partir de snapshot do benchmark."""

    method: str
    psnr: float
    ssim: float
    lpips: float
    fps: float
    vram_gb: float
    train_seconds: float
    inference_seconds: float
    frame_time_ms: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p99_ms: float


@dataclass(frozen=True)
class MethodComparison:
    """Posições de ranking por métrica e rank médio de um método."""

    method: str
    metrics: MethodMetrics
    rank_psnr: int
    rank_ssim: int
    rank_lpips: int
    rank_fps: int
    rank_vram_gb: int
    rank_train_seconds: int
    rank_inference_seconds: int
    average_rank: float


def _to_float(payload: dict, key: str, default: float = 0.0) -> float:
    """Converte com segurança valor de payload JSON para float."""
    try:
        value = payload.get(key, default)
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _load_snapshot(snapshot_file: str | Path) -> list[MethodMetrics]:
    """Carrega snapshot JSON com métricas por método."""
    path = Path(snapshot_file)
    if not path.exists():
        raise FileNotFoundError(f"Snapshot de metricas nao encontrado: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Formato invalido de snapshot: esperado objeto JSON por metodo")

    methods: list[MethodMetrics] = []
    for method, values in payload.items():
        if not isinstance(values, dict):
            continue
        methods.append(
            MethodMetrics(
                method=str(method),
                psnr=_to_float(values, "psnr"),
                ssim=_to_float(values, "ssim"),
                lpips=_to_float(values, "lpips"),
                fps=_to_float(values, "fps"),
                vram_gb=_to_float(values, "vram_gb"),
                train_seconds=_to_float(values, "train_seconds"),
                inference_seconds=_to_float(values, "inference_seconds"),
                frame_time_ms=_to_float(values, "frame_time_ms"),
                latency_p50_ms=_to_float(values, "latency_p50_ms"),
                latency_p90_ms=_to_float(values, "latency_p90_ms"),
                latency_p99_ms=_to_float(values, "latency_p99_ms"),
            )
        )

    if not methods:
        raise ValueError("Snapshot de metricas vazio ou sem metodos validos")

    return methods


def _rank(methods: list[MethodMetrics], accessor, descending: bool) -> dict[str, int]:
    """Retorna mapeamento de ranking para o acessor de métrica."""
    sorted_methods = sorted(methods, key=accessor, reverse=descending)
    return {entry.method: index + 1 for index, entry in enumerate(sorted_methods)}


def _build_comparison(methods: list[MethodMetrics]) -> list[MethodComparison]:
    """Constrói comparação ranqueada considerando todas as métricas."""
    rank_psnr = _rank(methods, lambda item: item.psnr, descending=True)
    rank_ssim = _rank(methods, lambda item: item.ssim, descending=True)
    rank_lpips = _rank(methods, lambda item: item.lpips, descending=False)
    rank_fps = _rank(methods, lambda item: item.fps, descending=True)
    rank_vram_gb = _rank(methods, lambda item: item.vram_gb, descending=False)
    rank_train = _rank(methods, lambda item: item.train_seconds, descending=False)
    rank_infer = _rank(methods, lambda item: item.inference_seconds, descending=False)

    compared: list[MethodComparison] = []
    for method in methods:
        avg_rank = (
            rank_psnr[method.method]
            + rank_ssim[method.method]
            + rank_lpips[method.method]
            + rank_fps[method.method]
            + rank_vram_gb[method.method]
            + rank_train[method.method]
            + rank_infer[method.method]
        ) / 7.0

        compared.append(
            MethodComparison(
                method=method.method,
                metrics=method,
                rank_psnr=rank_psnr[method.method],
                rank_ssim=rank_ssim[method.method],
                rank_lpips=rank_lpips[method.method],
                rank_fps=rank_fps[method.method],
                rank_vram_gb=rank_vram_gb[method.method],
                rank_train_seconds=rank_train[method.method],
                rank_inference_seconds=rank_infer[method.method],
                average_rank=avg_rank,
            )
        )

    return sorted(compared, key=lambda item: (item.average_rank, item.method))


def _resolve_artifacts_root(snapshot_file: str | Path) -> Path:
    """Infere pasta raiz de artefatos com base no caminho do snapshot."""
    path = Path(snapshot_file).resolve()
    if path.exists():
        return path.parent.parent
    return Path("./artifacts").resolve()


def _find_method_image(artifacts_root: Path, method_id: str, kind: str) -> Path | None:
    """Localiza imagem de exemplo de render/referência de um método."""
    candidates = [
        artifacts_root / f"metrics-{method_id}" / method_id / kind / "frame_0000.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _encode_image_base64(path: Path | None) -> str | None:
    """Retorna data URI da imagem quando o caminho existir."""
    if not path or not path.exists():
        return None
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _plot_metric_chart(comparison: list[MethodComparison], key: str, title: str, ylabel: str) -> str | None:
    """Renderiza gráfico de barras de métrica e retorna como data URI."""
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception:
        return None

    methods = [item.method for item in comparison]
    raw_values = [getattr(item.metrics, key) for item in comparison]
    values = np.nan_to_num(np.array(raw_values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)

    fig, ax = plt.subplots(figsize=(4.4, 3.0), dpi=150)
    ax.bar(methods, values, color="#1f6f8b")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _render_chart_section(comparison: list[MethodComparison]) -> str:
    """Monta HTML de gráficos quando matplotlib estiver disponível."""
    charts = [
        ("psnr", "PSNR", "PSNR (dB)"),
        ("ssim", "SSIM", "SSIM"),
        ("lpips", "LPIPS", "LPIPS"),
        ("fps", "FPS", "Quadros por segundo"),
        ("vram_gb", "VRAM", "VRAM (GB)"),
        ("train_seconds", "Tempo de treino", "Segundos"),
        ("inference_seconds", "Tempo de inferência", "Segundos"),
    ]

    cards = []
    for key, title, ylabel in charts:
        chart_uri = _plot_metric_chart(comparison, key=key, title=title, ylabel=ylabel)
        if chart_uri:
            cards.append(
                "<div class='chart-card'>"
                f"<img src='{chart_uri}' alt='Chart {escape(title)}' />"
                "</div>"
            )

    if not cards:
        return ""

    return (
        "<h2>Gráficos</h2>"
        "<div class='charts'>"
        f"{''.join(cards)}"
        "</div>"
    )


def _render_visual_section(comparison: list[MethodComparison], artifacts_root: Path) -> str:
    """Monta cards HTML com renders e referências de exemplo."""
    cards = []
    for item in comparison:
        render_uri = _encode_image_base64(_find_method_image(artifacts_root, item.method, "renders"))
        ref_uri = _encode_image_base64(_find_method_image(artifacts_root, item.method, "references"))
        if not render_uri and not ref_uri:
            continue

        render_block = (
            f"<img src='{render_uri}' alt='Render {escape(item.method)}' />"
            if render_uri
            else "<p>Render não encontrado.</p>"
        )
        ref_block = (
            f"<img src='{ref_uri}' alt='Referência {escape(item.method)}' />"
            if ref_uri
            else "<p>Referência não encontrada.</p>"
        )
        cards.append(
            "<div class='visual-card'>"
            f"<strong>{escape(item.method)}</strong>"
            "<div>"
            f"{render_block}"
            "</div>"
            "<div>"
            f"{ref_block}"
            "</div>"
            "</div>"
        )

    if not cards:
        return ""

    return (
        "<h2>Comparação visual</h2>"
        "<div class='visual-grid'>"
        f"{''.join(cards)}"
        "</div>"
    )


def _render_html(comparison: list[MethodComparison], generated_at: str, snapshot_file: str | Path) -> str:
    """Renderiza relatório HTML como string única."""
    winner = comparison[0]
    artifacts_root = _resolve_artifacts_root(snapshot_file)
    charts_html = _render_chart_section(comparison)
    visuals_html = _render_visual_section(comparison, artifacts_root)

    raw_rows = []
    ranking_rows = []
    for row in comparison:
        raw_rows.append(
            "<tr>"
            f"<td>{escape(row.method)}</td>"
            f"<td>{row.metrics.psnr:.3f}</td>"
            f"<td>{row.metrics.ssim:.3f}</td>"
            f"<td>{row.metrics.lpips:.3f}</td>"
            f"<td>{row.metrics.fps:.3f}</td>"
            f"<td>{row.metrics.vram_gb:.3f}</td>"
            f"<td>{row.metrics.train_seconds:.3f}</td>"
            f"<td>{row.metrics.inference_seconds:.3f}</td>"
            f"<td>{row.metrics.frame_time_ms:.3f}</td>"
            f"<td>{row.metrics.latency_p50_ms:.3f}</td>"
            f"<td>{row.metrics.latency_p90_ms:.3f}</td>"
            f"<td>{row.metrics.latency_p99_ms:.3f}</td>"
            "</tr>"
        )
        ranking_rows.append(
            "<tr>"
            f"<td>{escape(row.method)}</td>"
            f"<td class='center'>{row.rank_psnr}</td>"
            f"<td class='center'>{row.rank_ssim}</td>"
            f"<td class='center'>{row.rank_lpips}</td>"
            f"<td class='center'>{row.rank_fps}</td>"
            f"<td class='center'>{row.rank_vram_gb}</td>"
            f"<td class='center'>{row.rank_train_seconds}</td>"
            f"<td class='center'>{row.rank_inference_seconds}</td>"
            f"<td class='center'>{row.average_rank:.2f}</td>"
            "</tr>"
        )

    return f"""
<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8" />
  <title>Relatório NVS Benchmark</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 28px; color: #202124; }}
    h1, h2 {{ margin: 0 0 12px 0; }}
    .meta {{ margin: 0 0 16px 0; color: #5f6368; font-size: 14px; }}
    .winner {{ margin: 0 0 22px 0; padding: 12px; border: 1px solid #d0d7de; border-radius: 8px; }}
    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 8px; text-align: left; font-size: 13px; }}
    th {{ background: #f6f8fa; }}
    .center {{ text-align: center; }}
    .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 24px; }}
    .chart-card {{ border: 1px solid #d0d7de; border-radius: 10px; padding: 12px; background: #fff; }}
    .chart-card img {{ width: 100%; height: auto; display: block; }}
    .visual-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; }}
    .visual-card {{ border: 1px solid #d0d7de; border-radius: 10px; padding: 12px; background: #fff; }}
    .visual-card img {{ width: 100%; height: auto; border-radius: 6px; margin-top: 8px; }}
  </style>
</head>
<body>
  <h1>Relatório de Comparação de Métodos</h1>
  <p class="meta">Gerado em: {escape(generated_at)}</p>
  <p class="meta">Snapshot: {escape(str(snapshot_file))}</p>

  <div class="winner">
    <strong>Método com melhor rank médio:</strong> {escape(winner.method)} (rank médio {winner.average_rank:.2f})
  </div>

  {charts_html}

  <h2>Métricas brutas</h2>
  <table>
    <thead>
      <tr>
        <th>Método</th>
        <th>PSNR</th>
        <th>SSIM</th>
        <th>LPIPS</th>
        <th>FPS</th>
        <th>VRAM (GB)</th>
        <th>Treino (s)</th>
        <th>Inferência (s)</th>
      </tr>
    </thead>
    <tbody>
      {''.join(raw_rows)}
    </tbody>
  </table>

  <h2>Ranking por métrica</h2>
  <table>
    <thead>
      <tr>
        <th>Método</th>
        <th class="center">PSNR</th>
        <th class="center">SSIM</th>
        <th class="center">LPIPS</th>
        <th class="center">FPS</th>
        <th class="center">VRAM</th>
        <th class="center">Treino</th>
        <th class="center">Inferência</th>
        <th class="center">Rank médio</th>
      </tr>
    </thead>
    <tbody>
      {''.join(ranking_rows)}
    </tbody>
  </table>

  {visuals_html}
</body>
</html>
"""


def _write_pdf_from_html(html_content: str, output_pdf: str | Path) -> tuple[bool, str | None]:
    """Tenta renderizar PDF a partir de HTML usando WeasyPrint."""
    try:
        from weasyprint import HTML

        HTML(string=html_content).write_pdf(str(output_pdf))
        return True, None
    except Exception as exc:
        return False, str(exc)


def generate_comparison_reports(
    *,
    snapshot_file: str | Path,
    output_dir: str | Path,
    report_name: str = "benchmark_report",
    generate_pdf: bool = True,
) -> dict[str, str]:
    """Gera relatórios comparativos HTML/PDF e retorna caminhos de saída."""
    methods = _load_snapshot(snapshot_file)
    comparison = _build_comparison(methods)

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_content = _render_html(comparison, generated_at=generated_at, snapshot_file=snapshot_file)
    html_path = root / f"{report_name}.html"
    html_path.write_text(html_content, encoding="utf-8")

    result = {
        "winner": comparison[0].method,
        "html_path": str(html_path),
    }

    if generate_pdf:
        pdf_path = root / f"{report_name}.pdf"
        ok, error_message = _write_pdf_from_html(html_content, pdf_path)
        if ok:
            result["pdf_path"] = str(pdf_path)
        else:
            result["pdf_error"] = error_message or "Falha desconhecida ao gerar PDF"

    return result


def generate_experiments_comparison_report(
    *,
    artifacts_dir: str | Path,
    run_ids: list[str],
    output_dir: str | Path,
    report_name: str = "experiments_comparison",
    generate_pdf: bool = False,
) -> dict[str, str]:
    """Generate a comparison report from selected experiment run IDs."""
    if not run_ids:
        raise ValueError("Lista de run_ids vazia.")

    artifacts_root = Path(artifacts_dir)
    history_file = artifacts_root / "experiments_history.json"
    if not history_file.exists():
        raise FileNotFoundError(f"Historico de experimentos nao encontrado: {history_file}")

    history_payload = json.loads(history_file.read_text(encoding="utf-8-sig"))
    experiments = history_payload.get("experiments", []) if isinstance(history_payload, dict) else []
    by_id = {str(item.get("run_id")): item for item in experiments if isinstance(item, dict)}

    selected: list[MethodMetrics] = []
    missing: list[str] = []
    for run_id in run_ids:
        row = by_id.get(run_id)
        if row is None:
            missing.append(run_id)
            continue

        method = str(row.get("method") or run_id)
        metrics_path = row.get("metrics_path")
        if isinstance(metrics_path, str) and metrics_path:
            candidate = Path(metrics_path)
            if not candidate.is_absolute():
                candidate = Path.cwd() / candidate
            if candidate.exists():
                try:
                    payload = json.loads(candidate.read_text(encoding="utf-8-sig"))
                    if isinstance(payload, dict) and payload:
                        method_payload = payload.get(method)
                        if isinstance(method_payload, dict):
                            selected.append(
                                MethodMetrics(
                                    method=f"{method} ({run_id})",
                                    psnr=_to_float(method_payload, "psnr"),
                                    ssim=_to_float(method_payload, "ssim"),
                                    lpips=_to_float(method_payload, "lpips"),
                                    fps=_to_float(method_payload, "fps"),
                                    vram_gb=_to_float(method_payload, "vram_gb"),
                                    train_seconds=_to_float(method_payload, "train_seconds"),
                                    inference_seconds=_to_float(method_payload, "inference_seconds"),
                                    frame_time_ms=_to_float(method_payload, "frame_time_ms"),
                                    latency_p50_ms=_to_float(method_payload, "latency_p50_ms"),
                                    latency_p90_ms=_to_float(method_payload, "latency_p90_ms"),
                                    latency_p99_ms=_to_float(method_payload, "latency_p99_ms"),
                                )
                            )
                            continue
                except Exception:
                    pass

        summary = row.get("metrics_summary") if isinstance(row.get("metrics_summary"), dict) else {}
        selected.append(
            MethodMetrics(
                method=f"{method} ({run_id})",
                psnr=_to_float(summary, "psnr"),
                ssim=_to_float(summary, "ssim"),
                lpips=_to_float(summary, "lpips"),
                fps=_to_float(summary, "fps"),
                vram_gb=0.0,
                train_seconds=0.0,
                inference_seconds=0.0,
                frame_time_ms=0.0,
                latency_p50_ms=0.0,
                latency_p90_ms=0.0,
                latency_p99_ms=0.0,
            )
        )

    if len(selected) < 2:
        raise ValueError("Comparacao requer ao menos 2 experimentos validos.")

    comparison = _build_comparison(selected)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html_content = _render_html(
        comparison,
        generated_at=generated_at,
        snapshot_file=f"experiments:{','.join(run_ids)}",
    )

    html_path = root / f"{report_name}.html"
    html_path.write_text(html_content, encoding="utf-8")
    result = {
        "winner": comparison[0].method,
        "html_path": str(html_path),
    }
    if missing:
        result["missing_runs"] = ",".join(missing)

    if generate_pdf:
        pdf_path = root / f"{report_name}.pdf"
        ok, error_message = _write_pdf_from_html(html_content, pdf_path)
        if ok:
            result["pdf_path"] = str(pdf_path)
        else:
            result["pdf_error"] = error_message or "Falha desconhecida ao gerar PDF"

    return result
