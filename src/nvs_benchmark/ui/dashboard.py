"""Dashboard web UI that embeds Viser as the 3D viewport."""

from __future__ import annotations

import json
import socket
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from datetime import datetime
from urllib.parse import parse_qs, urlparse
from urllib.parse import urlencode

from nvs_benchmark.core.experiments import ExperimentManager
from nvs_benchmark.install import install_item_by_id, load_install_catalog
from nvs_benchmark.reporting import generate_experiments_comparison_report
from nvs_benchmark.ui.preview import (
    capture_preview_camera_render,
    get_preview_runtime_state,
    nearest_scene_frame_index,
    refresh_preview_scene,
    run_preview_ui,
)


def _find_available_port(host: str, preferred_port: int, max_tries: int = 20) -> int:
    """Find an available TCP port, starting from preferred_port."""
    for offset in range(max_tries):
        candidate = preferred_port + offset
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex((host, candidate)) != 0:
                return candidate
    return preferred_port


def _to_web_path(path: Path, base_dir: Path) -> str:
    """Convert absolute path to dashboard-served web path."""
    rel = path.resolve().relative_to(base_dir.resolve())
    return "/" + rel.as_posix()


def _discover_scenes(base_dir: Path) -> list[dict]:
    """Discover scene transforms and frame files under data/ for quick testing."""
    data_root = base_dir / "data"
    if not data_root.exists():
        return []

    scenes: list[dict] = []
    transforms_files = sorted(data_root.rglob("transforms_*.json"))
    for transforms_file in transforms_files:
        try:
            payload = json.loads(transforms_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        frames = payload.get("frames", []) if isinstance(payload, dict) else []
        if not isinstance(frames, list):
            continue

        split_name = transforms_file.stem.replace("transforms_", "")
        scene_dir = transforms_file.parent
        scene_name = scene_dir.name
        scene_id = scene_dir.as_posix()

        split_frames: list[dict] = []
        for index, frame in enumerate(frames):
            if index >= 60:
                break
            if not isinstance(frame, dict):
                continue
            file_path = frame.get("file_path")
            if not isinstance(file_path, str) or not file_path:
                continue

            candidate = (scene_dir / file_path).resolve()
            candidates = [candidate]
            if candidate.suffix == "":
                candidates.extend([candidate.with_suffix(".png"), candidate.with_suffix(".jpg"), candidate.with_suffix(".jpeg")])

            image_path = None
            for option in candidates:
                if option.exists():
                    image_path = option
                    break
            if image_path is None:
                continue

            try:
                web_path = _to_web_path(image_path, base_dir)
            except Exception:
                continue

            split_frames.append(
                {
                    "label": f"{index:03d} - {image_path.name}",
                    "web_path": web_path,
                }
            )

        existing = next((item for item in scenes if item["id"] == scene_id), None)
        split_entry = {"name": split_name, "frames": split_frames}
        if existing is None:
            scenes.append(
                {
                    "id": scene_id,
                    "label": scene_name,
                    "transforms": _to_web_path(transforms_file, base_dir),
                    "splits": [split_entry],
                }
            )
        else:
            existing["splits"].append(split_entry)

    return scenes


def _first_image_in_dir(path: Path) -> Path | None:
    """Return first image candidate found in a directory tree."""
    if not path.exists() or not path.is_dir():
        return None
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        matches = sorted(path.rglob(pattern))
        if matches:
            return matches[0]
    return None


def _find_method_render_frame(base_dir: Path, method_id: str, frame_index: int | None) -> Path | None:
    """Find a render image for method, preferring nearest-frame index when available."""
    safe_method = method_id.strip()
    if not safe_method:
        return None

    candidates: list[Path] = []
    if frame_index is not None and frame_index >= 0:
        frame_name = f"frame_{int(frame_index):04d}.png"
        candidates.extend(
            [
                base_dir / "artifacts" / f"custom-{safe_method}" / safe_method / "renders" / frame_name,
                base_dir / "artifacts" / safe_method / "renders" / frame_name,
            ]
        )

    candidates.extend(
        [
            base_dir / "artifacts" / f"custom-{safe_method}" / safe_method / "renders" / "frame_0000.png",
            base_dir / "artifacts" / safe_method / "renders" / "frame_0000.png",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _parse_timestamp_day(value: object) -> str | None:
    """Normalize timestamp string to YYYY-MM-DD when possible."""
    if not isinstance(value, str) or not value.strip():
        return None

    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(text)
        return parsed.date().isoformat()
    except Exception:
        # Fallback for loosely formatted timestamps.
        if len(value) >= 10 and value[4] == "-" and value[7] == "-":
            return value[:10]
        return None


def _safe_float(value: object) -> float | None:
    """Convert arbitrary JSON value to finite float."""
    try:
        num = float(value)
    except Exception:
        return None
    if num != num:
        return None
    if num in (float("inf"), float("-inf")):
        return None
    return num


def _build_experiments_timeline(
    items: list[dict],
    *,
    metric: str,
    method: str | None,
    dataset: str | None,
    status: str | None,
    include_smoke: bool,
) -> dict:
    """Aggregate experiment metrics into daily temporal points."""
    supported = {"psnr", "ssim", "lpips", "fps"}
    metric_name = metric if metric in supported else "psnr"

    grouped: dict[str, list[float]] = {}
    methods_seen: set[str] = set()

    for row in items:
        run_id = str(row.get("run_id") or "")
        row_method = str(row.get("method") or "")
        row_dataset = str(row.get("dataset") or "")
        row_status = str(row.get("status") or "")

        if not include_smoke and (run_id.startswith("smoke-") or run_id.startswith("metrics-")):
            continue
        if method and row_method != method:
            continue
        if dataset and row_dataset != dataset:
            continue
        if status and row_status != status:
            continue

        timestamp = row.get("timestamp")
        day = _parse_timestamp_day(timestamp)
        if not day:
            continue

        summary = row.get("metrics_summary")
        metric_value = None
        if isinstance(summary, dict):
            metric_value = _safe_float(summary.get(metric_name))
        if metric_value is None:
            continue

        grouped.setdefault(day, []).append(metric_value)
        if row_method:
            methods_seen.add(row_method)

    points = []
    for day in sorted(grouped.keys()):
        values = grouped[day]
        avg = sum(values) / len(values)
        points.append({"date": day, "value": round(avg, 6), "count": len(values)})

    latest_delta = None
    if len(points) >= 2:
        previous = points[-2]
        latest = points[-1]
        latest_delta = {
            "previous_date": previous["date"],
            "latest_date": latest["date"],
            "delta": round(float(latest["value"]) - float(previous["value"]), 6),
        }

    return {
        "metric": metric_name,
        "methods": sorted(methods_seen),
        "points": points,
        "latest_delta": latest_delta,
    }


class _DashboardHandler(SimpleHTTPRequestHandler):
    """Serve static assets and a tiny JSON API for installs."""

    def __init__(self, *args, base_dir: Path, catalog_path: Path, preview_port: int, **kwargs):
        self._base_dir = base_dir
        self._catalog_path = catalog_path
        self._preview_port = int(preview_port)
        super().__init__(*args, directory=str(base_dir), **kwargs)

    def _json_response(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path

        if route in {"/", "/index.html"}:
            self.send_response(302)
            self.send_header("Location", "/web_ui/index.html")
            self.end_headers()
            return
        if route == "/api/catalog":
            catalog = load_install_catalog(self._catalog_path)
            self._json_response(
                {
                    "datasets": [
                        {**item.__dict__, "installed": Path(item.path).exists() if item.path else False}
                        for item in catalog.datasets
                    ],
                    "methods": [
                        {**item.__dict__, "installed": Path(item.path).exists() if item.path else False}
                        for item in catalog.methods
                    ],
                    "notes": catalog.notes,
                }
            )
            return
        if route == "/api/scenes":
            self._json_response({"scenes": _discover_scenes(self._base_dir)})
            return
        if route == "/api/experiments":
            manager = ExperimentManager(output_dir=self._base_dir / "artifacts")
            query = parse_qs(parsed.query)
            method = query.get("method", [None])[0]
            dataset = query.get("dataset", [None])[0]
            status = query.get("status", [None])[0]
            include_smoke_raw = query.get("include_smoke", ["false"])[0]
            include_smoke = str(include_smoke_raw).strip().lower() in {"1", "true", "yes", "on"}
            limit_raw = query.get("limit", [""])[0]
            limit = int(limit_raw) if limit_raw.isdigit() else None
            items = manager.list(method=method, dataset=dataset, status=status, limit=limit)

            if not include_smoke:
                filtered_items: list[dict] = []
                for item in items:
                    run_id = str(item.get("run_id") or "")
                    if run_id.startswith("smoke-") or run_id.startswith("metrics-"):
                        continue
                    filtered_items.append(item)
                items = filtered_items

            enriched: list[dict] = []
            for item in reversed(items):
                row = dict(item)
                rendered_dir = row.get("rendered_dir")
                if isinstance(rendered_dir, str) and rendered_dir:
                    candidate = Path(rendered_dir)
                    if not candidate.is_absolute():
                        candidate = self._base_dir / candidate
                    preview = _first_image_in_dir(candidate)
                    if preview is not None:
                        try:
                            row["render_preview_url"] = _to_web_path(preview, self._base_dir)
                        except Exception:
                            row["render_preview_url"] = None
                    else:
                        row["render_preview_url"] = None
                else:
                    row["render_preview_url"] = None
                enriched.append(row)
            self._json_response({"experiments": enriched})
            return
        if route == "/api/experiments-timeline":
            manager = ExperimentManager(output_dir=self._base_dir / "artifacts")
            query = parse_qs(parsed.query)
            metric = str(query.get("metric", ["psnr"])[0] or "psnr")
            method = query.get("method", [None])[0]
            dataset = query.get("dataset", [None])[0]
            status = query.get("status", [None])[0]
            include_smoke_raw = query.get("include_smoke", ["false"])[0]
            include_smoke = str(include_smoke_raw).strip().lower() in {"1", "true", "yes", "on"}

            rows = manager.list(limit=None)
            timeline = _build_experiments_timeline(
                rows,
                metric=metric,
                method=method,
                dataset=dataset,
                status=status,
                include_smoke=include_smoke,
            )
            self._json_response(timeline)
            return
        if route == "/api/preview-state":
            payload = get_preview_runtime_state(self._preview_port)
            self._json_response(payload)
            return
        if route == "/api/training-status":
            manager = ExperimentManager(output_dir=self._base_dir / "artifacts")
            rows = manager.list(limit=1)
            latest = rows[-1] if rows else None

            metrics_file = self._base_dir / "artifacts" / "metrics" / "latest_preview.json"
            metrics_mtime = None
            if metrics_file.exists():
                try:
                    metrics_mtime = metrics_file.stat().st_mtime
                except Exception:
                    metrics_mtime = None

            checkpoint_exists = False
            checkpoint_path = None
            if isinstance(latest, dict):
                raw_checkpoint = latest.get("checkpoint_path")
                if isinstance(raw_checkpoint, str) and raw_checkpoint:
                    checkpoint = Path(raw_checkpoint)
                    if not checkpoint.is_absolute():
                        checkpoint = (self._base_dir / checkpoint).resolve()
                    checkpoint_exists = checkpoint.exists()
                    checkpoint_path = str(checkpoint)

            self._json_response(
                {
                    "latest_run": latest,
                    "metrics_mtime": metrics_mtime,
                    "checkpoint": {
                        "path": checkpoint_path,
                        "exists": checkpoint_exists,
                    },
                    "preview": get_preview_runtime_state(self._preview_port),
                }
            )
            return
        return super().do_GET()

    def list_directory(self, path: str):  # type: ignore[override]
        """Disable directory listing."""
        self.send_error(404, "File not found")
        return None

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        route = parsed.path

        if route == "/api/report-compare":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                self._json_response({"error": "invalid-json"}, status=400)
                return

            run_ids = payload.get("run_ids")
            if not isinstance(run_ids, list):
                self._json_response({"error": "run_ids deve ser lista"}, status=400)
                return

            generate_pdf = bool(payload.get("generate_pdf", False))
            report_name = str(payload.get("report_name") or "experiments_comparison")
            try:
                result = generate_experiments_comparison_report(
                    artifacts_dir=self._base_dir / "artifacts",
                    run_ids=[str(item) for item in run_ids],
                    output_dir=self._base_dir / "artifacts" / "reports",
                    report_name=report_name,
                    generate_pdf=generate_pdf,
                )
                html_path = result.get("html_path")
                if isinstance(html_path, str):
                    html_file = Path(html_path)
                    if html_file.exists():
                        result["html_path"] = _to_web_path(html_file, self._base_dir)

                pdf_path = result.get("pdf_path")
                if isinstance(pdf_path, str):
                    pdf_file = Path(pdf_path)
                    if pdf_file.exists():
                        result["pdf_path"] = _to_web_path(pdf_file, self._base_dir)
                self._json_response(result)
            except Exception as exc:
                self._json_response({"error": str(exc)}, status=400)
            return

        if route == "/api/render-camera":
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                self._json_response({"ok": False, "error": "invalid-json"}, status=400)
                return

            width = int(payload.get("width") or 960)
            height = int(payload.get("height") or 540)
            quality = str(payload.get("quality") or "manual").strip().lower()
            method_id = str(payload.get("method_id") or "").strip()
            preview_mode = str(payload.get("preview_mode") or "auto").strip().lower()

            nearest_idx = nearest_scene_frame_index(self._preview_port)
            if preview_mode != "viser":
                artifact_path = _find_method_render_frame(self._base_dir, method_id, nearest_idx)
                if artifact_path is not None:
                    try:
                        web_path = _to_web_path(artifact_path, self._base_dir)
                    except Exception:
                        self._json_response({"ok": False, "error": "render-path-outside-base"}, status=500)
                        return
                    self._json_response(
                        {
                            "ok": True,
                            "image_url": web_path,
                            "camera": get_preview_runtime_state(self._preview_port).get("camera") or {},
                            "quality": quality,
                            "source": "artifact",
                            "frame_index": nearest_idx,
                            "preview_mode": preview_mode,
                        }
                    )
                    return

            if preview_mode == "artifact":
                self._json_response(
                    {
                        "ok": False,
                        "error": "artifact-not-found",
                        "preview_mode": preview_mode,
                        "frame_index": nearest_idx,
                    },
                    status=409,
                )
                return

            output_name = "live_camera_render.jpg" if quality == "manual" else "live_camera_preview.jpg"

            capture = capture_preview_camera_render(
                port=self._preview_port,
                width=width,
                height=height,
                output_name=output_name,
            )
            if not capture.get("ok"):
                self._json_response(capture, status=409)
                return

            image_path = Path(str(capture.get("path")))
            try:
                web_path = _to_web_path(image_path, self._base_dir)
            except Exception:
                self._json_response({"ok": False, "error": "render-path-outside-base"}, status=500)
                return

            self._json_response(
                {
                    "ok": True,
                    "image_url": web_path,
                    "camera": capture.get("camera") or {},
                    "width": capture.get("width"),
                    "height": capture.get("height"),
                    "quality": quality,
                    "source": "viser_capture",
                    "preview_mode": preview_mode,
                }
            )
            return

        if route == "/api/preview-refresh":
            result = refresh_preview_scene(self._preview_port)
            status = 200 if result.get("ok") else 409
            self._json_response(result, status=status)
            return

        if route != "/api/install":
            return super().do_POST()

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._json_response({"error": "invalid-json"}, status=400)
            return

        item_id = str(payload.get("item_id") or "")
        if not item_id:
            self._json_response({"error": "missing-item-id"}, status=400)
            return

        catalog = load_install_catalog(self._catalog_path)
        try:
            messages = install_item_by_id(catalog=catalog, item_id=item_id, execute=True)
            self._json_response({"messages": messages})
        except Exception as exc:  # pragma: no cover - runtime errors
            self._json_response({"error": str(exc)}, status=500)


def _start_dashboard_server(host: str, port: int, base_dir: Path, catalog_path: Path, preview_port: int) -> ThreadingHTTPServer:
    handler = lambda *args, **kwargs: _DashboardHandler(  # noqa: E731
        *args, base_dir=base_dir, catalog_path=catalog_path, preview_port=preview_port, **kwargs
    )
    server = ThreadingHTTPServer((host, port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def run_dashboard_ui(
    *,
    host: str = "127.0.0.1",
    dashboard_port: int = 8780,
    viser_port: int = 8765,
    open_browser: bool = True,
    metrics_file: str | None = "./artifacts/metrics/latest_preview.json",
    scene_transforms_file: str | None = "./data/blender_synthetic/transforms_train.json",
    install_catalog_file: str | None = "./configs/install_catalog.json",
) -> None:
    """Launch dashboard UI and Viser viewport side-by-side."""
    base_dir = Path(__file__).resolve().parents[3]
    catalog_path = Path(install_catalog_file or "./configs/install_catalog.json").resolve()
    resolved_viser_port = _find_available_port(host, viser_port)

    _start_dashboard_server(host, dashboard_port, base_dir, catalog_path, resolved_viser_port)

    metrics_web_path = "/artifacts/metrics/latest_preview.json"
    if metrics_file:
        try:
            resolved_metrics = Path(metrics_file)
            if not resolved_metrics.is_absolute():
                resolved_metrics = (base_dir / resolved_metrics).resolve()
            if resolved_metrics.exists():
                metrics_web_path = _to_web_path(resolved_metrics, base_dir)
        except Exception:
            metrics_web_path = "/artifacts/metrics/latest_preview.json"

    # Usar caminhos relativos à raiz web (sempre a partir de base_dir)
    # Os caminhos devem ser relativos para que o cliente JavaScript possa usá-los como URLs HTTP
    query = urlencode(
        {
            "viser_port": str(resolved_viser_port),
            "metrics_file": metrics_web_path,
            "artifacts_root": "/artifacts",
        }
    )
    dashboard_url = f"http://{host}:{dashboard_port}/web_ui/index.html?{query}"
    print(f"Dashboard completo disponivel em: {dashboard_url}")
    if open_browser:
        webbrowser.open(dashboard_url)

    # Start Viser (widget-only) without opening browser; the dashboard will embed it.
    run_preview_ui(
        host=host,
        port=resolved_viser_port,
        open_browser=False,
        metrics_file=metrics_file,
        scene_transforms_file=scene_transforms_file,
        install_catalog_file=install_catalog_file,
        minimal=True,
    )
