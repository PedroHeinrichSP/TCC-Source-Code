#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -x "$PROJECT_ROOT/venv/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
else
  PYTHON_BIN="python"
fi

HOST="${1:-127.0.0.1}"
PORT="${2:-8765}"
METRICS_FILE="${3:-./artifacts/metrics/latest_preview.json}"
SCENE_FILE="${4:-./data/blender_synthetic/nerf_synthetic/lego/transforms_train.json}"

if [[ ! -f "$SCENE_FILE" ]]; then
  if [[ -f "./data/blender_synthetic/nerf_synthetic/lego/transforms_train.json" ]]; then
    SCENE_FILE="./data/blender_synthetic/nerf_synthetic/lego/transforms_train.json"
  elif [[ -f "./data/blender_synthetic/lego/lego/transforms_train.json" ]]; then
    SCENE_FILE="./data/blender_synthetic/lego/lego/transforms_train.json"
  else
    echo "Cena real nao encontrada. Execute ./scripts/setup_lego.ps1 (Windows) ou instale Blender Synthetic em ./data/blender_synthetic." >&2
    exit 1
  fi
fi

"$PYTHON_BIN" -m nvs_benchmark.cli ui-preview \
  --host "$HOST" \
  --port "$PORT" \
  --metrics-file "$METRICS_FILE" \
  --scene-transforms-file "$SCENE_FILE"
