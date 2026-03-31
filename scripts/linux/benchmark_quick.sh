#!/bin/bash
set -e

METHOD="${1:-nerf_static}"
DATASET="${2:-blender_synthetic}"
ROOT="${3:-./data/blender_synthetic/nerf_synthetic/lego}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Project root is one level above scripts/
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"

# Preserve an already-active environment. If none is active, pick a local fallback.
if [ -n "${VIRTUAL_ENV:-}" ]; then
    echo "Using active virtual environment: $VIRTUAL_ENV"
elif [ -f ".venv-mx330-311/bin/activate" ]; then
    source .venv-mx330-311/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "❌ Virtual environment not found."
    echo "Expected one of:"
    echo "  - active shell venv (recommended)"
    echo "  - ./.venv-mx330-311"
    echo "  - ./venv"
    echo "Activate/create one and try again."
    exit 1
fi

echo "Python in use: $(python -c 'import sys; print(sys.executable)')"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║              Quick Benchmark (Single Method)               ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Method:  $METHOD"
echo "Dataset: $DATASET"
echo "Scene:   $ROOT"
echo ""

# Optional CPU-only mode. Leave GPU enabled by default.
if [ "${NVS_FORCE_CPU:-0}" = "1" ]; then
    export CUDA_VISIBLE_DEVICES=""
    echo "CPU-only mode enabled (NVS_FORCE_CPU=1)."
fi

# Resolver raiz do dataset
if [ ! -d "$ROOT" ]; then
    ALT_ROOT="./data/blender_synthetic/nerf_synthetic/lego"
    if [ -d "$ALT_ROOT" ]; then
        ROOT="$ALT_ROOT"
        echo "Using alternate path: $ROOT"
    else
        echo "Dataset not found at '$ROOT'. Run './scripts/download_dataset.sh' first."
        exit 1
    fi
fi

echo "🚀 Starting benchmark..."
echo ""

PRESET="${NVS_PRESET:-quick}"
CMD=(
    python -m nvs_benchmark.cli method-run
    --method "$METHOD"
    --dataset "$DATASET"
    --root "$ROOT"
    --split train
    --output-dir ./artifacts
    --log-dir ./logs
    --compute-metrics
    --snapshot-file ./artifacts/metrics/latest.json
    --append-snapshot
    --preset "$PRESET"
    --adaptive-preset
)

if [ -n "${NVS_EXTRA_JSON:-}" ]; then
    CMD+=(--extra-json "$NVS_EXTRA_JSON")
fi

"${CMD[@]}"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅ Benchmark complete!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Results saved to:"
echo "  - Metrics:  ./artifacts/metrics/latest.json"
echo "  - Logs:     ./logs/runs/"
echo ""
echo "Next steps:"
echo "  - View results:       bash ./scripts/preview.sh"
echo "  - Generate report:    bash ./scripts/generate_report.sh"
echo ""
