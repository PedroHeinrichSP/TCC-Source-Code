#!/bin/bash
set -e

METHOD="${1:-nerf_static}"
DATASET="${2:-blender_synthetic}"
ROOT="${3:-./data/blender_synthetic/nerf_synthetic/lego}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"

source venv/bin/activate 2>/dev/null || true

echo "╔════════════════════════════════════════════════════════════╗"
echo "║              Quick Benchmark (Single Method)               ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Method:  $METHOD"
echo "Dataset: $DATASET"
echo "Scene:   $ROOT"
echo ""

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

python -m nvs_benchmark.cli method-run \
    --method "$METHOD" \
    --dataset "$DATASET" \
    --root "$ROOT" \
    --split train \
    --output-dir ./artifacts \
    --log-dir ./logs \
    --compute-metrics \
    --snapshot-file ./artifacts/metrics/latest.json \
    --append-snapshot

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
