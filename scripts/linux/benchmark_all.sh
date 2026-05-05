#!/bin/bash
set -e

DATASET="${1:-blender_synthetic}"
ROOT="${2:-./data/blender_synthetic/nerf_synthetic/lego}"
OUTPUT_DIR="${3:-./artifacts}"
REPORT_NAME="${4:-benchmark_all}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "❌ Virtual environment not found at ./venv"
    echo "Run './scripts/setup.sh' first to create the environment."
    exit 1
fi

echo "╔════════════════════════════════════════════════════════════╗"
echo "║            Full Benchmark Suite (All Methods)              ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "Dataset: $DATASET"
echo "Scene:   $ROOT"
echo ""

if [ ! -d "$ROOT" ]; then
    echo "❌ Dataset not found at '$ROOT'"
    echo "Run './scripts/download_dataset.sh' first."
    exit 1
fi

METHODS=("nerf_static" "nerf_dynamic" "gs_static" "gs_dynamic")
SNAPSHOT_FILE="$OUTPUT_DIR/metrics/$REPORT_NAME.json"

mkdir -p "$(dirname "$SNAPSHOT_FILE")"

echo "⏱️  Starting benchmark suite (this may take a while)..."
echo ""

for METHOD in "${METHODS[@]}"; do
    echo "────────────────────────────────────────────────────────────"
    echo "Running $METHOD..."
    echo "────────────────────────────────────────────────────────────"
    
    # Try-catch pattern using subshell
    if python -m nvs_benchmark.cli method-run \
        --method "$METHOD" \
        --dataset "$DATASET" \
        --root "$ROOT" \
        --split train \
        --output-dir "$OUTPUT_DIR" \
        --log-dir ./logs \
        --compute-metrics \
        --snapshot-file "$SNAPSHOT_FILE" \
        --append-snapshot 2>&1; then
        echo "[✓] $METHOD completed"
    else
        echo "[⚠] $METHOD failed or skipped (possible missing GPU/dependencies)"
    fi
    echo ""
done

echo "════════════════════════════════════════════════════════════"
echo "✅ Benchmark suite complete!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "Results: $SNAPSHOT_FILE"
echo ""
echo "Next steps:"
echo "  - View results:       bash ./scripts/generate_report.sh"
echo "  - Generate report:    bash ./scripts/generate_report.sh"
echo ""
