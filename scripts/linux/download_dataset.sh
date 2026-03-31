#!/bin/bash
set -e

DATASET="${1:-blender_synthetic}"
CATALOG_FILE="${2:-./configs/install_catalog.json}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"

source venv/bin/activate 2>/dev/null || true

echo "╔════════════════════════════════════════════════════════════╗"
echo "║              Download / Setup Dataset                      ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

echo "Available datasets:"
echo "  • blender_synthetic : Synthetic scenes (Lego, Chair, etc.) - ~500MB"
echo "  • d_nerf : Dynamic scenes (human motion) - ~1GB"
echo ""

if [[ ! "$DATASET" =~ ^(blender_synthetic|d_nerf)$ ]]; then
    echo "⚠️  Invalid dataset: $DATASET"
    exit 1
fi

echo "📥 Downloading $DATASET..."
echo ""

python -m nvs_benchmark.cli install \
    --catalog-file "$CATALOG_FILE" \
    --only datasets \
    --execute

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅ Dataset downloaded successfully!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "✓ Dataset location: ./data/$DATASET"
echo "✓ Ready for benchmarking!"
echo ""
echo "Next step: Run a benchmark"
echo "  bash ./scripts/benchmark_quick.sh"
echo ""
