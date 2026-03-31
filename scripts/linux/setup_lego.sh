#!/bin/bash
set -e

CATALOG_FILE="${1:-./configs/install_catalog.json}"
LEGO_ROOT="${2:-./data/blender_synthetic/nerf_synthetic/lego}"

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
echo "║            Setup Lego Scene (Blender Synthetic)            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

echo "[1/3] Installing Blender synthetic dataset via catalog..."
if ! python -m nvs_benchmark.cli install \
    --catalog-file "$CATALOG_FILE" \
    --only datasets \
    --execute; then
    echo "❌ Failed to install dataset"
    exit 1
fi

LEGO_PATH="$PROJECT_ROOT/$LEGO_ROOT"
LEGACY_LEGO_PATH="$PROJECT_ROOT/data/blender_synthetic/lego"

if [ ! -d "$LEGO_PATH" ] && [ -d "$LEGACY_LEGO_PATH" ]; then
    LEGO_PATH="$LEGACY_LEGO_PATH"
fi

if [ ! -d "$LEGO_PATH" ]; then
    echo "❌ Lego scene not found. Check extraction at ./data/blender_synthetic"
    exit 1
fi

echo ""
echo "[2/3] Validating Lego dataset structure at: $LEGO_PATH"
if ! python -m nvs_benchmark.cli dataset-check \
    --dataset blender_synthetic \
    --root "$LEGO_PATH" \
    --split train; then
    echo "❌ Dataset validation failed"
    exit 1
fi

echo ""
echo "[3/3] Setup complete."
echo "✓ Lego scene root: $LEGO_PATH"
echo "Use this root in subsequent commands:"
echo "  bash ./scripts/benchmark_quick.sh nerf_static blender_synthetic \"$LEGO_PATH\""
echo ""
