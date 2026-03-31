#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"

echo "[1/3] Upgrading pip..."
python -m pip install --upgrade pip

echo "[2/3] Installing nvs_benchmark package..."
python -m pip install -e .

echo "[3/3] Validating installation..."
python -m nvs_benchmark.cli status

echo ""
echo "✅ Setup complete (local Python environment)"
