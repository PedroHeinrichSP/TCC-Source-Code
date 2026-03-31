#!/bin/bash
set -e

HOST="${1:-127.0.0.1}"
PORT="${2:-8765}"
METRICS_FILE="${3:-./artifacts/metrics/latest.json}"

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
    exit 1
fi

echo "Python in use: $(python -c 'import sys; print(sys.executable)')"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║          3D Interactive Viewer (Viser)                     ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "🌐 Starting 3D viewer..."
echo ""
echo "Opening browser at: http://$HOST:$PORT"
echo ""
echo "Features:"
echo "  - Compare method results side-by-side"
echo "  - Rotate, zoom, pan the 3D scene"
echo "  - View rendered frames from each method"
echo "  - Check performance metrics (FPS, VRAM)"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Abrir navegador (funciona no macOS/Linux com xdg-open)
sleep 0.5
if command -v xdg-open &> /dev/null; then
    xdg-open "http://$HOST:$PORT" &
elif command -v open &> /dev/null; then
    open "http://$HOST:$PORT" &
fi

echo ""

python -m nvs_benchmark.cli ui-preview \
    --host "$HOST" \
    --port "$PORT" \
    --metrics-file "$METRICS_FILE"
