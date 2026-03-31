#!/bin/bash
set -e

HOST="${1:-127.0.0.1}"
PORT="${2:-8765}"
METRICS_FILE="${3:-./artifacts/metrics/latest.json}"

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
