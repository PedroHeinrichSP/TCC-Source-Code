#!/bin/bash
set -e

SNAPSHOT_FILE="${1:-./artifacts/metrics/latest.json}"
REPORT_NAME="${2:-benchmark_report}"
OUTPUT_DIR="${3:-./artifacts/reports}"

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
echo "║                Generate HTML Report                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

if [ ! -f "$SNAPSHOT_FILE" ]; then
    echo "Metrics file not found: $SNAPSHOT_FILE"
    echo "Run a benchmark first with: bash ./scripts/benchmark_quick.sh"
    exit 1
fi

echo "📊 Generating report from metrics..."
echo ""
echo "Input:  $SNAPSHOT_FILE"
echo "Output: $OUTPUT_DIR/$REPORT_NAME.html"
echo ""

python -m nvs_benchmark.cli report-generate \
    --snapshot-file "$SNAPSHOT_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --report-name "$REPORT_NAME" \
    --log-dir ./logs \
    --no-pdf

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅ Report generated successfully!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "📄 HTML Report:"
echo "   $OUTPUT_DIR/$REPORT_NAME.html"
echo ""

# Tentar abrir no navegador padrão
if command -v xdg-open &> /dev/null; then
    xdg-open "$OUTPUT_DIR/$REPORT_NAME.html" &
    echo "🌐 Opening report in browser..."
elif command -v open &> /dev/null; then
    open "$OUTPUT_DIR/$REPORT_NAME.html" &
    echo "🌐 Opening report in browser..."
else
    echo "💡 Open manually: $OUTPUT_DIR/$REPORT_NAME.html"
fi

echo ""
