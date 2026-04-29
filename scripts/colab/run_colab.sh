#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -x "$PROJECT_ROOT/venv/bin/python" ]]; then
	PYTHON_BIN="$PROJECT_ROOT/venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
	PYTHON_BIN="python3"
else
	PYTHON_BIN="python"
fi

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -e .

"$PYTHON_BIN" -m nvs_benchmark.cli status
"$PYTHON_BIN" -m nvs_benchmark.cli methods-check --output-dir ./artifacts --log-dir ./logs
"$PYTHON_BIN" -m nvs_benchmark.cli metrics-check --output-dir ./artifacts --snapshot-file ./artifacts/metrics/latest_preview.json --log-dir ./logs
"$PYTHON_BIN" -m nvs_benchmark.cli report-generate --snapshot-file ./artifacts/metrics/latest_preview.json --output-dir ./artifacts/reports --report-name benchmark_report --log-dir ./logs --strict-snapshot --min-methods 1 --require-finite-metrics
