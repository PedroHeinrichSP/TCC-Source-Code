#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║         NVS Benchmark - Environment Setup                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Passo 1: Criar ambiente virtual
echo "📦 Step 1/3: Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Ativar venv
source venv/bin/activate

# Passo 2: Instalar dependências
echo ""
echo "📥 Step 2/3: Installing dependencies..."
python -m pip install --upgrade pip > /dev/null 2>&1
python -m pip install -e . > /dev/null 2>&1
echo "✓ Dependencies installed"

# Passo 3: Validar a instalação
echo ""
echo "✔️  Step 3/3: Validating installation..."
python -m nvs_benchmark.cli status
echo "✓ Setup validated"

# Opcional: Baixar dataset
echo ""
echo "📊 Would you like to download a dataset for benchmarking?"
read -p "Download blender_synthetic (500MB) now? (yes/no): " response

if [ "$response" = "yes" ] || [ "$response" = "y" ]; then
    echo ""
    python -m nvs_benchmark.cli install \
        --catalog-file ./configs/install_catalog.json \
        --only datasets \
        --execute
    echo "✓ Dataset downloaded successfully"
fi

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "✅ Setup complete!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  1. Run a quick benchmark:    bash ./scripts/benchmark_quick.sh"
echo "  2. View 3D results:          bash ./scripts/preview.sh"
echo "  3. Generate HTML report:     bash ./scripts/generate_report.sh"
echo ""
