#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Project root is one level above scripts/
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
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
if ! python -m pip install --upgrade pip; then
    echo "❌ Failed to upgrade pip"
    exit 1
fi
if ! python -m pip install -e .; then
    echo "❌ Failed to install nvs_benchmark package"
    echo "Make sure setup.py or pyproject.toml is in the project root."
    exit 1
fi
echo "✓ Dependencies installed"

# Passo 3: Validar a instalação
echo ""
echo "✔️  Step 3/3: Validating installation..."
if ! python -m nvs_benchmark.cli status; then
    echo "❌ Failed to validate nvs_benchmark installation"
    exit 1
fi
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
