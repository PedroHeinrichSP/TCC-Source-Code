#!/bin/bash
set -e

REPO_URL="${1:-https://github.com/graphdeco-inria/gaussian-splatting}"
TARGET_DIR="${2:-./third_party/gaussian_splatting}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

TARGET_PATH="$PROJECT_ROOT/$TARGET_DIR"
TARGET_PARENT="$(dirname "$TARGET_PATH")"

# Criar diretório pai se não existir
mkdir -p "$TARGET_PARENT"

if [ -d "$TARGET_PATH" ]; then
    echo "[skip] Repository already exists: $TARGET_PATH"
else
    echo "╔════════════════════════════════════════════════════════════╗"
    echo "║         Setup 3D Gaussian Splatting Repository             ║"
    echo "╚════════════════════════════════════════════════════════════╝"
    echo ""
    
    # Tentar usar git se disponível
    if command -v git &> /dev/null; then
        echo "[info] Cloning with git..."
        if ! git clone --depth 1 "$REPO_URL" "$TARGET_PATH"; then
            echo "❌ Failed to clone from $REPO_URL"
            exit 1
        fi
    else
        echo "[warn] Git not found. Using wget/curl + unzip fallback..."
        
        ZIP_URL="$REPO_URL/archive/refs/heads/main.zip"
        ZIP_FILE="$TARGET_PARENT/gaussian_splatting_main.zip"
        EXTRACT_ROOT="$TARGET_PARENT/gaussian_splatting_extract"
        
        # Download com wget ou curl
        if command -v wget &> /dev/null; then
            if ! wget -q "$ZIP_URL" -O "$ZIP_FILE"; then
                echo "❌ Failed to download from $ZIP_URL"
                exit 1
            fi
        elif command -v curl &> /dev/null; then
            if ! curl -L -o "$ZIP_FILE" "$ZIP_URL"; then
                echo "❌ Failed to download from $ZIP_URL"
                exit 1
            fi
        else
            echo "❌ Neither git, wget, nor curl found. Cannot download repository."
            exit 1
        fi
        
        # Extrair
        if [ -d "$EXTRACT_ROOT" ]; then
            rm -rf "$EXTRACT_ROOT"
        fi
        
        if ! unzip -q "$ZIP_FILE" -d "$EXTRACT_ROOT"; then
            echo "❌ Failed to extract $ZIP_FILE"
            rm -f "$ZIP_FILE"
            exit 1
        fi
        
        # Encontrar diretório interno e mover
        INNER_DIR=$(find "$EXTRACT_ROOT" -maxdepth 1 -type d -name "gaussian-splatting*" | head -1)
        if [ -z "$INNER_DIR" ]; then
            echo "❌ Failed to extract repository structure from ZIP"
            rm -rf "$EXTRACT_ROOT" "$ZIP_FILE"
            exit 1
        fi
        
        mv "$INNER_DIR" "$TARGET_PATH"
        rm -rf "$EXTRACT_ROOT" "$ZIP_FILE"
    fi
fi

# Validar estrutura
TRAIN_FILE="$TARGET_PATH/train.py"
RENDER_FILE="$TARGET_PATH/render.py"

if [ ! -f "$TRAIN_FILE" ] || [ ! -f "$RENDER_FILE" ]; then
    echo "❌ 3DGS repository incomplete at $TARGET_PATH"
    echo "Missing: train.py and/or render.py"
    exit 1
fi

echo ""
echo "[ok] 3DGS repository ready at: $TARGET_PATH"
echo "[next] Configure dependencies (CUDA/PyTorch/GPU support) as needed"
echo ""
echo "To use in benchmarks:  bash ./scripts/benchmark_quick.sh gs_static"
echo ""
