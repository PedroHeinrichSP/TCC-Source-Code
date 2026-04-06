# ============================================================
# NVS Benchmark — Imagem Docker com CUDA + PyTorch
# ============================================================
# Uso:
#   docker build -t nvs-benchmark .
#   docker run --gpus all -p 8780:8780 -v $(pwd)/data:/workspace/data nvs-benchmark
#
# Versões:
#   CUDA 12.1 + cuDNN 8 + PyTorch 2.2 + Python 3.11

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ─── Metadados ────────────────────────────────────────────────────────────────
LABEL maintainer="TCC NVS Benchmark"
LABEL description="Benchmark educacional para métodos de Novel View Synthesis"
LABEL version="0.1.0"

# ─── Variáveis de ambiente ────────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV WORKSPACE=/workspace

# ─── Dependências do sistema ──────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    git \
    git-lfs \
    curl \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # WeasyPrint deps (PDF)
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libcairo2 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    shared-mime-info \
    && rm -rf /var/lib/apt/lists/*

# ─── Python 3.11 como padrão ─────────────────────────────────────────────────
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# ─── Diretório de trabalho ────────────────────────────────────────────────────
WORKDIR $WORKSPACE

# ─── PyTorch com CUDA 12.1 ───────────────────────────────────────────────────
# Instalado antes do resto para aproveitar cache de layer
RUN pip install --upgrade pip && \
    pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# ─── Dependências do projeto ─────────────────────────────────────────────────
COPY pyproject.toml requirements.txt ./
COPY src/ ./src/

RUN pip install -e ".[dev]"

# ─── Código do projeto ───────────────────────────────────────────────────────
# (já copiado acima; configs/scripts separados para evitar rebuild desnecessário)
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY web_ui/ ./web_ui/

# ─── Diretórios de dados e artefatos ─────────────────────────────────────────
RUN mkdir -p \
    $WORKSPACE/data \
    $WORKSPACE/artifacts/metrics \
    $WORKSPACE/artifacts/reports \
    $WORKSPACE/logs \
    $WORKSPACE/third_party

# ─── Portas ──────────────────────────────────────────────────────────────────
# 8780 = Dashboard Web UI
# 8765 = Viser 3D Viewer
EXPOSE 8780 8765

# ─── Healthcheck ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import nvs_benchmark; print('ok')" || exit 1

# ─── Ponto de entrada padrão ─────────────────────────────────────────────────
CMD ["python", "-m", "nvs_benchmark.cli", "ui-preview", \
     "--host", "0.0.0.0", \
     "--dashboard-port", "8780", \
     "--viser-port", "8765"]
