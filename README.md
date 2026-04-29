# 📊 NVS Benchmark - Suite de Benchmarks para Síntese de Novas Perspectivas

Um framework educacional **amigável para estudantes** projetado para comparar e testar métodos de síntese de novas perspectivas (novel view synthesis) no seu computador.

## 🎯 O Que Este Projeto Faz?

Esta é uma ferramenta educacional que te ajuda a:
- **Executar** diferentes algoritmos de síntese de visão (NeRF, Gaussian Splatting, e mais)
- **Medir** a qualidade (PSNR, SSIM, LPIPS) e o desempenho (FPS, memória, tempo de treinamento) deles
- **Comparar** resultados lado a lado com visualizações interativas
- **Gerar** belos relatórios em HTML com gráficos e imagens

Perfeito para **estudantes de visão computacional** que estão aprendendo sobre renderização 3D, redes neurais e síntese de novas perspectivas!

## 🚀 Início Rápido (5 Minutos)

### Pré-requisitos
- Python 3.10+
- Ambiente virtual (recomendado)

### Windows (PowerShell)
```powershell
# 1. Configurar o ambiente
powershell -ExecutionPolicy Bypass -File ./scripts/windows/powershell/setup.ps1

# 2. Executar seu primeiro benchmark
powershell -ExecutionPolicy Bypass -File ./scripts/windows/powershell/benchmark_quick.ps1

# 3. Visualizar os resultados no navegador
powershell -ExecutionPolicy Bypass -File ./scripts/windows/powershell/preview.ps1
```

### Linux / macOS / Colab
```bash
# 1. Configurar o ambiente
bash ./scripts/linux/setup.sh

# 2. Executar seu primeiro benchmark
bash ./scripts/linux/benchmark_quick.sh

# 3. Visualizar os resultados no navegador
bash ./scripts/linux/preview.sh
```

## 📁 Estrutura do Projeto (Focada no Estudante)

```
.
├── scripts/              ← Execute estes scripts! (setup, benchmark, preview)
├── src/nvs_benchmark/    ← Código principal da biblioteca
│   ├── cli.py           ← Interface de linha de comando
│   ├── methods/         ← Implementações das redes neurais
│   ├── data/            ← Carregamento de datasets
│   ├── evaluation/       ← Métricas (PSNR, SSIM, etc)
│   └── ui/              ← Visualizador 3D interativo
├── configs/             ← Configurações para cada método
├── data/                ← Datasets ficam aqui
├── artifacts/           ← Resultados, métricas, relatórios
└── notebooks/           ← Notebooks Jupyter para aprendizado
```

## 🎮 Métodos Disponíveis

| Método | Tipo | Requer GPU | Velocidade |
|--------|------|-----------|-------|
| **nerf_static** | Neural Radiance Fields | Opcional | Média |
| **nerf_dynamic** | Dynamic NeRF (D-NeRF) | Opcional | Lenta |
| **gs_static** | 3D Gaussian Splatting | Sim | Rápida ⚡ |
| **gs_dynamic** | 4D Gaussian Splatting | Sim | Rápida ⚡ |
| **external** | Seu próprio modelo! | Varia | Varia |

## 📊 Datasets Suportados

| Dataset | Tipo | Tamanho | Notas |
|---------|------|------|-------|
| **blender_synthetic** | Cenas 3D Sintéticas | ~500MB | Recomendado para aprendizado |
| **d_nerf** | Cenas dinâmicas | ~1GB | Para benchmarks temporais |
| **mipnerf360** | Cenas reais 360° | ~7.7GB | Formato LLFF/COLMAP, cenário unbounded |
| **tanks_and_temples** | Cenas de grande escala | ~15GB | Requer download/registro manual |
| **custom** | Seu próprio dataset | Qualquer | Usando o formato transforms_train.json |

## ⚙️ Windows PowerShell - Comandos Comuns

No Windows, os scripts utilizam **PowerShell**. Execute-os desta forma:
```powershell
powershell -ExecutionPolicy Bypass -File ./scripts/windows/powershell/NOME_DO_SCRIPT.ps1
```

(Não são necessárias alterações globais de segurança!)

## 🎓 Tutorial: Executando seu Primeiro Benchmark

### Passo 1: Configuração (Apenas uma Vez)
```powershell
powershell -ExecutionPolicy Bypass -File ./scripts/windows/powershell/setup.ps1
```
Isso fará com que:
- Crie um ambiente virtual Python
- Instale as dependências
- Valide sua configuração

### Passo 2: Baixar um Dataset
```powershell
powershell -ExecutionPolicy Bypass -File ./scripts/windows/powershell/download_dataset.ps1 -Dataset blender_synthetic
```

### Passo 3: Executar um Benchmark Rápido
```powershell
powershell -ExecutionPolicy Bypass -File ./scripts/windows/powershell/benchmark_quick.ps1
```
Isso executa o NeRF (compatível com CPU) na cena Lego e leva ~2 minutos.

### Passo 4: Visualizar Resultados Interativos
```powershell
powershell -ExecutionPolicy Bypass -File ./scripts/windows/powershell/preview.ps1
```
Abre um visualizador 3D no seu navegador em `http://127.0.0.1:8765`

### Passo 5: Gerar um Relatório
```powershell
powershell -ExecutionPolicy Bypass -File ./scripts/windows/powershell/generate_report.ps1
```
Cria um relatório em HTML contendo métricas, gráficos e comparações.

---

## 🔧 Avançado: Suite Completa de Benchmarks

Deseja testar todos os métodos? Use:
```powershell
powershell -ExecutionPolicy Bypass -File ./scripts/windows/powershell/benchmark_all.ps1
```

Isso executa:
- NeRF (estático) - Funciona bem em CPU
- NeRF Dinâmico - Funciona bem em CPU
- Gaussian Splatting - GPU OBRIGATÓRIA
- 4D Gaussian Splatting - GPU OBRIGATÓRIA

(Os métodos que exigem GPU serão pulados se você não possuir uma GPU NVIDIA)

---

## 💡 Adicionando Seu Próprio Modelo

Deseja rodar o benchmark na sua própria rede neural? Crie um arquivo de configuração:

**`configs/my_method.json`:**
```json
{
  "train_command": "python my_train.py --data $NVS_DATASET_ROOT --out $NVS_OUTPUT_DIR",
  "infer_command": "python my_infer.py --ckpt $NVS_CHECKPOINT_PATH --out $NVS_RENDERED_DIR",
  "checkpoint_path": "./checkpoints/model.ckpt",
  "rendered_dir": "./renders",
  "frames": 120
}
```

Em seguida execute:
```powershell
python -m nvs_benchmark.cli method-run `
  --method external `
  --dataset blender_synthetic `
  --root ./data/blender_synthetic/lego `
  --split train `
  --compute-metrics `
  --extra-file ./configs/my_method.json
```

---

## 📚 Recursos de Aprendizado

### Notebooks
- `notebooks/nvs_benchmark_pipeline.ipynb` - Passo a passo completo com explicações
- `notebooks/nvs_benchmark_colab.ipynb` - Fluxo pronto para Google Colab (setup, benchmark, relatório e download)
- `notebooks/nvs_benchmark_colab_full_matrix.ipynb` - Execução completa (métodos x datasets) com relatório geral

### Documentação
- `docs/FLUXO_DE_DADOS.md` - Explicação do fluxo de dados
- `docs/TESTS_BANCADA.md` - Especificações de teste

### Referência de Comandos
```python
# Ver todos os comandos disponíveis
python -m nvs_benchmark.cli --help

# Testar um método específico
python -m nvs_benchmark.cli method-run --help

# Gerar relatórios
python -m nvs_benchmark.cli report-generate --help

# Visualizar a interface iterativa em 3D
python -m nvs_benchmark.cli ui-preview --help
```

---

## 🆘 Solução de Problemas

### "ModuleNotFoundError: No module named 'nvs_benchmark'"
Execute o setup primeiro:
```powershell
powershell -ExecutionPolicy Bypass -File ./scripts/windows/powershell/setup.ps1
```

### O Script não roda (Erro de ExecutionPolicy)
Use os invólucros `.cmd` fornecidos ao invés dos `.ps1`:
```powershell
./scripts/windows/cmd/benchmark_quick.cmd
./scripts/windows/cmd/preview.cmd
```

### GPU Não Detectada?
Não se preocupe! A maioria dos métodos funciona bem em CPU. Para métodos acelerados por GPU:
```bash
# Verifique se existe uma GPU NVIDIA disponível
python -c "import torch; print(torch.cuda.is_available())"
```

Se o retorno for falso (false), você está rodando em CPU - e tudo bem! Os métodos baseados em Gaussian Splatting farão o fallback para o NeRF automaticamente.

### Faltou Memória (Out of Memory)?
Reduza o tamanho do dataset ou use cenas menores. A cena Lego do `blender_synthetic` é a menor de todas (~100MB).

---

## 📖 Detalhes do Projeto

### Arquitetura

```
nvs_benchmark/
├── core/          ← Tipos comuns e classes base
├── methods/       ← Implementações de algoritmos (NeRF, GS, etc)
├── data/          ← Carregadores dos datasets
├── evaluation/    ← Métricas de Qualidade (PSNR, SSIM, LPIPS)
├── runtime/       ← Perfilamento de Hardware & Logs
├── reporting/     ← Geração de relatório HTML/PDF
└── ui/            ← Visualizador 3D iterativo (Viser)
```

### Conceitos Chave

**Contrato de Datasets**: Cada dataset precisa fornecer um arquivo `transforms_train.json` contento as poses de câmera e os caminhos das imagens.

**Contrato de Métodos**: Cada método implementa:
- `TrainRequest` → treina através de um dataset
- `InferenceRequest` → renderiza vistas da cena
- E avalia automaticamente: FPS, VRAM e métricas de qualidade.

**Métricas**:
- **PSNR** - Peak Signal-to-Noise Ratio (maior = melhor)
- **SSIM** - Structural Similarity (maior = melhor)
- **LPIPS** - Learned Perceptual Image Patch Similarity (menor = melhor)
- **FPS** - Frames Por Segundo (maior = mais rápido)
- **VRAM** - Pico de uso da memória de vídeo da GPU
- **Time** - Tempo de treinamento e inferência

---

## 🔗 Referências

| Método | Artigo | Repositório |
|--------|-------|------|
| NeRF | [Mildenhall et al. 2020](https://arxiv.org/abs/2003.08934) | [Oficial](https://github.com/bmild/nerf) |
| D-NeRF | [Li et al. 2021](https://arxiv.org/abs/2011.13961) | [Oficial](https://github.com/albertpumarola/D-NeRF) |
| 3D-GS | [Kerbl et al. 2023](https://arxiv.org/abs/2308.04079) | [Oficial](https://github.com/graphdeco-inria/gaussian-splatting) |
| 4D-GS | [Hu et al. 2024](https://arxiv.org/abs/2402.03307) | [Oficial](https://github.com/hustvl/4DGaussians) |

---

## 💬 Dúvidas?

- Verifique `docs/` para detalhes de arquitetura
- Leia `notebooks/` para exemplos passo a passo detalhados
- Olhe `tests/` para conferir padrões de uso da ferramenta
- Ou abra uma _issue_ no projeto!

---

## 📄 Licença

Projeto educacional - acesse LICENSE para mais detalhes.

**Feito para ajudar os estudantes que estão aprendendo visão computacional e renderização neural! 🎓**
