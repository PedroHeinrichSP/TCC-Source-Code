# 🐧 Guia de Scripts do Linux / macOS

Todos os scripts nesta pasta são shell scripts (`.sh`) para sistemas Unix (e derivados).

## 🚀 Início Rápido

```bash
# 1. Navegue até esta pasta
cd scripts/linux

# 2. Torne os scripts executáveis (apenas na primeira vez)
chmod +x *.sh

# 3. Execute o setup
./setup.sh

# 4. Execute o benchmark
./benchmark_quick.sh

# 5. Visualize os resultados
./preview.sh

# 6. Gere relatórios
./generate_report.sh
```

---

## 📋 Scripts Disponíveis

### Scripts Principais

| Script | Tempo | O Que Ele Faz |
|--------|------|-------------|
| `setup.sh` | 5 min | Cria venv, instala e valida dependências (3 estágios) |
| `setup_local.sh` | 3 min | Setup minimalista sem venv (usa Python do sistema) |
| `download_dataset.sh` | 5-10 min | Baixa datasets de benchmark |
| `benchmark_quick.sh` | 2-5 min | Teste rápido em um único método (padrão: nerf_static) |
| `benchmark_all.sh` | 30+ min | Comparação completa entre todos os 4 métodos |
| `preview.sh` | - | Visualizador web 3D iterativo (Viser) |
| `generate_report.sh` | 1 min | Relatório HTML com gráficos e métricas |
| `run_ui_preview.sh` | - | Visualizador alternativo (legacy) |

### Scripts Especializados de Setup

| Script | O Que Ele Faz |
|--------|-------------|
| `setup_lego.sh` | Setup especializado para dataset Lego (3 estágios: download + validação) |
| `setup_gs_static.sh` | Clona repositório 3D Gaussian Splatting (com fallback wget/unzip se git não disponível) |

---

## 🎯 Fluxos de Trabalho Comuns

### Teste Rápido (2-5 minutos) - Recomendado para Começar
```bash
cd scripts/linux
chmod +x *.sh                # Permissão de execução (apenas 1ª vez)
./setup.sh                   # Setup com 3 estágios (cria venv, instala, valida)
./download_dataset.sh        # Baixa dataset Blender Synthetic (~500MB)
./benchmark_quick.sh         # Benchmark rápido (nerf_static por padrão)
./preview.sh                 # Visualizador 3D interativo
```

### Setup Especializado para Lego
```bash
./setup.sh           # Setup base
./setup_lego.sh      # Download + validação específica de Lego (3 estágios)
./benchmark_quick.sh nerf_static ./data/blender_synthetic/nerf_synthetic/lego
```

### Setup para Gaussian Splatting (3DGS)
```bash
./setup.sh               # Setup base
./setup_gs_static.sh     # Clona repositório 3DGS (usa git ou wget/unzip)
./benchmark_quick.sh gs_static
```

### Suite Completa de Benchmark (30+ minutos)
```bash
./setup.sh           # Setup base
./download_dataset.sh
./benchmark_all.sh   # Executa todos os 4 métodos em sequência:
                     #   - nerf_static
                     #   - nerf_dynamic  
                     #   - gs_static
                     #   - gs_dynamic
./generate_report.sh # Cria relatório em HTML com comparação
./preview.sh         # Visualiza resultados
```

### Setup Local (sem venv, usa Python do Sistema)
```bash
./setup_local.sh     # Install direto no Python do sistema
./download_dataset.sh
./benchmark_quick.sh
```

---

## � Melhorias de Robustez (v2.0)

Os scripts Linux foram melhorados para:

✅ **Mostrar erros reais**: `pip install` não redireciona mais stderr para `/dev/null`, então você verá mensagens de erro concretas  
✅ **Validação de venv**: Scripts verificam se venv existe before activating, e falham com mensagem clara caso contrário  
✅ **Tratamento de erros com `set -e`**: Todos os scripts param na primeira falha (não continuam silenciosamente)  
✅ **Detecção automática de GPU/CUDA**: Métodos NeRF ajustam tipo de tensor (CPU vs CUDA) automaticamente  
✅ **Fallback wget/unzip para 3DGS**: Se git não estiver instalado, `setup_gs_static.sh` usa wget/curl + unzip  
✅ **Scripts especializados**: Agora há 4 scripts de setup especializados (setup_lego.sh, setup_gs_static.sh, setup_local.sh)  

---

### Ver ajuda para um script
```bash
cd scripts/linux
bash benchmark_quick.sh --help
```

### Executar com um dataset customizado
```bash
./benchmark_quick.sh gs_static dataset_customizado ./caminho/para/dataset
```

### Visualizador 3D em Host/Porta Customizados
```bash
./preview.sh 192.168.1.100 9000
```

---

## 📂 Estrutura do Diretório

```
linux/
├── setup.sh
├── download_dataset.sh
├── benchmark_quick.sh
├── benchmark_all.sh
├── preview.sh
├── generate_report.sh
└── README.md          ← Este mesmo arquivo
```

---

## ❓ FAQ

**P: Estou recebendo o erro de "Permission denied" (Permissão negada)**
R: Os scripts requerem permissões de execução. Execute:
```bash
chmod +x *.sh
```

**P: Os scripts ativam o venv automaticamente?**
R: Sim, eles procuram por `./venv` e o ativam. Ou eles tentarão usar o Python nativo do sistema caso não encontrem um venv criado.

**P: Posso executar de qualquer diretório?**
R: É melhor permanecer na pasta `scripts/linux/`, ou então declarar caminhos absolutos completos para a execução.

**P: Como posso saber se estou usando CUDA/GPU que estão disponíveis?**
R: Os scripts fazem detecção automática. Caso uma GPU seja encontrada, os modelos de Gaussian Splatting irão utilizá-la. Caso não encontrem, eles cairão nos modelos nativos de CPU (Métodos NeRF normais).

**P: Como proceder em um macOS?**
R: Os scripts funcionam normalmente em macOS! O procedimento é idêntico ao Linux.

---

## 📚 Documentações Relacionadas

- Guia principal: `../../README.md`
- Todos os scripts: `../README.md`
- Usuários Windows: `../windows/README.md`
- Usuários Colab: `../colab/README.md`

---

**🎓 Suporte completo a todos Sistemas Operacionais - escolha o seu método preferido!**
