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

| Script | Tempo | O Que Ele Faz |
|--------|------|-------------|
| `setup.sh` | 5 min | Instala e valida dependências |
| `download_dataset.sh` | 5-10 min | Baixa datasets de benchmark |
| `benchmark_quick.sh` | 2 min | Teste rápido em um único método |
| `benchmark_all.sh` | 30+ min | Comparação completa entre todos os métodos |
| `preview.sh` | - | Visualizador web 3D iterativo |
| `generate_report.sh` | 1 min | Relatório HTML com gráficos |

---

## 🎯 Fluxos de Trabalho Comuns

### Teste Rápido (2 minutos)
```bash
cd scripts/linux
chmod +x *.sh
./setup.sh              # Apenas uma vez
./download_dataset.sh   # Apenas uma vez
./benchmark_quick.sh
./preview.sh
```

### Suite Completa de Benchmark (30+ minutos)
```bash
cd scripts/linux
./benchmark_all.sh      # Executa todos os 4 métodos
./generate_report.sh    # Cria relatório em HTML
```

### Método Customizado
```bash
cd scripts/linux
./benchmark_quick.sh nerf_dynamic    # Executa D-NeRF ao invés de NeRF
./benchmark_quick.sh gs_static       # Executa Gaussian Splatting
```

---

## 🔧 Uso Avançado

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
