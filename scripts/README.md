# 📝 Guia de Scripts

Todos os scripts estão organizados por **sistema operacional** para facilitar a navegação.

## 🎯 Escolha Seu Sistema Operacional

### 🪟 **Usuários Windows** (Duas Opções)
```
scripts/windows/
├── cmd/           ← COMESSE AQUI (mais simples, sem flags adicionais)
│   ├── setup.cmd
│   ├── benchmark_quick.cmd
│   ├── download_dataset.cmd
│   └── ...
├── powershell/    ← Para usuários avançados (mais controle)
│   ├── setup.ps1
│   ├── benchmark_quick.ps1
│   └── ...
└── README.md      ← Guia completo para Windows
```

**Início Rápido:**
```cmd
cd scripts\windows\cmd
setup.cmd
benchmark_quick.cmd
preview.cmd
```

**Leia:** [Guia de Scripts para Windows](windows/README.md)

---

### 🐧 **Usuários Linux / macOS**
```
scripts/linux/
├── setup.sh
├── benchmark_quick.sh
├── download_dataset.sh
├── preview.sh
├── generate_report.sh
└── README.md        ← Guia completo para Linux
```

**Início Rápido:**
```bash
cd scripts/linux
chmod +x *.sh               # Uma vez: tornar executável
./setup.sh
./benchmark_quick.sh
./preview.sh
```

**Leia:** [Guia de Scripts para Linux/macOS](linux/README.md)

---

### ☁️ **Google Colab (Nuvem)**
```
scripts/colab/
├── setup_colab.sh
├── run_colab.sh
└── README.md        ← Guia completo para Colab
```

**Início Rápido:**
1. Abra o [Google Colab](https://colab.research.google.com/)
2. Crie um novo notebook
3. Use preferencialmente o notebook dedicado: `notebooks/nvs_benchmark_colab.ipynb`
4. Ou, na primeira célula, cole e execute:
```python
!git clone https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git TCC
%cd /content/TCC
!bash scripts/colab/setup_colab.sh
```

**Leia:** [Guia de Scripts para Colab](colab/README.md)

---

## 📋 Referência de Scripts

Todos os sistemas operacionais possuem os mesmos scripts com funcionalidade similar:

| Script | Propósito | Windows | Linux | Colab |
|--------|---------|---------|-------|-------|
| Setup | Instala & valida dependências | ✅ | ✅ | ✅ |
| Download Dataset | Baixa dados de benchmark | ✅ | ✅ | — |
| Benchmark Quick | Teste com método único | ✅ | ✅ | via notebook |
| Benchmark All | Compara todos os métodos | ✅ | ✅ | manual |
| Preview | View 3D interativo | ✅ | ✅ | — |
| Generate Report | Relatório HTML & gráficos | ✅ | ✅ | ✅ |

---

## 🚀 Fluxo Típico (Qualquer SO)

1. **Setup** (uma única vez)
   ```
   ./setup  (Windows: setup.cmd ou setup.ps1)
            (Linux:   ./setup.sh)
            (Colab:   bash setup_colab.sh)
   ```

2. **Download Dataset** (uma única vez)
   ```
   ./download_dataset  (ou download_dataset.sh)
   ```

3. **Run Benchmark** (Executar Benchmark)
   ```
   ./benchmark_quick  (ou ./benchmark_quick.sh)
   ```

4. **View Results** (Visualizar Resultados)
   ```
   ./preview  (ou ./preview.sh)
   Abre um viewer 3D no link http://127.0.0.1:8765
   ```

5. **Generate Report** (Gerar Relatórios)
   ```
   ./generate_report  (ou ./generate_report.sh)
   Gera um relatório HTML contendo gráficos
   ```

---

## 🆘 Solução de Problemas por SO

### Windows: Erro de ExecutionPolicy
**Problema:** "cannot be loaded because running scripts is disabled"
**Solução:** Substitua por usar a versão `.cmd` ao invés de `.ps1`, ou adicione a flag ExecutionPolicy:
```powershell
powershell -ExecutionPolicy Bypass -File script.ps1
```

### Linux: Permission Denied
**Problema:** "permission denied" erro para executar scripts `.sh`
**Solução:** Transforme-os em executáveis:
```bash
chmod +x scripts/linux/*.sh
```

### Colab: Module Not Found
**Problema:** "ModuleNotFoundError: No module named 'nvs_benchmark'"
**Solução:** Re-execute a célula de setup:
```python
!pip install -e . > /dev/null 2>&1
```

---

## 📂 Estrutura Completa do Diretório

```
scripts/
├── windows/
│   ├── cmd/                      ← Inicie a jornada Windows por aqui!
│   │   ├── setup.cmd
│   │   ├── benchmark_quick.cmd
│   │   ├── benchmark_all.cmd
│   │   ├── download_dataset.cmd
│   │   ├── preview.cmd
│   │   ├── generate_report.cmd
│   │   └── (scripts legados...)
│   ├── powershell/               ← Windows para avançados
│   │   ├── setup.ps1
│   │   ├── benchmark_quick.ps1
│   │   ├── benchmark_all.ps1
│   │   ├── download_dataset.ps1
│   │   ├── preview.ps1
│   │   ├── generate_report.ps1
│   │   └── (scripts legados...)
│   └── README.md                 ← Guia Windows
├── linux/                        ← Linux/macOS/WSL
│   ├── setup.sh
│   ├── benchmark_quick.sh
│   ├── benchmark_all.sh
│   ├── download_dataset.sh
│   ├── preview.sh
│   ├── generate_report.sh
│   └── README.md                 ← Guia Linux
├── colab/                        ← Google Colab
│   ├── setup_colab.sh
│   ├── run_colab.sh
│   └── README.md                 ← Guia Colab
└── README.md                     ← Este mesmo arquivo
```

---

## 💡 Dicas Rápidas

✅ **Iniciantes de Windows:** Escolha usar a pasta `scripts/windows/cmd/`  
✅ **Avançados do Windows:** Escolha usar `scripts/windows/powershell/` com parâmetros customizados  
✅ **Linux/macOS:** Escolha usar `scripts/linux/` (não esqueça dos arquivos `.sh` com permissões `chmod +x`)  
✅ **Nuvem gratuita:** Use Google Colab (também inclui GPUs de graça!)  
✅ **Suporte Multiplataforma:** Todos os scripts fazem exatamente a mesma coisa, apenas com sintaxe condizente do sistema

---

## 📚 Documentação

- **Guia Principal:** [`../../README.md`](../README.md)
- **Detalhes sobre Windows:** [`windows/README.md`](windows/README.md)
- **Detalhes sobre Linux:** [`linux/README.md`](linux/README.md)
- **Detalhes sobre Colab:** [`colab/README.md`](colab/README.md)
- **Resumo de Refatoramento:** [`../../docs/REFACTORING_SUMMARY.md`](../docs/REFACTORING_SUMMARY.md)

---

## 🎓 Ciclo de Aprendizado

1. Leia o arquivo principal `README.md` (5 min) - Para obter o conceito do projeto
2. Escolha seu SO → Leia o arquivo `README.md` de acordo do SO (5 min)
3. Execute Passo 1: O script de `setup` (5 min)
4. Execute Passo 2: O método de benchmark com `benchmark_quick` (2 min)
5. Execute Passo 3: Realize acesso interativo através do `preview` para ver resultados
6. Explore nossos cadernos (notebooks) interativos para obter mais conhecimentos baseados em prática

**O primeiro teste e seus resultados levarão mais ou menos:~20 minutos!** ⚡

---

**🎓 Realizado para estudantes. Escolha o seu Sistema Operacional e inicie agora mesmo!**
