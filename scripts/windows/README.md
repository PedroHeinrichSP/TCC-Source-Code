# 🪟 Guia de Scripts do Windows

## Escolha o Seu Método de Execução Preferido

### Opção 1️⃣: **CMD (Mais Simples - Recomendado para Iniciantes)**
Não haverá avisos de ExecutionPolicy do PowerShell. Apenas clique duas vezes ou execute diretamente.

```cmd
# Navegue até a subpasta cmd
cd c:\Users\SeuNome\Projetos\TCC\scripts\windows\cmd

# E então execute qualquer script:
setup.cmd
benchmark_quick.cmd
benchmark_all.cmd
preview.cmd
download_dataset.cmd
generate_report.cmd
```

**Prós:**
- ✅ Sem problemas com ExecutionPolicy
- ✅ Comandos de uma linha só, sem flags extras
- ✅ Funciona no Prompt de Comando convencional

**Contras:**
- ❌ Menos flexível (sem parâmetros customizados)

---

### Opção 2️⃣: **PowerShell (Maior Controle)**
Mais flexibilidade para passar parâmetros customizados.

```powershell
# Navegue até a subpasta powershell
cd c:\Users\SeuNome\Projetos\TCC\scripts\windows\powershell

# Execute ignorando a ExecutionPolicy (sem alterações globais)
powershell -ExecutionPolicy Bypass -File setup.ps1
powershell -ExecutionPolicy Bypass -File benchmark_quick.ps1 -Method nerf_static

# Ou com parâmetros customizados:
powershell -ExecutionPolicy Bypass -File benchmark_quick.ps1 -Method gs_static
```

**Prós:**
- ✅ Pode passar parâmetros customizados
- ✅ Scripts mais poderosos
- ✅ Mesma dinâmica que scripts shell de Linux

**Contras:**
- ❌ Requer a flag de ignorar a ExecutionPolicy (temporário)

---

## 📋 Scripts Disponíveis

| Script | Tempo | O Que Ele Faz |
|--------|------|-------------|
| `setup.{cmd,ps1}` | 5 min | Instala e valida dependências |
| `download_dataset.{cmd,ps1}` | 5-10 min | Baixa datasets de benchmark |
| `benchmark_quick.{cmd,ps1}` | 2 min | Teste rápido em um único método |
| `benchmark_all.{cmd,ps1}` | 30+ min | Comparação completa entre todos os métodos |
| `preview.{cmd,ps1}` | - | Visualizador web 3D iterativo |
| `generate_report.{cmd,ps1}` | 1 min | Relatório HTML com gráficos |

---

## 🚀 Início Rápido (CMD - Mais Fácil)

```cmd
# 1. Setup (somente uma vez)
cd scripts\windows\cmd
setup.cmd

# 2. Baixar dados
download_dataset.cmd

# 3. Executar o benchmark rápido
benchmark_quick.cmd

# 4. Visualizar os resultados
preview.cmd

# 5. Gerar relatórios
generate_report.cmd
```

---

## 🛠️ PowerShell com Parâmetros Customizados

```powershell
cd scripts\windows\powershell

# Executar NeRF Dinâmico ao invés do nerf_static
powershell -ExecutionPolicy Bypass -File benchmark_quick.ps1 -Method nerf_dynamic

# Executar Gaussian Splatting
powershell -ExecutionPolicy Bypass -File benchmark_quick.ps1 -Method gs_static

# Gerar relatório com um nome customizado
powershell -ExecutionPolicy Bypass -File generate_report.ps1 -ReportName meus_resultados
```

---

## 📂 Estrutura do Diretório

```
windows/
├── cmd/           ← Clique aqui para a experiência mais simples
│   ├── setup.cmd
│   ├── benchmark_quick.cmd
│   ├── benchmark_all.cmd
│   ├── preview.cmd
│   ├── generate_report.cmd
│   ├── download_dataset.cmd
│   └── (scripts legados)
├── powershell/    ← Para usuários avançados que querem maior controle
│   ├── setup.ps1
│   ├── benchmark_quick.ps1
│   ├── benchmark_all.ps1
│   ├── preview.ps1
│   ├── generate_report.ps1
│   ├── download_dataset.ps1
│   └── (scripts legados)
└── README.md      ← Este mesmo arquivo
```

---

## ❓ FAQ

**P: Qual eu devo usar, CMD ou PowerShell?**
R: Inicie utilizando o CMD (sem flags extras), e mude para o PowerShell caso precise de parâmetros customizados.

**P: Erro de "ExecutionPolicy"?**
R: Você está tentando executar comandos .ps1 sem a flag bypass. Use:
```powershell
powershell -ExecutionPolicy Bypass -File script.ps1
```
Ou simplesmente use a versão em `.cmd` ao invés disso!

**P: Posso executar de qualquer diretório?**
R: É melhor navegar até a pasta dos scripts primeiro, ou escrever o caminho absoluto completo.

**P: Preciso fazer o setup todas as vezes?**
R: Não, apenas uma vez. Após isso, utilize os scripts de benchmark/preview/report.

---

## 📚 Documentações Relacionadas

- Guia principal: `../../README.md`
- Todos os scripts: `../README.md`
- Mudanças detalhadas: `../../docs/REFACTORING_SUMMARY.md`

---

**🎓 Precisa de Linux ou Colab? Veja a documentação pai `scripts/README.md`**
