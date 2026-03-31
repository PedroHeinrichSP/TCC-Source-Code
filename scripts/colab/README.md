# ☁️ Guia de Scripts do Google Colab

Execute benchmarks de NVS diretamente no Google Colab (diretamente no navegador, nenhuma instalação necessária!).

## 🚀 Início Rápido no Colab

### Opção 1: Link Direto da Web (O mais Fácil)
1. Abra o [Google Colab](https://colab.research.google.com/)
2. Clique em **File (Arquivo)** → **Open Notebook (Abrir Notebook)** → **GitHub**
3. Cole o seguinte link: `https://github.com/SEU_REPOSITORIO/blob/main/notebooks/nvs_benchmark_pipeline.ipynb`
4. Clique no primeiro resultado e execute as células do notebook em sequência

### Opção 2: Fazer Upload & Executar o Script
1. Abra o [Google Colab](https://colab.research.google.com/)
2. Crie um novo notebook
3. Na primeira célula, cole os comandos:
```python
!git clone https://github.com/SEU_REPOSITORIO TCC
%cd TCC/scripts/colab
!bash setup_colab.sh
```
4. Em seguida, execute as células restantes de configuração

---

## 📋 Scripts Disponíveis

| Script | Notas |
|--------|-------|
| `setup_colab.sh` | Instala dependências dentro do Colab |
| `run_colab.sh` | Pipeline completo (setup → benchmark → relatório) |

---

## 🎯 Fluxo de Trabalho Completo no Colab

### Célula 1: Configuração e Clonagem do Repositório
```python
# Montar o Google Drive (opcional, para salvar resultados)
from google.colab import drive
drive.mount('/content/drive')

# Clonar e configurar
!git clone https://github.com/SEU_REPOSITORIO /content/TCC
%cd /content/TCC
!pip install -e . > /dev/null 2>&1
!echo "✓ NVS Benchmark instalado"
```

### Célula 2: Baixar Dataset
```python
!python -m nvs_benchmark.cli install \
    --catalog-file ./configs/install_catalog.json \
    --only datasets \
    --execute
```

### Célula 3: Benchmark Rápido
```python
!python -m nvs_benchmark.cli method-run \
    --method nerf_static \
    --dataset blender_synthetic \
    --root ./data/blender_synthetic/nerf_synthetic/lego \
    --split train \
    --output-dir ./artifacts \
    --log-dir ./logs \
    --compute-metrics \
    --snapshot-file ./artifacts/metrics/latest.json \
    --append-snapshot
```

### Célula 4: Gerar Relatório
```python
!python -m nvs_benchmark.cli report-generate \
    --snapshot-file ./artifacts/metrics/latest.json \
    --output-dir ./artifacts/reports \
    --report-name colab_results \
    --log-dir ./logs \
    --no-pdf
```

### Célula 5: Baixar Resultados
```python
# Comprimir e baixar os resultados
import shutil
shutil.make_archive('/content/results', 'zip', '/content/TCC/artifacts')

from google.colab import files
files.download('/content/results.zip')
```

---

## 💾 Usando o Google Drive para Armazenamento

### Montar o Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Salvar Resultados no Drive
```python
import shutil
import os

# Copiar os artefatos gerados para o Google Drive
drive_dir = '/content/drive/My Drive/NVS_Benchmark'
os.makedirs(drive_dir, exist_ok=True)
shutil.copytree('/content/TCC/artifacts/metrics', f'{drive_dir}/metrics')
shutil.copytree('/content/TCC/artifacts/reports', f'{drive_dir}/reports')
```

### Carregar Resultados Anteriores
```python
# Carregar as métricas da sua última execução salva
import json

metrics_file = '/content/drive/My Drive/NVS_Benchmark/metrics/latest.json'
with open(metrics_file) as f:
    metrics = json.load(f)
print(metrics)
```

---

## 🖼️ Visualizar Resultados no Colab

### Apresentar Relatório HTML
```python
from IPython.display import IFrame
IFrame(src='/content/TCC/artifacts/reports/colab_results.html', width=1200, height=800)
```

### Exibir Tabela de Métricas
```python
import pandas as pd
import json

with open('/content/TCC/artifacts/metrics/latest.json') as f:
    data = json.load(f)

# Criar a tabela de comparação
df = pd.DataFrame(data).T
print(df[['fps', 'psnr', 'ssim', 'vram_gb']])
```

---

## ⚡ Acesso de GPU no Colab

O Colab inclui **uso gratuito de GPU!** Siga os passos de como habilitar:

1. Clique em **Runtime** → **Change runtime type (Alterar tipo de tempo de execução)**
2. Selecione **GPU** abaixo de "Hardware accelerator (Acelerador de hardware)"
3. Aperte **Save (Salvar)**

Prontinho! Seus benchmarks ficarão de 10-100x mais rápidos!

---

## ❓ FAQ

**P: Eu posso executar os métodos de Gaussian Splatting (GS) no Colab?**
R: Sim! Considerando que a GPU tenha sido ativada, os 4 métodos irão executar de forma perfeita. Os métodos GS rodarão entre 50-100x mais rapidamente comparados à performance do CPU nativo.

**P: Quanto tempo demora um benchmark completo?**
R: Entre 30 a 45 minutos usando as placas de vídeo do Colab (GPU K80 ou T4), de ~2-3 horas rodando no processador (CPU).

**P: Eu posso guardar os resultados dentro do meu Google Drive?**
R: Com toda certeza! Veja a seção "Usando o Google Drive para Armazenamento" lendo o arquivo acima.

**P: O tempo gratuíto da placa de vídeo (GPU) do Colab estará 100% à minha disposição independente do que aconteça?**
R: Na grande parte do tempo a resposta será sim, mas se for muito sobrecarregada por solicitações do usuário ou por conta de liminares fixas, ela poderá ficar indisponível ou inacessível brevemente. Para resolver isto deve-se assinar o Colab Premium.

**P: Posso executar o ambiente gráfico e iterativo visualizador web (preview)?**
R: O navegador 3D de _sandbox_ iterativo rodado junto do Viser não funciona muito bem ou se adapta na restrição do _sandbox_ do ambiente da página do colab. Crie em contrapartida os relatórios exportados nos formulários em HTML.

---

## 📚 Documentações Relacionadas

- Guia principal: `../../README.md`
- Todos os scripts: `../README.md`
- Usuários de Windows: `../windows/README.md`
- Usuários de Linux/Mac: `../linux/README.md`
- Tutorial sobre como funciona em notebooks: `../../notebooks/nvs_benchmark_pipeline.ipynb`

---

## 🎓 Dicas de Ouro para Colab

✅ Ative a  Aceleração por Computação de Vídeo (GPU) de Forma Nativa e Simples  
✅ Guarde Permanentemente as Metricas Encontradas Durante Suas Modificações  
✅ Salve e Compartilhe para Ajudar à Outros Colab'ers como Ocorreu O Seu Retorno de Custo Sem Fim ($0,00)!  
✅ Exporte os dados em planilhamento `*.CSV` ou similar por código  
✅ Faça do Compartilhamento de Conhecimentos Para Seus Familiares, Professores, Avaliadores, Amigos ou Co-Criadores Através do Envio Direto de Seus _Scripts_ do Colab.

**Não Precisa Ser Necessáriamente Baixado, Benchmark Dentro de Seu Atual Navegador!** 🌐
