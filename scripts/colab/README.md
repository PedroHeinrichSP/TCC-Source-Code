# Guia de Scripts para Google Colab

Este guia foca em uma execucao simples e reproduzivel no Colab.

## Inicio rapido

1. Abra o Google Colab.
2. Ative GPU em Runtime > Change runtime type > Hardware accelerator = GPU.
3. Use um dos fluxos abaixo.

## Fluxo recomendado: notebook pronto

Use o notebook dedicado ao Colab:

- `notebooks/nvs_benchmark_colab.ipynb`

Ele executa:

- clone do repositorio
- instalacao do pacote
- benchmark rapido
- geracao de relatorio HTML
- download dos artefatos

## Fluxo por script (shell)

Se preferir executar por comandos no Colab:

```python
!git clone https://github.com/SEU_USUARIO/SEU_REPOSITORIO.git /content/TCC
%cd /content/TCC
!bash ./scripts/colab/setup_colab.sh
!bash ./scripts/colab/run_colab.sh
```

## Scripts disponiveis

| Script | Funcao |
|---|---|
| `scripts/colab/setup_colab.sh` | Atualiza pip, instala projeto e valida status da CLI |
| `scripts/colab/run_colab.sh` | Executa status, checks e gera relatorio HTML |

## Onde ficam os resultados

- metricas: `artifacts/metrics/`
- relatorios: `artifacts/reports/`
- logs de execucao: `logs/runs/`

## Limites no Colab

- A preview 3D interativa com Viser pode nao funcionar de forma estavel no sandbox do Colab.
- Para Colab, prefira relatorios HTML e analise offline dos artefatos.

## Documentacao relacionada

- `README.md`
- `scripts/README.md`
- `notebooks/README.md`
