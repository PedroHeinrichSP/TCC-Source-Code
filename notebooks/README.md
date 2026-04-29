# Notebooks

Este diretorio concentra notebooks para execucao e analise do benchmark.

## Notebooks principais

- `notebooks/nvs_benchmark_pipeline.ipynb`
	- Fluxo local completo: instalacao, verificacoes, geracao de relatorio e preview da UI.
- `notebooks/nvs_benchmark_colab.ipynb`
	- Fluxo focado em Google Colab: clone, setup, benchmark rapido, relatorio HTML e download dos artefatos.
- `notebooks/nvs_benchmark_colab_full_matrix.ipynb`
	- Execucao unica no Colab (one-click) para matriz completa de metodos x cenas, com consolidacao e relatorio geral.
- `notebooks/analyzing_results.ipynb`
	- Analise exploratoria das metricas geradas.
- `notebooks/how_to_use_benchmark.ipynb`
	- Guia de comandos e uso geral da CLI.

## Recomendacao por ambiente

- Local (Windows/Linux/macOS): use `nvs_benchmark_pipeline.ipynb`.
- Google Colab: use `nvs_benchmark_colab.ipynb`.

## Rastreabilidade

- Logs de execucao: `./logs/runs/<run_id>/`
- Artefatos e metricas: `./artifacts/`
- Validacao manual de UX/UI e frustums 3D: `docs/TESTS_BANCADA.md`
