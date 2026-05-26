# Notebooks

Este diretorio concentra exemplos e analises do benchmark.

## Principais notebooks

- `nvs_benchmark_local_pc.ipynb`
- `nvs_benchmark_local_blender_synthetic.ipynb`
- `nvs_benchmark_colab.ipynb`

## Uso recomendado

- Local generico: `nvs_benchmark_local_pc.ipynb`
- Local focado em Blender Synthetic: `nvs_benchmark_local_blender_synthetic.ipynb`
- Colab: `nvs_benchmark_colab.ipynb`

## Observacao sobre `gs_dynamic`

O notebook do Colab inclui preparacao e probes para `gs_dynamic`, mas isso nao garante execucao real do `4DGaussians` no runtime padrao do Colab. Consulte [docs/4DGS_COLAB_LIMITACOES.md](/mnt/c/Users/Admin/Projetos/TCC/docs/4DGS_COLAB_LIMITACOES.md) antes de usar `gs_dynamic` em comparacoes finais.
