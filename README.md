# NVS Benchmark

Benchmark educacional para comparacao de metodos de Novel View Synthesis.

## Fluxo Rapido

### Windows

```powershell
powershell -ExecutionPolicy Bypass -File ./scripts/windows/powershell/setup.ps1
powershell -ExecutionPolicy Bypass -File ./scripts/windows/powershell/benchmark_quick.ps1
powershell -ExecutionPolicy Bypass -File ./scripts/windows/powershell/generate_report.ps1
```

### Linux / macOS

```bash
bash ./scripts/linux/setup.sh
bash ./scripts/linux/benchmark_quick.sh
bash ./scripts/linux/generate_report.sh
```

## Estrutura

- `src/nvs_benchmark/` - biblioteca principal e CLI.
- `scripts/` - wrappers por sistema operacional e Colab.
- `configs/` - presets, catálogos e configurações.
- `data/` - datasets locais.
- `artifacts/` - métricas, relatórios e saídas geradas.
- `docs/` - documentação de fluxo e referências do projeto.
- `notebooks/` - exemplos e análise.
- `tests/` - testes automatizados.

## Comandos da CLI

- `python -m nvs_benchmark.cli init`
- `python -m nvs_benchmark.cli dataset-check --dataset blender_synthetic --root ./data/blender_synthetic/nerf_synthetic/lego`
- `python -m nvs_benchmark.cli method-run --method nerf_static --dataset blender_synthetic --root ./data/blender_synthetic/nerf_synthetic/lego --split train --compute-metrics`
- `python -m nvs_benchmark.cli metrics-check --output-dir ./artifacts --snapshot-file ./artifacts/metrics/latest_preview.json --log-dir ./logs`
- `python -m nvs_benchmark.cli report-generate --snapshot-file ./artifacts/metrics/latest_preview.json --output-dir ./artifacts/reports --report-name benchmark_report --no-pdf --log-dir ./logs`

## Nota sobre `gs_static`

O metodo `gs_static` usa a integracao oficial do Gaussian Splatting e requer ambiente compatível com CUDA para execucao real. Em hosts sem CUDA, os comandos de smoke da CLI registram `gs_static` como `skipped` por incompatibilidade de hardware, em vez de reportar sucesso falso.

## Documentacao

- [Fluxo de execucao](docs/FLUXO_DE_EXECUCAO.md)
- [Guia de scripts](scripts/README.md)
- [Guia de notebooks](notebooks/README.md)
