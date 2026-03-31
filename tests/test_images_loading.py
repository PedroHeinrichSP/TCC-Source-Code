#!/usr/bin/env python
"""Teste para verificar se as imagens estão sendo carregadas corretamente."""

from pathlib import Path
from nvs_benchmark.ui.preview import _resolve_artifacts_root, _find_method_image

# Teste com caminho absoluto
metrics_file = r'C:\Users\Admin\Projetos\TCC\artifacts\metrics\latest_preview.json'
artifacts_root = _resolve_artifacts_root(metrics_file)
print(f'artifacts_root: {artifacts_root}')
print()

for method in ['nerf_static', 'nerf_dynamic', 'gs_static', 'gs_dynamic']:
    render_path = _find_method_image(artifacts_root, method, 'renders')
    ref_path = _find_method_image(artifacts_root, method, 'references')
    render_status = 'Found' if render_path else 'NOT FOUND'
    ref_status = 'Found' if ref_path else 'NOT FOUND'
    print(f'{method:15} | render: {render_status:9} | ref: {ref_status:9}')
    if render_path:
        print(f'  → {render_path}')
    if ref_path:
        print(f'  → {ref_path}')
