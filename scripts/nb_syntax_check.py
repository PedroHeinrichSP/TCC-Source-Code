import json
import sys
from pathlib import Path
import py_compile

nb_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('notebooks/nvs_benchmark_colab.ipynb')
out_path = Path('scripts/nvs_benchmark_colab_script.py')

nb = json.loads(nb_path.read_text(encoding='utf-8'))
with out_path.open('w', encoding='utf-8') as out:
    out.write('# Auto-generated script from notebook: ' + str(nb_path) + '\n')
    for cell in nb.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue
        out.write('\n# ---- cell ----\n')
        src = cell.get('source', [])
        # src is a list of strings; write as-is
        for line in src:
            out.write(line)
        out.write('\n')

print('Wrote script to', out_path)

try:
    py_compile.compile(str(out_path), doraise=True)
    print('Syntax check: OK')
except py_compile.PyCompileError as e:
    print('Syntax check: FAILED')
    print(e)
    # print the file with line numbers to help debugging
    txt = out_path.read_text(encoding='utf-8')
    for i, l in enumerate(txt.splitlines(), start=1):
        print(f'{i:04d}: {l}')
    sys.exit(2)


