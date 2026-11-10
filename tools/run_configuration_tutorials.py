"""Execute the public tutorial notebooks against this checkout, preserving outputs."""
from pathlib import Path
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'validation'
os.environ.update(PYTHONPATH=str(ROOT), MPLCONFIGDIR=str(OUT / 'mpl'),
                  OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
os.environ['JUPYTER_PATH'] = str(OUT / 'jupyter')
os.environ['JUPYTER_RUNTIME_DIR'] = str(OUT / 'jupyter/runtime')

import nbformat
from nbclient import NotebookClient

kernel = OUT / 'jupyter/kernels/qasp-validation'
kernel.mkdir(parents=True, exist_ok=True)
(OUT / 'jupyter/runtime').mkdir(exist_ok=True)
(kernel / 'kernel.json').write_text(json.dumps({
    'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
    'display_name': 'QASP configuration validation', 'language': 'python',
    'env': {'PYTHONPATH': str(ROOT)},
}))
results = []
for source in sorted((ROOT / 'tutorials').rglob('*.ipynb')):
    relative = source.relative_to(ROOT / 'tutorials')
    destination = OUT / 'executed_tutorials' / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    notebook = nbformat.read(source, as_version=4)
    result = {'notebook': str(relative)}
    started = time.monotonic()
    try:
        NotebookClient(notebook, timeout=240, kernel_name='qasp-validation',
                       resources={'metadata': {'path': str(destination.parent)}}).execute()
        result['status'] = 'passed'
    except Exception as exc:
        result.update(status='failed', error=str(exc))
    result['seconds'] = time.monotonic() - started
    result['images'] = sum('image/png' in output.get('data', {})
                          for cell in notebook.cells for output in cell.get('outputs', []))
    nbformat.write(notebook, destination)
    results.append(result)
    (OUT / 'tutorials.json').write_text(json.dumps(results, indent=2))
    print(relative, result['status'], round(result['seconds'], 1), flush=True)
sys.exit(int(any(r['status'] != 'passed' for r in results)))
