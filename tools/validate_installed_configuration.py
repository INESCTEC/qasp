"""Validate an installed wheel without importing the source checkout.

Build/install the candidate into the active environment first. This script
copies tests and their two example fixtures into an isolated working folder,
records package provenance, then runs the complete regression suite.
"""
from pathlib import Path
import os
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'validation'
CHECK = OUT / 'wheel-check'
(CHECK / 'tests').mkdir(parents=True, exist_ok=True)
(CHECK / 'examples').mkdir(exist_ok=True)
for source in (ROOT / 'tests').glob('*.py'):
    shutil.copy2(source, CHECK / 'tests' / source.name)
for name in ['pqdqn_cartpole_experiment.py', 'pqdqn_paper_figures.py']:
    shutil.copy2(ROOT / 'examples' / name, CHECK / 'examples' / name)
(CHECK / 'pytest.ini').write_text('[pytest]\naddopts = --import-mode=importlib\npythonpath = tests\n')
(CHECK / 'tests/conftest.py').write_text('''import numpy as np
import torch
import qasp
assert "site-packages" in qasp.__file__, qasp.__file__
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass
''')
env = dict(os.environ, PYTHONPATH=str(CHECK / 'tests'), MPLBACKEND='Agg',
           MPLCONFIGDIR=str(OUT / 'mpl'), OMP_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', NUMBA_NUM_THREADS='1')
provenance = r'''
from pathlib import Path
import hashlib, importlib.metadata as m, json, platform, sys
import numpy
import qasp
root = Path(sys.argv[1]); output = root / 'validation'
installed = Path(qasp.__file__).parent
assert 'site-packages' in str(installed), installed
source = root / 'qasp'
checks = []
for path in sorted(source.rglob('*.py')):
    relative = path.relative_to(source)
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    actual = hashlib.sha256((installed / relative).read_bytes()).hexdigest()
    checks.append({'file': str(relative), 'sha256': expected, 'matches': expected == actual})
assert all(c['matches'] for c in checks)
assert not (installed / 'rl/trl.py').exists()
wheel = next((output / 'dist').glob('*.whl'))
report = {'python': sys.version, 'platform': platform.platform(), 'qasp_location': str(installed),
          'interpreter': sys.executable, 'candidate': 'unpublished local fixes based on 0.1.3',
          'versions': {n: m.version(n) for n in ['qasp', 'torch', 'pennylane', 'pennylane-lightning', 'torchrl', 'numpy', 'quimb', 'cotengra']},
          'source_files': checks, 'wheel_sha256': hashlib.sha256(wheel.read_bytes()).hexdigest()}
(output / 'environment.json').write_text(json.dumps(report, indent=2))
print(f'Verified {len(checks)} installed source files and removal of the legacy trainer.')
'''
subprocess.run([sys.executable, '-c', provenance, str(ROOT)], cwd=CHECK, env=env, check=True)
with (OUT / 'requirements-lock.txt').open('w') as output:
    subprocess.run([sys.executable, '-m', 'pip', 'freeze'], cwd=CHECK, env=env, stdout=output, check=True)
(OUT / 'requirements-dependencies.txt').write_text(''.join(
    line for line in (OUT / 'requirements-lock.txt').read_text().splitlines(keepends=True)
    if not line.lower().startswith(('qasp==', 'qasp @'))
))
with (OUT / 'pip-check.txt').open('w') as output:
    subprocess.run([sys.executable, '-m', 'pip', 'check'], cwd=CHECK, env=env, stdout=output, check=True)
command = [sys.executable, '-m', 'pytest', 'tests', '-q', '--timeout=120',
           f'--junitxml={OUT / "installed-suite.xml"}', '--cov=qasp', '--cov-branch',
           f'--cov-report=json:{OUT / "installed-coverage.json"}',
           f'--cov-report=html:{OUT / "coverage-html"}']
with (OUT / 'installed-suite.log').open('w') as output:
    result = subprocess.run(command, cwd=CHECK, env=env, stdout=output, stderr=subprocess.STDOUT)
print(f'pytest exit code {result.returncode}; results: {OUT / "installed-suite.log"}')
sys.exit(result.returncode)
