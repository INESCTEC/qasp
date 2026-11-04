#!/usr/bin/env python3
"""Run the complete suite against the current interpreter and record evidence.

By default tests import the installed QASP wheel from a temporary directory.
Use --source explicitly when checking a development checkout (e.g. torchlane).
"""
from __future__ import annotations
import argparse
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source', action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    env.update({name: '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')})
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['MPLCONFIGDIR'] = str(output / 'matplotlib')
    env['MPLBACKEND'] = 'Agg'  # Drawing checks must not open macOS GUI windows.
    env['NUMBA_CACHE_DIR'] = str(output / 'numba')
    if args.source:
        env['PYTHONPATH'] = str(repo)
    report = {'python': platform.python_version(), 'platform': platform.platform(),
              'mode': 'source checkout' if args.source else 'installed wheel',
              'packages': dict(sorted((d.metadata['Name'].lower().replace('_', '-'), d.version)
                                     for d in metadata.distributions()))}
    with tempfile.TemporaryDirectory(prefix='qasp-validation-') as cwd:
        probe = subprocess.run([sys.executable, '-c',
            'import qasp; print(qasp.__file__)'], cwd=cwd, env=env,
            capture_output=True, text=True)
        report['import_exit_code'] = probe.returncode
        report['import_path'] = probe.stdout.strip()
        (output / 'import.log').write_text(probe.stdout + probe.stderr)
        if not args.source and probe.returncode == 0:
            assert not Path(probe.stdout.strip()).is_relative_to(repo), 'Source checkout masked installed wheel'
        check = subprocess.run([sys.executable, '-m', 'pip', 'check'], cwd=cwd,
                               env=env, capture_output=True, text=True)
        report['pip_check'] = {'exit_code': check.returncode, 'output': (check.stdout + check.stderr).strip()}
        command = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                   str(repo / 'tests'), '--junitxml=' + str(output / 'pytest.xml')]
        start = time.monotonic()
        with (output / 'pytest.log').open('w') as log:
            result = subprocess.run(command, cwd=cwd, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=1800)
        report['pytest_exit_code'] = result.returncode
        report['elapsed_seconds'] = round(time.monotonic() - start, 2)
        if (output / 'pytest.xml').exists():
            suites = ET.parse(output / 'pytest.xml').getroot()
            report['tests'] = {key: sum(int(s.get(key, '0')) for s in suites.iter('testsuite'))
                               for key in ('tests', 'failures', 'errors', 'skipped')}
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'packages'}, indent=2))
    return int(bool(report['import_exit_code'] or report['pip_check']['exit_code'] or report['pytest_exit_code']))


if __name__ == '__main__':
    raise SystemExit(main())
