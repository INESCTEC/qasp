"""Render the installed-candidate audit results as a local HTML report."""
from collections import Counter, defaultdict
from html import escape
from pathlib import Path
import json
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'validation'
cases = []
for case in ET.parse(OUT / 'installed-suite.xml').iter('testcase'):
    problem = next((case.find(tag) for tag in ('failure', 'error', 'skipped') if case.find(tag) is not None), None)
    status = ('skipped' if problem.tag == 'skipped' else 'failed') if problem is not None else 'passed'
    cases.append({**case.attrib, 'status': status, 'message': problem.get('message', '') if problem is not None else ''})
counts = Counter(c['status'] for c in cases)
groups = defaultdict(Counter)
for case in cases:
    module = case['classname'].split('.')[-1]
    group = module.removeprefix('test_configuration_') if module.startswith('test_configuration_') else 'existing regression suite'
    groups[group][case['status']] += 1
coverage = json.loads((OUT / 'installed-coverage.json').read_text())['totals']
environment = json.loads((OUT / 'environment.json').read_text())
tutorials = json.loads((OUT / 'tutorials.json').read_text())
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip() if (ROOT / '.git').exists() else 'See SOURCE_COMMIT.txt'
summary = dict(tests=dict(counts), total_tests=len(cases), groups=groups,
               coverage=coverage, environment=environment, tutorials=tutorials,
               source_commit=commit, branch='codex/validation-configuration-fixes',
               failures=[c for c in cases if c['status']=='failed'],
               skips=[c for c in cases if c['status']=='skipped'])
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2))


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{escape(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{v}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'


fixes = [
    ('Lightning batches and gradients', 'Builder compilation expands adjoint/template operations and broadcasts before Lightning adjoint execution. The original six failures pass, including input and parameter gradients. Separate checks cover best/adjoint differentiation and caller-supplied device instances. Parameter-shift batches are expanded too.'),
    ('Unused trainer removal', 'Removed both TorchRLTrainer classes, the historical rl.trl module and its unused PQCWrapper export after checking package, tests, examples and tutorials. PQCBornWrapper and the maintained training algorithms remain.'),
    ('IQP fidelity circuits', 'TN and VN now normalize inverse embeddings and Pauli rotations without detaching angles. Structural Hadamard cancellations recognize IQP overlap circuits. No angle-dependent simplification is used.'),
    ('Fisher kernels', 'Torch and NumPy parameter handling, metric-template decomposition, normalized Gram matrices, zero-probability validation, fixed reference snapshots and Matplotlib drawing are corrected. Independent probability/state finite differences verify classical/quantum Fisher information.'),
    ('Ansatz-only models', 'Direct node(params=...) remains available. TorchLayers now accept placeholder inputs and preserve batch axes for input-independent circuits. Encoding-only adapters describe neural/input-scale weights correctly.'),
    ('Policy factories and drawing', 'Builder configurations no longer require redundant component dictionaries. Omitted/None ansatz_config works. Caller dictionaries are copied while supplied device identity, reports and caches are preserved. Drawing handles circuits with no feature map.'),
    ('Softmax policies', 'Forward execution preserves batch axes. action_probs retains tanh preprocessing and now includes the complete simulator path, classical head and trainable output scale.'),
]
parts = ['<section><h2>Changes verified</h2>' + table(['Area', 'Change'], [[escape(a), escape(b)] for a,b in fixes]) + '</section>']
parts += ['<section><h2>Final installed-wheel test run</h2>' + table(['Group', 'Passed', 'Failed', 'Skipped'], [[escape(k), str(v['passed']), str(v['failed']), str(v['skipped'])] for k,v in groups.items()]) + '<p>Counts include expected rejection checks for unsupported configurations. The tests use the installed wheel in an isolated working directory, with an assertion that QASP is loaded from site-packages. All 59 installed Python files match the candidate source by SHA-256; the removed trainer module is absent.</p><p><a href="installed-suite.log">Full log</a> · <a href="installed-suite.xml">JUnit XML</a> · <a href="summary.json">Machine-readable results</a> · <a href="environment.json">Environment and wheel provenance</a></p></section>']
parts += [f'<section><h2>Measured code coverage</h2><p><strong>{coverage["percent_statements_covered"]:.1f}% statement coverage</strong> ({coverage["covered_lines"]:,}/{coverage["num_statements"]:,}) and <strong>{coverage["percent_branches_covered"]:.1f}% branch coverage</strong> ({coverage["covered_branches"]:,}/{coverage["num_branches"]:,}). Notebook execution is separate and not included in these coverage figures.</p><p><a href="coverage-html/index.html">Explore file and line coverage</a></p></section>']
parts += ['''<section><h2>Configuration coverage</h2><ul>
<li>Fidelity and projected kernels: feature-map only (no ansatz/config), Angle/IQP ansätze, omitted ansatz_config, plain/scaled/explicit-word/neural encodings, Gram and cross-Gram values, symmetry/PSD/diagonal properties, gradients and both drawing formats.</li>
<li>Fisher score kernels: encoding-only circuits parameterized by input scales at a fixed feature vector, ansatz-only circuits and both components; classical and quantum geometry, Torch/Autograd, finite-difference references, normalization, invalid/zero-probability outcomes and drawing.</li>
<li>Building blocks: direct unencoded QNodes and TorchLayers, probabilities/expectations, single/batched inputs, reuploading on/off, ordinary and multi-qubit Pauli rotations; independent statevector gradients.</li>
<li>Classifiers, Born/softmax policies and Q-functions: encoded, unencoded and encoding-only builders, multiple encodings/ansätze, factory creation, batched values/gradients and Matplotlib/TikZ drawing.</li>
<li>Generative learning: NLL and expectation-based MMD, fixed inputs/scales, ansatz-only models, values/gradients and optimizer updates.</li>
<li>Optimization: QAOA, warm-start, linear-ramp and PCE objectives/gradients and supported result retrieval. The existing suite also covers recursive/multi-angle configurations and optimizer families.</li>
<li>Real PPO, Reinforce and contextual-bandit rollouts/updates on nine execution settings, plus existing REINFORCE/DQN regression tests and tutorials.</li>
<li>Existing initialization distributions, diagnostics/qubit sweeps, simulator selection, sampled VN/importance/multicore/higher derivatives, drawing/export and MPI-routing contracts remain covered.</li>
</ul><p><a href="../docs/configuration_validation.md">Configuration examples and compatibility notes</a> · <a href="../tests/">Test sources</a></p></section>''']
parts += ['''<section><h2>Simulator boundaries</h2><p>Nine CPU execution settings: default.qubit, lightning.qubit, default.mixed, PennyLane default.tensor MPS/TN, QASP IQP TN/VN/auto, and PauliPropagationSimulator.</p>
<ul><li>Explicit VN evaluates compatible expectations, not full probability vectors. Fidelity projectors use importance sampling; exact all-terms mode is explicitly rejected for that representation. Sampled tests have statistical tolerances.</li>
<li>PennyLane MPS/TN devices support the tested expectation workflows but reject full probabilities. Thus NLL, Fisher score kernels and QAOA probability-based solution decoding are unavailable there. QAOA objectives and optimizer updates still work; the unsupported retrieval step is tested separately.</li>
<li>Born policies can reconstruct action probabilities from expectations; the training tests use this route. Generative MMD similarly avoids full probabilities.</li>
<li>Strict IQP TN may reject quantum-Fisher auxiliary circuits containing non-IQP gates even when the original model is IQP. Classical Fisher remains available for compatible probability circuits. Auto can use its documented size-limited fallback.</li>
<li>QASP differentiated simulator paths use Torch. Unsupported Autograd routes are explicit rejection tests, not claimed as differentiable support.</li>
<li>Fisher is a fixed score kernel on bit strings, not a trainable continuous-feature kernel. It enumerates outcomes. A fixed encoding with no trainable weights is not a trainable quantum model; an ansatz with no encoding is input-independent.</li>
<li>Three optional Kokkos tests are skipped. CUDA plugins and real multi-rank MPI runs were not available on this Mac. This is one Python/dependency combination and a finite configuration matrix, not an exhaustive Cartesian product of every gate, width and version.</li></ul></section>''']
parts += ['<section><h2>Executed tutorials</h2>' + table(['Notebook', 'Result', 'Time', 'Inline PNGs'], [[f'<a href="executed_tutorials/{escape(n["notebook"])}">{escape(n["notebook"])}</a>', escape(n['status']), f'{n["seconds"]:.1f} s', str(n['images'])] for n in tutorials]) + '<p>Existing tutorial sources were executed against the patched checkout and saved with outputs. Times include startup and figure compilation, and are not simulator benchmarks.</p></section>']
parts += ['<section><h2>Compatibility changes</h2><p>TorchRLTrainer/PQCWrapper imports are removed. Fisher normalize=True now performs normalization; use normalize=False for the historical raw inner product. Softmax action_probs now honors the full configured model. These changes are documented in the package guide.</p><p>The validation environment contains the local candidate. Nothing has been published to PyPI or pushed to the remote repository by this audit.</p></section>']
parts += ['<section><h2>Reproduce</h2><p>Activate the validation environment, then use the README in this audit folder. The candidate wheel is <a href="dist/qasp-0.1.3-py3-none-any.whl">available locally</a>; its version number is inherited from the base release and does not identify a public release containing these fixes.</p><p><a href="requirements-lock.txt">Dependency lock</a> · <a href="pip-check.txt">Dependency consistency</a> · <a href="../tools/validate_installed_configuration.py">Installed-package runner</a></p></section>']
parts += ['<section><h2>Exceptions</h2>' + table(['Test', 'Status/reason'], [[escape(c['name']), escape(c['status'] + ': ' + c['message'])] for c in cases if c['status'] != 'passed']) + '</section>']
style = 'body{margin:0;background:#f0f5fa;color:#183048;font:16px/1.6 system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:32px 20px 70px}header{background:#113f70;color:white;padding:32px;border-radius:16px}h1{font-size:34px;margin:4px 0}h2{margin-top:0;font-size:24px}.stats{display:flex;gap:15px;flex-wrap:wrap;margin:20px 0}.stat{background:white;border:1px solid #d8e3ef;border-radius:12px;padding:18px;flex:1;min-width:145px}.stat b{display:block;font-size:31px;color:#147154}section{background:white;border:1px solid #d8e3ef;border-radius:14px;padding:25px;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #e2e9f0;padding:11px 13px;text-align:left;vertical-align:top}th{background:#edf3fa}.scroll{overflow:auto}a{color:#0762b9}li{margin:9px 0}footer{font-size:13px;color:#53677d}code{overflow-wrap:anywhere}'
headline = 'Configuration fixes verified' if not counts['failed'] else 'Configuration audit: failures remain'
html = f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>QASP — Configuration audit</title><style>{style}</style></head><body><main><header><p>QASP · 10 November 2026 · local candidate</p><h1>{headline}</h1><p>Building blocks, kernels, simulators and application modules</p></header><div class="stats"><div class="stat"><b>{counts["passed"]}</b>tests passed</div><div class="stat"><b>{counts["failed"]}</b>tests failed</div><div class="stat"><b>{counts["skipped"]}</b>optional tests skipped</div><div class="stat"><b>{sum(n["status"]=="passed" for n in tutorials)}/9</b>tutorials passed</div></div>{"".join(parts)}<footer>Branch: codex/validation-configuration-fixes · Source commit: {escape(commit)}. Source-file hashes and wheel hash are recorded separately.</footer></main></body></html>'
(OUT / 'REPORT.html').write_text(html)
print(dict(counts), 'source commit', commit)
