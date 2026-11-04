# Dependency compatibility

Validated on **4 November 2026**, on an Apple Silicon Mac running macOS 27.
The table describes specific tested combinations, not every possible combination
inside the version ranges in [pyproject.toml](../pyproject.toml).

## Tested installations

Each clean environment was created without access to system site-packages,
installed QASP from a wheel, and passed `python -m pip check`. Tests ran from a
temporary directory to prevent the repository from hiding the installed package.

| Profile | Python | PennyLane | PyTorch | NumPy | TorchRL / TensorDict | Result |
|---|---|---|---|---|---|---|
| [Lower bounds](../constraints/py310-lower.txt) | 3.10.17 | 0.42.3 | 2.6.0 | 1.26.4 | 0.7.2 / 0.7.2 | 746 passed, 3 skipped |
| [Current Python 3.10](../constraints/py310-current.txt) | 3.10.17 | 0.42.3 | 2.14.1 | 2.2.6 | 0.14.1 / 0.14.3 | 746 passed, 3 skipped |
| [Intermediate versions](../constraints/py311-middle.txt) | 3.11.11 | 0.43.3 | 2.8.0 | 2.3.5 | 0.9.2 / 0.9.1 | 746 passed, 3 skipped |
| [Current Python 3.12](../constraints/py312-current.txt) | 3.12.8 | 0.45.1 | 2.15.0 | 2.5.4 | 0.14.1 / 0.14.3 | 746 passed, 3 skipped |

The three skips are tests requiring the separately installed
`pennylane-lightning-kokkos` plugin. The unchanged `torchlane` reference
installation also passed **749 tests**, including those Kokkos tests. It uses
PennyLane 0.42.2, PyTorch 2.7.0 and NumPy 2.2.6, but is **not** a recommended fresh
installation: `pip check` reports two pre-existing PennyLane-Qiskit conflicts,
and PennyLane 0.42.2 predates the safer lower bound used here. It was not modified.

The [machine-readable record](dependency_compatibility.json) contains versions,
counts, failure investigations and validation scope. The constraint snapshots
also pin transitive and development dependencies. They are reproducibility
profiles for these Python versions on macOS arm64; they are not portable lockfiles
or a requirement to install every listed package.

## Reproduce a profile

From the repository root, create a fresh environment with the Python version
specified by the profile. For example:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -c constraints/py312-current.txt '.[dev]'
python tools/check_compatibility.py --output /tmp/qasp-compatibility-results
```

Use `py310-lower.txt` or `py310-current.txt` with Python 3.10, and
`py311-middle.txt` with Python 3.11. The runner verifies imports, runs `pip check`,
executes the complete regression suite and writes JSON, JUnit XML and logs.
It exits unsuccessfully for an import, dependency or test failure. For an
explicit source-checkout check in an existing environment, add `--source`.
Do not use that flag when validating a built wheel.

The `current` filenames mean the snapshots taken on the date above. They do not
silently follow new releases. To explore versions within QASP's declared ranges,
install without `-c` in a separate environment and rerun the checker before
claiming compatibility or replacing a snapshot.

## Bounds and observed failures

- **PennyLane >=0.42.3,<0.46.** The 0.42.3 patch adds the Autoray `<0.8`
  constraint missing in 0.42.2. Tested PennyLane lines are 0.42, 0.43 and 0.45;
  0.44 was not separately tested. Python 3.10 resolves to 0.42.3 because newer
  PennyLane releases require Python >=3.11.
- **Autograd >=1.8,<1.9.** Autograd 1.9.1 with PennyLane 0.42.3 caused three
  generative-learning/input-scaling gradient tests to fail, even though
  `pip check` passed. Autograd 1.8.0 passes. Current PennyLane also constrains
  Autograd below 1.9; see its [0.45.1 release notes](https://docs.pennylane.ai/en/stable/development/release_notes.html#release-0-45-1-current-release).
- **SciPy on Python 3.10 / macOS arm64.** The 1.15.2 and 1.15.3 wheels failed
  during import on macOS 27 with a Mach-O `__thread_bss` error in PROPACK.
  SciPy 1.14.1 loads and passes. Package metadata therefore conservatively uses
  `<1.15` for that Python/platform combination. This does not assert that the
  same wheels fail on every macOS release. Other platforms/Python versions use
  the broader tested release window; SciPy 1.16.3 and 1.18.1 pass in the
  Python 3.11/3.12 profiles.
- **Other packages.** Lower bounds match the older tested profile. Upper bounds
  stop at the next unvalidated release line beyond this matrix. These bounds
  express the current validation window, not evidence that every excluded newer
  version is broken. Dependency resolvers must also satisfy the libraries' own
  constraints; in particular, keep TorchRL and TensorDict paired as recorded.
  Ray's `rllib` extra selects Gymnasium 1.0.0, 1.1.1 or 1.2.2 in these profiles.

## Test changes and coverage

The added integration test routes a real QASP policy through TensorDict and a
TorchRL probabilistic actor, checks probabilities, and backpropagates through an
Adam update on `default.qubit`, `IQPSimulator` and `PauliPropagationSimulator`.
The rest of the suite covers simulators, gradients, initializers, classifiers,
kernels, generative learning, optimization and maintained RL training paths.

Three test assumptions were corrected without changing simulator algorithms:

- Kokkos tests now check for the Kokkos plugin itself, rather than assuming it
  comes with the base Lightning installation. An installed plugin that fails
  execution still fails its tests.
- PennyLane 0.45 returns a complex-typed statevector reference for an overlapping
  product of commuting projectors. The test asserts a zero imaginary part, then
  compares real values and gradients. QASP's result must remain real.
- The cross-backend Adam comparison uses `eps=1e-6` to avoid amplifying round-off
  in mathematically zero gradients. Observed gradient differences were around
  `1e-14`, while default Adam epsilon produced parameter differences around
  `3e-9`. Existing value, gradient and parameter tolerances remain unchanged.

This validation covers CPU execution on one Mac. It does not establish Windows,
Linux, CUDA, MPI multi-rank or Ray-cluster compatibility. MPI-routing tests use
mocks; Ray is installed/imported but no distributed training job was run.
PennyLane's optional MPS/TN device dependencies (`quimb`, etc.) are not validated
by this matrix; QASP's own TN backend is covered.

Historical `qasp.rl.trl` classes remain prototypes, as documented in the
[package guide](package_guide.md). The separate external
`qasp.rl.torchrl.TorchRLTrainer` adapter is also outside the supported training
matrix: its old TorchRL 0.7.2/Gymnasium 1.0 combination rejects environment
construction, and the source references legacy APIs. Passing the maintained
REINFORCE/DQN tests or the actor/gradient integration test does not certify those
legacy trainer classes.
