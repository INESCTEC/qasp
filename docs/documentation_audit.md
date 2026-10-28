# Documentation audit — 14 October 2026

This is the historical documentation audit, before the PQCBM update and planner
separation. Its coverage totals and test counts describe that earlier source
snapshot. The Deucalion planner is now maintained on its development branch;
the current package guide describes the modules shipped on `main`.

This pass reviewed all 52 Python modules under `qasp/`, added missing source
docstrings and corrected descriptions against the implementation. It added
[the package guide](package_guide.md) with six runnable examples and a complete
source inventory, and reconciled the README's module descriptions.

## Coverage

| Measure | Before | After |
| --- | ---: | ---: |
| Modules with docstrings | 38 / 52 | 52 / 52 |
| Classes/functions/methods with docstrings | 277 / 662 | 662 / 662 |
| Missing definition docstrings | 385 | 0 |

The definition count includes constructors, private helpers, nested definitions
and repeated definitions in conditional branches. It excludes notebook cells,
external dependencies, lambdas and dataclass-generated methods. Coverage is
presence-based; it does not prove semantic correctness or feature maturity.
All newly documented definitions were read to describe their actual behavior.
Public entry points include parameter/configuration semantics, shapes, returns,
mutation and limitations where relevant; small internal helpers use shorter
purpose/contract descriptions.

46 Python source files changed during this documentation pass. Earlier feature,
notebook, paper and result changes already in the working tree were preserved.
The comparison baseline is the source snapshot taken at the start of this task,
not Git HEAD.

## Verification

- All 52 modules parsed and compiled in memory without generating bytecode files.
- Removing module/class/function docstrings from the before/after syntax trees
  produced identical executable ASTs for every module. Additional edits to code
  comments clarify existing behavior; no function signatures, constants or
  executable statements changed.
- All six Python examples in `package_guide.md` executed sequentially in a shared
  namespace: circuit/gradient, TikZ source, classifier training, fidelity kernel,
  QUBO energy and PQDQN output/head shapes. They are smoke examples, not learning
  or convergence evidence.
- Existing suite: **252 passed, 1 failed out of 253**, in 42.34 seconds. The failure
  is `test_sparse_projectors_match_probs_and_gradients[lightning.kokkos]` because
  the `lightning.kokkos` plugin is unavailable. The corresponding Lightning-qubit
  test passed. This pass did not install a backend or alter tests to hide the
  environment limitation.
- The test run used Python 3.13 in `/private/tmp/qasp-slurm-venv`, an Agg Matplotlib
  backend and one OMP/OpenBLAS/MKL thread. Dependency deprecation warnings remain.

The test command was:

```bash
PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/qasp-mpl \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/private/tmp/qasp-slurm-venv/bin/python -m pytest tests/test_*.py -q
```

[Machine-readable audit](documentation_audit.json) records per-module coverage,
source hashes and the exact failed-test message. The hashes identify source at
this pass; future code/documentation edits can legitimately change them.

## Kokkos verification in `torchlane`

After the user identified the intended Conda environment, the previously failing
`test_sparse_projectors_match_probs_and_gradients[lightning.kokkos]` was rerun
using `conda run -n torchlane python`: **1 passed in 29.81 seconds**.
Device creation also succeeded. This confirms the earlier failure was caused
by the missing plugin in the temporary test environment; the tested Kokkos
probability and gradient comparison works in `torchlane` without code changes.

This environment uses Python 3.10.17, PennyLane 0.42.2, Lightning and
Lightning-Kokkos 0.42.0, Torch 2.7.0 and pytest 9.1.1. The earlier full-suite
result remains recorded above for provenance. Only the previously failing test
was rerun here, so this is not a claim of 253 tests passing in one environment.

## Material corrections and implementation boundaries

These are documentation findings, not fixes to the underlying implementation:

1. **Construction:** `build()` returns a plain quantum function; `compile()`
   chooses the backend/QNode. Input encoding belongs to the ansatz. Corrected
   outdated parameter-shape formulas, measurement names and initializer labels.
2. **Readouts:** classifier groups use means; PQDQN per-action groups use sums;
   shared/preset Q-function features are individually fed to an affine head.
   Preset decomposition size is independent of the environment action count.
3. **Policy contracts:** the factory's `PQCBorn` selector constructs
   `PQCBornVariance`. `PQCSoftmax.forward()` returns logits and includes the head;
   the legacy `action_probs()` path bypasses it. The current softmax head is
   biased, and its None/non-None convention differs from PQDQN's boolean flag.
4. **PP:** `lightning.pp` is an internal QASP route. Hard pruning differentiates
   retained coefficients, not selection masks; it is not a straight-through
   mask estimator. Approximation need not preserve gradients or learning.
5. **Diagnostics:** the current diagnoser measures cost/output concentration,
   not gradient variance. `fourier_analysis` performs no Fourier transform;
   legacy decomposition lists are heuristic generators, not executable gates.
6. **Trainer differences:** documented REINFORCE's legacy CartPole freezing
   rule, the distinct `Reinforce` implementation, PPO's rollout resets and
   truncation masks, and DQN's frame/update units and separate evaluation env.
7. **Generative learning (superseded by the PQCBM update):** the original
   audit recorded default-loss and natural-gradient failures. The subsequent
   `genlearn` implementation fixes both, introduces `PQCBM`, expectation MMD,
   drawing and device sampling, and adds numerical tests. See
   [the current genlearn guide](genlearn.md) and
   [its validation record](genlearn_validation.md). `BornMachine` is now a
   compatibility wrapper; its integer-sample default is retained.
8. **Historical paths:** `UQC`, adaptive `LearnablePQCBorn`, `rl.trl`, legacy
   policy setters and the Fisher kernel have specific restrictions recorded
   in the guide/source. `QFKernel.normalize` is unused.
9. **Optimization/HPC:** candidate energy differs from optimized expectation or
   PCE surrogate; optimizer status does not certify a global binary optimum.
   Planner memory estimates and dated profiles are not measured runtime/memory
   guarantees, and Slurm generation does not submit jobs.

The cancelled ten-qubit PP-vs-Lightning 400-episode learning comparison remains
cancelled with partial data preserved. No experiments were launched, resumed or
modified as part of this documentation pass. Completed earlier experiments must
retain their original implementation/settings provenance when cited in the paper.

## PQDQN integration on planner-free main

The prepared integration was checked in an isolated checkout in `torchlane`:
**256 tests passed**, with no failures or skips and one expected deprecated
`BornMachine` warning. The selection was `tests/test_*.py`, including PQDQN,
DeepQLearning, Pauli propagation, drawings, classifiers, kernels, optimization,
genlearn and MPI routing. No multi-node or long learning run was repeated.

All six package-guide Python examples executed successfully as written. The
first example imports QASP before Torch to avoid a duplicate OpenMP runtime
initialization observed with Torch-first imports in this Conda environment.
Current source coverage is 49/49 modules and 657/657 definitions with docstrings.
All selected Python files compile, and the seven selected notebooks have no
saved error outputs. Existing experiment results are preserved; the cancelled
ten-qubit CartPole comparison retains its explicit cancellation notice and
partial-result status. Planner imports, examples and guide sections are absent
from the current distribution.
