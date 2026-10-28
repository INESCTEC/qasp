# PQCBM validation record — 14 October 2026

The `genlearn` update introduces `PQCBM`, `create_qcbm`, `draw_qcbm`, and
`PauliMMD`, and retains `BornMachine` as a deprecated compatibility wrapper.
The current API and mathematical conventions are documented in
[genlearn.md](genlearn.md). This record supersedes the generative-learning
limitations in the earlier documentation audit.

## Automated checks

Environment: Conda `torchlane`, Python 3.10.17, PennyLane 0.42.2,
Torch 2.7.0, Lightning and Lightning-Kokkos 0.42.0. Runs used one CPU thread
for the numerical libraries and disabled bytecode output.

- **70 tests passed** after separating the Deucalion planner from `main`:
  `test_genlearn.py` (42 cases) and `test_mpi_devices.py` (28 cases).
- The run emitted one expected `BornMachine` deprecation warning from an
  existing compatibility test.
- The updated notebook MPI-configuration cell was executed successfully; the
  other seven cells retain their earlier saved experiment outputs.
- Earlier runs of 190 broader tests and 110 focused tests included the planner
  integration preserved on `codex/deucalion-slurm-planner`. Those historical
  counts do not describe the current `main` test selection.

The numerical tests compare exact expectation-mixture MMD with a dense
Gaussian kernel at nonunit bandwidths, including gradients. They compare
values, gradients, and optimizer updates for Angle and IQP ansatzes on
`default.qubit`, `lightning.qubit`, `lightning.kokkos`, `default.mixed`, and
unpruned `lightning.pp`. They also exercise Autograd/Torch NLL and MMD,
feature maps and trainable input scales, natural gradients, finite-shot
parameter-shift gradients and the independent-expectation product, sampled
mask statistics, data U-statistic correction, 70-wire PP without probabilities,
custom QNode wire order, drawings, device sampling, and deterministic MPI routing.

## Executed example

[The notebook](../examples/pqcbm_generative_learning.ipynb) contains eight
code cells, all executed in order in one IPython process with saved text and
image outputs. It calls the independently runnable
[experiment script](../examples/pqcbm_generative_learning.py).

The target assigns equal mass to `000` and `111`. The model uses three qubits,
two YZ layers with enabled CNOT chains, seed 7, Adam learning rate 0.05,
120 updates, and exact MMD with Gaussian standard deviations 0.5, 1 and 2.

| Quantity | Measured value |
| --- | ---: |
| Initial MMD² | 0.3320805126706118 |
| Final MMD² | 0.0000166783726707788 |
| Probability on target support | 0.9925771886455247 |
| Total variation to target | 0.0074228113544753 |
| Maximum initial gradient difference, PP versus statevector | 1.39e-16 |
| Maximum final probability difference, PP versus statevector | 2.22e-16 |

PP used no pruning or weight truncation. Its learned circuit was sampled
2,000 times on `default.qubit` with sampling seed 19. Full probabilities were
used only for small-system diagnostics. See the saved
[summary](../examples/results/pqcbm_small_experiment/summary.json),
[parameters](../examples/results/pqcbm_small_experiment/trained_parameters.json),
[numerical arrays](../examples/results/pqcbm_small_experiment/results.npz), and
[machine-readable validation](../examples/results/pqcbm_small_experiment/validation.json).

The [Quantikz source](../examples/results/pqcbm_small_experiment/pqcbm.tex)
compiled successfully to a
[vector PDF](../examples/results/pqcbm_small_experiment/pqcbm.pdf).
Its rendered preview and the learning/distribution plot were visually checked.

## Boundaries on claims

This validates a small experiment and local backend integration. No actual
multi-node MPI execution, GPU, tensor-network or quantum-hardware run was
performed here. The
[cluster smoke test](../tests/tests_deucalion/pqcbm_mpi_smoke.py) is supplied
for execution with an MPI-enabled Lightning installation; local routing tests
use a serial stand-in.

The implementation follows the reference paper's Pauli-Z expectation-mixture
MMD identity. It does not reproduce its specialized IQP Monte Carlo estimator,
data-dependent initialization, or thousand-qubit training experiments.
Truncated/pruned PP changes expectations and gradients and has no general
polynomial-time or unbiasedness guarantee. Bandwidths consistently mean
standard deviations as in Eq. (3) of the supplied v1 PDF; its Eq. (17) prints
inconsistent bandwidth notation, explained in the module guide.
