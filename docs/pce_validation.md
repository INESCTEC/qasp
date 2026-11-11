# PCE implementation validation

Validated on 11 November 2026 using Python 3.12.8, PennyLane 0.45.1,
Lightning 0.45.0, Torch 2.15.0, Quimb 1.15.0 and Cotengra 0.8.3. Torch
double precision and one intra-operation thread were used for strict comparisons.

## Correctness and regressions

The package suite passed **1,990 tests**, with three skips for the uninstalled
optional Lightning Kokkos plugin. The PCE tests in
[`tests/test_pce_reference.py`](../tests/test_pce_reference.py) cover:

- The original paper's unweighted loss normalization and weighted regularizer;
  positive rescaling of graph weights; disconnected graphs, isolated vertices,
  signed-edge extension and explicit scale overrides.
- Encoding orders 1–4, multiple axis subsets, capacity, uniqueness and order;
  order-dependent default alpha through order five; finite hyperparameters.
- Independent dense NumPy circuit evolution and finite-difference gradients,
  with QUBO soft-spin energy checked through a Bernoulli reference calculation.
- NumPy/Autograd and Torch on default.qubit and lightning.qubit; exact
  PauliPropagationSimulator; PennyLane default.tensor MPS and TN with
  parameter-shift; compatible IQP TN, VN enumeration and auto execution.
- Sampled VN's empirical loss gradient with a fixed sample bank, including
  equivalent one-worker and four-worker results and parallel backward execution.
- Torch training, result-energy consistency, zero-objective gradients,
  finite-shot gradients and consistency between recorded correlators and raw decoding.
- One-pass classical improvement against naive full-cost recomputation on all
  bitstrings of three random five-variable QUBOs, including linear terms and
  nonsymmetric matrices. Disabling postprocessing preserves the raw output.
- TikZ readout labels retain alternative measurement bases rather than
  overwriting them. The drawing suite and rendered notebook were checked after
  increasing PCE's default row separation to accommodate those labels.

There are no expected-failure markers in the PCE regression tests. The earlier
audit's normalization, weighted-scale, postprocessing, alpha and nonfinite-input
findings are now covered by passing regression tests.

## Executed examples

The updated optimization algorithms notebook and the website's complete PCE
workflow executed successfully. The new
[PCE–QAOA tutorial](../tutorials/optimization/pce_vs_qaoa.ipynb) completed all
cells, including Matplotlib and inline TikZ rendering. Its independent
simulator checks found maximum absolute loss differences below 1.3e-14 and
gradient differences below 1.7e-9 relative to statevector finite differences
on the documented small circuits.

The tutorial uses three fixed graphs, three random initializations per graph,
and 60 Adam updates for each method. Its result summaries were:

| Instance | PCE qubits / QAOA qubits | PCE raw mean ratio | PCE improved mean ratio | QAOA raw / improved mean ratio |
| --- | --- | --- | --- | --- |
| Four-cycle | 3 / 4 | 1.0000 | 1.0000 | 1.0000 / 1.0000 |
| Six-vertex chorded ring | 3 / 6 | 1.0000 | 1.0000 | 1.0000 / 1.0000 |
| Weighted random, eight vertices | 3 / 8 | 0.9301 | 0.9624 | 1.0000 / 1.0000 |

Ratios divide the candidate cut by an independently enumerated exact optimum.
Every restart is retained; both methods receive the same single-bit sweep.
The PCE and QAOA circuits have different parameter counts (16 versus 4) and
different output rules. These small demonstrations do not establish a general
performance or learning advantage. Timing columns report local observations,
with cold loss/gradient warmup separated from optimizer plus readout time;
they are not a controlled simulator performance benchmark.

The repository notebook has cleared outputs. An executed local copy, six
rendered figures, test logs and JUnit reports are retained in
`validation/pce-audit-2026-11-11/` in the working checkout. The legacy audit
report in that directory describes the pre-fix release and is retained as
provenance, not the current implementation status.

## Scope

The implementation follows the paper's encoding, MaxCut loss and final
single-bit improvement protocol. It retains QASP's customizable ansatz and
optimizer; it does not claim to reproduce the paper's large-instance results.
Analytic exact comparisons do not validate approximate tensor truncation,
Pauli pruning or the statistical accuracy of finite-shot/Monte Carlo training.
Those change the estimated nonlinear objective and require separate studies.

Reference: [Sciorilli et al., Nature Communications 16, 476 (2025)](https://doi.org/10.1038/s41467-024-55346-z).
