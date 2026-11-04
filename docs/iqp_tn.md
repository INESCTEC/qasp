# Exact IQP tensor-network simulation

Research reproduction commands and recorded benchmark discussions in this guide refer to the [pinned upstream checkout](https://github.com/andre-sequeira10/QASP/tree/3bda8622bd6f5ddfb7c05a0dfd0538c1b0ccd718). This distribution keeps the runtime API and compact tutorials; see the [repository layout policy](repository_layout.md).

For the unified exact/sampled device, see [IQPSimulator](iqp_simulator.md). It adds a Torch Van den Nest correlator estimator and automatic selection; this legacy TN device keeps its existing behavior.

`IQPTNSimulator` is a PennyLane device integrated with QASP's device factory and
`CircuitBuilder`. It contracts a doubled circuit **and its requested measurement**.
It supports arbitrary diagonal interaction graphs and hypergraphs, rather than a
fixed chain of three-qubit blocks. The compact Born-policy path produces action
probabilities directly and does not allocate a basis-to-action table.

This implementation is based on `main` commit
`60433c9543a755a7a41acda422b389457c10071c`, on branch
`codex/iqp-tn-simulator`. It does not include the separate circuit-cutting feature.

## Choosing a backend

```python
from qasp import IQPTNSimulator, IQPTNOptions

# QASP builders and models accept these device arguments:
device = "auto"
device = "IQPTNSimulator"  # alias: "iqp.tn"
device = IQPTNSimulator   # factory creates a device with the caller's wires/shots
device = IQPTNSimulator(wires=5, options=IQPTNOptions(
    auto_max_flops=100_000_000,
    auto_max_memory_bytes=256 * 1024**2,
))

node = builder.compile(device="auto", diff_method="backprop")
result = node(inputs=features, params=weights)
print(node.device.last_report)
```

Selection is deferred until an actual parameterized tape exists. `auto` inspects
operations, wire labels, requested measurements/decoder, the numerical interface,
batch size, precision and gradient requirements. Supported, affordable plans use
TN contraction; unsupported or over-budget plans use `default.qubit`. The report
explains why. Automatic MPI selection retains `lightning.kokkos`; an explicit TN
request with MPI fails clearly. Explicit requests for other plugins are unchanged.

**Explicit TN execution ignores the automatic budgets.** It reports the estimate,
warns when the estimate exceeds those budgets, and attempts the exact contraction.
It never silently switches to a statevector or truncates tensors. Optional
`hard_max_flops` and `hard_max_memory_bytes` reject estimated costs even in forced
mode; neither is enabled by default. These are estimate limits, not OS memory caps.
A real resource failure remains an error. Test forced behavior with small circuits
and deliberately tiny automatic budgets, as in the example below.

`fallback_max_wires=24` prevents automatic fallback from accidentally allocating a
huge statevector. Exceeding it gives an explanatory error; raise it explicitly only
when the fallback allocation is intended. `auto` is not a guarantee that any
circuit will fit available memory. `device_kwargs` carries TN options through QASP
APIs that expose it; alternatively pass a configured device instance. Existing
instances retain their own wires and shots, following the existing factory contract.

## Supported circuit contract

The input is a constructed PennyLane `QuantumScript`, not arbitrary unanalyzed
Python code. Gates must admit this representation, allowing commuting operations
on separate wires to appear interleaved:

\[
 |\psi\rangle = \left(\bigotimes_w V_w\right)
 D\left(\bigotimes_w U_w\right)|0\cdots0\rangle,
 \qquad D=\prod_e D_e.
\]

Supported multiwire diagonal operations are `IsingZZ`, `MultiRZ`, `PauliRot` with
only `I` and `Z` letters, `CZ`, and `ControlledPhaseShift`. Hyperedges can have any
size; Z-string rotations are decomposed into rank-three parity tensors, never a
`2**k` gate matrix. Identity letters are excluded from the phase parity. Angles
may be trainable, shared across gates, input-dependent, or exactly zero.

Commuting rotations in a shared **product Pauli basis** are also accepted:
`IsingXX`, `IsingYY`, and Pauli strings with a consistent nonidentity X/Y/Z axis
on each wire. This includes `AngleAnsatz` RX/RXX/RXXX families, matching feature
encoding, and multiple such commuting layers. The template name does not determine
eligibility. A repeated IQPAnsatz can still introduce noncommuting Hadamards.

For each interacting wire choose \(B_w P_w B_w^\dagger=Z\):
\(B_w=H\) for X, \(B_w=R_X(\pi/2)\) for Y, and identity for Z. With
\(B=\bigotimes_w B_w\),

\[
U_{\rm post}\left[\prod_e e^{-i\theta_e P_e/2}\right]U_{\rm pre}
=(U_{\rm post}B^\dagger)
\left[\prod_e e^{-i\theta_e Z_e/2}\right](B U_{\rm pre}).
\]

The simulator absorbs these fixed local changes into the existing preparation
and readout tensors and passes live angles unchanged to Z-string rotations.
Single-wire gates between interactions must commute with that wire's selected
axis. Compatible rotations become RZ gates; arbitrary supported local gates
before/after the interval retain their order. No large gate matrix, parameter
specialization or statevector is introduced. Traced inference and eager Torch
backpropagation use the same normalization, including higher derivatives.

This requires a product basis, not merely pairwise commutation: XX and YY on the
same two wires commute but require an entangling diagonalization, so explicit TN
execution rejects that family and auto may fall back. The existing Z-diagonal
normalization and resource guards remain in place.

Local dressing accepts `Hadamard`, the Pauli gates, `Identity`, `S`, `T`, `SX`,
`RX/RY/RZ`, `Rot`, `PhaseShift`, one-wire `PauliRot`/`MultiRZ`, one-wire
`QubitUnitary`, and their adjoints. Local diagonal gates may occur between
interactions on that wire. A noncommuting local gate between two multiwire
interactions is rejected: for example, `RZZ; H; RZZ` does not become one IQP layer
merely because the template is named `IQPAnsatz` or an angle happens to be zero.
Local phases outside the interaction interval retain their order with respect to
local dressing. Barriers and global phases have no effect on supported outputs.
`AngleEmbedding` and `IQPEmbedding` are expanded into their actual gates before
validation; multiple noncommuting rounds remain subject to the same checks.

Unsupported examples include CNOT networks, generic multiwire unitaries,
entangled state preparation, channels/noise, mid-circuit measurements and control,
and unsimplified adjoints of multiwire interactions. Equivalent circuits expressed
in unsupported primitives may therefore fall back even if a separate algebraic
rewrite could put them into the supported form. There is no numerical or
parameter-value-based gate elimination. Initial wires start in `|0>`; arbitrary
product states can be supplied using supported local unitaries.

## Readout is part of the network

```python
import pennylane as qml
from qasp import IQPTNSimulator, DecoderSpec, action_probs

@qml.qnode(IQPTNSimulator(wires=5), interface="torch", diff_method="backprop")
def policy(theta):
    for w in range(5):
        qml.Hadamard(w)
        qml.RZ(theta[w], w)
    qml.MultiRZ(theta[5], wires=[0, 2, 4])
    for w in range(5):
        qml.Hadamard(w)
    return action_probs(
        wires=[4, 1, 0, 3],
        decoder=DecoderSpec("majority-modulo", 3, blocks=((0, 1), (2, 3))),
    )
```

Decoder block entries above are **positions in the measured-wire list**. This
low-level ordering is preserved. QASP `PQCBornVariance.majority_blocks` instead
uses physical wires and resolves them to positions before constructing the
measurement. `CircuitBuilder` retains its existing sorted measurement-wire order.

| Readout | Exact construction | Output / limitations |
|---|---|---|
| Global majority | Saturating count automaton | Two actions; ties go to zero |
| Majority-modulo | One count automaton per block, then modular code accumulation | Arbitrary action count; even/odd and overlapping blocks at the low-level API; collisions aggregate exactly |
| Parity / recursive parity | XOR automata for successively shorter prefixes, then modular accumulation | Same prefix order as QASP's parity/hidden decoder |
| Contiguous bins | Lexicographic comparator automata for bin boundaries; difference of CDFs | Near-balanced bins; first remainder bins are larger; O(actions × measured wires) comparator tensors |
| `qml.probs(wires=...)` | Leave measured output bits open, trace the others | Marginals supported; **full probability output still has `2**n` entries** |
| Pauli expectations/sums | Local measurement tensors with Pauli insertions | Analytic, one network per Pauli term |
| Variance | Compute `E[O²] - E[O]²` | Analytic; squaring a Pauli sum can increase the term count |
| Bit samples/counts | Sequential conditional contractions | Finite shots; no dense distribution required; `all_outcomes=False` |

`DecoderSpec("majority")` gives binary global majority.
`DecoderSpec("parity", A)` and `DecoderSpec("recursive-parity", A)` are aliases
of `parity-like`; `contiguous` aliases `contiguous-like`. Samples can be decoded
with `spec.decode(bits)`, without enumerating the basis and without overflowing a
64-bit integer on long bit strings. `spec.aggregate(probabilities)` is explicitly
a **dense** fallback utility and uses O(`2**n`) storage.

Arbitrary Python decoder functions are not inferred as compact tensors. Use a
supported `DecoderSpec`, or explicitly request `qml.probs` and apply a user-defined
dense aggregation outside the device. The latter exposes the exponential output
size to the planner; it is not a compact-decoder claim. New compact readouts can
extend the automaton compiler in `qasp/simulators/iqp_tn.py`.

`ActionProbsMP` belongs to this backend and its automatic fallback adapter. Using
it directly on an unrelated PennyLane device raises instead of silently returning
basis probabilities with the wrong shape. During fallback the adapter requests
ordinary basis probabilities and applies the same decoder.

`PQCBornVariance(training_method="probs", device="auto" or "IQPTNSimulator")`
automatically queues compact action readout and skips its exponential table.
Existing policy losses, samplers, training algorithms, and explicit other-device
routes are retained. Classifier and legacy Born-model probability decoders continue
to request their ordinary dense basis outputs; those requests are costed as dense.
A 45-qubit low-width regression uses the public `PQCBornVariance` route and makes
both table creation and statevector fallback fail the test if invoked.

## Contraction and differentiation

For each wire introduce ket/bra computational bits `z_w,z'_w` and output bit `y_w`.
The local tensors are

\[
 A_w(z,z')=(U_w)_{z0}(U_w)^*_{z'0},\qquad
 B_w(z,z',y)=(V_w)_{yz}(V_w)^*_{yz'}.
\]

A diagonal interaction contributes `D_e(z_e) D_e(z'_e)*`. For a Z-string rotation,

\[
 D_e(z_e)=\exp[-i\theta_e(-1)^{\oplus_{w\in e}z_w}/2].
\]

The XOR is implemented with tensors `T(a,b,c)=1[c=a xor b]`, followed by a length-two
phase vector. This has O(|e|) tensors even for a large hyperedge. Multiplying the
local tensors and summing internal bits yields the Born probability. Unmeasured
outputs insert the identity (a local trace). Pauli outputs replace the readout by
`(V† P V)_{z',z}`. Classical action automata share the physical `y` indices with
this same doubled network, so correlations between blocks are retained.

For a majority block with `k` bits, use count state
`c_j=min(floor(k/2)+1, c_(j-1)+y_j)`, starting at zero. The terminal logical bit is
one exactly in the absorbing strict-majority state. A parity block instead uses
`c_j=(c_(j-1)+y_j) mod 2`. Combining logical bits in measured-block order uses
`r_j=(2*r_(j-1)+b_j) mod A`; the final open `r` index is the action. This directly
accounts for ties and modulo collisions without a Fourier expansion of majority.

`opt_einsum.contract_path` selects a contraction order from tensor **shapes**.
`contract_expression` caches the structural path in a bounded 512-entry LRU keyed
by shapes, labeled connectivity, output order and planner strategy. Numerical gate
values are rebuilt every time; the path cache cannot retain old trainable coefficients.
Topology/readout changes therefore receive the appropriate structural plan.
Supported planner strategies are `greedy`, `auto`, and `auto-hq` (default `greedy`).
The planner is heuristic; it does not certify optimal treewidth.

Tensor values and contractions use Torch, including the phase exponentials and
local matrices. Autodiff differentiates both the quantum angles and their classical
input projections, with no detached trainable values and no pruning at zero.
Analytic `best`/`backprop` supports Torch, including tested second derivatives and
an optimizer step. Broadcast tapes expand into scalar tapes, whose paths are
reused; their batch count is included in the estimates. QASP's existing
`NeuralEmbedding` expects one input row: evaluate rows individually and stack them.
Vectorized Torch encoders that construct broadcast angles work directly, and are
covered separately by input/weight gradient tests.

Non-Torch differentiable interfaces use fallback in auto mode and produce an
informative unsupported-interface error in explicit mode. Explicit adjoint
requests are not TN derivatives; auto can delegate supported analytic adjoint
measurements to `default.qubit`. Optimization modules' automatic derivative choice
uses backprop for the TN device. Explicit choices of other plugins retain their
previous derivative defaults.

## Complexity depends on the circuit, readout and contraction order

Factoring a k-wire Pauli phase needs O(k) small parity tensors. The majority
readout has a different storage cost: its counter dimensions grow with block
size, and the current implementation stores dense transition tensors. A k-bit
strict-majority block uses O(k) tensors but Theta(k^3) dense entries. Compact
readout avoids the required exponential basis-output table; it does not imply
linear storage or efficient contraction for every circuit.

For a fixed dense contraction path, let U_s contain the labels in the two operands
at step s, with index dimensions d_i. Arithmetic work is bounded by a constant
times `sum_s product_(i in U_s) d_i`, in addition to encoding and tensor
construction. Gradient arithmetic has the same order for a scalar loss, while
saved intermediates increase memory. Readout counters change both connectivity
and dimensions. Greedy planning does not guarantee an optimal path.

See the [scientific manuscript](https://github.com/andre-sequeira10/QASP/blob/3bda8622bd6f5ddfb7c05a0dfd0538c1b0ccd718/docs/iqp_tn_paper/iqp_tn_simulator_paper.pdf) for proofs, readout automata,
weighted-width bounds, the neural chain rule and limitations. The new
[commuting AngleAnsatz benchmark](https://github.com/andre-sequeira10/QASP/blob/3bda8622bd6f5ddfb7c05a0dfd0538c1b0ccd718/docs/iqp_tn_angle_policy_comparison.md) reports
matched-seed inference, warmup and full-gradient measurements against PennyLane
MPS/TN. Its local RX/RXX/RXXX family admits an exact MPS of bounded bond dimension;
these timings are implementation comparisons, not an asymptotic advantage claim.

## Optional inference acceleration

`inference_execution="torchscript"` enables a bounded cache of traced contraction
kernels and static decoder tensors **only inside `torch.no_grad()` or
`torch.inference_mode()`**. `inference_execution="cached"` reuses static tensors
without tracing. The default remains `"eager"`: compilation has a startup cost,
and gains depend on the circuit, readout and number of repeated evaluations.

```python
import torch
import pennylane as qml
from qasp import IQPTNSimulator, action_probs

dev = IQPTNSimulator(3, inference_execution="torchscript")

@qml.qnode(dev, interface="torch", diff_method="backprop")
def policy(theta):
    for w in range(3):
        qml.Hadamard(w)
        qml.RZ(theta[w], w)
    qml.MultiRZ(theta[3], wires=[0, 1, 2])
    for w in range(3):
        qml.Hadamard(w)
    return action_probs(range(3))

theta = torch.tensor([.2, -.4, .6, .7], requires_grad=True)
with torch.no_grad():
    policy(theta)  # build and validate the kernel; account for this warmup
    probabilities = policy(theta + .1)  # uses the current angles
loss = -policy(theta)[1].log()  # unchanged eager autograd implementation
loss.backward()
print(dev.inference_cache_info())
dev.clear_inference_cache()
```

The same option can be passed through `CircuitBuilder.compile(device_kwargs=...)`
or by supplying a configured device instance to a model. Calling `model.eval()`
alone does not disable gradients and therefore does not activate these caches.
With gradients enabled, even untrainable inputs use the original eager path.
Forward-mode dual tensors and autocast use eager contractions as well.

Only the contraction expression is traced. The tape, gate-matrix construction,
input encoding and readout assembly remain eager. Every numerical operand,
including static constants, is an explicit kernel input; parameter-dependent
tensors are rebuilt on every call. The kernel key guards labeled connectivity,
output indices, planner choice, exact tensor shapes/strides, dtype/device,
conjugation flags, gradient flags and inference-mode context. Semantically
different observables can share a contraction kernel when their only difference
is a current tensor operand. No tensor value is frozen into that kernel.

PennyLane broadcast expansion still produces scalar tapes; changing a supported
batch length can reuse their kernels. There is no new assumption of arbitrary
batch-rank polymorphism. Parameter updates and checkpoint loads need no cache
flush because the new tensors are passed explicitly. Changing topology, shapes,
readout connectivity or precision selects the corresponding guarded kernel.

`inference_cache_size=32` bounds kernel entries per device, including failed
compilation entries. `inference_constant_bytes=16*1024**2` bounds cached constant
payloads and byte-string keys; constants also have a 512-entry cap. Python/JIT
object overhead is outside those limits. Setting either limit to zero disables
that cache component. These optional cache limits supplement the existing planner
estimate; automatic selection, explicit hard limits and fallback rules are
unchanged. They are checked before numerical execution and compilation.

New kernels must match eager execution bit for bit on the tracing inputs and
warmed checks. Compiler errors, execution errors and detected mismatches select
eager contraction with a warning and an inspectable failure. This check is not a
proof for all floating-point inputs; the regression suite additionally exercises
changed parameters, checkpoints, graph/readout changes, gradients and rollouts.
The feature is tested on CPU with Torch 2.7.0; no GPU speedup is claimed.

See the [inference integration report](https://github.com/andre-sequeira10/QASP/blob/3bda8622bd6f5ddfb7c05a0dfd0538c1b0ccd718/docs/iqp_tn_inference.md) for before/after
forward and backward timings, warmup costs, exact seeded training replays and
why the paper-specific post-freeze trainer shortcut is not enabled globally.

## Planning, cost limits and sampling

`device.plan(tape, batch_size=..., requires_grad=..., interface="torch")` inspects
structure and returns `IQPTNReport` without materializing numerical contraction
tensors. `last_report` and `last_reports` expose execution decisions. In a
broadcast batch, the estimates deliberately reserve storage for all tape results
and saved gradients. Calls with no trainable Torch values can return NumPy data.

The estimate sums planner operation costs, multiplies by four to cost complex
arithmetic, and reserves a factor of three for forward/backward when gradients
are required. Memory sums input sizes, output sizes and every intermediate size, adds the largest
intermediate as working storage, multiplies by complex element bytes (8 or 16),
batch size, and the backward factor. This conservative tensor-storage proxy is
not a measured process peak; it excludes Python, planner, allocator and kernel
workspace overhead. It also does not bound the memory of a user's external neural
network. `complex128` is the default; `complex64` is selectable. In the 50-qubit
compact-rotation test, complex64 accumulates about `5.4e-6` observable error,
consistent with the selected precision; that test uses a width-scaled tolerance
of `100 * eps(float32)` rather than a double-precision threshold. A full-output
lower-bound check precedes expensive output construction. Precision is never
silently reduced to meet an automatic budget.

Costs depend on graph structure, hyperedge ordering, measurement automata and the
chosen path. Degree three alone does **not** guarantee polynomial contraction.
An expensive plan is not evidence of classical hardness; a feasible finite plan
is not a proof of asymptotic polynomial scaling. Auto budgets are selection rules,
not performance promises.

With finite shots, probability/action readouts return multinomial frequencies
from the contracted probabilities. Bit samples/counts use repeated prefix-
conditioned contractions, costing roughly `shots × measured wires` contractions.
They are nondifferentiable draws: use `diff_method=None`. Analytic probabilities,
finite-shot samples/frequencies, and additive estimators are distinct interfaces;
no additive-estimation algorithm is implemented here. Significant negative or
nonfinite conditional probabilities raise. Sampling clips only small negative
roundoff (tolerance `1e-6` relative to conditional mass) and renormalizes; analytic
outputs and gradients are not clipped. Shot vectors, observable samples,
finite-shot observable moments, and `counts(all_outcomes=True)` are unsupported on
the TN path and may fall back in auto mode.

Statevectors, density matrices, noisy channels, source mid-circuit control, GPU/MPI
TN execution, general tensor-network circuit simulation, and automatic algebraic
rewriting of arbitrary functions are outside this implementation's tested contract.
The old `cut_circuit=True` and Pauli-propagation paths are retained separately;
there is no combined cutting/TN integration in this main-based branch.

## Reproduction and validation

```bash
conda activate torchlane
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/tmp/iqptn-mpl
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
python examples/iqp_tn_simulator.py
python -m pytest tests/test_iqp_tn.py tests/test_iqp_tn_applications.py -q
python examples/benchmark_iqp_tn.py --output docs/benchmarks/iqp_tn.json
```

The executable example compares automatic selection, deliberately over-budget
automatic fallback, and forced TN execution on the same five-qubit circuit. All
three must agree on action probabilities and gradients. Tests compare randomized
small graphs over three seeds: chains, stars, cycles, a cubic graph and RZZZ/higher
hyperedges; partial/reordered readout; even/odd and overlapping majority blocks;
non-power-of-two action counts; shared and special/zero angles; second derivatives;
input/parameter gradients; batches; topology changes; shots/counts; and an update.
Public API tests exercise Born policies with neural input scales, classifiers,
generative learning, kernels, and optimization, distinguishing TN execution from
fallback when the constructed circuit is unsupported.

The benchmark records first-forward time, three warm-path forward/backward samples,
and OS process peak RSS from fresh subprocesses per case. Process memory includes
imports and the Torch runtime; it must not be read as tensor memory alone. It
compares statevectors only up to 16 qubits. The 32/45-qubit chain cases run TN only.
See [machine-readable results](https://github.com/andre-sequeira10/QASP/blob/3bda8622bd6f5ddfb7c05a0dfd0538c1b0ccd718/docs/benchmarks/iqp_tn.json) and the validation summary
in [test results](https://github.com/andre-sequeira10/QASP/blob/3bda8622bd6f5ddfb7c05a0dfd0538c1b0ccd718/docs/benchmarks/iqp_tn_validation.json). These are local experiments,
not hardware claims or scaling theorems.

A separate [comparison with PennyLane `default.tensor`](https://github.com/andre-sequeira10/QASP/blob/3bda8622bd6f5ddfb7c05a0dfd0538c1b0ccd718/docs/iqp_tn_pennylane_comparison.md)
measures its MPS and TN methods on matched circuits. It separates native observable
execution from majority readout through dense state extraction and reports full
gradient costs, accuracy, and process memory. The optimized rerun includes a
matched eager baseline, explicit no-grad inference, and first-call/warmup costs.

For the public RL API, see [policy construction and Torch training examples](iqp_tn_policies.md)
and the [executed policy notebook](https://github.com/andre-sequeira10/QASP/blob/3bda8622bd6f5ddfb7c05a0dfd0538c1b0ccd718/examples/iqp_tn_rl_policies.ipynb).

## Recorded results (27 October 2026)

The `torchlane` runs passed **284 distinct tests**: a 283-test regression suite,
followed by a 106-test core rerun that adds the output-memory regression. There
were no failures or skips in these completed runs. The environment used Python
3.10.17, Torch 2.7.0, PennyLane 0.42.2 and opt_einsum 3.4.0.

The table reports warm-path medians from the recorded CPU benchmark. RSS is
the peak of the entire subprocess, not just the tensor contraction.

| Graph | Qubits | Backend | Forward ms | Backward ms | Peak process MiB |
|---|---:|---|---:|---:|---:|
| chain | 8 | tn | 3.91 | 2.25 | 292.8 |
| chain | 8 | statevector | 4.44 | 2.94 | 293.6 |
| chain | 16 | tn | 10.89 | 7.12 | 298.8 |
| chain | 16 | statevector | 56.19 | 53.42 | 401.7 |
| chain | 32 | tn | 19.91 | 18.16 | 302.0 |
| chain | 45 | tn | 31.95 | 27.63 | 308.9 |
| star | 8 | tn | 3.61 | 2.02 | 296.4 |
| star | 8 | statevector | 4.06 | 2.91 | 294.8 |
| cycle | 12 | tn | 6.45 | 3.96 | 294.6 |
| cycle | 12 | statevector | 18.28 | 8.85 | 301.5 |
| cubic | 6 | tn | 5.01 | 2.52 | 294.4 |
| cubic | 6 | statevector | 3.96 | 1.95 | 294.4 |
| hypergraph | 12 | tn | 7.05 | 6.30 | 296.6 |
| hypergraph | 12 | statevector | 8.89 | 5.96 | 300.3 |

Across the small cases with both backends, the largest measured action-probability
error was `4.66e-15`. Full vector gradient comparisons are covered by the tests;
the benchmark records gradient norms as a compact reproducibility check.

## References

- Markov and Shi, *Simulating Quantum Computation by Contracting Tensor Networks*,
  SIAM J. Comput. 38, 963–981 (2008), [arXiv:quant-ph/0511069](https://arxiv.org/abs/quant-ph/0511069).
- Smith and Gray, *opt_einsum — A Python package for optimizing contraction order
  for einsum-like expressions*, JOSS 3, 753 (2018), [doi:10.21105/joss.00753](https://doi.org/10.21105/joss.00753).
- [opt_einsum path-planning API](https://optimized-einsum.readthedocs.io/en/stable/autosummary/opt_einsum.contract_path.html)
  and [reusable expressions](https://optimized-einsum.readthedocs.io/en/stable/api.html).
- [PennyLane custom-device interface](https://docs.pennylane.ai/en/stable/code/api/pennylane.devices.Device.html).

The chain-of-triples proof of concept in the separate IQP study was inspected as
motivation. Its block-size restrictions and hand-derived contraction are not
assumed here; this implementation compiles the actual tape and readout.
