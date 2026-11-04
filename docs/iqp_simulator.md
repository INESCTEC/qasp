# Unified IQP simulation in QASP

Research reproduction commands and recorded benchmark discussions in this guide refer to the [pinned upstream checkout](https://github.com/andre-sequeira10/QASP/tree/3bda8622bd6f5ddfb7c05a0dfd0538c1b0ccd718). This distribution keeps the runtime API and compact tutorials; see the [repository layout policy](repository_layout.md).

`IQPSimulator` combines exact tensor-network contraction with a differentiable
Van den Nest estimator for Pauli-Z correlations. It implements the algorithm in
[IQPopt, Sections 2.1–2.4](https://arxiv.org/abs/2501.04776v1) using Torch, without
a JAX dependency. The estimator can be faster for a polynomial list of
correlations, especially when exact contraction has large width. It returns
additive-error estimates; TN remains exact.

## Public configuration

~~~python
from qasp import IQPSimulator

device = IQPSimulator(
    wires=20, backend="auto",  # or "tn", "van_den_nest"
    vn_samples=4096, vn_sampling="resample", seed=2026,
)
node = builder.compile(device=device, diff_method="backprop")

# Existing QASP model dictionaries:
config = {
    # Existing ansatz, encoder and model settings ...
    "device": "IQPSimulator",  # alias "iqp"; class or instance also accepted
    "device_kwargs": {
        "backend": "van_den_nest", "vn_samples": 8192, "seed": 2026,
    },
    "diff_method": "backprop",
}
~~~

`PQCSoftmax` and `PQCBornVariance` now accept `device_kwargs` directly.
See the executed [notebook](../examples/iqp_simulator.ipynb) for ordinary public
policy configuration, neural encoding, Adam updates and generative MMD.

`IQPTNSimulator`, `"iqp.tn"` and the legacy device string `"auto"` retain their
previous exact-TN/statevector behavior. The new two-method selector is
`IQPSimulator(backend="auto")`. Its defaults are also configurable through the
exported `IQPOptions` dataclass.

## Supported circuits and encoding

The estimator accepts the user's construction
\[
U_X(\theta)=\exp\!\left(-i\sum_j\theta_jX^{g_j}\right)
=H^{\otimes n}\exp\!\left(-i\sum_j\theta_jZ^{g_j}\right)H^{\otimes n},
\]
initialized in \(|0^n\rangle\). Supports can overlap arbitrarily and have any
order. RX, IsingXX, X-only PauliRot, recognized adjoints, and equivalent
H-framed Z gates are supported. Compatible CZ and controlled-phase gates expand
into commuting rotations. Zero angles do not relax the structural checks.

Encoding uses the same gates with angles
\(\alpha_l(s)=\langle s,\lambda_l\rangle+b_l\).
`NeuralEmbedding` supplies the linear case; appending a constant feature supplies
bias. Angles, inputs, encoder parameters and observable coefficients remain live
Torch tensors. PennyLane uses \(R_P(\phi)=e^{-i\phi P/2}\), so \(\phi_j=2\theta_j\).

Every estimator measurement must be `qml.expval` of a Z string or a real weighted
sum of I/Z strings. The optional importance mode additionally accepts compact
computational-basis projectors and their supported sums/products (below). Arbitrary local preparation, open Hadamard frames,
incompatible rotations, X/Y readout, variances, probabilities, output
samples/counts and finite quantum shots are outside this estimator. TN retains
its broader circuit/readout contract. Generic exponential operators must be
expressed as supported rotation primitives.

## Estimator and uncertainty

For \(z\) uniformly distributed over \(\{0,1\}^n\), and physical gate angles \(\phi\),
\[
\mu_a=\langle Z^a\rangle
=\mathbb E_z\cos\!\left[\sum_j\phi_j(-1)^{g_j\cdot z}
 {\bf1}\{g_j\cdot a=1\pmod2\}\right].
\]
A mean over \(S\) independent uniform bitstrings is unbiased, with standard
deviation at most \(S^{-1/2}\) for each string. These are classical auxiliary
samples, not samples from the quantum output distribution.

For each returned \(H_r=\sum_k c_{rk}Z^{a_k}\), let \(L_r=\sum_k|c_{rk}|\).
At a fixed parameter point independent of the samples, with probability at least
\(1-\delta\), all \(M\) returned expectations simultaneously satisfy
\[
|\widehat{\langle H_r\rangle}-\langle H_r\rangle|
\le L_r\sqrt{2\log(2M/\delta)/S}.
\]
Thus \(H=\sum_{k=1}^K P_k\) has \(L=K\): estimation remains polynomial when the
description size, coefficient norm and inverse absolute tolerance are
polynomial, but fixed accuracy on the sum can require more samples than on one
term. Shared samples correlate the term estimates. The reported empirical
standard error of each weighted sum includes those correlations.

`vn_error=0.02, vn_confidence=0.99` selects a sufficient count for each returned
weighted expectation; `vn_samples` remains a minimum.
`vn_max_samples=1_000_000` guards large requests. An explicit estimator request
raises if the bound requires more samples.
`last_report` contains `samples`, `standard_errors` and
`hoeffding_half_widths`. These are fixed-point value bounds, not guarantees of
relative accuracy, gradient accuracy, or accuracy across an adaptive training
trajectory. Reusing a fitted fixed sample bank is not a fresh confidence bound.

## Gradients and sampling

Writing the integrand as \(f_a=\cos\omega_a\), with
\(b_j=(-1)^{g_j\cdot z}{\bf1}\{g_j\cdot a=1\}\), gives
\[
\partial_{\phi_j}f_a=-b_j\sin\omega_a,\qquad
\partial_{\phi_i}\partial_{\phi_j}f_a=-b_i b_j\cos\omega_a.
\]
Torch differentiates the empirical mean and classical encoder. Exhaustive
small-system averaging validates values, input/parameter/coefficient gradients
and second derivatives against full-state execution.

Sampled derivatives of a linear expectation objective are unbiased at fixed
parameters. A nonlinear loss of sampled expectations generally has bias:
\(\mathbb E[\widehat\mu^2]=\mu^2+\operatorname{Var}(\widehat\mu)\).
Generative MMD now uses independent estimator executions for its quadratic
model term, including with `shots=None`. The `independent_estimates()` context
forces fresh replicas even when the device otherwise uses fixed samples.
This correction does not make arbitrary nonlinear policy losses or projected
kernel values unbiased.

- `vn_sampling="resample"`: fresh seeded samples each execution. Backward reuses
  that forward call's samples.
- `"fixed"`: a reproducible empirical objective for debugging and finite
  differences. `device.resample(seed=...)` replaces the bank.
- `"enumerate"`: exact uniform averaging for validation, exponential in qubit
  count and limited to 16 wires by default.

`vn_batch_size=1024` and `vn_observable_batch_size=64` control chunks.
`vn_checkpoint=True` regenerates chunks from captured seeds in backward instead
of retaining every sample matrix. Later executions cannot change earlier graphs.
Disabling checkpointing trades memory for speed.

## Cost and automatic selection

Let \(G\) count gate occurrences, \(K\) distinct Z strings, \(M\) returned sums,
and \(R=\sum_j|g_j|\) the sparse support size. The implemented parity and matrix
operations cost
\[
O\!\left(S(n+R+GK+KM)+GK\lceil n/w\rceil\right),
\]
plus parsing; \(w\) is the integer word size. For sample chunk \(B\) and observable
chunk \(C\), working storage is
\(O(Bn+BG+BC+BM+KG+KM+R)\), plus output and checkpoint graph bookkeeping.
Reverse-mode work has the same polynomial order for the empirical objective.
TN's exact cost instead depends on contraction width.

`auto` checks eligibility and resource estimates for the actual tape/readout,
dtype, batch and backward. Its score also includes small-operation TN overhead,
controlled by `auto_tn_operation_overhead=20000` in equivalent operations.
Pilot timings informed this proxy; it is not a portable timing calibration.
Zero restores arithmetic-only comparison. Both scores and arithmetic estimates
appear in `last_report`.

Eligible large estimator cases skip TN planning above
`auto_tn_max_wires=64` or `auto_tn_max_gates=512`.
Parameter-independent TN reports have a bounded 64-entry cache;
`clear_plan_cache()` clears it. Selection does not draw samples or time user
circuits. It is a heuristic, not a guarantee of the fastest backend.

`allow_approximate=False` disables sampled selection. Forced backends do not
silently switch methods. `hard_max_flops` and `hard_max_memory_bytes` constrain
estimates, not OS resources. In automatic mode, the existing bounded
`default.qubit` fallback remains when neither IQP route qualifies.
`mode="explicit"` disables it; `fallback_max_wires` bounds its size.

## Module coverage

| Consumer | Estimator route | Other readouts |
|---|---|---|
| RL | Softmax on Z moments, including neural encoding | Compact Born probabilities use TN |
| Classifiers | Expectation readout on Z moments | Probability readout uses TN |
| Quantum kernels | Projected Z-moment features | Fidelity/probability kernels use TN or bounded fallback |
| Generative learning | Moment MMD with independent model replicas | Full distributions and output samples use TN or another device |
| Optimization | IQP diagonal Hamiltonians; PCE with Z correlations | Other axes/non-IQP layers use TN/fallback where eligible |

The actual circuit and measurement determine support. An exponentially large
observable list, arbitrary QAOA layers, or arbitrary encoders do not become
efficient by selecting this device. Torch is the differentiable interface for
both native routes; JAX/Autograd differentiation has not been added to them.

## Reproducing the comparison

The expanded [notebook](../examples/iqp_simulator.ipynb) compares QASP TN,
QASP Van den Nest, PennyLane MPS and PennyLane TN through 100 qubits on chains,
rings, stars, breadth-first binary trees, disjoint four-qubit blocks and random
three-body circuits. It reports inference, full gradients, warmup, accuracy,
resource limits and the actual `auto` decisions. These are CPU measurements in
`torchlane`, with one numerical thread and float64/complex128.

~~~sh
python examples/benchmark_iqp_structures.py
python examples/inspect_iqp_auto.py
python examples/summarize_iqp_structures.py
~~~

The benchmark resumes an existing output file. Choose a new `--output` path for
a fresh timing run. Raw measurements are in
[upstream docs/benchmarks/iqp_structures.json](https://github.com/andre-sequeira10/QASP/blob/3bda8622bd6f5ddfb7c05a0dfd0538c1b0ccd718/docs/benchmarks/iqp_structures.json); all circuit supports, angles and seeds are
saved. The summarizer regenerates the report tables and figures. The notebook
loads those saved measurements; running it does not launch the full benchmark.

There are two seeds and two warm repetitions, with independent full-state checks
at eight qubits and cross-backend exact checks at larger sizes where available.
Every job is bounded by phase/record time and observed worker RSS; QASP TN also
uses an explicit planning budget. A missing or timed-out gradient is not a
completed timing. Exact contraction and 4096-sample estimation have different
accuracy contracts. The sample count, four fixed readouts and parameter-shift
versus reverse-mode differentiation all affect the comparison.

The installed PennyLane native PauliRot path applies MPOs without requesting
compression, and wrapped ring inputs produce an MPS shape error. Native outcomes
are retained. A separate control uses equivalent elementary gates and an MPS
cutoff of 1e-12 with no bond cap; this numerical truncation is checked against
available exact references. One eight-qubit random case exceeds the strict
gradient tolerance; a separate zero-cutoff check resolves it without replacing
the original record:

~~~sh
python examples/benchmark_iqp_gradient_estimators.py --output docs/benchmarks/iqp_structures_lowered.json --lowering elementary --cpu-cutoff 1e-12 --families chain ring star tree blocks random --sizes 8 16 32 64 100 --gradients parameter-shift
python examples/summarize_iqp_controls.py
~~~

Both native and lowered inputs are reported, because lowering adds gates and
need not be faster for every structure. The installed device uses its
`auto-hq` option for local expectations, but its product-observable overlap
handler does not forward that option and uses quimb's default `greedy` strategy.
The historical metadata naming `auto-hq` refers to the device option, not to
every contraction. These defaults remain unchanged in the extended runs. These comparisons do not establish an
optimal PennyLane configuration. QASP uses default eager execution here; earlier
compiled-policy timings remain a separate experiment.

PennyLane `default.tensor` does not expose backpropagation or adjoint gradients
in the tested version. Parameter-shift is an exact-gradient choice, not the only
estimator. The supplementary script compares native SPSA with one and four random
directions and records gradient error as well as time:

~~~sh
python examples/benchmark_iqp_gradient_estimators.py
python examples/summarize_iqp_spsa.py
~~~

SPSA uses two shifted circuits per direction, but does not provide the same
accuracy as the complete analytic gradient. Its step size, direction count,
seed and measured error are part of each record. Neither timing study measures
optimization convergence.

`auto` chooses only between QASP's internal algorithms. The saved selection
probes show that a larger sampling budget can favor TN, `allow_approximate=False`
disables sampled selection, probabilities exclude the estimator, and eligible
circuits above 64 wires or 512 recorded operations (including basis changes) bypass TN planning. The latter shortcut also
applies to easy chains and blocks; it is not a proof of runtime optimality.
No selection coefficients were changed for this extension.

The earlier two-backend measurements remain in
[upstream docs/benchmarks/iqp_backends.json](https://github.com/andre-sequeira10/QASP/blob/3bda8622bd6f5ddfb7c05a0dfd0538c1b0ccd718/docs/benchmarks/iqp_backends.json). Prior
[PennyLane policy comparisons](https://github.com/andre-sequeira10/QASP/blob/3bda8622bd6f5ddfb7c05a0dfd0538c1b0ccd718/docs/iqp_tn_angle_policy_comparison.md) retain their
original measurements and interpretation.

## Extended execution budgets

The follow-up retries missing PennyLane full-gradient measurements with a
300-second backward limit, 90 seconds for other phases, an 800-second record
limit and a 4-GiB sampled worker RSS ceiling. A 512-MiB available-system-memory
guard also protects the host. Runs remain serial in `torchlane`. Original files
are preserved, and completed original measurements are reused.

~~~sh
python examples/benchmark_iqp_extended.py
python examples/benchmark_iqp_gradient_estimators.py --output docs/benchmarks/iqp_spsa_extended.json --families chain --sizes 100 --directions 4 --phase-timeout 300 --timeout 1000 --rss-mib 4096
python examples/summarize_iqp_extended.py
~~~

The original controllers reused phase labels and polled them every 0.1 seconds.
Repeated inference calls shared a window, and a brief forward transition could
be missed between backward calls. Consequently, an original 20-second timeout
marker is a monitoring-window threshold, not always a single-call lower bound.
The extended full-gradient runner uses distinct labels for each repetition.

Each new record has two warm inference repetitions and one warm complete-gradient
repetition after the initial calls; original records retain their two warm
gradient repetitions. Both circuit seeds are required for a paired measured
point. The first resource stop ends new attempts at that and larger sizes for
the same representation/family/device. Larger points are labelled **not
attempted**, not timed out. Native ring-MPS execution errors are retained;
additional time does not repair a device error.

The extended summary keeps missing-value reasons and forecasts separate from
measured times. Hollow squares identify single completed seeds, while paired
points still require both seeds. Dotted curves and hollow diamonds are finite-range power-law
extrapolations, admitted only with at least three paired sizes, a successful
largest-size holdout check and a good log-space fit. Shading is model sensitivity,
not a confidence interval. Random graphs and breadth-first MPS trees are not
extrapolated. No forecast is used as a measured speedup or memory requirement.

The four missing 100-qubit, four-direction SPSA cases are also retried in
`iqp_spsa_extended.json`, using a 300-second monitored-window limit and 1000 seconds per
record with the original two warm repetitions. Their timing and gradient-error
summary is kept separate from the complete-gradient comparison.

## References

- M. Van den Nest, *Simulating quantum computers with probabilistic methods*,
  Quantum Information and Computation 11, 784–812 (2011),
  [arXiv:0911.1624](https://arxiv.org/abs/0911.1624).
- E. Recio-Armengol and J. Bowles, *IQPopt: Fast optimization of instantaneous
  quantum polynomial circuits in JAX* (2025), Sections 2.1–2.4,
  [arXiv:2501.04776v1](https://arxiv.org/abs/2501.04776v1).

## Optional importance sampling of observable terms

Set `vn_observable_sampling="importance"` to draw observable terms as well as
auxiliary bitstrings. The default, `"all_terms"`, continues to evaluate every
supplied Pauli term. These are separate controls from `vn_sampling`, which
selects fixed or resampled random banks. Importance mode cannot be combined
with `vn_sampling="enumerate"`: enumerating bitstrings would not remove the
remaining randomness over observable terms.

For an explicit diagonal sum \(H=\sum_{a=1}^{C}c_a Z^{u_a}\), the proposal is
\[
q_a=(1-\eta)\frac{|c_a|}{\sum_b|c_b|}+\frac{\eta}{C},
\qquad Y=\frac{c_a}{q_a}f_{u_a}(z),\quad a\sim q,\quad z\sim\mathrm{Unif}\{0,1\}^n.
\]
Here \(f\) is the VN integrand above, and `vn_term_uniform_mix` sets
\(\eta\) (default 0.05). If all coefficients vanish, the proposal is uniform.
Then \(\mathbb E Y=\langle H\rangle\). The positive uniform component keeps
zero-valued trainable coefficients in the support. The proposal is detached
from Torch; differentiation of the live numerator and integrand gives unbiased
pointwise derivatives of the expectation, including coefficient derivatives at
zero. Differentiating the discrete sampling decision is unnecessary. Proposals
are rebuilt at each execution, so finite differences that perturb coefficients
also change the proposal; they are not a validation of the fixed-proposal
empirical derivative. Angle finite differences with a fixed bank are valid.

The same implementation accepts a compact sum of products of Z words and
computational-basis projectors. In particular,
\[
|x\rangle\langle x|=2^{-m}\sum_{a\in\{0,1\}^m}(-1)^{a\cdot x}Z^a.
\]
An independent uniform subset samples this expansion without constructing its
\(2^m\) terms. For a coefficient times a product of projectors, independent
subsets are drawn for the factors; Z masks combine by XOR and signs multiply.
Sampling an additive component with probability \(q_a\) cancels the projector
expansion's uniform normalization. Duplicate expansion paths are permitted and
remain unbiased. Incompatible overlapping projectors also cancel in expectation.
The exact TN backend can now contract these compact projector observables
directly, providing an exact alternative without a Pauli expansion.

The envelope is \(B=\max_a |c_a|/q_a\). `vn_error` and the reported Hoeffding
interval use \(B\), with half-width
\(B\sqrt{2\log(2M/\delta)/S}\) for \(M\) returned means. A single projector
has \(B=1\), independent of its exponential Pauli expansion. The report includes
`observable_components`, `observable_log2_term_counts` (expansion paths before
cancellation), `coefficient_l1_bounds` and `importance_weight_bounds`.

This does **not** supply an efficient generic majority sampler, relative-error
estimation of exponentially small probabilities, or quantum output samples.
An explicit list of terms still takes time proportional to its description to
parse and build a proposal. Products of sums are not automatically expanded.
A large coefficient norm can make the required sample count impractical.
Replacing exact means by noisy means inside a nonlinear training loss generally
biases that loss and its gradient; the pointwise unbiasedness statement applies
to linear expectation values. Resampling and independent evaluation remain
important during training.

```python
from qasp import IQPSimulator
import pennylane as qml
import torch

# Configure numerical pools once in the application, before substantial work.
torch.set_num_threads(1)
device = IQPSimulator(
    100, backend="van_den_nest", vn_observable_sampling="importance",
    vn_samples=32768, vn_batch_size=1024, vn_workers=4,
    vn_sampling="resample", seed=183,
)

@qml.qnode(device, interface="torch", diff_method="backprop")
def probability(theta):
    for w in range(100):
        qml.RX(theta[w], w)
    return qml.expval(qml.Projector([0]*100, wires=range(100)))

angles = torch.full((100,), 0.1, dtype=torch.float64, requires_grad=True)
value = probability(angles)
gradient, = torch.autograd.grad(value, angles)
```

## Single-node multicore execution

`vn_workers` is a positive integer (default **1**). It controls a bounded CPU
thread pool over sample chunks, in both all-terms and importance modes. With
`vn_parallel_backward=True` (the default), CPU gradient evaluation uses the
same bounded scheduling mechanism. At most
`min(vn_workers, ceil(samples / vn_batch_size))` chunks run concurrently. Each
chunk has a seed allocated in a deterministic order; changing workers preserves
the sample bank and reduction order. Changing batch size changes the bank.
Grad mode and inference mode are propagated into worker threads, and checkpoint
closures retain their own samples even after later forward calls or resampling.
A separate pool is scoped to each forward or backward evaluation and closed
when that phase ends; no device-global executor or persistent worker threads
are retained.
The reported `cpu_workers` is the effective chunk concurrency, and memory
planning includes concurrent working buffers. This is a conservative proxy,
not an operating-system memory cap.

Generator supports are grouped by arity and their parities computed with
vectorized XOR reductions. Gather blocks are bounded to approximately 256 Ki
elements, except when one support alone requires a larger batch-by-arity
buffer. For \(G\) generators with total support size \(R\), \(K\) explicit
Pauli terms, \(M\) returned means and batch size \(b\), all-terms processing
costs \(O(S(n+R+GK+KM))\), with the existing term batching limiting large
intermediate arrays. Joint term sampling replaces the per-sample enumeration
of \(K\) terms by sampled masks: the numerical work is
\(O(S[n+R+M(n+R+G)+D])\), where \(D\) is the total sampled component
support/projector-description size per sample. Proposal construction and
component grouping add description-dependent overhead. None of these changes
makes the worst-case coefficient norm or required accuracy free.

Torch's `torch.set_num_threads(t)` is a separate, application-wide setting for
numerical kernels. QASP never changes it. VN's explicit workers now parallelize
**both forward chunks and their reverse-mode gradient contributions** on CPU.
Workers compute local vector-Jacobian products with `torch.autograd.grad`; the
coordinator reduces them in deterministic chunk order before Torch continues
through the original circuit-parameter and encoder graphs. Workers do not write
parameter `.grad` buffers or concurrently traverse the classical encoder.
Set `vn_parallel_backward=False` to recover the former eager-autograd scheduling
path for comparison. A single worker uses the same new reduction algorithm
serially. Accelerator tensors with one worker retain eager autograd.
TN uses Torch's numerical threading and does not use `vn_workers`.
More threads can slow small operations,
and multiplying many outer workers by many numerical threads can oversubscribe
the allocation. Start with either `vn_workers > 1, torch.set_num_threads(1)` or
`vn_workers=1, torch.set_num_threads(t)`, then measure the actual training loss.
Multicore VN workers currently require CPU tensors; use one worker on GPU/MPS.
No MPI, multi-node distribution or automatic thread tuner is implemented.

All QASP models that pass `device_kwargs` use the same option, for example:

```python
config["device"] = "IQPSimulator"
config["device_kwargs"] = {
    "backend": "van_den_nest", "vn_workers": 4,
    "vn_samples": 8192, "vn_sampling": "resample",
    "vn_observable_sampling": "importance", "seed": 183,
}
config["diff_method"] = "backprop"
```

The circuit and readout must still satisfy the chosen backend's contract.
Softmax policies with compatible Z expectations can use VN; compact Born
majority readouts continue to use TN through `backend="auto"`. The automatic
backend selector compares conservative work/memory proxies; it does not divide
VN cost by worker count or promise optimal runtime on a particular CPU.

See [the executed example notebook](https://github.com/andre-sequeira10/QASP/blob/3bda8622bd6f5ddfb7c05a0dfd0538c1b0ccd718/examples/iqp_importance_multicore.ipynb)
and [the M1 benchmark results](https://github.com/andre-sequeira10/QASP/blob/3bda8622bd6f5ddfb7c05a0dfd0538c1b0ccd718/docs/benchmarks/iqp_multicore_m1.md). The benchmark
script records first-call costs, additional warmups, warm forward and backward
samples, inference times, peak process RSS, versions and source hashes. Historical
paper benchmarks remain unchanged and describe their recorded implementation.


## Parallel chunk gradients

For a returned vector $y=S^{-1}\sum_r v_r(\phi,c)$ and upstream cotangent
$u=\partial L/\partial y$, each worker computes

\[
 g_r=J_{v_r}(\phi,c)^\top(u/S),\qquad
 \nabla_{\phi,c} L=\sum_r g_r.
\]

The implementation sums the unnormalized chunk outputs first; the existing
normalization sends $u/S$ to the custom backward function. This computes a
vector-Jacobian product, without constructing a full observable-by-parameter
Jacobian. The same formula applies to all-terms and importance-sampled
observables. Importance proposals and all random seeds are captured by each
forward call and remain fixed during its backward replay.

With `vn_checkpoint=True`, forward stores only the input tensors and replay
metadata. Each backward worker regenerates one chunk and differentiates its
local graph. At most `vn_workers` chunks are in flight; first-order scratch
memory scales with effective concurrent chunks, rather than all samples.
With `vn_checkpoint=False`, independent local graphs are retained during
forward and reused in first-order backward, trading memory for recomputation.
Repeated backward calls with `retain_graph=True` remain supported.

When `create_graph=True` requests higher derivatives, chunks are replayed using
the original inputs rather than detached local leaves. This preserves dependence
on angles, coefficients and upstream gradients, including mixed derivatives
through neural encoders and nonlinear losses. Higher-order graphs retain their
own intermediates, so the bounded first-order scratch claim is not a bound on
Hessian/third-derivative memory. Tests compare first, second and third derivatives
with the previous path, and check `gradcheck`, `gradgradcheck`, later forwards,
resampling, and both checkpoint settings. This is Torch eager reverse-mode
support; forward-mode and `torch.func` transformations are not added here.

Configure the device through the existing public dictionary or constructor:

```python
torch.set_num_threads(1)  # Configure once in the application.
device = IQPSimulator(
    100, backend="van_den_nest", vn_workers=4,
    vn_parallel_backward=True, vn_checkpoint=True,
    vn_samples=32768, vn_batch_size=1024, seed=183,
)
```

The worker count still cannot exceed the number of chunks: 4,096 samples in
chunks of 1,024 expose only four tasks. Larger worker counts require additional
chunks, and smaller chunks can increase scheduling overhead. Increasing sample
count changes the estimator's computational/statistical budget; it is not a
free parallel speedup. No distributed execution was added.

The [parallel-backward comparison](https://github.com/andre-sequeira10/QASP/blob/3bda8622bd6f5ddfb7c05a0dfd0538c1b0ccd718/docs/benchmarks/iqp_parallel_backward.md) reruns
both the former and new schedulers on the same implementation, rather than
comparing only against older machine timings. It includes the six circuit
families, separate first/warm forward and backward measurements, and larger
workloads with up to eight effective workers. Historical paper and multicore
measurements remain associated with their recorded source revisions.
