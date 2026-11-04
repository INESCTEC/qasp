# Generative learning with PQCBM

`qasp.genlearn` learns distributions of binary strings using parameterized
quantum circuit Born machines. A circuit prepares
`|psi(theta)> = U(theta)|0...0>`; computational-basis measurement generates
`q_theta(x) = |<x|psi(theta)>|²`. The training dataset is a matrix of zeros
and ones, with one output bitstring per row.

The public entry points are `PQCBM`, `create_qcbm`, `draw_qcbm`, and `PauliMMD`.
They are available from `qasp.genlearn` and from `qasp`. Probability-loss helpers
`nll_loss`, `kl_divergence_loss`, and `mmd_loss` are in `qasp.genlearn`.

For a complete runnable example, open
[the experiment notebook](../examples/pqcbm_generative_learning.ipynb).
For cluster execution, use
[the MPI smoke test](../tests/tests_deucalion/pqcbm_mpi_smoke.py).

## Quick start

```python
import numpy as np
from qasp import create_qcbm, draw_qcbm

# Equal-frequency all-zero and all-one target strings.
data = np.array([[0, 0, 0], [1, 1, 1]] * 64)
config = {
    "ansatz": "AngleAnsatz",
    "ansatz_config": {
        "n_layers": 2,
        "rotations": "YZ",
        "include_imprimitive": True, "imprimitive": "CNOT",
        "imprimitive_pattern": "chain",
    },
    "loss": "mmd",
    "mmd_config": {
        "bandwidths": [0.5, 1.0, 2.0],
        "mode": "exact",  # Small-system validation; use sampled at scale.
    },
    "device": "default.qubit",
    "seed": 7,
    "lr": 0.05,
}
model = create_qcbm(config, n_qubits=3, data=data)
params, history = model.train(100)
bits = model.sample(1000)         # shape (1000, 3), direct device samples
probabilities = model.get_probs() # shape (8,), small-system diagnostic
tex = draw_qcbm(model, output="tikz", tikz_options={
    "standalone": True, "symbolic_parameters": True, "compress": True,
})
```

The factory copies its config and supplied `CircuitBuilder`; it does not change
caller-owned definitions. Dimensions can also be inside `config`.
`PQCBM.from_config(config, n_qubits=..., data=...)` is equivalent. Explicit
`n_qubits` must agree with the dictionary. Data may instead be passed later to
`model.train(steps, data=...)`.

## Defining the circuit

Use the same flat component dictionaries as the classifier and RL factories.
`ansatz` accepts `"AngleAnsatz"` / `"angle"`, `"IQPAnsatz"` / `"iqp"`, or a
compatible template class. `ansatz_config` contains that template's constructor
arguments; `n_layers` may be a callable of the number of qubits. The default is
`AngleAnsatz` with the template's defaults.

A Born machine normally has no feature map: its observations are the strings
it generates. An optional feature map encodes **one fixed conditioning vector**,
shared by every training observation. It is not automatically fed the target
bitstring. For example:

```python
conditional_config = {
    **config,
    "feature_map": "AngleEmbedding",
    "feature_map_config": {
        "features": 2, "rotations": "X", "input_scaling": True,
    },
    "inputs": [0.25, -0.4],
}
conditional = create_qcbm(conditional_config, 3, data)
```

`NeuralEmbedding` and explicit feature-map classes also work. `features` may
be a callable of `n_qubits`. Pass `inputs` with exactly the embedding's feature
count. Input scaling parameters are trained together with ansatz parameters;
set `input_scaling_params` to initialize them explicitly, or use
`input_scaling_init=Uniform(0.8, 1.2)` from `qasp.initialization`.
Without either setting they start at one. `init=Normal(0, 0.2)` independently
initializes ansatz weights; both distributions use the model's local `seed`.
Explicit arrays override their own group. Reuploading follows the ansatz's
`reuploading` setting. Training a
conditional model on different input vectors for different rows is outside
this trainer's contract; separate conditioning vectors require an external
training loop.

For custom circuits, supply `circuit_builder=builder` or positional
`PQCBM(builder, initial_parameters, data, ...)`. The builder must expose an
ansatz and matching `n_qubits`; its measurement is selected for the objective.
A probability-returning PennyLane QNode is also accepted with explicit initial
parameters. Its trainable argument must be named `params`, and its terminal
measurement must be one computational-basis probability readout. Supply
`n_qubits` for a readout smaller than the device register. Bit columns follow
the QNode's measured wire order. An existing QNode retains its device,
differentiation method and shots; it cannot be reconfigured by device options.

## Training API

| Setting or method | Meaning |
| --- | --- |
| `loss="nll"` | Default; full-probability negative log likelihood |
| `loss="mmd"` | Gaussian MMD squared using Pauli-Z expectations |
| `interface="torch"` | Default; live float64 Torch parameters and Adam |
| `interface="autograd"` | PennyLane NumPy arrays and PennyLane Adam |
| `lr=0.1` | Positive optimizer learning rate |
| `batch_size=None` | Full data; otherwise sample rows without replacement |
| `seed=0` | Local RNG for initialization and batches; default MMD seed too |
| `init="random_0_2pi"` | Ansatz initializer from `qasp.initialization` (e.g. `Normal`, `Beta`, `TruncatedNormal`, `LogNormal`), legacy string, or tensor callable |
| `input_scaling_init=None` | Independent input-scale initializer; omitted means ones |
| `init_params` | Explicit initial tensor, matching ansatz shape |
| `train(steps, ...)` / `fit(steps, ...)` | Returns `(ansatz_params, pre_update_losses)` |
| `loss(data=None)` | Differentiable scalar at current weights and fixed masks |
| `parameters()` | Live ansatz and optional scale tensors for external optimizers |
| `moments(masks)` | Differentiable Z expectations; masks have shape `(M,n)` |
| `sample(shots, return_bits=True)` | Device samples, shape `(shots,n)` |
| `sample(shots, return_bits=False)` | Big-endian indices, at most 62 bits |
| `get_probs()` | Full `2**n` vector, guarded by `max_probability_qubits=20` |
| `shannon_entropy()` | Empirical data and full model entropies, in nats |

`PQCBM` is a model/trainer object, not a `torch.nn.Module`; `train()` performs
optimization. To use a different Torch optimizer, create the model and assign,
for example, `model.opt = torch.optim.SGD(model.parameters(), lr=0.05)`, or use
`loss()` / `backward()` in your own loop. Autograd models also accept a
PennyLane optimizer via `optimizer=`. Torch also accepts an optimizer factory,
for example `optimizer=lambda p: torch.optim.SGD(p, lr=0.05)`, which binds
to the newly created parameters; unrelated optimizer instances are rejected. A custom `train(..., loss_fn=callback)`
receives `(probabilities, batch_indices)` and must return a differentiable
scalar; this path requires probability readout and does not work on PP.
`kl_divergence_loss` expects a target **distribution**, so it needs an adapter
before use as this callback.

Loss histories contain values **before** each parameter update. `model.loss()`
after training evaluates the final parameters. Sampled-MMD histories also
contain observable-sampling noise; compare checkpoints on fixed masks or exact
MMD for small systems. `train(..., resample_observables=False)` holds masks
fixed. For an external optimizer, call `model.mmd.resample()` once per step,
then keep the masks fixed during that step's differentiation.

`natural_grad=True` retains the older Fisher update as a tested small-system
option. It requires `interface="autograd"`, analytic shots, no trainable input
scales, and a positive `lambda_reg` (default 0.01). `fisher="quantum"` uses four
times the full adjoint metric tensor; `"classical"` uses the probability Fisher.
`commuting=True` reuses the initial Fisher matrix: it is a caller-selected
approximation, not an automatically established property of the ansatz. This
path forms a dense parameter-by-parameter matrix and is not the scalable MMD
training route. Plugin support for the transforms still applies.

## MMD as a mixture of expectation values

For binary strings, squared Euclidean distance is Hamming distance. We use the
Gaussian kernel with **standard deviation** `sigma`:

\[
k_\sigma(x,y)=\exp[-\|x-y\|^2/(2\sigma^2)].
\]

Let `a` be a binary mask, `Z_a` the product of Pauli-Z operators at its nonzero
positions, and `mu_p(a)=E_p[(-1)^(x dot a)]`. Independently include each bit of
`a` with probability

\[
r_\sigma=\tfrac12(1-e^{-1/(2\sigma^2)}).
\]

Then

\[
\operatorname{MMD}^2(p,q_\theta)
=\mathbb E_{a\sim\operatorname{Bernoulli}(r_\sigma)^{\otimes n}}
 [ (\mu_p(a)-\langle Z_a\rangle_\theta)^2 ].
\]

This implements the expectation-mixture construction in Sec. 4.2, Eq. (16) of
[Recio-Armengol, Ahmed and Bowles (2025), arXiv:2503.02934v1](https://arxiv.org/abs/2503.02934v1).
There is a bandwidth-notation inconsistency in the supplied v1: Eq. (3) uses
`2 sigma²`, while Eq. (17) prints `2 sigma`. QASP consistently uses the
standard-deviation convention of Eq. (3), including the square in the mask
probability. Regression tests compare the entire expression and its gradient
against the dense Gaussian-kernel definition at nonunit bandwidths.

`bandwidths=[...]` defines an equal-weight mixture of Gaussian kernels. In
sampled mode, choose a bandwidth uniformly for each mask, then draw its bits.
In exact mode, average the corresponding product-Bernoulli weights over all
bandwidths. The identity mask has discrepancy zero but retains its probability
mass; dropping it and renormalizing would change the loss.

| `mmd_config` option | Default | Behavior |
| --- | --- | --- |
| `mode` | `"sampled"` | Monte Carlo masks; `"exact"` enumerates all `2**n` masks |
| `n_observables` | `128` | Number of sampled masks, including duplicates/identity |
| `bandwidths` | `[1.0]` | Positive finite standard deviations |
| `estimator` | `"empirical"` | Compare to empirical data; `"unbiased"` corrects the data-square term |
| `seed` | Model seed | Isolated RNG for mask sampling |
| `max_exact_qubits` | `16` | Guard against accidental exponential enumeration |

The empirical target moment is the mean parity of the observed bitstrings.
For `estimator="unbiased"`, with `N>=2` target observations, the data-square
term is replaced by `(N*mu_hat²-1)/(N-1)`, removing its diagonal sample bias.
This estimator can be negative; QASP does not clip it. Relative to a population
distribution, unbiasedness assumes independent data observations. Repeated
training on a fixed dataset is empirical-risk optimization.

With analytic quantum expectations, the model square is exact. With finite
shots, QASP evaluates each model expectation **twice independently** and uses
`(mu_q_1-mu_p)*(mu_q_2-mu_p)`. This avoids the upward bias of squaring a single
shot estimate and supports differentiation of both factors. Observable
sampling, data minibatching, finite shots and PP approximation are distinct
sources of error. Pruned/truncated PP estimates generally do not have the
unbiasedness guarantees of exact expectations.

The implementation uses the paper's expectation-mixture identity, with QASP's
PennyLane and PP engines. It does **not** implement the paper's specialized
IQP Monte Carlo estimator in Eq. (14)/(21), its initialization strategy, or its
thousand-qubit experiments. General Pauli propagation can still grow
exponentially. A sampled MMD objective avoids enumerating outputs; it does not
by itself make arbitrary circuits efficiently simulable.

## Devices, Pauli propagation, and sampling

`device`, `diff_method`, `shots`, `mpi` (or `enable_mpi`) and `device_kwargs`
follow the shared `CircuitBuilder` / `create_device` convention. Compatible
installed PennyLane plugins are passed through, including device objects.
They remain responsible for gate, measurement and differentiation support.
Conflicting MPI aliases and unsupported requests raise errors rather than
silently choosing serial execution.

| Backend | Training route and boundaries |
| --- | --- |
| `default.qubit` | NLL or MMD; analytic backprop, or finite-shot parameter shift |
| `lightning.qubit` | NLL with a supported probability gradient; MMD supports analytic adjoint |
| `lightning.kokkos`, `lightning.gpu` | Same readout distinction; MPI needs a compatible MPI build and launch |
| `default.mixed` | NLL/MMD where the circuit and differentiation are supported |
| `default.tensor`, `lightning.tensor`, other plugins | Shared plugin route; install dependencies and use plugin-supported gates/gradients |
| `default.clifford` | Shared builder Clifford decomposition route; arbitrary trainable rotations are not an exact Clifford simulation |
| `lightning.pp` | QASP analytic differentiable PP; Torch + MMD, no finite shots or MPI |

Use `diff_method="best"` for backend selection, `"adjoint"` for supported
analytic expectation backends, and `"parameter-shift"` for finite shots.
Lightning backends do not implement ordinary statevector backprop. NLL needs
full probabilities, so an adjoint method supporting only expectation outputs
is unsuitable for that readout. Quantum hardware plugins can run supported
circuits with finite shots and parameter shift; access and plugin installation
are external to this module.

PP controls are `pp_truncation=None`, `pp_topM=20000`, `pp_tau=1e-10`,
`pp_prune_every=5`, and `pp_prune=True`. Truncation limits Pauli weight;
pruning removes terms based on detached coefficient magnitudes and retains
gradients through surviving terms. For a small-system exact reference, use
`pp_truncation=None, pp_prune=False`. Exact PP is not necessarily cheap.
QASP deduplicates masks and handles identity moments classically.

A PP backend computes expectation values, not a normalized probability
vector or samples. Deploy the same learned circuit to a sampling backend:

```python
pp = create_qcbm({**config, "device": "lightning.pp", "pp_prune": False}, 3, data)
pp.train(20)
bits = pp.sample(1000, device="default.qubit", device_kwargs={"seed": 12})
```

`sample()` uses direct quantum sampling and does not call `get_probs()`.
For large models choose an appropriate installed hardware/simulator plugin;
explicitly choosing a statevector simulator still incurs exponential state
memory. Diagnostic `get_probs(device=...)` is also available, subject to its
size guard. The returned bit matrix scales with shots times qubits.

## Quantikz and publication figures

```python
from pathlib import Path
tex = model.draw(output="tikz", tikz_options={
    "standalone": True,
    "symbolic_parameters": True,
    "compress": True,
    "wire_labels": [r"q_0", r"q_1", r"q_2"],
    "gate_colors": False,
})
Path("pqcbm.tex").write_text(tex)
```

`draw_qcbm(model, ...)`, `model.draw(...)`, and
`draw_qcbm(config=config, n_qubits=3, ...)` share the same exporter. Return
formats are `matplotlib`/`mpl`, `text`/`ascii`, and `tikz`/`tex`/`latex`.
Matplotlib returns `(figure, axes)`; other formats return strings. Set
`standalone=False` for a snippet to insert into a paper. The circuit drawing
shows computational-basis measurement; the classical MMD calculation is not
shown as a quantum gate. A config drawing with a feature map uses zero inputs if none are supplied;
pass `inputs=...` to draw a particular conditioning vector. Drawing traces
gates without executing the quantum circuit or altering the model's training measurement. Compile standalone
source with a LaTeX installation containing Quantikz for vector PDF output.

## MPI execution

MPI distributes **one simulator statevector**. It does not distribute data
batches among the ranks of that statevector. All ranks must use identical
parameters, input scales, target data, masks, optimizer state and circuit-call
order. The model's explicit seed makes its initialization, minibatch choices
and masks reproducible across ranks. Do not choose rank-dependent seeds or
execute training only on rank zero. `seed=None` is rejected with MPI.
Plugin shot RNG seeds are separate from the model seed.

```python
cluster_config = {
    **config,
    "device": "lightning.kokkos",
    "diff_method": "adjoint",
    "mpi": True,
    "seed": 7,
    "mmd_config": {"bandwidths": [1.0, 2.0], "n_observables": 128},
    "device_kwargs": {"batch_obs": True},
}
# Run this construction AND train on every rank in the same MPI world.
model = create_qcbm(cluster_config, n_qubits=12,
                    data=np.array([[0] * 12, [1] * 12] * 64))
model.train(20)
```

Configure the cluster's Python environment and MPI-enabled Lightning plugin
for its installed MPI stack. All ranks execute the same training calls; only
rank zero should write shared results. Launch and resource allocation are
managed by your cluster's scheduler.

MMD training uses expectation readouts rather than a full `2**n` probability
output. Statevectors still require exponential memory, and exact-mask MMD
still has exponential classical storage. The target parity matrix occupies
batch rows times sampled masks, while masks occupy sampled masks times qubits.
Account for these classical buffers when choosing resources. NLL, entropy,
probability diagnostics and natural-gradient Fisher matrices retain additional
dense costs.

The collective smoke test compares a distributed model with a serial
reference and checks agreement across ranks:

```sh
mpiexec -n 2 python tests/tests_deucalion/pqcbm_mpi_smoke.py \
    --device lightning.kokkos --qubits 6
```

## Migration and validation

`BornMachine` is a deprecated compatibility subclass of `PQCBM`. It preserves
Autograd and integer sampling defaults, while fixing the old default NLL
import and natural-gradient update failures. New `PQCBM` defaults to Torch
and bit-matrix samples. Probability callbacks retain their two-argument
contract. Prefer explicit `n_qubits` for custom circuits and flat parameters
matching the ansatz's reported shape.

The tests in [test_genlearn.py](../tests/test_genlearn.py) check dense-kernel
MMD equivalence and gradients, backend agreement, feature-map/input-scale
training, Fisher updates, finite-shot gradients, unbiased corrections,
large-width expectation-only execution, sampling, drawing, configuration
errors, reproducibility, and MPI argument routing. Local routing tests use a
serial stand-in and are not evidence of real multi-node execution. The
notebook records a small experiment with exact diagnostics and a PP comparison;
it is not a reproduction of the reference paper's large-scale benchmarks.
