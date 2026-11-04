# Differentiable Pauli propagation

`PauliPropagationSimulator` is QASP's shared analytic Heisenberg simulator. It
lives in `qasp.simulators`, alongside `IQPSimulator`, and is exported from `qasp`.
It supports noncommuting circuits, including repeated encoding, arbitrary
mixed-axis Pauli rotations, and CNOT/CZ entanglers. Exact propagation is the default.

Use the same execution dictionary in compatible QASP models:

```python
execution = {
    "device": "PauliPropagationSimulator",
    "device_kwargs": {"k": None, "prune": False},
    "diff_method": "backprop",
    "shots": None,
}
```

`device="pauli.propagation"` is an alias. You may instead pass the class or a
configured `PauliPropagationSimulator(wires=...)` instance. As with other QASP
devices, an instance is reused as-is; do not also pass `device_kwargs` for it.
The string names are resolved by QASP's `create_device` helper, not registered
as external PennyLane plugins.

## A direct neural policy

This model is constructed entirely through QASP. RX/RY/RZ and mixed Pauli
rotations need not commute. The neural encoder and circuit weights train in the
same Torch graph.

```python
import numpy as np
import torch
from qasp import AngleAnsatz, NeuralEmbedding
from qasp.rl.policies import PQCSoftmax

policy = PQCSoftmax(
    n_qubits=3, n_actions=2,
    ansatz=AngleAnsatz,
    ansatz_config={
        "n_qubits": 3, "n_layers": 2,
        "rotations": ["Y 0", "Z 1", "X 2", "XYZ 0 1 2"],
        "include_imprimitive": True, "imprimitive": "CNOT",
        "imprimitive_pattern": "chain", "reuploading": True,
    },
    feature_map=NeuralEmbedding,
    feature_map_config={"n_qubits": 3, "features": 2, "rotations": "Y"},
    observables=["Z 0", "XZ 1 2"], fc_layer=None,
    device="PauliPropagationSimulator",
    device_kwargs={"k": None, "prune": False},
    diff_method="backprop", shots=None,
).double()
x = torch.tensor(np.array([.2, -.4]), dtype=torch.float64, requires_grad=True)
optimizer = torch.optim.Adam(policy.parameters(), lr=.01)
optimizer.zero_grad()
loss = -torch.log_softmax(policy(x), dim=-1)[1]
loss.backward()
assert torch.isfinite(x.grad).all()
assert all(p.grad is not None and torch.isfinite(p.grad).all()
           for p in policy.parameters())
optimizer.step()
```

Run [the complete application example](../examples/pauli_propagation_modules.py)
from the repository root:

```sh
conda run -n torchlane python examples/pauli_propagation_modules.py
```

It exercises a neural policy, PQDQN, classifier, fidelity kernel, MMD Born
machine, two-layer QAOA and PCE. It uses public QASP constructors and factories,
without hand-written QNodes. Numerical parity checks live in
[test_pauli_propagation_applications.py](../tests/test_pauli_propagation_applications.py).

## Application coverage

| Module | Supported analytic paths | Configuration notes |
| --- | --- | --- |
| RL | `PQCSoftmax`, `PQCBornVariance` (the `create_model("PQCBorn", ...)` route), and `PQDQN` | Pauli expectations or Born probabilities; neural PQDQN encodings use `batch_mode="sequential"`, as on other ordinary devices. |
| Classifiers | Expectation and probability readouts, including classical decoders | `PQCClassifier` / `create_classifier`; usual Torch losses and optimizers. |
| Kernels | Global/local fidelity projectors and projected Pauli kernels | Trainable Gram matrices and parameter gradients; kernel convenience methods retain their existing input-conversion behavior. |
| Generative learning | Expectation-based MMD and probability-based NLL | `create_qcbm`, default Torch interface; MMD avoids full probability vectors. |
| Optimization | QAOA, including noncommuting multilayer and warm-start circuits, and PCE | Use `backend="torch"`. Automatic differentiation-method selection chooses backprop; explicit `diff_method="backprop"` also works. |
| Circuit construction | `CircuitBuilder` expectation, variance and probability measurements | Returns an ordinary QNode/TorchLayer; no model-specific PP adapter is needed. |

Backend support does not remove existing restrictions in model APIs or historical
prototypes. It does not enable statevector-dependent diagnostics, natural-gradient
paths requiring unsupported measurements, finite-shot samplers, or sparse-matrix
readouts restricted to other devices. For sampling during generative evaluation,
select an explicit sampling device through that model's existing sampling API.

## Algorithm and derivatives

For a gate sequence $U=U_G\cdots U_1$ and observable
$O=\sum_P c_P P$, propagate the Pauli expansion backward and evaluate
$\langle0|U^\dagger O U|0\rangle$. Clifford gates permute Pauli words and change
signs. A rotation $R_P(\theta)=\exp(-i\theta P/2)$ updates each word $Q$ as

$$
R_P(\theta)^\dagger Q R_P(\theta)=
\begin{cases}
Q,&[P,Q]=0,\\
\cos\theta\,Q+i\sin\theta\,PQ,&\{P,Q\}=0.
\end{cases}
$$

The engine stores words as packed integer X/Z masks and coefficients as Torch
scalars. It merges equal words by adding their coefficients. Final contraction
keeps words containing only I and Z, whose expectation on the all-zero state is
one. State preparation by unitary gates and computational `BasisState` preparation
are supported; amplitude-vector `StatePrep` is explicitly rejected.

Native gates include H, CNOT, CZ, Pauli gates, S/T, phase shifts, RX/RY/RZ,
IsingXX/YY/ZZ, MultiRZ and **arbitrary PauliRot words**, including mixed axes and
identity factors. PennyLane decomposes supported higher-level unitary templates,
controlled rotations and adjoints into this set. Unsupported decompositions raise
an error; there is no statevector fallback.

Torch differentiates the coefficient arithmetic, including gate parameters,
shared angles, input features, neural weights and real observable coefficients.
Zero-valued angles and coefficients retain their derivatives. Parameters outside
the observable's light cone receive zero derivatives. Tested exact paths support
second derivatives. Broadcast inputs are expanded into scalar tapes, retaining
their graphs; this implementation does not introduce a parallel worker pool.
Backpropagation uses Torch; NumPy values can be evaluated without differentiation.
Explicit PennyLane parameter-shift differentiation also works on the tested exact
rotation/expectation path, including Autograd. Torch backprop is the recommended
training route; device adjoint derivatives are not provided.

Expectations accept Pauli sums and computational-basis projectors. Variance is
computed as $\langle O^2\rangle-\langle O\rangle^2$, by multiplying the Pauli
sentences. This multiplication itself can increase the number of terms
quadratically in the observable's term count.

For probability measurements on an ordered subset of $m$ wires, the device
computes the $2^m$ Z-string moments and applies a fast Walsh–Hadamard transform:

$$
p(b)=2^{-m}\sum_{S\subseteq\{0,\ldots,m-1\}}
(-1)^{\sum_{j\in S}b_j}\langle Z_S\rangle.
$$

The transform costs $O(m2^m)$; it does not eliminate the $2^m$ moment evaluations
or outputs. A computational projector uses
$|b\rangle\langle b|=\prod_j(I+(-1)^{b_j}Z_j)/2$.
`max_probability_wires=16` bounds each probability/projector expansion. It counts
readout wires, not total register width, and can be changed explicitly.

For $G$ decomposed gates and at most $S$ retained terms per propagated observable,
the propagation work is roughly $O(GS)$ for bounded-support gates, with extra
factors for gate support and large integer masks. A live coefficient dictionary
uses $O(S)$ entries; training also retains the arithmetic graph across gates and
observables. **S can grow exponentially**, up to the $4^n$ Pauli basis. Neither
this simulator nor its readout limit provides a universal polynomial runtime or
a global memory cap. Favor sparse observables and small backward light cones;
use IQPSimulator when its commuting-circuit contract fits the problem.

## Exact and approximate settings

`PauliPropagationOptions` accepts these settings directly or through
`device_kwargs`:

| Option | Default | Meaning |
| --- | --- | --- |
| `k` | `None` | Project the initial observable to weight at most k, then truncate during propagation. |
| `prune` | `False` | Enable coefficient pruning. |
| `topM` | `None` | With pruning enabled, retain at most M terms at pruning points. |
| `tau` | `0.0` | With pruning enabled, drop magnitudes below this threshold. |
| `prune_every` | `5` | Prune every this many gates, plus at the end; zero means final-only. |
| `max_probability_wires` | `16` | Reject larger dense readouts/projector expansions before allocating their terms. |

Pruning uses detached coefficient magnitudes. Autograd differentiates surviving
paths with the selected mask fixed; it does not differentiate the discrete
selection decision. Weight truncation is independent of `prune`. These controls
change both values and gradients and carry no general error bound. Approximate
probabilities need not remain nonnegative, and approximate variances need not
remain nonnegative either. The simulator does not silently clip or renormalize
them. Use exact defaults for probability losses unless you validate the chosen
approximation for your workload.

Only analytic execution (`shots=None`) is supported. State/density-matrix
outputs, channels/noise, mid-circuit measurements and MPI are unsupported.

## Migration and compatibility

The implementation of `DifferentiablePauliProp` now lives in
[qasp/simulators/_pauli_propagation.py](../qasp/simulators/_pauli_propagation.py).
The public device is
[qasp/simulators/pauli_propagation.py](../qasp/simulators/pauli_propagation.py).
These imports still identify the same low-level engine class:

```python
from qasp import DifferentiablePauliProp
from qasp.simulators import DifferentiablePauliProp
from qasp.rl.differentiable_pp_sim import DifferentiablePauliProp  # compatibility
```

Existing `device="lightning.pp"` model adapters and their `pp_*` defaults are
preserved. For new code, use the shared device and move its approximation settings
into `device_kwargs`:

| Old model argument | New device option |
| --- | --- |
| `pp_truncation` | `k` |
| `pp_prune` | `prune` |
| `pp_topM` | `topM` |
| `pp_tau` | `tau` |
| `pp_prune_every` | `prune_every` |

The legacy engine/adapters default to pruning, whereas the **new device defaults
to exact execution**. Preserve all desired approximation options explicitly when
migrating an existing experiment. The new device applies `k` to the initial
observable as well as propagated words. The legacy adapters retain their
historical initial-observable handling, so revalidate truncated results when
migrating high-weight observables. `FastTruncatedPauliProp` remains the separate,
non-autograd engine under `qasp.rl`; it is not used by this device.
