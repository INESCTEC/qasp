# Differentiable Pauli propagation

Simulate noncommuting circuits by evolving observables backward. Start with exact execution, train with Torch, and validate optional approximations against a small full-state reference.

Run the complete guide from the repository root with `python examples/pauli_propagation_basics.py`.

## 1. When to choose Pauli propagation

Use this simulator when you need a few Pauli expectation values from a circuit with a compact backward observable expansion. The circuit can contain noncommuting gates and repeated feature encoding. It does not need the commuting structure required by the IQP simulator.

The device stores a sparse operator expansion instead of a state vector. Circuit width alone does not determine its cost: gates, observables and retained Pauli terms matter. Start with exact defaults on a small instance; then increase width or enable a validated approximation. Dense readouts can remove the computational benefit.

## 2. First circuit: build → compile → differentiate

This ansatz-only example prepares RY(theta) on wire 0, applies CNOT(0, 1), and measures Z on wire 1. No feature map or TorchLayer is needed. The compiled node receives the trainable params explicitly, and returns a sequence with one expectation.

The answer is $\langle Z_1\rangle=\cos\theta$, with derivative $-\sin\theta$. The code checks both identities. CircuitBuilder selects its Torch interface internally; compile the device with backprop and analytic shots.

```python
import torch
from qasp import AngleAnsatz, CircuitBuilder

ansatz = AngleAnsatz(
    2, n_layers=1, rotations=["Y 0"],
    include_imprimitive=True, imprimitive="CNOT",
    imprimitive_pattern=[(0, 1)], include_final_imprimitive=True,
)
builder = CircuitBuilder(2)
builder.set_ansatz(ansatz)
builder.set_measurement("expectation", observables=["Z 1"])
builder.build()
node = builder.compile(
    device="PauliPropagationSimulator",
    diff_method="backprop", shots=None,
)
theta = torch.tensor([0.37], dtype=torch.float64, requires_grad=True)
value = node(params=theta)[0]
gradient = torch.autograd.grad(value, theta)[0]
torch.testing.assert_close(value, torch.cos(theta[0]))
torch.testing.assert_close(gradient, -torch.sin(theta))
print("Exact value and gradient:", value.item(), gradient.item())
```

## 3. What happens to the observable?

For $U=G_D\cdots G_1$ and $O=\sum_P c_P P$, start with O and conjugate by gates in reverse circuit order:

$$
\langle O\rangle=\langle0^{\otimes n}|U^\dagger O U|0^{\otimes n}\rangle.
$$
 Each term is a Pauli word with a real coefficient. Clifford gates map a word to another word with a possible sign.

For $R_P(\theta)=e^{-i\theta P/2}$, a commuting word Q is unchanged; an anticommuting word branches:

$$
R_P^\dagger Q R_P=\cos\theta\,Q+i\sin\theta\,PQ.
$$
 Equal output words are merged by adding coefficients. The engine stores packed integer X/Z masks and differentiable Torch coefficients.

In the example,

$$
Z_1\xrightarrow{\mathrm{CNOT}^{\dagger}(\cdot)\mathrm{CNOT}}Z_0Z_1\xrightarrow{R_Y^{\dagger}(\cdot)R_Y}\cos\theta\,Z_0Z_1-\sin\theta\,X_0Z_1.
$$
 On the initial all-zero state, I/Z-only words contribute their coefficients and words with X or Y contribute zero. Apply this filter only after all reversed gates: an intermediate X or Y can turn back into Z.

## 4. Exact defaults and every approximation setting

Exact means no deliberate term truncation: k=None and prune=False. Floating-point rounding remains. Set options in device_kwargs, or pass a PauliPropagationOptions object as options to a device instance. Do not combine options with individual option keywords.

Pauli weight counts non-identity factors. For example, Z0 has weight 1 and X0 Z7 has weight 2, irrespective of register width. A small k can discard a contribution even when the measured observable itself has small weight.

Use k alone for a weight cutoff; use prune=True with tau and/or topM for coefficient selection. Combining them applies both. topM is enforced at pruning points; intermediate expansion, initial observable conversion and the Torch graph can require more storage.

| Option | Type | Default | Meaning |
| --- | --- | --- | --- |
| `k` | int \| None | `None` | Maximum number of non-identity factors in each retained Pauli word. Projects the initial observable and truncates during propagation, even when prune=False. This is not circuit depth or gate arity. |
| `prune` | bool | `False` | Enable coefficient-based selection using tau and topM. Does not enable or disable the independent weight cutoff k. |
| `tau` | float ≥ 0 | `0.0` | With prune=True, discard terms whose coefficient magnitude is strictly below tau, after merging equal words. An absolute threshold, not an output-error tolerance. |
| `topM` | positive int \| None | `None` | With prune=True, retain at most this many largest-magnitude terms after thresholding at each pruning point. This is not a peak-memory cap. |
| `prune_every` | int ≥ 0 | `5` | With prune=True, apply coefficient selection every this many reversed gates and once at the end. Zero means final-only selection. Weight truncation still applies throughout. |
| `max_probability_wires` | int ≥ 0 | `16` | Maximum wires in a dense probability or computational-projector expansion. Limits readout width, not circuit width or total memory. |

## 5. A cutoff can change the answer substantially

Continue the first circuit with k=1. Its backward CNOT produces the weight-two word Z0 Z1, so truncation removes it. The result and derivative become zero although the exact values at theta=0.37 are approximately 0.9323 and −0.3616. prune=False does not turn off k.

A larger cutoff is not a general guarantee of monotonically decreasing output or gradient error: discarded paths can interfere later. Validate the quantities your loss uses.

```python
truncated = builder.compile(
    device="PauliPropagationSimulator",
    diff_method="backprop", shots=None,
    device_kwargs={"k": 1, "prune": False},
)
approximate = truncated(params=theta)[0]
approximate_gradient = torch.autograd.grad(approximate, theta)[0]
print("Weight-one result:", approximate.item(), approximate_gradient.item())
# Both are zero: the backward CNOT produces Z0 Z1, which k=1 removes.
torch.testing.assert_close(approximate, torch.zeros_like(approximate))
torch.testing.assert_close(approximate_gradient, torch.zeros_like(theta))
```

## 6. Configure and train a neural policy

The same device fields work in compatible QASP modules. This example uses create_policy, a NeuralEmbedding with one- and two-qubit Pauli rotation strings, reuploading, and a noncommuting ZY ansatz. Normal comes from qasp.initialization.

PQCSoftmax returns logits; log_softmax turns them into log action probabilities. This demonstrates one action-loss update, rather than a full environment rollout. Torch owns the circuit and embedding parameters, so policy(x) takes only input features. Input derivatives are checked too.

```python
import torch
from qasp.initialization import Normal
from qasp.rl import create_policy

execution = {
    "device": "PauliPropagationSimulator",
    "device_kwargs": {"k": None, "prune": False},
    "diff_method": "backprop", "shots": None,
}
config = {
    "n_actions": 2, "observables": ["Z 0", "ZZ 0 1"], "fc_layer": None,
    "feature_map": "NeuralEmbedding",
    "feature_map_config": {"features": 2, "rotations": ["Y 0", "YY 0 1"]},
    "ansatz": "AngleAnsatz",
    "ansatz_config": {
        "n_layers": 2, "rotations": "ZY", "reuploading": True,
        "include_imprimitive": True, "imprimitive": "CNOT",
        "imprimitive_pattern": "chain",
    },
    "init": Normal(0.0, 0.3), "input_scaling_init": Normal(0.0, 0.5),
    **execution,
}
policy = create_policy("PQCSoftmax", config, n_qubits=2).double()
x = torch.tensor([0.2, -0.4], dtype=torch.float64, requires_grad=True)
optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
optimizer.zero_grad()
loss = -torch.log_softmax(policy(x), dim=-1)[1]
loss.backward()
assert x.grad is not None and torch.isfinite(x.grad).all()
assert all(p.grad is not None and torch.isfinite(p.grad).all()
           for p in policy.parameters())
optimizer.step()
print("One action-loss update:", loss.item())
```

## 7. Check outputs AND gradients before scaling

Continue from the policy above. Reuse exactly the same state_dict and input in every comparison. Check the maximum absolute logit error and the gradient error for the action loss, including every parameter and the input features.

For this two-qubit example, k=2 retains every possible weight and must match exact propagation. The pruned example deliberately limits terms and need not match. Errors depend on initialization; these settings are a demonstration, not recommended universal tolerances.

Repeat this check over representative inputs, random initializations and checkpoints after training. For larger instances without a full-state reference, compare increasing budgets and state clearly that this is convergence evidence rather than an error certificate.

```python
from copy import deepcopy

def make_reference(device, options):
    cfg = deepcopy(config)
    cfg.update(device=device, device_kwargs=options)
    model = create_policy("PQCSoftmax", cfg, n_qubits=2).double()
    model.load_state_dict(policy.state_dict())  # Identical trained parameters.
    return model

def evaluate_with_gradients(model):
    features = x.detach().clone().requires_grad_(True)
    logits = model(features)
    loss = -torch.log_softmax(logits, dim=-1)[1]
    grads = torch.autograd.grad(loss, (*model.parameters(), features))
    return logits.detach(), torch.cat([g.detach().reshape(-1) for g in grads])

reference = make_reference("default.qubit", {})
ref_value, ref_grad = evaluate_with_gradients(reference)
settings = {
    "exact": {"k": None, "prune": False},
    "weight 1": {"k": 1, "prune": False},
    "weight 2": {"k": 2, "prune": False},
    "pruned": {"prune": True, "tau": 1e-4, "topM": 8, "prune_every": 1},
}
for name, options in settings.items():
    model = make_reference("PauliPropagationSimulator", options)
    result, grads = evaluate_with_gradients(model)
    value_error = (result - ref_value).abs().max().item()
    gradient_error = (grads - ref_grad).abs().max().item()
    print(f"{name:>8}: max output error={value_error:.3e}, "
          f"max loss-gradient error={gradient_error:.3e}")
    if name in {"exact", "weight 2"}:
        torch.testing.assert_close(result, ref_value, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(grads, ref_grad, atol=1e-10, rtol=1e-10)
```

## 8. Expectations, variances and probabilities have different costs

Expectations accept real Pauli sums and computational-basis projectors. Variances evaluate $\langle O^2\rangle-\langle O\rangle^2$; an L-term observable can require $O(L^2)$ products before merging to form $O^2$.

Probabilities on m measured wires require all $2^m$ Z-string moments:

$$
p(b)=2^{-m}\sum_{S\subseteq\{0,\ldots,m-1\}}(-1)^{\sum_{j\in S}b_j}\langle Z_S\rangle.
$$
 The Walsh–Hadamard reconstruction adds $O(m2^m)$ work. It does not remove the exponential number of moments or outputs.

A basis-state projector expands as $|b\rangle\langle b|=2^{-m}\prod_j(I+(-1)^{b_j}Z_j)$. max_probability_wires=16 bounds a readout/projector expansion, not the total circuit width. Measuring a few wires in a large circuit differs from requesting its entire probability vector. Weight truncation does not avoid the initial projector conversion.

Approximate probabilities and variances can become negative. QASP does not silently clip or renormalize them. Start with exact execution for probability losses; validate any approximation before using it for training.

## 9. What differentiability guarantees

Torch backpropagation differentiates coefficient arithmetic through the retained expansion, including circuit angles, shared parameters, features, neural embedding weights and real observable coefficients. Exact tests cover zero angles/coefficients and second derivatives. Keep trainable quantities as Torch tensors; converting to NumPy or detaching them breaks the graph.

Coefficient selection uses detached magnitudes. With pruning enabled, the derivative treats the currently selected terms as fixed; it does not differentiate the event of a term entering or leaving the expansion. The resulting gradient can differ from the exact circuit gradient. This is not a straight-through estimator.

Use diff_method="backprop" with Torch. Explicit parameter-shift also works on tested exact rotation/expectation paths, including Autograd. A device adjoint derivative is not implemented. NumPy can be used for evaluation without native backpropagation.

## 10. Module integration

Reuse execution from the policy example in the model configuration. Device classes and configured instances also work; an instance keeps its own wires/options and must not also receive device_kwargs. QASP resolves the string PauliPropagationSimulator (or pauli.propagation); it is not a separately registered PennyLane plugin.

Device support does not override a model’s measurement or batching contract. The complete application script exercises a policy, Q-function, classifier, fidelity kernel, generative model and both optimization routes.

| Module | Use with this simulator | Readout or execution detail |
| --- | --- | --- |
| Building blocks | `CircuitBuilder.build()` → `compile()` | Expectations, variances and bounded probability readouts; use Torch tensors for training. |
| RL | `create_policy` with PQCSoftmax, PQCBorn or PQDQN | Pauli logits or Born readouts. Neural PQDQN inputs require sequential batching. |
| Classifiers | `create_classifier` | Expectation scores or probabilities, including classical decoder layers. |
| Kernels | `create_kernel` for fidelity or projected kernels | Trainable Gram matrices; a global fidelity projector has an exponential Pauli expansion. State-output-dependent paths are unsupported. |
| Generative learning | `create_qcbm` with MMD or NLL | Expectation-based MMD can avoid dense probabilities. NLL uses the bounded dense readout; sampling needs an explicit sampling device. |
| Optimization | QAOA and PCE | Use backend="torch" and diff_method="backprop". Noncommuting multilayer and warm-start circuits are supported when decomposable. |

- [Runnable cross-module example](https://github.com/INESCTEC/qasp/blob/main/examples/pauli_propagation_modules.py)
- [Runnable guide examples](https://github.com/INESCTEC/qasp/blob/main/examples/pauli_propagation_basics.py)

## 11. Scaling and supported operations

For D decomposed gates and at most S active terms per observable, propagation is roughly $O(DS)$ for bounded-support gates, with extra support and integer-mask costs. A live dictionary uses $O(S)$ entries; backpropagation also retains an arithmetic graph across gates and observables. Multiple readouts require multiple propagations.

In the worst case $S\leq4^n$ still allows exponential work. Restricting weight to a fixed k limits the retained word space to $\sum_{j=0}^{k}3^j{n\choose j}$, but gives no general accuracy guarantee. Initial observable expansion, intermediate terms and the training graph have their own costs. No universal hundred-qubit runtime or memory bound follows.

Native gates include H, CNOT, CZ, Pauli gates, S/T, phase shifts, RX/RY/RZ, Ising rotations, MultiRZ and arbitrary PauliRot words, including mixed axes and identity factors. Higher unitary templates, controlled rotations and adjoints must decompose into supported gates. The initial state is all-zero; BasisState preparation is decomposed into gates. Unsupported operations raise an error with no statevector fallback.

Only shots=None is supported. Amplitude-vector StatePrep, noise/channels, mid-circuit measurements, state/density outputs and finite-shot sampling are unsupported. Broadcast inputs become scalar tapes evaluated sequentially. This backend has no dedicated worker pool or MPI implementation; QASP’s distributed facilities belong to other execution routes.

## 12. References and the scope of theoretical results

Rall et al. give foundational Pauli-propagation simulation algorithms, including stochastic estimators. Begušić, Hejazi and Chan study truncated Heisenberg expansions through Clifford perturbation theory. Rudolph et al. give a broad account from bit-level representations to applications. QASP uses deterministic sparse propagation with its own weight and coefficient controls; citing these works does not imply that it implements every algorithm in them.

Angrisani et al. prove efficient observable estimation under random-circuit ensemble assumptions, including invariance of layer distributions under single-qubit rotations. Their high-probability guarantee over such ensembles is not a pointwise guarantee for an arbitrary trained circuit. QASP does not certify these assumptions or attach their error bound to k, tau or topM.

- [Rall et al. (2019) — Simulation of Qubit Quantum Circuits via Pauli Propagation](https://doi.org/10.1103/PhysRevA.99.062337)
- [Begušić, Hejazi & Chan (2023 preprint) — Clifford perturbation theory](https://arxiv.org/abs/2306.04797)
- [Angrisani et al. (2025) — Classically Estimating Observables of Noiseless Quantum Circuits](https://doi.org/10.1103/lh6x-7rc3)
- [Rudolph et al. (2026) — Pauli Propagation: A Computational Framework for Simulating Quantum Systems](https://doi.org/10.1103/6vd7-l9bn)

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
