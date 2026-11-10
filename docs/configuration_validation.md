# Configuration and simulator validation

The configuration regression tests compare outputs and input/parameter
gradients with independent statevector execution. Fisher information is also
checked against finite differences of probabilities and quantum state vectors.
These are CPU correctness checks, not simulation-speed or learning-advantage
benchmarks. Run `python -m pytest tests/test_configuration_*.py` with the test
dependencies and optional `quimb`/`cotengra` installed.

## Construction choices

| Configuration | Behavior |
| --- | --- |
| Feature map only, no `ansatz` or `ansatz_config` | Supported directly by fidelity/projected kernel factories. Scaled/neural encodings need fixed scales or a scale initializer. |
| `ansatz` with no `ansatz_config` | Uses that template's defaults. |
| Ansatz only, no feature map | Direct builders accept `node(params=...)`. TorchLayers accept placeholder inputs and preserve batch axes; the circuit is input-independent. Generative learning and optimization naturally support unencoded circuits. |
| Preconfigured `CircuitBuilder` | Supported by classifier, policy and Q-function constructors/factories, including unencoded ansätze and encoding-only adapters. |
| Feature map plus ansatz | Angle/Neural embeddings and Angle/IQP ansätze are tested with scaling, explicit Pauli words, batches and reuploading. |

A fixed encoding with no variational weights or trainable input scales is not
a trainable quantum model. An unencoded ansatz produces the same prediction
for every input; it can serve as an input-independent baseline.

```python
import torch
from qasp import AngleAnsatz, CircuitBuilder

builder = CircuitBuilder(2)
builder.set_ansatz(AngleAnsatz(2, n_layers=1, rotations="Y"))
builder.set_measurement("expectation", observables=["Z 0", "Z 1"])
builder.build()
node = builder.compile(device="default.qubit", diff_method="backprop")
theta = torch.tensor([0.2, 0.4], dtype=torch.float64, requires_grad=True)
expectations = node(params=theta)
sum(expectations).backward()

layer = builder.create_torchlayer(params=theta.detach()).double()
outputs = layer(torch.zeros((3, 2), dtype=torch.float64))
```

## Kernels without an ansatz

```python
import numpy as np
from qasp import create_kernel

X = np.array([[0.2, -0.3], [0.6, 0.4]])
for kind in ("fidelity", "projected"):
    kernel = create_kernel({
        "kernel_type": kind,
        "feature_map": "AngleEmbedding",
        "feature_map_config": {"features": 2, "rotations": "X"},
        "device": "default.qubit",
    }, n_qubits=2)
    print(kind, kernel.gram(X))
```

Fisher kernels have a different domain. The following constructs an
encoding-only probability circuit, differentiates its input scales at a fixed
feature vector, then compares binary outcomes:

```python
import pennylane as qml
from qasp import AngleEmbedding
from qasp.qkernels import QFKernel

embedding = AngleEmbedding(2, features=2, rotations="X", input_scaling=True)
features = qml.numpy.array([0.3, 0.7], requires_grad=False)

@qml.qnode(qml.device("default.qubit", wires=2), interface="autograd")
def probability_circuit(scales):
    embedding(features, input_scaling_params=scales)
    return qml.probs(wires=[0, 1])

scales = qml.numpy.array([0.6, 0.9], requires_grad=True)
kernel = QFKernel(probability_circuit, scales, fisher="classical")
K = kernel.gram([[0, 0], [0, 1], [1, 0], [1, 1]])
```

The test suite additionally constructs this circuit through CircuitBuilder and
checks quantum Fisher, ansatz-only circuits, mixed configurations, Torch and
Autograd, normalization, cross-Gram matrices, zero probabilities and drawing.

## Execution boundaries

| Backend | Tested behavior / boundary |
| --- | --- |
| `default.qubit`, `default.mixed` | Analytic values and gradients, compatible state/probability/expectation workflows. |
| `lightning.qubit` | Adjoint expectations and parameter-shift probabilities. Builder preprocessing expands templates/adjoints and batches to preserve gradients. |
| `default.tensor` MPS / TN | Expectations and parameter-shift training. Full probabilities are unsupported by these PennyLane devices: NLL, Fisher score kernels and QAOA probability-based solution decoding are explicit rejection tests. |
| `IQPSimulator`, TN | Compatible IQP circuits, including inverse encodings in fidelity overlaps and classical Fisher probability circuits with Torch. Quantum Fisher auxiliary circuits can be non-IQP and are then rejected. |
| `IQPSimulator`, VN | Diagonal expectation values. Fidelity projectors use `vn_observable_sampling="importance"`; sampled comparisons use statistical tolerances. No full probabilities. |
| `IQPSimulator`, auto | Selects a compatible IQP backend or the documented size-limited fallback. Auto is not a guarantee that every circuit executes as IQP. |
| `PauliPropagationSimulator` | Exact small-circuit comparisons with pruning disabled, Torch training, probabilities/expectations, classical and compatible quantum Fisher workflows. |

Born policies can use `training_method="expectation"` to reconstruct action
probabilities from Pauli expectations on VN and PennyLane tensor devices.
Generative MMD likewise uses expectations; NLL requires full probabilities.
Unsupported combinations are tested as explicit rejections, not silently
replaced with a different simulator.

GPU plugins, optional Kokkos and real MPI/distributed runs require separate
hardware validation. The local tests cover nine CPU execution configurations;
they do not exhaust every gate, width, topology and dependency version.

## Removed trainer prototypes

The unused `TorchRLTrainer` classes and their legacy `PQCWrapper` were removed
after a repository-wide usage check. `PQCBornWrapper` remains for external
TorchRL actors. Maintained REINFORCE, Reinforce, PPO, contextual-bandit and
DeepQLearning implementations remain; real short learning loops verify finite
statistics, gradients and weight updates.
