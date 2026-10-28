# QASP

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/qasp-logo-dark.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/qasp-logo.png">
    <img src="docs/assets/qasp-logo.png" alt="QASP — Quantum Agent Software Package" width="600">
  </picture>
</p>

**QASP** (**Q**uantum **A**gent **S**oftware **P**ackage) is a Python toolkit for building, training, and analyzing parameterized quantum circuit (PQC) models with a strong focus on:

- quantum reinforcement learning,
- expectation-value and Born-probability quantum classification,
- contextual bandits and policy-gradient training,
- Born-rule and softmax quantum policies,
- reusable feature maps and ansatz templates,
- quantum-kernel experiments,
- QUBO optimization with QAOA, warm starts, RQAOA, LR-QAOA, and PCE,
- initialization-concentration diagnostics and exploratory Lie-closure analyses,
- approximate simulation via truncated Pauli propagation.

The package is organized so that the same circuit-building primitives can be
reused across classification, optimization, RL, generative-learning, and
kernel workflows.

---

## Highlights

- Build PQCs from composable **feature maps** and **ansätze**
- Train **Born** and **softmax** policies on Gym/Gymnasium-style environments
- Run lightweight **REINFORCE** training loops directly from QASP
- Evaluate trainability with utilities such as:
  - model-agnostic cost-concentration diagnosis,
  - policy variance,
  - exploratory Lie-closure dimensions (with documented heuristic limitations)
- Use PennyLane **lightning.qubit** / **lightning.kokkos** and QASP's internal differentiable **lightning.pp** route
- Compare different observable/readout designs, including local `Z`, pairwise `ZZ`, and parity-based observables
- Build global-fidelity, local-fidelity, and projected quantum kernels, including trainable IQP encodings
- Optimize trainable kernels with centered kernel-target alignment
- Train binary and multiclass PQC classifiers with MSE, hinge, binary cross-entropy, or categorical cross-entropy
- Decode Born probabilities into classes with contiguous, parity, or majority partitions
- Evaluate classifiers with accuracy, balanced accuracy, precision, recall, F1, ROC AUC, PR AUC, confusion matrices, log loss, and Brier score
- Solve QUBOs with standard, multi-angle, warm-start, recursive, and linear-ramp QAOA
- Define custom QAOA mixers and PCE parameterized circuits with QASP dictionaries
- Export [paper-ready circuit drawings](#exporting-publication-ready-tikz-circuits) as editable TikZ/Quantikz, with symbolic parameters, gate colors, and repeated-layer compression

---

## Documentation

Start with the [package guide](docs/package_guide.md) for the complete module
map, executable examples, data shapes and implementation boundaries. Every
Python module, class and function under `qasp` has an in-source docstring.
The [documentation audit](docs/documentation_audit.md) records verification and
environment limitations. Focused guides cover [Q-functions](docs/qfunctions.md)
and [generative learning](docs/genlearn.md).

## Installation

### From source

```bash
git clone https://github.com/andre-sequeira10/QASP
cd QASP
python -m pip install -e .
```

### Development install

```bash
python -m pip install -e .[dev]
```

---

## Dependencies

QASP relies primarily on:

- `pennylane`
- `torch`
- `numpy`
- `gymnasium`
- `torchrl`
- `tensordict`
- `matplotlib`
- `pandas`
- `ray[rllib]`

The exact dependency list should be kept in `pyproject.toml`.

---

## Package layout

```text
qasp/
├── circuits/
│   ├── constructor.py      # CircuitBuilder and compilation utilities
│   ├── templates.py        # Ansatz templates such as IQPAnsatz and AngleAnsatz
│   ├── tikz.py             # Quantikz/TikZ circuit export
│   └── uqc.py              # Historical circuit prototype (see limitations)
├── feature_maps/
│   └── templates.py        # AngleEmbedding and NeuralEmbedding
├── genlearn/
│   ├── models.py           # PQCBM training and device sampling
│   ├── factory.py          # Dictionary-based create_qcbm
│   ├── drawing.py          # draw_qcbm / Quantikz export
│   └── losses.py           # NLL and expectation-mixture MMD
├── qkernels/
│   ├── kernels.py          # Fisher-information kernel utilities
│   ├── quantum_kernels.py  # Fidelity, local-fidelity, and projected kernels
│   ├── factory.py          # Configuration-driven kernel construction
│   ├── kta.py              # Kernel-target alignment
│   └── use_cases/          # Kernel experiments and examples
├── classifiers/
│   ├── models.py           # PQCClassifier expectation/Born readouts
│   ├── decoders.py         # Contiguous, parity, and majority class decoders
│   ├── losses.py           # MSE, hinge, BCE, and categorical CE
│   ├── metrics.py          # Classification evaluation metrics
│   ├── training.py         # Compact PyTorch training/evaluation loop
│   └── factory.py          # create_classifier configuration factory
├── optimization/
│   ├── problem.py          # QUBO model and QUBO-to-Ising conversion
│   ├── qaoa.py             # standard, multi-angle, warm-start, and LR-QAOA
│   ├── recursive.py        # recursive QAOA elimination and reconstruction
│   ├── pce.py              # Pauli Correlation Encoding with QASP ansätze
│   └── optimizers.py       # reusable SciPy optimizer adapters
├── diagnostics/
│   └── barren_plateau.py   # classifier/kernel/policy cost concentration
├── hpc/                    # Workload/memory planning, Slurm generation and execution
├── devices.py              # Shared device/MPI option handling
└── rl/
    ├── model.py            # create_model, draw_model
    ├── policies.py         # PQCBorn, PQCSoftmax, related policy classes
    ├── metrics.py          # Legacy probability variance / heuristic Lie closure
    ├── pp_sim.py           # truncated Pauli propagation simulator
    ├── differentiable_pp_sim.py
    ├── torchrl.py          # DQN, REINFORCE, PPO, bandits and TorchRL adapter
    ├── qfunctions/         # PQDQN models, Pauli readouts and drawings
    ├── envs.py             # Windy target-navigation environment
    └── trl.py              # Historical TorchRL prototype (see limitations)

tests/
├── test_feature_maps.ipynb
├── test_qml_models.ipynb
├── test_policy_training.ipynb
├── optimization_algorithms.ipynb
├── trainability_born_paper.ipynb
└── tests_deucalion/        # cluster scripts and experiment drivers
```

---

## Top-level API

The package re-exports the most common entry points from `qasp.__init__`.

### Circuits

- `CircuitBuilder`
- `IQPAnsatz`
- `AngleAnsatz`
- `UQC`

### Feature maps

- `AngleEmbedding`
- `NeuralEmbedding`

### Generative learning

- `PQCBM`, `create_qcbm`, `draw_qcbm`, `PauliMMD`
- `BornMachine` (deprecated compatibility wrapper)

See [the generative-learning guide](docs/genlearn.md) and
[the runnable PQCBM notebook](examples/pqcbm_generative_learning.ipynb) for
expectation-based MMD, differentiable Pauli propagation, sampling, and MPI.

A three-qubit Born machine generates bitstrings by measuring a two-layer
`AngleAnsatz`. This example builds its sampling circuit and saves the complete
paper-ready LaTeX document; no training is needed to draw the topology.

<!-- readme-circuit: generative -->

```python
from pathlib import Path
from qasp import create_qcbm, draw_qcbm

model = create_qcbm({
    "ansatz": "AngleAnsatz",
    "ansatz_config": {
        "n_layers": 2,
        "rotations": "YZ",
        "include_imprimitive": True,
        "imprimitive": "CNOT",
        "imprimitive_pattern": "chain",
    },
    "device": "default.qubit",
    "seed": 7,
}, n_qubits=3)
tex = draw_qcbm(model, output="tikz", tikz_options={
    "standalone": True, "symbolic_parameters": True, "compress": True,
})
Path("born_machine.tex").write_text(tex, encoding="utf-8")
```

[![Three-qubit Born machine with Y/Z rotations, CNOT-chain entanglers, a compact second layer, and computational-basis measurements](docs/assets/circuits/born_machine.png)](docs/assets/circuits/born_machine.pdf)

[Editable TikZ](docs/assets/circuits/born_machine.tex) · [Vector PDF](docs/assets/circuits/born_machine.pdf)

### Kernels

- `QFKernel`
- `FidelityKernel`
- `ProjectedKernel`
- `create_kernel`, `draw_kernel`
- `KernelTargetAlignment`, `align_kernel`, `target_alignment`

### Quantum classification

- `PQCClassifier`, `QuantumClassifier`
- `create_classifier`, `draw_classifier`
- `ClassifierTrainer`, `ClassifierLoss`
- `ClassProbabilityDecoder`
- `mean_squared_error`, `hinge_loss`
- `binary_cross_entropy`, `categorical_cross_entropy`
- `classification_metrics`, `confusion_matrix`, `accuracy_score`, `f1_score`
- `roc_auc_score`, `pr_auc_score`, `average_precision_score`

### Combinatorial optimization

- `QUBO`
- `QAOA`, `WarmStartQAOA`
- `RecursiveQAOA`, `RQAOA`
- `LRQAOA`, `LinearRampQAOA`
- `PauliCorrelationEncoding`, `PCE`
- `create_optimizer`
- `draw_optimization`, `draw_qaoa`, `draw_pce`, `draw_recursive_qaoa`

<!-- readme-circuit: optimization -->

```python
from qasp import QAOA, QUBO

problem = QUBO.maxcut(3, [(0, 1), (1, 2), (0, 2)])
mixer = {
    "rotations": ["X 0", "X 1", "X 2", "XX 0 1"],
    "initial_state": "plus",
}

solver = QAOA(
    problem,
    n_layers=2,
    parametrization="multi_angle",
    mixer=mixer,
)
```

```python
result = solver.solve(optimizer="COBYLA", optimizer_options={"maxiter": 100})
print(result.bitstring, result.energy)
```

Draw the same two-layer, multi-angle QAOA circuit, including its custom
single-qubit `X` and two-qubit `XX` mixer. Symbolic labels show the parameter
structure independently of training. To show fitted numerical values instead,
pass `parameters=result.parameters` and set `symbolic_parameters=False`.

<!-- readme-circuit: optimization -->

```python
from pathlib import Path
from qasp import draw_optimization

tex = draw_optimization(
    solver, parameters=solver.initial_parameters("zeros"), output="tikz",
    tikz_options={"standalone": True, "symbolic_parameters": True, "compress": True},
)
Path("qaoa.tex").write_text(tex, encoding="utf-8")
```

[![Two-layer multi-angle triangle MaxCut QAOA circuit with a custom X and XX mixer](docs/assets/circuits/qaoa.png)](docs/assets/circuits/qaoa.pdf)

[Editable TikZ](docs/assets/circuits/qaoa.tex) · [Vector PDF](docs/assets/circuits/qaoa.pdf)

Warm starts accept either a relaxed solution vector or `True` to compute a
bounded continuous relaxation. PCE accepts the same `ansatz` / `ansatz_config`
dictionary convention used by the classifier and RL factories. See
`tests/optimization_algorithms.ipynb` for executable examples of every
algorithm.

QAOA minimizes the expectation value of the QUBO Hamiltonian directly. With
analytic execution (`shots=None`), the default differentiation method is
adjoint. With finite-shot training, it switches to parameter-shift; either use
that gradient with a gradient-aware optimizer or train the noisy expectation
directly with a derivative-free optimizer such as COBYLA.

```python
# Analytic Hamiltonian training with adjoint gradients.
analytic = QAOA(problem, n_layers=2)
analytic_result = analytic.solve(optimizer="BFGS", use_gradients=True)

# Finite-shot training followed by an independent 2,000-shot solution estimate.
sampled = QAOA(problem, n_layers=2, shots=500)
sampled_result = sampled.solve(
    optimizer="COBYLA",
    result_shots=2_000,
)
print(sampled_result.counts)
print(sampled_result.bitstring, sampled_result.solution_probability)

# Resample trained parameters at any later shot count.
counts = sampled.solution_counts(sampled_result.parameters, shots=10_000)
```

QAOA and PCE can train with either the default NumPy/Autograd backend or a
PyTorch backend. The NumPy backend supports the existing SciPy methods plus
PennyLane-native optimizers; the Torch backend uses `torch.optim` and keeps the
QNode, loss, gradients, and parameters as Torch tensors during training.

```python
# PyTorch + adjoint differentiation on an analytic simulator.
torch_solver = QAOA(problem, n_layers=2, backend="torch")
torch_result = torch_solver.solve(
    optimizer="Adam",
    optimizer_options={"max_steps": 100, "lr": 0.05},
)

# PyTorch + parameter-shift differentiation with finite-shot training.
noisy_torch = QAOA(problem, backend="torch", shots=1_000)
noisy_result = noisy_torch.solve(
    optimizer="RMSprop",
    optimizer_options={"max_steps": 50, "lr": 0.02},
    result_shots=5_000,
)

# PennyLane optimizers use the NumPy/Autograd backend.
qng_result = QAOA(problem, backend="numpy").solve(
    optimizer="QNG",
    optimizer_options={"max_steps": 50, "stepsize": 0.02, "lam": 1e-3},
)
spsa_result = QAOA(problem, shots=1_000).solve(
    optimizer="SPSA",
    optimizer_options={"max_steps": 100, "a": 0.05, "c": 0.1},
)
```

Available optimizer families are exported as `SCIPY_OPTIMIZERS`,
`PENNYLANE_OPTIMIZERS`, and `TORCH_OPTIMIZERS`:

- SciPy: COBYLA, Nelder-Mead, Powell, BFGS, L-BFGS-B, and SLSQP.
- PennyLane: gradient descent, Adam, Adagrad, Momentum, Nesterov Momentum,
  RMSProp, quantum natural gradient (QNG), SPSA, quantum-natural SPSA
  (QNSPSA), and Shot Adaptive optimization.
- PyTorch: Adam, AdamW, SGD, RMSprop, Adagrad, Adadelta, ASGD, and LBFGS.

`QNG`, `QNSPSA`, and `ShotAdaptive` require a single scalar quantum objective,
so they apply directly to QAOA. PCE's loss is a nonlinear classical function
of several Pauli expectations; it supports the ordinary PennyLane gradient
optimizers and every listed Torch optimizer, but not those three QNode-only
optimizers. `ShotAdaptive` additionally requires finite-shot QAOA execution.

Optimization circuits use the same drawing outputs as the rest of QASP. TikZ
exports default to symbolic QAOA parameters and can produce a complete,
standalone paper-ready document:

```python
tikz = sampled.draw(
    sampled_result,
    output="tikz",
    tikz_options={
        "standalone": True,
        "compress": True,
        "wire_labels": [r"q_0", r"q_1", r"q_2"],
        "gate_colors": {
            "single": "DCECF8",
            "two": "E4D8F2",
            "measurement": "E8E5F5",
        },
    },
)

# The high-level helper also constructs and draws a model from configuration.
tikz = draw_optimization(
    problem=problem,
    algorithm="multi_angle_qaoa",
    config={"n_layers": 2},
    output="tikz",
)
```

Use `output="matplotlib"` for a figure/axes pair or `output="text"` for an
ASCII circuit. PCE exposes the same `.draw(...)` method. After `RQAOA.solve()`,
`rqaoa.draw(step=0, ...)` draws any constituent recursive QAOA step.

### Quantum-classifier examples

The classifier factory uses the same class-and-config notation as QASP's other
modules. Probability readout returns a normalized distribution over any
discrete number of classes; expectation readout returns one binary score or one
score per class.

The two readout families have different mathematical meanings:

- **Probability readout** measures computational-basis Born probabilities and
  sums them into class probabilities with a contiguous, parity, or majority
  decoder. These outputs are genuine normalized quantum measurement
  probabilities and naturally pair with binary or categorical cross entropy.
- **Expectation readout** measures user-selected observables. For binary
  classification QASP uses
  `score = mean(<O_1>, ..., <O_M>)`, predicts class 1 exactly when `score > 0`,
  and encodes MSE/hinge targets as `{-1, +1}`. The sigmoid values returned by
  `predict_proba` are classical score post-processing, not Born probabilities.
  Multiclass expectation readout accepts either one observable per class or
  one nonempty observable group per class. It averages each group into its
  class score, applies temperature-softmax in `predict_proba`, and predicts
  with `argmax`.

<!-- readme-circuit: classifier -->

```python
import torch
from qasp import ClassifierTrainer, create_classifier, draw_classifier

config = {
    "feature_map": "AngleEmbedding",
    "feature_map_config": {"rotations": "Y", "input_scaling": True},
    "ansatz": "AngleAnsatz",
    "ansatz_config": {
        "n_layers": 2,
        "rotations": "ZY",
        "include_imprimitive": True,
        "imprimitive_pattern": "ring",
        "reuploading": True,
    },
    "readout": "probabilities",
    "decoder": "parity",       # contiguous, parity, or majority
    "init": "small_normal",
}

classifier = create_classifier(config, n_qubits=4, n_classes=3)
```

```python
trainer = ClassifierTrainer(
    classifier,
    objective="categorical_cross_entropy",
    learning_rate=0.02,
)
trainer.fit(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train),
    epochs=20,
    batch_size=16,
    validation_data=(X_test, y_test),
)
metrics = trainer.evaluate(X_test, y_test)
# Includes metrics["roc_auc"] and metrics["pr_auc"]. PR AUC is step-wise
# average precision, which is usually more informative for rare positives.
```

Export this classifier's Born-probability circuit with symbolic input scaling
and a compact second reuploading layer. The parity class decoder is classical
post-processing and is not drawn as a quantum gate.

<!-- readme-circuit: classifier -->

```python
from pathlib import Path

tex = draw_classifier(
    classifier, inputs=torch.zeros(4), output="tikz",
    tikz_options={"standalone": True, "symbolic_parameters": True, "compress": True},
)
Path("classifier_born.tex").write_text(tex, encoding="utf-8")
```

[![Four-qubit Born classifier with input scaling, reuploading, and a compact second layer](docs/assets/circuits/classifier_born.png)](docs/assets/circuits/classifier_born.pdf)

[Editable TikZ](docs/assets/circuits/classifier_born.tex) · [Vector PDF](docs/assets/circuits/classifier_born.pdf)

Binary expectation-value classification with an averaged custom observable and
MSE is configured directly in the same notation:

<!-- readme-circuit: classifier -->

```python
expectation_config = {
    **config,
    "readout": "expectation",
    "decoder": None,
    "observable": ["Z 0", "ZZ 0 1", "X 1"],
}
classifier = create_classifier(expectation_config, n_qubits=4, n_classes=2)
trainer = ClassifierTrainer(classifier, objective="mean_squared_error")

# Publication-ready standalone Quantikz source with symbolic parameters.
tex = draw_classifier(
    classifier,
    inputs=torch.zeros(4),
    output="tikz",
    tikz_options={"standalone": True, "symbolic_parameters": True, "compress": True},
)
Path("classifier_expectation.tex").write_text(tex, encoding="utf-8")
```

[![The same four-qubit classifier with Z0, Z0Z1, and X1 expectation readouts](docs/assets/circuits/classifier_expectation.png)](docs/assets/circuits/classifier_expectation.pdf)

[Editable TikZ](docs/assets/circuits/classifier_expectation.tex) · [Vector PDF](docs/assets/circuits/classifier_expectation.pdf)


For a three-class expectation classifier, nested groups define the observables
whose expectation values are averaged into each class score:

```python
multiclass_config = {
    **config,
    "readout": "expectation",
    "decoder": None,
    "observables": [
        ["Z 0", "X 0", "ZZ 0 1"],  # class 0
        ["Z 1", "X 1"],             # class 1
        ["Z 2", "X 2", "ZZ 2 3"],  # class 2
    ],
    "temperature": 0.8,
}
classifier = create_classifier(multiclass_config, n_qubits=4, n_classes=3)
scores = classifier(X_train[:8])
probabilities = classifier.predict_proba(X_train[:8])
```

Expectation classifiers can use QASP's differentiable Pauli-propagation
backend with the same configuration used by observable-based RL policies:

```python
pp_config = {
    **multiclass_config,
    "device": "lightning.pp",
    "diff_method": "backprop",
    "pp_truncation": 3,   # maximum retained Pauli weight; None disables it
    "pp_topM": 20_000,    # maximum retained terms; None disables it
    "pp_tau": 1e-10,      # coefficient pruning threshold
    "pp_prune_every": 5,
    "pp_prune": True,
}
classifier = create_classifier(pp_config, n_qubits=4, n_classes=3)
trainer = ClassifierTrainer(
    classifier,
    objective="categorical_cross_entropy",
    learning_rate=0.02,
)
trainer.fit(X_train, y_train, epochs=20, batch_size=16)
```

`lightning.pp` is a QASP logical backend, not a PennyLane device. It supports
analytic expectation readout with Pauli observables and differentiates through
the retained propagation paths. Classifier batches are propagated sample by
sample and stacked without breaking autograd. Born-probability readout, finite
shots, and MPI are rejected. Exact propagation on the supported gate set is
obtained with `pp_truncation=None`, `pp_topM=None`, `pp_tau=0.0`, and
`pp_prune=False`; truncated runs should always be checked against a feasible
exact state-vector reference.

For majority decoding, every majority block produces one code bit (strict
majority; even-block ties map to zero), blocks are interpreted most-significant
first, and the resulting integer is reduced modulo `n_classes`. Default blocks
are odd and maximize qubit coverage. Non-power-of-two class counts and explicit
even blocks can induce a non-uniform reference class prior; inspect
`classifier.probability_decoder.reference_class_prior`, or use
`strict_majority_blocks=True` to require equal, disjoint odd blocks of size at
least three with no unused measured wires.

For executable, assertion-driven sklearn integrations using Breast Cancer,
Iris, Wine, and Digits, see
`examples/quantum_classifiers_sklearn_validation.ipynb`.

For cross-module validation of cost and circuit-expectation concentration on
classifiers, trainable kernels, and RL policies, see
`examples/barren_plateau_diagnoser_validation.ipynb`.

### IQP quantum-kernel example

An IQP ansatz turns a kernel into a trainable QASP model while preserving the
same feature-map, device, and differentiation configuration used elsewhere in
the package:

<!-- readme-circuit: kernel -->

```python
from qasp import create_kernel

config = {
    "kernel_type": "fidelity",  # or projected
    "feature_map": "AngleEmbedding",
    "feature_map_config": {
        "rotations": "Y",
        "imprimitive_pattern": "ring",
    },
    "ansatz": "IQPAnsatz",
    "ansatz_config": {
        "n_layers": 2,
        "imprimitive_pattern": "ring",
        "reuploading": True,
    },
    "device": "default.qubit",
}

kernel = create_kernel(config, n_qubits=4)
```

```python
K_train = kernel.gram(X_train)
K_test = kernel.gram(X_test, X_train)

# Localized fidelity uses the same class with explicit projector subsets.
local_config = {
    **config,
    "projectors": ["P 0 1", "P 0 2", "P 0 1 2"],
}
local_kernel = create_kernel(local_config, n_qubits=4)
```

The fidelity circuit evaluates an overlap between the two encoded samples.
This example keeps every gate visible because mirrored layer compression
requires `reuploading=False`. Open the vector PDF to inspect the full-width
circuit. The parameter labels below explicitly reuse the same angles in the
adjoint half. Each half's `f_i` symbols denote its respective input sample.

<!-- readme-circuit: kernel -->

```python
from pathlib import Path
import numpy as np
from qasp import draw_kernel

n_angles = kernel.qlayer.state_dict()["params"].numel()
labels = [rf"\theta_{{{i}}}" for i in range(1, n_angles + 1)]
tex = draw_kernel(
    kernel=kernel, x=np.zeros(4), y=np.ones(4), output="tikz",
    tikz_options={
        "standalone": True,
        "symbolic_parameters": True,
        "parameter_labels": labels + labels[::-1],
    },
)
Path("fidelity_kernel.tex").write_text(tex, encoding="utf-8")
```

[![Four-qubit two-layer IQP fidelity overlap circuit with data reuploading and the adjoint encoding](docs/assets/circuits/fidelity_kernel.png)](docs/assets/circuits/fidelity_kernel.pdf)

[Editable TikZ](docs/assets/circuits/fidelity_kernel.tex) · [Vector PDF](docs/assets/circuits/fidelity_kernel.pdf)

With no `projectors` key, `FidelityKernel` measures the global all-zero
projector. Each explicit projector label `"P i j ..."` measures the probability that the listed
qubits are jointly in the all-zero state, with the identity on every unlisted
qubit. A single projector containing every qubit recovers global fidelity.
For proper subsets, this localized similarity is not guaranteed to be PSD for
arbitrary encoding circuits; inspect the Gram spectrum before using it with a
method that assumes a Mercer kernel.

See `examples/iqp_quantum_kernels.py` for global and localized modes of
`FidelityKernel`, projected features, and trainable kernel-target alignment.
For an assertion-driven notebook covering circuit drawings, Gram diagnostics,
gradients, and precomputed-kernel SVMs for every kernel class, run
`examples/iqp_quantum_kernels_validation.ipynb`.

### RL / training / analysis

- `create_model`
- `draw_model`
- `PQCBorn`
- `PQCSoftmax`
- `PQCBornVariance`
- `LearnablePQCBorn`
- `FastTruncatedPauliProp`
- `REINFORCE`
- `TorchRLTrainer`
- `ContextualBanditTrainer`
- `policy_variance`
- `compute_DLA_generators`
- `fourier_analysis`
- `hadamard_decomposition`
- `cz_decomposition`
- `cnot_decomposition`

---

## Distributed statevector simulation (MPI)

Pass `device="lightning.kokkos", mpi=True` to QAOA (including warm-start and
linear-ramp variants), PCE, RQAOA, classifiers, kernels, RL policies, or
`CircuitBuilder.compile`. Classifier and RL configurations also retain the
older `enable_mpi=True` alias where it was previously supported. Kernels
already accepted `mpi`; all named device paths now forward it, including
`lightning.gpu` and custom MPI-capable plugins. Unsupported plugins raise
their own error; QASP does not fall back to a serial device.

```python
from qasp import QAOA, QUBO, PCE, RecursiveQAOA, create_classifier, create_kernel

problem = QUBO.maxcut(30, [(i, i + 1) for i in range(29)])
solver = QAOA(
    problem, device="lightning.kokkos", mpi=True,
    backend="torch", diff_method="adjoint", seed=42,
    device_kwargs={"batch_obs": True},
)
result = solver.solve(
    optimizer="Adam", optimizer_options={"max_steps": 50, "lr": 0.02},
    result_shots=1000, include_probabilities=False,
)
# Counts-only results avoid allocating a 2**n probability vector on each rank.
print(result.bitstring, result.solution_probability)

# PCE also accepts device/mpi/device_kwargs inside its circuit dictionary.
pce = PCE(problem, device="lightning.kokkos", mpi=True, seed=42)
recursive = RecursiveQAOA(
    problem, device="lightning.kokkos", mpi=True, cutoff=10,
    qaoa_options={"seed": 42},
)
# Use result_shots=... and include_probabilities=False in recursive.solve()
# for counts-only step results and direct Z/ZZ correlation measurements.

# Add these options to existing classifier/kernel factory dictionaries:
distributed_options = {
    "device": "lightning.kokkos", "mpi": True, "diff_method": "adjoint",
}
# classifier = create_classifier({**classifier_config, **distributed_options}, n_qubits, n_classes)
# kernel = create_kernel({**kernel_config, **distributed_options}, n_qubits)
```

Launch the same script on every MPI rank using the cluster's MPI-enabled
PennyLane/Lightning installation. All ranks must execute the same circuit
calls in the same order, with identical parameters, data, optimizer state,
and minibatches. Seed NumPy and Torch identically before model construction;
pass the same explicit `seed` to QAOA/PCE and stochastic optimizers. Only
printing and file writes should be restricted to rank zero. Do not scatter
different kernel pairs or training examples across ranks sharing one
distributed statevector. QASP forwards the device configuration; it does not
synchronize a user's classical training loop or replace it with distributed
data-parallel training.

Use adjoint differentiation for supported analytic expectation measurements,
and parameter-shift for finite shots. Classifiers and RL policies default to `diff_method="best"`
so PennyLane can select a method supported by the device. An explicitly chosen
`backprop` method is not supported by Lightning. Measurements, observables,
and gradient transforms must be supported by the installed plugin version.
Full-state outputs, full probability readouts, PQCBM NLL, and QFKernel
can still require exponential host memory; MPI distributes the
simulator state, not those classical arrays. Prefer expectation readouts or
small measurement registers for classifiers. PQCBM MMD trains directly on
Pauli-Z expectations without constructing the full probability vector. `include_probabilities=False`
requires `result_shots` and returns `probabilities=None`, with sparse `counts`
and `solution_probability` retained. The default result format is unchanged.
Keep enough qubits per MPI rank in compressed PCE registers and at the RQAOA
cutoff; RQAOA does not shrink the MPI communicator as variables are eliminated.

Additional device constructor options are accepted via `device_kwargs` in
optimization, classifiers, kernels, PQCBM, and the circuit builder.
For an already constructed PennyLane device/QNode, enable MPI at creation;
QASP reuses it and cannot turn a serial object into an MPI object.
The existing `lightning.pp` Pauli-propagation backend is not an MPI statevector
device and rejects `mpi=True`.

Cluster validation (small deterministic circuits, serial-reference comparisons,
gradients, a QAOA optimizer step, and sampled results):

```bash
mpiexec -n 2 python tests/tests_deucalion/mpi_modules_smoke.py --device lightning.kokkos --qubits 6
```

Use your scheduler's equivalent launcher, such as `srun`, when required.
The plugin must be compiled with MPI support; see the official
[Lightning-Kokkos MPI documentation](https://docs.pennylane.ai/projects/lightning/en/stable/lightning_kokkos/device.html).

## Manual circuit construction example

QASP can also be used without the policy classes. You can directly combine a feature map, an ansatz template, and `CircuitBuilder` to create a differentiable torch-compatible quantum layer.

The example below follows the same style used in `tests/test_qml_models.ipynb`.

<!-- readme-circuit: manual -->

```python
from qasp import AngleEmbedding, AngleAnsatz, CircuitBuilder
import pennylane as qml
from pennylane import numpy as np
import torch

# 1. Define a feature map
fm = AngleEmbedding(
    features=4,
    n_qubits=4,
    rotations="YZ",
    input_scaling=True,
)

# 2. Define an ansatz that reuses the feature map
ansatz = AngleAnsatz(
    n_qubits=4,
    n_layers=3,
    rotations="ZY",
    feature_map=fm,
    include_imprimitive=True,
    imprimitive="CZ",
    imprimitive_pattern="all_to_all",
    reuploading=True,
)

# 3. Inspect required parameter shapes
shapes = ansatz.get_param_shapes()
print("Parameter shapes:", shapes)

# Example initialization
input_scaling_params = np.random.uniform(-1.0, 1.0, size=shapes[0])
params = np.random.uniform(-np.pi, np.pi, size=shapes[1])

# 4. Build a circuit with an expectation-value readout
builder = CircuitBuilder(n_qubits=4)
builder.set_ansatz(ansatz)
builder.set_measurement(
    "expectation",
    wires=[0, 1, 2, 3],
    observables=["Z 0", "Z 1", "ZZ 0 1", "ZZ 2 3"],
)

circuit = builder.build()
qnode = builder.compile(device="lightning.qubit", diff_method="adjoint")

# 5. Create a torch layer
qlayer = builder.create_torchlayer(
    params=torch.tensor(params, requires_grad=True),
    input_scaling_params=torch.tensor(input_scaling_params, requires_grad=True),
)

# 6. Evaluate the layer on sample input
x = torch.tensor([0.1, -0.2, 0.3, 0.5], dtype=torch.float32)
out = qlayer(x)
print("Quantum layer output:", out)
```

Export this same builder, passing its input and both parameter arrays. The
first layer includes the feature map; subsequent blocks include reuploading
and are labelled as data-dependent unitaries.

<!-- readme-circuit: manual -->

```python
state = qlayer.state_dict()
builder.save_tikz(
    "manual_circuit.tex",
    inputs=x, params=state["params"],
    input_scaling_params=state["input_scaling_params"],
    standalone=True, symbolic_parameters=True, compress=True,
)
```

[![Four-qubit custom circuit with YZ input encoding, ZY variational rotations, all-to-all CZ entanglers, and three reuploading layers](docs/assets/circuits/manual_circuit.png)](docs/assets/circuits/manual_circuit.pdf)

[Editable TikZ](docs/assets/circuits/manual_circuit.tex) · [Vector PDF](docs/assets/circuits/manual_circuit.pdf)

This lower-level workflow is useful when:

- designing a custom model architecture,
- testing a new feature map or ansatz,
- studying trainability or Fourier structure without committing to a specific RL algorithm,
- integrating a QASP circuit into a larger PyTorch model.

---

## High level RL policies from dictionaries and `create_model`

This is the main high-level workflow for reinforcement learning with QASP.

<!-- readme-circuit: neural_policy -->

```python
from qasp import create_model, draw_model
import torch

n_qubits = 6

fm_config = {
    "features": 4,
    "rotations": "Z",
}

ansatz_config = {
    "n_layers": 3,
    "rotations": "ZY",
    "reuploading": False,
    "include_imprimitive": True,
    "imprimitive": "CZ",
    "imprimitive_pattern": "all_to_all",
    "include_final_imprimitive": False,
    "include_feature_map_first": False,
    "include_hadamards_initial": True,
}

observables = [f"Z {i}" for i in range(n_qubits)]
observables += [f"ZZ {i} {i+1}" for i in range(n_qubits - 1)]

policy_model_config = {
    "ansatz": "AngleAnsatz",
    "ansatz_config": ansatz_config,
    "feature_map": "NeuralEmbedding",
    "feature_map_config": fm_config,
    "n_actions": 2,
    "init": "normal_0_1",
    "device": "lightning.qubit",
    "diff_method": "adjoint",
    "fc_layer": True,
    "observables": observables,
}

policy = create_model("PQCSoftmax", policy_model_config, n_qubits=n_qubits)
```

To visualize the resulting circuit:

```python
draw_model(
    model=policy,
    policy_class="PQCSoftmax",
    n_qubits=n_qubits,
    num_inputs=4,
    inputs=torch.zeros(4),
)
```

Save a paper-ready figure of this six-qubit neural-embedding policy.
This configuration places its feature map after the first variational layer,
so the export shows all three layers without automatic compression. The
classical neural feature map and final affine policy head are not quantum gates.
Open the vector PDF to inspect the individual gates at full size.

<!-- readme-circuit: neural_policy -->

```python
from pathlib import Path

tikz = draw_model(
    model=policy, inputs=torch.zeros(4), output="tikz",
    tikz_options={"standalone": True, "symbolic_parameters": True},
)
Path("neural_policy.tex").write_text(tikz, encoding="utf-8")
```

[![Six-qubit softmax policy with neural Z encoding, three ZY layers, all-to-all CZ entanglers, and Z and ZZ readouts](docs/assets/circuits/neural_policy.png)](docs/assets/circuits/neural_policy.pdf)

[Editable TikZ](docs/assets/circuits/neural_policy.tex) · [Vector PDF](docs/assets/circuits/neural_policy.pdf)

### Exporting publication-ready TikZ circuits

QASP can export any `CircuitBuilder`, PennyLane QNode, quantum function, or
quantum tape as editable vector-quality LaTeX using the `quantikz` package:

```python
from qasp import circuit_to_tikz, save_tikz

# `params` and `inputs` are the same values used to execute the circuit.
tikz = circuit_to_tikz(
    policy.circuit,
    inputs=torch.zeros(4),
    params=policy.qlayer.state_dict()["params"],
    symbolic_parameters=True,
)
print(tikz)

# A complete, directly compilable .tex document:
save_tikz(
    "policy_circuit.tex",
    policy.circuit,
    inputs=torch.zeros(4),
    params=policy.qlayer.state_dict()["params"],
    standalone=True,
)
```

For a builder, the same operations are also available as methods:

```python
tikz = builder.to_tikz(params=params, symbolic_parameters=True)
builder.save_tikz("circuit.tex", params=params, standalone=True)
```

By default, the result is a snippet ready to paste into a paper or Beamer
presentation (add `\\usepackage{xcolor}` and `\\usepackage{quantikz}` to the preamble). Set
`standalone=True` for a complete document. Use `wire_labels`,
`parameter_labels`, or `parameter_formatter` to match the notation of a paper.
An `AngleEmbedding` block is expanded into its individual rotations. With
symbolic parameters, encoded angles are labelled `f_i`, or
`f_i\\lambda_i` when input scaling is enabled; variational angles retain
the independent `\\theta_i` sequence.
The default pastel theme uses light blue for single-qubit gates, purple for
two-qubit gates, pale violet for larger gates, and lavender for measurements.
Override any color with an HTML hex value or an existing `xcolor` expression:

```python
tikz = builder.to_tikz(
    params=params,
    gate_colors={
        "single": "DCECF8",
        "two": "E4D8F2",
        "multi": "violet!10",
        "measurement": "blue!8",
    },
)
```

Set `gate_colors=False` to generate an unfilled monochrome circuit.

For deep repeated circuits, set `compress=True`. QASP keeps the data-encoding
prefix visible, draws every gate in the first variational layer inside a dashed
`U(\\theta_1)` enclosure, and replaces later layers with register-wide unitary
blocks:

```python
tikz = draw_model(
    model=policy,
    inputs=torch.zeros(4),
    output="tikz",
    tikz_options={
        "standalone": True,
        "symbolic_parameters": True,
        "compress": True,
        "max_explicit_layers": 4,
    },
)
```

For `n_layers <= max_explicit_layers`, every later block from
`U(\\theta_2)` through `U(\\theta_L)` is displayed. Larger circuits show
`U(\\theta_2)`, an ellipsis, and `U(\\theta_L)`. Layer boundaries are inferred
automatically for `AngleAnsatz` and `IQPAnsatz`. With data reuploading, each
compressed block includes its feature map and is labelled as data-dependent;
the encoding must precede the first layer. Input-scaling symbols include the
layer index. Mirrored overlap compression still requires `reuploading=False`.

The same exporter is available through `draw_kernel`. Fidelity and local
fidelity kernels use both samples; projected kernels only need `x`:

```python
from qasp import draw_kernel

tikz = draw_kernel(
    kernel=kernel,
    x=X_train[0],
    y=X_train[1],  # omit for a projected kernel
    output="tikz",
    tikz_options={
        "standalone": True,
        "symbolic_parameters": True,
        "compress": True,
        "max_explicit_layers": 4,
    },
)
```

Compression applies to trainable kernels built with a repeated
`AngleAnsatz` or `IQPAnsatz` and `reuploading=False`. Feature-map-only
kernels can still be exported to TikZ, but have no variational layers to
compress. For fidelity and local-fidelity circuits, QASP also draws the
matching adjoint blocks in reverse order, from
`U^\\dagger(\\theta_L)` through `U^\\dagger(\\theta_1)`.

For runnable drawing-only examples covering IQP and AngleAnsatz policies,
fidelity, local-fidelity, and projected kernels, and a directly assembled
CircuitBuilder, see the [circuit-drawing notebook](examples/drawing_quantum_circuits.ipynb).
Each module example above and the CartPole examples below include their
own drawing, editable LaTeX, and vector PDF. Export calls can run immediately
after model construction; training is not required. Symbols describe the
circuit structure rather than a fitted parameter value.
Readout meters are schematic: tensor-product observables use local basis
meters, and overlapping observables share wires. The observable lists in the
code specify the complete set of expectations, including correlations and
measurements in different bases.

QASP returns LaTeX source. With a TeX distribution containing `standalone`,
`xcolor`, and `quantikz`, compile any saved figure (replace the filename with
the one from your example):

```bash
pdflatex -interaction=nonstopmode -halt-on-error born_machine.tex
```

For inclusion in an existing paper, set `standalone=False` and add
`\usepackage{xcolor}` and `\usepackage{quantikz}` to its preamble, or use
`\includegraphics{born_machine.pdf}` with the `graphicx` package.

To regenerate every README figure from the marked construction/export code
blocks, without running the training examples:

```bash
python examples/render_readme_circuits.py
```

This uses QASP's installed dependencies, `pdflatex`, and Poppler's `pdftoppm`.
The editable sources, vector PDFs, and PNG previews are written to
[`docs/assets/circuits/`](docs/assets/circuits/). Use `--source-only` to export
LaTeX without a TeX installation. The generator executes only the blocks
marked `readme-circuit` in this README; blocks sharing a name share variables.

---

## Reinforcement learning example with REINFORCE

QASP includes a built-in `REINFORCE` trainer for policy-gradient experiments.

### CartPole with a softmax PQC policy

```python
from qasp import *

agent = REINFORCE(
    policy=policy,
    env_name="CartPole-v1",
    gamma=0.99,
    lr=1e-2,
    batch_episodes=16,
    logits=True,  # PQCSoftmax.forward returns action logits.
    normalize_mode="clip",
    baseline_type="ema",
    normalize_returns=True,
    entropy_coef=0.01,
    max_grad_norm=1.0,
    fc_layer=True,
)

returns = agent.train(num_episodes=500)
```

### Typical REINFORCE controls

The trainer supports several knobs that are useful in experiments:

- `baseline_type="ema"` or `baseline_type="value"`
- `normalize_mode` for state preprocessing
- `entropy_coef` for exploration pressure
- `batch_episodes` for lower-variance policy-gradient updates
- `max_grad_norm` for gradient clipping
- `fc_layer=True` for policies that use a classical readout head on top of quantum expectations

This makes it straightforward to compare PQC policies across environments such as CartPole, Acrobot, and contextual bandits.

---

## Policy classes in `qasp.rl.policies`

QASP currently exposes several policy-model families.

### `PQCBorn`

The direct class is a legacy sampler whose `forward` returns an action and log
probability. The factory selector `create_model("PQCBorn", ...)` instead constructs
`PQCBornVariance`, the current probability-returning decoder model.

### `PQCSoftmax`

An expectation-score model with an optional biased affine head. `forward()`
returns logits; use a logits-aware trainer or `Categorical(logits=...)`. The legacy
`action_probs()` shortcut bypasses the head and PP path. `fc_layer=None` disables
the head; any non-None value currently creates one.

### `PQCBornVariance`

The current Born-policy implementation, with contiguous, prefix-parity, hidden
and majority-modulo decoders. It supports compatible probability, expectation
reconstruction and majority sparse-projector paths; backend restrictions are
documented on the constructor.

### `LearnablePQCBorn`

A historical adaptive-measurement prototype. Its setup and postprocessing have
known limitations; it is not a validated replacement for the fixed-decoder APIs.

---

## Trainability and analysis utilities

The lazy top-level `barren_plateau_diagnoser` works with a constructed QASP
classifier, trainable kernel, or RL policy. It repeatedly samples the selected
trainable parameters and reports both the unbiased finite-sample variance of
one scalar cost and the variance of every raw quantum expectation output,
together with their running variance after every sample:

```python
import torch
from qasp import barren_plateau_diagnoser, target_alignment

diagnosis = barren_plateau_diagnoser(
    classifier,
    n_samples=2000,
    initialization="random_-pi_pi",
    state=torch.tensor(X_train[0]),  # fixed input: variance over parameters
    targets=torch.tensor(y_train[0]),
    objective="mean_squared_error",
    seed=7,
    return_samples=True,
)

print(diagnosis["variance"])
print(diagnosis["cost_variance"])
print(diagnosis["expectation_value_variance"])
print(diagnosis["expectation_variance"])
cost_variance_curve = diagnosis["cost_variance_by_sample"]
expectation_variance_curve = diagnosis["expectation_value_variance_by_sample"]
```

The actual system-size scaling experiment uses a model factory and a qubit
list. Each width receives its own independent ensemble of `n_samples`
initializations:

```python
qubit_list = [2, 4, 6, 8, 10, 12, 14, 16]

diagnosis = barren_plateau_diagnoser(
    model_factory=lambda n: create_classifier(classifier_config(n), n, 2),
    qubit_list=qubit_list,
    n_samples=2000,
    initialization="random_-pi_pi",
    state_factory=lambda n: torch.zeros(n),  # fixed input at each width
    seed=7,
)

# Arrays aligned one-to-one with qubit_list.
cost_variance_vs_qubits = diagnosis["cost_variance"]
expectation_variance_vs_qubits = diagnosis["expectation_value_variance"]
per_width_details = diagnosis["results_by_qubit"]
```

Here `classifier_config(n)` denotes a user factory for the same circuit family
at width `n`. The sweep also accepts kernel and policy factories. Raw-output
dimensionality can change with width, so componentwise values are stored as
`expectation_variance_by_qubit[n]`; scalar cost and mean-expectation variances
are returned as arrays that can be plotted directly against `qubit_list` on a
logarithmic variance axis.

`variance` remains a compatibility alias for `cost_variance`. Observable
circuits report the variance of each measured expectation before decoder,
scaling, softmax, or loss post-processing. For Born/probability circuits, QASP
uses the raw basis probabilities to reconstruct one local Pauli-Z expectation
per measured qubit. `expectation_value_variance` is the scalar variance of the
arithmetic mean of those raw outputs and therefore includes their covariance;
`expectation_variance` retains the componentwise variances.
`expectation_function(model, sampled_state)` can override this default when a
custom quantum statistic is required. With `return_samples=True`, the
corresponding raw values are available as `cost_samples`,
`expectation_value_samples`, and `expectation_samples`.

When `targets` are omitted, the classifier default is the class probability
at `output_index`; the policy default is the action probability at that index;
and the kernel default is `k(x, y)` for a supplied pair `state=(x, y)`. For an
actual task objective, pass `cost_function(model, sampled_state)`. For example,
a trainable kernel can diagnose negative target alignment on a fixed dataset:

```python
diagnosis = barren_plateau_diagnoser(
    kernel,
    n_samples=500,
    state=X_train,
    cost_function=lambda model, X: -target_alignment(model.gram(X), y_train),
    initialization="small_normal",
    seed=11,
)
```

If `state=None`, QASP samples a fresh embedded input per initialization with
`state_initialization` (a pair for kernels). This estimates joint variance over
parameters and inputs. Supplying `state` instead isolates parameter variance
conditional on that fixed input. `state_sampler` supports custom per-sample
data generation, and `parameter_scope="all"` includes trainable classical
post-processing parameters rather than only the quantum `qlayer`. The original
model parameters and train/evaluation mode are restored even if a sample
fails.

These are initialization variances of the cost and of expectation values—not
the within-circuit quantum measurement variance
`<O^2> - <O>^2`. They are useful concentration signals, but the standard
barren-plateau claim concerns how gradient variance scales with system size;
a controlled cost-variance sweep alone does not establish that stronger
claim without a corresponding gradient analysis.

The legacy `qasp.rl.metrics` module provides probability-variance sampling and
heuristic Lie-closure utilities. `compute_DLA_generators` returns closure sizes;
the named gate-decomposition helpers supply heuristic generator lists rather
than executable unitary decompositions. `fourier_analysis` performs no Fourier
transform and has legacy import limitations. Use the [package guide](docs/package_guide.md)
to distinguish these prototypes from the current concentration diagnostics.

---

## Notebooks and experiment scripts

The repository includes examples and experiments in:

- `tests/test_qml_models.ipynb`
- `tests/test_feature_maps.ipynb`
- `tests/test_policy_training.ipynb`
- `tests/trainability_born_paper.ipynb`
- `tests/optimization_algorithms.ipynb`
- `tests/tests_deucalion/`

The notebooks are a good starting point for:

- inspecting circuit templates,
- checking parameter shapes,
- drawing models,
- reproducing RL experiments,
- running trainability benchmarks.
- testing QUBO optimization algorithms and custom circuit dictionaries.

---

### Quantum deep Q-learning

`qasp.rl.qfunctions` provides `PQDQN` and `create_qfunction` for dictionary-driven
quantum action-value networks, with shared or per-action Pauli strings and exact
parity/majority-modulo Pauli features with configurable `preset_actions`.
The fully connected layer includes a bias for presets and shared observables;
`n_actions` independently controls the final number of Q-values.
PQDQN also supports `device="lightning.pp"` for differentiable Pauli propagation,
including configurable truncation/pruning and the same DeepQLearning trainer.
`qasp.rl.torchrl.DeepQLearning` adds replay-based DQN/Double DQN with a frozen
target network and epsilon-greedy exploration.

`draw_qfunction(model, output="tikz", tikz_options=...)` and `model.draw(...)`
use the same Quantikz exporter as policies and kernels, including full/compact
views, symbolic parameters, pastel colors, and standalone or paper snippets.
The example notebooks compile vector PDFs and display PDF-rendered previews.

See [the Q-function guide](docs/qfunctions.md),
[observable validation notebook](examples/pqdqn_observables_validation.ipynb),
the [CartPole training notebook](examples/cartpole_pqdqn_deep_q_learning.ipynb),
and the [10-qubit PP/Lightning benchmark](examples/pqdqn_pauli_propagation_benchmark.ipynb).
The benchmark records inference and gradient timings alongside truncation errors.
Runtime improvements should be assessed separately from approximation quality
and reinforcement-learning performance.

## End-to-end CartPole examples

Each example below runs independently after installing QASP and its dependencies.
Both use four qubits for CartPole's four observations, two actions, and the
`default.qubit` simulator. Six short episodes make these examples quick checks
of construction and training; they do not establish convergence. Increase the
training budget and episode cap for a learning experiment.

### Policy gradients with softmax or Born policies

Choose `PQCSoftmax` to turn learned action logits into a categorical policy, or
`PQCBorn` to decode circuit probabilities into action probabilities. The
`create_model("PQCBorn", ...)` factory constructs the current
`PQCBornVariance` implementation. Set the trainer's `logits` flag to match the
selected model's output.

<!-- readme-circuit: cartpole_policy -->

```python
from qasp.rl import create_model, draw_model
from copy import deepcopy
import torch

torch.manual_seed(7)
policy_kind = "PQCSoftmax"  # Change to "PQCBorn" for a Born policy.
config = {
    "n_actions": 2,
    "ansatz": "AngleAnsatz",
    "ansatz_config": {
        "n_layers": 2,
        "rotations": "YZ",
        "reuploading": True,
        "include_imprimitive": True,
        "imprimitive": "CNOT",
        "imprimitive_pattern": "ring",
    },
    "feature_map": "AngleEmbedding",
    "feature_map_config": {
        "features": 4,
        "rotations": "Y",
        "input_scaling": True,
    },
    "device": "default.qubit",
    "diff_method": "backprop",
}
if policy_kind == "PQCSoftmax":
    config.update(observables=["Z 0", "Z 1", "Z 2", "Z 3"], fc_layer=True)
else:
    config.update(
        policy="parity-like",
        parity_cnot_decomposition=False,
        training_method="probs",
    )

# Copy nested dictionaries because the policy factory resolves them in place.
policy = create_model(policy_kind, deepcopy(config), n_qubits=4)
```

```python
from qasp.rl import REINFORCE

with torch.no_grad():
    output = policy(torch.zeros(4))
    probabilities = output.softmax(dim=-1) if policy_kind == "PQCSoftmax" else output
    action = torch.distributions.Categorical(probs=probabilities).sample().item()
print("Initial action probabilities:", probabilities.tolist(), "sampled action:", action)

agent = REINFORCE(
    policy=policy,
    env_name="CartPole-v1",
    device="cpu",
    logits=(policy_kind == "PQCSoftmax"),
    normalize_mode="tanh",
    lr=0.01,
    batch_episodes=2,
    max_steps=50,
    baseline_type="ema",
)
try:
    agent.env.reset(seed=7)  # Seed subsequent environment resets as well.
    returns = agent.train(num_episodes=6, log_every=2)
finally:
    agent.env.close()
print("Training episode returns:", returns)

```

Save the paper-ready circuit for the selected policy. The drawing below
shows the default softmax policy's four `Z` readouts before the classical head.

<!-- readme-circuit: cartpole_policy -->

```python
from pathlib import Path

policy_tex = draw_model(
    model=policy, inputs=torch.zeros(4), output="tikz",
    tikz_options={"standalone": True, "symbolic_parameters": True, "compress": True},
)
Path("cartpole_policy.tex").write_text(policy_tex, encoding="utf-8")
```

[![CartPole softmax policy with scaled Y inputs, YZ rotations, CNOT-ring entanglers, and a compact second reuploading layer](docs/assets/circuits/cartpole_policy.png)](docs/assets/circuits/cartpole_policy.pdf)

[Editable TikZ](docs/assets/circuits/cartpole_policy.tex) · [Vector PDF](docs/assets/circuits/cartpole_policy.pdf)

For comparison, build the Born-policy variant from the same ansatz and feature
map. Its computational-basis measurements feed the classical parity decoder.

<!-- readme-circuit: cartpole_policy -->

```python
born_config = deepcopy(config)
born_config.pop("observables", None)
born_config.pop("fc_layer", None)
born_config.update(
    policy="parity-like", parity_cnot_decomposition=False, training_method="probs",
)
born_policy = create_model("PQCBorn", born_config, n_qubits=4)
born_tex = draw_model(
    model=born_policy, inputs=torch.zeros(4), output="tikz",
    tikz_options={"standalone": True, "symbolic_parameters": True, "compress": True},
)
Path("cartpole_born.tex").write_text(born_tex, encoding="utf-8")
```

[![The same CartPole circuit with computational-basis measurements for Born parity decoding](docs/assets/circuits/cartpole_born.png)](docs/assets/circuits/cartpole_born.pdf)

[Editable TikZ](docs/assets/circuits/cartpole_born.tex) · [Vector PDF](docs/assets/circuits/cartpole_born.pdf)


`REINFORCE.train()` returns unshaped training-episode returns. Its
`normalize_mode` preprocesses observations before the policy call. The legacy
trainer freezes CartPole optimization when its trailing training mean reaches
195; use independent evaluation when assessing performance on CartPole-v1.

### Q-learning with PQDQN

`create_qfunction` constructs a `PQDQN`. This example measures four Pauli-Z
features and learns an affine map to two Q-values. `DeepQLearning` trains with
replay minibatches, a separate target network, and epsilon-greedy exploration;
Double DQN is enabled explicitly below. Greedy evaluation uses a separate
environment and performs no training updates.

<!-- readme-circuit: pqdqn -->

```python
from qasp.rl import create_qfunction, draw_qfunction
import torch

torch.manual_seed(7)
config = {
    "ansatz": "AngleAnsatz",
    "ansatz_config": {
        "n_layers": 2,
        "rotations": "YZ",
        "reuploading": True,
        "include_imprimitive": True,
        "imprimitive": "CNOT",
        "imprimitive_pattern": "ring",
    },
    "feature_map": "AngleEmbedding",
    "feature_map_config": {
        "features": 4,
        "rotations": "Y",
        "input_scaling": True,
    },
    "observables": ["Z 0", "Z 1", "Z 2", "Z 3"],
    "fc_layer": True,
    "fc_bias": True,
    "input_activation": "tanh",
    "device": "default.qubit",
    "diff_method": "backprop",
    "batch_mode": "broadcast",
}
qfunction = create_qfunction(config, n_qubits=4, n_actions=2)
```

```python
from qasp.rl import DeepQLearning

with DeepQLearning(
    qfunction,
    "CartPole-v1",
    lr=0.01,
    batch_size=8,
    learning_starts=16,
    train_frequency=4,
    target_update_interval=10,  # Measured in optimizer updates.
    double_dqn=True,
    epsilon_decay_steps=200,
    max_steps=50,
    seed=7,
) as trainer:
    history = trainer.train(num_episodes=6, log_every=2)
    evaluation_returns = trainer.evaluate(num_episodes=3, seed=10000)

print("Training episode returns:", [row["return"] for row in history])
print("Mean greedy evaluation return:", sum(evaluation_returns) / len(evaluation_returns))
with torch.no_grad():
    q_values = qfunction(torch.zeros(4))
print("Q-values:", q_values.tolist(), "greedy action:", q_values.argmax().item())

```

Export this PQDQN's quantum feature circuit; the learned affine map from
four measured expectations to two Q-values is classical and is not drawn.

<!-- readme-circuit: pqdqn -->

```python
from pathlib import Path

qfunction_tex = draw_qfunction(
    qfunction, inputs=torch.zeros(4), output="tikz",
    tikz_options={"standalone": True, "symbolic_parameters": True, "compress": True},
)
Path("pqdqn.tex").write_text(qfunction_tex, encoding="utf-8")
```

[![Four-qubit PQDQN with scaled Y encoding, two reuploading YZ layers, CNOT-ring entanglers, and Pauli-Z feature readouts](docs/assets/circuits/pqdqn.png)](docs/assets/circuits/pqdqn.pdf)

[Editable TikZ](docs/assets/circuits/pqdqn.tex) · [Vector PDF](docs/assets/circuits/pqdqn.pdf)


The affine head produces unconstrained action values, so greedy action
selection uses `argmax`. To use a predefined readout, replace `observables`
with `"parity-like"` or `"majority-modulo"` and set `preset_actions=2` in the
model dictionary. These presets supply Pauli features to the same learned head.

For longer experiments, saved results, and additional readout choices, see the
[CartPole PQDQN notebook](examples/cartpole_pqdqn_deep_q_learning.ipynb),
[Q-function guide](docs/qfunctions.md), and
[policy training notebook](tests/test_policy_training.ipynb).
