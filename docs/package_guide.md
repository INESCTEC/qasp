# QASP package guide

This guide maps the installed `qasp` package to its source APIs, data contracts,
and examples. Source docstrings document implementation details, including
private helpers. The descriptions below distinguish working application paths
from retained prototypes. Documentation coverage is not a claim that every
backend or historical branch is validated.

## Architecture and conventions

The main construction path is:

```text
feature-map configuration + ansatz configuration
    → gate-queueing template
    → CircuitBuilder: measurements → quantum function → backend/QNode
    → trainable TorchLayer
    → application readout (classes, action probabilities, Q-values or kernel)
    → application loss and optimizer
```

Optimization also builds problem-specific QAOA circuits directly. Shared
`qasp.devices.create_device` propagates backend/MPI options without silently
falling back to a serial simulator. Lazy top-level exports defer some imports;
they do not remove optional runtime dependencies.

- Circuit angles are radians. Template weights are flat tensors; ask
  `get_param_shapes()` rather than guessing sizes from qubit count.
- Feature-map **axis strings**, such as `"YZ"`, apply axes across wires;
  **lists of specifications**, such as `["Y 0", "ZZ 0 1"]`, describe individual
  rotations. These forms can have different shape and batching requirements.
- A Pauli specification pairs a word with wires: `"XZ 0 2"` means
  (X_0Z_2), and `"-Z 1"` includes a minus sign. Q-function parsing additionally
  accepts full-register words such as `"XIZ"`. `"I 0"` has constant expectation
  one; it supplies no state-dependent information.
- `shots=None` normally means analytic execution. A sampling readout needs
  finite shots. Supported measurements/differentiation methods depend on the
  installed device, not just the name of a QASP model.
- A Torch tensor device (CPU/CUDA) and a PennyLane simulator backend are distinct.
  Move a model to its intended dtype/device before creating its optimizer.
- Factories differ in copying behavior: classifier, kernel and Q-function
  dictionaries are copied; the legacy policy factory shallow-copies and mutates
  nested component dictionaries. Deep-copy policy configs before reusing them.
- Model construction does not train it. Sampling/measurement, analytic
  expectation evaluation and classical decoding are separate operations.

## Circuits and feature maps

Sources: [constructor](../qasp/circuits/constructor.py),
[ansätze](../qasp/circuits/templates.py),
[feature maps](../qasp/feature_maps/templates.py).

`AngleEmbedding` encodes features as rotation angles, optionally multiplied by
trainable scales. `NeuralEmbedding` uses learned linear projections of a feature
vector to rotation angles, with no projection bias. Its current reduction is
single-sample; use sequential evaluation for it and explicit-word encodings.

`AngleAnsatz` provides axis/Pauli-word rotations with optional fixed CZ/CNOT
entanglers on chain, ring, all-to-all or explicit edges. For `"YZ"` on (n)
qubits, each layer has (2n) variational angles. Fixed entanglers add no angles.
`IQPAnsatz` queues initial Hadamards, optional local RZ gates, trainable ZZ edges
and layer Hadamards. Both can encode initially and reupload between layers.
Input-scale storage is allocated per layer even when a configuration does not
use every allocated scale.

`CircuitBuilder.build()` returns a **plain quantum function**. After configuring
and building, `compile()` returns the QNode, and `create_torchlayer()` allocates
trainable weights. Attach encoding to the ansatz: `set_feature_map()` alone only
stores builder metadata. Valid measurement names include `"variance"`, not
`"var"`. Explicit backend selection is preferable to `"auto"` when a device
plugin is not installed. Auto does not cover state/variance selection.

```python
from qasp.circuits import CircuitBuilder, AngleAnsatz
from qasp.feature_maps import AngleEmbedding
import torch

fm = AngleEmbedding(n_qubits=2, features=2, rotations="Y", input_scaling=True)
ansatz = AngleAnsatz(
    n_qubits=2, n_layers=2, rotations="YZ", feature_map=fm,
    reuploading=True, include_imprimitive=True,
    imprimitive="CNOT", imprimitive_pattern="chain",
)
builder = CircuitBuilder(n_qubits=2)
builder.set_ansatz(ansatz)
builder.set_measurement("expectation", observables=["Z 0", "Z 1"])
builder.build()
builder.compile(device="default.qubit", diff_method="backprop")
layer = builder.create_torchlayer(initialization="small_normal")
features = layer(torch.tensor([0.1, -0.2])).reshape(-1)
assert features.shape == (2,)
features.sum().backward()
```

The adaptive-measurement and circuit-cutting adapters are experimental, with
more restrictive call signatures. `circuits.uqc.UQC` is a retained historical
function with missing imports and an old broadcast API, not a supported template.
The initializer named `discrete_0_pi_over_2` in the builder currently samples
0 or **pi**; its legacy name should not be used as a mathematical specification.
Prefer the shared objects in `qasp.initialization`: for example,
`initialization=Normal(0, 0.2), input_scaling_init=Uniform(0.8, 1.2)` when
creating a Torch layer. Model dictionaries use `init` for the first setting.
`Choice((0, np.pi / 2))` specifies discrete values without legacy ambiguity.
See the [initialization API](../README.md#parameter-initialization) and
[executed cross-module notebook](../examples/initialization_across_modules.ipynb)
for defaults, reproducibility, explicit arrays and optimization models.

## Drawing

Source: [TikZ exporter](../qasp/circuits/tikz.py). Application adapters:
`draw_model`, `draw_classifier`, `draw_kernel`, `draw_qfunction`,
`draw_optimization`, and model `.draw()` methods where exposed.

Use `output="tikz"` to obtain Quantikz/TikZ **source**. `standalone=True` creates a
complete LaTeX document; compilation to PDF requires a suitable TeX installation
with Quantikz. `compress=True` draws a complete first layer and represents later
layers schematically, with supported reuploading boundaries. Compression does
not change simulation. Numeric labels describe the supplied weights/input;
symbolic labels show the parameter structure.

```python
from qasp.circuits import circuit_to_tikz

tex = circuit_to_tikz(
    builder,
    circuit_kwargs={"inputs": torch.zeros(2), **layer.state_dict()},
    standalone=True,
    symbolic_parameters=True,
)
assert "quantikz" in tex
```

Meters display quantum measurements. They do not draw a Q-function's affine
head or encode all classical product/reconstruction coefficients. Explain those
in the caption/equation or accompanying readout metadata.
See [drawing_quantum_circuits.ipynb](../examples/drawing_quantum_circuits.ipynb).

## Classification

Sources: [models](../qasp/classifiers/models.py),
[factory](../qasp/classifiers/factory.py), [decoders](../qasp/classifiers/decoders.py),
[losses](../qasp/classifiers/losses.py), [metrics](../qasp/classifiers/metrics.py),
[training](../qasp/classifiers/training.py).

`PQCClassifier` has two readout families. Probability readout groups Born
probabilities using contiguous, prefix-parity or majority decoding. Expectation
readout uses one signed binary score, or one score per multiclass label;
observable **groups are averaged**, unlike Q-function per-action sums.
`predict_proba()` applies the appropriate sigmoid/softmax to expectation scores.
`compute_loss()` supports MSE, hinge, BCE and categorical cross-entropy.
`ClassifierTrainer` performs minibatch Torch updates and reports loss and
classification metrics. Targets are integer class labels.

```python
from qasp.classifiers import create_classifier, ClassifierTrainer

classifier_config = {
    "ansatz": "AngleAnsatz",
    "ansatz_config": {"n_layers": 1, "rotations": "YZ"},
    "feature_map": "AngleEmbedding",
    "feature_map_config": {"features": 2, "rotations": "Y"},
    "readout": "probabilities",
    "decoder": "contiguous",
    "device": "default.qubit",
    "diff_method": "backprop",
}
classifier = create_classifier(classifier_config, n_qubits=2, n_classes=2)
X = torch.tensor([[0.1, 0.2], [-0.2, 0.3], [0.5, -0.1], [-0.3, -0.4]])
y = torch.tensor([0, 1, 0, 1])
classifier_trainer = ClassifierTrainer(classifier)
classifier_history = classifier_trainer.fit(X, y, epochs=1, batch_size=2)
assert classifier.predict_proba(X).shape == (4, 2)
```

Confusion matrices use true labels in rows, predictions in columns. Macro metrics
include configured absent classes with zero recall. Brier score sums squared
class errors before averaging examples. PR AUC uses stepwise average precision;
ROC/PR AUC without both binary classes is undefined (NaN), not zero. These are
metrics, not independent significance tests.
See [classifier validation](../examples/quantum_classifiers_sklearn_validation.ipynb).

## Quantum kernels

Sources: [kernel models](../qasp/qkernels/quantum_kernels.py),
[factory](../qasp/qkernels/factory.py), [alignment](../qasp/qkernels/kta.py),
[legacy Fisher kernel](../qasp/qkernels/kernels.py).

`FidelityKernel` measures the global all-zero projector on the overlap circuit
by default. Explicit projector labels such as `"P 0 2"` define localized
zero-state probabilities; multiple projectors are averaged. They must not be
reported as global fidelity unless the measurement actually covers every wire.
`ProjectedKernel` forms features (f_j(x)=\langle O_j\rangle_x) and computes
\(\exp[-\gamma\sum_j w_j(f_j(x)-f_j(y))^2]\).

Without an ansatz, `EncodingAnsatz` repeats only the feature map. Supplying an
ansatz creates trainable quantum weights. `gram(X, return_tensor=True)` retains
the graph for those weights; ordinary NumPy-returning convenience methods detach
it. Input conversion in these APIs does not promise gradients with respect to
user inputs. `KernelTargetAlignment` maximizes normalized Frobenius alignment
with the same-label target matrix, optionally double-centered. This is training
against labels, so validation must use held-out data.

```python
import numpy as np
from qasp.qkernels import create_kernel

kernel = create_kernel({
    "kernel_type": "fidelity",
    "feature_map": "AngleEmbedding",
    "feature_map_config": {"features": 2, "rotations": "Y"},
    "device": "default.qubit",
}, n_qubits=2)
K = kernel.gram(np.array([[0.1, 0.2], [0.3, -0.2]]))
assert K.shape == (2, 2)
```

The legacy `QFKernel` uses probability-score vectors and a regularized Fisher
inverse. It enumerates measured outcomes, can divide by zero probabilities, and
its `normalize` option is currently unused. These restrictions differ from the
modern fidelity/projected API. See [kernel validation](../examples/iqp_quantum_kernels_validation.ipynb).

## Optimization

Sources: [QUBO](../qasp/optimization/problem.py), [QAOA](../qasp/optimization/qaoa.py),
[recursive QAOA](../qasp/optimization/recursive.py), [PCE](../qasp/optimization/pce.py),
[optimizers](../qasp/optimization/optimizers.py), [results](../qasp/optimization/results.py),
[drawing](../qasp/optimization/drawing.py).

`QUBO` represents \(c+x^TQx\) with binary variables; an off-diagonal pair has
coefficient (Q_{ij}+Q_{ji}), without an implicit factor of one half. Its Ising
conversion uses (x_i=(1-Z_i)/2). `QUBO.maxcut` minimizes the **negative** cut.
`QAOA` supports shared or multi-angle parameters, standard/custom mixers and
warm starts. `WarmStartQAOA` enables the relaxed initialization path.
`LRQAOA` uses a deterministic angle ramp rather than a classical training loop.
`RecursiveQAOA` uses measured correlations to eliminate variables, solves a small
remaining problem exactly and reconstructs the original bitstring.

`PauliCorrelationEncoding` maps binary variables to Pauli correlations, trains
a smooth surrogate on fewer qubits when the encoding permits, and decodes a
binary candidate. The optimized surrogate and candidate QUBO energy are
**different quantities**. Backend adapters support SciPy, PennyLane/Autograd and
Torch optimizers with different callable/gradient contracts.

```python
from qasp.optimization import QUBO

problem = QUBO.from_terms(2, linear={0: -1, 1: -1}, quadratic={(0, 1): 2})
assert problem.energy("10") == -1.0
assert problem.energy("11") == 0.0
```

`OptimizationResult.energy` evaluates the selected binary candidate;
`objective` records the optimized quantum/surrogate objective. Probabilities
may be absent for counts-only or decoded results. Optimizer success does not
certify a globally optimal binary solution. Separate finite-shot **training**
from independent `result_shots` used to estimate the output distribution.
Full basis-probability arrays grow exponentially; counts-only output avoids
allocating that dense result but not the simulator state itself.
See [optimization notebook](../examples/optimization_algorithms.ipynb).

## RL policies, Q-functions and trainers

Sources: [policies](../qasp/rl/policies.py), [factory/drawing](../qasp/rl/model.py),
[Q-functions](../qasp/rl/qfunctions/models.py), [observables](../qasp/rl/qfunctions/observables.py),
[Q-function drawing](../qasp/rl/qfunctions/drawing.py), [trainers](../qasp/rl/torchrl.py),
[navigation environment](../qasp/rl/envs.py).

| Model/API | Returned quantity | Readout distinction |
| --- | --- | --- |
| `create_model("PQCBorn", ...)` | `PQCBornVariance` action probabilities | Fixed basis decoder; probability, supported moment reconstruction or majority sparse projectors |
| `PQCSoftmax.forward` | Action logits | Expectations, scaling and optional **biased** affine head; apply categorical softmax afterward |
| `PQDQN.forward` / `q_values` | Unconstrained Q-values | Observable features, optional affine head; no softmax |
| Direct legacy `PQCBorn.forward` | Sampled action and log probability | Historical interface, not the factory's current Born model |

`PQCSoftmax.action_probs()` is a legacy shortcut: it bypasses its classical head,
PP route and trainable scaling. Use its `forward()` with a logits-aware trainer.
Its head is disabled by `fc_layer=None`; even `False` currently creates a head.
PQDQN instead validates `fc_layer` as a boolean.

For PQDQN, \(z(s)\) is the vector of readout features and
\(Q(s)=Wz(s)+b\) when the head is enabled:

| Observable specification | Features before the head | Default head bias |
| --- | --- | --- |
| `["Z 0", "Z 1"]` | Two independent expectations | Yes |
| `[["I 0", "Z 0"], ["I 0", "-Z 0"]]` | One sum per action; exactly one nested group per action | No |
| `"parity"` / `"majority"`, `preset_mode="features"` | Distinct nonidentity Pauli terms from decoder-projector support, each separately trainable through the head | Yes |

`preset_actions=a` selects the **decomposition**, independently of output
`n_actions`. For four qubits, parity `a=2` gives (Z_0Z_1Z_2Z_3); parity `a=4`
gives (Z_0Z_1Z_2Z_3,Z_0Z_1Z_2,Z_3). Majority uses strict block majority (ties
zero); features are the nonidentity support terms, without fixed majority
coefficients applied to the final Q-function. Historical `preset_mode="projectors"`
reconstructs fixed probabilities and must be identified separately.
A flat shared list can map to any supported action count through the head; a
single nested group cannot bypass the one-group-per-action validation.

```python
from qasp.rl.qfunctions import create_qfunction

q_config = {
    "ansatz": "AngleAnsatz",
    "ansatz_config": {"n_layers": 2, "rotations": "YZ"},
    "feature_map": "AngleEmbedding",
    "feature_map_config": {"features": 2, "rotations": "Y", "input_scaling": True},
    "observables": ["Z 0", "Z 1"],
    "fc_layer": True,
    "device": "default.qubit",
    "diff_method": "backprop",
}
qmodel = create_qfunction(q_config, n_qubits=2, n_actions=4)
qvalues = qmodel(torch.zeros(3, 2))
assert qvalues.shape == (3, 4)
assert qmodel.fc_layer.bias is not None
```

See [the detailed Q-function guide](qfunctions.md) for training, drawings,
readout metadata and the corrected experiments.

| Trainer | Method and interface | Important experiment contract |
| --- | --- | --- |
| `DeepQLearning` | Replay, target network, smooth-L1 TD loss; optional Double DQN | Target sync counts optimizer steps; epsilon counts environment frames; time-limit truncation bootstraps |
| `REINFORCE` | Episode batches with EMA/value baseline; configurable logits | CartPole optimization freezes at an available trailing up-to-100 mean of 195; this is not a CartPole-v1 solving certificate |
| `Reinforce` + `ReinforceConfig` | Separate Monte Carlo implementation with online normalization | Different configuration, learning-rate groups and stopping rules; not an alias of `REINFORCE` |
| `PPO` | Clipped likelihood ratios and GAE with separate critic | Resets each rollout; both termination and truncation mask GAE; environment time limit applies |
| `ContextualBanditTrainer` | Advantage-weighted regression from one-step replay | Requires a probability-compatible policy interface |
| `TorchRLTrainer` | Adapter to external TorchRL actors/losses | Optional dependency/API compatibility; reporting averages are training returns |

`DeepQLearning` owns environments created by name and closes them through its
context manager. Caller-supplied environment instances remain caller-owned;
evaluation must use a separate instance. Its seed does not initialize model
weights: seed Torch before constructing the model. No automatic checkpointing
is provided by this trainer; example runners implement their own persistence.

`WindyTargetNavigationEnv` exposes six normalized features and three/four thrust
actions, with inertial motion, wind and successive waypoints.
`ContextualBanditEnv` supplies one-step contexts and dense/Bernoulli rewards;
its nominal oracle/bounds assume nonnegative reward scaling and the normalized
oracle omits the step formula's epsilon adjustment.

`LearnablePQCBorn` and `rl.trl.TorchRLTrainer` are historical prototypes with
known setup/postprocessing limitations. They should not be described as validated
adaptive-measurement or general PPO support. Legacy policy weight setters may
call a nonexistent QNode method; use Torch state dictionaries instead.

## Pauli propagation

Sources: [shared device](../qasp/simulators/pauli_propagation.py),
[differentiable engine](../qasp/simulators/_pauli_propagation.py),
[non-autograd engine](../qasp/rl/pp_sim.py).

Use `device="PauliPropagationSimulator"`, `shots=None` and
`diff_method="backprop"` for the shared analytic Torch device across RL,
classifiers, fidelity/projected kernels, generative learning and optimization
(`backend="torch"`). It supports noncommuting circuits, arbitrary Pauli rotations,
Pauli expectations/variances and bounded probability readouts. Exact execution
is the default. See the [shared-device guide](pauli_propagation.md) and
[application example](../examples/pauli_propagation_modules.py).

`lightning.pp` remains a legacy QASP execution route, **not a PennyLane plugin name**.
Its model-level `pp_*` controls below are preserved; the new device uses
`device_kwargs` and different, exact defaults.
The differentiable engine propagates observables backward through supported
operations and contracts them with the all-zero state. Coefficients remain Torch
scalars connected to gate/input-scaling parameters. PQDQN uses live parameters
when tracing each sample; PP currently loops over scalar tapes even when the
model's batch mode says broadcast.

The approximation controls are distinct:

- `pp_truncation=k`: discard Pauli words above weight (k).
- `pp_tau`: remove coefficients below a magnitude threshold.
- `pp_topM`: retain at most the largest (M) terms.
- `pp_prune_every`: coefficient-pruning interval; zero means final-only pruning.
- `pp_prune=False`: disable coefficient pruning, **not** Pauli-weight truncation.

Masks are chosen using detached magnitudes. Gradients propagate through kept
coefficients, with no derivative through mask selection and no straight-through
mask estimator. Setting no weight cutoff and disabling coefficient pruning
provides an untruncated reference for the supported gate set. Truncation changes
both predictions and gradients; timing or differentiability alone does not
establish learning or an approximation-error bound.

The non-autograd `FastTruncatedPauliProp` uses Python complex coefficients and
must not be substituted for differentiable training. See
[PP benchmark](https://github.com/andre-sequeira10/QASP/blob/3bda8622bd6f5ddfb7c05a0dfd0538c1b0ccd718/examples/pqdqn_pauli_propagation_benchmark.ipynb).
The subsequent one-seed 400-episode CartPole comparison was **cancelled**; its
partial data are not a completed learning study and must not be restarted by
a documentation or paper update.

## Diagnostics

Source: [barren_plateau_diagnoser](../qasp/diagnostics/barren_plateau.py).

The diagnoser estimates finite-sample **cost/output concentration** under
parameter initialization, for one model or a factory over qubit widths. It
reports scalar cost variance, variance of the mean raw expectation and
componentwise expectation variances; those are different statistics. It does
not calculate a gradient-variance certificate. Fixed inputs isolate parameter
variation; sampled inputs mix parameter and input variability. Default parameter
scope is the quantum layer; `"all"` also includes trainable classical parameters.
Parameters/mode are restored after the diagnostic.

A PQDQN is classified as a generic model: supply a scalar
`cost_function(model, state)`, such as a selected Q-value or actual TD loss.
Do not label cost concentration alone proof of a barren plateau or inability
to learn. See [diagnostic validation](https://github.com/andre-sequeira10/QASP/blob/3bda8622bd6f5ddfb7c05a0dfd0538c1b0ccd718/examples/barren_plateau_diagnoser_validation.ipynb).

[Legacy RL metrics](../qasp/rl/metrics.py) retain `policy_variance` and heuristic
Lie-closure utilities. `compute_DLA_generators` returns closure **sizes**, not
generators. `fourier_analysis` performs no Fourier transform and has legacy
import limitations. The named gate-decomposition helpers are heuristic generator
lists, not executable unitary decompositions.

## Generative learning

Sources: [PQCBM](../qasp/genlearn/models.py),
[factory](../qasp/genlearn/factory.py), [drawing](../qasp/genlearn/drawing.py),
[losses](../qasp/genlearn/losses.py). See the [complete genlearn guide](genlearn.md)
and [experiment notebook](../examples/pqcbm_generative_learning.ipynb).

`PQCBM` and `create_qcbm` accept the package's ansatz/feature-map dictionaries,
a circuit builder, or an existing probability QNode. NLL learns from full
probabilities; Gaussian MMD learns a mixture of Pauli-Z expectations, using
sampled masks or exact small-system enumeration. Torch and Autograd training
are supported; analytic Autograd also supports the Fisher update. Fixed
conditioning inputs and trainable input scales are optional.

`lightning.pp` supports differentiable analytic MMD with explicit pruning and
truncation controls. Other backends use the shared device configuration,
including MPI-enabled statevectors. `sample()` executes a sampling device and
returns bit rows; PP models require an explicit sampling backend. `draw_qcbm`
exports Matplotlib, text, and Quantikz. `BornMachine` remains a deprecated
compatibility wrapper
with Autograd and integer-sample defaults; its old default-loss and
natural-gradient failures have been fixed.

## Documentation maintenance

Source contracts should state inputs and shapes, return values, mutation/resource
ownership, supported configurations and limitations. Keep executable examples
small; full learning experiments belong in notebooks/runners with saved metadata.
When modifying behavior, update the relevant source docstring and guide together.

[Documentation audit](https://github.com/andre-sequeira10/QASP/blob/3bda8622bd6f5ddfb7c05a0dfd0538c1b0ccd718/docs/documentation_audit.md) records coverage, syntax/code-change
checks, example execution and test limitations for this pass. The source inventory
below includes every Python module under `qasp`, including export-only modules;
notebook cells and external dependencies are outside its denominator.

## Complete source inventory

| Module | Role |
| --- | --- |
| [qasp/__init__.py](../qasp/__init__.py) | QASP: reusable quantum circuits and application-level learning models. |
| [qasp/circuits/__init__.py](../qasp/circuits/__init__.py) | Circuit templates, mutable circuit construction and Quantikz/TikZ rendering exports. |
| [qasp/circuits/constructor.py](../qasp/circuits/constructor.py) | Mutable construction of QASP quantum functions, QNodes and Torch layers. |
| [qasp/circuits/templates.py](../qasp/circuits/templates.py) | Gate-queueing IQP and axis/Pauli-word ansatz templates. |
| [qasp/circuits/tikz.py](../qasp/circuits/tikz.py) | Export PennyLane circuits as publication-ready Quantikz/TikZ source. |
| [qasp/circuits/uqc.py](../qasp/circuits/uqc.py) | Legacy illustrative universal-quantum-classifier circuit function. |
| [qasp/classifiers/__init__.py](../qasp/classifiers/__init__.py) | Quantum classifiers, objectives, training, decoders, and metrics. |
| [qasp/classifiers/decoders.py](../qasp/classifiers/decoders.py) | Class-probability decoders for Born-rule quantum classifiers. |
| [qasp/classifiers/factory.py](../qasp/classifiers/factory.py) | Configuration-driven construction for QASP quantum classifiers. |
| [qasp/classifiers/losses.py](../qasp/classifiers/losses.py) | Differentiable objectives for discrete quantum classifiers. |
| [qasp/classifiers/metrics.py](../qasp/classifiers/metrics.py) | Dependency-light evaluation metrics for QASP classifiers. |
| [qasp/classifiers/models.py](../qasp/classifiers/models.py) | Parameterized-quantum-circuit classifiers built from QASP primitives. |
| [qasp/classifiers/training.py](../qasp/classifiers/training.py) | Small PyTorch training loop for QASP classifiers. |
| [qasp/devices.py](../qasp/devices.py) | Shared device creation for QASP's circuit and algorithm modules. |
| [qasp/diagnostics/__init__.py](../qasp/diagnostics/__init__.py) | Model-agnostic trainability diagnostics. |
| [qasp/diagnostics/barren_plateau.py](../qasp/diagnostics/barren_plateau.py) | Finite-sample cost-concentration diagnostics for QASP models. |
| [qasp/feature_maps/__init__.py](../qasp/feature_maps/__init__.py) | Input-encoding templates: elementwise angle scaling and neural weighted rotations. |
| [qasp/feature_maps/templates.py](../qasp/feature_maps/templates.py) | Feature encoders that enqueue PennyLane rotations in an active quantum tape. |
| [qasp/genlearn/__init__.py](../qasp/genlearn/__init__.py) | Quantum circuit Born machines, factories, drawing and MMD training. |
| [qasp/genlearn/factory.py](../qasp/genlearn/factory.py) | Dictionary-based PQCBM construction. |
| [qasp/genlearn/drawing.py](../qasp/genlearn/drawing.py) | Matplotlib, text and Quantikz drawing of PQCBM circuits. |
| [qasp/genlearn/losses.py](../qasp/genlearn/losses.py) | Probability losses and Gaussian MMD as a mixture of Pauli-Z expectations. |
| [qasp/genlearn/models.py](../qasp/genlearn/models.py) | PQCBM training, sampling, feature maps and device integration. |
| [qasp/optimization/__init__.py](../qasp/optimization/__init__.py) | Combinatorial optimization with QASP circuits and optimizers. |
| [qasp/optimization/drawing.py](../qasp/optimization/drawing.py) | Circuit drawing helpers for QASP optimization algorithms. |
| [qasp/optimization/optimizers.py](../qasp/optimization/optimizers.py) | Optimizer adapters used by the optimization module. |
| [qasp/optimization/pce.py](../qasp/optimization/pce.py) | Pauli Correlation Encoding with QASP parameterized circuits. |
| [qasp/optimization/problem.py](../qasp/optimization/problem.py) | QUBO problem definitions and QUBO-to-Ising conversion utilities. |
| [qasp/optimization/qaoa.py](../qasp/optimization/qaoa.py) | QAOA, warm-start QAOA, multi-angle QAOA, and LR-QAOA. |
| [qasp/optimization/recursive.py](../qasp/optimization/recursive.py) | Recursive QAOA for reducing and exactly finishing QUBO problems. |
| [qasp/optimization/results.py](../qasp/optimization/results.py) | Result containers shared by QASP optimization algorithms. |
| [qasp/qkernels/__init__.py](../qasp/qkernels/__init__.py) | Quantum feature kernels, differentiable alignment losses and kernel training utilities. |
| [qasp/qkernels/factory.py](../qasp/qkernels/factory.py) | Dictionary factories and circuit drawing for fidelity/projected kernels. |
| [qasp/qkernels/kernels.py](../qasp/qkernels/kernels.py) | Legacy score/Fisher kernel over measured computational-basis bit strings. |
| [qasp/qkernels/kta.py](../qasp/qkernels/kta.py) | Centered kernel-target alignment and optimization of quantum kernel weights. |
| [qasp/qkernels/quantum_kernels.py](../qasp/qkernels/quantum_kernels.py) | Fidelity and projected quantum kernels built on QASP circuit components. |
| [qasp/rl/__init__.py](../qasp/rl/__init__.py) | Reinforcement-learning models, trainers, and Pauli simulators. |
| [qasp/rl/differentiable_pp_sim.py](../qasp/rl/differentiable_pp_sim.py) | Packed-Pauli Heisenberg propagation with Torch-autograd coefficients. |
| [qasp/rl/envs.py](../qasp/rl/envs.py) | Small self-contained Gymnasium environments for policy benchmarks. |
| [qasp/rl/metrics.py](../qasp/rl/metrics.py) | Legacy policy-output sampling and exploratory Lie-closure utilities. |
| [qasp/rl/model.py](../qasp/rl/model.py) | Dictionary-based RL model creation and circuit drawing adapters. |
| [qasp/rl/policies.py](../qasp/rl/policies.py) | Quantum policy models and computational-basis action decoders. |
| [qasp/rl/pp_sim.py](../qasp/rl/pp_sim.py) | Packed-Pauli Heisenberg propagation with Python complex coefficients. |
| [qasp/rl/qfunctions/__init__.py](../qasp/rl/qfunctions/__init__.py) | Quantum action-value functions for discrete-action reinforcement learning. |
| [qasp/rl/qfunctions/drawing.py](../qasp/rl/qfunctions/drawing.py) | Draw the quantum circuit of a PQDQN using QASP's existing exporters. |
| [qasp/rl/qfunctions/models.py](../qasp/rl/qfunctions/models.py) | Parametrized quantum deep Q-networks built from QASP circuits. |
| [qasp/rl/qfunctions/observables.py](../qasp/rl/qfunctions/observables.py) | Pauli features from the exact Z support of QASP decoder projectors. |
| [qasp/rl/torchrl.py](../qasp/rl/torchrl.py) | Discrete-action RL trainers, baselines and a synthetic contextual bandit. |
| [qasp/rl/trl.py](../qasp/rl/trl.py) | Historical TorchRL PPO prototype; use qasp.rl.torchrl for maintained trainers. |
