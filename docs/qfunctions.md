# Quantum Q-functions and deep Q-learning

`qasp.rl.qfunctions.PQDQN` produces one **Q-value** per discrete action. It uses
QASP's `CircuitBuilder`, feature maps and ansatz templates, plus an optional
`torch.nn.Linear` readout (with bias for shared and preset features). There is no softmax. All circuit,
input-scaling and readout parameters participate in training.

## Configuration

```python
from qasp.rl import create_qfunction, DeepQLearning

config = {
    "ansatz": "AngleAnsatz",       # IQPAnsatz or a template class also works
    "ansatz_config": {
        "n_layers": 4,
        "rotations": "YZ",
        "reuploading": True,
        "include_imprimitive": True,
        "imprimitive": "CNOT",
        "imprimitive_pattern": "ring",
    },
    "feature_map": "AngleEmbedding",  # NeuralEmbedding or a class also works
    "feature_map_config": {
        "features": 4,
        "rotations": "Y",
        "input_scaling": True,
    },
    "observables": ["Z 0", "Z 1", "Z 2", "Z 3"],
    "fc_bias": True,
    "fc_layer": True,
    "device": "default.qubit",
    "diff_method": "backprop",
    "batch_mode": "broadcast",
}
model = create_qfunction(config, n_qubits=4, n_actions=2)
# Equivalent: PQDQN.from_config(config, n_qubits=4, n_actions=2).
# Legacy factory: create_model("PQDQN", config_with_n_actions, n_qubits=4).

with DeepQLearning(model, "CartPole-v1", seed=7) as trainer:
    history = trainer.train(num_episodes=500)
    returns = trainer.evaluate(num_episodes=20, seed=10000)
```

Dictionaries are copied. `n_layers`, `features`, and `n_actions` can use the
policy-style callables of `n_qubits`. A preconfigured `circuit_builder` can be
supplied instead of feature-map/ansatz configuration; it is copied before its
measurement is set. `shots`, `mpi`, and `device_kwargs` pass through to QASP's
circuit compiler. The custom `lightning.pp` engine is supported for analytic
expectations through Torch autograd; see the configuration below.

Inputs have shape `(..., features)` and outputs `(..., n_actions)`. Inputs use
`tanh` by default; set `input_activation=None` when preprocessing externally.
The default `batch_mode="sequential"` supports existing feature maps including
NeuralEmbedding and explicit rotation lists. `batch_mode="broadcast"` is a
faster opt-in for compatible circuits, such as AngleEmbedding with string
rotations and backpropagation. Parameter-shift training should use sequential
mode. Move the model to its intended dtype/device **before** creating a trainer.

## Train with differentiable Pauli propagation

`device="lightning.pp"` selects QASP's `DifferentiablePauliProp` engine. PQDQN
traces a tape using its live trainable parameters and evaluates all Pauli
expectations with that engine. A PennyLane QNode holds parameter shapes and
initialization only; no statevector execution is used for the forward pass.
Kokkos is not required. The same feature aggregation and affine head apply to
shared, per-action, parity, and majority configurations.

Using the circuit dictionary above:

```python
import torch
from qasp.rl import create_qfunction, DeepQLearning

pp_config = dict(
    config,
    device="lightning.pp",
    diff_method="backprop",       # Torch autograd through Pauli coefficients
    batch_mode="sequential",
    shots=None,
    pp_truncation=None,            # no Pauli-weight truncation
    pp_topM=None,
    pp_tau=0.0,
    pp_prune=False,                # unpruned reference calculation
)
torch.manual_seed(7)
model = create_qfunction(pp_config, n_qubits=4, n_actions=2)
with DeepQLearning(model, "CartPole-v1", seed=7, batch_size=16) as trainer:
    history = trainer.train(num_episodes=100)
    evaluation = trainer.evaluate(num_episodes=20, seed=10000)
```

All circuit, trainable embedding/input-scaling, and classical head parameters
retain their Torch gradient connections. Existing model checkpoints can be
loaded into an otherwise identical PP configuration. Target-network copies,
replay minibatches, greedy evaluation and TikZ/text/Matplotlib drawing remain
available. `batch_mode="broadcast"` is accepted for configuration compatibility,
but PP evaluates each sample individually; it does not perform native parameter
broadcasting. Arbitrary leading batch dimensions are preserved.

To use approximate propagation, choose explicit controls, for example:

```python
pp_config.update(
    pp_truncation=4,   # maximum retained Pauli weight during propagation
    pp_topM=20000,     # maximum retained terms at each pruning step
    pp_tau=1e-10,      # coefficient-magnitude threshold
    pp_prune_every=5,
    pp_prune=True,
)
```

The executed [10-qubit benchmark](../examples/pqdqn_pauli_propagation_benchmark.ipynb)
compares `lightning.pp` at `pp_truncation=2,3` against `lightning.qubit` with
adjoint differentiation, using identical four-layer CNOT-ring models and inputs.
The benchmark includes inference and
forward/backward timings, feature/Q-value errors, and quantum-gradient errors.
In this measured example, `k=2` is faster but both low cutoffs give poor gradient
approximations; these timings do not demonstrate equivalent learning performance.

Defaults are `pp_truncation=None`, `pp_topM=20000`, `pp_tau=1e-10`,
`pp_prune_every=5`, and `pp_prune=True`. `pp_prune_every=0` applies coefficient
pruning only at the end. `pp_prune=False` disables coefficient/top-M pruning,
**not** Pauli-weight truncation; set `pp_truncation=None` as well for the
untruncated reference. Hard pruning decisions use detached coefficient
magnitudes: gradients follow retained paths, not the discrete selection mask.
Approximation can remove informative paths, so check outputs/gradients against
an unpruned small model before choosing aggressive limits.

PP requires `shots=None`, `mpi=False`, and `diff_method="backprop"` or `"best"`.
The engine decomposes templates to its supported gates (CNOT, CZ, H, RX/RY/RZ,
one-qubit PauliRot and XX/YY/ZZ PauliRot). Unsupported gates raise an error.
AngleEmbedding, NeuralEmbedding, AngleAnsatz and IQPAnsatz have integration
coverage. Unpruned Pauli expansions can grow exponentially; this backend is
not guaranteed to be faster than statevector simulation on small circuits.
The recorded five-seed CartPole comparison still uses default.qubit/backprop;
the PP integration is separately checked by value/gradient equivalence and a
short real CartPole training test, not a new convergence experiment.

## Observable choices

| Configuration | Features supplied to the classical head | Default bias |
| --- | --- | --- |
| `["Z 0", "X 1", "ZZ 0 1"]` | Three individual Pauli expectations | Yes |
| `[["Z 0", "-Z 1"], ["ZZ 0 1"]]` | One sum per action | No |
| `"parity-like"` with `preset_actions=a` | Distinct non-identity Z terms in the a-way parity decomposition | Yes |
| `"majority-modulo"` with `preset_actions=a` | Distinct non-identity Z terms in the a-way majority decomposition | Yes |

Strings use QASP notation, e.g. `"-XZ 0 2"`; full words such as `"XIZI"`
are also accepted. Nested lists must contain one non-empty group per environment
action, with unit coefficients (a leading minus sign supplies a negative term).

For shared and preset readouts, the default is **Q(s) = W z(s) + b**, where
z contains individual expectation values. There is no softmax and no fixed
probability reconstruction in the network. `fc_bias` can explicitly override
the default in any readout. Shared observables need no identity term: the bias
provides the constant offset. User-supplied identity terms are still accepted.
`fc_layer=False` requires exactly `n_actions` features and returns them directly;
these bounded values usually cannot represent unscaled long-horizon returns.

### Preset size is independent of environment actions

`preset_actions` is the user's a=|A| parameter for selecting the decomposition.
It defaults to `n_actions` but need not equal it. `n_actions` always specifies
the number of final Q-values required by the environment.

```python
parity_model = create_qfunction(
    dict(config, observables="parity-like", preset_actions=4, fc_bias=True),
    n_qubits=4, n_actions=2,
)
assert parity_model.observable_strings == (
    "ZZZZ 0 1 2 3", "ZZZ 0 1 2", "Z 3",
)
assert parity_model.fc_layer.in_features == 3
assert parity_model.fc_layer.out_features == 2
```

For a=2, parity supplies the single full-register product Z0 ... Z(n-1).
For a=4, it supplies Z0 ... Z(n-1), Z0 ... Z(n-2), and Z(n-1).
The last term is the product of the two prefix parity observables. More generally,
QASP computes the Pauli support of the decoder's action projectors, removes
identity, deduplicates the terms, and feeds their expectations independently to
the trainable head. Coefficients used to reconstruct probabilities are **not**
applied. Features are ordered by decreasing support size then mask order.
Non-power-of-two a follows the existing decoder's modulo convention.

```python
majority_model = create_qfunction(
    dict(config, observables="majority-modulo", preset_actions=4,
         majority_blocks=[[0, 1, 2], [3]], fc_bias=True),
    n_qubits=4, n_actions=2,
)
```

Majority follows the same construction using `ceil(log2(a))` disjoint blocks.
Each block uses strict majority, with ties assigned to zero. Blocks contain
physical wire labels; `measurement_qubits` selects their ordered domain.
Default blocks follow QASP's balanced odd-size convention and can leave one
wire unused. Explicit blocks make the choice reproducible.
For a three-wire majority, the signed majority observable is
(Z0 + Z1 + Z2 - Z0 Z1 Z2)/2, but the Q-network receives **four separate features**
Z0, Z1, Z2, Z0 Z1 Z2. For a=4 with blocks [0,1,2] and [3], the union of block
terms and their products yields nine independent input features to the head.
Even-size majority blocks can produce substantially larger Pauli supports.

Construction uses a Walsh-Hadamard transform, not dense operator matrices.
It is exponential in measured wires: at most 16 measured wires, 2**22
projector-by-basis coefficients and 2**22 feature-matrix entries are allowed.
`observable_strings` exposes the terms; `readout_coefficients` is identity for
preset features. These are construction limits, not a scalable majority method.

### Historical projector readouts

The earlier implementation reconstructed action probabilities before the head.
That is a different model. Explicit `preset_mode="projectors", fc_bias=False`
reproduces its named-readout convention and checkpoint shapes. To restore an
old shared or per-action checkpoint, keep its original strings and set
`fc_bias=False`. Old results remain under
`examples/results/pqdqn_cartpole_4layer_cnot_ring_400ep_5seeds/`, with original
source/notebook snapshots in `reproduction_before_feature_presets/`.
Never mix those results with the corrected feature-based experiment.

## Draw a quantum Q-network

`PQDQN.draw()` supports Matplotlib (default), text, and Quantikz/TikZ:

```python
# Use the existing model's current weights.
tikz_source = model.draw(
    output="tikz",
    tikz_options={"standalone": True, "symbolic_parameters": True},
)

# Optional: write a compilable LaTeX source file (requires Quantikz to compile).
from pathlib import Path
Path("pqdqn.tex").write_text(tikz_source)

# Numeric gates for a particular raw observation; tanh is applied if configured.
tikz_source = model.draw(inputs=[0.1, 0.2, -0.1, 0.3], output="tikz")
figure, axes = model.draw(output="matplotlib")
print(model.draw(output="text"))
```

The equivalent helper is `qasp.rl.draw_qfunction(model, output="tikz")`. To
construct and draw directly from a dictionary, use
`draw_qfunction(config=config, n_qubits=4, n_actions=2, output="tikz")`.
The existing `draw_model(model=model, output="tikz")` and
`draw_model(policy_class="PQDQN", policy_config_kwargs=config_with_n_actions,
n_qubits=4, output="tikz")` entry points also support PQDQN.

Omitted inputs use a zero vector with the configured feature count. Supply one
observation, not a batch. Drawing uses the trained circuit and input-scaling
weights without modifying them. TikZ options pass through to QASP's exporter,
including `wire_labels`, `gate_colors`, and `compress`. For AngleAnsatz and
IQPAnsatz with data reuploading, compression expands the first complete
encoding/variational layer and represents later layers by data-dependent blocks.
The feature map must precede the first layer. Mirrored overlap compression
still requires `reuploading=False`. `tex` and `latex` are aliases for `tikz`.

Use the same compact paper options as `examples/drawing_quantum_circuits.ipynb`:

```python
compact_source = model.draw(
    output="tikz",
    tikz_options={"standalone": True, "symbolic_parameters": True,
                  "compress": True, "max_explicit_layers": 4},
)
```

The notebook helper `examples/pqdqn_paper_figures.py` compiles these sources
with `pdflatex`/Quantikz and renders the PDFs with `pdftoppm` or macOS `sips`.
It uses QASP's pastel palette. Set `standalone=False` for a paper snippet or
`gate_colors=False` for monochrome. Reuploaded AngleEmbedding scaling symbols
include their layer index, so independently trained scales are not confused.

The figure shows quantum gates and Pauli measurements. The fixed decomposition
coefficients and classical fully connected layer are outside the quantum circuit
and are not drawn. TikZ output is LaTeX source; generating a PDF requires compilation.

## DeepQLearning

`qasp.rl.torchrl.DeepQLearning` uses a manual PyTorch learning loop and the
Gymnasium API, alongside the existing trainers in that file. It accepts a
PQDQN or another differentiable module returning `(..., n_actions)` values.

- Uniform replay with a bounded capacity, warmup, training frequency and
  configurable minibatch/gradient-step counts.
- A frozen target network, synchronized every `target_update_interval`
  **optimizer updates**; Double DQN by default (`double_dqn=False` for vanilla).
- Huber loss, gradient clipping, and explicit non-finite checks.
- Linear epsilon decay measured in environment steps, using a local seeded RNG.
- True termination masks the Bellman continuation value. Gymnasium time limits
  and the trainer's rollout cap retain bootstrapping.
- Greedy evaluation on a separate environment, with fixed evaluation seeds and
  no updates to exploration, replay, or training counters.

Only zero-based `gymnasium.spaces.Discrete` actions are supported. Observations
are flattened; use a deterministic `observation_transform` for custom inputs.
A provided environment instance remains caller-owned. Pass a separate instance
to `evaluate(env=...)` in that case; environments created by name are managed
by the trainer. Use the context manager or `close()` to release them.

Seed PyTorch before constructing a model to reproduce its initialization.
The trainer's `seed` controls exploration, replay sampling, and training resets.
Each call to `train` continues the existing replay, counters, and schedule.
Returned records contain `episode`, `return`, `steps`, `frames`, `updates`,
`epsilon`, and mean minibatch `loss` (`None` before updates begin).

Save model weights with `torch.save(model.state_dict(), path)` and restore into
a model made with the same configuration. This is a model checkpoint, not an
exact trainer-resume checkpoint (optimizer, replay and RNG state are separate).

## Executed examples

- [Observable validation and CartPole smoke runs](../examples/pqdqn_observables_validation.ipynb)
- [CartPole learning experiment](../examples/cartpole_pqdqn_deep_q_learning.ipynb)

The notebooks record dependency versions, seeds, actual evaluation returns,
and learning curves. Short smoke runs validate execution and parameter updates;
they do not establish convergence. Learning results depend on initialization,
observable choice, circuit capacity, and training budget.

### Earlier two-layer pilot result

The earlier two-layer, single-seed learning experiment ran 600 episodes with seed 7 and 4,785 replay
updates. Greedy evaluation on 20 fixed seeds increased from 9.4 to 50.15;
the final model averaged 55.1 on 100 held-out episodes (standard deviation
31.75). This configuration **did not solve CartPole**. Its final held-out mean
is below the notebook's 475 reference. The checkpoint, raw returns, and plot
are saved under `examples/results/pqdqn_cartpole/`.

The earlier validation and pilot results apply to the original readout convention;
the corrected feature/bias implementation has separate validation and results.

### Corrected four-layer comparison

Both notebooks use the same four-layer YZ AngleAnsatz, Y encoding with trainable
scalings, data reuploading, and CNOT rings after every layer including the last.
There are four configurations: shared Z features with bias; the original
per-action sums without bias as a control; parity a=2; majority a=2
with block [0,1,2,3]. All presets use bias.
Preset sizes do not change CartPole's two output Q-values.

Seeds 7–11 and 400 episodes yield 20 independent runs. The backend remains
`default.qubit` with `backprop`. All other training settings are unchanged.
The corrected dataset is saved separately in
`examples/results/pqdqn_cartpole_pauli_features_bias_v2_400ep_5seeds/`.
The validation drawings are in `examples/results/pqdqn_validation_pauli_features_bias_v2/`.
Every explicitly created model has compact and expanded TikZ/PDF drawings;
readout metadata records the preset size, individual terms and presence of bias.

A trailing 20-episode mean is computed within each seed, then mean and sample
standard deviation across five seeds are plotted. The first 19 episodes are
omitted; shading is ±1 SD, not a confidence interval. Held-out evaluation uses
100 episodes per trained model, with five models as independent replicates.
Equal episode budgets need not imply equal environment steps.

The runner refuses reuse when the recorded specification changes. The v2
manifest records `readout_semantics="pauli_features_affine_v2"`, making the
changed feature interpretation explicit. Incomplete runs restart; completed
matching runs can be reloaded without retraining.
