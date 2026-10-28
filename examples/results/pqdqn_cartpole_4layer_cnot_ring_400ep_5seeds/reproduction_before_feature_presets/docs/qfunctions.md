# Quantum Q-functions and deep Q-learning

`qasp.rl.qfunctions.PQDQN` produces one **Q-value** per discrete action. It uses
QASP's `CircuitBuilder`, feature maps and ansatz templates, plus an optional
`torch.nn.Linear(..., bias=False)` readout. There is no softmax. All circuit,
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
    "observables": ["I 0", "Z 0", "Z 1", "Z 2", "Z 3"],
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
circuit compiler. The custom `lightning.pp` engine is not supported here.

Inputs have shape `(..., features)` and outputs `(..., n_actions)`. Inputs use
`tanh` by default; set `input_activation=None` when preprocessing externally.
The default `batch_mode="sequential"` supports existing feature maps including
NeuralEmbedding and explicit rotation lists. `batch_mode="broadcast"` is a
faster opt-in for compatible circuits, such as AngleEmbedding with string
rotations and backpropagation. Parameter-shift training should use sequential
mode. Move the model to its intended dtype/device **before** creating a trainer.

## Observable choices

| Configuration | Quantum features before the classical layer |
| --- | --- |
| `["Z 0", "X 1", "ZZ 0 1"]` | A shared vector of three expectation values |
| `[["Z 0", "-Z 1"], ["ZZ 0 1"]]` | One sum per action: `<Z0> - <Z1>` and `<Z0 Z1>` |
| `"parity-like"` | Exact action-projector expectations from nested prefix parities |
| `"majority-modulo"` | Exact action-projector expectations from majority block bits, modulo the number of actions |

Strings use QASP notation, e.g. `"-XZ 0 2"`. Full-register words such as
`"XIZI"` are also accepted. Each nested list has unit coefficients; use a
leading minus sign for negative terms. Nested lists must contain exactly one
non-empty group per action. A flat list may contain any positive number of
observables when the classical layer is enabled.

`fc_layer=True` (default) learns a bias-free map from those features to action
values. For shared strings, including `"I 0"` supplies a constant quantum
feature so the linear layer can learn a common Q-value offset without a bias
parameter. Named projector decompositions already include their identity term.
`fc_layer=False` returns the quantum features directly and requires one feature
per action. Those values are bounded; this is useful for inspecting the
observables but generally cannot represent long-horizon, unscaled returns.

### Predefined Pauli-Z decompositions

```python
majority_model = create_qfunction(
    dict(config,
         observables="majority-modulo",
         measurement_qubits=[0, 1, 2, 3],
         majority_blocks=[[0, 1, 2, 3]]),
    n_qubits=4, n_actions=2,
)
```

These names follow the same action partitions as `PQCBornVariance`. They build
an exact Pauli-Z decomposition of **each action projector**, including its
coefficient and identity term. Measurement uses the Pauli terms, followed by a
fixed coefficient matrix, then the optional trainable linear layer. There is
no sampling-based decoder or approximation to the majority observable.

Parity uses `ceil(log2(n_actions))` nested prefix parity bits in most-significant
bit order, then reduces the code modulo `n_actions`. Majority uses one bit per
disjoint block, with strict majority and ties resolving to zero. Explicit
blocks contain physical wire labels. Default blocks are balanced odd-sized
blocks and can leave one measured wire unused, matching QASP's existing policy.
For four measured wires and two actions, pass an explicit four-wire block if
you want all four wires involved in majority voting.

The decomposition uses a Walsh-Hadamard transform, not dense operator matrices.
Its construction still scales exponentially in measured wires and is limited
to 16 measured wires and at most 2**22 action-by-basis coefficients.
`observable_strings` and `readout_coefficients` expose the
result. For more wires, supply explicit Pauli strings.

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

Validation covered 35 new tests for the Q-functions and DQN trainer. A selected
regression run passed 102 tests, with two skips and the unavailable optional
`lightning.kokkos` case excluded. The two notebooks were executed with saved
outputs; all four observable configurations completed CartPole replay updates.

### Four-layer readout comparison

Both notebooks now use the same four-layer AngleAnsatz with Y/Z rotations,
data reuploading, and CNOT rings after every layer (including the final layer).
The four readouts are shared strings, per-action sums, parity, and majority-modulo.
The CartPole comparison trains each readout with seeds 7–11, 400 episodes per
seed, for 20 independent runs. All other training settings remain those of the
original experiment. Equal episode budgets do not imply equal environment steps.

Every model created explicitly in the notebooks has compact and expanded TikZ
sources, vector PDFs, and previews rendered from those PDFs. The CartPole
gallery covers all 20 readout/seed configurations before the training cell.
The validation notebook also draws the two Born reference models and its four
fresh smoke-training models. The fully connected readout is classical and is
not shown as a quantum gate.

The comparison computes a trailing 20-episode mean separately within each seed,
then plots the mean and sample standard deviation across five seeds at each
episode. The first 19 episodes are omitted. Shading is one standard deviation,
not a confidence interval. Final held-out returns are summarized across five
trained models, with 100 evaluation episodes per model.

The runner lives in `examples/pqdqn_cartpole_experiment.py` and the notebook
passes its settings explicitly. Completed runs are reused only when their
specifications match; incomplete runs restart. Current results go to
`examples/results/pqdqn_cartpole_4layer_cnot_ring_400ep_5seeds/`, preserving the earlier pilot.
Each readout/seed directory contains `circuit.tex`, `circuit.pdf`, `circuit.png`,
expanded `circuit_full.*` figures, `circuit_readout.json`, `training.csv`,
`evaluation.csv`, `model.pt`, and `summary.json`. Aggregate CSVs and comparison
plots are produced after all runs finish.
