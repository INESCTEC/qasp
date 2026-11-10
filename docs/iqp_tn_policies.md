# Dictionary-configured QASP policies on IQPTNSimulator

Research reproduction commands and recorded benchmark discussions in this guide refer to the [pinned upstream checkout](https://github.com/andre-sequeira10/QASP/tree/3bda8622bd6f5ddfb7c05a0dfd0538c1b0ccd718). This distribution keeps the runtime API and compact tutorials; see the [repository layout policy](repository_layout.md).

The unified [IQPSimulator](iqp_simulator.md) also works with these policy configurations. See the [new notebook](../examples/iqp_simulator.ipynb) for sampled Z-correlator training with neural encoding.

Set `"device": "IQPTNSimulator"` in the ordinary QASP policy configuration.
There is no custom policy wrapper and no need to import an example module.
`create_policy("PQCBorn", config, n_qubits=...)` builds the current
`qasp.rl.policies.PQCBornVariance`; `"PQCSoftmax"` builds `PQCSoftmax`.

The [executed notebook](https://github.com/andre-sequeira10/QASP/blob/3bda8622bd6f5ddfb7c05a0dfd0538c1b0ccd718/examples/iqp_tn_rl_policies.ipynb) shows the complete
configuration dictionaries, direct policy construction, Torch updates,
full-state comparisons and a short CartPole run. Every training loop is visible
in the notebook. The [standalone validation script](https://github.com/andre-sequeira10/QASP/blob/3bda8622bd6f5ddfb7c05a0dfd0538c1b0ccd718/examples/iqp_tn_rl_policies.py)
uses internal helpers only to run its repeated checks; they are not part of the
public policy API or required by the notebook.

## Use the existing dictionary factory

```python
from copy import deepcopy
import numpy as np
import torch
from qasp.rl.model import create_policy
from qasp.rl.policies import PQCBornVariance

n_qubits = 5
policy_config = {
    "ansatz": "IQPAnsatz",
    "ansatz_config": {
        "n_layers": 1,
        "imprimitive_pattern": "chain",
    },
    "feature_map": "AngleEmbedding",
    "feature_map_config": {
        "features": 4,
        "rotations": ["Z 0", "Z 1", "Z 2", "Z 3"],
        "input_scaling": True,
    },
    "n_actions": 2,
    "policy": "majority-modulo",
    "majority_blocks": [[0, 1, 2, 3, 4]],
    "training_method": "probs",
    "parity_cnot_decomposition": False,
    "device": "IQPTNSimulator",
    "diff_method": "backprop",
    "shots": None,
}
policy = create_policy("PQCBorn", deepcopy(policy_config), n_qubits=n_qubits).double()
assert isinstance(policy, PQCBornVariance)
observation = torch.tensor([.2, -.4, .3, .1], dtype=torch.float64)
probabilities = policy(observation)
print("Probabilities:", probabilities.detach().tolist())
print("Executed backend:", policy.circuit.device.last_report.selected)
assert policy.circuit.device.last_report.selected == "IQPTNSimulator"
```

Use a deep copy when reusing configurations: the existing factory fills in
nested ansatz/feature-map dictionaries. For an axis string such as `"Z"`, its
AngleEmbedding route currently sets the feature count to `n_qubits`; the explicit
rotation list above preserves four input features on five wires.

`policy(observation)` and `policy.action_probs(observation)` return the same
Born action probabilities. The compact TN decoder avoids materializing the
full basis distribution. A different supported Born decoder can be selected
with `"policy": "parity-like"`, `"contiguous-like"` or `"hidden-like"`.

## Or instantiate the imported policy directly

Direct constructors accept classes rather than registry-name strings. Both
routes use the same policy implementation and device:

```python
from qasp.rl.policies import PQCBornVariance
from qasp import IQPAnsatz
from qasp.feature_maps import AngleEmbedding

direct_config = deepcopy(policy_config)
direct_config["ansatz"] = IQPAnsatz
direct_config["feature_map"] = AngleEmbedding
direct_config["ansatz_config"]["n_qubits"] = n_qubits
direct_config["feature_map_config"]["n_qubits"] = n_qubits
policy_direct = PQCBornVariance(n_qubits=n_qubits, **direct_config).double()
policy_direct.load_state_dict(policy.state_dict())
torch.testing.assert_close(policy_direct(observation), policy(observation))
print("Direct constructor and dictionary factory agree.")
```

The direct `PQCBorn` import is a legacy sampling class. The factory's
`"PQCBorn"` selector refers to the current `PQCBornVariance` probability policy.

## Commuting AngleAnsatz with RX, RXX and RXXX

The simulator now recognizes Pauli rotations sharing one product Pauli basis.
For the X family it inserts local Hadamards internally, using
`H X H = Z`, and absorbs those gates into its preparation/readout tensors.
The same exact contraction and Torch gradient path then handle the circuit.
No statevector or dense multiwire gate matrix is needed for this conversion.

```python
angle_config = deepcopy(policy_config)
angle_config["ansatz"] = "AngleAnsatz"
angle_config["ansatz_config"] = {
    "n_layers": 2,
    "rotations": (
        [f"X {w}" for w in range(n_qubits)]
        + [f"XX {w} {w+1}" for w in range(n_qubits - 1)]
        + [f"XXX {w} {w+1} {w+2}" for w in range(n_qubits - 2)]
    ),
    "include_imprimitive": False,
    "include_hadamards_initial": False,
    "reuploading": True,
}
angle_config["feature_map_config"]["rotations"] = ["X 0", "X 1", "X 2", "X 3"]
angle_policy = create_policy("PQCBorn", deepcopy(angle_config), n_qubits=n_qubits).double()
print("Commuting-X probabilities:", angle_policy(observation).detach().tolist())
print("Executed backend:", angle_policy.circuit.device.last_report.selected)
assert angle_policy.circuit.device.last_report.selected == "IQPTNSimulator"

```

Here `"X 0"`, `"XX 0 1"` and `"XXX 0 1 2"` are AngleAnsatz's syntax for RX,
RXX and RXXX. `include_imprimitive=False` prevents extra fixed CNOT/CZ gates.
Multiple commuting layers and matching X-axis reuploading are supported.
Product-Y families and consistent per-wire X/Y/Z axes are also supported.

Compatibility depends on the **actual gates and their ordering**, not the
ansatz class name. A one-layer IQPAnsatz is supported with the encoding shown
above. Repeated IQPAnsatz layers insert Hadamards between interactions, so they
do not automatically qualify. Noncommuting feature encoding inside an
interaction block can also make an AngleAnsatz unsupported.

Pairwise commutation alone is not sufficient for this implementation: XX and YY
on the same two wires commute but do not share a product Pauli basis. Their joint
diagonalization requires an entangling basis change, which this device does not
implement. Explicit TN selection rejects unsupported circuits; `"device": "auto"`
may choose its bounded statevector fallback. Inspect
`policy.circuit.device.last_report.selected` and `.reason` to see what ran.

## Neural embedding in the same configuration

The same commuting AngleAnsatz can learn its input projection:

```python
neural_config = deepcopy(angle_config)
neural_config["feature_map"] = "NeuralEmbedding"
neural_config["feature_map_config"] = {"features": 4, "rotations": "X"}
neural_policy = create_policy("PQCBorn", deepcopy(neural_config), n_qubits=n_qubits).double()
x = observation.clone().requires_grad_()
loss = -neural_policy(x)[1].log()
loss.backward()
encoder_gradient = dict(neural_policy.named_parameters())["qlayer.input_scaling_params"].grad
assert encoder_gradient is not None and torch.isfinite(encoder_gradient).all()
assert x.grad is not None and torch.isfinite(x.grad).all()
```

`NeuralEmbedding` applies a learned linear projection from four input features
onto each wire's X rotation, independently in each reuploading layer. It has no
bias or activation. The ordinary Torch optimizer sees both those projection
weights and the variational circuit weights. Evaluate one observation at a time
and stack outputs for a batch, as shown in the notebook's three Adam updates.
The notebook also compares complete gradients and updated weights against
`default.qubit`.

## Torch training and optional optimized rollout

Use ordinary `torch.optim.Adam(policy.parameters(), ...)`, `loss.backward()`
and `optimizer.step()`. Gradients reach circuit parameters, connected feature-map
parameters and differentiable inputs. The notebook compares all of these with
`default.qubit`, including the parameters after Adam updates.

```python
observations = torch.tensor([
    [.2, -.4, .3, .1], [-.3, .1, .4, -.2], [.5, .2, -.1, .3],
], dtype=torch.float64)
actions = torch.tensor([0, 1, 0])
advantages = torch.tensor([1., -.4, .7], dtype=torch.float64)
optimizer = torch.optim.Adam(policy.parameters(), lr=.01)
for step in range(3):
    optimizer.zero_grad()
    probabilities = torch.stack([policy(x) for x in observations])
    distribution = torch.distributions.Categorical(probs=probabilities)
    loss = -(distribution.log_prob(actions) * advantages).mean()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in policy.parameters())
    optimizer.step()
    print({"step": step + 1, "loss": float(loss.detach())})
```

The simulator name selects default eager execution. Only to customize device
options, place an instance in the same dictionary entry:

```python
from qasp import IQPTNSimulator

optimized_config = deepcopy(angle_config)
optimized_config["device"] = IQPTNSimulator(n_qubits, inference_execution="torchscript")
optimized_policy = create_policy("PQCBorn", optimized_config, n_qubits=n_qubits).double()
with torch.no_grad():
    optimized_policy(observation)  # warmup
    p = optimized_policy(observation)
    action = torch.distributions.Categorical(probs=p).sample()
cache = optimized_policy.circuit.device.inference_cache_info()
loss = -optimized_policy(observation)[action].log()
loss.backward()
assert optimized_policy.circuit.device.inference_cache_info() == cache
print("Inference cache:", cache)
```

`torch.no_grad()` activates the inference cache. The first call includes tracing;
subsequent calls reuse contractions with current parameters. Training outside
that context keeps eager autograd. `policy.eval()` alone does not disable
recording. Recompute rollout probabilities with autograd before training.

For `PQCSoftmax`, construct it through `create_policy("PQCSoftmax", config, ...)`
and use `Categorical(logits=policy(observation))` to include its trainable head.
Its legacy `action_probs()` shortcut bypasses that head. The notebook shows a
complete dictionary and optimizer step on the commuting AngleAnsatz.

These examples use `training_method="probs"`, `shots=None` and
`diff_method="backprop"`. The Lightning-specific `sparse_projectors`/adjoint path
is not the IQP TN configuration. The older sampled `PQCBorn` and experimental
`LearnablePQCBorn` interfaces are not validated by these examples.

## Execute and inspect

```bash
conda activate torchlane
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/private/tmp/iqptn-mpl
export NUMBA_CACHE_DIR=/private/tmp/iqptn-numba
python examples/iqp_tn_rl_policies.py --steps 3 --episodes 4 \
    --output docs/benchmarks/iqp_tn_policies.json
```

The notebook and script run actual public QASP policies, with no-grad rollout
and eager training. Four CartPole episodes demonstrate execution and finite
gradients, not convergence. See [recorded checks](https://github.com/andre-sequeira10/QASP/blob/3bda8622bd6f5ddfb7c05a0dfd0538c1b0ccd718/docs/benchmarks/iqp_tn_policies.json)
and the [precise simulator contract](iqp_tn.md).

The [AngleAnsatz benchmark](https://github.com/andre-sequeira10/QASP/blob/3bda8622bd6f5ddfb7c05a0dfd0538c1b0ccd718/docs/iqp_tn_angle_policy_comparison.md) compares angle
and neural embeddings with PennyLane MPS and TN at 4, 8 and 12 qubits, including
warmup, inference, training forward and complete backward timings. Its native
softmax and dense-state-adapted Born comparisons are reported separately.
The [scientific manuscript](https://github.com/andre-sequeira10/QASP/blob/3bda8622bd6f5ddfb7c05a0dfd0538c1b0ccd718/docs/iqp_tn_paper/iqp_tn_simulator_paper.pdf) derives the algorithms and
contraction-width complexity bounds. The [earlier performance report](https://github.com/andre-sequeira10/QASP/blob/3bda8622bd6f5ddfb7c05a0dfd0538c1b0ccd718/docs/iqp_tn_pennylane_comparison.md)
retains the Z-basis experiments at their original source commit; these are
separate gate-only workloads, not timings of full RL training episodes.

## Validation recorded 27 October 2026

The updated notebook executed all twelve code cells in `torchlane`. The core,
inference, application and new product-basis suites passed **219 tests**.
The 47 product-basis cases cover X/Y/mixed product bases, double/single precision, compact
readouts, parameter and input gradients, Hessians, cached inference, unsupported
families, and dictionary-configured Born policies with matching Adam updates.

The earlier standalone policy check also passed for all seven configurations. Its
commuting-X case matched full-state outputs within `2.22e-16` and parameter/input
gradients within `2.22e-16`. The notebook separately checks two commuting layers
with X feature reuploading.

The [new validation record](https://github.com/andre-sequeira10/QASP/blob/3bda8622bd6f5ddfb7c05a0dfd0538c1b0ccd718/docs/benchmarks/iqp_tn_angle_validation.json) includes the
108 benchmark workers and two additional neural-policy regression cases. These
compare Born and softmax outputs, encoder/input gradients and three Adam updates
with `default.qubit`, including cached rollouts after parameter changes.
