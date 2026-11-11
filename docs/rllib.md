# Optional RLlib integration

Keep defining and using your models through `qasp.rl` as before. The new
`qasp.rl.rllib` submodule is an explicit opt-in; existing model code, exports,
trainers and dependencies have not been changed. There is no separate adapter
package and no replacement model-definition format. The source checkout includes
this integration; PyPI QASP 0.3.0 predates it. Install this checkout with
`python -m pip install -e . "ray[rllib]==2.60.0"` to use the integration and the
PennyLane gate-metadata compatibility fix. Restart existing notebook kernels.

```python
from qasp.rl import PQCSoftmax, PQDQN

policy = PQCSoftmax(...)  # your existing constructor and arguments
q_model = PQDQN(...)     # your existing constructor and arguments

# Only if you choose RLlib:
from qasp.rl.rllib import ppo_module_spec, dqn_module_spec
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig

ppo_config = (
    PPOConfig()
    .environment("CartPole-v1")
    .rl_module(rl_module_spec=ppo_module_spec(policy))
)
dqn_config = (
    DQNConfig()
    .environment("CartPole-v1")
    .training(dueling=False, num_atoms=1)
    .rl_module(rl_module_spec=dqn_module_spec(q_model))
)
```

Continue with RLlib's normal `config.build_algo()` and `algo.train()` calls.
Models must match the environment's observation and action dimensions. See the
complete [CartPole notebook](../examples/cartpole_rllib.ipynb) and
[runnable example](../examples/rllib_cartpole.py).

## What the adapters do

`ppo_module_spec(policy)` snapshots the existing policy and adds an independent
classical critic (two hidden layers of 64 units by default). You may pass your
own PyTorch critic with `critic=my_critic`. The policy's observations are not
rescaled or normalized by the adapter. `PQCSoftmax.forward()` supplies logits;
`PQCBornVariance.forward()` supplies probabilities. Legacy `PQCBorn` and
`LearnablePQCBorn` use their existing `action_probs()` entrypoints. Their existing
preprocessing behavior is retained. The adapter evaluates policies per sample
to accommodate QASP methods that flatten batch axes, then reconstructs the batch.
For deterministic policies compatible with `torch.vmap`, explicitly opt into
`ppo_module_spec(policy, policy_batch_mode="vmap")`. The CartPole notebook uses
this with analytic `default.qubit`/`backprop`, and uses PQDQN's existing
`batch_mode="broadcast"`. Output/gradient parity is tested against the sequential
path. Finite-shot or other simulator backends may not support these modes; the
adapter default remains sequential.

`dqn_module_spec(q_model)` wraps existing scalar Q-values without a softmax.
RLlib handles standard/Double DQN losses, replay, exploration and target updates.
Target networks are independent and have frozen parameters. Set `dueling=False`
and `num_atoms=1`: distributional and dueling heads are outside this adapter's scope.

The adapters currently accept flat `Box` observations and zero-based `Discrete`
actions. The Q-network must preserve leading batch dimensions. Born probabilities
are normalized and floored at `1e-8` before conversion to categorical log-weights;
this small explicit smoothing avoids taking the logarithm of zero. Override
`probability_floor` if needed. `policy_method` can select a non-default entrypoint,
with explicit `policy_output="logits"` or `"probs"` where inference is ambiguous.

## Hybrid models

Use standard PyTorch composition:

```python
from torch import nn

hybrid_policy = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), policy)
critic = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 1))
spec = ppo_module_spec(hybrid_policy, critic=critic, policy_output="logits")
```

For arbitrary hybrid models, explicitly declare the meaning of the policy output.
All trainable quantum and classical components must be registered as PyTorch
submodules or parameters, as in ordinary PyTorch training.

## Original objects and trained weights

Creating a spec captures the model's current architecture and weights. Later
changes to the original model do not change that spec. RLlib trains independent
copies, leaving the caller's model and gradients unchanged. After training:

```python
trained_policy = ppo.get_module().policy
trained_q_model = dqn.get_module().q_network

# Optional: explicitly copy trained weights into your original QASP objects.
policy.load_state_dict(trained_policy.state_dict())
q_model.load_state_dict(trained_q_model.state_dict())
```

These are ordinary QASP/PyTorch models. Their state dictionaries can be saved and
loaded with PyTorch independently of RLlib. Full RLlib checkpoints additionally
preserve optimizer/training state; enable DQN's `store_buffer_in_checkpoints=True`
to retain replay contents. Resuming does not promise bit-identical RNG trajectories.

For evaluation of a selected checkpoint, load its learner module directly. In
Ray 2.60, restoring an older checkpoint into a running algorithm can leave its
local rollout module with newer weights:

```python
from pathlib import Path
from ray.rllib.core.rl_module.rl_module import RLModule

module = RLModule.from_checkpoint(str(Path(checkpoint) /
    "learner_group/learner/rl_module/default_policy"))
```

## Distributed workers and Slurm

Use native `.env_runners(num_env_runners=...)` settings to distribute rollouts.
The supplied examples keep a single learner. Each worker gets an independent
model copy via Ray/cloudpickle. If a simulator owns resources that cannot be
serialized, pass a zero-argument factory instead of an instance:

```python
def make_policy():
    return PQCSoftmax(...)  # exactly the usual QASP constructor

spec = ppo_module_spec(make_policy)
```

The factory builds the model within each worker, and RLlib synchronizes weights.
Use identical architectures on every worker. Existing weights can be loaded in
the factory if needed. MPI-backed circuits are rejected: distributing rollout
workers is separate from distributing one quantum state. GPU simulation also
requires backend-specific validation; the provided recipes use CPU simulation.

The [Slurm script](../examples/slurm/rllib_cartpole.sbatch) starts one Ray runtime
on each of two exclusive CPU nodes. Install the same QASP including this optional
submodule, Ray 2.60.0, and simulator dependencies on every node:

```bash
export QASP_PYTHON=/shared/envs/qasp-validation/bin/python
export QASP_EXAMPLE=/shared/QASP/examples/rllib_cartpole.py
export QASP_OUTPUT=/shared/results/cartpole
export QASP_ALGORITHM=ppo  # or dqn
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION examples/slurm/rllib_cartpole.sbatch
```

Adjust account, partition, wall time and CPU allocation for the cluster.
`QASP_WORKERS` overrides the default allocated CPU count minus one. The script
limits thread pools to one and reserves a core for learning. Node hostnames and
Ray's network ports must be reachable inside the allocation. Start Ray outside
the QASP repository root when testing an installed release: Ray prepends the
driver's working directory to worker import paths. The examples do this explicitly.

## Development validation against PyPI

The new optional submodule is not yet part of the published QASP 0.3.0 wheel.
To test it against that exact release without replacing any published model or
trainer code, use the existing `qasp-validation` environment:

```bash
.venvs/qasp-validation/bin/python -I tests/rllib/prepare_environment.py
MPLCONFIGDIR=/tmp/qasp-rllib-mpl OMP_NUM_THREADS=1 \
  .venvs/qasp-validation/bin/python -I -m pytest tests/rllib -q
```

The preparation script stages only `qasp/rl/rllib.py` beside the installed package
and verifies all 62 existing PyPI Python sources are unchanged. It creates no new
distribution and does not edit package metadata. Once this addition is released
as part of QASP, this development-only step is unnecessary.

The source also removes deprecated PennyLane `id` gate arguments and stores
QAOA drawing labels as QASP metadata. These changes preserve circuit values and
gradients. The warning was treated as an error in regression tests.

Validation uses QASP 0.3.0, Ray 2.60.0, PyTorch 2.15.0 and PennyLane 0.45.1.
Tests cover existing API use with Ray imports blocked, unchanged caller objects,
model-output parity, hybrid and quantum gradients, target updates, checkpoint
roundtrips, distributed PPO/DQN rollouts, worker provenance and resumed training.

## Full CartPole training result

The notebook was executed in `qasp-validation` on 2026-11-11 with training seed 7,
16 local vector environments, analytic quantum simulation, and the versions above.
Checkpoint selection used 50 validation episodes (seeds 10000–10049). A separate
100-episode greedy evaluation (seeds 40000–40099) gave:

| Model | Training steps | Mean test return | Target ≥475 |
| --- | ---: | ---: | --- |
| PPO: PQCSoftmax, NeuralEmbedding, classical critic | 15,643 | 497.49 | Passed |
| Double DQN: PQDQN, angle encoding, nonlinear classical head | 20,782 | 500.00 | Passed |

Both models use four qubits and two reuploading layers. PPO uses learning rate
0.001 and five optimization epochs per 512-step batch. DQN uses learning rate
0.001, mean-square Bellman loss, three-step returns, replay, and soft target
updates (`tau=0.01`). All predictions pass through trainable quantum features.
The notebook stops at validation mean ≥495, with a limit of 500 iterations.

A compact [validation record](rllib_validation.json) retains all 100 test returns
for each algorithm. After the gate-metadata fix, 174 regression tests passed with
the deprecated ID warning treated as an error; the full notebook emitted no such
warning and both models met the target again.

The earlier small-model settings did not solve CartPole; smoke-test success alone
did not establish convergence. The notebook loads the selected checkpoint directly before final evaluation.
Its repository copy has cleared outputs; executed copies and checkpoints remain
local, following the repository layout policy.
This demonstrates convergence for one training seed; it is not a multi-seed
reliability benchmark. Distributed worker tests passed, but this convergence run
was local and actual multi-node Slurm scheduling was not tested.

To repeat the full run:

```bash
.venvs/qasp-validation/bin/python -I tests/rllib/execute_notebook.py --full
```
