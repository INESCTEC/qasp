"""Build the notebook demonstrating unchanged QASP model construction."""
from pathlib import Path
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb = nbf.v4.new_notebook()
nb.cells = [
md('''# Use existing QASP models with RLlib on CartPole

Construct your models with **`qasp.rl` exactly as before**. If you choose RLlib, import two optional adapters from **`qasp.rl.rllib`** and pass the existing objects to native `PPOConfig` or `DQNConfig`. Existing QASP trainers and model APIs are unchanged.

The full recipe was validated with **QASP 0.3.0**, the optional integration, and Ray 2.60.0 in `qasp-validation`. Install this source checkout before running the notebook:

```bash
python -m pip install -e ".[dev]" "ray[rllib]==2.60.0"
```

No separate adapter package is needed. Choose the interpreter where QASP was installed; this validation uses `qasp-validation`. The source also removes deprecated PennyLane gate IDs without changing circuit calculations.

Full mode trains for up to 500 iterations per algorithm, selects a checkpoint using 50 validation episodes, and evaluates 100 separate test episodes against a mean-return target of 475. Run the cells to reproduce the recorded full result in `docs/rllib.md`. Repository copies have cleared outputs; executed copies and checkpoints stay local.'''),
code('''import os
import sys
import tempfile
import json
import time
from pathlib import Path
from copy import deepcopy

START_DIRECTORY = Path.cwd().resolve()
# While validating against PyPI, prevent the repository copy from shadowing it.
for entry in list(sys.path):
    candidate = Path(entry or Path.cwd()).resolve()
    if (candidate / "qasp" / "__init__.py").exists() and "site-packages" not in candidate.parts:
        sys.path.remove(entry)
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "qasp-rllib-mpl"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")

import gymnasium as gym
import numpy as np
import torch
from torch import nn
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule

# Existing imports and model classes: no change.
from qasp.circuits import AngleAnsatz
from qasp.feature_maps import AngleEmbedding, NeuralEmbedding
from qasp.initialization import Normal, Constant
from qasp.rl import PQCSoftmax, PQCBornVariance, PQDQN

# Opt in only when using RLlib.
from qasp.rl.rllib import ppo_module_spec, dqn_module_spec

import qasp
from importlib.metadata import version
print("Python:", sys.executable)
print("QASP:", version("qasp"), "Ray:", ray.__version__)
print("Model package:", qasp.__file__)
torch.set_num_threads(1)
torch.manual_seed(7)'''),
md('''## Define the models as usual

These are ordinary QASP objects. PPO uses `PQCSoftmax` with QASP’s `NeuralEmbedding` and a classical critic. DQN uses `PQDQN` with angle encoding and a classical nonlinear output head. Every policy/Q-value passes through its trainable quantum circuit. You can call these models directly or use existing QASP trainers; RLlib does not change their constructors or APIs.'''),
code('''policy = PQCSoftmax(
    n_qubits=4, n_actions=2,
    ansatz=AngleAnsatz,
    ansatz_config=dict(n_qubits=4, n_layers=2, rotations="YZ", reuploading=True),
    feature_map=NeuralEmbedding,
    feature_map_config=dict(n_qubits=4, features=4, rotations="Y"),
    observables=["Z 0", "Z 1", "Z 2", "Z 3"], fc_layer=True,
    device="default.qubit", diff_method="backprop",
    init=Normal(std=1.0), input_scaling_init=Normal(std=0.5),
)

q_model = PQDQN(
    n_qubits=4, n_actions=2,
    ansatz=AngleAnsatz,
    ansatz_config=dict(n_qubits=4, n_layers=2, rotations="YZ", reuploading=True),
    feature_map=AngleEmbedding,
    feature_map_config=dict(n_qubits=4, features=4, rotations="Y", input_scaling=True),
    observables=["Z 0", "Z 1", "Z 2", "Z 3"], fc_layer=True,
    device="default.qubit", diff_method="backprop", batch_mode="broadcast",
    init=Normal(std=1.0), input_scaling_init=Constant(0.5),
)

# Standard PyTorch composition: every prediction still passes through PQDQN's
# trainable quantum features. There is no classical bypass of the circuit.
q_model.fc_layer = nn.Sequential(nn.Linear(4, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))

probe = torch.tensor([0.02, -0.03, 0.01, 0.04])
print("Usual policy logits:", policy(probe))
print("Usual Q-values:", q_model(probe))
original_policy_state = deepcopy(policy.state_dict())
original_q_state = deepcopy(q_model.state_dict())'''),
md('''## Optional hybrid model or custom critic

For other hybrid models, use the same `nn.Sequential` composition you would normally use. Pass `policy_output="logits"` or `"probs"` explicitly because the adapter cannot infer the meaning of an arbitrary hybrid model's output. A custom classical critic is also an ordinary PyTorch module.

The main PPO run below uses the original `policy`. To train the hybrid version instead, substitute `hybrid_spec` for `ppo_module_spec(policy)` in the PPO configuration.'''),
code('''hybrid_policy = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), deepcopy(policy))
classical_critic = nn.Sequential(nn.Linear(4, 64), nn.Tanh(), nn.Linear(64, 1))
hybrid_spec = ppo_module_spec(hybrid_policy, critic=classical_critic, policy_output="logits")
print("Hybrid output:", hybrid_policy(probe))'''),
md('''## Choose local or distributed rollouts

Each rollout worker receives an independent copy of the model. `NUM_ENV_RUNNERS=0` samples locally; `2` starts two rollout processes. Under Slurm, set `RAY_ADDRESS="auto"` to attach to an existing cluster. Distributed rollout support does not distribute a single quantum state with MPI.

RLlib trains its copies. The original objects remain usable as before; later cells show how to retrieve the trained weights.'''),
code('''SMOKE = os.environ.get("QASP_NOTEBOOK_SMOKE", "0") == "1"
NUM_ENV_RUNNERS = 0
RAY_ADDRESS = None
ITERATIONS = 2 if SMOKE else 500
EVAL_EVERY = 1 if SMOKE else 10
VALIDATION_EPISODES = 2 if SMOKE else 50
FINAL_EPISODES = 3 if SMOKE else 100
RUN_DIRECTORY = Path(tempfile.mkdtemp(prefix="qasp-rllib-", dir=START_DIRECTORY))
os.chdir(RUN_DIRECTORY)  # Ray workers inherit cwd; avoid the source package root.
if not ray.is_initialized():
    ray.init(address=RAY_ADDRESS, include_dashboard=False,
             **({} if RAY_ADDRESS else {"num_cpus": max(2, NUM_ENV_RUNNERS + 1)}))
print("Outputs:", RUN_DIRECTORY)
print("Mode:", "smoke check" if SMOKE else "full training")'''),
md('''## Native RLlib configurations

The integration is the `rl_module_spec=...` line. Everything else is normal RLlib configuration. PPO gets a default two-layer classical critic; DQN uses scalar action values and Double DQN with replay and target-network updates.

`ppo_module_spec` automatically recognizes `PQCSoftmax` logits and `PQCBornVariance` probabilities. It also handles legacy `PQCBorn` through its existing `action_probs` method. No input normalization is added by the adapter. This example opts into vectorized analytic circuit evaluation for speed. DQN uses a mean-square Bellman loss and soft target updates; these are native RLlib training settings.'''),
code('''ppo_config = (
    PPOConfig()
    .environment("CartPole-v1")
    .framework("torch")
    .rl_module(rl_module_spec=ppo_module_spec(policy, policy_batch_mode="vmap"))
    .env_runners(num_env_runners=NUM_ENV_RUNNERS, num_envs_per_env_runner=16,
                 rollout_fragment_length="auto", sample_timeout_s=300)
    .learners(num_learners=0, num_gpus_per_learner=0)
    .training(lr=1e-3, train_batch_size_per_learner=64 if SMOKE else 512,
              minibatch_size=32 if SMOKE else 128, num_epochs=1 if SMOKE else 5)
    .debugging(seed=7, log_level="ERROR")
)

dqn_config = (
    DQNConfig()
    .environment("CartPole-v1")
    .framework("torch")
    .rl_module(rl_module_spec=dqn_module_spec(q_model))
    .env_runners(num_env_runners=NUM_ENV_RUNNERS, num_envs_per_env_runner=16,
                 rollout_fragment_length=32, sample_timeout_s=300)
    .learners(num_learners=0, num_gpus_per_learner=0)
    .training(dueling=False, num_atoms=1, double_q=True, lr=1e-3, gamma=0.99,
              td_error_loss_fn="mse", tau=0.01,
              train_batch_size_per_learner=16 if SMOKE else 64,
              num_steps_sampled_before_learning_starts=16 if SMOKE else 1000,
              target_network_update_freq=32,
              epsilon=[[0, 1.0], [40000, 0.02]], n_step=3,
              training_intensity=4.0,
              store_buffer_in_checkpoints=True,
              replay_buffer_config={"type": "PrioritizedEpisodeReplayBuffer",
                                    "capacity": 1000 if SMOKE else 50000,
                                    "alpha": 0.6, "beta": 0.4})
    .reporting(min_time_s_per_iteration=0, min_sample_timesteps_per_iteration=64 if SMOKE else 512)
    .debugging(seed=7, log_level="ERROR")
)'''),
md('''## Evaluation and a bounded training loop

Validation uses seeds starting at 10000; final testing uses new seeds starting at 40000. Full algorithm checkpoints include optimizer state and, for this DQN configuration, replay contents. The helper below only organizes logging and evaluation; training is `algo.train()` from RLlib.'''),
code('''def evaluate(module, episodes, seed):
    env = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(episodes)])
    returns = np.zeros(episodes)
    finished = np.zeros(episodes, dtype=bool)
    try:
        # VectorEnv seeds sub-environment i with seed+i, matching serial evaluation.
        obs, _ = env.reset(seed=seed)
        while not finished.all():
            with torch.no_grad():
                out = module.forward_inference({Columns.OBS: torch.as_tensor(obs)})
                actions = out[Columns.ACTIONS] if Columns.ACTIONS in out else out[Columns.ACTION_DIST_INPUTS].argmax(-1)
            obs, reward, terminated, truncated, _ = env.step(actions.cpu().numpy())
            returns += reward * (~finished)
            finished |= terminated | truncated
    finally:
        env.close()
    return returns.tolist()


def train_and_evaluate(config, name):
    # Ray 2.60 storage keeps logs in this experiment's directory.
    from ray.train._internal.storage import StorageContext
    storage = StorageContext(storage_path=str(RUN_DIRECTORY), experiment_dir_name="cartpole", trial_dir_name=name)
    algo = config.algo_class(config=config, storage=storage)
    history = []
    started = time.monotonic()
    best_score = -float("inf")
    best_checkpoint = None
    try:
        for iteration in range(ITERATIONS):
            result = algo.train()
            progress = {"algorithm": name, "iteration": iteration + 1,
                        "steps": result["num_env_steps_sampled_lifetime"],
                        "training_return": result.get("env_runners", {}).get("episode_return_mean"),
                        "elapsed_seconds": time.monotonic() - started}
            with (RUN_DIRECTORY / f"{name}-progress.jsonl").open("a") as stream:
                stream.write(json.dumps(progress) + "\\n")
            if (iteration + 1) % EVAL_EVERY == 0 or iteration == ITERATIONS - 1:
                returns = evaluate(algo.get_module(), VALIDATION_EPISODES, 10000)
                record = {"iteration": iteration + 1,
                          "steps": result["num_env_steps_sampled_lifetime"],
                          "validation_return": float(np.mean(returns))}
                history.append(record)
                if record["validation_return"] > best_score:
                    best_score = record["validation_return"]
                    best_checkpoint = algo.save_to_path(str(RUN_DIRECTORY / f"{name}-best-checkpoint"))
                (RUN_DIRECTORY / f"{name}-validation.json").write_text(json.dumps(history, indent=2))
                print(name, record)
                # Stop using validation only; never use the held-out test to train
                # or select a checkpoint. The maximum budget remains bounded.
                if not SMOKE and best_score >= 495:
                    break
            if (iteration + 1) % 10 == 0:
                algo.save_to_path(str(RUN_DIRECTORY / f"{name}-latest-checkpoint"))
        # Select solely on validation episodes; test seeds are untouched until now.
        # Load the selected learner weights directly. A running algorithm's local
        # rollout copy can retain newer weights after restore_from_path().
        module = RLModule.from_checkpoint(str(Path(best_checkpoint) /
            "learner_group/learner/rl_module/default_policy"))
        final_returns = evaluate(module, FINAL_EPISODES, 40000)
        (RUN_DIRECTORY / f"{name}-test.json").write_text(json.dumps({"seed": 40000, "returns": final_returns}, indent=2))
        checkpoint = best_checkpoint
        trained_model = module.policy if name == "ppo" else module.q_network
        trained_weights = {k: v.detach().cpu().clone() for k, v in trained_model.state_dict().items()}
        torch.save(trained_weights, RUN_DIRECTORY / f"{name}-weights.pt")
        return history, final_returns, checkpoint
    finally:
        algo.stop()'''),
code('''ppo_history, ppo_returns, ppo_checkpoint = train_and_evaluate(ppo_config, "ppo")
dqn_history, dqn_returns, dqn_checkpoint = train_and_evaluate(dqn_config, "dqn")

# Wrapping and training did not mutate the original user-created models.
for key, value in policy.state_dict().items():
    torch.testing.assert_close(value, original_policy_state[key])
for key, value in q_model.state_dict().items():
    torch.testing.assert_close(value, original_q_state[key])
print("Original QASP objects remain unchanged.")'''),
md('''## Results

A short smoke run only checks that models, gradients, RLlib and checkpointing work together. To assess the CartPole target, use full mode and review the measured result. Extend the budget or tune the model and learning settings if the target is not reached.'''),
code('''import matplotlib.pyplot as plt
import pandas as pd

fig, ax = plt.subplots(figsize=(8, 4))
for name, history in [("PPO", ppo_history), ("DQN", dqn_history)]:
    ax.plot([row["steps"] for row in history], [row["validation_return"] for row in history], marker="o", label=name)
ax.axhline(475, color="gray", linestyle="--", label="475-return target")
ax.set(xlabel="Environment steps", ylabel="Mean validation return", title="Existing QASP models trained with RLlib")
ax.legend()
fig.tight_layout()
fig.savefig(RUN_DIRECTORY / "learning-curves.png", dpi=160)
plt.show()
report = pd.DataFrame([
    {"algorithm": name, "mean_return": np.mean(values), "episodes": len(values),
     "target_met": not SMOKE and np.mean(values) >= 475}
    for name, values in [("PPO", ppo_returns), ("DQN", dqn_returns)]
])
report.to_csv(RUN_DIRECTORY / "evaluation.csv", index=False)
display(report)'''),
md('''## Use the trained models through the usual QASP API

Weights can be loaded into an ordinary QASP model with PyTorch's standard `load_state_dict`. That model remains usable independently of RLlib. To continue RLlib training instead, restore the full algorithm checkpoint; it retains its original worker topology.'''),
code('''trained_policy = deepcopy(policy)
trained_policy.load_state_dict(torch.load(RUN_DIRECTORY / "ppo-weights.pt", weights_only=True))
trained_q_model = deepcopy(q_model)
trained_q_model.load_state_dict(torch.load(RUN_DIRECTORY / "dqn-weights.pt", weights_only=True))
print("Trained policy, normal QASP call:", trained_policy(probe))
print("Trained Q-values, normal QASP call:", trained_q_model(probe))
print("Algorithm checkpoints:", ppo_checkpoint, dqn_checkpoint)

# Continue training with RLlib:
# from ray.rllib.algorithms.algorithm import Algorithm
# resumed = Algorithm.from_checkpoint(ppo_checkpoint)
# try:
#     next_result = resumed.train()
# finally:
#     resumed.stop()'''),
md('''## Slurm and worker-local factories

`examples/slurm/rllib_cartpole.sbatch` runs the native RLlib example in `examples/rllib_cartpole.py` using two exclusive CPU nodes. Each node needs the same QASP installation including the optional module and Ray 2.60.0.

```bash
export QASP_PYTHON=/shared/envs/qasp-validation/bin/python
export QASP_EXAMPLE=/shared/QASP/examples/rllib_cartpole.py
export QASP_OUTPUT=/shared/results/cartpole
export QASP_ALGORITHM=ppo  # or dqn
sbatch --account=YOUR_ACCOUNT --partition=YOUR_PARTITION examples/slurm/rllib_cartpole.sbatch
```

Model instances are snapshotted with their current weights. If a simulator holds resources that cannot be serialized, pass a zero-argument factory that uses your **same normal QASP constructor**: `ppo_module_spec(make_policy)` or `dqn_module_spec(make_q_model)`. The factory builds the model inside each worker; RLlib synchronizes weights. MPI-backed circuits are not supported by these rollout adapters.

Local worker tests do not verify Slurm networking or multi-node performance.'''),
code('''ray.shutdown()
os.chdir(START_DIRECTORY)
print("Artifacts:", RUN_DIRECTORY)'''),
]
nb.metadata = {"kernelspec":{"display_name":"Python (qasp-validation)","language":"python","name":"qasp-validation"},"language_info":{"name":"python","version":"3.12"}}
path = Path(__file__).resolve().parents[2] / 'examples/cartpole_rllib.ipynb'
nbf.write(nb,path)
print(path)
