"""Use ordinary QASP models with native RLlib configurations (local or Slurm).

Example: python -I examples/rllib_cartpole.py --algorithm ppo --workers 2 --smoke
The installed QASP must include the optional qasp.rl.rllib module; for development
against PyPI 0.3.0, see tests/rllib/prepare_environment.py.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile

import gymnasium as gym
import numpy as np
import ray
import torch
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.core.columns import Columns

from qasp.circuits import AngleAnsatz
from qasp.feature_maps import AngleEmbedding
from qasp.rl import PQCSoftmax, PQCBornVariance, PQDQN
from qasp.rl.rllib import ppo_module_spec, dqn_module_spec


def make_models(policy_kind="softmax"):
    # Normal QASP constructor arguments. No RLlib-specific model schema.
    common = dict(
        n_qubits=4, n_actions=2,
        ansatz=AngleAnsatz,
        ansatz_config=dict(n_qubits=4, n_layers=2, rotations="YZ", reuploading=True),
        feature_map=AngleEmbedding,
        feature_map_config=dict(n_qubits=4, features=4, rotations="Y", input_scaling=True),
        device="default.qubit", diff_method="backprop",
    )
    if policy_kind == "born":
        policy = PQCBornVariance(**common, policy="parity-like", measurement_qubits=[0, 1, 2, 3])
    else:
        policy = PQCSoftmax(**common, observables=["Z 0", "Z 1", "Z 2", "Z 3"], fc_layer=True)
    q_model = PQDQN(**common, observables=["Z 0", "Z 1", "Z 2", "Z 3"], fc_layer=True)
    return policy, q_model


def make_config(algorithm, *, workers=0, smoke=False, policy_kind="softmax", seed=7):
    torch.manual_seed(seed)
    policy, q_model = make_models(policy_kind)
    if algorithm == "ppo":
        config = (PPOConfig()
                  .rl_module(rl_module_spec=ppo_module_spec(policy))
                  .training(lr=3e-4, train_batch_size_per_learner=64 if smoke else 512,
                            minibatch_size=32 if smoke else 128,
                            num_epochs=1 if smoke else 5))
    else:
        config = (DQNConfig()
                  .rl_module(rl_module_spec=dqn_module_spec(q_model))
                  .training(dueling=False, num_atoms=1, double_q=True, lr=1e-3,
                            n_step=1, train_batch_size_per_learner=16 if smoke else 64,
                            num_steps_sampled_before_learning_starts=16 if smoke else 1000,
                            target_network_update_freq=32 if smoke else 500,
                            epsilon=[[0, 1.0], [20000, 0.02]],
                            store_buffer_in_checkpoints=True,
                            replay_buffer_config={"type": "PrioritizedEpisodeReplayBuffer",
                                                  "capacity": 1000 if smoke else 50000,
                                                  "alpha": .6, "beta": .4}))
    return (config.environment("CartPole-v1").framework("torch")
            .env_runners(num_env_runners=workers, num_cpus_per_env_runner=1,
                         rollout_fragment_length="auto" if algorithm == "ppo" else 32,
                         sample_timeout_s=300)
            .learners(num_learners=0, num_gpus_per_learner=0)
            .resources(num_cpus_for_main_process=1)
            .debugging(seed=seed, log_level="ERROR")
            .reporting(min_time_s_per_iteration=0,
                       min_sample_timesteps_per_iteration=64 if smoke else 512))


def evaluate(module, episodes=10, seed=10000):
    env = gym.make("CartPole-v1")
    returns = []
    try:
        for episode in range(episodes):
            obs, _ = env.reset(seed=seed + episode)
            total = 0.0
            while True:
                with torch.no_grad():
                    out = module.forward_inference({Columns.OBS: torch.as_tensor(obs).unsqueeze(0)})
                    action = out[Columns.ACTIONS] if Columns.ACTIONS in out else out[Columns.ACTION_DIST_INPUTS].argmax(-1)
                obs, reward, terminated, truncated, _ = env.step(int(action.item()))
                total += reward
                if terminated or truncated:
                    break
            returns.append(total)
    finally:
        env.close()
    return {"mean": float(np.mean(returns)), "returns": returns}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=["ppo", "dqn"], default="ppo")
    parser.add_argument("--policy", choices=["softmax", "born"], default="softmax")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--address", default=None)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.workers < 0 or args.iterations < 1:
        parser.error("workers must be nonnegative and iterations must be positive")
    output = (args.output or Path(tempfile.mkdtemp(prefix="qasp-cartpole-"))).resolve()
    checkpoint = args.resume.resolve() if args.resume else None
    output.mkdir(parents=True, exist_ok=True)
    # Ray prepends cwd to workers' imports. Keep the source checkout from
    # shadowing the installed QASP on those workers.
    os.chdir(output)
    torch.set_num_threads(1)
    ray.init(address=args.address, include_dashboard=False,
             **({} if args.address else {"num_cpus": max(2, args.workers + 1)}))
    algo = None
    try:
        if checkpoint:
            from ray.rllib.algorithms.algorithm import Algorithm
            algo = Algorithm.from_checkpoint(str(checkpoint))
        else:
            # Ray 2.60 storage replaces its older logger_creator argument.
            from ray.train._internal.storage import StorageContext
            storage = StorageContext(storage_path=str(output), experiment_dir_name="cartpole", trial_dir_name=args.algorithm)
            config = make_config(args.algorithm, workers=args.workers, smoke=args.smoke, policy_kind=args.policy)
            algo = config.algo_class(config=config, storage=storage)
        for _ in range(args.iterations):
            result = algo.train()
            record = {"iteration": result["training_iteration"],
                      "env_steps": result["num_env_steps_sampled_lifetime"],
                      "training_return": result.get("env_runners", {}).get("episode_return_mean")}
            if record["training_return"] is not None:
                record["training_return"] = float(record["training_return"])
            print(json.dumps(record), flush=True)
            with (output / "history.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
        saved = algo.save_to_path(str(output / f"checkpoint-{algo.iteration:06d}"))
        final = evaluate(algo.get_module(), episodes=2 if args.smoke else 100, seed=20000)
        (output / "evaluation.json").write_text(json.dumps(final, indent=2))
        print("Evaluation:", final, "Checkpoint:", saved, flush=True)
    finally:
        if algo is not None:
            algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
