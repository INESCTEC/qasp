import os
import json
import random
import numpy as np
import torch


import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

sys.path.insert(0, PROJECT_ROOT)

from qasp import *
from qasp.rl.torchrl import ContextualBanditEnv


import qasp


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_observables(n_qubits: int, observable_mode: str):
    if observable_mode == "Z":
        return [f"Z {i}" for i in range(n_qubits)]

    if observable_mode == "Z_ZZ_linear":
        observables = [f"Z {i}" for i in range(n_qubits)]
        observables += [f"ZZ {i} {i+1}" for i in range(n_qubits - 1)]
        return observables

    if observable_mode == "Z_ZZ_all":
        observables = [f"Z {i}" for i in range(n_qubits)]
        for i in range(n_qubits - 1):
            for j in range(i + 1, n_qubits):
                observables.append(f"ZZ {i} {j}")
        return observables

    raise ValueError(f"Unknown observable_mode: {observable_mode}")


def main():
    env_path = os.environ.get("BANDIT_ENV_PATH", "bandit_env_6f_20a_100c.npz")
    observable_mode = os.environ["OBS_MODE"]   # Z, Z_ZZ_linear, Z_ZZ_all
    run_id = int(os.environ["RUN_ID"])
    base_seed = int(os.environ.get("BASE_SEED", "1234"))
    seed = base_seed + 1000 * run_id + hash(observable_mode) % 1000

    set_seed(seed)

    # Load shared benchmark environment
    env = ContextualBanditEnv.load(env_path)

    n_qubits = 6

    fm_config = {
        "features": 6,
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

    observables = build_observables(n_qubits, observable_mode)

    policy_model_config_softmax = {
        "ansatz": "AngleAnsatz",
        "ansatz_config": ansatz_config,
        "feature_map": "NeuralEmbedding",
        "feature_map_config": fm_config,
        "n_actions": 20,
        "init": "normal_0_1",
        "device": "lightning.kokkos",
        "diff_method": "adjoint",
        "fc_layer": True,
        "observables": observables,
    }

    policy = create_model("PQCSoftmax", policy_model_config_softmax, n_qubits=n_qubits)

    agent = REINFORCE(
        policy=policy,
        env_name=env,
        max_steps=1,
        gamma=1.0,
        lr=1e-2,
        batch_episodes=32,
        logits=True,              # PQCSoftmax usually returns probabilities
        normalize_mode=None,       # contexts are already in [-1, 1]
        baseline_type="ema",
        normalize_returns=True,
        entropy_coef=0.01,
        max_grad_norm=1.0,
        fc_layer=True,
    )

    returns = agent.train(num_episodes=5000, log_every=50)

    result = {
        "experiment": "bandit_softmax",
        "observable_mode": observable_mode,
        "n_qubits": n_qubits,
        "run_id": run_id,
        "seed": seed,
        "env_path": env_path,
        "oracle_value": float(env.get_oracle_value()),
        "reward_bounds": list(map(float, env.get_reward_bounds())),
        "final_return_mean_last200": float(np.mean(returns[-200:])),
        "returns": list(map(float, returns)),
    }

    out_name = f"results_bandit_softmax_{observable_mode}_q{n_qubits}_run{run_id}.json"
    with open(out_name, "w") as f:
        json.dump(result, f)

    print(f"[INFO] Saved {out_name}", flush=True)


if __name__ == "__main__":
    main()