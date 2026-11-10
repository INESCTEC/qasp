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

import qasp


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    # Read parameters from environment
    n_qubits = int(os.environ["N_QUBITS"])
    run_id = int(os.environ["RUN_ID"])
    #base_seed = int(os.environ.get("BASE_SEED", "1234"))
    #seed = base_seed + 1000 * n_qubits + run_id

    #set_seed(seed)

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

    policy_model_config = {
        "ansatz": "AngleAnsatz",
        "ansatz_config": ansatz_config,
        "feature_map": "NeuralEmbedding",
        "feature_map_config": fm_config,
        "n_actions": 2,
        "init": "normal_0_1",
        "policy": "parity-like",
        "device": "lightning.kokkos",
        "diff_method": "best",
        "softmax_activation": False,
        "output_scaling": None,
        "training_method": "expectation",
        "torchrl": False,
        "parity_cnot_decomposition": False,
    }

    print(f"[INFO] n_qubits={n_qubits}, run_id={run_id}", flush=True)

    policy = create_policy("PQCBorn", policy_model_config, n_qubits=n_qubits)

    agent = REINFORCE(
        policy=policy,
        env_name="CartPole-v1",
        gamma=0.99,
        lr=1e-1,
        batch_episodes=5,
        logits=False,
        normalize_mode="clip",
        normalize_returns=True,
        max_steps=200,
        baseline_type="ema",
        entropy_coef=0.0,
        max_grad_norm=None,
    )

    returns = agent.train(num_episodes=500)

    result = {
        "n_qubits": n_qubits,
        "run_id": run_id,
        "final_return": float(np.mean(returns[-100:])),
        "returns": list(map(float, returns)),
    }

    out_name = f"results_nv1_parity_q{n_qubits}_run{run_id}.json"
    with open(out_name, "w") as f:
        json.dump(result, f)

    print(f"[INFO] Saved {out_name}", flush=True)


if __name__ == "__main__":
    main()