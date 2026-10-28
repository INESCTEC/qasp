import os
import json
import random
import numpy as np
import torch

from itertools import combinations

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


def z_string(wires):
    return "Z" * len(wires) + " " + " ".join(str(w) for w in wires)


def xor_product_wires(list_of_wire_sets):
    counts = {}
    for wires in list_of_wire_sets:
        for w in wires:
            counts[w] = counts.get(w, 0) + 1
    product_wires = [w for w, c in counts.items() if c % 2 == 1]
    return sorted(product_wires)


def build_parity_observables(n_qubits, n_action_bits):
    base_labels = [chr(ord("A") + i) for i in range(n_action_bits)]

    base_wires = [
        list(range(n_qubits - i))
        for i in range(n_action_bits)
    ]

    observable_labels = []
    observable_wires = []
    observables = []
    subsets = []

    for r in range(1, n_action_bits + 1):
        for subset in combinations(range(n_action_bits), r):
            label = "".join(base_labels[i] for i in subset)
            wires = xor_product_wires([base_wires[i] for i in subset])

            if len(wires) == 0:
                continue

            observable_labels.append(label)
            observable_wires.append(wires)
            observables.append(z_string(wires))
            subsets.append(subset)

    return observable_labels, observable_wires, observables, subsets


def main():
    env_path = os.environ.get("BANDIT_ENV_PATH", "bandit_env_6f_20a_100c.npz")
    n_action_bits = int(os.environ["N_ACTION_BITS"])
    run_id = int(os.environ["RUN_ID"])

    base_seed = int(os.environ.get("BASE_SEED", "1234"))
    seed = base_seed + 1000 * n_action_bits + run_id

    set_seed(seed)

    env = ContextualBanditEnv.load(env_path)

    n_qubits = 6

    observable_labels, observable_wires, observables_parity, subsets = build_parity_observables(
        n_qubits=n_qubits,
        n_action_bits=n_action_bits,
    )

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
        "observables": observables_parity,
    }

    policy = create_model("PQCSoftmax", policy_model_config_softmax, n_qubits=n_qubits)

    agent = REINFORCE(
        policy=policy,
        env_name=env,
        max_steps=1,
        gamma=1.0,
        lr=1e-2,
        batch_episodes=32,
        logits=True,
        normalize_mode=None,
        baseline_type="ema",
        normalize_returns=True,
        entropy_coef=0.01,
        max_grad_norm=1.0,
        fc_layer=True,
    )

    returns = agent.train(num_episodes=5000, log_every=50)

    result = {
        "experiment": "bandit_softmax_parity",
        "n_qubits": n_qubits,
        "n_action_bits": n_action_bits,
        "run_id": run_id,
        "seed": seed,
        "env_path": env_path,
        "oracle_value": float(env.get_oracle_value()),
        "reward_bounds": list(map(float, env.get_reward_bounds())),
        "final_return_mean_last200": float(np.mean(returns[-200:])),
        "returns": list(map(float, returns)),
    }

    out_name = f"results_bandit_softmax_parity_q{n_qubits}_abits{n_action_bits}_run{run_id}.json"
    with open(out_name, "w") as f:
        json.dump(result, f)

    print(f"[INFO] Saved {out_name}", flush=True)


if __name__ == "__main__":
    main()