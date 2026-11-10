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

import qasp


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def z_string(wires):
    """
    Convert a list of wires into QASP observable string format.

    Example:
        wires = [0, 1, 3]
        returns "ZZZ 0 1 3"
    """
    if len(wires) == 0:
        raise ValueError("Empty wire set produced an identity observable; not allowed here.")
    return "Z" * len(wires) + " " + " ".join(str(w) for w in wires)


def xor_product_wires(list_of_wire_sets):
    """
    Product of Pauli-Z strings.

    Since Z_i * Z_i = I, repeated wires cancel mod 2.
    This returns the symmetric difference of all wire sets.
    """
    counts = {}
    for wires in list_of_wire_sets:
        for w in wires:
            counts[w] = counts.get(w, 0) + 1

    product_wires = [w for w, c in counts.items() if c % 2 == 1]
    return sorted(product_wires)


def build_parity_observables(n_qubits, n_action_bits):
    """
    Build all nontrivial parity-product observables for |A| = 2^n_action_bits.

    Requires: n_action_bits <= n_qubits for this construction.
    """
    if n_action_bits > n_qubits:
        raise ValueError(
            f"Invalid configuration: n_action_bits={n_action_bits} > n_qubits={n_qubits} "
            "for the current parity observable construction."
        )

    base_labels = [chr(ord("A") + i) for i in range(n_action_bits)]

    # A has n wires, B has n-1 wires, C has n-2 wires, ...
    base_wires = [list(range(n_qubits - i)) for i in range(n_action_bits)]

    observable_labels = []
    observable_wires = []
    observables = []
    subsets = []

    for r in range(1, n_action_bits + 1):
        for subset in combinations(range(n_action_bits), r):
            label = "".join(base_labels[i] for i in subset)
            wires = xor_product_wires([base_wires[i] for i in subset])

            # Skip identity if it appears
            if len(wires) == 0:
                continue

            observable_labels.append(label)
            observable_wires.append(wires)
            observables.append(z_string(wires))
            subsets.append(subset)

    return observable_labels, observable_wires, observables, subsets


def main():
    # Read parameters from environment
    n_qubits = int(os.environ["N_QUBITS"])
    n_action_bits = int(os.environ["N_ACTION_BITS"])
    run_id = int(os.environ["RUN_ID"])
    #base_seed = int(os.environ.get("BASE_SEED", "1234"))

    #seed = base_seed + 10000 * n_qubits + 100 * n_action_bits + run_id
    #set_seed(seed)

    n_actions = 2

    observable_labels, observable_wires, observables_parity, subsets = build_parity_observables(
        n_qubits=n_qubits,
        n_action_bits=n_action_bits,
    )

    print(f"[INFO] n_qubits={n_qubits}, n_action_bits={n_action_bits}, n_actions={n_actions}, run_id={run_id}", flush=True)
    print(f"[INFO] observables_parity={observables_parity}", flush=True)

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

    policy_model_config_softmax = {
        "ansatz": "AngleAnsatz",
        "ansatz_config": ansatz_config,
        "feature_map": "NeuralEmbedding",
        "feature_map_config": fm_config,
        "n_actions": n_actions,
        "init": "normal_0_1",
        "device": "lightning.kokkos",
        "diff_method": "adjoint",
        "softmax_activation": False,
        "fc_layer": True,
        "observables": observables_parity,
    }

    policy = create_policy("PQCSoftmax", policy_model_config_softmax, n_qubits=n_qubits)

    agent = REINFORCE(
        policy=policy,
        env_name="CartPole-v1",
        gamma=0.99,
        lr=1e-1,
        batch_episodes=5,
        logits=True,
        normalize_mode="clip",
        normalize_returns=True,
        max_steps=200,
        baseline_type="ema",
        entropy_coef=0.0,
        max_grad_norm=None,
        fc_layer=True,

    )

    returns = agent.train(num_episodes=500)

    result = {
        "n_qubits": n_qubits,
        "n_action_bits": n_action_bits,
        "n_actions": n_actions,
        "run_id": run_id,
        "observables_parity": observables_parity,
        "final_return_last100": float(np.mean(returns[-100:])),
        "returns": list(map(float, returns)),
    }

    out_name = f"results_v2_softmax_parity_q{n_qubits}_abits{n_action_bits}_run{run_id}.json"
    with open(out_name, "w") as f:
        json.dump(result, f)

    print(f"[INFO] Saved {out_name}", flush=True)


if __name__ == "__main__":
    main()