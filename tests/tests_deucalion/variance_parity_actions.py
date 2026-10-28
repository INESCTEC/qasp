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
    n_actions = int(os.environ["N_ACTIONS"])
    print(f"[INFO] Running variance test for n_actions = {n_actions}", flush=True)

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
        "n_actions": n_actions,
        "init": "normal_0_1",
        "policy": "parity-like",
        "device": "lightning.kokkos",
        "diff_method": "best",
        "softmax_activation": False,
        "output_scaling": None,
        "training_method": "expectation",
        "torchrl": False,
        "measurement_qubits": lambda n: [i for i in range(n)],
        "parity_cnot_decomposition": False,
    }

    variance_qubit = policy_variance(
        policy_class="PQCBorn",
        policy_config_kwargs=policy_model_config,
        n_samples=2000,
        qubit_list=[4, 6, 8, 9, 10, 12, 14, 16],
        parallel=True,
    )

    result = {
        "n_actions": n_actions,
        "qubit_list": [4, 6, 8, 9, 10, 12, 14, 16],
        "variance_qubit": variance_qubit,
    }

    out_name = f"variance_actions_{n_actions}.json"
    with open(out_name, "w") as f:
        json.dump(result, f)

    print(f"[INFO] Saved {out_name}", flush=True)


if __name__ == "__main__":
    main()