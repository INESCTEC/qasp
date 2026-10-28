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
        "rotations" : "ZY",#["Y {}".format(i) for i in range(n_qubits)]+["Z {}".format(i) for i in range(n_qubits)]+["Y {}".format(i) for i in range(n_qubits)],
        "reuploading": False,
        "include_imprimitive": True,
        "imprimitive": "CZ",
        "imprimitive_pattern": "all_to_all",
        "include_final_imprimitive": False,
        "include_feature_map_first": False,
        "include_hadamards_initial": True
    }
    globalz = "Z"*n_qubits+" {}".format(" ".join(str(wire) for wire in list(range(n_qubits))))

    observables = []
    #add all 1-local Z observables
    for i in range(n_qubits):
        observables.append("Z {}".format(i))
    #add all 2-local ZZ observables
    for i in range(n_qubits-1):
            observables.append("ZZ {} {}".format(i, i+1))
    #add chain ZZZ observables
    #for i in range(n_qubits-2):
    #    observables.append("ZZZ {} {} {}".format(i, i+1, i+2))

    #add ZZZ on 0,1,2 and ZZZ on 3,4,5
    #observables.append("ZZZ 0 1 2")
    #observables.append("ZZZ 3 4 5")

    observables.append(globalz)


    policy_model_config_softmax = {
    "ansatz": "AngleAnsatz",
        "ansatz_config": ansatz_config,
        "feature_map": "NeuralEmbedding",
        "feature_map_config": fm_config,
        "n_actions": 2,
        "init": "normal_0_1",
        "device": "lightning.kokkos",
        "diff_method": "adjoint",
        #"spsa_num_directions": 10,
        "softmax_activation": False,
        "fc_layer":True,
        #"shots": 5000,
        "observables": observables
    }

    policy = create_model("PQCSoftmax", policy_model_config_softmax, n_qubits=n_qubits)

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
        "run_id": run_id,
        "final_return": float(np.mean(returns[-100:])),
        "returns": list(map(float, returns)),
    }

    out_name = f"results_Z_ZZ_linear_globalz_softmax_q{n_qubits}_run{run_id}.json"
    with open(out_name, "w") as f:
        json.dump(result, f)

    print(f"[INFO] Saved {out_name}", flush=True)


if __name__ == "__main__":
    main()