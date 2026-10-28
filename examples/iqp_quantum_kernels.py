"""Small, executable IQP quantum-kernel use cases.

Run from the repository root with::

    python examples/iqp_quantum_kernels.py
"""

import numpy as np
import torch

from qasp import KernelTargetAlignment, create_kernel, target_alignment


X_TRAIN = np.array(
    [
        [0.10, 0.20, 0.15],
        [0.20, 0.10, 0.12],
        [0.90, 1.10, 1.00],
        [1.00, 0.80, 0.92],
    ]
)
Y_TRAIN = np.array([0, 0, 1, 1])
X_TEST = np.array([[0.15, 0.25, 0.18], [0.95, 0.85, 0.90]])
N_QUBITS = 3


def iqp_config(kernel_type, init="zeros"):
    return {
        "kernel_type": kernel_type,
        "feature_map": "AngleEmbedding",
        "feature_map_config": {
            "rotations": "Y",
            "imprimitive_pattern": "ring",
        },
        "ansatz": "IQPAnsatz",
        "ansatz_config": {
            "n_layers": 2,
            "imprimitive_pattern": "ring",
            "reuploading": True,
        },
        "init": init,
        "device": "default.qubit",
    }


def fidelity_use_case():
    kernel = create_kernel(iqp_config("fidelity"), n_qubits=N_QUBITS)
    train_gram = kernel.gram(X_TRAIN)
    test_gram = kernel.gram(X_TEST, X_TRAIN)
    print("IQP fidelity train Gram:\n", np.round(train_gram, 3))
    print("IQP fidelity test-vs-train shape:", test_gram.shape)


def localized_fidelity_use_case():
    fidelity = create_kernel(iqp_config("fidelity"), n_qubits=N_QUBITS)
    config = iqp_config("fidelity")
    config["projectors"] = ["P 0 1", "P 0 2", " P 0 1 2"]
    local = create_kernel(config, n_qubits=N_QUBITS)

    local_gram = local.gram(X_TRAIN)
    print("Local-projector subsets:", local.projector_wires)
    print("IQP local-projector Gram:\n", np.round(local_gram, 3))

    global_config = iqp_config("fidelity")
    global_config["projectors"] = ["P 0 1 2"]
    global_projector = create_kernel(global_config, n_qubits=N_QUBITS)
    error = np.max(
        np.abs(global_projector.gram(X_TRAIN) - fidelity.gram(X_TRAIN))
    )
    print("All-qubit projector/global fidelity maximum error:", f"{error:.3e}")


def projected_use_case():
    config = iqp_config("projected")
    config.update(
        {
            "observables": [
                "X 0", "Y 0", "Z 0",
                "X 1", "Y 1", "Z 1",
                "X 2", "Y 2", "Z 2",
            ],
            "gamma": 0.5,
        }
    )
    kernel = create_kernel(config, n_qubits=N_QUBITS)
    print("IQP projected Gram:\n", np.round(kernel.gram(X_TRAIN), 3))


def target_alignment_use_case():
    torch.manual_seed(3)
    kernel = create_kernel(
        iqp_config("fidelity", init="small_normal"), n_qubits=N_QUBITS
    )
    before = target_alignment(kernel.gram(X_TRAIN), Y_TRAIN)
    history = KernelTargetAlignment(kernel).fit(
        X_TRAIN, Y_TRAIN, steps=5, lr=0.02, seed=3, verbose=False
    )
    after = target_alignment(kernel.gram(X_TRAIN), Y_TRAIN)
    print(f"IQP target alignment: {before:.4f} -> {after:.4f}")
    print("Optimization history:", np.round(history["alignment"], 4))


if __name__ == "__main__":
    fidelity_use_case()
    localized_fidelity_use_case()
    projected_use_case()
    target_alignment_use_case()
