"""Run direct QASP models with one shared differentiable PP device configuration.

Run from the repository root: python examples/pauli_propagation_modules.py
Uses analytic, exact propagation on small noncommuting circuits. For training,
use torchlane (or an environment with QASP's optional RL dependencies installed).
"""
from copy import deepcopy
import json

import numpy as np  # Load the numerical runtime before Torch in torchlane.
import torch

from qasp import AngleAnsatz, NeuralEmbedding
from qasp.rl.policies import PQCSoftmax
from qasp.rl import PQDQN
from qasp.classifiers import create_classifier
from qasp.qkernels import create_kernel
from qasp.genlearn import create_qcbm
from qasp.optimization import QAOA, PCE, QUBO


EXECUTION = {
    "device": "PauliPropagationSimulator",
    "device_kwargs": {"k": None, "prune": False},
    "diff_method": "backprop",
    "shots": None,
}
ANSATZ = {
    "n_layers": 2,
    "rotations": ["Y 0", "Z 1", "X 2", "XYZ 0 1 2"],
    "include_imprimitive": True, "imprimitive": "CNOT",
    "imprimitive_pattern": "chain", "reuploading": True,
}


def run_examples():
    torch.manual_seed(2026)
    # Direct policy import, configuration dictionary, neural input encoding.
    policy = PQCSoftmax(
        n_qubits=3, n_actions=2,
        ansatz=AngleAnsatz, ansatz_config={"n_qubits": 3, **deepcopy(ANSATZ)},
        feature_map=NeuralEmbedding,
        feature_map_config={"n_qubits": 3, "features": 2, "rotations": "Y"},
        observables=["Z 0", "XZ 1 2"], fc_layer=None, **deepcopy(EXECUTION),
    ).double()
    x = torch.tensor(np.array([.2, -.4]), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=.01)
    optimizer.zero_grad()
    policy_loss = -torch.log_softmax(policy(x), dim=-1)[1]
    policy_loss.backward()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in policy.parameters())
    optimizer.step()

    # The same fields work in value-based RL, with sequential neural encoding.
    qfunction = PQDQN(
        n_qubits=3, n_actions=2, ansatz="AngleAnsatz", ansatz_config=deepcopy(ANSATZ),
        feature_map="NeuralEmbedding", feature_map_config={"features": 2, "rotations": "Y"},
        observables="majority", preset_actions=2, batch_mode="sequential", **deepcopy(EXECUTION),
    ).double()
    qvalues = qfunction(x.detach().unsqueeze(0))
    qvalues.square().mean().backward()

    classifier = create_classifier({
        "ansatz": "AngleAnsatz", "ansatz_config": deepcopy(ANSATZ),
        "feature_map": "AngleEmbedding", "feature_map_config": {"features": 3, "rotations": "Y"},
        "readout": "probabilities", **deepcopy(EXECUTION),
    }, n_qubits=3, n_classes=2).double()
    features = np.array([[.2, .3, .1], [-.4, .1, .6]])
    scores = classifier(torch.tensor(features, dtype=torch.float64))
    scores.square().sum().backward()

    kernel = create_kernel({
        "kernel_type": "fidelity", "ansatz": "AngleAnsatz", "ansatz_config": deepcopy(ANSATZ),
        "feature_map": "AngleEmbedding", "feature_map_config": {"rotations": "Y"},
        **deepcopy(EXECUTION),
    }, n_qubits=3)
    gram = kernel.gram(features, return_tensor=True)
    gram.sum().backward()

    born = create_qcbm({
        "ansatz": "AngleAnsatz", "ansatz_config": deepcopy(ANSATZ),
        "loss": "mmd", "mmd_config": {"mode": "exact"}, **deepcopy(EXECUTION),
    }, n_qubits=3, data=[[0, 0, 0], [1, 1, 1]])
    born_loss = born.loss()
    born_loss.backward()  # loss="nll" also works, using dense probabilities.

    problem = QUBO.maxcut(3, [(0, 1), (1, 2), (0, 2)])
    qaoa = QAOA(problem, n_layers=2, backend="torch", seed=2026, **deepcopy(EXECUTION))
    result = qaoa.solve(optimizer="Adam", optimizer_options={"maxiter": 2, "lr": .01})
    pce = PCE(problem, backend="torch", seed=2026, **deepcopy(EXECUTION))
    pce_result = pce.solve(optimizer="Adam", optimizer_options={"maxiter": 2, "lr": .01})
    return {
        "policy_loss": float(policy_loss.detach()), "qvalues": qvalues.detach().tolist(),
        "classifier_scores": scores.detach().tolist(), "kernel_gram": gram.detach().tolist(),
        "mmd_loss": float(born_loss.detach()), "qaoa_energy": float(result.energy),
        "pce_objective": float(pce_result.objective),
        "pce_raw_energy": float(pce_result.metadata["raw_energy"]),
        "pce_improved_energy": float(pce_result.energy),
    }


if __name__ == "__main__":
    print(json.dumps(run_examples(), indent=2))
