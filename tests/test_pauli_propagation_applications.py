"""Public QASP APIs on the shared PP device, compared with default.qubit."""
from copy import deepcopy
import numpy as np
import pennylane as qml
import pytest
import torch

from qasp import CircuitBuilder, AngleAnsatz
from qasp.feature_maps import AngleEmbedding
from qasp.rl.model import create_policy
from qasp.rl import PQDQN
from qasp.classifiers import PQCClassifier
from qasp.qkernels import create_kernel
from qasp.genlearn import create_qcbm
from qasp.optimization import QAOA, PCE, QUBO

DEVICE = "PauliPropagationSimulator"
ANSATZ = {"n_layers": 2, "rotations": ["Y 0", "Z 1", "X 2", "XYZ 0 1 2"],
          "include_imprimitive": True, "imprimitive": "CNOT", "imprimitive_pattern": "chain", "reuploading": True}


def assert_grads(a, b, parameters_a, parameters_b, atol=2e-9):
    torch.testing.assert_close(a, b, atol=atol, rtol=atol)
    weights = torch.arange(1, a.numel() + 1, dtype=a.dtype).reshape(a.shape)
    ga = torch.autograd.grad((a * weights).sum(), parameters_a)
    gb = torch.autograd.grad((b * weights).sum(), parameters_b)
    for u, v in zip(ga, gb): torch.testing.assert_close(u, v, atol=atol, rtol=atol)


@pytest.mark.parametrize("policy", ["PQCSoftmax", "PQCBorn"])
def test_neural_policies_adam_updates_match_statevector(policy):
    cfg = {"ansatz": "AngleAnsatz", "ansatz_config": deepcopy(ANSATZ),
           "feature_map": "NeuralEmbedding", "feature_map_config": {"features": 2, "rotations": "Y"},
           "n_actions": 2, "device": DEVICE, "diff_method": "backprop", "shots": None}
    if policy == "PQCSoftmax": cfg.update(observables=["Z 0", "XZ 1 2"], fc_layer=None)
    else: cfg.update(policy="parity-like", training_method="probs", parity_cnot_decomposition=False)
    torch.manual_seed(21)
    a = create_policy(policy, deepcopy(cfg), n_qubits=3).double()
    b = create_policy(policy, {**deepcopy(cfg), "device": "default.qubit"}, n_qubits=3).double()
    b.load_state_dict(a.state_dict())
    optimizers = [torch.optim.Adam(m.parameters(), lr=.01) for m in (a, b)]
    before = [p.detach().clone() for p in a.parameters()]
    for _ in range(2):
        values, grads = [], []
        for model, optimizer in zip((a, b), optimizers):
            x = torch.tensor([.2, -.4], dtype=torch.float64, requires_grad=True)
            optimizer.zero_grad()
            value = model(x)
            loss = -torch.log_softmax(value, -1)[0] if policy == "PQCSoftmax" else -value[0].log()
            loss.backward()
            values.append(value.detach())
            grads.append([p.grad.clone() for p in model.parameters()] + [x.grad.clone()])
            optimizer.step()
        torch.testing.assert_close(*values, atol=2e-9, rtol=2e-9)
        for u, v in zip(*grads): torch.testing.assert_close(u, v, atol=2e-8, rtol=2e-8)
    assert any(not torch.equal(p, q) for p, q in zip(a.parameters(), before))
    for u, v in zip(a.parameters(), b.parameters()): torch.testing.assert_close(u, v, atol=2e-8, rtol=2e-8)


@pytest.mark.parametrize("batch_mode", ["broadcast", "sequential"])
def test_pqdqn_batch_majority_features(batch_mode):
    cfg = dict(n_qubits=3, n_actions=2, ansatz="AngleAnsatz", ansatz_config=deepcopy(ANSATZ),
               feature_map="NeuralEmbedding", feature_map_config={"features": 2, "rotations": "Y"},
               observables="majority", preset_actions=2, batch_mode=batch_mode, diff_method="backprop")
    if batch_mode == "broadcast":
        cfg.update(feature_map="AngleEmbedding", feature_map_config={"features": 3, "rotations": "Y", "input_scaling": True})
    a = PQDQN(**cfg, device=DEVICE).double()
    b = PQDQN(**cfg, device="default.qubit").double()
    b.load_state_dict(a.state_dict())
    x = torch.tensor([[[.2, -.4], [-.1, .3]]], dtype=torch.float64, requires_grad=True)
    if batch_mode == "broadcast":
        x = torch.tensor([[[.2, -.4, .1], [-.1, .3, .2]]], dtype=torch.float64, requires_grad=True)
    assert_grads(a(x), b(x), (*a.parameters(), x), (*b.parameters(), x))


def builder():
    b = CircuitBuilder(3)
    b.set_ansatz(AngleAnsatz(3, **deepcopy(ANSATZ), feature_map=AngleEmbedding(3, rotations="Y", features=3)))
    return b


@pytest.mark.parametrize("readout", ["expectation", "probabilities"])
def test_classifier_batch_values_and_gradients(readout):
    cfg = dict(n_qubits=3, n_classes=2, readout=readout, init="normal_0_1", diff_method="backprop")
    if readout == "expectation": cfg["observables"] = ["Z 0", "XY 1 2"]
    a = PQCClassifier(circuit_builder=builder(), **cfg, device=DEVICE).double()
    b = PQCClassifier(circuit_builder=builder(), **cfg, device="default.qubit").double()
    b.load_state_dict(a.state_dict())
    x = torch.tensor([[.2, .3, .7], [-.5, .2, .1]], dtype=torch.float64, requires_grad=True)
    assert_grads(a(x), b(x), (*a.parameters(), x), (*b.parameters(), x))


@pytest.mark.parametrize("kind", ["projected", "fidelity"])
def test_kernel_gram_and_parameter_gradients(kind):
    cfg = {"kernel_type": kind, "feature_map": "AngleEmbedding", "feature_map_config": {"rotations": "Y"},
           "ansatz": "AngleAnsatz", "ansatz_config": deepcopy(ANSATZ),
           "init": "normal_0_1", "diff_method": "backprop"}
    a = create_kernel({**deepcopy(cfg), "device": DEVICE}, n_qubits=3)
    b = create_kernel({**deepcopy(cfg), "device": "default.qubit"}, n_qubits=3)
    b.qlayer.load_state_dict(a.qlayer.state_dict())
    x = np.array([[.2, .5, .1], [.3, -.4, .6]])
    assert_grads(a.gram(x, return_tensor=True), b.gram(x, return_tensor=True),
                 tuple(a.parameters()), tuple(b.parameters()), atol=5e-6)


@pytest.mark.parametrize("loss", ["mmd", "nll"])
def test_genlearn_losses_and_adam(loss):
    cfg = {"ansatz": "AngleAnsatz", "ansatz_config": deepcopy(ANSATZ), "loss": loss,
           "seed": 41, "diff_method": "backprop"}
    if loss == "mmd": cfg["mmd_config"] = {"mode": "exact"}
    a = create_qcbm({**deepcopy(cfg), "device": DEVICE}, 3, [[0, 0, 0], [1, 1, 1]])
    b = create_qcbm({**deepcopy(cfg), "device": "default.qubit"}, 3, [[0, 0, 0], [1, 1, 1]])
    assert_grads(a.loss(), b.loss(), tuple(a.parameters()), tuple(b.parameters()))
    opt = torch.optim.Adam(a.parameters(), lr=.01)
    before = a.params.detach().clone()
    opt.zero_grad(); a.loss().backward(); opt.step()
    assert not torch.equal(before, a.params)


@pytest.mark.parametrize("algorithm", ["qaoa", "warm-qaoa", "pce"])
def test_optimization_auto_diff_noncommuting_circuits(algorithm):
    problem = QUBO.maxcut(3, [(0, 1), (1, 2), (0, 2)])
    kwargs = dict(backend="torch", seed=31)
    cls = PCE if algorithm == "pce" else QAOA
    if algorithm != "pce": kwargs["n_layers"] = 2
    if algorithm == "warm-qaoa": kwargs["warm_start"] = [.2, .6, .4]
    a = cls(problem, device=DEVICE, **kwargs)
    b = cls(problem, device="default.qubit", diff_method="backprop", **kwargs)
    assert a.diff_method == "backprop"
    t = torch.tensor(a.initial_parameters(), dtype=torch.float64, requires_grad=True)
    fa, fb = (a.loss_tensor, b.loss_tensor) if algorithm == "pce" else (a.expectation_tensor, b.expectation_tensor)
    assert_grads(fa(t), fb(t), (t,), (t,))
    if algorithm != "pce":
        np.testing.assert_allclose(a.probabilities(t.detach().numpy()), b.probabilities(t.detach().numpy()), atol=2e-12)
