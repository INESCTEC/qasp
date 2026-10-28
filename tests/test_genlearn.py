"""Numerical contracts for PQCBM, moment MMD, drawings, and MPI routing."""

from copy import deepcopy

import numpy as np
import pennylane as qml
import pytest
import torch

from qasp import PQCBM, create_qcbm, draw_qcbm
from qasp.genlearn import BornMachine, PauliMMD
from qasp.genlearn.losses import basis_indices, mmd_loss

DATA = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0], [0, 1, 1]])
CONFIG = {
    "ansatz": "AngleAnsatz",
    "ansatz_config": {
        "n_layers": 2,
        "rotations": "YZ",
        "include_imprimitive": True,
        "imprimitive": "CNOT",
        "imprimitive_pattern": "chain",
    },
    "loss": "mmd",
    "mmd_config": {"mode": "exact", "bandwidths": [0.6, 1.3]},
    "seed": 3,
}


def model(**kwargs):
    return create_qcbm({**CONFIG, **kwargs}, 3, DATA)


def dense_mmd(probs, data, bandwidths):
    bits = (np.arange(8)[:, None] >> np.arange(2, -1, -1)) & 1
    distances = ((bits[:, None] - bits[None, :]) ** 2).sum(axis=-1)
    kernel = np.mean([np.exp(-distances / (2 * b**2)) for b in bandwidths], axis=0)
    empirical = np.bincount(basis_indices(data), minlength=8) / len(data)
    delta = probs - torch.tensor(empirical)
    return delta @ torch.tensor(kernel) @ delta


def test_factory_and_exact_mmd_value_and_gradient():
    cfg = deepcopy(CONFIG)
    m = create_qcbm(cfg, 3, DATA)
    assert cfg == CONFIG
    exact = m.loss()
    expected = dense_mmd(m.get_probs(), DATA, [0.6, 1.3])
    torch.testing.assert_close(exact, expected, atol=1e-12, rtol=1e-12)
    a = torch.autograd.grad(exact, m.params)[0]
    b = torch.autograd.grad(expected, m.params)[0]
    torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)


@pytest.mark.parametrize(
    "device", ["lightning.qubit", "lightning.kokkos", "default.mixed", "lightning.pp"]
)
@pytest.mark.parametrize("ansatz", ["AngleAnsatz", "IQPAnsatz"])
def test_backend_values_gradients_and_update(device, ansatz):
    ac = (
        CONFIG["ansatz_config"]
        if ansatz == "AngleAnsatz"
        else {"n_layers": 2, "imprimitive_pattern": "chain"}
    )
    reference = model(ansatz=ansatz, ansatz_config=ac)
    candidate = model(
        ansatz=ansatz,
        ansatz_config=ac,
        device=device,
        pp_prune=False,
        pp_tau=0,
        pp_topM=None,
    )
    x, y = reference.loss(), candidate.loss()
    torch.testing.assert_close(x, y, atol=2e-6, rtol=2e-6)
    gx = torch.autograd.grad(x, reference.params)[0]
    gy = torch.autograd.grad(y, candidate.params)[0]
    torch.testing.assert_close(gx, gy, atol=2e-6, rtol=2e-6)
    reference.train(1)
    candidate.train(1)
    torch.testing.assert_close(reference.params, candidate.params, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("interface", ["torch", "autograd"])
@pytest.mark.parametrize("loss", ["nll", "mmd"])
def test_training_default_loss_and_feature_scaling(interface, loss):
    cfg = {
        "n_qubits": 2,
        "ansatz": "AngleAnsatz",
        "ansatz_config": {"rotations": "Y", "n_layers": 2, "reuploading": True},
        "feature_map": "AngleEmbedding",
        "feature_map_config": {"features": 2, "rotations": "X", "input_scaling": True},
        "inputs": [0.3, 0.6],
        "interface": interface,
        "loss": loss,
        "seed": 8,
    }
    m = create_qcbm(cfg, data=[[0, 0], [1, 1]])
    before = np.asarray(qml.math.toarray(m.input_scaling_params)).copy()
    initial = float(qml.math.toarray(m.loss()))
    _, history = m.train(15)
    assert np.isfinite(history).all()
    assert float(qml.math.toarray(m.loss())) < initial
    assert not np.allclose(before, qml.math.toarray(m.input_scaling_params))


@pytest.mark.parametrize("fisher", ["quantum", "classical"])
@pytest.mark.parametrize("commuting", [False, True])
def test_natural_gradient_is_executable(fisher, commuting):
    m = PQCBM(
        n_qubits=2,
        ansatz_config={"rotations": "Y"},
        init_params=[0.6, 0.8],
        data=[[0, 0], [0, 1]],
        interface="autograd",
        natural_grad=True,
        fisher=fisher,
        commuting=commuting,
        lr=0.02,
    )
    before = float(m.loss())
    _, history = m.train(3)
    assert len(history) == 3
    assert np.isfinite(history).all()
    assert float(m.loss()) < before


def test_unbiased_data_term_matches_kernel_u_statistic():
    m = model(
        mmd_config={"mode": "exact", "bandwidths": [0.9], "estimator": "unbiased"}
    )
    biased = dense_mmd(m.get_probs(), DATA, [0.9])
    kxx = np.exp(-((DATA[:, None] - DATA[None, :]) ** 2).sum(-1) / (2 * 0.9**2))
    correction = (1 - kxx.mean()) / (len(DATA) - 1)
    torch.testing.assert_close(m.loss(), biased - correction, atol=1e-12, rtol=1e-12)


def test_sampled_masks_match_bernoulli_kernel_and_reproduce():
    a = PauliMMD(4, bandwidths=[0.75], n_observables=50000, seed=7)
    b = PauliMMD(4, bandwidths=[0.75], n_observables=50000, seed=7)
    np.testing.assert_array_equal(a.masks, b.masks)
    expected = (1 - np.exp(-1 / (2 * 0.75**2))) / 2
    assert abs(a.masks.mean() - expected) < 0.003
    assert (~a.masks.any(axis=1)).any()
    a.resample()
    assert not np.array_equal(a.masks, b.masks)


def test_cross_estimator_removes_model_square_noise():
    # Average every pair of independent +/-1 realizations of a Bernoulli model.
    mu, target = 0.3, -0.2
    probability = {1: (1 + mu) / 2, -1: (1 - mu) / 2}
    result = sum(
        px
        * py
        * float(
            mmd_loss(
                torch.tensor([float(x)]),
                [target],
                model_moments_2=torch.tensor([float(y)]),
            )
        )
        for x, px in probability.items()
        for y, py in probability.items()
    )
    assert result == pytest.approx((mu - target) ** 2, abs=1e-7)


def test_finite_shots_mmd_executes_twice_and_has_gradients(monkeypatch):
    m = model(shots=3000, diff_method="parameter-shift", device_kwargs={"seed": 71})
    count = []
    original = m.moments

    def wrapped(*args, **kwargs):
        count.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(m, "moments", wrapped)
    loss = m.loss()
    assert len(count) == 2
    loss.backward()
    assert torch.isfinite(m.params.grad).all()
    assert m.sample(1).shape == (1, 3)


def test_large_width_pp_never_allocates_probabilities(monkeypatch):
    m = create_qcbm(
        {
            "n_qubits": 70,
            "device": "lightning.pp",
            "loss": "mmd",
            "ansatz_config": {"rotations": "Y", "include_imprimitive": False},
            "mmd_config": {"n_observables": 8, "bandwidths": [6]},
            "pp_prune": False,
        },
        data=np.zeros((3, 70), int),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("A probability vector was requested")

    monkeypatch.setattr(m, "get_probs", forbidden)
    assert np.isfinite(m.train(1)[1]).all()
    assert m.moments(np.zeros((2, 70), int)).tolist() == [1, 1]


def test_sampling_pp_deployment_and_legacy_alias():
    m = model(device="lightning.pp")
    with pytest.raises(ValueError, match="explicit"):
        m.sample(4)
    with pytest.raises(ValueError, match="explicit"):
        m.get_probs()
    bits = m.sample(40, device="default.qubit", device_kwargs={"seed": 9})
    assert bits.shape == (40, 3)
    assert np.isin(bits, [0, 1]).all()
    with pytest.warns(DeprecationWarning):
        old = BornMachine(n_qubits=2, data=[[0, 0]])
    assert old.sample(4).shape == (4,)
    assert np.isfinite(old.train(1)[1]).all()


def test_drawing_preserves_model_and_covers_formats():
    m = model(device="lightning.pp")
    measurement = m.circuit_builder.measurement
    tex = draw_qcbm(
        m,
        output="tikz",
        tikz_options={
            "standalone": True,
            "symbolic_parameters": True,
            "compress": True,
        },
    )
    assert "quantikz" in tex and "document" in tex
    assert m.circuit_builder.measurement == measurement
    assert "Probs" in m.draw(output="text")
    figure, _ = m.draw(output="matplotlib")
    import matplotlib.pyplot as plt

    plt.close(figure)
    assert "quantikz" in draw_qcbm(config={**CONFIG, "n_qubits": 3}, output="tex")


def test_existing_qnode_nll_mmd_and_drawing():
    @qml.qnode(qml.device("default.qubit", wires=2))
    def circuit(params):
        qml.RY(params[0], 0)
        qml.CNOT([0, 1])
        return qml.probs(wires=[0, 1])

    for loss in ["nll", "mmd"]:
        m = PQCBM(circuit, [0.5], [[0, 0], [1, 1]], loss=loss)
        assert np.isfinite(m.train(2)[1]).all()
        assert "quantikz" in m.draw(output="tex")


def test_mpi_forwarding_and_reproducibility(monkeypatch):
    calls = []
    original = qml.device

    def local(name, **kwargs):
        calls.append((name, kwargs.copy()))
        result = original(
            "default.qubit", wires=kwargs["wires"], shots=kwargs.get("shots")
        )
        result._mpi = kwargs.get("mpi", False)
        return result

    monkeypatch.setattr(qml, "device", local)
    config = {
        **CONFIG,
        "n_qubits": 3,
        "device": "lightning.kokkos",
        "mpi": True,
        "device_kwargs": {"batch_obs": False},
        "mmd_config": {"n_observables": 12},
        "batch_size": 2,
    }
    a, b = create_qcbm(config, data=DATA), create_qcbm(config, data=DATA)
    assert calls[-1][1]["mpi"] is True and calls[-1][1]["batch_obs"] is False
    np.testing.assert_allclose(a.train(3)[1], b.train(3)[1])
    torch.testing.assert_close(a.params, b.params)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"device": "lightning.pp", "loss": "nll", "mmd_config": None}, "requires"),
        ({"device": "lightning.pp", "mpi": True}, "mpi=False"),
        ({"device": "lightning.pp", "shots": 20}, "shots=None"),
        ({"device": "lightning.pp", "interface": "autograd"}, "requires"),
        ({"mpi": True, "enable_mpi": False}, "agree"),
        ({"mpi": True, "seed": None}, "seed"),
        ({"mmd_config": {"bandwidths": [0]}}, "bandwidths"),
        ({"mmd_config": {"n_observables": 0}}, "n_observables"),
        ({"batch_size": 0}, "batch_size"),
    ],
)
def test_invalid_configuration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        model(**kwargs)


def test_binary_and_exact_size_validation():
    with pytest.raises(ValueError, match="binary"):
        create_qcbm(CONFIG, 3, [[0, 0.5, 1]])
    with pytest.raises(ValueError, match="enumerates"):
        PauliMMD(70, mode="exact")
    with pytest.raises(ValueError, match="two"):
        PauliMMD(2, estimator="unbiased").target_statistics([[0, 1]])
    assert basis_indices([[1, 0, 1], [0, 1, 0]]).tolist() == [5, 2]


def test_qnode_probability_wire_order_and_validation():
    @qml.qnode(qml.device("default.qubit", wires=["a", "b", "c"]))
    def circuit(params):
        qml.RY(params[0], "b")
        return qml.probs(wires=["b", "a"])

    m = PQCBM(
        circuit,
        [0.7],
        [[0, 0], [1, 0]],
        n_qubits=2,
        loss="mmd",
        mmd_config={"mode": "exact"},
    )
    np.testing.assert_allclose(m.moments([[1, 0], [0, 1]]).detach(), [np.cos(0.7), 1])
    assert m.sample(20)[:, 1].sum() == 0

    @qml.qnode(qml.device("default.qubit", wires=1))
    def invalid(params):
        qml.RY(params[0], 0)
        return qml.expval(qml.Z(0))

    with pytest.raises(ValueError, match="probability"):
        PQCBM(invalid, [0.2], [[0]])


@pytest.mark.parametrize("device", ["default.qubit", "lightning.pp"])
def test_neural_embedding_trains_all_weights(device):
    m = create_qcbm(
        {
            "n_qubits": 2,
            "ansatz": "IQPAnsatz",
            "ansatz_config": {"n_layers": 2},
            "feature_map": "NeuralEmbedding",
            "feature_map_config": {"features": 3, "rotations": "Y"},
            "inputs": [0.2, 0.5, 0.8],
            "loss": "mmd",
            "device": device,
            "mmd_config": {"mode": "exact"},
        },
        data=[[0, 0], [1, 1]],
    )
    m.loss().backward()
    assert m.params.grad is not None and torch.isfinite(m.params.grad).all()
    assert (
        m.input_scaling_params.grad is not None
        and m.input_scaling_params.grad.abs().sum() > 0
    )


def test_draw_encoded_config_accepts_explicit_or_default_inputs():
    config = {
        **CONFIG,
        "feature_map": "AngleEmbedding",
        "feature_map_config": {"features": lambda n: n, "rotations": "Y"},
    }
    for inputs in (None, [0.1, 0.2, 0.3]):
        assert "quantikz" in draw_qcbm(
            config=config, n_qubits=3, inputs=inputs, output="tikz"
        )
    assert "inputs" not in config


def test_optimizer_factory_binds_live_parameters_and_rejects_wrong_owner():
    m = model(optimizer=lambda p: torch.optim.SGD(p, lr=0.1))
    before = m.params.detach().clone()
    m.train(1)
    assert not torch.allclose(before, m.params)
    unrelated = torch.nn.Parameter(torch.zeros(1))
    with pytest.raises(ValueError, match="own this model"):
        model(optimizer=torch.optim.Adam([unrelated]))


def test_mpi_rejects_nondeterministic_mask_seed():
    with pytest.raises(ValueError, match="MMD seed"):
        model(mpi=True, mmd_config={"seed": None})
