"""Shared PP device: statevector agreement, gradients and explicit limitations."""
import numpy as np
import pennylane as qml
import pytest
import torch

from qasp import PauliPropagationSimulator, PauliPropagationOptions, DifferentiablePauliProp
from qasp.devices import create_device
from qasp.qkernels import create_kernel
from qasp.simulators import action_probs, DecoderSpec


def nodes(fn, wires=3, **options):
    return [qml.QNode(fn, dev, interface="torch", diff_method="backprop")
            for dev in (PauliPropagationSimulator(wires, **options), qml.device("default.qubit", wires=wires))]


@pytest.mark.parametrize("seed", [3, 17, 41])
def test_noncommuting_values_jacobians_and_observable_coefficients(seed, monkeypatch):
    def circuit(t, c):
        qml.BasisState(np.array([1, 0, 1]), wires=[0, 1, 2])
        qml.Rot(t[0], t[1], t[2], 0)
        qml.CNOT([0, 1]); qml.CZ([1, 2]); qml.Hadamard(2)
        qml.PauliRot(t[3], "XYZ", [2, 0, 1])
        qml.IsingYY(t[0], [1, 2]); qml.MultiRZ(t[2], [0, 1, 2])
        qml.ctrl(qml.RY, control=2)(t[1], wires=0)
        qml.adjoint(qml.Rot)(t[2], t[1], t[0], wires=1)
        obs = c[0] * qml.X(0) + c[1] * (qml.Z(1) @ qml.Y(2))
        return qml.expval(obs), qml.var(obs), qml.probs(wires=[2, 0])
    actual, reference = nodes(circuit)
    g = torch.Generator().manual_seed(seed)
    t = torch.randn(4, dtype=torch.float64, generator=g, requires_grad=True)
    c = torch.tensor([0., .4], dtype=torch.float64, requires_grad=True)
    expected = reference(t, c)
    expected_grad = torch.autograd.functional.jacobian(reference, (t, c))
    # Prove the device does not dispatch to statevector execution.
    monkeypatch.setattr(qml.devices.DefaultQubit, "execute", lambda *a, **kw: pytest.fail("statevector fallback"))
    result = actual(t, c)
    for a, b in zip(result, expected):
        torch.testing.assert_close(a, b, atol=2e-12, rtol=2e-12)
    result_grad = torch.autograd.functional.jacobian(actual, (t, c))
    for a, b in zip(result_grad, expected_grad):
        for u, v in zip(a, b):
            torch.testing.assert_close(u, v, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("word", ["XYZ", "IIII", "IXYZ", "XXXX", "ZZZZ", "ZYXY"])
@pytest.mark.parametrize("angle", [0., .4, np.pi / 2])
def test_general_pauli_rotations_and_second_derivatives(word, angle):
    def f(t):
        for w in range(4):
            qml.RX(.21 * (w+1), w); qml.RY(.39, w)
        qml.PauliRot(t, word, wires=range(len(word)))
        return qml.expval(qml.Y(0) @ qml.Z(2))
    t = torch.tensor(angle, dtype=torch.float64, requires_grad=True)
    result = []
    for node in nodes(f, wires=4):
        value = node(t)
        grad = torch.autograd.grad(value, t, create_graph=True)[0]
        hess = torch.autograd.grad(grad, t)[0]
        result.append(torch.stack((value, grad, hess)))
    torch.testing.assert_close(*result, atol=2e-12, rtol=2e-12)


def test_wire_labels_batch_partial_probabilities_and_empty_circuit():
    def f(t):
        qml.RY(t, "b"); qml.CNOT(["b", "a"])
        return qml.probs(wires=["idle", "a"]), qml.expval(qml.Projector([1, 0], wires=["a", "b"]))
    x = torch.tensor([0., .4, -.2], dtype=torch.float64, requires_grad=True)
    a, b = [n(x) for n in nodes(f, wires=["b", "idle", "a"])]
    for u, v in zip(a, b): torch.testing.assert_close(u, v, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(torch.autograd.grad(a[0][:, 1].sum(), x)[0],
                               torch.autograd.grad(b[0][:, 1].sum(), x)[0])
    dev = PauliPropagationSimulator(["a", "b"])
    out = qml.QNode(lambda: qml.probs(), dev, diff_method=None)()
    np.testing.assert_allclose(out, [1., 0., 0., 0.])


def test_only_observable_is_trainable_and_unused_parameters_have_zero_gradients():
    def f(c):
        qml.RY(.7, 0)
        return qml.expval(c[0] * qml.Z(0) + c[1] * qml.X(0))
    c = torch.tensor([0., 0.], dtype=torch.float64, requires_grad=True)
    a, b = [node(c) for node in nodes(f, 1)]
    torch.testing.assert_close(torch.autograd.grad(a, c)[0], torch.autograd.grad(b, c)[0])
    def independent(x):
        qml.RY(x, 1)
        return qml.expval(qml.Z(0))
    x = torch.tensor(.5, dtype=torch.float64, requires_grad=True)
    value = nodes(independent, 2)[0](x)
    g = torch.autograd.grad(value, x, create_graph=True)[0]
    assert g.item() == 0
    assert torch.autograd.grad(g, x)[0].item() == 0


def test_imports_resolver_options_and_legacy_identity():
    from qasp.rl.differentiable_pp_sim import DifferentiablePauliProp as Legacy
    from qasp.simulators import DifferentiablePauliProp as Shared
    from qasp.rl import DifferentiablePauliProp as RL
    assert Legacy is Shared is RL is DifferentiablePauliProp
    assert Shared.__module__ == "qasp.simulators._pauli_propagation"
    for spec in ("PauliPropagationSimulator", "pauli.propagation", PauliPropagationSimulator):
        assert isinstance(create_device(spec, wires=2), PauliPropagationSimulator)
    dev = PauliPropagationSimulator(2, options=PauliPropagationOptions(k=2))
    assert create_device(dev, wires=2) is dev
    with pytest.raises(ValueError, match="mpi"):
        create_device("PauliPropagationSimulator", wires=2, mpi=True)
    with pytest.raises(ValueError, match="shots"):
        PauliPropagationSimulator(2, shots=10)


@pytest.mark.parametrize("kwargs", [{"k": -1}, {"topM": 0}, {"tau": -1}, {"tau": float("nan")},
                                       {"prune_every": -2}, {"max_probability_wires": -1}, {"prune": "yes"}])
def test_option_validation(kwargs):
    with pytest.raises(ValueError): PauliPropagationSimulator(2, **kwargs)


@pytest.mark.parametrize("kind", ["state", "sample", "noise", "stateprep", "hermitian", "finite-shots"])
def test_unsupported_modes_are_explicit(kind):
    dev = PauliPropagationSimulator(2)
    def f():
        if kind == "noise": qml.DepolarizingChannel(.1, 0)
        if kind == "stateprep": qml.StatePrep(np.array([1., 0.]), wires=[0])
        if kind == "state": return qml.state()
        if kind == "sample": return qml.sample(wires=[0])
        if kind == "hermitian": return qml.expval(qml.Hermitian(np.eye(2), 0))
        return qml.probs(wires=[0])
    tape = qml.tape.make_qscript(f)()
    if kind == "finite-shots": tape = tape.copy(shots=10)
    with pytest.raises(ValueError, match="PauliPropagationSimulator"):
        dev.execute(tape)


def test_probability_budget_is_on_readout_not_register():
    def f(t):
        qml.RY(t, 99)
        return qml.probs(wires=[99, 2])
    dev = PauliPropagationSimulator(100, max_probability_wires=2)
    t = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
    result = qml.QNode(f, dev, interface="torch")(t)
    torch.testing.assert_close(result, torch.stack((torch.cos(t/2)**2, 0*t, torch.sin(t/2)**2, 0*t)))
    with pytest.raises(MemoryError, match="max_probability_wires"):
        qml.QNode(lambda: qml.probs(wires=[0, 1, 2]), dev, diff_method=None)()


def test_pruning_is_opt_in_and_preserves_finite_gradients():
    def f(t):
        qml.RY(t, 0); qml.CNOT([0, 1]); qml.RX(t, 1)
        return qml.expval(qml.Z(0) + .01 * qml.Z(1))
    t = torch.tensor(.4, dtype=torch.float64, requires_grad=True)
    exact, ref = nodes(f, 2)
    torch.testing.assert_close(exact(t), ref(t), atol=1e-12, rtol=1e-12)
    approx = qml.QNode(f, PauliPropagationSimulator(2, prune=True, topM=1, tau=.05, prune_every=1), interface="torch")
    value = approx(t)
    assert torch.isfinite(torch.autograd.grad(value, t)[0])
    assert not torch.isclose(value, exact(t))


def test_python_float_precision():
    def f(x):
        qml.RY(x, 0)
        return qml.expval(.123456789123 * qml.Z(0))
    x = torch.tensor(.31, dtype=torch.float64, requires_grad=True)
    a = qml.QNode(f, PauliPropagationSimulator(1), interface="torch")(x)
    torch.testing.assert_close(a, .123456789123 * torch.cos(x), atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("kind", ["projected", "fidelity"])
def test_nontrainable_kernel(kind):
    cfg = {"kernel_type": kind, "feature_map": "AngleEmbedding", "feature_map_config": {"rotations": "Y"}}
    a = create_kernel({**cfg, "device": "PauliPropagationSimulator"}, n_qubits=2)
    b = create_kernel({**cfg, "device": "default.qubit"}, n_qubits=2)
    x = np.array([[.2, .7], [-.1, .3]])
    np.testing.assert_allclose(a.gram(x), b.gram(x), atol=1e-12, rtol=1e-12)


def test_compact_measurement_semantics():
    decoder = DecoderSpec("majority")
    def circuit(x, compact):
        qml.RY(x, 0); qml.CNOT([0, 1]); qml.RX(.4, 2)
        return action_probs([2, 0, 1], decoder) if compact else qml.probs(wires=[2, 0, 1])
    x = torch.tensor(.4, dtype=torch.float64, requires_grad=True)
    a = qml.QNode(lambda x: circuit(x, True), PauliPropagationSimulator(3), interface="torch")(x)
    b = decoder.aggregate(qml.QNode(lambda x: circuit(x, False), qml.device("default.qubit", wires=3), interface="torch")(x))
    torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(torch.autograd.grad(a[0], x)[0], torch.autograd.grad(b[0], x)[0])


def test_projector_arithmetic_coefficients():
    def f(x):
        qml.RY(x[0], 0); qml.CNOT([0, 1])
        return qml.expval(x[1] * qml.Projector([0], [1]) + .25 * qml.Z(0))
    x = torch.tensor([.4, 0.], dtype=torch.float64, requires_grad=True)
    values = [qml.QNode(f, d, interface="torch")(x) for d in (PauliPropagationSimulator(2), qml.device("default.qubit", wires=2))]
    torch.testing.assert_close(*values)
    torch.testing.assert_close(*[torch.autograd.grad(v, x)[0] for v in values])


def test_parameter_shift_autograd_explicit():
    def f(t):
        qml.RY(t, 0)
        return qml.expval(qml.Z(0))
    node = qml.QNode(f, PauliPropagationSimulator(1), interface="autograd", diff_method="parameter-shift")
    t = qml.numpy.array(.3, requires_grad=True)
    np.testing.assert_allclose(qml.grad(node)(t), -np.sin(t), atol=1e-12)


@pytest.mark.parametrize("with_gates", [False, True])
def test_initial_weight_projection_and_dropped_coefficient_gradients(with_gates):
    def f(c):
        if with_gates:
            qml.RZ(.2, 0)
        return qml.expval(c * (qml.Z(0) @ qml.Z(1)) + .5 * qml.Z(0))
    c = torch.tensor(.3, dtype=torch.float64, requires_grad=True)
    node = qml.QNode(f, PauliPropagationSimulator(2, k=1), interface="torch")
    value = node(c)
    torch.testing.assert_close(value, torch.tensor(.5, dtype=torch.float64))
    grad = torch.autograd.grad(value, c, create_graph=True)[0]
    assert grad.item() == 0
    assert torch.autograd.grad(grad, c)[0].item() == 0
