"""Independent Pauli-string encoding checks through kernel and RL public APIs.

References use RY/IsingYY (or RX/IsingXX) and matrix-vector projections, never
the feature-map implementation under test. Batch sizes intentionally differ
from both feature count and rotation count. Fisher kernels are fixed snapshots.
"""
from copy import deepcopy

import matplotlib
import numpy as np
import pennylane as qml
import pytest
import torch

from qasp import AngleEmbedding, NeuralEmbedding, create_kernel, create_policy, draw_kernel, draw_policy
from qasp.qkernels import QFKernel
from configuration_helpers import execution

# Circuit rendering tests must also run in headless macOS and CI sessions.
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def double_precision():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def reference_encoding(embedding, inputs, scales):
    """Queue explicit primitive gates with independently calculated angles."""
    neural = isinstance(embedding, NeuralEmbedding)
    if neural:
        if isinstance(inputs, torch.Tensor):
            inputs = inputs.to(scales)
        angles = inputs @ scales.reshape(-1, embedding.features).T
    else:
        angles = inputs * scales if embedding.input_scaling else inputs
    if isinstance(embedding.rotations, str):
        for axis in embedding.rotations:
            for wire in range(embedding.n_qubits if neural else embedding.features):
                {"X": qml.RX, "Y": qml.RY, "Z": qml.RZ}[axis](angles[..., wire], wire)
    else:
        for i, specification in enumerate(embedding.rotations):
            word, *wires = specification.split()
            gate = {"X": qml.RX, "Y": qml.RY, "Z": qml.RZ,
                    "XX": qml.IsingXX, "YY": qml.IsingYY, "ZZ": qml.IsingZZ}[word]
            gate(angles[..., i], wires=[int(w) for w in wires])


class ReferenceNeural(NeuralEmbedding):
    def __call__(self, inputs=None, input_scaling_params=None, **kwargs):
        reference_encoding(self, inputs, input_scaling_params)


class ReferenceAngle(AngleEmbedding):
    def __call__(self, inputs=None, input_scaling_params=None, **kwargs):
        reference_encoding(self, inputs, input_scaling_params)


def options(backend):
    if backend == "adjoint":
        return {"device": "default.qubit", "diff_method": "adjoint"}
    return execution(backend)


def components(embedding, reupload=True, axis="Y"):
    features = 4 if embedding == "neural" else 2
    config = {
        "feature_map": "NeuralEmbedding" if embedding == "neural" else "AngleEmbedding",
        "feature_map_config": {"features": features, "rotations": [f"{axis} 0", f"{axis*2} 0 1"]},
        "ansatz": "AngleAnsatz",
        "ansatz_config": {"n_layers": 2, "rotations": "ZY", "reuploading": reupload,
                          "include_imprimitive": True, "imprimitive": "CZ",
                          "imprimitive_pattern": "all_to_all", "include_final_imprimitive": False},
    }
    if embedding == "angle":
        config["feature_map_config"]["input_scaling"] = True
    if axis == "X":
        config["ansatz_config"].update(rotations=["X 0", "X 1", "XX 0 1"], include_imprimitive=False)
    return config


def set_weights(parameters):
    with torch.no_grad():
        for i, parameter in enumerate(parameters):
            parameter.copy_(torch.linspace(0.21 + i * 0.07, 0.93, parameter.numel()).reshape(parameter.shape))


def compare_gradients(actual, expected, actual_parameters, reference_parameters, atol=1e-7):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=atol)
    weights = torch.linspace(0.3, 1.2, actual.numel()).reshape(actual.shape)
    gradients = torch.autograd.grad((actual * weights).sum(), actual_parameters)
    reference_gradients = torch.autograd.grad((expected * weights).sum(), reference_parameters)
    for got, ref in zip(gradients, reference_gradients):
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got, ref, atol=atol, rtol=atol)
    return gradients


@pytest.mark.parametrize("rotations", [["Y 0", "YY 0 1"], "Y", "ZY"])
@pytest.mark.parametrize("features", [2, 4])
@pytest.mark.parametrize("batch_size", [2, 5])
@pytest.mark.parametrize("diff_method", ["backprop", "adjoint"])
def test_neural_strings_against_independent_projection(rotations, features, batch_size, diff_method):
    embedding = NeuralEmbedding(2, features=features, rotations=rotations)

    def make_node(reference):
        @qml.qnode(qml.device("default.qubit", wires=2), interface="torch", diff_method=diff_method)
        def node(inputs, scales, angles):
            if reference:
                reference_encoding(embedding, inputs, scales)
            else:
                embedding(inputs, scales)
            qml.RZ(angles[0], 0)
            qml.RY(angles[1], 1)
            return qml.expval(qml.Z(0)), qml.expval(qml.Z(0) @ qml.Z(1))
        return node

    inputs = torch.linspace(-0.7, 0.8, features * batch_size).reshape(batch_size, features).requires_grad_()
    scales = torch.linspace(0.2, 1.1, 2 * features).requires_grad_()
    angles = torch.tensor([0.3, -0.4], requires_grad=True)
    node, reference = make_node(False), make_node(True)
    actual = torch.stack(node(inputs, scales, angles), dim=-1)
    expected = torch.stack(reference(inputs, scales, angles), dim=-1)
    individual = torch.stack([torch.stack(node(row, scales, angles)) for row in inputs])
    assert actual.shape == (batch_size, 2)
    torch.testing.assert_close(actual, individual, atol=1e-10, rtol=1e-10)
    compare_gradients(actual, expected, [inputs, scales, angles], [inputs, scales, angles], atol=1e-9)


def make_policy(kind, config):
    if kind == "softmax":
        return create_policy("PQCSoftmax", {**config, "n_actions": 2,
                             "observables": ["Z 0", "ZZ 0 1"], "fc_layer": None}, 2).double()
    if kind == "qfunction":
        return create_policy("PQDQN", {**config, "n_actions": 2,
                             "observables": ["Z 0", "ZZ 0 1"], "input_activation": None,
                             "fc_layer": False}, 2).double()
    return create_policy("PQCBorn", {**config, "n_actions": 2, "policy": "parity-like",
                         "parity_cnot_decomposition": False,
                         "training_method": "probs" if kind == "born_probs" else "expectation"}, 2).double()


@pytest.mark.parametrize("backend", ["statevector", "adjoint", "lightning", "mixed", "pp", "mps", "tn"])
@pytest.mark.parametrize("embedding", ["angle", "neural"])
@pytest.mark.parametrize("reupload", [False, True])
@pytest.mark.parametrize("kind", ["softmax", "born", "qfunction"])
def test_policy_strings_batches_gradients_and_optimizer(backend, embedding, reupload, kind):
    cfg = components(embedding, reupload)
    actual = make_policy(kind, {**cfg, **options(backend)})
    reference_cfg = deepcopy(cfg)
    reference_cfg["feature_map"] = ReferenceNeural if embedding == "neural" else ReferenceAngle
    reference = make_policy(kind, {**reference_cfg, **options("statevector")})
    set_weights(actual.parameters())
    reference.load_state_dict(actual.state_dict())
    features = cfg["feature_map_config"]["features"]
    x = torch.linspace(-0.6, 0.8, 5 * features).reshape(5, features).requires_grad_()
    a, b = actual(x), reference(x)
    assert a.shape == (5, 2)
    torch.testing.assert_close(a, torch.stack([actual(row) for row in x]), atol=1e-8, rtol=1e-8)
    compare_gradients(a, b, [*actual.parameters(), x], [*reference.parameters(), x])
    # Equal feature/batch counts can silently hide axis errors; test a second
    # batch and a rollout-style pair of leading dimensions explicitly.
    for batch in (x[:2], x[:4].reshape(2, 2, features)):
        torch.testing.assert_close(actual(batch), reference(batch), atol=1e-8, rtol=1e-8)
    if kind != "qfunction":
        probabilities = actual.action_probs(x)
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(5), atol=1e-8, rtol=1e-8)
        assert probabilities.min() >= -1e-10
    before = [p.detach().clone() for p in actual.parameters()]
    optimizer = torch.optim.Adam(actual.parameters(), lr=0.01)
    optimizer.zero_grad()
    output = actual(x.detach())
    loss = -output[:, 0].clamp_min(1e-8).log().mean() if kind == "born" else output.square().mean()
    loss.backward()
    optimizer.step()
    assert all(torch.isfinite(p).all() for p in actual.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, actual.parameters()))


@pytest.mark.parametrize("backend", ["statevector", "lightning", "pp"])
@pytest.mark.parametrize("embedding", ["angle", "neural"])
def test_born_probability_readout_custom_strings(backend, embedding):
    cfg = components(embedding)
    opts = options(backend)
    if backend == "lightning":
        opts["diff_method"] = "parameter-shift"
    model = make_policy("born_probs", {**cfg, **opts})
    cfg["feature_map"] = ReferenceNeural if embedding == "neural" else ReferenceAngle
    reference = make_policy("born_probs", {**cfg, **options("statevector")})
    set_weights(model.parameters())
    reference.load_state_dict(model.state_dict())
    x = torch.linspace(-0.6, 0.7, 3 * cfg["feature_map_config"]["features"]).reshape(3, -1).requires_grad_()
    compare_gradients(model(x), reference(x), [*model.parameters(), x], [*reference.parameters(), x])


def kernel_config(embedding, layout, kind):
    cfg = components(embedding, reupload=layout == "reupload")
    if layout == "encoding":
        cfg.pop("ansatz")
        cfg.pop("ansatz_config")
        cfg["reps"] = 2
        count = 8 if embedding == "neural" else 2
        cfg["input_scaling_params"] = np.linspace(0.3, 0.9, count)
    cfg["kernel_type"] = kind
    if kind == "projected":
        cfg.update(observables=["Z 0", "Z 1", "ZZ 0 1"], weights=[0.4, 0.7, 1.1], gamma=0.8)
    return cfg


@pytest.mark.parametrize("backend", ["statevector", "adjoint", "lightning", "mixed", "pp", "mps", "tn"])
@pytest.mark.parametrize("embedding", ["angle", "neural"])
@pytest.mark.parametrize("layout", ["encoding", "ansatz", "reupload"])
@pytest.mark.parametrize("kind", ["fidelity", "projected"])
def test_kernel_strings_values_and_parameter_gradients(backend, embedding, layout, kind):
    cfg = kernel_config(embedding, layout, kind)
    actual = create_kernel({**deepcopy(cfg), **options(backend)}, 2)
    cfg["feature_map"] = ReferenceNeural if embedding == "neural" else ReferenceAngle
    reference = create_kernel({**cfg, **options("statevector")}, 2)
    if actual.trainable:
        actual.qlayer.double()
        reference.qlayer.double()
        set_weights(actual.parameters())
        reference.qlayer.load_state_dict(actual.qlayer.state_dict())
    f = actual.nfeat
    x = np.linspace(-0.7, 0.8, 3 * f).reshape(3, f)
    y = np.linspace(0.6, -0.9, 2 * f).reshape(2, f)
    gram = actual.gram(x)
    np.testing.assert_allclose(gram, reference.gram(x), atol=3e-6, rtol=3e-6)
    np.testing.assert_allclose(gram, gram.T, atol=2e-6)
    np.testing.assert_allclose(gram.diagonal(), 1, atol=2e-6)
    assert np.linalg.eigvalsh(gram).min() >= -3e-6
    if actual.trainable:
        a, b = actual.gram(x, y, return_tensor=True), reference.gram(x, y, return_tensor=True)
        assert a.shape == (3, 2)
        gradients = compare_gradients(a, b, actual.parameters(), reference.parameters(), atol=3e-6)
        # The learned encoder must participate even when common final gates
        # cancel in a fidelity kernel without reuploading.
        scale_index = [name for name, _ in actual.qlayer.named_parameters()].index("input_scaling_params")
        assert gradients[scale_index].abs().max() > 1e-7
        before = [p.detach().clone() for p in actual.parameters()]
        optimizer = torch.optim.Adam(actual.parameters(), lr=0.01)
        optimizer.zero_grad()
        actual.gram(x, y, return_tensor=True).sum().backward()
        optimizer.step()
        assert any(not torch.equal(a, b) for a, b in zip(before, actual.parameters()))
        assert np.isfinite(actual.gram(x)).all()
    else:
        np.testing.assert_allclose(actual.gram(x, y), reference.gram(x, y), atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("embedding", ["angle", "neural"])
@pytest.mark.parametrize("backend", ["statevector", "lightning", "pp"])
@pytest.mark.parametrize("layout", ["encoding", "ansatz"])
@pytest.mark.parametrize("fisher", ["classical", "quantum"])
def test_fisher_custom_strings_against_state_finite_differences(embedding, backend, layout, fisher):
    cls = NeuralEmbedding if embedding == "neural" else AngleEmbedding
    config = {"features": 2, "rotations": ["Y 0", "YY 0 1"]}
    if embedding == "angle":
        config["input_scaling"] = True
    fm = cls(2, **config)
    scale_count = 4 if embedding == "neural" else 2
    count = scale_count + (4 if layout == "ansatz" else 0)
    opts = options(backend)
    if backend == "lightning":
        opts["diff_method"] = "parameter-shift"
    from qasp import CircuitBuilder
    builder = CircuitBuilder(2, n_ancillas=1)
    # Compile the device with QASP's normal resolution, then use the public
    # Fisher interface with a single parameter-vector QNode.
    from qasp.qkernels.quantum_kernels import EncodingAnsatz
    builder.set_ansatz(EncodingAnsatz(2, feature_map=fm))
    builder.set_measurement("probs", wires=[0, 1])
    builder.build()
    device = builder.compile(**opts).device

    def gates(weights, reference=False):
        inputs = torch.tensor([0.43, -0.71], dtype=weights.dtype)
        if reference:
            reference_encoding(fm, inputs, weights[:scale_count])
        else:
            fm(inputs, weights[:scale_count])
        if layout == "ansatz":
            qml.RZ(weights[scale_count], 0)
            qml.RZ(weights[scale_count + 1], 1)
            qml.CZ([0, 1])
            qml.RY(weights[scale_count + 2], 0)
            qml.RY(weights[scale_count + 3], 1)

    @qml.qnode(device, interface="torch", diff_method=opts["diff_method"])
    def node(weights):
        gates(weights)
        return qml.probs(wires=[0, 1])

    @qml.qnode(qml.device("default.qubit", wires=2), interface="torch", diff_method="backprop")
    def state(weights):
        gates(weights, reference=True)
        return qml.state()

    weights = torch.linspace(0.3, 1.1, count).requires_grad_()
    kernel = QFKernel(node, weights, fisher=fisher, lambda_reg=0.02)
    psi = state(weights).detach().numpy()
    eps = 1e-5
    plus = [state(weights + eps * v).detach().numpy() for v in torch.eye(count)]
    minus = [state(weights - eps * v).detach().numpy() for v in torch.eye(count)]
    state_jac = np.column_stack([(a-b)/(2*eps) for a,b in zip(plus,minus)])
    p = np.abs(psi)**2
    jac = 2 * np.real(psi.conj()[:, None] * state_jac)
    scores = jac / p[:, None]
    if fisher == "classical":
        fim = jac.T @ scores
    else:
        tangent = state_jac.conj().T @ psi
        fim = 4 * np.real(state_jac.conj().T @ state_jac - np.outer(tangent, tangent.conj()))
    f_reg = fim + 0.02 * np.eye(count)
    expected = scores @ np.linalg.solve(f_reg, scores.T)
    np.testing.assert_allclose(kernel.Freg, f_reg, atol=1e-6, rtol=1e-6)
    outcomes = [[0, 0], [0, 1], [1, 0], [1, 1]]
    np.testing.assert_allclose(kernel.gram(outcomes, normalize=False), expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("embedding", ["angle", "neural"])
def test_policy_and_fidelity_drawings_share_encoding_weights(embedding):
    cfg = components(embedding)
    x = np.linspace(-0.4, 0.6, cfg["feature_map_config"]["features"])
    policy = make_policy("softmax", {**cfg, **options("statevector")})
    tex = draw_policy(model=policy, inputs=x, output="tikz", tikz_options={"symbolic_parameters": True})
    kernel = create_kernel({**cfg, "kernel_type": "fidelity", **options("statevector")}, 2)
    overlap = draw_kernel(kernel, x=x, y=-x, output="tikz", tikz_options={"symbolic_parameters": True})
    assert r"\theta_{8}" in tex and r"\theta_{9}" not in tex
    for layer in (1, 2):
        for index in ((0, 1) if embedding == "neural" else (1, 2)):
            symbol = rf"\lambda}}_{{{layer},{index}}}" if embedding == "neural" else rf"\lambda_{{{layer},{index}}}"
            assert symbol in tex
            assert overlap.count(symbol) == 2
    for index in range(1, 9):
        assert overlap.count(rf"\theta_{{{index}}}") == 2


@pytest.mark.parametrize("embedding", ["angle", "neural"])
@pytest.mark.parametrize("backend", ["iqp_tn", "iqp_vn", "iqp_auto"])
def test_iqp_compatible_custom_strings(embedding, backend):
    cfg = components(embedding, axis="X")
    reference_cfg = deepcopy(cfg)
    reference_cfg["feature_map"] = ReferenceNeural if embedding == "neural" else ReferenceAngle
    actual = make_policy("softmax", {**cfg, **options(backend)})
    reference = make_policy("softmax", {**reference_cfg, **options("statevector")})
    set_weights(actual.parameters())
    reference.load_state_dict(actual.state_dict())
    x = torch.linspace(-0.7, 0.8, 3 * cfg["feature_map_config"]["features"]).reshape(3, -1).requires_grad_()
    compare_gradients(actual(x), reference(x), [*actual.parameters(), x], [*reference.parameters(), x])
    for config in (cfg, reference_cfg):
        config.update(kernel_type="projected", observables=["Z 0", "ZZ 0 1"])
    kernel = create_kernel({**cfg, **options(backend)}, 2)
    baseline = create_kernel({**reference_cfg, **options("statevector")}, 2)
    kernel.qlayer.double()
    baseline.qlayer.double()
    set_weights(kernel.parameters())
    baseline.qlayer.load_state_dict(kernel.qlayer.state_dict())
    inputs = x.detach().numpy()
    compare_gradients(kernel.gram(inputs, return_tensor=True), baseline.gram(inputs, return_tensor=True),
                      kernel.parameters(), baseline.parameters(), atol=3e-6)


@pytest.mark.parametrize("embedding", ["angle", "neural"])
@pytest.mark.parametrize("layout", ["encoding", "ansatz", "reupload"])
@pytest.mark.parametrize("kind", ["fidelity", "projected"])
def test_kernel_strings_against_independent_state_overlaps(embedding, layout, kind):
    """Verify the whole Gram construction, including shared inverse weights."""
    cfg = kernel_config(embedding, layout, kind)
    kernel = create_kernel({**cfg, **options("statevector")}, 2)
    if kernel.trainable:
        kernel.qlayer.double()
        set_weights(kernel.parameters())
        scales = kernel.qlayer.input_scaling_params
        angles = kernel.qlayer.params
    else:
        scales = torch.tensor(cfg["input_scaling_params"])
        angles = torch.empty(0)
    width = 8 if embedding == "neural" else 2

    @qml.qnode(qml.device("default.qubit", wires=2), interface="torch", diff_method="backprop")
    def state(x, scales, angles):
        if layout == "encoding":
            for _ in range(2):
                reference_encoding(kernel.fm, x, scales)
        else:
            for layer in range(2):
                if layer == 0 or layout == "reupload":
                    reference_encoding(kernel.fm, x, scales[layer * width:(layer + 1) * width])
                qml.RZ(angles[4 * layer], 0)
                qml.RZ(angles[4 * layer + 1], 1)
                qml.RY(angles[4 * layer + 2], 0)
                qml.RY(angles[4 * layer + 3], 1)
                if layer == 0:
                    qml.CZ([0, 1])
        return qml.state()

    x = np.linspace(-0.7, 0.8, 3 * kernel.nfeat).reshape(3, kernel.nfeat)
    # Match the public trainable kernel's documented float32 feature conversion.
    inputs = torch.as_tensor(x, dtype=torch.float32 if kernel.trainable else torch.float64)
    states = torch.stack([state(row, scales, angles) for row in inputs])
    if kind == "fidelity":
        expected = (states.conj() @ states.T).abs().square()
    else:
        z_values = states.real.new_tensor([[1, 1, -1, -1], [1, -1, 1, -1], [1, -1, -1, 1]])
        features = states.abs().square() @ z_values.T
        distance = ((features[:, None] - features[None, :]).square() * torch.tensor(cfg["weights"])).sum(-1)
        expected = torch.exp(-cfg["gamma"] * distance)
    actual = kernel.gram(x, return_tensor=kernel.trainable)
    if kernel.trainable:
        compare_gradients(actual, expected.to(actual.dtype), kernel.parameters(), kernel.parameters(), atol=3e-6)
    else:
        np.testing.assert_allclose(actual, expected.detach().numpy(), atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("embedding", ["angle", "neural"])
@pytest.mark.parametrize("backend", ["statevector", "lightning", "pp"])
def test_custom_strings_contextual_bandit_training(embedding, backend):
    from qasp.rl.torchrl import ContextualBanditTrainer, ContextualBanditEnv, BanditAWRConfig

    cfg = components(embedding, reupload=True)
    policy = make_policy("born", {**cfg, **options(backend)}).float()
    set_weights(policy.parameters())
    before = [p.detach().clone() for p in policy.parameters()]
    env = ContextualBanditEnv(n_features=cfg["feature_map_config"]["features"],
                             n_actions=2, n_contexts=16, seed=37)
    trainer = ContextualBanditTrainer(env, policy, n_actions=2,
                cfg=BanditAWRConfig(batch_size=5, updates_per_epoch=2, device="cpu", seed=37))
    try:
        history = trainer.train(epochs=2, episodes_per_epoch=8, log_every=1)
        assert len(history) == 2 and np.isfinite(history).all()
        assert all(torch.isfinite(p).all() for p in policy.parameters())
        assert any(not torch.equal(a, b) for a, b in zip(before, policy.parameters()))
        assert policy.qlayer.input_scaling_params.grad is not None
        assert torch.isfinite(policy.qlayer.input_scaling_params.grad).all()
        assert policy.qlayer.input_scaling_params.grad.abs().max() > 0
    finally:
        env.close()


@pytest.mark.parametrize("embedding", ["angle", "neural"])
@pytest.mark.parametrize("input_kind", ["numpy", "list", "torch", "omitted"])
@pytest.mark.parametrize("output", ["matplotlib", "tikz"])
def test_policy_drawing_accepts_input_containers(embedding, input_kind, output, monkeypatch):
    config = components(embedding)
    policy = make_policy("softmax", {**config, **options("statevector")})
    features = config["feature_map_config"]["features"]
    data = np.linspace(-0.4, 0.7, features)
    arguments = {}
    if input_kind == "omitted":
        counts = []
        def random_inputs(count):
            counts.append(count)
            return data
        monkeypatch.setattr(np.random, "random", random_inputs)
    else:
        arguments["inputs"] = {"numpy": data, "list": data.tolist(), "torch": torch.tensor(data)}[input_kind]
    result = draw_policy(model=policy, output=output, **arguments)
    if output == "matplotlib":
        import matplotlib.pyplot as plt
        figure, _ = result
        assert figure.axes
        plt.close(figure)
    else:
        assert r"\begin{quantikz}" in result
    if input_kind == "omitted":
        assert counts == [features]


@pytest.mark.parametrize("device_instance", [False, True])
@pytest.mark.parametrize("word", ["YY", "XYZ", "IXY"])
def test_mps_pauli_rotation_decomposition_values_and_gradients(device_instance, word):
    from qasp import AngleAnsatz, CircuitBuilder

    pytest.importorskip("quimb")
    pytest.importorskip("cotengra")
    def builder():
        result = CircuitBuilder(3)
        words = ["Y 0", f"{word} " + " ".join(map(str, range(len(word)))), "Y 1"]
        result.set_ansatz(AngleAnsatz(3, n_layers=2, rotations=words,
                         include_imprimitive=True, imprimitive="CZ", include_final_imprimitive=False))
        result.set_measurement("expectation", observables=["Z 0", "Z 1", "ZZ 0 1"])
        result.build()
        return result
    device = qml.device("default.tensor", wires=4, method="mps") if device_instance else "default.tensor"
    node = builder().compile(device=device, method="mps", diff_method="parameter-shift")
    reference = builder().compile(device="default.qubit", diff_method="backprop")
    if device_instance:
        assert node.device is device
    weights = torch.linspace(0.2, 0.9, 6).requires_grad_()
    compare_gradients(torch.stack(node(params=weights)), torch.stack(reference(params=weights)),
                      [weights], [weights], atol=1e-8)
