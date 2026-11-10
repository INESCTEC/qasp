"""Readout semantics, differentiability, configuration and batching regressions."""

from copy import deepcopy

import numpy as np
import pennylane as qml
import pytest
import torch

from qasp.rl import PQDQN, create_qfunction
from qasp.rl.policies import _PolicyDecoder
from qasp.rl.qfunctions.observables import parse_pauli, predefined_observables


def config(**kwargs):
    cfg = dict(
        n_qubits=4,
        n_actions=2,
        ansatz="AngleAnsatz",
        ansatz_config=dict(n_layers=2, rotations="YZ", reuploading=True),
        feature_map="AngleEmbedding",
        feature_map_config=dict(rotations="Y", input_scaling=True, features=4),
        observables=["I 0", "Z 0", "Z 1", "Z 2", "Z 3"],
        diff_method="backprop",
    )
    return dict(cfg, **kwargs)


@pytest.mark.parametrize(
    "observables",
    [
        ["Z 0", "X 1", "ZZ 0 3"],
        [["Z 0", "-Z 1"], ["X 2"]],
        "parity-like",
        "majority-modulo",
    ],
)
def test_batched_outputs_and_gradients(observables):
    torch.manual_seed(5)
    model = PQDQN(**config(observables=observables)).double()
    broadcast = PQDQN(
        **config(observables=observables, batch_mode="broadcast")
    ).double()
    broadcast.load_state_dict(model.state_dict())
    x = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    expected = model(x)
    actual = broadcast(x)
    assert actual.shape == (2, 3, 2)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(model(x[0, 0]), actual[0, 0])
    assert (model.fc_layer.bias is not None) == (
        isinstance(observables, str) or isinstance(observables[0], str)
    )
    expected.square().sum().backward()
    actual.square().sum().backward()
    for (name, p), (_, q) in zip(
        model.named_parameters(), broadcast.named_parameters()
    ):
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        torch.testing.assert_close(p.grad, q.grad)
    assert x.grad.abs().sum() > 0
    assert model.qlayer.params.grad.abs().sum() > 0
    assert model.qlayer.input_scaling_params.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "name,n_actions,wires,blocks",
    [
        ("parity-like", 2, [3, 0, 2], None),
        ("parity-like", 3, [3, 0, 2], None),
        ("majority-modulo", 2, [3, 0, 2, 1], [[3, 0, 2, 1]]),
        ("majority-modulo", 3, [3, 0, 2, 1], [[3, 0], [2, 1]]),
    ],
)
def test_decompositions_match_basis_action_projectors(name, n_actions, wires, blocks):
    strings, coefficients = predefined_observables(
        name, 4, n_actions, wires, blocks, mode="projectors"
    )
    operators = [parse_pauli(s, 4) for s in strings]
    diagonal = np.stack(
        [np.diag(qml.matrix(op, wire_order=range(4))).real for op in operators]
    )
    positions = (
        None if blocks is None else [[wires.index(w) for w in b] for b in blocks]
    )
    decoder = _PolicyDecoder(name, n_actions, len(wires), positions)
    expected = np.zeros((n_actions, 16))
    for state in range(16):
        measured = int("".join(format(state, "04b")[w] for w in wires), 2)
        expected[int(decoder.action_index[measured]), state] = 1
    np.testing.assert_allclose(coefficients @ diagonal, expected, atol=1e-12)


def test_direct_action_sums_and_unbounded_fc():
    model = PQDQN(**config(observables=[["I 0", "Z 0"], ["-Z 1"]], fc_layer=False))
    x = torch.zeros(4)
    raw = model.qlayer(x).flatten()
    torch.testing.assert_close(model(x), torch.stack([raw[0] + raw[1], raw[2]]))
    model = PQDQN(**config(observables=["I 0"]))
    with torch.no_grad():
        model.fc_layer.weight.copy_(torch.tensor([[50.0], [-20.0]]))
        model.fc_layer.bias.zero_()
    torch.testing.assert_close(model(x), torch.tensor([50.0, -20.0]))


def test_factory_preserves_dictionary_and_serialization():
    cfg = config()
    original = deepcopy(cfg)
    model = create_qfunction(cfg)
    assert cfg == original
    restored = PQDQN.from_config(cfg)
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(model(torch.ones(4)), restored(torch.ones(4)))
    from qasp.rl.model import create_policy

    assert isinstance(create_policy("PQDQN", cfg, 4), PQDQN)


@pytest.mark.parametrize("spec", ["", "A 0", "ZZ 0", "Z 4", "ZZ 0 0", "Z", "Z x"])
def test_invalid_strings(spec):
    with pytest.raises(ValueError):
        PQDQN(**config(observables=[spec]))


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(observables=[]),
        dict(observables=[["Z 0"]]),
        dict(observables="unknown"),
        dict(observables=["Z 0"], fc_layer=False),
        dict(observables="majority-modulo", majority_blocks=[[0, 0]]),
        dict(observables="parity-like", measurement_qubits=[0, 0]),
        dict(observables="parity-like", preset_actions=1),
        dict(observables="parity-like", preset_actions=32),
        dict(observables="parity-like", preset_mode="invalid"),
        dict(observables=["Z 0"], preset_actions=4),
        dict(fc_bias="yes"),
    ],
)
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        PQDQN(**config(**kwargs))


def test_parameter_shift_and_neural_embedding():
    model = PQDQN(
        **config(
            diff_method="parameter-shift",
            feature_map="NeuralEmbedding",
            feature_map_config=dict(rotations="Y", features=4),
        )
    )
    model(torch.randn(2, 4)).square().sum().backward()
    assert model.qlayer.params.grad.abs().sum() > 0
    assert model.qlayer.input_scaling_params.grad.abs().sum() > 0


def test_finite_shots_and_iqp_with_explicit_rotations():
    model = PQDQN(
        **config(
            n_qubits=2,
            ansatz="IQPAnsatz",
            ansatz_config=dict(n_layers=1),
            feature_map_config=dict(
                features=2, rotations=["Y 0", "Y 1"], input_scaling=True
            ),
            observables=["Z 0", "X 1"],
            diff_method="parameter-shift",
            shots=100,
            device_kwargs={"seed": 7},
        )
    )
    result = model(torch.tensor([[0.1, 0.2], [0.2, -0.1]]))
    assert result.shape == (2, 2)
    result.square().sum().backward()
    assert torch.isfinite(model.qlayer.params.grad).all()


def test_circuit_builder_is_copied_and_singleton_batches():
    from qasp.circuits import AngleAnsatz, CircuitBuilder
    from qasp.feature_maps import AngleEmbedding

    builder = CircuitBuilder(n_qubits=2)
    builder.set_ansatz(
        AngleAnsatz(
            n_qubits=2, feature_map=AngleEmbedding(2, rotations="Y", features=2)
        )
    )
    builder.set_measurement("probs")
    model = PQDQN(
        n_qubits=2,
        n_actions=2,
        observables=["Z 0"],
        circuit_builder=builder,
        batch_mode="broadcast",
    )
    assert builder.measurement[0] == "probs"
    assert model(torch.zeros(1, 2)).shape == (1, 2)
    assert model(torch.zeros(0, 2)).shape == (0, 2)
    assert model(torch.zeros(2)).shape == (2,)


@pytest.mark.parametrize(
    "observables",
    [["Z 0", "X 1"], [["Z 0", "-Z 1"], ["Z 2"]], "parity-like", "majority-modulo"],
)
def test_draw_qfunction_tikz_and_legacy_entry_point(observables):
    from qasp import draw_qfunction
    from qasp.rl import draw_policy

    model = PQDQN(**config(observables=observables)).double()
    # Drawing must preserve trained weights, gradients, and mode.
    model(torch.ones(4)).square().sum().backward()
    with torch.no_grad():
        model.qlayer.params.add_(0.25)
    model.eval()
    before = {name: value.clone() for name, value in model.state_dict().items()}
    gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
    options = {"standalone": True, "symbolic_parameters": True}
    source = model.draw(output="tikz", tikz_options=options)
    assert r"\documentclass" in source
    assert r"\begin{quantikz}" in source
    assert r"\theta" in source
    assert source == draw_qfunction(model, output="tex", tikz_options=options)
    assert source == draw_policy(model=model, output="tikz", tikz_options=options)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])
    for name, value in model.named_parameters():
        torch.testing.assert_close(value.grad, gradients[name])
    assert not model.training
    assert options == {"standalone": True, "symbolic_parameters": True}


def test_drawing_from_config_infers_neural_feature_count_without_mutation():
    from qasp.rl import draw_policy, draw_qfunction

    cfg = config(
        feature_map="NeuralEmbedding",
        feature_map_config={"rotations": "Y", "features": 3},
    )
    before = deepcopy(cfg)
    assert r"\begin{quantikz}" in draw_qfunction(config=cfg, output="tikz")
    assert r"\begin{quantikz}" in draw_policy(
        policy_class="PQDQN", policy_config_kwargs=cfg, n_qubits=4, output="tikz"
    )
    assert cfg == before
    model = PQDQN(**cfg)
    with pytest.raises(ValueError, match="one observation"):
        model.draw(inputs=torch.zeros(2, 3), output="tikz")


def test_drawing_uses_current_weights_and_forward_input_activation():
    import matplotlib.pyplot as plt

    model = PQDQN(**config()).double()
    with torch.no_grad():
        model.qlayer.input_scaling_params.fill_(1.0)
        model.qlayer.params.fill_(0.123456)
    x = torch.tensor([2.0, 0.1, -0.2, 0.3], dtype=torch.float64)
    source = model.draw(inputs=x, output="tikz", decimals=6)
    assert "0.123456" in source
    assert f"{torch.tanh(x[0]).item():.6f}" in source
    model.input_activation = None
    assert source != model.draw(inputs=x, output="tikz", decimals=6)
    assert "0.123456" in model.draw(inputs=x, output="text", decimals=6)
    figure, _ = model.draw(inputs=x, output="matplotlib")
    assert figure.axes
    plt.close(figure)


@pytest.mark.parametrize("ansatz", ["AngleAnsatz", "IQPAnsatz"])
def test_paper_drawing_compresses_complete_reuploading_layers(ansatz):
    from qasp import draw_qfunction
    from qasp.circuits.tikz import _builder_compression_metadata

    model = PQDQN(
        **config(ansatz=ansatz, ansatz_config={"n_layers": 4, "reuploading": True})
    )
    state = model.qlayer.state_dict()
    tape = qml.tape.make_qscript(model.circuit_builder.circuit)(
        inputs=torch.zeros(4),
        params=state["params"],
        input_scaling_params=state["input_scaling_params"],
    )
    metadata = _builder_compression_metadata(model.circuit_builder, tape)
    start, count = (
        metadata["first_layer_start"],
        metadata["first_layer_operation_count"],
    )
    assert tape.operations[start].name == "AngleEmbedding"
    assert tape.operations[start + count].name == "AngleEmbedding"
    assert len(tape.operations) == start + metadata["layer_operation_count"]
    options = {
        "compress": True,
        "standalone": True,
        "symbolic_parameters": True,
        "gate_colors": {"single": "DCECF8"},
    }
    source = draw_qfunction(model, output="tikz", tikz_options=options)
    assert r"U_{4}(\mathbf{f};\Theta_{4})" in source
    assert r"\lambda_{1,1}" in source
    assert r"\lambda_{2,1}" not in source
    assert "inner sep=3pt" in source
    full = model.draw(output="tikz", tikz_options={"symbolic_parameters": True})
    assert r"\lambda_{4,4}" in full


@pytest.mark.parametrize(
    "a,expected",
    [
        (2, ["ZZZZ 0 1 2 3"]),
        (4, ["ZZZZ 0 1 2 3", "ZZZ 0 1 2", "Z 3"]),
    ],
)
def test_parity_features_and_independent_preset_size(a, expected):
    model = PQDQN(**config(observables="parity", preset_actions=a))
    assert list(model.observable_strings) == expected
    assert model.fc_layer.in_features == a - 1
    assert model.fc_layer.out_features == 2
    x = torch.tensor([0.2, -0.4, 0.1, 0.3])
    raw = model.qlayer(torch.tanh(x)).flatten()
    torch.testing.assert_close(model.quantum_features(x), raw)
    torch.testing.assert_close(
        model(x), model.fc_layer.weight @ raw + model.fc_layer.bias
    )


def test_majority_features_are_individual_paulis_not_fixed_majority():
    # Majority on three bits: (Z0 + Z1 + Z2 - Z0 Z1 Z2)/2.
    strings, coefficients = predefined_observables("majority", 3, 2)
    assert strings == ["ZZZ 0 1 2", "Z 0", "Z 1", "Z 2"]
    np.testing.assert_array_equal(coefficients, np.eye(4))
    # Two majority bits on blocks of size 3 and 1: union of block supports
    # and products. No fixed coefficients or probability reconstruction.
    model = PQDQN(
        **config(
            observables="majority", preset_actions=4, majority_blocks=[[0, 1, 2], [3]]
        )
    )
    expected = {
        "ZZZZ 0 1 2 3",
        "ZZZ 0 1 2",
        "ZZ 0 3",
        "ZZ 1 3",
        "ZZ 2 3",
        "Z 0",
        "Z 1",
        "Z 2",
        "Z 3",
    }
    assert set(model.observable_strings) == expected
    assert model.fc_layer.in_features == 9 and model.fc_layer.out_features == 2
    raw = model.qlayer(torch.tanh(torch.ones(4))).flatten()
    torch.testing.assert_close(model.quantum_features(torch.ones(4)), raw)


def test_affine_head_bias_and_explicit_legacy_configuration():
    for observables in [["Z 0"], "parity", "majority"]:
        model = PQDQN(**config(observables=observables))
        with torch.no_grad():
            model.fc_layer.weight.zero_()
            model.fc_layer.bias.copy_(torch.tensor([4.0, -3.0]))
        torch.testing.assert_close(model(torch.zeros(4)), torch.tensor([4.0, -3.0]))
        model(torch.ones(4)).sum().backward()
        torch.testing.assert_close(model.fc_layer.bias.grad, torch.ones(2))
    legacy = PQDQN(
        **config(observables="parity", preset_mode="projectors", fc_bias=False)
    )
    assert legacy.fc_layer.bias is None
    torch.testing.assert_close(
        legacy.quantum_features(torch.ones(4)).sum(), torch.tensor(1.0)
    )
