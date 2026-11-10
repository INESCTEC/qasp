"""Known Fourier coefficients and off-grid predictions of public QASP models."""

import numpy as np
import pennylane as qml
import pytest
import torch

from qasp import (
    AngleAnsatz,
    AngleEmbedding,
    CircuitBuilder,
    FidelityKernel,
    IQPAnsatz,
    NeuralEmbedding,
    PQCClassifier,
    ProjectedKernel,
    expressivity_analyzer,
)
from qasp.rl import LearnablePQCBorn, PQCBornVariance, PQCSoftmax, PQDQN


def builder(n=1, features=1, rotations="Y", neural=False, scaled=False, **ansatz):
    b = CircuitBuilder(n)
    if neural:
        fm = NeuralEmbedding(n, rotations=rotations, features=features)
    else:
        fm = AngleEmbedding(n, rotations=rotations, features=features, input_scaling=scaled)
    b.set_ansatz(AngleAnsatz(n, feature_map=fm, **ansatz))
    return b


def classifier(b, **kwargs):
    return PQCClassifier(circuit_builder=b, n_qubits=b.n_qubits, init="zeros", **kwargs).double()


def test_phase_sign_active_support_and_live_weights():
    model = classifier(builder(scaled=True, rotations="X"), readout="expectation")
    with torch.no_grad():
        model.qlayer.input_scaling_params.fill_(-np.sqrt(2))
    model.eval()
    model.qlayer.params.grad = torch.ones_like(model.qlayer.params)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.inference_mode():
        s = expressivity_analyzer(model, observables=["Y 0", "Z 0", "I 0", "X 0"])
    # RX(-sqrt(2)*x) yields +sin(sqrt(2)*x) for Y and cos(sqrt(2)*x) for Z.
    np.testing.assert_allclose(s.frequencies[:, 0], [-np.sqrt(2), 0, np.sqrt(2)])
    np.testing.assert_allclose(
        s.coefficients, [[-0.5j, 0.5, 0, 0], [0, 0, 1, 0], [0.5j, 0.5, 0, 0]], atol=1e-12
    )
    np.testing.assert_array_equal(s.active_counts, [2, 2, 1, 0])
    np.testing.assert_array_equal(s.magnitudes, abs(s.coefficients))
    assert not model.training
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, before[k])
    torch.testing.assert_close(model.qlayer.params.grad, torch.ones_like(model.qlayer.params))
    with torch.no_grad():
        model.qlayer.input_scaling_params.zero_()
    constant = expressivity_analyzer(model, observables=["Y 0", "Z 0"])
    np.testing.assert_array_equal(constant.frequencies, [[0]])
    np.testing.assert_allclose(constant.evaluate([[0.2], [-1.7]]), [[0, 1], [0, 1]])


@pytest.mark.parametrize("rotations", ["Y", ["Y 0", "XZ 0 1"]])
def test_neural_projection_reupload_and_classifier_groups(rotations):
    b = builder(
        2,
        2,
        rotations=rotations,
        neural=True,
        n_layers=2,
        reuploading=True,
        include_imprimitive=True,
    )
    model = classifier(b, readout="expectation", observables=["X 0", "ZZ 0 1"], output_scaling=-2)
    weights = torch.tensor(
        [0.375, -0.75, -0.25, 0.625, 0.375, -0.75, 0.125, 0.25], dtype=torch.float64
    )
    with torch.no_grad():
        model.qlayer.params.copy_(torch.linspace(-0.8, 0.9, model.qlayer.params.numel()))
        model.qlayer.input_scaling_params.copy_(weights)
    result = expressivity_analyzer(model)
    assert result.n_evaluations == 5 * 3 * 3
    vectors = weights.reshape(4, 2).numpy()
    from itertools import product

    expected = {tuple(h @ vectors) for h in np.array(list(product([-1, 0, 1], repeat=4)))}
    assert set(map(tuple, result.frequencies)) == expected
    x = torch.tensor([[0.12, -0.6], [1.6, 0.41], [-0.39, 0.87]], dtype=torch.float64)
    expected_scores = np.array([[model(row).item()] for row in x])
    np.testing.assert_allclose(result.evaluate(x), expected_scores, atol=1e-12)


def test_collisions_add_complex_coefficients_before_taking_magnitudes():
    # Three uploads along commensurate but unequal slopes: RY(x) RY(2x) RY(3x).
    model = classifier(builder(n_layers=3, reuploading=True, scaled=True), readout="expectation")
    with torch.no_grad():
        model.qlayer.input_scaling_params.copy_(torch.tensor([1.0, 2.0, 3.0]))
    s = expressivity_analyzer(model, observables="Z 0")
    np.testing.assert_array_equal(s.frequencies[:, 0], np.arange(-6, 7))
    np.testing.assert_array_equal(s.active_counts, [2])
    np.testing.assert_allclose(s.coefficients[[0, -1]], 0.5, atol=1e-12)
    np.testing.assert_allclose(s.evaluate([[0.7], [1.3]])[:, 0], np.cos(6 * np.array([0.7, 1.3])))


def test_close_frequencies_and_multi_axis_projection():
    fm = NeuralEmbedding(2, features=2, rotations="XY")
    k = ProjectedKernel(
        n_qubits=2,
        feature_map=fm,
        input_scaling_params=[0.5, -0.25, 0.5 + 2**-39, -0.25],
        observables=["X 1"],
    )
    s = expressivity_analyzer(k)
    assert s.n_frequencies == 25
    assert s.n_evaluations == 25
    x = [[0.72, 0.14], [-0.97, 0.4]]
    np.testing.assert_allclose(s.evaluate(x), k.features(x), atol=1e-12)


@pytest.mark.parametrize("readout", ["probabilities", "expectation"])
def test_classifier_matches_live_output(readout):
    b = builder(3, 2, include_imprimitive=True)
    model = classifier(b, readout=readout, n_classes=3)
    x = torch.tensor([0.37, -0.91], dtype=torch.float64)
    np.testing.assert_allclose(
        expressivity_analyzer(model).evaluate(x), model(x).detach(), atol=1e-12
    )


def test_q_values_include_affine_head_in_embedding_coordinates():
    model = PQDQN(
        n_actions=2,
        n_qubits=2,
        circuit_builder=builder(2, 2),
        observables=[["Z 0", "ZZ 0 1"], ["X 1"]],
        fc_bias=True,
    ).double()
    with torch.no_grad():
        model.fc_layer.weight.copy_(torch.tensor([[2.0, -1.0], [0.0, 0.0]]))
        model.fc_layer.bias.copy_(torch.tensor([-0.1, 0.25]))
    s = expressivity_analyzer(model)
    assert s.input_activation == "tanh"
    assert s.output_labels == ("Q[0]", "Q[1]")
    assert s.active_counts[1] == 1
    x = torch.tensor([[0.14, -0.33], [0.23, 2.5]], dtype=torch.float64)
    np.testing.assert_allclose(s.evaluate(x.tanh()), model(x).detach(), atol=1e-12)


@pytest.mark.parametrize("policy", ["softmax", "probs", "expectation", "auto"])
def test_policy_spectrum_matches_quantum_readout(policy):
    kwargs = dict(
        n_qubits=2,
        n_actions=3,
        ansatz=IQPAnsatz,
        ansatz_config=dict(n_qubits=2, n_layers=2, reuploading=True, imprimitive_pattern="chain"),
        feature_map=AngleEmbedding,
        feature_map_config=dict(n_qubits=2, features=2, rotations="Z"),
    )
    if policy == "softmax":
        model = PQCSoftmax(**kwargs, fc_layer=True, output_scaling=1.7, observables=["Z 0", "X 1"])
    else:
        model = PQCBornVariance(
            **kwargs,
            policy="parity-like",
            training_method="expectation" if policy == "expectation" else "probs",
            device="auto" if policy == "auto" else "default.qubit",
        )
    model.double()
    s = expressivity_analyzer(model)
    x = torch.tensor([0.48, -0.31], dtype=torch.float64)
    np.testing.assert_allclose(s.evaluate(x), model(x).detach().reshape(-1), atol=1e-10)


@pytest.mark.parametrize("trainable", [False, True])
def test_joint_and_conditional_local_fidelity(trainable):
    options = (
        dict(
            ansatz=AngleAnsatz,
            ansatz_config=dict(n_layers=1, include_imprimitive=True),
            input_scaling_init="ones",
        )
        if trainable
        else dict(input_scaling_params=[0.25, -0.375, 0.5, 0.125])
    )
    k = FidelityKernel(
        n_qubits=2,
        feature_map=NeuralEmbedding(2, features=2, rotations="Y"),
        projectors=["P 0", "P 1"],
        **options,
    )
    x = np.array([[0.2, 0.7], [-0.31, 1.2]])
    y = np.array([0.55, -0.43])
    joint = expressivity_analyzer(k)
    conditional = expressivity_analyzer(k, reference=y)
    assert (joint.frequencies.shape[1], conditional.frequencies.shape[1]) == (4, 2)
    for actual in [joint.evaluate(x, y), conditional.evaluate(x)]:
        np.testing.assert_allclose(actual[:, 0], [k.kernel(z, y) for z in x], atol=2e-7)


def test_adaptive_policy_from_config_reconstructs_a_fixed_measurement():
    config = {
        "n_qubits": 3, "n_actions": 2, "n_qubits_measure": 1,
        "feature_map": NeuralEmbedding,
        "feature_map_config": {"n_qubits": 3, "features": 2, "rotations": "Y"},
        "ansatz": AngleAnsatz,
        "ansatz_config": {"n_qubits": 3, "n_layers": 1, "rotations": "Y"},
        "init": "zeros", "input_scaling_init": "ones",
    }
    model = LearnablePQCBorn(**config).double()
    x = torch.tensor([0.25, -0.7], dtype=torch.float64)
    wires, _ = model.measurement_selector(x)
    probabilities = model.qlayer(torch.cat([x, x.new_tensor(wires)]))
    spectrum = expressivity_analyzer(model, observables=f"Z {wires[0]}")
    np.testing.assert_allclose(
        spectrum.evaluate(x), (probabilities[0] - probabilities[1]).item(), atol=1e-12
    )


def test_width_sweep_tracks_allowed_vs_observable_specific_active_frequencies():
    def factory(n):
        return classifier(builder(n, n, include_imprimitive=False))

    sweep = expressivity_analyzer(
        model_factory=factory, qubit_list=[3, 1, 2], observables=lambda n: ["Z 0", "Z" * n]
    )
    assert sweep.qubit_list == [3, 1, 2]
    np.testing.assert_array_equal(sweep.frequency_counts, [27, 3, 9])
    for n, s in sweep.results_by_qubit.items():
        np.testing.assert_array_equal(s.active_counts, [2, 2**n])
    rows = sweep.summary()
    assert len(rows) == 6
    assert rows[1]["non_dc_power"] == pytest.approx(1 / 8)


def test_guardrails_and_explicit_observables():
    m = classifier(builder(2, 2))
    with pytest.raises(ValueError, match="requires 9 evaluations"):
        expressivity_analyzer(m, max_evaluations=8)
    for not_model in [builder(), torch.nn.Linear(2, 2), object()]:
        with pytest.raises(TypeError, match="QASP RL policy"):
            expressivity_analyzer(not_model)
    with pytest.raises(ValueError, match="reference"):
        expressivity_analyzer(m, reference=[0, 0])
    with pytest.raises(ValueError, match="wire"):
        expressivity_analyzer(m, observables=qml.Z(2))
    with pytest.raises(ValueError, match="distinct positive"):
        expressivity_analyzer(model_factory=lambda n: m, qubit_list=[2, 2])
    with pytest.raises(ValueError, match="3 data qubits"):
        expressivity_analyzer(model_factory=lambda n: m, qubit_list=[3])
    s = expressivity_analyzer(m, observables=qml.Z(0) + qml.X(1), atol=10)
    np.testing.assert_array_equal(s.active_counts, [0])
    np.testing.assert_allclose(s.evaluate([0.52, -0.2]), [np.cos(0.52) + np.sin(-0.2)])


def test_long_pauli_word_and_identity_generator():
    kernel = ProjectedKernel(
        n_qubits=5,
        feature_map=AngleEmbedding(5, features=2, rotations=["XYZXY 0 1 2 3 4", "III 0 2 4"]),
        observables=["Z 0", "X 2"],
    )
    s = expressivity_analyzer(kernel)
    np.testing.assert_array_equal(s.frequencies, [[-1, 0], [0, 0], [1, 0]])
    x = [[0.81, 1.5], [-1.75, 0.37]]
    np.testing.assert_allclose(s.evaluate(x), kernel.features(x), atol=1e-12)


def test_reupload_collisions_with_noncommuting_variational_blocks():
    model = classifier(
        builder(n_layers=3, reuploading=True, scaled=True),
        readout="expectation",
        observables=["Z 0"],
    )
    with torch.no_grad():
        model.qlayer.input_scaling_params.copy_(torch.tensor([1.0, -2.0, 3.0]))
        model.qlayer.params.copy_(torch.linspace(0.81, -1.54, 6))
    s = expressivity_analyzer(model)
    assert s.n_frequencies == 13 < s.n_evaluations
    x = torch.tensor([[-1.2], [0.56], [2.13]], dtype=torch.float64)
    expected = np.array([[model(z).item()] for z in x])
    np.testing.assert_allclose(s.evaluate(x), expected, atol=1e-12)


def test_tensor_observables_snapshot_complex_entries_and_learned_scale():
    model = classifier(builder())
    scale = torch.tensor(1.25, requires_grad=True)
    # A Hermitian with a complex off-diagonal checks the observable snapshot.
    matrix = torch.tensor([[1, 1 - 1j], [1 + 1j, -1]], dtype=torch.complex128, requires_grad=True)
    s = expressivity_analyzer(model, observables=[scale * qml.Z(0), qml.Hermitian(matrix, 0)])
    x = torch.tensor([0.81], dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        scale.zero_()
        matrix.zero_()
    np.testing.assert_allclose(s.evaluate(x), [1.25 * np.cos(0.81), np.cos(0.81) + np.sin(0.81)])
    assert scale.grad is None and matrix.grad is None
