import pennylane as qml
import pytest
import torch

from qasp.circuits import CircuitBuilder
from qasp.classifiers import (
    ClassProbabilityDecoder,
    ClassifierTrainer,
    PQCClassifier,
    accuracy_score,
    average_precision_score,
    binary_cross_entropy,
    balanced_majority_blocks,
    categorical_cross_entropy,
    classification_metrics,
    confusion_matrix,
    create_classifier,
    draw_classifier,
    hinge_loss,
    mean_squared_error,
    pr_auc_score,
    roc_auc_score,
)


class _FeatureMap:
    input_scaling = False


class _ToyAnsatz:
    feature_map = _FeatureMap()
    n_layers = 1

    def __init__(self, n_qubits):
        self.n_qubits = n_qubits

    def __call__(self, inputs=None, input_scaling_params=None, params=None):
        del input_scaling_params
        for wire in range(self.n_qubits):
            qml.RY(params[wire] + inputs[..., wire], wires=wire)
        for wire in range(self.n_qubits - 1):
            qml.CNOT(wires=[wire, wire + 1])

    def get_param_shapes(self):
        return ((self.n_qubits,),)


def _builder(n_qubits):
    builder = CircuitBuilder(n_qubits=n_qubits)
    builder.set_ansatz(_ToyAnsatz(n_qubits))
    return builder


def test_contiguous_decoder_is_balanced_and_differentiable():
    decoder = ClassProbabilityDecoder("contiguous", 3, 3)
    assert decoder.class_index.tolist() == [0, 0, 0, 1, 1, 1, 2, 2]

    logits = torch.arange(8, dtype=torch.float64, requires_grad=True)
    probabilities = torch.softmax(logits, dim=-1)
    class_probabilities = decoder(probabilities)
    torch.testing.assert_close(class_probabilities.sum(), torch.ones((), dtype=torch.float64))
    (-torch.log(class_probabilities[1])).backward()
    assert torch.isfinite(logits.grad).all()


def test_parity_and_majority_decoders_match_sample_decoding():
    parity = ClassProbabilityDecoder("parity", 3, 3)
    samples = torch.tensor(
        [[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 1, 1]]
    )
    basis_indices = torch.tensor([0, 1, 2, 7])
    torch.testing.assert_close(
        parity.decode_samples(samples), parity.class_index[basis_indices]
    )

    majority = ClassProbabilityDecoder(
        "majority", 4, 6, majority_blocks=((0, 1, 2), (3, 4, 5))
    )
    majority_samples = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 1, 1, 0],
        ]
    )
    torch.testing.assert_close(
        majority.decode_samples(majority_samples), torch.tensor([0, 1, 2, 3])
    )


@pytest.mark.parametrize(
    ("n_classes", "n_qubits", "blocks"),
    [
        (2, 3, ((0, 1, 2),)),
        (3, 6, ((0, 1, 2), (3, 4, 5))),
        (4, 6, ((0, 1, 2), (3, 4, 5))),
        (5, 9, ((0, 1, 2), (3, 4, 5), (6, 7, 8))),
    ],
)
def test_majority_decoder_matches_exhaustive_reference_definition(
    n_classes, n_qubits, blocks
):
    decoder = ClassProbabilityDecoder(
        "majority",
        n_classes,
        n_qubits,
        majority_blocks=blocks,
    )
    expected = []
    n_code_bits = len(blocks)
    for basis_index in range(1 << n_qubits):
        bits = [int(bit) for bit in format(basis_index, f"0{n_qubits}b")]
        majority_bits = [
            int(sum(bits[position] for position in block) > len(block) // 2)
            for block in blocks
        ]
        code = sum(
            bit << (n_code_bits - 1 - index)
            for index, bit in enumerate(majority_bits)
        )
        expected.append(code % n_classes)

    torch.testing.assert_close(decoder.class_index, torch.tensor(expected))
    uniform_basis = torch.full((1 << n_qubits,), 1.0 / (1 << n_qubits))
    torch.testing.assert_close(
        decoder(uniform_basis), decoder.reference_class_prior
    )


def test_majority_block_balance_and_strict_validation():
    wire_blocks, position_blocks, unused = balanced_majority_blocks(
        range(4), 4
    )
    assert wire_blocks == ((0, 1, 2), (3,))
    assert position_blocks == wire_blocks
    assert unused == tuple()
    decoder = ClassProbabilityDecoder(
        "majority", 4, 4, majority_blocks=position_blocks
    )
    torch.testing.assert_close(decoder.basis_state_counts, torch.full((4,), 4))

    with pytest.raises(ValueError, match="equal odd block sizes"):
        balanced_majority_blocks(
            range(4),
            4,
            majority_blocks=((0, 1), (2, 3)),
            strict=True,
        )

    modulo_decoder = ClassProbabilityDecoder(
        "majority",
        3,
        6,
        majority_blocks=((0, 1, 2), (3, 4, 5)),
    )
    torch.testing.assert_close(
        modulo_decoder.basis_state_counts, torch.tensor([32, 16, 16])
    )


def test_classifier_losses_cover_binary_and_multiclass_objectives():
    binary_scores = torch.tensor([-0.8, 0.7], requires_grad=True)
    binary_targets = torch.tensor([0, 1])
    assert float(hinge_loss(binary_scores, binary_targets).detach()) == pytest.approx(0.25)
    assert float(mean_squared_error(binary_scores, binary_targets).detach()) == pytest.approx(0.065)

    binary_probs = torch.tensor([[0.9, 0.1], [0.2, 0.8]], requires_grad=True)
    bce = binary_cross_entropy(binary_probs, binary_targets)
    expected_bce = -0.5 * (torch.log(torch.tensor(0.9)) + torch.log(torch.tensor(0.8)))
    assert float(bce.detach()) == pytest.approx(float(expected_bce))

    class_probs = torch.tensor(
        [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]], requires_grad=True
    )
    targets = torch.tensor([0, 2])
    cce = categorical_cross_entropy(class_probs, targets)
    cce.backward()
    assert torch.isfinite(class_probs.grad).all()

    class_scores = torch.tensor([[1.2, 0.2, -0.1], [0.1, 0.7, 0.4]])
    assert float(hinge_loss(class_scores, torch.tensor([0, 1]))) == pytest.approx(0.35)


def test_metrics_match_known_confusion_matrix():
    targets = torch.tensor([0, 0, 1, 1, 2, 2])
    predictions = torch.tensor([0, 1, 1, 1, 2, 0])
    expected = torch.tensor([[1, 1, 0], [0, 2, 0], [1, 0, 1]])
    torch.testing.assert_close(confusion_matrix(targets, predictions, 3), expected)
    assert accuracy_score(targets, predictions) == pytest.approx(4 / 6)
    report = classification_metrics(targets, predictions, n_classes=3)
    assert set(report) >= {
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "confusion_matrix",
    }


def test_auc_metrics_match_sklearn_on_imbalanced_binary_data_with_ties():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    targets = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    positive_scores = torch.tensor(
        [0.05, 0.10, 0.20, 0.20, 0.30, 0.40, 0.50, 0.55, 0.80, 0.55, 0.90]
    )
    probabilities = torch.stack((1.0 - positive_scores, positive_scores), dim=1)

    expected_roc = sklearn_metrics.roc_auc_score(targets, positive_scores)
    expected_pr = sklearn_metrics.average_precision_score(targets, positive_scores)
    assert float(roc_auc_score(targets, probabilities)) == pytest.approx(expected_roc)
    assert float(pr_auc_score(targets, probabilities)) == pytest.approx(expected_pr)
    assert float(average_precision_score(targets, positive_scores)) == pytest.approx(
        expected_pr
    )

    report = classification_metrics(
        targets,
        (positive_scores >= 0.5).long(),
        n_classes=2,
        probabilities=probabilities,
    )
    assert float(report["roc_auc"]) == pytest.approx(expected_roc)
    assert float(report["pr_auc"]) == pytest.approx(expected_pr)


def test_multiclass_auc_metrics_match_sklearn_one_vs_rest():
    sklearn_metrics = pytest.importorskip("sklearn.metrics")
    targets = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2])
    probabilities = torch.tensor(
        [
            [0.80, 0.10, 0.10],
            [0.55, 0.30, 0.15],
            [0.45, 0.40, 0.15],
            [0.20, 0.65, 0.15],
            [0.35, 0.50, 0.15],
            [0.10, 0.15, 0.75],
            [0.15, 0.35, 0.50],
            [0.25, 0.25, 0.50],
            [0.20, 0.45, 0.35],
        ]
    )
    encoded = torch.nn.functional.one_hot(targets, num_classes=3)

    expected_roc = sklearn_metrics.roc_auc_score(
        targets, probabilities, multi_class="ovr", average="weighted"
    )
    expected_pr = sklearn_metrics.average_precision_score(
        encoded, probabilities, average="weighted"
    )
    assert float(
        roc_auc_score(targets, probabilities, average="weighted")
    ) == pytest.approx(expected_roc)
    assert float(
        pr_auc_score(targets, probabilities, average="weighted")
    ) == pytest.approx(expected_pr)
    assert roc_auc_score(targets, probabilities, average=None).shape == (3,)
    assert pr_auc_score(targets, probabilities, average=None).shape == (3,)


@pytest.mark.parametrize("decoder", ["contiguous", "parity"])
def test_probability_classifier_shapes_normalization_and_gradients(decoder):
    torch.manual_seed(7)
    model = PQCClassifier(
        circuit_builder=_builder(3),
        n_qubits=3,
        n_classes=3,
        readout="probabilities",
        decoder=decoder,
        measurement_qubits=[0, 1, 2],
        init="random_-1_1",
    )
    inputs = torch.linspace(-0.6, 0.6, 12).reshape(4, 3)
    targets = torch.tensor([0, 1, 2, 0])
    probabilities = model(inputs)
    assert probabilities.shape == (4, 3)
    torch.testing.assert_close(
        probabilities.sum(dim=-1), torch.ones(4), atol=1e-6, rtol=0
    )
    loss = model.compute_loss(inputs, targets, "categorical_cross_entropy")
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert torch.isfinite(next(model.parameters()).grad).all()


def test_majority_probability_classifier_exposes_resolved_blocks():
    model = PQCClassifier(
        circuit_builder=_builder(6),
        n_qubits=6,
        n_classes=3,
        readout="probabilities",
        decoder="majority",
        majority_blocks=[[0, 1, 2], [3, 4, 5]],
        init="zeros",
    )
    assert model.majority_blocks == ((0, 1, 2), (3, 4, 5))
    assert model.unused_majority_qubits == tuple()
    probabilities = model(torch.zeros(2, 6))
    assert probabilities.shape == (2, 3)


def test_binary_and_multiclass_expectation_readouts():
    binary = PQCClassifier(
        circuit_builder=_builder(2),
        n_qubits=2,
        n_classes=2,
        readout="expectation",
        init="random_-1_1",
    )
    binary_inputs = torch.tensor([[0.1, -0.2], [0.3, 0.4]])
    binary_scores = binary(binary_inputs)
    assert binary_scores.shape == (2, 1)
    binary_probabilities = binary.predict_proba(binary_inputs)
    assert binary_probabilities.shape == (2, 2)
    binary.compute_loss(binary_inputs, torch.tensor([0, 1]), "hinge").backward()

    multiclass = PQCClassifier(
        circuit_builder=_builder(2),
        n_qubits=2,
        n_classes=3,
        readout="expectation",
        init="random_-1_1",
    )
    class_scores = multiclass(binary_inputs)
    assert class_scores.shape == (2, 3)
    probabilities = multiclass.predict_proba(binary_inputs)
    assert probabilities.shape == (2, 3)
    torch.testing.assert_close(
        probabilities.sum(dim=-1), torch.ones(2), atol=1e-6, rtol=0
    )


def test_multiclass_expectation_averages_one_observable_group_per_class():
    torch.manual_seed(12)
    model = PQCClassifier(
        circuit_builder=_builder(2),
        n_qubits=2,
        n_classes=3,
        readout="expectation",
        observables=[
            ["Z 0", "X 0"],
            ["Z 1"],
            ["ZZ 0 1", "X 1", "Y 1"],
        ],
        temperature=0.7,
        init="random_-1_1",
    )
    inputs = torch.tensor([[0.1, -0.2], [0.3, 0.4]])
    raw_expectations = model.expectation_values(inputs)
    scores = model(inputs)

    assert raw_expectations.shape == (2, 6)
    assert model.observable_groups == (
        ("Z 0", "X 0"),
        ("Z 1",),
        ("ZZ 0 1", "X 1", "Y 1"),
    )
    expected_scores = torch.stack(
        (
            raw_expectations[:, 0:2].mean(dim=-1),
            raw_expectations[:, 2:3].mean(dim=-1),
            raw_expectations[:, 3:6].mean(dim=-1),
        ),
        dim=-1,
    )
    torch.testing.assert_close(scores, expected_scores)
    torch.testing.assert_close(
        model.predict_proba(inputs), torch.softmax(expected_scores / 0.7, dim=-1)
    )
    torch.testing.assert_close(
        model.predict(inputs), expected_scores.argmax(dim=-1)
    )

    loss = model.compute_loss(
        inputs, torch.tensor([0, 2]), "categorical_cross_entropy"
    )
    loss.backward()
    assert torch.isfinite(next(model.parameters()).grad).all()


def test_multiclass_expectation_observable_groups_are_validated():
    with pytest.raises(ValueError, match="3 observable groups"):
        PQCClassifier(
            circuit_builder=_builder(2),
            n_qubits=2,
            n_classes=3,
            readout="expectation",
            observables=[["Z 0"], ["Z 1"]],
        )

    with pytest.raises(ValueError, match="group 1 must be nonempty"):
        PQCClassifier(
            circuit_builder=_builder(2),
            n_qubits=2,
            n_classes=3,
            readout="expectation",
            observables=[["Z 0"], [], ["Z 1"]],
        )

    with pytest.raises(ValueError, match="Do not mix"):
        PQCClassifier(
            circuit_builder=_builder(2),
            n_qubits=2,
            n_classes=3,
            readout="expectation",
            observables=["Z 0", ["Z 1"], ["ZZ 0 1"]],
        )


def test_differentiable_pauli_propagation_matches_exact_scores_and_gradients():
    torch.manual_seed(19)
    common = dict(
        circuit_builder=_builder(2),
        n_qubits=2,
        n_classes=3,
        readout="expectation",
        observables=[
            ["Z 0", "X 0"],
            ["Z 1"],
            ["ZZ 0 1", "X 1"],
        ],
        init="small_normal",
    )
    exact = PQCClassifier(**common, device="default.qubit")
    propagated = PQCClassifier(
        **common,
        device="lightning.pp",
        diff_method="backprop",
        pp_truncation=None,
        pp_topM=None,
        pp_tau=0.0,
        pp_prune=False,
    )
    propagated.qlayer.load_state_dict(exact.qlayer.state_dict())

    inputs = torch.tensor([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.2]])
    targets = torch.tensor([0, 1, 2])
    torch.testing.assert_close(
        propagated.expectation_values(inputs),
        exact.expectation_values(inputs),
        atol=2e-6,
        rtol=2e-6,
    )
    torch.testing.assert_close(
        propagated.predict_proba(inputs),
        exact.predict_proba(inputs),
        atol=2e-6,
        rtol=2e-6,
    )

    exact_loss = exact.compute_loss(inputs, targets, "categorical_cross_entropy")
    propagated_loss = propagated.compute_loss(
        inputs, targets, "categorical_cross_entropy"
    )
    exact_loss.backward()
    propagated_loss.backward()
    torch.testing.assert_close(propagated_loss, exact_loss, atol=2e-6, rtol=2e-6)
    for exact_parameter, propagated_parameter in zip(
        exact.qlayer.parameters(), propagated.qlayer.parameters()
    ):
        torch.testing.assert_close(
            propagated_parameter.grad,
            exact_parameter.grad,
            atol=2e-6,
            rtol=2e-6,
        )


def test_classifier_trainer_optimizes_pauli_propagation_parameters():
    config = {
        "feature_map": "AngleEmbedding",
        "feature_map_config": {
            "rotations": "Y",
            "input_scaling": True,
        },
        "ansatz": "AngleAnsatz",
        "ansatz_config": {
            "n_layers": 1,
            "rotations": "Y",
            "include_imprimitive": True,
            "imprimitive_pattern": "chain",
        },
        "readout": "expectation",
        "observables": ["Z 0", "ZZ 0 1", "X 1"],
        "device": "lightning.pp",
        "diff_method": "backprop",
        "init": "small_normal",
        "pp_truncation": 2,
        "pp_topM": 8,
        "pp_tau": 1e-10,
        "pp_prune_every": 1,
        "pp_prune": True,
    }
    model = create_classifier(config, n_qubits=2, n_classes=2)
    before = {
        name: parameter.detach().clone()
        for name, parameter in model.qlayer.named_parameters()
    }
    trainer = ClassifierTrainer(
        model, objective="mean_squared_error", learning_rate=0.03
    )
    history = trainer.fit(
        torch.tensor(
            [[-0.8, -0.4], [-0.4, -0.2], [0.3, 0.5], [0.7, 0.9]]
        ),
        torch.tensor([0, 0, 1, 1]),
        epochs=2,
        batch_size=2,
        shuffle=False,
    )

    assert len(history["train_loss"]) == 2
    assert all(torch.isfinite(torch.tensor(history["train_loss"])))
    for name, parameter in model.qlayer.named_parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert not torch.equal(parameter.detach(), before[name])


def test_pauli_propagation_classifier_rejects_incompatible_execution_modes():
    with pytest.raises(ValueError, match="supports only expectation readout"):
        PQCClassifier(
            circuit_builder=_builder(2),
            n_qubits=2,
            n_classes=2,
            readout="probabilities",
            device="lightning.pp",
        )

    with pytest.raises(ValueError, match="requires shots=None"):
        PQCClassifier(
            circuit_builder=_builder(2),
            n_qubits=2,
            n_classes=2,
            readout="expectation",
            device="lightning.pp",
            shots=100,
        )


def test_binary_expectation_averages_user_observables_for_mse_and_sign_prediction():
    torch.manual_seed(14)
    model = PQCClassifier(
        circuit_builder=_builder(2),
        n_qubits=2,
        n_classes=2,
        readout="expectation",
        observable=["Z 0", "ZZ 0 1", "X 1"],
        init="random_-1_1",
    )
    inputs = torch.tensor([[0.1, -0.2], [0.4, 0.3], [-0.5, 0.2]])
    targets = torch.tensor([0, 1, 0])
    expectation_values = model.expectation_values(inputs)
    scores = model(inputs)
    assert expectation_values.shape == (3, 3)
    torch.testing.assert_close(
        scores, expectation_values.mean(dim=-1, keepdim=True)
    )
    torch.testing.assert_close(
        model.predict(inputs), (scores.squeeze(-1) > 0).long()
    )

    loss = model.compute_loss(inputs, targets, "mean_squared_error")
    loss.backward()
    assert torch.isfinite(next(model.parameters()).grad).all()

    with pytest.raises(ValueError, match="either observable or observables"):
        PQCClassifier(
            circuit_builder=_builder(2),
            n_qubits=2,
            n_classes=2,
            readout="expectation",
            observable=["Z 0"],
            observables=["Z 1"],
        )

    with pytest.raises(ValueError, match="decoder applies only to probability"):
        PQCClassifier(
            circuit_builder=_builder(2),
            n_qubits=2,
            n_classes=2,
            readout="expectation",
            decoder="contiguous",
            observable=["Z 0"],
        )


def test_draw_classifier_supports_text_matplotlib_and_paper_tikz():
    import matplotlib.pyplot as plt

    model = PQCClassifier(
        circuit_builder=_builder(2),
        n_qubits=2,
        n_classes=2,
        readout="expectation",
        observable=["Z 0", "ZZ 0 1", "X 1"],
        init="zeros",
    )
    inputs = torch.tensor([0.1, -0.2])
    text_diagram = draw_classifier(model, inputs=inputs, output="text")
    assert "RY" in text_diagram

    tikz = model.draw(
        inputs=inputs,
        output="tikz",
        tikz_options={"standalone": True, "symbolic_parameters": True},
    )
    assert "\\documentclass" in tikz
    assert "\\begin{quantikz}" in tikz

    figure, _ = draw_classifier(model, inputs=inputs, output="matplotlib")
    assert figure.axes
    plt.close(figure)


def test_factory_and_trainer_use_qasp_configuration_notation():
    config = {
        "feature_map": "AngleEmbedding",
        "feature_map_config": {
            "rotations": "Y",
            "input_scaling": False,
        },
        "ansatz": "AngleAnsatz",
        "ansatz_config": {
            "n_layers": 1,
            "rotations": "Y",
            "include_imprimitive": True,
            "imprimitive_pattern": "chain",
        },
        "readout": "probabilities",
        "decoder": "contiguous",
        "init": "small_normal",
    }
    model = create_classifier(config, n_qubits=2, n_classes=2)
    text_diagram = draw_classifier(
        config=config,
        n_qubits=2,
        n_classes=2,
        inputs=torch.tensor([0.1, -0.2]),
        output="text",
    )
    assert "AngleEmbedding" in text_diagram
    trainer = ClassifierTrainer(model, learning_rate=0.02)
    inputs = torch.tensor(
        [[-0.8, -0.6], [-0.5, -0.7], [0.6, 0.5], [0.8, 0.7]]
    )
    targets = torch.tensor([0, 0, 1, 1])
    history = trainer.fit(inputs, targets, epochs=2, batch_size=2)
    assert len(history["train_loss"]) == 2
    evaluation = trainer.evaluate(inputs, targets)
    assert evaluation["confusion_matrix"].shape == (2, 2)
    assert 0.0 <= evaluation["accuracy"] <= 1.0
    assert 0.0 <= evaluation["roc_auc"] <= 1.0
    assert 0.0 <= evaluation["pr_auc"] <= 1.0
