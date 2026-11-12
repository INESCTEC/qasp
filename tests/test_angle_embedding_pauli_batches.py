"""Custom Pauli encodings must select features, not rows of a training batch."""
import pennylane as qml
import pytest
import torch

from qasp import AngleEmbedding


@pytest.mark.parametrize("batch_size", [2, 5])
@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize("reupload", [False, True])
@pytest.mark.parametrize("diff_method", ["backprop", "adjoint"])
def test_custom_pauli_batch_values_and_gradients(batch_size, scaled, reupload, diff_method):
    """Compare independent gates and gradients, including equal batch/feature sizes."""
    embedding = AngleEmbedding(2, features=2, rotations=["Y 0", "YY 0 1"],
                               input_scaling=scaled)

    def make_node(reference):
        @qml.qnode(qml.device("default.qubit", wires=2), interface="torch",
                   diff_method=diff_method)
        def node(inputs, scales, angles):
            for layer in range(2 if reupload else 1):
                if reference:
                    qml.RY(inputs[..., 0] * (scales[layer, 0] if scaled else 1), wires=0)
                    qml.IsingYY(inputs[..., 1] * (scales[layer, 1] if scaled else 1), wires=[0, 1])
                else:
                    embedding(inputs, scales[layer], layer_index=layer + 1)
                qml.RZ(angles[layer, 0], wires=0)
                qml.RY(angles[layer, 1], wires=1)
            return qml.expval(qml.Z(0) @ qml.Z(1))
        return node

    inputs = torch.linspace(-0.71, 0.83, batch_size * 2, dtype=torch.float64).reshape(batch_size, 2).requires_grad_()
    scales = torch.tensor([[0.8, 1.1], [1.2, 0.6]], dtype=torch.float64, requires_grad=True)
    angles = torch.tensor([[0.3, -0.4], [0.7, 0.2]], dtype=torch.float64, requires_grad=True)
    node, reference = make_node(False), make_node(True)
    actual = node(inputs, scales, angles)
    expected = reference(inputs, scales, angles)
    individual = torch.stack([node(row, scales, angles) for row in inputs])
    assert actual.shape == (batch_size,)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(actual, individual, atol=1e-10, rtol=1e-10)
    weights = torch.linspace(0.3, 1.2, batch_size, dtype=torch.float64)
    parameters = [inputs, angles] + ([scales] if scaled else [])
    gradients = torch.autograd.grad((actual * weights).sum(), parameters)
    reference_gradients = torch.autograd.grad((expected * weights).sum(), parameters)
    individual_gradients = torch.autograd.grad((individual * weights).sum(), parameters)
    for got, ref, single in zip(gradients, reference_gradients, individual_gradients):
        assert torch.isfinite(got).all()
        torch.testing.assert_close(got, ref, atol=1e-9, rtol=1e-9)
        torch.testing.assert_close(got, single, atol=1e-9, rtol=1e-9)


def test_custom_pauli_encoding_still_accepts_feature_list():
    embedding = AngleEmbedding(2, features=2, rotations=["Y 0", "YY 0 1"])
    tape = qml.tape.make_qscript(embedding)([0.2, -0.4])
    assert [op.data[0] for op in tape.operations] == [0.2, -0.4]
    assert [op.hyperparameters["pauli_word"] for op in tape.operations] == ["Y", "YY"]


@pytest.mark.parametrize("scaled", [False, True])
@pytest.mark.parametrize("reupload", [False, True])
def test_custom_pauli_drawing_labels_features_and_layer_scales(scaled, reupload):
    from qasp import AngleAnsatz, CircuitBuilder
    from qasp.circuits import circuit_to_tikz

    embedding = AngleEmbedding(3, features=2, rotations=["Y 0", "YY 0 1"],
                               input_scaling=scaled)
    ansatz = AngleAnsatz(3, n_layers=2, feature_map=embedding, rotations="ZY",
                        reuploading=reupload, include_imprimitive=False)
    builder = CircuitBuilder(3)
    builder.set_ansatz(ansatz)
    builder.set_measurement("expectation", observables=["ZZZ 0 1 2"])
    builder.build()
    arguments = {"inputs": torch.tensor([0.2, -0.4]), "params": torch.zeros(12)}
    if scaled:
        arguments["input_scaling_params"] = torch.ones(4)
    tex = circuit_to_tikz(builder, circuit_kwargs=arguments, symbolic_parameters=True)
    assert tex.count(r"f_{1}") == (2 if reupload else 1)
    assert tex.count(r"f_{2}") == (2 if reupload else 1)
    assert r"\theta_{12}" in tex
    assert r"\theta_{13}" not in tex
    if scaled:
        for layer in range(1, 3 if reupload else 2):
            for feature in (1, 2):
                index = f"{layer},{feature}" if reupload else str(feature)
                assert rf"f_{{{feature}}}\lambda_{{{index}}}" in tex
    else:
        assert r"\lambda" not in tex
