import numpy as np
import pytest
import torch

from qasp.qkernels import (
    FidelityKernel,
    KernelTargetAlignment,
    ProjectedKernel,
    create_kernel,
    draw_kernel,
    parse_fidelity_projectors,
    target_alignment,
)


X = np.array(
    [[0.10, 0.20], [0.20, 0.10], [0.90, 1.10], [1.00, 0.80]]
)
Y = np.array([[0.15, 0.25], [0.95, 0.85]])
LABELS = np.array([0, 0, 1, 1])


def feature_kernel_config(kernel_type):
    return {
        "kernel_type": kernel_type,
        "feature_map": "AngleEmbedding",
        "feature_map_config": {
            "rotations": "Y",
            "imprimitive_pattern": "chain",
        },
        "reps": 2,
        "device": "default.qubit",
    }


def iqp_kernel_config(kernel_type, init="zeros"):
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


@pytest.mark.parametrize(
    ("kernel_type", "kernel_class"),
    [
        ("fidelity", FidelityKernel),
        ("projected", ProjectedKernel),
    ],
)
def test_factory_builds_feature_map_kernels(kernel_type, kernel_class):
    kernel = create_kernel(feature_kernel_config(kernel_type), n_qubits=2)
    gram = kernel.gram(X)

    assert isinstance(kernel, kernel_class)
    assert gram.shape == (4, 4)
    assert np.allclose(gram, gram.T, atol=1e-7)
    assert np.allclose(np.diag(gram), 1.0, atol=1e-7)


@pytest.mark.parametrize("kernel_type", ["local", "local_fidelity"])
def test_factory_rejects_removed_local_fidelity_types(kernel_type):
    with pytest.raises(ValueError, match="Unknown kernel_type"):
        create_kernel(feature_kernel_config(kernel_type), n_qubits=2)


def test_fidelity_cross_gram_has_expected_shape_and_values():
    kernel = create_kernel(feature_kernel_config("fidelity"), n_qubits=2)
    cross_gram = kernel.gram(Y, X)

    assert cross_gram.shape == (2, 4)
    assert np.allclose(cross_gram, kernel.gram(X, Y).T, atol=1e-7)


def test_iqp_fidelity_kernel_is_psd_and_differentiable():
    torch.manual_seed(7)
    kernel = create_kernel(
        iqp_kernel_config("fidelity", init="small_normal"), n_qubits=2
    )

    gram = kernel.gram(X, return_tensor=True)
    gram.sum().backward()

    eigenvalues = np.linalg.eigvalsh(gram.detach().numpy())
    assert gram.requires_grad
    assert eigenvalues.min() >= -1e-6
    assert all(parameter.grad is not None for parameter in kernel.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in kernel.parameters())


def test_iqp_default_fidelity_is_the_global_all_qubit_projector():
    default = create_kernel(iqp_kernel_config("fidelity"), n_qubits=2)
    explicit_config = iqp_kernel_config("fidelity")
    explicit_config["projectors"] = ["P 0 1"]
    explicit = create_kernel(explicit_config, n_qubits=2)

    assert default.projector_labels == ["P 0 1"]
    assert default.projector_wires == [[0, 1]]
    assert np.allclose(default.gram(X), explicit.gram(X), atol=1e-6)


def test_two_qubit_local_fidelity_is_average_of_single_qubit_projectors():
    combined_config = iqp_kernel_config("fidelity")
    combined_config["projectors"] = ["P 0", "P 1"]
    combined = create_kernel(combined_config, n_qubits=2)

    first_config = iqp_kernel_config("fidelity")
    first_config["projectors"] = ["P 0"]
    first = create_kernel(first_config, n_qubits=2)
    second_config = iqp_kernel_config("fidelity")
    second_config["projectors"] = ["P 1"]
    second = create_kernel(second_config, n_qubits=2)

    expected = 0.5 * (first.gram(X) + second.gram(X))
    assert np.allclose(combined.gram(X), expected, atol=1e-7)


def test_user_defined_local_projector_subsets_are_parsed_and_evaluated():
    X_three_qubits = np.column_stack([X, 0.5 * (X[:, 0] + X[:, 1])])
    labels = ["P 0 1", "P 0 2", " P 0 1 2"]
    config = iqp_kernel_config("fidelity")
    config["projectors"] = labels
    kernel = create_kernel(config, n_qubits=3)

    gram = kernel.gram(X_three_qubits)

    assert kernel.projector_labels == ["P 0 1", "P 0 2", "P 0 1 2"]
    assert kernel.projector_wires == [[0, 1], [0, 2], [0, 1, 2]]
    assert len(parse_fidelity_projectors(labels, n_qubits=3)) == 3
    assert kernel.n_projectors == 3
    raw_projector_probabilities = kernel._overlap_out(
        X_three_qubits[0], X_three_qubits[1]
    )
    assert np.isclose(
        kernel.kernel(X_three_qubits[0], X_three_qubits[1]),
        raw_projector_probabilities.detach().numpy().mean(),
    )
    assert np.allclose(gram, gram.T, atol=1e-7)
    assert np.allclose(np.diag(gram), 1.0, atol=1e-7)


def test_iqp_projected_kernel_is_psd_and_differentiable():
    torch.manual_seed(11)
    config = iqp_kernel_config("projected", init="small_normal")
    config.update({"observables": ["X 0", "Y 0", "Z 0", "Z 1"], "gamma": 0.5})
    kernel = create_kernel(config, n_qubits=2)

    gram = kernel.gram(X, return_tensor=True)
    gram.sum().backward()

    eigenvalues = np.linalg.eigvalsh(gram.detach().numpy())
    assert eigenvalues.min() >= -1e-6
    assert all(parameter.grad is not None for parameter in kernel.parameters())


def test_iqp_kernel_target_alignment_updates_parameters():
    torch.manual_seed(3)
    kernel = create_kernel(
        iqp_kernel_config("fidelity", init="small_normal"), n_qubits=2
    )
    before = [parameter.detach().clone() for parameter in kernel.parameters()]

    history = KernelTargetAlignment(kernel).fit(
        X, LABELS, steps=2, lr=0.02, seed=3, verbose=False
    )

    assert len(history["alignment"]) == 2
    assert np.isfinite(history["alignment"]).all()
    assert any(
        not torch.allclose(old, new)
        for old, new in zip(before, kernel.parameters())
    )


@pytest.mark.parametrize(
    "projectors",
    [[], ["Z 0"], ["P"], ["P a"], ["P 0 0"], ["P 2"]],
)
def test_fidelity_rejects_invalid_projectors(projectors):
    config = feature_kernel_config("fidelity")
    config["projectors"] = projectors

    with pytest.raises(ValueError):
        create_kernel(config, n_qubits=2)

def test_input_scaling_and_data_shapes_are_validated():
    config = feature_kernel_config("fidelity")
    config["feature_map"] = "NeuralEmbedding"
    config["feature_map_config"] = {"rotations": "Z", "features": 2}

    with pytest.raises(ValueError, match="input_scaling_params"):
        create_kernel(config, n_qubits=2)

    kernel = create_kernel(feature_kernel_config("fidelity"), n_qubits=2)
    with pytest.raises(ValueError, match="shape"):
        kernel.gram(np.ones((3, 3)))


def test_target_alignment_validates_dimensions():
    with pytest.raises(ValueError, match="square"):
        target_alignment(np.ones((2, 3)), [0, 1])
    with pytest.raises(ValueError, match="labels"):
        target_alignment(np.eye(3), [0, 1])


@pytest.mark.parametrize(
    "kernel_type", ["fidelity", "projected"]
)
def test_draw_kernel_exports_standalone_tikz(kernel_type):
    kernel = create_kernel(feature_kernel_config(kernel_type), n_qubits=2)

    tikz = draw_kernel(
        kernel=kernel,
        x=X[0],
        y=X[1] if kernel_type != "projected" else None,
        output="tikz",
        tikz_options={"standalone": True},
    )

    assert "\\documentclass" in tikz
    assert "\\begin{quantikz}" in tikz
    assert "qaspSingleGate" in tikz


def test_draw_kernel_supports_compressed_trainable_layers():
    config = iqp_kernel_config("fidelity")
    config["ansatz_config"]["n_layers"] = 3
    config["ansatz_config"]["reuploading"] = False
    kernel = create_kernel(config, n_qubits=2)

    tikz = draw_kernel(
        kernel=kernel,
        x=X[0],
        y=X[1],
        output="tikz",
        tikz_options={"compress": True},
    )

    assert "dashed" in tikz
    assert "U(\\theta_{1})" in tikz
    assert "U(\\theta_{2})" in tikz
    assert "U(\\theta_{3})" in tikz
    assert "U^{\\dagger}(\\theta_{3})" in tikz
    assert "U^{\\dagger}(\\theta_{1})" in tikz


@pytest.mark.parametrize(("embedding", "rotations"), [
    ("AngleEmbedding", "YZ"), ("AngleEmbedding", ["Y 0", "ZZ 0 1"]),
    ("NeuralEmbedding", "Y"),
])
def test_fidelity_draw_shares_weights_and_distinguishes_inputs(embedding, rotations):
    import pennylane as qml

    cfg = {
        "feature_map": embedding,
        "feature_map_config": {"features": 2, "rotations": rotations},
        "ansatz": "AngleAnsatz",
        "ansatz_config": {"n_layers": 2, "rotations": "YZ", "reuploading": True},
        "device": "default.qubit",
    }
    if embedding == "AngleEmbedding":
        cfg["feature_map_config"]["input_scaling"] = True
    kernel = create_kernel(cfg, n_qubits=2)
    tex = draw_kernel(kernel, x=X[0], y=Y[1], output="tikz",
                      tikz_options={"symbolic_parameters": True})
    for index in range(1, 9):
        theta = rf"\theta_{{{index}}}"
        assert tex.count(theta) == 2
        assert tex.count(theta + r")^{\dagger}") == 1
    assert r"\theta_{9}" not in tex
    if embedding == "AngleEmbedding":
        for layer in (1, 2):
            for site in (1, 2):
                for data in ("x", "y"):
                    assert rf"{data}_{{{site}}}\lambda_{{{layer},{site}}}" in tex
        assert r"\lambda_{3," not in tex
    else:
        for data in ("x", "y"):
            assert rf"\langle\mathbf{{{data}}},\boldsymbol{{\lambda}}_{{2,1}}\rangle" in tex

    @qml.qnode(qml.device("default.qubit", wires=2), interface="torch")
    def state(features):
        kernel.ansatz(inputs=features, params=kernel.qlayer.params,
                      input_scaling_params=kernel.qlayer.input_scaling_params)
        return qml.state()

    x, y = kernel._x_tensor(X[0]), kernel._x_tensor(Y[1])
    reference = torch.vdot(state(y), state(x)).abs().square()
    actual = kernel._overlap_out(X[0], Y[1]).squeeze()
    torch.testing.assert_close(actual.double(), reference, atol=2e-6, rtol=2e-6)
    for observed, expected in zip(torch.autograd.grad(actual, kernel.parameters()),
                                  torch.autograd.grad(reference, kernel.parameters())):
        torch.testing.assert_close(observed, expected, atol=2e-6, rtol=2e-6)
