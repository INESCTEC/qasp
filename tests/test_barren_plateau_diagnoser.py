import numpy as np
import pytest
import torch

import qasp
from qasp import barren_plateau_diagnoser, create_classifier, create_kernel


def _ansatz_config():
    return {
        "n_layers": 1,
        "rotations": "Y",
        "include_imprimitive": False,
        "reuploading": False,
    }


def _feature_map_config():
    return {
        "features": 2,
        "rotations": "Y",
        "imprimitive_pattern": "chain",
    }


def _classifier():
    return create_classifier(
        {
            "ansatz": "AngleAnsatz",
            "ansatz_config": _ansatz_config(),
            "feature_map": "AngleEmbedding",
            "feature_map_config": _feature_map_config(),
            "readout": "expectation",
            "observables": ["Z 0"],
            "init": "zeros",
            "device": "default.qubit",
            "diff_method": "backprop",
        },
        n_qubits=2,
        n_classes=2,
    )


def _classifier_for_width(n_qubits):
    return create_classifier(
        {
            "ansatz": "AngleAnsatz",
            "ansatz_config": _ansatz_config(),
            "feature_map": "AngleEmbedding",
            "feature_map_config": {
                "features": n_qubits,
                "rotations": "Y",
                "imprimitive_pattern": "chain",
            },
            "readout": "expectation",
            "observables": ["Z 0"],
            "init": "zeros",
            "device": "default.qubit",
            "diff_method": "backprop",
        },
        n_qubits=n_qubits,
        n_classes=2,
    )


def test_top_level_diagnoser_is_public_and_lazy_exported():
    assert callable(qasp.barren_plateau_diagnoser)
    assert "barren_plateau_diagnoser" in qasp.__all__


def test_classifier_diagnosis_is_seeded_unbiased_and_restores_model():
    classifier = _classifier()
    classifier.train()
    before = [parameter.detach().clone() for parameter in classifier.parameters()]

    result = barren_plateau_diagnoser(
        classifier,
        n_samples=8,
        initialization="random_-pi_pi",
        state=torch.tensor([0.2, -0.4]),
        output_index=1,
        seed=17,
        return_samples=True,
    )

    assert result["model_type"] == "classifier"
    assert result["cost"] == "class_probability[1]"
    assert result["state_mode"] == "fixed"
    assert result["variance"] == pytest.approx(
        np.var(result["samples"], ddof=1)
    )
    assert result["cost_variance"] == pytest.approx(result["variance"])
    assert result["variance_by_sample"][-1] == pytest.approx(result["variance"])
    np.testing.assert_allclose(
        result["expectation_variance"],
        np.var(result["expectation_samples"], axis=0, ddof=1),
    )
    np.testing.assert_allclose(
        result["expectation_variance_by_sample"][-1],
        result["expectation_variance"],
    )
    assert result["expectation_value_variance"] == pytest.approx(
        np.var(result["expectation_value_samples"], ddof=1)
    )
    assert result["expectation_value_variance_by_sample"][-1] == pytest.approx(
        result["expectation_value_variance"]
    )
    np.testing.assert_array_equal(result["sample_counts"], np.arange(1, 9))
    assert result["n_parameters"] > 0
    assert classifier.training
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(before, classifier.parameters())
    )

    repeated = barren_plateau_diagnoser(
        classifier,
        n_samples=8,
        state=torch.tensor([0.2, -0.4]),
        output_index=1,
        seed=17,
        return_samples=True,
    )
    np.testing.assert_array_equal(result["samples"], repeated["samples"])


def test_qubit_sweep_returns_one_independent_variance_estimate_per_width():
    widths = [2, 3, 4]
    result = barren_plateau_diagnoser(
        model_factory=_classifier_for_width,
        qubit_list=widths,
        n_samples=7,
        state_factory=torch.zeros,
        output_index=1,
        seed=31,
        return_samples=True,
    )

    assert result["mode"] == "qubit_sweep"
    np.testing.assert_array_equal(result["qubit_list"], widths)
    assert result["n_samples_per_qubit"] == 7
    assert result["state_mode"] == "fixed"
    assert result["cost_variance"].shape == (len(widths),)
    assert result["expectation_value_variance"].shape == (len(widths),)
    np.testing.assert_array_equal(result["variance"], result["cost_variance"])

    for index, n_qubits in enumerate(widths):
        child = result["results_by_qubit"][n_qubits]
        assert child["n_qubits"] == n_qubits
        assert child["seed"] == 31 + index
        assert child["n_samples"] == 7
        assert result["cost_variance"][index] == pytest.approx(
            np.var(result["cost_samples_by_qubit"][n_qubits], ddof=1)
        )
        assert result["expectation_value_variance"][index] == pytest.approx(
            np.var(
                result["expectation_value_samples_by_qubit"][n_qubits], ddof=1
            )
        )
        np.testing.assert_allclose(
            result["expectation_variance_by_qubit"][n_qubits],
            np.var(
                result["expectation_samples_by_qubit"][n_qubits],
                axis=0,
                ddof=1,
            ),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"qubit_list": []}, "at least one"),
        ({"qubit_list": [2, 2]}, "duplicate"),
        ({"qubit_list": [0, 2]}, "positive integer"),
        ({"qubit_list": [2]}, "model_factory"),
    ],
)
def test_qubit_sweep_validates_its_width_contract(kwargs, message):
    with pytest.raises(ValueError, match=message):
        barren_plateau_diagnoser(n_samples=3, **kwargs)


def test_classifier_can_diagnose_its_configured_loss_with_random_states():
    result = barren_plateau_diagnoser(
        _classifier(),
        n_samples=5,
        state=None,
        state_initialization="random_-1_1",
        targets=1,
        objective="mean_squared_error",
        seed=4,
        return_samples=True,
    )

    assert result["state_mode"] == "random"
    assert result["cost"] == "mean_squared_error"
    assert np.isfinite(result["samples"]).all()


def test_probability_classifier_reports_z_expectation_variance():
    classifier = create_classifier(
        {
            "ansatz": "AngleAnsatz",
            "ansatz_config": _ansatz_config(),
            "feature_map": "AngleEmbedding",
            "feature_map_config": _feature_map_config(),
            "readout": "probabilities",
            "decoder": "contiguous",
            "init": "zeros",
            "device": "default.qubit",
            "diff_method": "backprop",
        },
        n_qubits=2,
        n_classes=2,
    )

    result = barren_plateau_diagnoser(
        classifier,
        n_samples=5,
        state=torch.tensor([0.2, -0.4]),
        seed=6,
        return_samples=True,
    )

    n_measured = len(classifier.measurement_qubits)
    assert result["expectation_samples"].shape == (5, n_measured)
    assert result["expectation_variance"].shape == (n_measured,)
    assert np.abs(result["expectation_samples"]).max() <= 1.0 + 1e-6


def test_trainable_kernel_uses_pair_similarity_cost():
    kernel = create_kernel(
        {
            "kernel_type": "fidelity",
            "feature_map": "AngleEmbedding",
            "feature_map_config": _feature_map_config(),
            "ansatz": "IQPAnsatz",
            "ansatz_config": {
                "n_layers": 2,
                "imprimitive_pattern": "ring",
                "reuploading": True,
            },
            "init": "zeros",
            "device": "default.qubit",
        },
        n_qubits=2,
    )
    state = (np.array([0.1, 0.2]), np.array([0.4, 0.7]))
    before = [parameter.detach().clone() for parameter in kernel.parameters()]

    result = barren_plateau_diagnoser(
        kernel, n_samples=5, state=state, seed=2, return_samples=True
    )

    assert result["model_type"] == "kernel"
    assert result["cost"] == "kernel_similarity"
    assert result["variance"] > 0
    assert result["expectation_variance"].shape == (1,)
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(before, kernel.parameters())
    )


def test_policy_action_probability_and_custom_scalar_cost():
    from qasp import create_policy

    policy = create_policy(
        "PQCBorn",
        {
            "ansatz": "AngleAnsatz",
            "ansatz_config": _ansatz_config(),
            "feature_map": "AngleEmbedding",
            "feature_map_config": _feature_map_config(),
            "n_actions": 2,
            "init": "zeros",
            "device": "default.qubit",
            "diff_method": "backprop",
            "shots": None,
            "softmax_activation": False,
            "output_scaling": None,
            "torchrl": False,
            "measurement_qubits": [0, 1],
            "parity_cnot_decomposition": False,
            "policy": "contiguous-like",
            "training_method": "probs",
        },
        n_qubits=2,
    )
    action_result = barren_plateau_diagnoser(
        policy,
        n_samples=5,
        state=torch.tensor([0.1, 0.2]),
        output_index=1,
        seed=8,
        return_samples=True,
    )
    custom_result = barren_plateau_diagnoser(
        policy,
        n_samples=5,
        state_sampler=lambda index, generator: torch.rand(
            2, generator=generator
        ),
        cost_function=lambda model, state: -torch.log(
            model.action_probs(state)[1].clamp_min(1e-8)
        ),
        seed=8,
    )

    assert action_result["model_type"] == "policy"
    assert action_result["cost"] == "action_probability[1]"
    assert action_result["expectation_variance"].shape == (2,)
    assert action_result["expectation_value_variance"] == pytest.approx(
        np.var(action_result["expectation_samples"].mean(axis=1), ddof=1)
    )
    assert np.logical_and(
        action_result["samples"] >= 0, action_result["samples"] <= 1
    ).all()
    assert custom_result["cost"] == "<lambda>"
    assert custom_result["state_mode"] == "sampler"


def test_non_trainable_kernel_and_nonscalar_cost_are_rejected():
    fixed_kernel = create_kernel(
        {
            "kernel_type": "fidelity",
            "feature_map": "AngleEmbedding",
            "feature_map_config": _feature_map_config(),
            "device": "default.qubit",
        },
        n_qubits=2,
    )
    with pytest.raises(ValueError, match="No trainable parameters"):
        barren_plateau_diagnoser(fixed_kernel, n_samples=3)

    classifier = _classifier()
    with pytest.raises(ValueError, match="exactly one scalar"):
        barren_plateau_diagnoser(
            classifier,
            n_samples=3,
            state=torch.zeros(2),
            cost_function=lambda model, state: model.predict_proba(state),
        )


@pytest.mark.parametrize("return_node", [False, True])
def test_building_block_sweep_matches_explicit_qnode_calls(return_node):
    from qasp import AngleAnsatz, AngleEmbedding, CircuitBuilder, Ones, Uniform

    nodes = {}
    def factory(n):
        builder = CircuitBuilder(n)
        builder.set_ansatz(AngleAnsatz(n, rotations="Y", feature_map=AngleEmbedding(
            n, features=n, rotations="Y", input_scaling=True)))
        builder.set_measurement("probs", wires=[0, n - 1])
        builder.build()
        nodes[n] = builder.compile(device="default.qubit")
        return nodes[n] if return_node else builder

    widths = [2, 4, 3]
    seed = 17
    init = Uniform(-1, 1)
    def fixed_input(n):
        return torch.linspace(-0.2, 0.5, n, dtype=torch.float64)

    sweep = barren_plateau_diagnoser(
        model_factory=factory, qubit_list=widths, state_factory=fixed_input,
        n_samples=6, initialization=init, input_scaling_init=Ones(),
        seed=seed, output_index=1, return_samples=True,
    )
    for i, n in enumerate(widths):
        x = fixed_input(n)
        generator = torch.Generator().manual_seed(seed + i)
        costs, expectations = [], []
        for _ in range(6):
            angles = torch.empty(n, dtype=x.dtype).uniform_(-1, 1, generator=generator)
            probabilities = nodes[n](x, params=angles, input_scaling_params=torch.ones_like(x))
            costs.append(probabilities[1].item())
            p = probabilities.detach().numpy().reshape(2, 2)
            expectations.append([p[0].sum() - p[1].sum(), p[:, 0].sum() - p[:, 1].sum()])
        np.testing.assert_allclose(sweep["cost_samples_by_qubit"][n], costs)
        np.testing.assert_allclose(sweep["expectation_samples_by_qubit"][n], expectations)
        assert sweep["cost_variance"][i] == pytest.approx(np.var(costs, ddof=1))
        assert sweep["results_by_qubit"][n]["n_parameters"] == 2 * n


def test_unencoded_compiled_expectations_need_no_state_or_layer():
    builder = qasp.CircuitBuilder(2)
    builder.set_ansatz(qasp.AngleAnsatz(2, rotations="Y"))
    builder.set_measurement("expectation", observables=["Z 0", "ZZ 0 1"])
    builder.build()
    node = builder.compile(device="default.qubit")
    r = barren_plateau_diagnoser(node, n_samples=4, initialization=qasp.Constant(0.3),
                                output_index=1, return_samples=True, seed=2)
    assert r["state_mode"] == "none"
    np.testing.assert_allclose(r["cost_samples"], np.cos(0.3)**2)
    np.testing.assert_allclose(r["expectation_mean"], [np.cos(0.3), np.cos(0.3)**2])


def test_builder_diagnostics_reject_uncompiled_and_state_readout():
    builder = qasp.CircuitBuilder(2)
    builder.set_ansatz(qasp.AngleAnsatz(2))
    builder.set_measurement("state")
    builder.build()
    with pytest.raises(ValueError, match="compile"):
        barren_plateau_diagnoser(builder, n_samples=2)
    node = builder.compile(device="default.qubit")
    with pytest.raises(ValueError, match="readout"):
        barren_plateau_diagnoser(node, n_samples=2)


def test_circuit_sweep_infers_features_and_validates_factory_width():
    builder = qasp.CircuitBuilder(3)
    builder.set_ansatz(qasp.AngleAnsatz(
        3, rotations="X", feature_map=qasp.AngleEmbedding(3, features=2, rotations="X")))
    builder.set_measurement("probs", wires=[1])
    builder.build()
    node = builder.compile(device="auto")
    def cost(bound_node, x):
        assert x.shape == (2,)
        return bound_node(x)[1]

    result = barren_plateau_diagnoser(model_factory=lambda n: node, qubit_list=[3],
                                     n_samples=3, seed=1, cost_function=cost,
                                     return_samples=True)
    p = result["cost_samples_by_qubit"][3]
    np.testing.assert_allclose(result["expectation_samples_by_qubit"][3][:, 0], 1 - 2*p)
    with pytest.raises(ValueError, match="3-qubit circuit for width 5"):
        barren_plateau_diagnoser(model_factory=lambda n: node, qubit_list=[5], n_samples=2)
