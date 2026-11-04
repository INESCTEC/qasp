from importlib.util import find_spec

import pennylane as qml
import pytest
import torch

from qasp.circuits.constructor import CircuitBuilder
from qasp.rl.policies import PQCBornVariance


class _FeatureMap:
    input_scaling = False


class _ToyAnsatz:
    """Small differentiable circuit that exercises all majority blocks."""

    feature_map = _FeatureMap()
    n_layers = 1

    def __init__(self, n_qubits):
        self.n_qubits = n_qubits

    def __call__(self, inputs=None, input_scaling_params=None, params=None):
        del input_scaling_params
        for wire in range(self.n_qubits):
            qml.RY(params[wire] + 0.1 * inputs[wire], wires=wire)
        for wire in range(self.n_qubits - 1):
            qml.CNOT(wires=[wire, wire + 1])

    def get_param_shapes(self):
        return ((self.n_qubits,),)


def _make_policy(training_method, device, diff_method):
    n_qubits = 6
    ansatz = _ToyAnsatz(n_qubits)
    builder = CircuitBuilder(n_qubits=n_qubits)
    builder.set_ansatz(ansatz)
    return PQCBornVariance(
        circuit_builder=builder,
        n_qubits=n_qubits,
        n_actions=3,
        policy="majority-modulo",
        majority_blocks=[[0, 1, 2], [3, 4, 5]],
        training_method=training_method,
        device=device,
        diff_method=diff_method,
        shots=None,
        init="random_-1_1",
    )


def _make_four_action_policy(training_method, device, diff_method):
    n_qubits = 6
    ansatz = _ToyAnsatz(n_qubits)
    builder = CircuitBuilder(n_qubits=n_qubits)
    builder.set_ansatz(ansatz)
    return PQCBornVariance(
        circuit_builder=builder,
        n_qubits=n_qubits,
        n_actions=4,
        policy="majority-modulo",
        majority_blocks=[[0, 1, 2], [3, 4, 5]],
        training_method=training_method,
        device=device,
        diff_method=diff_method,
        shots=None,
        init="random_-1_1",
    )


def _make_binary_policy(training_method, device, diff_method):
    n_qubits = 5
    ansatz = _ToyAnsatz(n_qubits)
    builder = CircuitBuilder(n_qubits=n_qubits, n_ancillas=1)
    builder.set_ansatz(ansatz)
    return PQCBornVariance(
        circuit_builder=builder,
        n_qubits=n_qubits,
        n_ancillas=1,
        n_actions=2,
        policy="majority-modulo",
        majority_blocks=[[0, 1, 2, 3, 4]],
        training_method=training_method,
        device=device,
        diff_method=diff_method,
        shots=None,
        init="random_-1_1",
    )


def _make_even_binary_policy(training_method, device, diff_method):
    n_qubits = 6
    ansatz = _ToyAnsatz(n_qubits)
    builder = CircuitBuilder(n_qubits=n_qubits)
    builder.set_ansatz(ansatz)
    return PQCBornVariance(
        circuit_builder=builder,
        n_qubits=n_qubits,
        n_actions=2,
        policy="majority-modulo",
        majority_blocks=[list(range(n_qubits))],
        training_method=training_method,
        device=device,
        diff_method=diff_method,
        shots=None,
        init="random_-1_1",
    )


def _make_parity_policy(n_actions, training_method, device, diff_method):
    n_qubits = 6
    ansatz = _ToyAnsatz(n_qubits)
    builder = CircuitBuilder(n_qubits=n_qubits)
    builder.set_ansatz(ansatz)
    return PQCBornVariance(
        circuit_builder=builder,
        n_qubits=n_qubits,
        n_actions=n_actions,
        policy="parity-like",
        measurement_qubits=list(range(n_qubits)),
        training_method=training_method,
        device=device,
        diff_method=diff_method,
        shots=None,
        init="random_-1_1",
    )


@pytest.mark.parametrize("lightning_device", ["lightning.qubit", "lightning.kokkos"])
def test_sparse_projectors_match_probs_and_gradients(lightning_device):
    pytest.importorskip("pennylane_lightning")
    if lightning_device == "lightning.kokkos" and find_spec("pennylane_lightning.lightning_kokkos") is None:
        pytest.skip("Optional lightning.kokkos plugin is not installed")
    torch.manual_seed(4)

    probability_policy = _make_policy("probs", "default.qubit", "backprop")
    sparse_policy = _make_policy("sparse_projectors", lightning_device, "adjoint")
    sparse_policy.qlayer.load_state_dict(probability_policy.qlayer.state_dict())

    inputs = torch.linspace(-0.4, 0.5, 6)
    expected = probability_policy(inputs)
    actual = sparse_policy(inputs)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual.sum(), torch.ones(()), atol=1e-6, rtol=0)

    expected_loss = -torch.log(expected[1]) + 0.2 * expected[2]
    actual_loss = -torch.log(actual[1]) + 0.2 * actual[2]
    expected_loss.backward()
    actual_loss.backward()

    expected_gradient = next(probability_policy.qlayer.parameters()).grad
    actual_gradient = next(sparse_policy.qlayer.parameters()).grad
    torch.testing.assert_close(
        actual_gradient, expected_gradient, atol=1e-6, rtol=1e-6
    )


def test_four_action_sparse_projectors_match_probs_and_gradients():
    torch.manual_seed(6)
    probability_policy = _make_four_action_policy(
        "probs", "default.qubit", "backprop"
    )
    sparse_policy = _make_four_action_policy(
        "sparse_projectors", "lightning.qubit", "adjoint"
    )
    sparse_policy.qlayer.load_state_dict(probability_policy.qlayer.state_dict())

    inputs = torch.linspace(-0.45, 0.35, 6)
    expected = probability_policy(inputs)
    actual = sparse_policy(inputs)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    expected_loss = -torch.log(expected[0]) + 0.2 * expected[-1]
    actual_loss = -torch.log(actual[0]) + 0.2 * actual[-1]
    expected_loss.backward()
    actual_loss.backward()
    for expected_parameter, actual_parameter in zip(
        probability_policy.qlayer.parameters(),
        sparse_policy.qlayer.parameters(),
    ):
        torch.testing.assert_close(
            actual_parameter.grad,
            expected_parameter.grad,
            atol=1e-6,
            rtol=1e-6,
        )


def test_sparse_projectors_validate_execution_configuration():
    with pytest.raises(ValueError, match="requires diff_method='adjoint'"):
        _make_policy("sparse_projectors", "lightning.qubit", "backprop")


def test_single_sparse_projector_includes_ancilla_identity():
    torch.manual_seed(9)
    probability_policy = _make_binary_policy("probs", "default.qubit", "backprop")
    sparse_policy = _make_binary_policy(
        "sparse_projectors", "lightning.qubit", "adjoint"
    )
    sparse_policy.qlayer.load_state_dict(probability_policy.qlayer.state_dict())

    inputs = torch.linspace(-0.2, 0.3, 5)
    expected = probability_policy(inputs)
    actual = sparse_policy(inputs)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    (-torch.log(actual[0])).backward()
    assert torch.isfinite(next(sparse_policy.qlayer.parameters()).grad).all()


def test_even_majority_block_uses_all_qubits_and_ties_go_to_zero():
    torch.manual_seed(12)
    probability_policy = _make_even_binary_policy(
        "probs", "default.qubit", "backprop"
    )
    sparse_policy = _make_even_binary_policy(
        "sparse_projectors", "lightning.qubit", "adjoint"
    )
    sparse_policy.qlayer.load_state_dict(probability_policy.qlayer.state_dict())

    samples = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1],
        ]
    )
    actions = probability_policy.policy_decoder.decode_samples(samples)
    torch.testing.assert_close(actions, torch.tensor([0, 0, 0, 1, 1]))
    assert probability_policy.majority_blocks == (tuple(range(6)),)
    assert probability_policy.unused_majority_qubits == tuple()

    inputs = torch.linspace(-0.3, 0.4, 6)
    expected = probability_policy(inputs)
    actual = sparse_policy(inputs)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    (-torch.log(expected[0])).backward()
    (-torch.log(actual[0])).backward()
    torch.testing.assert_close(
        next(sparse_policy.qlayer.parameters()).grad,
        next(probability_policy.qlayer.parameters()).grad,
        atol=1e-6,
        rtol=1e-6,
    )


@pytest.mark.parametrize("n_actions", [3, 4])
def test_multiaction_parity_expectations_match_probs_and_gradients(n_actions):
    torch.manual_seed(21)
    probability_policy = _make_parity_policy(
        n_actions, "probs", "default.qubit", "backprop"
    )
    expectation_policy = _make_parity_policy(
        n_actions, "expectation", "lightning.qubit", "adjoint"
    )
    expectation_policy.qlayer.load_state_dict(probability_policy.qlayer.state_dict())

    inputs = torch.linspace(-0.35, 0.25, 6)
    expected = probability_policy(inputs)
    actual = expectation_policy(inputs)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual.sum(), torch.ones(()), atol=1e-6, rtol=0)

    expected_loss = -torch.log(expected[0]) + 0.15 * expected[-1]
    actual_loss = -torch.log(actual[0]) + 0.15 * actual[-1]
    expected_loss.backward()
    actual_loss.backward()
    torch.testing.assert_close(
        next(expectation_policy.qlayer.parameters()).grad,
        next(probability_policy.qlayer.parameters()).grad,
        atol=1e-6,
        rtol=1e-6,
    )
