"""Local tests of MPI routing, independent of an MPI-enabled plugin build."""

import numpy as np
import pennylane as qml
import pytest
import torch

from qasp.circuits import AngleAnsatz, CircuitBuilder
from qasp.classifiers import create_classifier
from qasp.classifiers import PQCClassifier
from qasp.devices import create_device
from qasp.genlearn import BornMachine
from qasp.optimization import LRQAOA, PCE, QAOA, QUBO, RecursiveQAOA, WarmStartQAOA
from qasp.qkernels import create_kernel


@pytest.fixture
def device_calls(monkeypatch):
    """Record plugin arguments, then execute the circuit on a local simulator."""
    original = qml.device
    calls = []

    def device(name, **kwargs):
        calls.append((name, kwargs.copy()))
        result = original(
            "default.qubit", wires=kwargs["wires"], shots=kwargs.get("shots")
        )
        result._mpi = kwargs.get("mpi", False)
        return result

    monkeypatch.setattr(qml, "device", device)
    return calls


def builder(n_qubits=3, n_ancillas=0):
    result = CircuitBuilder(n_qubits=n_qubits, n_ancillas=n_ancillas)
    result.set_ansatz(AngleAnsatz(n_qubits=n_qubits, n_layers=1, rotations="Y"))
    result.build()
    return result


@pytest.mark.parametrize("device", ["lightning.kokkos", "lightning.gpu", "custom.mpi"])
def test_builder_forwards_mpi_to_every_explicit_plugin(device_calls, device):
    circuit = builder(n_ancillas=1)
    node = circuit.compile(device=device, mpi=True, shots=20, device_kwargs={"seed": 7})
    name, options = device_calls[-1]
    assert name == device
    assert options["mpi"] is True
    assert options["wires"] == 4
    assert options["shots"] == 20
    assert options["seed"] == 7
    assert node.device is circuit.device


@pytest.mark.parametrize("measurement", ["probs", "sample", "expectation"])
def test_auto_mpi_keeps_ancillas_and_shots(device_calls, measurement):
    circuit = builder(n_ancillas=2)
    circuit.set_measurement(
        measurement,
        observables=[qml.PauliZ(0)] if measurement == "expectation" else None,
    )
    circuit.build()
    circuit.compile(device="auto", mpi=True, shots=30)
    name, options = device_calls[-1]
    assert name == "lightning.kokkos"
    assert options["mpi"] is True
    assert options["wires"] == 5
    assert options["shots"] == 30


def test_serial_devices_do_not_receive_mpi_and_invalid_requests_fail():
    device = create_device("default.qubit", wires=2)
    assert device is create_device(device, wires=2)
    with pytest.raises(TypeError):
        create_device("default.qubit", wires=2, mpi=True)
    with pytest.raises(ValueError, match="existing device"):
        create_device(device, wires=2, mpi=True)
    with pytest.raises(ValueError, match="explicit arguments"):
        create_device("default.qubit", wires=2, device_kwargs={"mpi": True})
    with pytest.raises(ValueError, match="does not support"):
        builder().compile(device="lightning.pp", mpi=True)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("variant", [QAOA, WarmStartQAOA, LRQAOA, PCE])
def test_optimization_reuses_distributed_device(device_calls, variant, backend):
    problem = QUBO.maxcut(3, [(0, 1), (1, 2)])
    solver = variant(
        problem, device="lightning.kokkos", mpi=True, backend=backend, seed=7
    )
    assert device_calls[-1][1]["mpi"] is True
    parameters = solver.initial_parameters()
    if variant is PCE:
        assert np.isfinite(solver.gradient(parameters)).all()
    else:
        assert np.isfinite(solver.gradient(parameters)).all()
        assert solver.sample_solutions(parameters, shots=12).shape == (12, 3)
    # Training, gradients, and sampling must not make a replacement serial device.
    assert len(device_calls) == 1


def test_pce_dictionary_and_recursive_options(device_calls):
    problem = QUBO.maxcut(3, [(0, 1), (1, 2)])
    PCE(problem, circuit={"device": "lightning.kokkos", "mpi": True})
    assert device_calls[-1][1]["mpi"] is True
    recursive = RecursiveQAOA(
        problem,
        device="lightning.kokkos",
        mpi=True,
        cutoff=2,
        optimizer_options={"maxiter": 3},
    )
    result = recursive.solve(result_shots=20, include_probabilities=False)
    assert len(result.bitstring) == 3
    assert all(options["mpi"] for _, options in device_calls)
    assert result.intermediate_results[0].probabilities is None


@pytest.mark.parametrize("alias", ["mpi", "enable_mpi"])
def test_classifier_factory_forwards_mpi_and_differentiates(device_calls, alias):
    config = {
        "ansatz": "AngleAnsatz",
        "ansatz_config": {"n_layers": 1, "rotations": "Y"},
        "feature_map": "AngleEmbedding",
        "feature_map_config": {"rotations": "Y"},
        "readout": "expectation",
        "observables": ["Z 0", "Z 1"],
        "device": "lightning.kokkos",
        alias: True,
        "diff_method": "adjoint",
    }
    model = create_classifier(config, n_qubits=2, n_classes=2)
    output = model(torch.tensor([0.1, 0.2]))
    output[0].backward()
    assert any(p.grad is not None for p in model.parameters())
    assert device_calls[-1][1]["mpi"] is True
    with pytest.raises(ValueError, match="must agree"):
        create_classifier({**config, "mpi": True, "enable_mpi": False}, 2, 2)


@pytest.mark.parametrize("kind", ["fidelity", "projected"])
def test_kernel_factory_forwards_mpi_and_executes(device_calls, kind):
    kernel = create_kernel(
        {
            "kernel_type": kind,
            "feature_map": "AngleEmbedding",
            "feature_map_config": {"rotations": "Y"},
            "device": "lightning.kokkos",
            "mpi": True,
            "device_kwargs": {"batch_obs": False},
        },
        n_qubits=2,
    )
    gram = kernel.gram([[0.1, 0.2], [0.3, 0.4]])
    assert np.isfinite(gram).all()
    assert device_calls[-1][1]["mpi"] is True
    assert device_calls[-1][1]["batch_obs"] is False


def test_born_machine_forwards_mpi(device_calls):
    BornMachine(
        builder(),
        qml.numpy.zeros((1, 3)),
        [[0, 0, 0]],
        device="lightning.kokkos",
        mpi=True,
    )
    assert device_calls[-1][1]["mpi"] is True


@pytest.mark.parametrize(
    "name", ["PQCBorn", "PQCSoftmax", "PQCBornVariance", "LearnablePQCBorn"]
)
def test_rl_policies_forward_mpi(monkeypatch, name):
    from qasp.rl import policies
    from qasp.feature_maps import AngleEmbedding

    circuit = builder()
    circuit.ansatz.feature_map = AngleEmbedding(n_qubits=3, features=3, rotations="Y")

    class Compiled(Exception):
        pass

    def compile(self, **kwargs):
        assert kwargs["mpi"] is True
        raise Compiled

    monkeypatch.setattr(CircuitBuilder, "compile", compile)
    options = {"n_qubits_measure": 1} if name == "LearnablePQCBorn" else {}
    if name == "PQCSoftmax":
        options["observables"] = ["Z 0", "Z 1"]
    with pytest.raises(Compiled):
        getattr(policies, name)(
            circuit_builder=circuit,
            n_qubits=3,
            device="lightning.kokkos",
            mpi=True,
            diff_method="best",
            **options,
        )


def test_counts_only_results_never_build_dense_probabilities(monkeypatch):
    solver = QAOA(QUBO.maxcut(3, [(0, 1)]), seed=1)

    def forbidden(*args, **kwargs):
        raise AssertionError("Dense probabilities requested")

    monkeypatch.setattr(solver, "probabilities", forbidden)
    monkeypatch.setattr(solver, "_sample_statistics", forbidden)
    result = solver.solve(
        optimizer_options={"maxiter": 3}, result_shots=50, include_probabilities=False
    )
    assert result.probabilities is None
    assert sum(result.counts.values()) == 50
    assert result.solution_probability == result.counts[result.bitstring] / 50
    assert 0 <= solver.probability_of("000", result.parameters, shots=20) <= 1


def test_direct_correlations_match_dense_reference():
    solver = QAOA(QUBO.maxcut(3, [(0, 1), (1, 2)]), seed=1)
    parameters = solver.initial_parameters()
    actual = solver.correlations(parameters)
    expected = RecursiveQAOA._correlations(solver.probabilities(parameters), 3)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, atol=1e-12)


def test_classifier_default_readout_does_not_enumerate_exponential_observables():
    from types import SimpleNamespace

    register = SimpleNamespace(measurement_qubits=list(range(40)))
    assert PQCClassifier._default_observables(register, 2) == ["Z 0", "Z 1"]
