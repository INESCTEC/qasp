import numpy as np
import pytest

from qasp.optimization import (
    LRQAOA,
    PENNYLANE_OPTIMIZERS,
    PCE,
    QAOA,
    QUBO,
    RecursiveQAOA,
    SCIPY_OPTIMIZERS,
    TORCH_OPTIMIZERS,
    WarmStartQAOA,
    build_pauli_correlation_encoding,
    create_optimizer,
    draw_optimization,
    draw_pce,
    draw_qaoa,
    draw_recursive_qaoa,
)


pytestmark = pytest.mark.filterwarnings("error:The 'id' argument is deprecated")

@pytest.fixture
def triangle_maxcut():
    return QUBO.maxcut(3, [(0, 1), (1, 2), (0, 2)])


def test_qubo_energy_ising_conversion_and_brute_force(triangle_maxcut):
    problem = triangle_maxcut
    offset, h, couplings = problem.to_ising()
    for index in range(2**problem.n_variables):
        bitstring = format(index, "03b")
        spins = 1.0 - 2.0 * np.asarray(list(map(int, bitstring)))
        ising_energy = offset
        ising_energy += sum(value * spins[i] for i, value in h.items())
        ising_energy += sum(
            value * spins[i] * spins[j]
            for (i, j), value in couplings.items()
        )
        assert ising_energy == pytest.approx(problem.energy(bitstring))
    bitstring, energy = problem.brute_force()
    assert len(bitstring) == 3
    assert energy == pytest.approx(-2.0)


def test_standard_qaoa_accepts_qasp_mixer_dictionary(triangle_maxcut):
    mixer = {
        "rotations": [
            "X 0",
            {"pauli": "Y", "wires": [1], "coefficient": 0.5},
            "XX 1 2",
        ],
        "initial_state": "plus",
    }
    solver = QAOA(
        triangle_maxcut,
        n_layers=2,
        mixer=mixer,
        parametrization="standard",
        seed=3,
    )
    assert solver.parameter_shape == (2, 2)
    probabilities = solver.probabilities(solver.initial_parameters("zeros"))
    assert probabilities.sum() == pytest.approx(1.0)
    assert len(solver.mixer_terms) == 3

    broadcast = QAOA(
        triangle_maxcut,
        mixer={"rotations": "XY", "wires": [0, 2]},
    )
    assert [term.pauli for term in broadcast.mixer_terms] == ["X", "X", "Y", "Y"]


def test_multi_angle_qaoa_assigns_one_parameter_per_gate(triangle_maxcut):
    solver = QAOA(
        triangle_maxcut, n_layers=2, parametrization="multi_angle", seed=4
    )
    assert len(solver.cost_terms) == 3
    assert len(solver.mixer_terms) == 3
    assert solver.parameter_shape == (2, 6)
    result = solver.solve(optimizer_options={"maxiter": 8})
    assert result.algorithm == "multi_angle_qaoa"
    assert result.parameters.shape == (12,)
    assert np.isfinite(result.objective)


def test_analytic_qaoa_uses_hamiltonian_and_adjoint_gradients(triangle_maxcut):
    solver = QAOA(triangle_maxcut, n_layers=1, seed=12)
    parameters = solver.initial_parameters("random")
    gradient = solver.gradient(parameters)
    epsilon = 1e-6
    finite_difference = np.empty_like(parameters)
    for index in range(parameters.size):
        forward = parameters.copy()
        backward = parameters.copy()
        forward[index] += epsilon
        backward[index] -= epsilon
        finite_difference[index] = (
            solver.expectation(forward) - solver.expectation(backward)
        ) / (2.0 * epsilon)

    assert solver.diff_method == "adjoint"
    assert solver.cost_hamiltonian is not None
    np.testing.assert_allclose(gradient, finite_difference, atol=1e-5)

    result = solver.solve(
        optimizer="BFGS",
        optimizer_options={"maxiter": 2},
        use_gradients=True,
    )
    assert np.isfinite(result.objective)


@pytest.mark.parametrize("parametrization", ["standard", "multi_angle"])
def test_warm_start_qaoa_supports_both_parametrizations(
    triangle_maxcut, parametrization
):
    solver = WarmStartQAOA(
        triangle_maxcut,
        n_layers=1,
        parametrization=parametrization,
        warm_start=[0.2, 0.8, 0.2],
    )
    result = solver.solve(
        initial_parameters="zeros", optimizer_options={"maxiter": 3}
    )
    assert "warm_start" in result.algorithm
    np.testing.assert_allclose(result.metadata["warm_start"], [0.25, 0.75, 0.25])
    assert result.probabilities.sum() == pytest.approx(1.0)


def test_lr_qaoa_uses_deterministic_opposing_ramps(triangle_maxcut):
    solver = LRQAOA(
        triangle_maxcut, n_layers=4, delta_gamma=0.6, delta_beta=0.3
    )
    np.testing.assert_allclose(
        solver.ramp_parameters[:, 0], [0.15, 0.3, 0.45, 0.6]
    )
    np.testing.assert_allclose(
        solver.ramp_parameters[:, 1], [0.3, 0.225, 0.15, 0.075]
    )
    result = solver.solve()
    assert result.algorithm == "lr_qaoa"
    assert result.metadata["ramp_parameters"].shape == (4, 2)


def test_finite_shot_training_and_post_training_sampling(triangle_maxcut):
    solver = QAOA(triangle_maxcut, n_layers=1, shots=200, seed=5)
    assert solver.training_shots == 200
    assert solver.diff_method == "parameter-shift"

    # Finite-shot parameter-shift gradients are supported.
    gradient = solver.gradient(solver.initial_parameters("zeros"))
    assert gradient.shape == (2,)
    assert np.isfinite(gradient).all()

    # COBYLA trains directly against noisy Hamiltonian expectation estimates.
    result = solver.solve(
        initial_parameters="zeros",
        optimizer="COBYLA",
        optimizer_options={"maxiter": 4},
        result_shots=500,
    )
    assert result.shots == 500
    assert sum(result.counts.values()) == 500
    assert result.probabilities.sum() == pytest.approx(1.0)
    assert result.solution_probability == pytest.approx(
        result.counts[result.bitstring] / 500
    )

    exact = solver.probabilities(result.parameters, shots=None)
    sampled = solver.estimate_probabilities(result.parameters, shots=100)
    assert exact.sum() == pytest.approx(1.0)
    assert sampled.sum() == pytest.approx(1.0)
    assert sum(solver.solution_counts(result.parameters, shots=50).values()) == 50

    with pytest.raises(ValueError, match="requires analytic execution"):
        QAOA(triangle_maxcut, shots=100, diff_method="adjoint")


def test_qaoa_drawing_supports_text_symbolic_tikz_and_compression(
    triangle_maxcut,
):
    import matplotlib.pyplot as plt

    solver = QAOA(triangle_maxcut, n_layers=3)
    text = draw_qaoa(solver, output="text")
    assert "H" in text
    assert "Probs" in text

    figure, axes = draw_qaoa(solver, output="matplotlib")
    assert figure is not None
    assert axes is not None
    plt.close(figure)

    tikz = solver.draw(
        output="tikz",
        tikz_options={"standalone": True},
    )
    assert tikz.startswith(r"\documentclass[tikz,border=4pt]{standalone}")
    assert r"\gamma_{1}" in tikz
    assert r"\beta_{1}" in tikz
    assert r"R_{ZZ}" in tikz
    assert tikz.count(r"\meter") == triangle_maxcut.n_variables

    compressed = draw_qaoa(
        solver,
        output="tikz",
        tikz_options={"compress": True},
    )
    assert r"U(\theta_{2})" in compressed
    assert r"U(\theta_{3})" in compressed


def test_all_qaoa_variants_have_paper_ready_parameter_labels(triangle_maxcut):
    multi = draw_optimization(
        problem=triangle_maxcut,
        algorithm="multi_angle_qaoa",
        config={"n_layers": 1},
        output="tikz",
    )
    assert r"\gamma_{1,1}" in multi
    assert r"\beta_{1,1}" in multi

    warm = WarmStartQAOA(
        triangle_maxcut,
        warm_start=[0.2, 0.8, 0.2],
    ).draw(output="tikz")
    assert r"\vartheta_{1}" in warm
    assert r"\beta_{1}" in warm

    ramp = LRQAOA(triangle_maxcut, n_layers=2).draw(output="tikz")
    assert r"\gamma_{1}" in ramp
    assert r"\beta_{2}" in ramp


def test_recursive_qaoa_reconstructs_original_assignment(triangle_maxcut):
    result = RecursiveQAOA(
        triangle_maxcut,
        cutoff=2,
        optimizer_options={"maxiter": 8},
    ).solve()
    assert len(result.bitstring) == 3
    assert result.energy == pytest.approx(-2.0)
    assert len(result.elimination_rules) == 1
    assert len(result.intermediate_results) == 1


def test_pce_uses_dictionary_defined_qasp_ansatz(triangle_maxcut):
    circuit = {
        "ansatz": "AngleAnsatz",
        "ansatz_config": {
            "n_layers": 1,
            "rotations": ["Y 0", "Y 1", "ZZ 0 1"],
            "include_hadamards_initial": True,
        },
        "device": "default.qubit",
    }
    solver = PCE(triangle_maxcut, circuit=circuit, seed=7)
    assert solver.n_qubits == 2
    assert solver.pauli_labels == ["XX 0 1", "YY 0 1", "ZZ 0 1"]
    result = solver.solve(optimizer_options={"maxiter": 5})
    assert len(result.bitstring) == 3
    assert result.metadata["expectations"].shape == (3,)
    assert np.isfinite(result.objective)

    tikz = draw_pce(
        solver,
        parameters=result,
        output="tikz",
        tikz_options={"standalone": True},
    )
    assert r"\begin{quantikz}" in tikz
    assert r"\theta_{1}" in tikz
    assert r"\meter" in tikz


def test_recursive_qaoa_drawing_selects_a_completed_step(triangle_maxcut):
    solver = RecursiveQAOA(
        triangle_maxcut,
        cutoff=2,
        optimizer_options={"maxiter": 4},
    )
    with pytest.raises(RuntimeError, match="before drawing"):
        draw_recursive_qaoa(solver, output="tikz")
    solver.solve()
    tikz = solver.draw(step=0, output="tikz")
    assert r"\begin{quantikz}" in tikz
    assert r"\gamma_{1}" in tikz


def test_pce_capacity_and_optimizer_factory():
    n_qubits, terms = build_pauli_correlation_encoding(100)
    assert n_qubits == 9
    assert len(terms) == 100
    optimizer = create_optimizer("powell", maxiter=2)
    result = optimizer(lambda values: float(np.sum(values**2)), [1.0, -1.0])
    assert result.fun >= 0.0


def test_optimizer_families_are_discoverable():
    assert {"COBYLA", "BFGS"} <= set(SCIPY_OPTIMIZERS)
    assert {"Adam", "QNG", "SPSA", "QNSPSA", "ShotAdaptive"} <= set(
        PENNYLANE_OPTIMIZERS
    )
    assert {"Adam", "AdamW", "SGD", "RMSprop", "LBFGS"} <= set(
        TORCH_OPTIMIZERS
    )


def test_torch_backend_supports_adam_and_adjoint_gradients(triangle_maxcut):
    solver = QAOA(triangle_maxcut, backend="torch", seed=21)
    parameters = solver.initial_parameters("random")
    gradient = solver.gradient(parameters)
    assert solver.interface == "torch"
    assert gradient.shape == (solver.n_parameters,)
    assert np.isfinite(gradient).all()

    result = solver.solve(
        optimizer="Adam",
        optimizer_options={"max_steps": 2, "lr": 0.05},
    )
    assert result.metadata["backend"] == "torch"
    assert result.metadata["optimizer"] == "Adam"
    assert len(result.history) == 2
    assert np.isfinite(result.objective)


def test_torch_backend_trains_with_finite_shots(triangle_maxcut):
    solver = QAOA(triangle_maxcut, backend="torch", shots=100, seed=22)
    result = solver.solve(
        initial_parameters="zeros",
        optimizer="SGD",
        optimizer_options={"max_steps": 1, "lr": 0.02},
        result_shots=100,
    )
    assert solver.diff_method == "parameter-shift"
    assert len(result.history) == 1
    assert sum(result.counts.values()) == 100


@pytest.mark.parametrize(
    ("optimizer", "options"),
    [
        ("Adam", {"max_steps": 1, "stepsize": 0.03}),
        ("QNG", {"max_steps": 1, "stepsize": 0.02, "lam": 1e-3}),
        ("SPSA", {"max_steps": 1, "a": 0.02, "c": 0.1}),
        (
            "QNSPSA",
            {
                "max_steps": 1,
                "stepsize": 0.02,
                "blocking": False,
                "seed": 23,
            },
        ),
    ],
)
def test_qaoa_supports_pennylane_optimizers(
    triangle_maxcut, optimizer, options
):
    solver = QAOA(triangle_maxcut, seed=23)
    result = solver.solve(optimizer=optimizer, optimizer_options=options)
    assert result.metadata["backend"] == "numpy"
    assert result.metadata["optimizer"] == optimizer
    assert len(result.history) == 1
    assert np.isfinite(result.objective)


def test_shot_adaptive_requires_and_uses_finite_shots():
    problem = QUBO.maxcut(2, [(0, 1)])
    with pytest.raises(ValueError, match="finite shots"):
        QAOA(problem).solve(
            optimizer="ShotAdaptive", optimizer_options={"max_steps": 1}
        )

    result = QAOA(problem, shots=100, seed=24).solve(
        optimizer="ShotAdaptive",
        optimizer_options={"max_steps": 1, "min_shots": 10},
    )
    assert result.metadata["optimizer"] == "ShotAdaptive"
    assert len(result.history) == 1


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_pce_supports_gradient_optimizers_on_both_backends(
    triangle_maxcut, backend
):
    solver = PCE(triangle_maxcut, backend=backend, seed=25)
    gradient = solver.gradient(solver.initial_parameters("random"))
    assert gradient.shape == (solver.n_parameters,)
    assert np.isfinite(gradient).all()

    result = solver.solve(
        optimizer="Adam",
        optimizer_options={"max_steps": 1, "lr": 0.02},
    )
    assert result.metadata["backend"] == backend
    assert result.metadata["optimizer"] == "Adam"
    assert np.isfinite(result.objective)


def test_pce_rejects_qnode_only_natural_gradient(triangle_maxcut):
    solver = PCE(triangle_maxcut, seed=26)
    with pytest.raises(ValueError, match="single scalar QNode"):
        solver.solve(optimizer="QNG", optimizer_options={"max_steps": 1})
