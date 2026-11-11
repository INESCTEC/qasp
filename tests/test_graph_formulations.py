"""Independent combinatorial oracles, encodings and solver integration checks."""

from itertools import combinations, permutations, product

import numpy as np
import pytest
import torch

from qasp import GraphSolution, PCE, QAOA, QUBO
from qasp.optimization import LRQAOA, RecursiveQAOA, WarmStartQAOA


CASES = [
    ("maxcut", 4, [(0, 1, 2), (1, 2, -0.5), (2, 3, 3)], {}),
    ("maximum_independent_set", 4, [(0, 1), (1, 2), (2, 3)], {"weights": [1, 3, 2, 1]}),
    ("minimum_vertex_cover", 4, [(0, 1), (1, 2), (2, 3)], {"weights": [1, 3, 2, 1]}),
    ("maximum_clique", 4, [(0, 1), (1, 2), (0, 2), (2, 3)], {"weights": [1, 3, 2, 5]}),
    ("maximum_matching", 4, [(0, 1, 3), (1, 2, 5), (2, 3, 3)], {}),
    ("minimum_dominating_set", 3, [(0, 1), (1, 2)], {"weights": [1, 3, 1]}),
    ("minimum_edge_cover", 3, [(0, 1, 2), (1, 2, 3), (0, 2, 1)], {}),
    ("graph_coloring", 3, [(0, 1), (1, 2), (0, 2)], {"n_colors": 3}),
    ("max_k_cut", 3, [(0, 1, 2), (1, 2, 3), (0, 2, 1)], {"n_parts": 2}),
    ("graph_partition", 4, [(0, 1, 2), (1, 2, 3), (2, 3, 1)], {}),
    ("densest_k_subgraph", 4, [(0, 1, 2), (1, 2, 3), (2, 3, 1)], {"k": 2}),
    ("hamiltonian_path", 3, [(0, 1), (1, 2)], {}),
    ("hamiltonian_cycle", 3, [(0, 1), (1, 2), (0, 2)], {}),
    ("traveling_salesperson", 3, [(0, 1, 2), (1, 2, 3), (0, 2, 1)], {}),
    ("hamiltonian_path", 3, [(1, 0), (0, 2)], {"directed": True}),
    ("hamiltonian_cycle", 3, [(1, 0), (0, 2), (2, 1)], {"directed": True}),
    ("traveling_salesperson", 3, [(0, 1, 1), (1, 2, 2), (2, 0, 3), (1, 0, 4), (2, 1, 5), (0, 2, 6)], {"directed": True}),
    ("maximum_matching", 2, [], {}),
    ("minimum_dominating_set", 2, [], {}),
    ("graph_partition", 3, [], {"subset_size": 0}),
    ("densest_k_subgraph", 3, [(0, 1)], {"k": 3}),
    ("hamiltonian_path", 1, [], {}),
    ("max_k_cut", 2, [(0, 1)], {"n_parts": 1}),
    ("maximum_independent_set", 3, [(0, 1)], {"weights": [0, 0, 0]}),
    ("hamiltonian_path", 4, [(0, 1), (1, 2), (2, 3)], {}),
    ("traveling_salesperson", 4, [(0, 1, 2), (0, 2, 7), (0, 3, 3), (1, 2, 1), (1, 3, 6), (2, 3, 4)], {}),
]


def classical_optimum(name, n, edges, options):
    """Solve directly over subsets, labelings or permutations; no QUBO helpers."""
    graph = {(e[0], e[1]): e[2] if len(e) == 3 else 1 for e in edges}
    weights = options.get("weights", [1] * n)
    candidates = []
    if name in {"hamiltonian_path", "hamiltonian_cycle", "traveling_salesperson"}:
        if not options.get("directed"):
            graph.update({(v, u): w for (u, v), w in list(graph.items())})
        for order in permutations(range(n)):
            route = order if name == "hamiltonian_path" else (*order, order[0])
            arcs = list(zip(route, route[1:]))
            if all(arc in graph for arc in arcs):
                candidates.append(sum(graph[arc] for arc in arcs) if name == "traveling_salesperson" else 0)
        return min(candidates) if candidates else None
    if name in {"graph_coloring", "max_k_cut"}:
        for labels in product(range(options.get("n_colors", options.get("n_parts"))), repeat=n):
            cut = sum(w for (u, v), w in graph.items() if labels[u] != labels[v])
            if name == "max_k_cut":
                candidates.append(cut)
            elif all(labels[u] != labels[v] for u, v in graph):
                candidates.append(0)
        return max(candidates) if candidates else None
    if name in {"maximum_matching", "minimum_edge_cover"}:
        for choice in product((0, 1), repeat=len(edges)):
            degrees = [0] * n
            for keep, edge in zip(choice, edges):
                if keep:
                    degrees[edge[0]] += 1
                    degrees[edge[1]] += 1
            valid = max(degrees) <= 1 if name == "maximum_matching" else min(degrees) >= 1
            if valid:
                candidates.append(sum(w * keep for w, keep in zip(graph.values(), choice)))
        return max(candidates) if name == "maximum_matching" else min(candidates)
    for bits in product((0, 1), repeat=n):
        chosen = {v for v in range(n) if bits[v]}
        score = sum(weights[v] for v in chosen)
        if name == "maximum_independent_set" and any(bits[u] and bits[v] for u, v in graph):
            continue
        if name == "minimum_vertex_cover" and any(not bits[u] and not bits[v] for u, v in graph):
            continue
        if name == "maximum_clique" and any((u, v) not in graph for u, v in combinations(sorted(chosen), 2)):
            continue
        if name == "minimum_dominating_set":
            dominated = chosen | {v for u, v in graph if u in chosen} | {u for u, v in graph if v in chosen}
            if len(dominated) != n:
                continue
        if name in {"graph_partition", "densest_k_subgraph"}:
            k = options.get("subset_size", n // 2) if name == "graph_partition" else options["k"]
            if len(chosen) != k:
                continue
        if name in {"maxcut", "graph_partition"}:
            score = sum(w for (u, v), w in graph.items() if bits[u] != bits[v])
        elif name == "densest_k_subgraph":
            score = sum(w for (u, v), w in graph.items() if bits[u] and bits[v])
        candidates.append(score)
    return min(candidates) if name.startswith("minimum_") or name == "graph_partition" else max(candidates)


@pytest.fixture(autouse=True)
def double_torch():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


@pytest.mark.parametrize("name,n,edges,options", CASES)
def test_all_global_minima_match_independent_combinatorial_optimum(name, n, edges, options):
    problem = getattr(QUBO, name)(n, edges, **options)
    states = np.array(list(product((0, 1), repeat=problem.n_variables)))
    energies = problem.offset + np.einsum("bi,ij,bj->b", states, problem.matrix, states)
    optimum = classical_optimum(name, n, edges, options)
    expected_energy = -optimum if problem.metadata["sense"] == "max" else optimum
    assert energies.min() == pytest.approx(expected_energy)
    for bits in states[np.isclose(energies, energies.min(), rtol=0, atol=1e-10)]:
        decoded = problem.decode(bits)
        assert isinstance(decoded, GraphSolution)
        assert decoded.feasible and decoded.encoding_feasible, decoded
        assert decoded.objective == pytest.approx(optimum)
        assert decoded.energy == pytest.approx(expected_energy)
    assert len(problem.variable_labels) == problem.n_variables
    assert len(set(problem.variable_labels)) == problem.n_variables
    assert problem.n_decision_variables <= problem.n_variables
    # Every computational basis energy, not only the optimum, must survive Ising conversion.
    offset, fields, couplings = problem.to_ising()
    z = 1 - 2 * states
    ising = np.full(len(states), offset)
    for i, h in fields.items():
        ising += h * z[:, i]
    for (i, j), coupling in couplings.items():
        ising += coupling * z[:, i] * z[:, j]
    np.testing.assert_allclose(ising, energies, atol=1e-11)


@pytest.mark.parametrize("name,n,edges,options", CASES[:14])
@pytest.mark.parametrize("solver_class", [QAOA, PCE])
def test_every_formulation_trains_and_decodes_with_both_solver_families(name, n, edges, options, solver_class):
    problem = getattr(QUBO, name)(n, edges, **options)
    solver = solver_class(problem, backend="torch", device="default.qubit", diff_method="backprop", seed=8)
    theta = np.linspace(0.1, 0.7, solver.n_parameters)
    gradient = solver.gradient(theta)
    assert gradient.shape == theta.shape and np.isfinite(gradient).all()
    result = solver.solve(theta, optimizer="Adam", optimizer_options={"max_steps": 2, "lr": 0.01})
    decoded = problem.decode(result.bitstring)
    assert np.isfinite(result.objective)
    assert decoded.energy == pytest.approx(result.energy)
    assert len(result.history) == 2


@pytest.mark.parametrize("device,diff,kwargs", [
    ("default.qubit", "backprop", {}),
    ("lightning.qubit", "adjoint", {}),
    ("PauliPropagationSimulator", "backprop", {"k": None, "prune": False}),
    ("IQPSimulator", "backprop", {"backend": "tn"}),
    ("IQPSimulator", "backprop", {"backend": "van_den_nest", "vn_sampling": "enumerate"}),
])
def test_constrained_pce_gradients_across_compatible_simulators(device, diff, kwargs):
    problem = QUBO.minimum_vertex_cover(3, [(0, 1), (1, 2)], weights=[1, 2, 1])
    circuit = {"ansatz": "AngleAnsatz", "ansatz_config": {"n_layers": 1,
               "rotations": ["X 0", "X 1", "X 2", "XX 0 1", "XX 1 2"],
               "include_imprimitive": False}}
    shared = dict(problem=problem, circuit=circuit, n_qubits=3, paulis="Z", backend="torch")
    solver = PCE(**shared, device=device, diff_method=diff, device_kwargs=kwargs)
    reference = PCE(**shared, device="default.qubit", diff_method="backprop")
    theta = np.linspace(-0.7, 0.8, solver.n_parameters)
    assert solver.loss(theta) == pytest.approx(reference.loss(theta), abs=1e-10)
    eps = 1e-6
    numeric = [(reference.loss(theta + eps * row) - reference.loss(theta - eps * row)) / (2 * eps)
               for row in np.eye(len(theta))]
    np.testing.assert_allclose(solver.gradient(theta), numeric, atol=2e-5, rtol=2e-5)


def test_other_qaoa_variants_accept_graph_formulations():
    problem = QUBO.maximum_independent_set(3, [(0, 1), (1, 2)])
    solvers = [QAOA(problem, parametrization="multi_angle", seed=2),
               WarmStartQAOA(problem, warm_start=[0.8, 0.2, 0.8], seed=2)]
    for solver in solvers:
        result = solver.solve(optimizer_options={"maxiter": 2})
        assert problem.decode(result.bitstring).energy == pytest.approx(result.energy)
    ramp = LRQAOA(problem).solve()
    assert problem.decode(ramp.bitstring).energy == pytest.approx(ramp.energy)
    recursive = RecursiveQAOA(problem, cutoff=2, qaoa_options={"seed": 2},
                              optimizer_options={"maxiter": 2}).solve()
    assert problem.decode(recursive.bitstring).energy == pytest.approx(recursive.energy)


def test_decoders_do_not_silently_repair_invalid_candidates():
    problem = QUBO.graph_coloring(2, [(0, 1)], 2)
    assert problem.decode("1010").violations  # equal colors
    assert problem.decode("1100").solution["colors"] == (None, None)
    assert problem.decode("1100").objective is None
    path = QUBO.hamiltonian_path(3, [(0, 1), (1, 2)])
    assert path.decode("000000000").solution["path"] is None
    # Permutation 0,2,1 uses a missing edge: preserve the candidate, flag it.
    bad_route = path.decode("100001010")
    assert bad_route.solution["path"] == (0, 2, 1)
    assert not bad_route.feasible and bad_route.objective is None


def test_graph_feasibility_and_slack_consistency_are_distinct():
    problem = QUBO.minimum_dominating_set(2, [(0, 1)])
    bad = problem.decode("1100")
    good = problem.decode("1111")
    assert bad.feasible and not bad.encoding_feasible
    assert good.feasible and good.encoding_feasible
    assert bad.objective == good.objective == 2
    assert bad.energy > good.energy
    empty_matching = QUBO.maximum_matching(2, [])
    assert empty_matching.n_decision_variables == 0
    assert empty_matching.decode("1").feasible
    assert not empty_matching.decode("1").encoding_feasible


@pytest.mark.parametrize("degree", range(1, 9))
def test_bounded_slacks_cover_exact_integer_range(degree):
    problem = QUBO.minimum_dominating_set(degree + 1, [(0, v) for v in range(1, degree + 1)])
    group = problem.metadata["slacks"][0]
    values = {sum(w * b for w, b in zip(group["weights"], bits))
              for bits in product((0, 1), repeat=len(group["indices"]))}
    assert values == set(range(degree + 1))


def test_metadata_and_normalization_contracts():
    assert len(QUBO.formulations()) == 14
    problem = QUBO.minimum_vertex_cover(3, [(2, 1), (0, 1)], weights={1: 2})
    assert problem.metadata["vertex_weights"] == (1, 2, 1)
    assert problem.metadata["edges"] == ((0, 1, 1), (1, 2, 1))
    copied = problem.metadata
    copied["name"] = "modified"
    assert problem.formulation == "minimum_vertex_cover"
    custom = QUBO(np.eye(2))
    assert custom.formulation is None and custom.metadata == {}
    assert custom.variable_labels == ("x[0]", "x[1]")
    with pytest.raises(ValueError, match="named graph"):
        custom.decode("01")
    # Preserve signed weights and duplicate-edge aggregation from existing MaxCut.
    maxcut = QUBO.maxcut(2, [(0, 1, 3), (1, 0, -1)])
    assert maxcut.energy("01") == -2
    assert maxcut.decode("01").objective == 2


def test_infeasible_graphs_remain_detectably_infeasible():
    triangle = [(0, 1), (1, 2), (0, 2)]
    for problem in (QUBO.graph_coloring(3, triangle, 2), QUBO.hamiltonian_cycle(3, triangle[:2]),
                    QUBO.traveling_salesperson(3, [(0, 1, 0), (1, 2, 0)])):
        bits, energy = problem.brute_force()
        assert energy > 0 and not problem.decode(bits).feasible
    with pytest.raises(ValueError, match="isolated"):
        QUBO.minimum_edge_cover(3, [(0, 1)])


def test_explicit_small_penalties_are_reported_not_claimed_sufficient():
    problem = QUBO.maximum_independent_set(2, [(0, 1)], weights=[4, 4], penalty=0.5)
    assert not problem.metadata["penalty_is_sufficient"]
    assert problem.metadata["sufficient_penalty_bound"] == 4
    assert not problem.decode(problem.brute_force()[0]).feasible


@pytest.mark.parametrize("factory,args,kwargs", [
    ("maximum_clique", (2.5, []), {}),
    ("maximum_clique", (True, []), {}),
    ("maximum_clique", (2, [(0, 2)]), {}),
    ("maximum_clique", (2, [(0.5, 1)]), {}),
    ("maximum_clique", (2, [(0, 0)]), {}),
    ("maximum_clique", (2, [(0, 1), (1, 0)]), {}),
    ("maximum_clique", (2, [(0, 1, 2)]), {}),
    ("maximum_clique", (2, []), {"weights": [-1, 1]}),
    ("maximum_clique", (2, []), {"weights": [1]}),
    ("maximum_clique", (2, []), {"weights": {2: 1}}),
    ("maximum_matching", (2, [(0, 1, -1)]), {}),
    ("maximum_matching", (2, [(0, 1, np.inf)]), {}),
    ("maximum_matching", (2, []), {"penalty": 0}),
    ("maximum_matching", (2, []), {"penalty": np.nan}),
    ("graph_coloring", (2, [], 0), {}),
    ("graph_partition", (2, []), {"subset_size": 3}),
    ("densest_k_subgraph", (2, [], -1), {}),
    ("hamiltonian_cycle", (2, [(0, 1)]), {}),
    ("hamiltonian_path", (2, [(0, 1)]), {"directed": "yes"}),
    ("traveling_salesperson", (3, [(0, 1, 1e308)]), {}),
])
def test_invalid_graph_inputs_fail_clearly(factory, args, kwargs):
    with pytest.raises(ValueError):
        getattr(QUBO, factory)(*args, **kwargs)


def test_small_nonzero_weights_are_not_lost_in_ising_conversion():
    problem = QUBO.maxcut(2, [(0, 1, 1e-12)])
    matrix = np.asarray(problem.hamiltonian().matrix(wire_order=[0, 1]))
    expected = np.array([problem.energy(bits) for bits in ("00", "01", "10", "11")])
    np.testing.assert_allclose(np.diag(matrix), expected, rtol=1e-12, atol=1e-25)


@pytest.mark.parametrize("name", QUBO.formulations())
def test_every_three_vertex_graph_against_independent_oracle(name):
    weighted = {"maxcut", "maximum_matching", "minimum_edge_cover", "max_k_cut",
                "graph_partition", "densest_k_subgraph", "traveling_salesperson"}
    options = {"graph_coloring": {"n_colors": 2}, "max_k_cut": {"n_parts": 2},
               "densest_k_subgraph": {"k": 2}}.get(name, {})
    if name in {"maximum_independent_set", "minimum_vertex_cover", "maximum_clique", "minimum_dominating_set"}:
        options = {"weights": [1, 1.5, 2]}
    possible = list(combinations(range(3), 2))
    for mask in product((0, 1), repeat=3):
        pairs = [edge for keep, edge in zip(mask, possible) if keep]
        edges = [(u, v, 0.5 * (j + 1)) for j, (u, v) in enumerate(pairs)] if name in weighted else pairs
        if name == "minimum_edge_cover" and len({v for edge in pairs for v in edge}) < 3:
            with pytest.raises(ValueError, match="isolated"):
                QUBO.minimum_edge_cover(3, edges)
            continue
        problem = getattr(QUBO, name)(3, edges, **options)
        expected = classical_optimum(name, 3, edges, options)
        states = np.array(list(product((0, 1), repeat=problem.n_variables)))
        energies = problem.offset + np.einsum("bi,ij,bj->b", states, problem.matrix, states)
        ground_states = states[np.isclose(energies, energies.min(), atol=1e-10, rtol=0)]
        if expected is None:
            assert energies.min() > 0
            assert all(not problem.decode(bits).feasible for bits in ground_states)
        else:
            for bits in ground_states:
                decoded = problem.decode(bits)
                assert decoded.feasible and decoded.encoding_feasible
                assert decoded.objective == pytest.approx(expected)


@pytest.mark.parametrize("bits", ["010", "02", [0.5, 1], [[0, 1]]])
def test_decoder_validates_binary_assignment(bits):
    with pytest.raises(ValueError):
        QUBO.maxcut(2, [(0, 1)]).decode(bits)
