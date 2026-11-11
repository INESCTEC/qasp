"""Fourteen graph problems through the same QUBO / QAOA interface.

Run from a checkout containing the graph factories. Exact references below use
at most nine binary variables; do not brute-force large one-hot/slack encodings.
"""

from qasp import Normal, QAOA, QUBO


def graph_problem_catalog():
    """Return one small instance of each named formulation, without training."""
    path = [(0, 1), (1, 2)]
    triangle = [*path, (0, 2)]
    costs = [(0, 1, 2.0), (1, 2, 3.0), (0, 2, 1.0)]
    return {
        "maxcut": QUBO.maxcut(3, costs),
        "maximum_independent_set": QUBO.maximum_independent_set(3, path, weights=[1, 1.5, 1]),
        "minimum_vertex_cover": QUBO.minimum_vertex_cover(3, path),
        "maximum_clique": QUBO.maximum_clique(3, triangle),
        "maximum_matching": QUBO.maximum_matching(3, costs),
        "minimum_dominating_set": QUBO.minimum_dominating_set(3, path),
        "minimum_edge_cover": QUBO.minimum_edge_cover(3, costs),
        "graph_coloring": QUBO.graph_coloring(3, triangle, n_colors=3),
        "max_k_cut": QUBO.max_k_cut(3, costs, n_parts=2),
        "graph_partition": QUBO.graph_partition(3, costs, subset_size=1),
        "densest_k_subgraph": QUBO.densest_k_subgraph(3, costs, k=2),
        "hamiltonian_path": QUBO.hamiltonian_path(3, path),
        "hamiltonian_cycle": QUBO.hamiltonian_cycle(3, triangle),
        "traveling_salesperson": QUBO.traveling_salesperson(3, costs),
    }


def main():
    """Print exact small references, then train and decode a four-vertex example."""
    for name, problem in graph_problem_catalog().items():
        bits, _ = problem.brute_force()
        solution = problem.decode(bits)
        assert solution.feasible and solution.encoding_feasible
        print(f"{name:26s} bits={problem.n_variables:2d} "
              f"{solution.sense}={solution.objective:g} {solution.solution}")

    problem = QUBO.maximum_independent_set(4, [(0, 1), (1, 2), (2, 3)], weights=[1, 2, 1, 2])
    solver = QAOA(problem, n_layers=2, backend="torch", device="default.qubit",
                  diff_method="backprop", init=Normal(0, 0.2), seed=7)
    result = solver.solve(optimizer="Adam", optimizer_options={"max_steps": 40, "lr": 0.05})
    print("QAOA candidate:", problem.decode(result.bitstring))
    print("Expected penalized energy:", result.objective)
    # This solver can be replaced with PCE or another QAOA variant. Decode with
    # the same original problem; optimizer success alone does not prove feasibility.


if __name__ == "__main__":
    main()
