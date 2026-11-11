"""Graph-to-QUBO reductions and decoders (no graph-library dependency).

All factories minimize the resulting QUBO. Maximization objectives are negated;
constraints use nonnegative penalties. See docs/graph_problems.md for equations,
sufficient penalty bounds, binary-variable layouts and references.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import combinations
import operator

import numpy as np

from .problem import GraphSolution


GRAPH_FORMULATIONS = (
    "maxcut", "maximum_independent_set", "minimum_vertex_cover", "maximum_clique",
    "maximum_matching", "minimum_dominating_set", "minimum_edge_cover",
    "graph_coloring", "max_k_cut", "graph_partition", "densest_k_subgraph",
    "hamiltonian_path", "hamiltonian_cycle", "traveling_salesperson",
)
_VERTEX_WEIGHTED = {
    "maximum_independent_set", "minimum_vertex_cover", "maximum_clique",
    "minimum_dominating_set",
}
_EDGE_WEIGHTED = {
    "maxcut", "maximum_matching", "minimum_edge_cover", "max_k_cut",
    "graph_partition", "densest_k_subgraph", "traveling_salesperson",
}
_ROUTES = {"hamiltonian_path", "hamiltonian_cycle", "traveling_salesperson"}
_MAXIMIZE = {
    "maxcut", "maximum_independent_set", "maximum_clique", "maximum_matching",
    "max_k_cut", "densest_k_subgraph",
}
_FEASIBILITY = {"graph_coloring", "hamiltonian_path", "hamiltonian_cycle"}


def _integer(value, label, low, high=None):
    try:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{label} must be an integer.") from exc
    if result < low or (high is not None and result > high):
        raise ValueError(f"{label} must be >= {low}" + (f" and <= {high}." if high is not None else "."))
    return result


def _weight(value, label, *, signed=False):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number.") from exc
    if not np.isfinite(result) or (not signed and result < 0):
        raise ValueError(f"{label} must be finite" + ("." if signed else " and nonnegative."))
    return result


def _edges(n, edges, *, weighted, directed=False, aggregate=False):
    result = {}
    for edge in edges:
        try:
            edge = tuple(edge)
        except TypeError as exc:
            raise ValueError("Each edge must be a pair or a weighted triple.") from exc
        if len(edge) not in ((2, 3) if weighted else (2,)):
            raise ValueError("Edges must be pairs" + (" or weighted triples." if weighted else "."))
        u, v = (_integer(x, "edge vertex", 0, n - 1) for x in edge[:2])
        if u == v:
            raise ValueError("Self loops are not supported.")
        key = (u, v) if directed else tuple(sorted((u, v)))
        w = _weight(edge[2] if len(edge) == 3 else 1, "edge weight", signed=aggregate)
        if key in result and not aggregate:
            raise ValueError(f"Duplicate edge {key}; supply a simple graph.")
        result[key] = result.get(key, 0.0) + w
        if not np.isfinite(result[key]):
            raise ValueError("Aggregated edge weights must be finite.")
    return tuple((u, v, w) for (u, v), w in sorted(result.items()))


def _vertex_weights(n, weights):
    result = [1.0] * n
    if weights is not None:
        if isinstance(weights, Mapping):
            for v, w in weights.items():
                result[_integer(v, "weight vertex", 0, n - 1)] = _weight(w, "vertex weight")
        else:
            values = list(weights)
            if len(values) != n:
                raise ValueError("weights must have one entry per vertex.")
            result = [_weight(w, "vertex weight") for w in values]
    return tuple(result)


class _Polynomial:
    """Accumulate an upper-triangular binary polynomial before dense allocation."""

    def __init__(self):
        self.linear = {}
        self.quadratic = {}
        self.offset = 0.0

    def single(self, i, value):
        self.linear[i] = self.linear.get(i, 0.0) + value

    def pair(self, i, j, value):
        if i == j:
            self.single(i, value)
        else:
            key = tuple(sorted((i, j)))
            self.quadratic[key] = self.quadratic.get(key, 0.0) + value

    def square(self, terms, constant, multiplier):
        self.offset += multiplier * constant * constant
        for i, a in terms.items():
            self.single(i, multiplier * (a * a + 2 * constant * a))
        for (i, a), (j, b) in combinations(terms.items(), 2):
            self.pair(i, j, 2 * multiplier * a * b)


def _slack_weights(upper):
    if upper == 0:
        return ()
    powers = tuple(1 << i for i in range(upper.bit_length() - 1))
    return (*powers, upper - sum(powers))


def _build_graph_qubo(qubo_class, name, n_vertices, edges, **options):
    """Construct one validated graph formulation and retain its decoding metadata."""
    if name not in GRAPH_FORMULATIONS:
        raise ValueError(f"Unknown graph formulation: {name}.")
    n = _integer(n_vertices, "n_vertices", 1)
    directed = options.get("directed", False)
    if not isinstance(directed, (bool, np.bool_)):
        raise ValueError("directed must be a boolean.")
    directed = bool(directed)
    if name in {"hamiltonian_cycle", "traveling_salesperson"} and n < 3:
        raise ValueError("Cycles require at least three vertices.")
    edges = _edges(n, edges, weighted=name in _EDGE_WEIGHTED, directed=directed,
                   aggregate=name == "maxcut")
    edge_set = {(u, v) for u, v, _ in edges}
    neighbors = [set() for _ in range(n)]
    incident = [[] for _ in range(n)]
    for e, (u, v, _) in enumerate(edges):
        neighbors[u].add(v)
        neighbors[v].add(u)
        incident[u].append(e)
        incident[v].append(e)
    weights = _vertex_weights(n, options.get("weights")) if name in _VERTEX_WEIGHTED else ()
    edge_weights = [w for _, _, w in edges]
    if name == "minimum_edge_cover" and any(not group for group in incident):
        raise ValueError("An edge cover does not exist for a graph with isolated vertices.")
    bound = 0.0
    if name in _VERTEX_WEIGHTED:
        bound = sum(weights) if name == "minimum_dominating_set" else max(weights)
    elif name == "maximum_matching":
        bound = max(edge_weights, default=0.0)
    elif name in {"minimum_edge_cover", "max_k_cut", "graph_partition", "densest_k_subgraph"}:
        bound = sum(edge_weights)
    elif name == "traveling_salesperson":
        bound = n * max(edge_weights, default=0.0)
    penalty = options.get("penalty")
    if not np.isfinite(bound):
        raise ValueError("Total objective scale is too large for finite penalties.")
    if penalty is None:
        penalty = bound + max(1.0, 0.1 * bound)
    penalty = _weight(penalty, "penalty")
    if penalty <= 0:
        raise ValueError("penalty must be positive.")
    sense = "max" if name in _MAXIMIZE else "feasibility" if name in _FEASIBILITY else "min"
    data = dict(name=name, n_vertices=n, edges=edges, vertex_weights=weights, sense=sense,
                penalty=None if name == "maxcut" else penalty,
                sufficient_penalty_bound=bound,
                penalty_is_sufficient=name == "maxcut" or penalty > bound,
                directed=directed)
    p = _Polynomial()
    labels = [f"vertex[{v}]" for v in range(n)]
    decision_count = n

    if name == "maxcut":
        for u, v, w in edges:
            p.single(u, -w)
            p.single(v, -w)
            p.pair(u, v, 2 * w)
    elif name in {"maximum_independent_set", "minimum_vertex_cover", "maximum_clique"}:
        for v, w in enumerate(weights):
            p.single(v, w if name == "minimum_vertex_cover" else -w)
        pairs = edge_set if name != "maximum_clique" else set(combinations(range(n), 2)) - edge_set
        for u, v in sorted(pairs):
            if name == "minimum_vertex_cover":
                p.offset += penalty
                p.single(u, -penalty)
                p.single(v, -penalty)
            p.pair(u, v, penalty)
    elif name in {"maximum_matching", "minimum_edge_cover", "minimum_dominating_set"}:
        if name != "minimum_dominating_set":
            labels = [f"edge[{u},{v}]" for u, v, _ in edges]
            decision_count = len(labels)
        objective_weights = weights if name == "minimum_dominating_set" else edge_weights
        for i, w in enumerate(objective_weights):
            p.single(i, -w if name == "maximum_matching" else w)
        if name == "maximum_matching":
            for group in incident:
                for e, f in combinations(group, 2):
                    p.pair(e, f, penalty)
            if not labels:
                labels.append("auxiliary_zero")
                p.single(0, penalty)
                data["fixed_zero"] = (0,)
        else:
            groups = [tuple(sorted({v} | neighbors[v])) for v in range(n)] if name == "minimum_dominating_set" else incident
            slacks = []
            for v, group in enumerate(groups):
                sw = _slack_weights(len(group) - 1)
                indices = tuple(range(len(labels), len(labels) + len(sw)))
                labels.extend(f"slack[{v},{j}]" for j in range(len(sw)))
                terms = {i: 1 for i in group}
                terms.update({i: -w for i, w in zip(indices, sw)})
                p.square(terms, -1, penalty)
                slacks.append(dict(vertex=v, decisions=tuple(group), indices=indices,
                                   weights=sw, upper=len(group) - 1))
            data["slacks"] = tuple(slacks)
    elif name in {"graph_coloring", "max_k_cut"}:
        key = "n_colors" if name == "graph_coloring" else "n_parts"
        k = _integer(options[key], key, 1)
        data[key] = k
        labels = [f"{'color' if name == 'graph_coloring' else 'group'}[{v},{c}]" for v in range(n) for c in range(k)]
        decision_count = len(labels)
        for v in range(n):
            p.square({v * k + c: 1 for c in range(k)}, -1, penalty)
        for u, v, w in edges:
            if name == "max_k_cut":
                p.offset -= w
            for c in range(k):
                p.pair(u * k + c, v * k + c, penalty if name == "graph_coloring" else w)
    elif name in {"graph_partition", "densest_k_subgraph"}:
        k = options.get("subset_size") if name == "graph_partition" else options["k"]
        if k is None:
            k = n // 2
        k = _integer(k, "subset_size" if name == "graph_partition" else "k", 0, n)
        data["subset_size"] = k
        p.square(dict.fromkeys(range(n), 1), -k, penalty)
        for u, v, w in edges:
            if name == "graph_partition":
                p.single(u, w)
                p.single(v, w)
            p.pair(u, v, -2 * w if name == "graph_partition" else -w)
    elif name in _ROUTES:
        labels = [f"position[{v},{t}]" for v in range(n) for t in range(n)]
        decision_count = len(labels)
        costs = {(u, v): w for u, v, w in edges}
        if not directed:
            costs.update({(v, u): w for u, v, w in edges})
        for v in range(n):
            p.square({v * n + t: 1 for t in range(n)}, -1, penalty)
        for t in range(n):
            p.square({v * n + t: 1 for v in range(n)}, -1, penalty)
        for t in range(n - 1 if name == "hamiltonian_path" else n):
            next_t = (t + 1) % n
            for u in range(n):
                for v in range(n):
                    if (u, v) not in costs:
                        p.pair(u * n + t, v * n + next_t, penalty)
                    elif name == "traveling_salesperson":
                        p.pair(u * n + t, v * n + next_t, costs[u, v])

    data.update(variable_labels=tuple(labels), n_decision_variables=decision_count)
    problem = qubo_class.from_terms(len(labels), linear=p.linear, quadratic=p.quadratic, offset=p.offset)
    problem._graph_data = data
    return problem


def _decode_graph_solution(problem, bits):
    """Evaluate graph constraints and original objectives, without silent repair."""
    d = problem._graph_data
    name, n, edges = d["name"], d["n_vertices"], d["edges"]
    selected = tuple(int(i) for i in np.flatnonzero(bits[:d["n_decision_variables"]]))
    violations, encoding_violations = [], []
    solution = {}
    objective = None
    if name in _VERTEX_WEIGHTED or name in {"maxcut", "graph_partition", "densest_k_subgraph"}:
        chosen = set(selected)
        solution = {"vertices": selected}
        if name in _VERTEX_WEIGHTED:
            objective = sum(d["vertex_weights"][v] for v in selected)
        if name == "maximum_independent_set":
            violations.extend(f"Adjacent selected vertices: {u}, {v}." for u, v, _ in edges if u in chosen and v in chosen)
        elif name == "minimum_vertex_cover":
            violations.extend(f"Uncovered edge: {u}, {v}." for u, v, _ in edges if u not in chosen and v not in chosen)
        elif name == "maximum_clique":
            existing = {(u, v) for u, v, _ in edges}
            violations.extend(f"Missing clique edge: {u}, {v}." for u, v in combinations(selected, 2) if (u, v) not in existing)
        elif name == "minimum_dominating_set":
            dominated = chosen.copy()
            for u, v, _ in edges:
                if u in chosen:
                    dominated.add(v)
                if v in chosen:
                    dominated.add(u)
            violations.extend(f"Undominated vertex: {v}." for v in range(n) if v not in dominated)
        elif name in {"maxcut", "graph_partition"}:
            objective = sum(w for u, v, w in edges if bits[u] != bits[v])
            solution = {"partition": (tuple(v for v in range(n) if v not in chosen), selected)}
        elif name == "densest_k_subgraph":
            objective = sum(w for u, v, w in edges if u in chosen and v in chosen)
        if name in {"graph_partition", "densest_k_subgraph"} and len(selected) != d["subset_size"]:
            violations.append(f"Selected {len(selected)} vertices; expected {d['subset_size']}.")
    elif name in {"maximum_matching", "minimum_edge_cover"}:
        chosen_edges = tuple(edges[e][:2] for e in selected)
        solution = {"edges": chosen_edges}
        objective = sum(edges[e][2] for e in selected)
        degrees = [0] * n
        for u, v in chosen_edges:
            degrees[u] += 1
            degrees[v] += 1
        if name == "maximum_matching":
            violations.extend(f"Multiple matching edges at vertex {v}." for v, degree in enumerate(degrees) if degree > 1)
        else:
            violations.extend(f"Uncovered vertex: {v}." for v, degree in enumerate(degrees) if degree == 0)
    elif name in {"graph_coloring", "max_k_cut"}:
        k = d["n_colors"] if name == "graph_coloring" else d["n_parts"]
        rows = bits.reshape(n, k)
        colors = tuple(int(np.flatnonzero(row)[0]) if row.sum() == 1 else None for row in rows)
        violations.extend(f"Vertex {v} does not select exactly one label." for v, c in enumerate(colors) if c is None)
        solution = {"colors" if name == "graph_coloring" else "labels": colors,
                    "groups": tuple(tuple(v for v, label in enumerate(colors) if label == c) for c in range(k))}
        if name == "graph_coloring":
            violations.extend(f"Equal colors across edge {u}, {v}." for u, v, _ in edges if colors[u] is not None and colors[u] == colors[v])
            objective = 0.0 if not violations else None
        elif not violations:
            objective = sum(w for u, v, w in edges if colors[u] != colors[v])
    elif name in _ROUTES:
        rows = bits.reshape(n, n)
        violations.extend(f"Vertex {v} does not occupy exactly one position." for v in range(n) if rows[v].sum() != 1)
        violations.extend(f"Position {t} does not contain exactly one vertex." for t in range(n) if rows[:, t].sum() != 1)
        route = None
        if not violations:
            route = tuple(int(np.flatnonzero(rows[:, t])[0]) for t in range(n))
            if name != "hamiltonian_path":
                route += (route[0],)
            costs = {(u, v): w for u, v, w in edges}
            if not d["directed"]:
                costs.update({(v, u): w for u, v, w in edges})
            arcs = tuple(zip(route, route[1:]))
            violations.extend(f"Missing route edge: {u}, {v}." for u, v in arcs if (u, v) not in costs)
            if not violations:
                objective = sum(costs[arc] for arc in arcs) if name == "traveling_salesperson" else 0.0
        solution = {"path" if name == "hamiltonian_path" else "tour": route}
    for group in d.get("slacks", ()):
        coverage = sum(int(bits[i]) for i in group["decisions"])
        slack = sum(int(bits[i]) * w for i, w in zip(group["indices"], group["weights"]))
        if coverage - 1 != slack:
            encoding_violations.append(f"Coverage slack for vertex {group['vertex']} is inconsistent.")
    for i in d.get("fixed_zero", ()):
        if bits[i]:
            encoding_violations.append(f"Auxiliary bit {i} must be zero.")
    return GraphSolution(
        formulation=name, bitstring="".join(map(str, bits)), feasible=not violations,
        encoding_feasible=not violations and not encoding_violations,
        objective=None if objective is None else float(objective), sense=d["sense"],
        energy=problem.energy(bits), solution=solution, violations=tuple(violations),
        encoding_violations=tuple(encoding_violations),
    )
