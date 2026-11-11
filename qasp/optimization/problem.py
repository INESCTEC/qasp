"""QUBO problem definitions and QUBO-to-Ising conversion utilities."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from itertools import product
from typing import Iterable, Mapping, Sequence

import numpy as np
import pennylane as qml


@dataclass(frozen=True)
class PauliTerm:
    """One weighted Pauli word used in an optimization Hamiltonian."""

    coefficient: float
    pauli: str
    wires: tuple[int, ...]


@dataclass(frozen=True)
class GraphSolution:
    """Decoded graph candidate, separating its original objective from QUBO cost.

    feasible tests the graph constraints. encoding_feasible additionally checks
    auxiliary/slack bits. objective is the original score or cost (None if the
    assignment cannot represent it); sense is max, min, or feasibility. energy
    includes all penalties. solution contains vertices, edges, colors or a route.
    The violation tuples explain invalid candidates instead of repairing them.
    """

    formulation: str
    bitstring: str
    feasible: bool
    encoding_feasible: bool
    objective: float | None
    sense: str
    energy: float
    solution: dict
    violations: tuple[str, ...] = ()
    encoding_violations: tuple[str, ...] = ()


class QUBO:
    """A quadratic unconstrained binary optimization problem.

    The objective is ``offset + x.T @ matrix @ x`` for binary variables
    ``x[i] in {0, 1}``.  Non-symmetric matrices are accepted; their two
    off-diagonal entries are combined into one quadratic coefficient.
    """

    def __init__(self, matrix, offset: float = 0.0):
        """Copy a finite square coefficient matrix and constant energy offset.

        The objective is ``offset + x.T @ matrix @ x`` for binary x. Off-diagonal
        entries are not implicitly halved: a pair coefficient is Q[i,j]+Q[j,i].
        Require at least one variable; invalid shape/nonfinite coefficients raise
        ValueError. Subsequent changes to the caller's array do not change this QUBO."""
        matrix = np.asarray(matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be a square two-dimensional array.")
        if matrix.shape[0] < 1:
            raise ValueError("A QUBO must contain at least one variable.")
        if not np.isfinite(matrix).all() or not np.isfinite(offset):
            raise ValueError("QUBO coefficients and offset must be finite.")
        self.matrix = matrix.copy()
        self.offset = float(offset)
        self._graph_data = None

    @classmethod
    def from_terms(
        cls,
        n_variables: int,
        *,
        linear: Mapping[int, float] | Sequence[float] | None = None,
        quadratic: Mapping[tuple[int, int], float] | None = None,
        offset: float = 0.0,
    ) -> "QUBO":
        """Build a QUBO from explicit linear and quadratic coefficients."""
        n_variables = int(n_variables)
        if n_variables < 1:
            raise ValueError("n_variables must be positive.")
        matrix = np.zeros((n_variables, n_variables), dtype=float)
        if linear is not None:
            items = linear.items() if isinstance(linear, Mapping) else enumerate(linear)
            for index, value in items:
                index = int(index)
                if index < 0 or index >= n_variables:
                    raise ValueError(f"Linear index {index} is outside the QUBO.")
                matrix[index, index] += float(value)
        if quadratic is not None:
            for pair, value in quadratic.items():
                if len(pair) != 2:
                    raise ValueError("Quadratic keys must be pairs of indices.")
                i, j = map(int, pair)
                if i == j:
                    matrix[i, i] += float(value)
                    continue
                if min(i, j) < 0 or max(i, j) >= n_variables:
                    raise ValueError(f"Quadratic index {(i, j)} is outside the QUBO.")
                matrix[min(i, j), max(i, j)] += float(value)
        return cls(matrix, offset=offset)

    @classmethod
    def from_ising(
        cls,
        n_variables: int,
        linear: Mapping[int, float],
        quadratic: Mapping[tuple[int, int], float],
        *,
        offset: float = 0.0,
    ) -> "QUBO":
        """Create a QUBO from ``offset + sum h_i Z_i + sum J_ij Z_i Z_j``."""
        h = {int(i): float(value) for i, value in linear.items()}
        couplings = {
            tuple(sorted(map(int, pair))): float(value)
            for pair, value in quadratic.items()
        }
        qubo_offset = float(offset) + sum(h.values()) + sum(couplings.values())
        qubo_linear = {
            i: -2.0 * h.get(i, 0.0)
            - 2.0
            * sum(value for pair, value in couplings.items() if i in pair)
            for i in range(int(n_variables))
        }
        qubo_quadratic = {pair: 4.0 * value for pair, value in couplings.items()}
        return cls.from_terms(
            n_variables,
            linear=qubo_linear,
            quadratic=qubo_quadratic,
            offset=qubo_offset,
        )

    @classmethod
    def maxcut(cls, n_variables: int, edges: Iterable[Sequence[float]]) -> "QUBO":
        """Return the minimization QUBO for weighted MaxCut.

        Each edge is ``(i, j)`` or ``(i, j, weight)``.  The returned objective
        is the negative cut value, so minimizing it maximizes the cut.
        """
        return cls._from_graph("maxcut", n_variables, edges)

    @classmethod
    def _from_graph(cls, name, n_vertices, edges, **options):
        from .formulations import _build_graph_qubo

        return _build_graph_qubo(cls, name, n_vertices, edges, **options)

    @classmethod
    def formulations(cls) -> tuple[str, ...]:
        """List all named graph-problem factory methods available on QUBO."""
        from .formulations import GRAPH_FORMULATIONS

        return GRAPH_FORMULATIONS

    @classmethod
    def maximum_independent_set(cls, n_vertices, edges, *, weights=None, penalty=None):
        """Maximize selected vertex weight with no selected adjacent vertices.

        Vertices are integers 0..n_vertices-1; edges are unique undirected pairs.
        weights is a nonnegative length-N sequence or vertex-to-weight mapping
        (omitted entries default to one). None gives unit weights. penalty is
        finite positive; None chooses a value greater than the largest weight.
        Uses N binary variables, one per vertex. decode reports the selected set.
        """
        return cls._from_graph("maximum_independent_set", n_vertices, edges, weights=weights, penalty=penalty)

    @classmethod
    def minimum_vertex_cover(cls, n_vertices, edges, *, weights=None, penalty=None):
        """Minimize selected vertex cost while covering every edge.

        edges are unique undirected pairs over 0..N-1. weights is a nonnegative
        length-N sequence or mapping, defaulting to unit costs. penalty=None
        chooses a value greater than the largest cost; explicit penalties must
        be finite positive. Uses N variables. decode checks uncovered edges.
        """
        return cls._from_graph("minimum_vertex_cover", n_vertices, edges, weights=weights, penalty=penalty)

    @classmethod
    def maximum_clique(cls, n_vertices, edges, *, weights=None, penalty=None):
        """Maximize vertex weight in a clique, penalizing selected nonedges.

        edges are unique undirected pairs over 0..N-1. weights is a nonnegative
        length-N sequence or mapping, default one. penalty=None exceeds the
        largest weight. An explicit penalty must be finite positive. Uses N
        variables; decode lists the vertices and any missing clique edges.
        """
        return cls._from_graph("maximum_clique", n_vertices, edges, weights=weights, penalty=penalty)

    @classmethod
    def maximum_matching(cls, n_vertices, edges, *, penalty=None):
        """Maximize the weight of a set of vertex-disjoint edges.

        edges are unique (u,v) or (u,v,nonnegative_weight) entries. One variable
        per sorted undirected edge; penalty=None exceeds the largest weight.
        A graph with no edges uses one fixed-zero auxiliary bit for its trivial
        empty matching because QUBO requires at least one binary variable.
        """
        return cls._from_graph("maximum_matching", n_vertices, edges, penalty=penalty)

    @classmethod
    def minimum_dominating_set(cls, n_vertices, edges, *, weights=None, penalty=None):
        """Minimize vertex cost so every vertex is selected or has a selected neighbor.

        edges are unique undirected pairs; nonnegative vertex weights default
        to one and accept a sequence or mapping. Uses N decision bits plus
        bounded binary slacks for closed-neighborhood coverage. penalty=None
        exceeds the sum of vertex costs. decode distinguishes graph feasibility
        from consistency of the slack bits.
        """
        return cls._from_graph("minimum_dominating_set", n_vertices, edges, weights=weights, penalty=penalty)

    @classmethod
    def minimum_edge_cover(cls, n_vertices, edges, *, penalty=None):
        """Minimize edge cost while making every vertex incident to a selected edge.

        edges are unique undirected pairs or triples with nonnegative weights.
        Uses |E| decision bits plus bounded coverage slacks. Isolated vertices
        make this problem infeasible and raise ValueError. penalty=None exceeds
        the sum of all edge costs; an explicit value must be finite positive.
        """
        return cls._from_graph("minimum_edge_cover", n_vertices, edges, penalty=penalty)

    @classmethod
    def graph_coloring(cls, n_vertices, edges, n_colors, *, penalty=None):
        """Find a proper coloring with at most n_colors using N*n_colors bits.

        edges are unique undirected pairs; n_colors is a positive integer.
        Each vertex has a one-hot color row, flattened vertex first. The energy
        is zero exactly for proper encodings; infeasible graphs retain positive
        minimum energy. penalty defaults to 1 and must be finite positive.
        """
        return cls._from_graph("graph_coloring", n_vertices, edges, n_colors=n_colors, penalty=penalty)

    @classmethod
    def max_k_cut(cls, n_vertices, edges, n_parts, *, penalty=None):
        """Maximize edge weight between at most n_parts groups.

        edges are unique pairs or triples with nonnegative weights; n_parts is
        a positive integer. Uses N*n_parts one-hot variables, vertex first.
        Empty groups are allowed. penalty=None exceeds the sum of edge weights.
        decode returns group labels and the original cut weight.
        """
        return cls._from_graph("max_k_cut", n_vertices, edges, n_parts=n_parts, penalty=penalty)

    @classmethod
    def graph_partition(cls, n_vertices, edges, *, subset_size=None, penalty=None):
        """Minimize cut weight with exactly subset_size vertices on the selected side.

        edges are unique pairs or triples with nonnegative weights. subset_size
        is in [0,N], default N//2 (balanced sides, including odd N). Uses N
        variables and a squared cardinality penalty. penalty=None exceeds the
        sum of edge weights. decode reports both sides and cardinality failures.
        """
        return cls._from_graph("graph_partition", n_vertices, edges, subset_size=subset_size, penalty=penalty)

    @classmethod
    def densest_k_subgraph(cls, n_vertices, edges, k, *, penalty=None):
        """Maximize internal edge weight among exactly k selected vertices.

        edges are unique pairs or triples with nonnegative weights; k is an
        integer in [0,N]. Uses N variables and a squared cardinality penalty.
        penalty=None exceeds the sum of edge weights; explicit values must be
        finite positive. decode reports internal weight and selected vertices.
        """
        return cls._from_graph("densest_k_subgraph", n_vertices, edges, k=k, penalty=penalty)

    @classmethod
    def hamiltonian_path(cls, n_vertices, edges, *, directed=False, penalty=None):
        """Find a path visiting every vertex once using N*N position variables.

        edges are unique pairs, undirected unless directed=True. Variables are
        flattened vertex first, then position. No closing edge is required.
        penalty defaults to 1; zero energy certifies a valid path. decode does
        not turn invalid one-hot assignments into a route by taking argmax.
        """
        return cls._from_graph("hamiltonian_path", n_vertices, edges, directed=directed, penalty=penalty)

    @classmethod
    def hamiltonian_cycle(cls, n_vertices, edges, *, directed=False, penalty=None):
        """Find a Hamiltonian cycle on at least three vertices using N*N bits.

        edges are unique pairs, undirected unless directed=True. Variables are
        vertex-major one-hot positions, including the last-to-first adjacency
        constraint. penalty defaults to 1. decode returns a closed route only
        for a valid permutation and separately checks all required edges.
        """
        return cls._from_graph("hamiltonian_cycle", n_vertices, edges, directed=directed, penalty=penalty)

    @classmethod
    def traveling_salesperson(cls, n_vertices, edges, *, directed=False, penalty=None):
        """Minimize a Hamiltonian cycle's total edge cost on at least three vertices.

        edges are unique pairs (unit cost) or triples with finite nonnegative
        costs, undirected unless directed=True. Missing edges are forbidden,
        not zero-cost travel. Uses N*N vertex-major position bits. The default
        penalty exceeds N times the largest edge cost, a sufficient bound when
        a tour exists. Explicit penalties must be finite positive.
        """
        return cls._from_graph("traveling_salesperson", n_vertices, edges, directed=directed, penalty=penalty)

    @property
    def formulation(self) -> str | None:
        """Named graph formulation, or None for a custom matrix/terms/Ising QUBO."""
        return None if self._graph_data is None else self._graph_data["name"]

    @property
    def metadata(self) -> dict:
        """Return a copy of graph inputs, encoding layout and penalty information."""
        return deepcopy(self._graph_data or {})

    @property
    def variable_labels(self) -> tuple[str, ...]:
        """Binary-variable labels in circuit-wire order, including auxiliary bits."""
        if self._graph_data is None:
            return tuple(f"x[{i}]" for i in range(self.n_variables))
        return self._graph_data["variable_labels"]

    @property
    def n_decision_variables(self) -> int:
        """Number of graph decision bits before any slack/fixed-zero auxiliary bits."""
        return self.n_variables if self._graph_data is None else self._graph_data["n_decision_variables"]

    def decode(self, bitstring: str | Sequence[int]) -> GraphSolution:
        """Validate and decode a named graph QUBO candidate without repairing it.

        The returned original objective and graph feasibility are distinct from
        penalized QUBO energy and slack consistency. A custom matrix has no
        inferred graph interpretation and raises ValueError. Pass a solver's
        result.bitstring, not its continuous parameters.
        """
        from .formulations import _decode_graph_solution

        if self._graph_data is None:
            raise ValueError("decode requires a named graph formulation; custom QUBOs have no graph decoder.")
        self.energy(bitstring)  # Validate size and binary entries consistently.
        bits = np.asarray(list(bitstring) if isinstance(bitstring, str) else bitstring, dtype=int)
        return _decode_graph_solution(self, bits)

    @property
    def n_variables(self) -> int:
        """Number of binary variables, equal to the coefficient matrix dimension."""
        return int(self.matrix.shape[0])

    @property
    def linear(self) -> dict[int, float]:
        """Return nonzero diagonal coefficients as an index-to-weight dictionary."""
        return {
            i: float(self.matrix[i, i])
            for i in range(self.n_variables)
            if self.matrix[i, i] != 0.0
        }

    @property
    def quadratic(self) -> dict[tuple[int, int], float]:
        """Return nonzero unordered pair coefficients Q[i,j]+Q[j,i], keyed by i<j."""
        terms = {}
        for i in range(self.n_variables):
            for j in range(i + 1, self.n_variables):
                value = float(self.matrix[i, j] + self.matrix[j, i])
                if value != 0.0:
                    terms[(i, j)] = value
        return terms

    def energy(self, bitstring: str | Sequence[int]) -> float:
        """Evaluate the QUBO objective on a binary assignment."""
        if isinstance(bitstring, str):
            if any(bit not in "01" for bit in bitstring):
                raise ValueError("A bitstring may contain only '0' and '1'.")
            bits = np.fromiter((int(bit) for bit in bitstring), dtype=float)
        else:
            bits = np.asarray(bitstring, dtype=float)
        if bits.shape != (self.n_variables,):
            raise ValueError(
                f"Expected {self.n_variables} bits, received shape {bits.shape}."
            )
        if not np.isin(bits, (0.0, 1.0)).all():
            raise ValueError("QUBO assignments must be binary.")
        return float(self.offset + bits @ self.matrix @ bits)

    def to_ising(
        self,
    ) -> tuple[float, dict[int, float], dict[tuple[int, int], float]]:
        """Return ``(offset, h, J)`` under the convention ``x=(1-Z)/2``."""
        linear = self.linear
        quadratic = self.quadratic
        offset = self.offset + 0.5 * sum(linear.values())
        offset += 0.25 * sum(quadratic.values())
        h = {
            i: -0.5 * linear.get(i, 0.0)
            - 0.25
            * sum(value for pair, value in quadratic.items() if i in pair)
            for i in range(self.n_variables)
        }
        h = {i: value for i, value in h.items() if value != 0.0}
        couplings = {
            pair: 0.25 * value
            for pair, value in quadratic.items()
            if value != 0.0
        }
        return float(offset), h, couplings

    def pauli_terms(self, *, include_constant: bool = False) -> list[PauliTerm]:
        """Return the nonzero Pauli terms in the Ising Hamiltonian."""
        offset, h, couplings = self.to_ising()
        terms = []
        if include_constant and offset != 0.0:
            terms.append(PauliTerm(offset, "I", (0,)))
        terms.extend(PauliTerm(value, "Z", (i,)) for i, value in h.items())
        terms.extend(
            PauliTerm(value, "ZZ", pair) for pair, value in couplings.items()
        )
        return terms

    def hamiltonian(self):
        """Return the exact PennyLane cost Hamiltonian, including its offset."""
        terms = self.pauli_terms(include_constant=True)
        if not terms:
            return qml.Hamiltonian([0.0], [qml.Identity(0)])
        observables = []
        coefficients = []
        pauli_map = {"I": qml.Identity, "X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}
        for term in terms:
            factors = [pauli_map[p](wire) for p, wire in zip(term.pauli, term.wires)]
            observable = factors[0]
            for factor in factors[1:]:
                observable = observable @ factor
            coefficients.append(term.coefficient)
            observables.append(observable)
        return qml.Hamiltonian(coefficients, observables)

    def brute_force(self) -> tuple[str, float]:
        """Return a minimum-energy assignment and its objective value."""
        best_bits = None
        best_energy = np.inf
        for bits in product((0, 1), repeat=self.n_variables):
            energy = self.energy(bits)
            if energy < best_energy:
                best_bits, best_energy = bits, energy
        return "".join(map(str, best_bits)), float(best_energy)

    def __repr__(self) -> str:
        """Return a concise description containing the variable count and constant offset."""
        return f"QUBO(n_variables={self.n_variables}, offset={self.offset})"


__all__ = ["GraphSolution", "PauliTerm", "QUBO"]
