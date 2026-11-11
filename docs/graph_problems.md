# Graph optimization with QUBO

QASP provides **14 named graph formulations**, in addition to arbitrary matrix,
linear/quadratic-term and Ising constructors. Every factory returns the same
`QUBO` object accepted by QAOA, multi-angle QAOA, warm-start QAOA, recursive QAOA,
linear-ramp QAOA and PCE. No graph-library dependency is required.

These factories are in the pending source update, not yet in the PyPI release.
See [PR #15](https://github.com/INESCTEC/qasp/pull/15).

## Choose a formulation

Let N be the graph's vertex count, M its edge count, K a label count, and d_v
the degree of vertex v. B denotes the **total binary-variable count**.

| Factory | Original problem | B |
| --- | --- | --- |
| `QUBO.maxcut(N, edges)` | Maximum weighted two-way cut | N |
| `QUBO.maximum_independent_set(N, edges, weights=None)` | Maximum-weight nonadjacent vertex set | N |
| `QUBO.minimum_vertex_cover(N, edges, weights=None)` | Minimum-cost vertices covering all edges | N |
| `QUBO.maximum_clique(N, edges, weights=None)` | Maximum-weight complete vertex subset | N |
| `QUBO.maximum_matching(N, edges)` | Maximum-weight vertex-disjoint edges | M; one fixed-zero auxiliary bit if M=0 |
| `QUBO.minimum_dominating_set(N, edges, weights=None)` | Minimum-cost vertices dominating the graph | N + sum ceil(log2(d_v+1)) |
| `QUBO.minimum_edge_cover(N, edges)` | Minimum-cost edges incident to every vertex | M + sum ceil(log2(d_v)) |
| `QUBO.graph_coloring(N, edges, n_colors=K)` | Proper coloring with at most K colors | NK |
| `QUBO.max_k_cut(N, edges, n_parts=K)` | Maximum cut across at most K groups | NK |
| `QUBO.graph_partition(N, edges, subset_size=None)` | Minimum cut with a fixed-size side | N |
| `QUBO.densest_k_subgraph(N, edges, k)` | Maximum internal edge weight on exactly k vertices | N |
| `QUBO.hamiltonian_path(N, edges, directed=False)` | Visit each vertex once along a path | N² |
| `QUBO.hamiltonian_cycle(N, edges, directed=False)` | Visit each vertex once and return to the start | N² |
| `QUBO.traveling_salesperson(N, edges, directed=False)` | Minimum-cost Hamiltonian cycle | N² |

All factories except MaxCut accept `penalty=None`. `QUBO.formulations()` lists
the method names. Coloring is feasibility with a supplied color budget, not
chromatic-number minimization; partitioning constrains vertex count, not weight.
Matching is **maximum**, not merely maximal. Empty groups are allowed in Max-K-Cut.
Paths/cycles are not general vehicle-routing or subtour-relaxed TSP models.

## Inputs and parameter choices

- Vertices are integers `0..N-1`, with N positive. Relabel external graph objects
  before passing their edges. Self loops are rejected. Isolated vertices are
  represented by N even when absent from the edge list.
- `edges` is an iterable of unique undirected `(u, v)` pairs. Weighted MaxCut,
  matching, edge cover, Max-K-Cut, partitioning, densest-k and TSP also accept
  `(u, v, weight)` triples; pairs have weight one. These new weighted factories
  require finite **nonnegative** weights/costs, including zero. Vertex-set,
  coloring and Hamiltonian feasibility factories take pairs only.
- For backward compatibility, MaxCut accepts finite signed weights and sums
  repeated/reversed edges. Other factories reject duplicate edges, including
  reversed duplicates in undirected graphs. `directed=True` on a route factory
  treats `(u,v)` and `(v,u)` as different arcs.
- Vertex `weights` accepts a length-N sequence or a mapping whose omitted
  entries default to one. Values must be finite and nonnegative. `None` gives
  unit weights. They mean benefits in independent set/clique and costs in covers.
- `n_colors`/`n_parts` are positive integers. `subset_size` and `k` lie in [0,N].
  `subset_size=None` selects floor(N/2), also on odd graphs. Cycles and TSP require
  N>=3. A single vertex is a valid Hamiltonian path.
- Edge cover rejects isolated vertices because no cover exists. Coloring and
  route factories do not solve feasibility during construction: an impossible
  instance has no zero-penalty feasible assignment and its decoder reports this.

## Create, train and decode

```python
from qasp import Normal, QAOA, QUBO

problem = QUBO.maximum_independent_set(
    4, [(0, 1), (1, 2), (2, 3)], weights=[1, 2, 1, 2],
)
solver = QAOA(problem, n_layers=2, backend="torch",
              device="default.qubit", diff_method="backprop",
              init=Normal(0, 0.2), seed=7)
result = solver.solve(optimizer="Adam",
                      optimizer_options={"max_steps": 40, "lr": 0.05})
candidate = problem.decode(result.bitstring)
print(candidate.solution["vertices"])
print(candidate.feasible, candidate.objective, candidate.energy)
```

`result.objective` is the training expectation or PCE surrogate.
`result.energy` is the penalized QUBO energy of the returned bitstring.
`problem.decode(...)` returns a `GraphSolution`:

| Attribute | Meaning |
| --- | --- |
| `formulation`, `bitstring` | Problem name and validated candidate in wire order |
| `solution` | Dictionary of `vertices`, `edges`, `partition`, `colors`/`labels` and `groups`, or `path`/`tour` |
| `objective`, `sense` | Original score/cost and `max`, `min` or `feasibility`; not the penalized energy |
| `feasible`, `violations` | Whether graph constraints hold and explanations when they do not |
| `encoding_feasible`, `encoding_violations` | Additionally requires consistent auxiliary/slack bits |
| `energy` | The full QUBO energy, including penalties and offset |

Decoding never repairs a candidate. A vertex set can be a valid dominating set
while its auxiliary bits encode incorrect slack: `feasible=True` then differs
from `encoding_feasible=False`. Invalid one-hot rows are not silently replaced
by their argmax. A malformed route is `None`; a permutation using a missing edge
is retained with violations. When a meaningful original objective is undefined,
`objective=None`. A valid coloring or Hamiltonian feasibility solution scores zero.
An infeasible selected subset still has a well-defined weight, so always inspect
`feasible` alongside that value. Optimizer success does not imply feasibility.

`variable_labels` gives the precise binary layout. `n_decision_variables`
excludes auxiliary bits, whereas `n_variables` includes them. `metadata` returns
a copy of the original graph, encoding, objective direction and penalty settings.
Treat a factory's coefficient matrix as read-only: editing it in place does not
rewrite its graph metadata. Construct a new `QUBO(matrix, offset=...)` for a
modified custom polynomial; custom QUBOs intentionally have no graph decoder.

## Penalties and equations

All energies are minimized. Let A>0 be the penalty, a_i the vertex benefit/cost,
w_ij the edge weight/cost, and x_i=1 indicate a selected vertex. Off-diagonal
coefficients are stored once in an upper-triangular matrix so that
E(x)=offset+xᵀQx; binary squares reduce using x_i²=x_i.

The default chooses A=L+max(1,0.1L), strictly above a sufficient bound L. An
explicit finite positive A is allowed even below the bound, to support penalty
experiments. `metadata["penalty_is_sufficient"]` records A>L;
`metadata["sufficient_penalty_bound"]` records L. A false flag means the bound
does not certify the choice, not that it must fail on that particular graph.
The default ensures every global minimum is feasible **when a feasible solution
exists**, in exact arithmetic. It does not guarantee a variational algorithm
will find that minimum. Floating-point conditioning still matters: rescale very
large or tiny coefficients rather than using unnecessarily huge penalties.

### Cuts, subsets and partitioning

With c_ij(x)=x_i+x_j-2x_ix_j, the cut indicator,

\[
E_{\rm MaxCut}=-\sum_{(i,j)\in E}w_{ij}c_{ij}(x).
\]

There is no constraint penalty and the decoded objective reverses the sign.
Independent set, vertex cover and clique use

\[
E_{\rm IS}=-\sum_i a_i x_i+A\sum_{(i,j)\in E}x_ix_j,
\qquad
E_{\rm VC}=\sum_i a_i x_i+A\sum_{(i,j)\in E}(1-x_i)(1-x_j),
\]
\[
E_{\rm clique}=-\sum_i a_i x_i+A\sum_{i<j:(i,j)\notin E}x_ix_j.
\]

Here L=max_i a_i. Removing a conflicting selected vertex in IS/clique, or
adding an endpoint to cover an uncovered edge, removes at least one penalty
while sacrificing at most L in objective.

Fixed-size partition and densest-k subgraph use

\[
E_{\rm partition}=\sum_{(i,j)\in E}w_{ij}c_{ij}(x)
 +A\left(\sum_i x_i-k\right)^2,
\quad
E_{\rm dense}=-\sum_{(i,j)\in E}w_{ij}x_ix_j
 +A\left(\sum_i x_i-k\right)^2.
\]

L=sum_E w bounds the entire objective range. Every cardinality violation incurs
at least A, and choosing any k vertices is feasible.

### Edge selection and coverage slacks

Let y_e=1 select edge e and delta(v) contain edges incident to v. Matching uses

\[
E_{\rm matching}=-\sum_e w_e y_e
 +A\sum_v\sum_{\{e,f\}\subseteq\delta(v)}y_ey_f.
\]

L=max_e w_e: removing a conflicting edge loses at most L and removes at least
one penalty. With no edges, the empty matching is encoded by a single fixed-zero
auxiliary bit with cost A times that bit.

For dominating set, C_v consists of v and its neighbors. Introduce bounded
nonnegative integer slacks s_v to enforce coverage:

\[
E_{\rm dominating}=\sum_i a_i x_i
 +A\sum_v\left(\sum_{i\in C_v}x_i-1-s_v\right)^2,
\quad 0\le s_v\le d_v,
\]
\[
E_{\rm edgecover}=\sum_e w_e y_e
 +A\sum_v\left(\sum_{e\in\delta(v)}y_e-1-s_v\right)^2,
\quad 0\le s_v\le d_v-1.
\]

L=sum_i a_i or sum_e w_e respectively bounds the feasible all-selected solution.
All objective costs are nonnegative; a violated equality costs at least A>L.
For upper bound U, the slack uses m=ceil(log2(U+1)) bits with weights
1,2,...,2^(m-2),U-(2^(m-1)-1); U=0 needs no bits. This represents exactly
the integers 0..U, possibly with redundant encodings. Decision bits come first,
followed by slacks in vertex order. Edge decision bits use sorted canonical edges.

### Coloring and multiway cuts

Let x_vc=1 assign vertex v to label c, flattened as index vK+c. Define
C_onehot=sum_v(sum_c x_vc-1)². Then

\[
E_{\rm coloring}=A C_{\rm onehot}
 +A\sum_{(u,v)\in E}\sum_c x_{uc}x_{vc},
\]
\[
E_{\rm MaxKCut}=-\sum_{(u,v)\in E}w_{uv}
 +\sum_{(u,v)\in E}w_{uv}\sum_c x_{uc}x_{vc}
 +A C_{\rm onehot}.
\]

Coloring has L=0 and default A=1; zero energy is equivalent to a proper coloring.
For Max-K-Cut, L=sum_E w: malformed one-hot encodings have energy at least A-L>0,
whereas every valid encoding has energy at most zero.

### Paths, cycles and traveling salesperson

Let x_vt=1 put vertex v at position t, with index vN+t. Use

\[
C_{\rm permutation}=\sum_v\left(\sum_t x_{vt}-1\right)^2
 +\sum_t\left(\sum_v x_{vt}-1\right)^2,
\qquad
C_{\rm missing}=\sum_{t}\sum_{(u,v)\notin E_{\rm arc}}x_{ut}x_{v,t+1}.
\]

Path transitions are t=0..N-2. Cycles include t=N-1 with t+1 taken modulo N.
An undirected edge creates both allowed arcs; self-arcs are forbidden.
Hamiltonian path/cycle minimize A(C_permutation+C_missing), with default A=1.
TSP adds the allowed travel costs:

\[
E_{\rm TSP}=A(C_{\rm permutation}+C_{\rm missing})
 +\sum_{t=0}^{N-1}\sum_{(u,v)\in E_{\rm arc}}w_{uv}x_{ut}x_{v,t+1}.
\]

L=N max_E w bounds any existing feasible tour. Missing edges are penalized,
never treated as zero-cost travel. The full N² layout retains rotational tour
symmetry; no start-vertex symmetry reduction is applied.

## Simulation, validation and examples

QAOA uses **B qubits**, not necessarily N. The dense QUBO matrix occupies O(B²)
memory: N²-variable routing models already require O(N⁴) matrix memory.
Statevector simulation scales exponentially in B. `brute_force()` also costs
2^B candidate evaluations and should be restricted to small reference instances.
Slack encodings use logarithmic bits per constraint but can still increase
quantum resources substantially. Check `problem.n_variables` before choosing a
simulation size. Graph construction is classical and independent of simulator.

PCE accepts the same QUBO, including slack bits, and chooses capacity for B
binary variables. Its continuous surrogate does not automatically preserve
hard graph constraints or ensure an optimum, and the general-QUBO field
extension should not be interpreted as a MaxCut-specific theoretical guarantee.
Always decode and check the returned candidate.

Use statevector or appropriate Pauli propagation execution for general QAOA.
An IQP device is only compatible when the **actual variational circuit** belongs
to its supported commuting family; arbitrary QAOA circuits do not. The new
factories do not remove existing device, gradient or measurement restrictions.

- [Executable catalog](../examples/graph_optimization_problems.py): all 14 factories,
  small exact references, and a Torch QAOA example.
- [Graph-problems notebook](../tutorials/optimization/graph_problems.ipynb): configurations,
  training, original-objective decoding, constraint checks, result plots and
  both Matplotlib and inline TikZ drawing.
- [Tests](../tests/test_graph_formulations.py): independent combinatorial optima,
  all small binary energies versus Ising, weighted/directed/invalid inputs,
  QAOA/PCE training and cross-simulator gradient checks.

## References

The implementations use standard binary penalty reductions, with explicit
weight bounds, variable layouts and decoding contracts as specified above.
See Andrew Lucas, *Ising formulations of many NP problems*, Frontiers in Physics
2, 5 (2014), [doi:10.3389/fphy.2014.00005](https://doi.org/10.3389/fphy.2014.00005),
for graph reductions and vertex-position route encodings; and Fred Glover,
Gary Kochenberger and Yu Du, *A Tutorial on Formulating and Using QUBO Models*
(2018), [arXiv:1811.11538](https://arxiv.org/abs/1811.11538), for squared-penalty
and auxiliary-variable construction.
