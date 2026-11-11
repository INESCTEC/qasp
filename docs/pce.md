# Pauli Correlation Encoding

PCE assigns binary variables to signs of same-axis Pauli correlations on a
compressed register. It follows the encoding, MaxCut loss and final classical
improvement step of [Sciorilli et al., *Towards large-scale quantum optimization
solvers with few qubits*, Nature Communications 16, 476 (2025)](https://doi.org/10.1038/s41467-024-55346-z).
The ansatz and classical optimizer are configurable; choosing another circuit
does not reproduce the paper's particular benchmark setup.

For (N) variables, order (r), and axis set (A), QASP chooses the smallest
(n) with (|A|\binom{n}{r}\geq N), unless `n_qubits` is specified. Correlations
are assigned in axis order, then lexicographic wire combinations. The defaults
are `correlation_order=2` and `paulis="XYZ"`.

## Objective and gradients

Let (c_i=\langle P_i\rangle), (s_i=\tanh(\alpha c_i)), and
(R=(N^{-1}\sum_i s_i^2)^2). For `QUBO.maxcut`, each undirected edge counted
once, the loss is exactly

\[
L=\sum_{(i,j)\in E}w_{ij}s_i s_j+\beta\nu R.
\]

`regularization` is (\beta), default **0.5**. For a connected graph with positive
weights, (\nu=w(G)/2+w(T_{\min})/4). Unweighted graphs therefore use
(\nu=|E|/2+(N-1)/4). Disconnected graphs use the minimum spanning forest,
equivalently summing this scale over connected components; isolated vertices
contribute zero. `regularization_scale` optionally overrides (\nu), including
zero. The default `alpha` is (n^{\lfloor r/2\rfloor}); it can be tuned explicitly.
All three numerical hyperparameters must be finite, with positive alpha and
nonnegative regularization and scale.

For a general QUBO with Ising energy (E(s)=c+\sum h_i s_i+\sum J_{ij}s_i s_j),
the data term is (2(E(s)-c)). The default scale uses the positive part of
weights (2J_{ij}) and adds (\sum_i|h_i|). This preserves scaling under a
positive rescaling of the QUBO. The positive-part convention for signed edges
follows the [authors' reference implementation](https://github.com/MarcoSciorilli/PCE/blob/0.1/src/loss.py);
the field term is a QASP extension, not a MaxCut bound for arbitrary QUBOs.
Use `regularization_scale` when a problem-specific scale is appropriate.

`loss_tensor(parameters)` differentiates the entire nonlinear function of
correlators. In particular, parameter-shift differentiates the underlying
expectations and applies the classical chain rule; shifting the already
nonlinear loss as if it were one expectation would be incorrect.

## Construct, train and inspect

```python
from qasp import Normal, PCE, QUBO, draw_pce

problem = QUBO.maxcut(3, [(0, 1, 1.0), (1, 2, 2.0), (0, 2, 3.0)])
config = {
    "correlation_order": 2,
    "regularization": 0.5,
    "init": Normal(0, 0.3),
    "backend": "torch",
    "device": "default.qubit",
    "diff_method": "backprop",
    "seed": 7,
}
model = PCE(problem, **config)
result = model.solve(
    optimizer="Adam",
    optimizer_options={"max_steps": 80, "lr": 0.03},
)
print("Encoding:", model.pauli_labels)
print("Regularizer scale:", model.regularization_scale)
print("Raw candidate:", result.metadata["raw_bitstring"], result.metadata["raw_energy"])
print("Improved candidate:", result.bitstring, result.energy)
print("Training loss:", result.objective)

fig, ax = draw_pce(model, parameters=result, output="matplotlib")
draw_pce(model, parameters=result, output="tikz", return_tex=False)
```

`solve(postprocess=True)` makes **one sequential sweep**, visiting variables in
index order and retaining only strictly improving flips. This is outside
autodiff and is not repeated until convergence. It takes (O(N+|E|)) operations
after adjacency construction. `result.energy` is the **original QUBO energy**
of the improved candidate; for MaxCut its negative is the cut weight.
`result.objective` remains the optimized PCE loss, not a cut value.

`decode(parameters)` always returns the **raw** candidate, with exact zero
correlations mapped to bit zero. `solve(postprocess=False)` returns this raw
candidate as the final result. `improve_bitstring(bits)` exposes the same sweep
for any candidate of this QUBO, including a QAOA result. Metadata always retains
the raw candidate, its energy, the readout correlations, and whether the sweep
was requested. With finite shots, the raw candidate and recorded correlations
come from the same readout.

## Simulator choices

| Simulator | PCE circuit | Differentiation |
| --- | --- | --- |
| `default.qubit` | General supported ansatz | `backprop` or `adjoint` |
| `lightning.qubit` | General supported ansatz | `adjoint` |
| `PauliPropagationSimulator` | Supported noncommuting gates | Torch `backprop`; `{"k": None, "prune": False}` for exact checks |
| `default.tensor`, method `mps` or `tn` | General supported ansatz | Explicit `parameter-shift`; requires Quimb |
| `IQPSimulator`, `tn` / `van_den_nest` / `auto` | Compatible commuting circuit and readout | Torch `backprop` |

PCE's default Y/Z/ZZ circuit is noncommuting and is not a VN circuit. For IQP
execution, supply commuting X/XX/XXX rotations and `paulis="Z"`, or another
supported IQP construction. Restricting the axis set changes the encoding
capacity and potentially the register size. Backend equivalence tests must
compare **the same circuit, encoding and parameters**.

VN enumeration gives an exact small-system reference. Sampled VN differentiates
the empirical nonlinear loss; this need not be an unbiased estimate of the
exact nonlinear loss or its gradient. Use a fixed sample bank for finite
difference checks. Finite-shot measurements have the analogous issue.
Approximate MPS truncation and Pauli pruning also change values and gradients.

In a TikZ diagram, slash-separated meter labels such as X/Y/Z denote alternative
readout settings on separate circuit preparations, not simultaneous measurements
in incompatible bases. `pauli_labels` lists the actual correlation observables.

For double-precision Torch/Lightning comparisons, set
`torch.set_default_dtype(torch.float64)` before constructing the models, in
addition to supplying float64 parameters. PennyLane's conversion of scalar
Lightning results uses Torch's default dtype.

## PCE versus QAOA

Run [the comparison tutorial](../tutorials/optimization/pce_vs_qaoa.ipynb) for
unweighted and weighted MaxCut graphs, exact small-instance optima, several
initializations, raw and improved cuts, training curves, timings and circuit
drawings. Both methods receive the same optimizer-step budget and the same
optional classical sweep. Their ansatz sizes, output rules and losses differ;
these small experiments are not evidence of a general performance advantage.
