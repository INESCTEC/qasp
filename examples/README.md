# QASP examples

This directory keeps small tutorials and their supporting scripts. Notebook
outputs are cleared in Git; run the cells to generate figures and results.
Saved output directories are ignored. Install QASP from the repository root
with `python -m pip install -e '.[dev]'` before opening a notebook.

| Topic | Start here |
| --- | --- |
| Fourier spectra across qubit counts and observables | [expressivity_sweep.py](expressivity_sweep.py) |
| Shared Pauli simulator across all five application modules | [pauli_propagation_modules.py](pauli_propagation_modules.py) |
| IQP TN/VN, neural embedding, gradients and auto selection | [iqp_simulator.ipynb](iqp_simulator.ipynb) |
| State initialization across models | [initialization_across_modules.ipynb](initialization_across_modules.ipynb) |
| Classifiers and sklearn validation | [quantum_classifiers_sklearn_validation.ipynb](quantum_classifiers_sklearn_validation.ipynb) |
| Fidelity and projected kernels | [iqp_quantum_kernels_validation.ipynb](iqp_quantum_kernels_validation.ipynb), [script](iqp_quantum_kernels.py) |
| Generative learning | [pqcbm_generative_learning.ipynb](pqcbm_generative_learning.ipynb), [script](pqcbm_generative_learning.py) |
| Policy-gradient RL and diagnostics | [cartpole_policies_and_variance.ipynb](cartpole_policies_and_variance.ipynb) |
| Q-functions, readouts and short DQN runs | [pqdqn_observables_validation.ipynb](pqdqn_observables_validation.ipynb) |
| QAOA variants and PCE | [optimization_algorithms.ipynb](optimization_algorithms.ipynb) |
| PCE versus QAOA, with simulator gradient checks | [pce_vs_qaoa.ipynb](../tutorials/optimization/pce_vs_qaoa.ipynb) |
| Circuit drawing | [drawing_quantum_circuits.ipynb](drawing_quantum_circuits.ipynb) |

Run a cross-module gradient check in the configured development environment:

```sh
conda run -n torchlane python examples/pauli_propagation_modules.py
```

`iqp_tn_simulator.py` demonstrates the exact-only compatibility device.
`pqdqn_cartpole_experiment.py` and `pqdqn_paper_figures.py` are supporting
helpers for the Q-function notebook and its regression test.
`render_readme_circuits.py` rebuilds the small documentation circuit gallery.
In Jupyter, module calls with `output="tikz", return_tex=False` display a figure. Omit `return_tex` and assign the result to keep the LaTeX as well. For a builder, import `show_tikz` from QASP and call `show_tikz(builder.to_tikz(...))`. Inline rendering uses pdfLaTeX, Quantikz, standalone and Ghostscript.

Drawing compilation requires a local LaTeX installation with Quantikz; numerical
examples and regression tests do not require compiling the gallery.

Notebook workloads vary: classifier dataset comparisons and RL training take
longer than the small simulator example. The generative notebook creates its
own parameter/results files before reading them. Historical research notebooks,
full timing sweeps, stored checkpoints and papers are available in the
[upstream archive](https://github.com/andre-sequeira10/QASP/tree/3bda8622bd6f5ddfb7c05a0dfd0538c1b0ccd718).
