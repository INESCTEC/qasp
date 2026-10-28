# QASP software paper

This folder contains a publication-oriented LaTeX preprint for QASP v0.1.0.

## Files

- `main.tex` — complete manuscript, including architecture, RL, kernel, parameterized-classifier, generative-learning, and combinatorial-optimization sections; capability tables; equations; code listings; limitations; and appendices.
- `references.bib` — bibliography for the software, QRL, gradient, and Pauli-propagation background.
- `render_rl_circuits.py` — builds four RL policies and exports their actual circuits through QASP's `draw_model(output="tikz")` path.
- `render_kernel_circuits.py` — builds the unified fidelity and projected kernels and exports their evaluated circuits through `draw_kernel(output="tikz")`.
- `render_classifier_circuits.py` — builds expectation- and Born-readout classifiers and exports their evaluated circuits through `draw_classifier(output="tikz")`.
- `render_optimization_circuits.py` — builds QAOA and Pauli-correlation-encoding solvers and exports their circuits through `draw_optimization(output="tikz")`.
- `render_pqdqn_results.py` — regenerates the vector held-out PQDQN comparison from the completed seed-level CartPole records without retraining.
- `figures/` — the INESC TEC cover logo, editable standalone Quantikz sources, vector PDFs used by LaTeX, and PNG previews for GitHub.
- `output/pdf/qasp_software_paper.pdf` — compiled publication-ready manuscript.

The draft was updated by auditing the local QASP checkout on branch
`codex/pqdqn-qfunctions`, including its uncommitted working-tree
changes. Recheck every API example against the clean release tag before
submission.

## Build

With a standard TeX Live or MacTeX installation:

```sh
latexmk -pdf main.tex
```

Alternatively:

```sh
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

The manuscript uses common LaTeX packages plus TikZ. It does not require shell escape or minted. Regenerating the standalone circuit assets additionally requires the `quantikz` package.

## Regenerate the circuit galleries

From the root of a QASP checkout with QASP's dependencies installed:

```sh
python qasp_paper/render_rl_circuits.py
python qasp_paper/render_kernel_circuits.py
python qasp_paper/render_classifier_circuits.py
python qasp_paper/render_optimization_circuits.py
python qasp_paper/render_pqdqn_results.py
```

The renderers call the public QASP factories and drawing entry points for
observable-softmax, Born parity, ancilla parity, majority–modulo, unified
fidelity, projected-kernel, parameterized-classifier, QAOA, and PCE circuits. QASP emits semantic gate colors,
symbolic parameter labels, optional repeated-layer compression, and the
editable Quantikz source; the scripts compile that source to vector PDF and
PNG without redrawing or altering the executed circuit topology.

## Submission blockers

Search `main.tex` for `TODO`. Before submission:

1. Add every author, affiliation, e-mail, ORCID, funding source, and contributor.
2. Tag the release and add an archival DOI (for example, Zenodo).
3. Confirm the public repository URL and update `pyproject.toml` in the QASP repository, whose current project URLs are placeholders.
4. Run all examples and tests against the tagged release in clean supported Python environments.
5. Archive the completed PQDQN manifest and raw records with the release, and add further benchmark results only after running the corresponding protocol in the validation section. The draft makes no unsupported speedup or quantum-advantage claim.
6. Review the AI-use disclosure and adapt it to the selected venue.

## Venue adaptation

The current document is a full software preprint suitable as a detailed companion to the GitHub release. For a submission to the Journal of Open Source Software, shorten it substantially and migrate the content into the JOSS template. For a full-length quantum-computing journal, retain the architecture and method sections and add tagged-release performance/accuracy figures.
