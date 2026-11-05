# Lightweight repository layout

This repository contains the QASP library synced from upstream revision
`3bda8622bd6f5ddfb7c05a0dfd0538c1b0ccd718`, its INESC TEC license, regression tests,
usage documentation and a curated set of examples. The implementation under
`qasp/` is unchanged by this cleanup.

## Migration to INESC TEC GitHub

The GitHub repository was initially imported from an older GitLab snapshot.
This update synchronizes the committed GitLab distribution at
`25e1e7ee9fbece7d0e8e25fc050d166073d7634c`, preserving the GitHub import commit
and the INESC TEC license. Local uncommitted notebook edits are excluded.
Repository links and the README contents list are adapted for GitHub; library
files match the committed GitLab distribution. The validation record below
describes the earlier GitLab cleanup.

## Retained material

- All library modules, including IQPSimulator and PauliPropagationSimulator.
- The package metadata and installation requirements.
- All runtime/API regression tests and the small experiment-runner regression.
- Cluster execution scripts and MPI smoke checks under `tests/tests_deucalion/`.
- Nine tutorials with cleared outputs and seven supporting Python examples.
- Essential API guides, logos and the small circuit-drawing gallery used by the README.

For guided, executable lessons by module, start with the [tutorial index](../tutorials/README.md). Generated tutorial figures stay in ignored local directories.

For existing examples, start with the [example index](../examples/README.md). The optimization tutorial
now lives under `examples/` rather than `tests/`.

## Research archive

The following are omitted from this distribution:

- `qasp_paper/` and [upstream docs/iqp_tn_paper/](https://github.com/andre-sequeira10/QASP/blob/3bda8622bd6f5ddfb7c05a0dfd0538c1b0ccd718/docs/iqp_tn_paper/), including compiled papers.
- Saved benchmark datasets, checkpoints, experiment runs and copied source snapshots.
- Repeated research notebooks, benchmark/report generators and generated previews.
- The two benchmark-reporting test files that specifically validate archived
  reporting scripts and saved timing datasets. Simulator correctness and worker
  gradient tests remain in the runtime regression suite.
- Bytecode, editor metadata, package-build metadata and temporary files.

The complete source material remains accessible in the
[pinned upstream research archive](https://github.com/andre-sequeira10/QASP/tree/3bda8622bd6f5ddfb7c05a0dfd0538c1b0ccd718).
Links to omitted files in retained guides point there. Research reproduction
commands in those guides are intended for that upstream checkout. They are
separate from the local example and regression-test commands.

## Keeping the repository small

Store new experiment output in ignored [upstream examples/results/](https://github.com/andre-sequeira10/QASP/blob/3bda8622bd6f5ddfb7c05a0dfd0538c1b0ccd718/examples/results/), [upstream tests/results/](https://github.com/andre-sequeira10/QASP/blob/3bda8622bd6f5ddfb7c05a0dfd0538c1b0ccd718/tests/results/),
[upstream docs/benchmarks/](https://github.com/andre-sequeira10/QASP/blob/3bda8622bd6f5ddfb7c05a0dfd0538c1b0ccd718/docs/benchmarks/), `output/` or a directory outside the repository. Keep essential
small test fixtures with their tests, rather than depending on saved experiments.
Before committing a notebook, clear its cell outputs and execution counts, and
remove saved widget state. Store papers and large datasets in the research
archive or an appropriate external artifact store, with links from documentation.

Run the local regression suite from the repository root:

```sh
conda run -n torchlane python -m pytest -q tests
```

This cleanup reduces the current checkout and future source snapshots. Existing
Git history still contains earlier committed artifacts. A full clone can therefore
remain substantially larger than the current files. To download only the latest
snapshot and its recent history, use a shallow clone:

```sh
git clone --depth 1 https://github.com/INESCTEC/qasp.git
```

No history rewrite or force push is part of this cleanup.

## Validation of this cleanup

The [validation record](repository_validation.json) records 746 passing tests in
`torchlane`, executed simulator/kernel examples, notebook checks and an isolated
wheel build/import check. The remaining long tutorials and cluster jobs were not
all executed during this cleanup.
