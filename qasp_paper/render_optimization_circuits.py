"""Render the optimization-circuit gallery used in the QASP paper.

Run this file from a QASP checkout so the figures are traced from the package
version being documented::

    python qasp_paper/render_optimization_circuits.py

The script uses QASP's public optimization constructors and
``draw_optimization(output="tikz")`` entry point.  It retains editable
Quantikz source, vector PDF, and PNG previews.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from qasp import PCE, QAOA, QUBO, draw_optimization

from render_rl_circuits import compile_quantikz


SEED = 43


def triangle_maxcut() -> QUBO:
    return QUBO.maxcut(3, [(0, 1), (1, 2), (0, 2)])


def optimization_models() -> dict[str, object]:
    problem = triangle_maxcut()
    standard = QAOA(problem, n_layers=1, seed=SEED)
    pce = PCE(
        problem,
        circuit={
            "ansatz": "AngleAnsatz",
            "ansatz_config": {
                "n_layers": 1,
                "rotations": ["Y 0", "Y 1", "ZZ 0 1"],
                "include_hadamards_initial": True,
            },
            "device": "default.qubit",
        },
        seed=SEED,
    )
    return {
        "optimization_qaoa": standard,
        "optimization_pce": pce,
    }


def render(name: str, model, output_dir: Path) -> None:
    parameters = model.initial_parameters("zeros")
    tikz = draw_optimization(
        model,
        parameters=parameters,
        output="tikz",
        tikz_options={
            "standalone": True,
            "symbolic_parameters": True,
            "row_sep": "0.40cm",
            "column_sep": "0.29cm",
        },
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    tex_path = output_dir / f"{name}.tex"
    tex_path.write_text(tikz, encoding="utf-8")
    compile_quantikz(tex_path, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figures",
    )
    args = parser.parse_args()
    np.random.seed(SEED)
    for name, model in optimization_models().items():
        render(name, model, args.output_dir)
        print(f"rendered {name}")


if __name__ == "__main__":
    main()
