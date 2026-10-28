"""Render the quantum-kernel Quantikz gallery used in the QASP paper.

Run from a QASP checkout so the figures are traced from the package version
being documented::

    python qasp_paper/render_kernel_circuits.py

The script calls the public ``create_kernel`` and
``draw_kernel(output="tikz")`` APIs and retains editable source, vector PDF,
and PNG previews.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from qasp import create_kernel, draw_kernel

from render_rl_circuits import compile_quantikz


SEED = 23
N_QUBITS = 3
X = np.array([0.10, 0.20, 0.15])
Y = np.array([0.90, 1.10, 1.00])


def base_config(kernel_type: str) -> dict:
    return {
        "kernel_type": kernel_type,
        "feature_map": "AngleEmbedding",
        "feature_map_config": {
            "rotations": "Y",
            "imprimitive_pattern": "ring",
        },
        "ansatz": "IQPAnsatz",
        "ansatz_config": {
            "n_layers": 2,
            "imprimitive_pattern": "ring",
            "reuploading": False,
        },
        "init": "small_normal",
        "device": "default.qubit",
    }


def kernel_specs() -> dict[str, tuple[dict, np.ndarray | None]]:
    localized = base_config("fidelity")
    localized["projectors"] = ["P 0 1"]

    projected = base_config("projected")
    projected.update(
        {
            "observables": ["Z 0", "Z 1", "Z 2"],
            "gamma": 0.5,
        }
    )
    return {
        "kernel_fidelity_global": (base_config("fidelity"), Y),
        "kernel_fidelity_localized": (localized, Y),
        "kernel_projected": (projected, None),
    }


def render(name: str, config: dict, y: np.ndarray | None, output_dir: Path) -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    kernel = create_kernel(config, n_qubits=N_QUBITS)
    tikz = draw_kernel(
        kernel=kernel,
        x=X,
        y=y,
        output="tikz",
        tikz_options={
            "standalone": True,
            "symbolic_parameters": True,
            "compress": True,
            "max_explicit_layers": 3,
            "row_sep": "0.38cm",
            "column_sep": "0.27cm",
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
    for name, (config, y) in kernel_specs().items():
        render(name, config, y, args.output_dir)
        print(f"rendered {name}")


if __name__ == "__main__":
    main()
