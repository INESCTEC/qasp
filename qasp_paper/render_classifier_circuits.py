"""Render the parameterized-classifier Quantikz gallery for the QASP paper.

Run from a QASP checkout so every panel is traced from the package version
being documented::

    python qasp_paper/render_classifier_circuits.py

The script calls the public ``create_classifier`` and
``draw_classifier(output="tikz")`` APIs, retaining editable Quantikz source,
vector PDF, and PNG previews.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from qasp import create_classifier, draw_classifier

from render_rl_circuits import compile_quantikz


SEED = 31
N_QUBITS = 4
INPUT = np.array([0.35, -0.20, 0.60, -0.45])


def base_config() -> dict:
    return {
        "feature_map": "AngleEmbedding",
        "feature_map_config": {
            "rotations": "Y",
            "input_scaling": False,
        },
        "ansatz": "AngleAnsatz",
        "ansatz_config": {
            "n_layers": 2,
            "rotations": "ZY",
            "reuploading": False,
            "include_imprimitive": True,
            "imprimitive": "CZ",
            "imprimitive_pattern": "ring",
            "include_final_imprimitive": False,
            "include_feature_map_first": True,
            "include_hadamards_initial": True,
        },
        "init": "small_normal",
        "device": "default.qubit",
        "diff_method": "backprop",
    }


def classifier_specs() -> dict[str, tuple[dict, int]]:
    expectation = base_config()
    expectation.update(
        {
            "readout": "expectation",
            "observable": ["Z 0", "X 1", "Z 2"],
        }
    )
    born = base_config()
    born.update(
        {
            "readout": "probabilities",
            "decoder": "parity",
            "measurement_qubits": [0, 1, 2, 3],
        }
    )
    return {
        "classifier_expectation": (expectation, 2),
        "classifier_born_parity": (born, 3),
    }


def render(name: str, config: dict, n_classes: int, output_dir: Path) -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    classifier = create_classifier(
        config, n_qubits=N_QUBITS, n_classes=n_classes
    )
    tikz = draw_classifier(
        classifier,
        inputs=INPUT,
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
    for name, (config, n_classes) in classifier_specs().items():
        render(name, config, n_classes, args.output_dir)
        print(f"rendered {name}")


if __name__ == "__main__":
    main()
