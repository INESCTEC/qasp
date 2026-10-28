"""Render the reinforcement-learning Quantikz gallery used in the QASP paper.

Run this file from a QASP checkout so that it imports the package under review::

    python qasp_paper/render_rl_circuits.py

Every panel is produced through QASP's public ``create_model`` and
``draw_model(output="tikz")`` functions.  The editable Quantikz source is
retained, PDF files are used by LaTeX, and PNG files support visual review.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from qasp import create_model, draw_model


SEED = 17


def ansatz_config(*, layers: int, reuploading: bool = False) -> dict:
    return {
        "n_layers": layers,
        "rotations": "ZY",
        "reuploading": reuploading,
        "include_imprimitive": True,
        "imprimitive": "CZ",
        "imprimitive_pattern": "ring",
        "include_final_imprimitive": False,
        "include_feature_map_first": True,
        "include_hadamards_initial": True,
    }


def softmax_policy():
    n_qubits = 4
    config = {
        "ansatz": "AngleAnsatz",
        "ansatz_config": ansatz_config(layers=2),
        "feature_map": "NeuralEmbedding",
        "feature_map_config": {"features": 4, "rotations": "Z"},
        "n_actions": 2,
        "observables": ["Z 0", "Z 1", "ZZ 0 1", "ZZ 2 3"],
        "fc_layer": True,
        "init": "small_normal",
        "device": "default.qubit",
        "diff_method": "backprop",
    }
    return create_model("PQCSoftmax", config, n_qubits=n_qubits), torch.zeros(4)


def born_parity_policy():
    n_qubits = 4
    config = {
        "ansatz": "AngleAnsatz",
        "ansatz_config": ansatz_config(layers=2),
        "feature_map": "AngleEmbedding",
        "feature_map_config": {
            "features": n_qubits,
            "rotations": "Y",
            "input_scaling": True,
        },
        "n_actions": 2,
        "policy": "parity-like",
        "training_method": "probs",
        "parity_cnot_decomposition": False,
        "init": "small_normal",
        "device": "default.qubit",
        "diff_method": "backprop",
    }
    return create_model("PQCBorn", config, n_qubits=n_qubits), torch.zeros(4)


def ancilla_parity_policy():
    n_qubits = 4
    config = {
        "ansatz": "AngleAnsatz",
        "ansatz_config": ansatz_config(layers=1, reuploading=False),
        "feature_map": "AngleEmbedding",
        "feature_map_config": {
            "features": n_qubits,
            "rotations": "Y",
            "input_scaling": False,
        },
        "n_actions": 2,
        "policy": "parity-like",
        "measurement_qubits": list(range(n_qubits)),
        "training_method": "expectation",
        "parity_cnot_decomposition": True,
        "init": "small_normal",
        "device": "default.qubit",
        "diff_method": "backprop",
    }
    return create_model("PQCBorn", config, n_qubits=n_qubits), torch.zeros(4)


def majority_modulo_policy():
    n_qubits = 6
    config = {
        "ansatz": "AngleAnsatz",
        "ansatz_config": ansatz_config(layers=1, reuploading=False),
        "feature_map": "AngleEmbedding",
        "feature_map_config": {
            "features": n_qubits,
            "rotations": "Y",
            "input_scaling": False,
        },
        "n_actions": 3,
        "policy": "majority-modulo",
        "measurement_qubits": list(range(n_qubits)),
        "majority_blocks": [[0, 1, 2], [3, 4, 5]],
        "strict_majority_blocks": True,
        "training_method": "probs",
        "init": "small_normal",
        "device": "default.qubit",
        "diff_method": "backprop",
    }
    return create_model("PQCBorn", config, n_qubits=n_qubits), torch.zeros(6)


POLICIES = {
    "rl_softmax_observable": softmax_policy,
    "rl_born_parity": born_parity_policy,
    "rl_ancilla_parity": ancilla_parity_policy,
    "rl_majority_modulo": majority_modulo_policy,
}


def compile_quantikz(tex_path: Path, output_dir: Path) -> None:
    """Compile standalone Quantikz and retain vector PDF plus PNG preview."""
    pdflatex = shutil.which("pdflatex")
    pdftoppm = shutil.which("pdftoppm")
    if pdflatex is None or pdftoppm is None:
        raise RuntimeError("Rendering needs pdflatex and pdftoppm on PATH.")

    with tempfile.TemporaryDirectory(prefix="qasp-quantikz-") as scratch:
        scratch_dir = Path(scratch)
        scratch_tex = scratch_dir / tex_path.name
        shutil.copy2(tex_path, scratch_tex)
        subprocess.run(
            [
                pdflatex,
                "-interaction=nonstopmode",
                "-halt-on-error",
                f"-output-directory={scratch_dir}",
                str(scratch_tex),
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        shutil.copy2(scratch_tex.with_suffix(".pdf"), output_dir / tex_path.with_suffix(".pdf").name)

    pdf_path = output_dir / tex_path.with_suffix(".pdf").name
    subprocess.run(
        [
            pdftoppm,
            "-singlefile",
            "-png",
            "-r",
            "220",
            str(pdf_path),
            str(output_dir / tex_path.stem),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def render(name: str, factory, output_dir: Path) -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model, observation = factory()
    output_dir.mkdir(parents=True, exist_ok=True)
    wire_labels = None
    if name == "rl_ancilla_parity":
        wire_labels = {
            **{wire: rf"q_{{{wire}}}" for wire in range(model.n_qubits)},
            model.n_qubits: "a_0",
        }
    tikz = draw_model(
        model=model,
        inputs=observation,
        output="tikz",
        tikz_options={
            "standalone": True,
            "symbolic_parameters": True,
            "compress": name in {"rl_softmax_observable", "rl_born_parity"},
            "max_explicit_layers": 3,
            "wire_labels": wire_labels,
            "row_sep": "0.38cm",
            "column_sep": "0.27cm",
        },
    )
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
    for name, factory in POLICIES.items():
        render(name, factory, args.output_dir)
        print(f"rendered {name}")


if __name__ == "__main__":
    main()
