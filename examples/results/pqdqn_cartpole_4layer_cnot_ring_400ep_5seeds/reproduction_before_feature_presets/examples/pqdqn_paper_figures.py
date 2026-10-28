"""Render QASP's Quantikz exports as vector PDFs and notebook previews.

Uses the same exporter, palette, and LaTeX workflow as
``drawing_quantum_circuits.ipynb``. Requires pdflatex with Quantikz and either
pdftoppm (optionally set QASP_PDFTOPPM) or macOS sips.
"""

import os
import shutil
import subprocess
from pathlib import Path

import torch

from qasp.circuits import circuit_to_tikz
from qasp.rl import draw_qfunction
from qasp.rl.qfunctions import PQDQN

PASTEL_COLORS = {
    "single": "DCECF8",
    "two": "E4D8F2",
    "multi": "EEE2F7",
    "measurement": "E8E5F5",
}
FULL_TIKZ = {
    "standalone": True,
    "symbolic_parameters": True,
    "gate_colors": PASTEL_COLORS,
}
COMPRESSED_TIKZ = {**FULL_TIKZ, "compress": True, "max_explicit_layers": 4}
_RENDERED_TIKZ = {}


def render_tikz(source, tex_path):
    """Compile actual TikZ, then rasterize the PDF for inline display."""
    tex_path = Path(tex_path).resolve()
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    tex_path.write_text(source, encoding="utf-8")
    pdf, png = tex_path.with_suffix(".pdf"), tex_path.with_suffix(".png")
    paths = {"tikz": str(tex_path), "pdf": str(pdf), "preview": str(png)}
    cached = _RENDERED_TIKZ.get(source)
    if cached and all(Path(cached[key]).exists() for key in ("pdf", "preview")):
        # Symbolic figures can match across seeds. Reuse only an exact source
        # match compiled successfully in this process; still save each figure.
        for key in ("pdf", "preview"):
            if Path(cached[key]) != Path(paths[key]):
                shutil.copyfile(cached[key], paths[key])
        return paths
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        raise RuntimeError(
            f"Saved {tex_path}; install pdflatex and Quantikz to render it."
        )
    result = subprocess.run(
        [
            pdflatex,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-no-shell-escape",
            tex_path.name,
        ],
        cwd=tex_path.parent,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(f"LaTeX failed for {tex_path}:\n{result.stdout[-4000:]}")
    pdftoppm = os.environ.get("QASP_PDFTOPPM") or shutil.which("pdftoppm")
    if pdftoppm:
        command = [
            pdftoppm,
            "-png",
            "-singlefile",
            "-r",
            "200",
            str(pdf),
            str(tex_path.with_suffix("")),
        ]
    elif shutil.which("sips"):
        command = [
            shutil.which("sips"),
            "-s",
            "format",
            "png",
            "-Z",
            "2000",
            str(pdf),
            "--out",
            str(png),
        ]
    else:
        raise RuntimeError(
            f"Saved {pdf}; install pdftoppm to create the notebook preview."
        )
    subprocess.run(command, check=True, capture_output=True, text=True)
    # Keep final artifacts tidy; compilation diagnostics remain on failure.
    for extension in (".aux", ".log"):
        tex_path.with_suffix(extension).unlink(missing_ok=True)
    _RENDERED_TIKZ[source] = paths
    return paths


def save_circuit_drawing(model, output_dir, name="circuit"):
    """Save compact and fully expanded Quantikz figures of the existing model.

    Compact blocks include encoding, rotations, and entanglers, with the first
    complete layer expanded. The PDFs are vector originals; PNGs are previews
    rendered from those PDFs. The classical readout is recorded separately.
    """

    def export(options):
        if isinstance(model, PQDQN):
            return draw_qfunction(model, output="tikz", tikz_options=options)
        # Born probability references used in the validation notebook.
        state = model.qlayer.state_dict()
        features = model.circuit_builder.ansatz.feature_map.features
        kwargs = {
            "inputs": torch.zeros(features, dtype=state["params"].dtype),
            "params": state["params"],
        }
        if "input_scaling_params" in state:
            kwargs["input_scaling_params"] = state["input_scaling_params"]
        return circuit_to_tikz(model.circuit_builder, circuit_kwargs=kwargs, **options)

    output_dir = Path(output_dir)
    paths = render_tikz(export(COMPRESSED_TIKZ), output_dir / f"{name}.tex")
    full = render_tikz(export(FULL_TIKZ), output_dir / f"{name}_full.tex")
    paths.update({"full_" + key: value for key, value in full.items()})
    if isinstance(model, PQDQN):
        import json

        readout = {
            "observable_strings": model.observable_strings,
            "readout_coefficients": model.readout_coefficients.detach().cpu().tolist(),
            "fc_layer": model.fc_layer is not None,
            "input_activation": model.input_activation,
            "note": "Measurement meters show local bases; this file specifies the Pauli products and classical aggregation.",
        }
        (output_dir / f"{name}_readout.json").write_text(json.dumps(readout, indent=2))
    return paths
