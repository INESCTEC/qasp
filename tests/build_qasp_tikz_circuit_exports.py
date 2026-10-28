from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import networkx as nx
import numpy as np
import pennylane as qml
import torch

from qasp.circuits import AngleAnsatz, CircuitBuilder, IQPAnsatz
from qasp.circuits import circuit_to_tikz, tape_to_tikz
from qasp.feature_maps import NeuralEmbedding
from qasp.rl.model import create_model


ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
OUTPUT = TESTS / "qasp_tikz_circuit_exports"
OUTPUT.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "single": "DCEAF7",
    "two": "F3D7B5",
    "multi": "E6D6F2",
    "measurement": "D9EAD3",
}

EXPANDER_EDGES = [
    (0, 5), (0, 6), (0, 7), (1, 2), (1, 4),
    (1, 7), (2, 6), (2, 8), (3, 4), (3, 7),
    (3, 9), (4, 9), (5, 6), (5, 8), (8, 9),
]


def wire_labels(n: int) -> list[str]:
    return [rf"q_{{{index + 1}}}" for index in range(n)]


def ring_edges(n: int) -> list[tuple[int, int]]:
    return [(index, (index + 1) % n) for index in range(n)]


def path_edges(n: int) -> list[tuple[int, int]]:
    return [(index, index + 1) for index in range(n - 1)]


def tex_header(title: str, sources: list[str], api_call: str) -> str:
    lines = [
        "% QASP-generated publication circuit.",
        f"% Figure: {title}",
        "% Source notebook cells: " + "; ".join(sources),
        f"% QASP API: {api_call}",
        "% Gate and wire geometry below is emitted by qasp.circuits.tikz.",
        "% Compile: latexmk -pdf -interaction=nonstopmode -halt-on-error FILE.tex",
    ]
    return "\n".join(lines) + "\n"


def save_tex(filename: str, source: str, header: str) -> Path:
    path = OUTPUT / filename
    path.write_text(header + source, encoding="utf-8")
    return path


def variance_formatter(value, operation, parameter_index, global_index):
    name = str(getattr(operation, "name", operation.__class__.__name__))
    wires = [int(wire) for wire in operation.wires]
    word = getattr(operation, "hyperparameters", {}).get("pauli_word")
    if name == "RZ" or word == "Z":
        return rf"\alpha_{{{wires[0] + 1}}}"
    if word == "ZZ":
        return rf"\beta_{{{wires[0] + 1},{wires[1] + 1}}}"
    return rf"\theta_{{{global_index + 1}}}"


def build_iqp_probability_builder(
    n: int, edges: list[tuple[int, int]], include_local: bool
) -> tuple[CircuitBuilder, int]:
    ansatz = IQPAnsatz(
        n_qubits=n,
        n_layers=1,
        imprimitive_pattern=edges,
        include_single_qubit=include_local,
    )
    builder = CircuitBuilder(n_qubits=n)
    builder.set_ansatz(ansatz)
    builder.set_measurement("probs", wires=range(n))
    builder.build()
    return builder, int(ansatz.get_param_shapes()[0][0])


def export_variance_circuits() -> list[dict]:
    rows = []
    specifications = [
        {
            "filename": "variance_pure_rxx_ring_n4.tex",
            "title": "Pure-RXX periodic ring IQP, n=4",
            "edges": ring_edges(4),
            "include_local": False,
            "sources": [
                "tests/iqp_variance_experiments.ipynb cells 5, 27, 28"
            ],
            "equivalence": "Native H-D_Z-H realization of edge-only R_XX ring",
        },
        {
            "filename": "variance_graph_iqp_path_n4.tex",
            "title": "Local-plus-edge graph-IQP path, n=4",
            "edges": path_edges(4),
            "include_local": True,
            "sources": [
                "tests/iqp_variance_experiments.ipynb cells 5, 6, 12, 13"
            ],
            "equivalence": "Native H-D_Z-H realization of local R_X plus path R_XX",
        },
    ]
    for spec in specifications:
        builder, count = build_iqp_probability_builder(
            4, spec["edges"], spec["include_local"]
        )
        params = np.linspace(0.2, 1.0, count)
        source = circuit_to_tikz(
            builder,
            circuit_kwargs={"params": params},
            standalone=True,
            wire_order=range(4),
            wire_labels=wire_labels(4),
            parameter_formatter=variance_formatter,
            gate_colors=PALETTE,
            row_sep="0.34cm",
            column_sep="0.20cm",
        )
        path = save_tex(
            spec["filename"],
            source,
            tex_header(
                spec["title"], spec["sources"],
                "qasp.circuits.circuit_to_tikz(builder, circuit_kwargs=..., standalone=True)",
            ),
        )
        rows.append({
            **spec,
            "tex": path.name,
            "n_qubits": 4,
            "n_layers": 1,
            "n_parameters": count,
            "measurement_wires": list(range(4)),
        })
    return rows


def repeated_iqp_rotations(n: int) -> list[str]:
    return (
        [f"Z {i}" for i in range(n)]
        + [f"ZZ {i} {(i + 1) % n}" for i in range(n)]
        + [f"X {i}" for i in range(n)]
        + [f"XX {i} {i + 1}" for i in range(n - 1)]
    )


def repeated_formatter(value, operation, parameter_index, global_index):
    name = str(getattr(operation, "name", operation.__class__.__name__))
    wires = [int(wire) for wire in operation.wires]
    word = getattr(operation, "hyperparameters", {}).get("pauli_word")
    if global_index < 5:
        return rf"\lambda_{{{wires[0] + 1}}}x"
    if name == "RZ" or word == "Z":
        return rf"\alpha_{{{wires[0] + 1}}}"
    if word == "ZZ":
        return rf"\beta_{{{wires[0] + 1},{wires[1] + 1}}}"
    if name == "RX" or word == "X":
        return rf"\gamma_{{{wires[0] + 1}}}"
    if word == "XX":
        return rf"\delta_{{{wires[0] + 1},{wires[1] + 1}}}"
    return rf"\theta_{{{global_index + 1}}}"


def export_repeated_iqp() -> dict:
    n = 5
    layers = 2
    feature_map = NeuralEmbedding(n_qubits=n, features=1, rotations="X")
    ansatz = AngleAnsatz(
        n_qubits=n,
        n_layers=layers,
        rotations=repeated_iqp_rotations(n),
        feature_map=feature_map,
        reuploading=False,
        include_imprimitive=False,
        imprimitive="CNOT",
        imprimitive_pattern="chain",
    )
    builder = CircuitBuilder(n_qubits=n)
    builder.set_ansatz(ansatz)
    builder.set_measurement("probs", wires=range(n))
    builder.build()
    shapes = ansatz.get_param_shapes()
    n_scaling = int(shapes[0][0])
    n_parameters = int(shapes[1][0])
    source = circuit_to_tikz(
        builder,
        circuit_kwargs={
            "inputs": np.ones(1),
            "input_scaling_params": np.linspace(0.2, 0.8, n_scaling),
            "params": np.linspace(0.15, 1.15, n_parameters),
        },
        standalone=True,
        wire_order=range(n),
        wire_labels=wire_labels(n),
        parameter_formatter=repeated_formatter,
        gate_colors=PALETTE,
        row_sep="0.30cm",
        column_sep="0.16cm",
        compress=True,
        max_explicit_layers=3,
    )
    source = source.replace(
        r"U(\theta_{1})", r"\mathcal{V}_{1}(\Theta_{1})"
    ).replace(
        r"U(\theta_{2})", r"\mathcal{V}_{2}(\Theta_{2})"
    )
    path = save_tex(
        "variance_repeated_iqp_n5_L2.tex",
        source,
        tex_header(
            "Repeated IQP-like variance architecture, n=5, L=2",
            ["tests/iqp_variance_experiments.ipynb cells 63, 64, 65"],
            "qasp.circuits.circuit_to_tikz(builder, compress=True, standalone=True)",
        ),
    )
    return {
        "filename": path.name,
        "title": "Repeated IQP-like variance architecture, n=5, L=2",
        "n_qubits": n,
        "n_layers": layers,
        "input_scaling_entries": n_scaling,
        "active_initial_embedding_entries": n,
        "ansatz_parameters": n_parameters,
        "program": "initial RX embedding; per layer: RZ, periodic RZZ, RX, open-chain RXX",
        "measurement_wires": list(range(n)),
        "sources": ["tests/iqp_variance_experiments.ipynb cells 63, 64, 65"],
    }


def policy_config(n: int, edges, layers: int, initializer: str, measured):
    return {
        "ansatz": "IQPAnsatz",
        "ansatz_config": {
            "n_layers": layers,
            "imprimitive_pattern": copy.deepcopy(edges),
            "reuploading": True,
            "include_single_qubit": True,
        },
        "feature_map": "NeuralEmbedding",
        "feature_map_config": {"features": 4, "rotations": "Z"},
        "n_actions": 2,
        "init": initializer,
        "device": "lightning.qubit",
        "diff_method": "adjoint",
        "shots": None,
        "softmax_activation": False,
        "output_scaling": None,
        "torchrl": False,
        "measurement_qubits": list(measured),
        "parity_cnot_decomposition": False,
        "policy": "parity-like",
        "training_method": "expectation",
    }


def cartpole_formatter_for(n: int):
    def formatter(value, operation, parameter_index, global_index):
        wires = [int(wire) for wire in operation.wires]
        word = getattr(operation, "hyperparameters", {}).get("pauli_word")
        name = str(getattr(operation, "name", operation.__class__.__name__))
        if global_index < n:
            return rf"\varphi_{{{wires[0] + 1}}}(\mathbf{{s}})"
        if name == "RZ" or word == "Z":
            return rf"\alpha_{{{wires[0] + 1}}}"
        if word == "ZZ":
            # The ten-qubit export keeps the exact QASP-traced endpoints but
            # suppresses edge indices so the nonlocal gates remain legible at
            # two-column width. The complete indexed edge list is embedded in
            # the TeX provenance comment, README, and manifest.
            if n == 10:
                return r"\beta"
            return rf"\beta_{{{wires[0] + 1},{wires[1] + 1}}}"
        return rf"\theta_{{{global_index + 1}}}"
    return formatter


def make_cartpole_model(n, edges, layers, initializer, measured):
    torch.manual_seed(3)
    np.random.seed(3)
    return create_model(
        "PQCBorn",
        policy_config(n, edges, layers, initializer, measured),
        n_qubits=n,
    )


def probability_copy(model, measured):
    builder = copy.deepcopy(model.circuit_builder)
    builder.set_measurement("probs", wires=list(measured))
    builder.build()
    return builder


def trace_builder(builder, kwargs):
    make_qscript = getattr(qml.tape, "make_qscript", None)
    if make_qscript is None:
        make_qscript = qml.transforms.make_tape
    return make_qscript(builder.circuit)(**kwargs)


def export_cartpole_template(
    *, filename, title, n, edges, initializer, measured, source_cells
) -> dict:
    models = {
        layers: make_cartpole_model(n, edges, layers, initializer, measured)
        for layers in (1, 2, 3)
    }
    operation_counts = {}
    for layers, model in models.items():
        builder = probability_copy(model, measured)
        state = model.qlayer.state_dict()
        kwargs = {
            "inputs": torch.zeros(4),
            "input_scaling_params": state["input_scaling_params"],
            "params": state["params"],
        }
        operation_counts[layers] = len(trace_builder(builder, kwargs).operations)

    model = models[3]
    builder = probability_copy(model, measured)
    state = model.qlayer.state_dict()
    kwargs = {
        "inputs": torch.zeros(4),
        "input_scaling_params": state["input_scaling_params"],
        "params": state["params"],
    }
    tape = trace_builder(builder, kwargs)
    first_program_count = 3 * n + len(edges)
    expected_counts = {
        layers: n + layers * first_program_count for layers in (1, 2, 3)
    }
    assert operation_counts == expected_counts
    source = tape_to_tikz(
        tape,
        wire_order=range(n),
        wire_labels=wire_labels(n),
        standalone=True,
        parameter_formatter=cartpole_formatter_for(n),
        gate_colors=PALETTE,
        row_sep="0.24cm" if n == 10 else "0.30cm",
        column_sep="0.05cm" if n == 10 else "0.17cm",
        compress=True,
        n_layers=3,
        first_layer_start=n,
        first_layer_operation_count=first_program_count,
        suffix_operation_count=0,
        max_explicit_layers=3,
    )
    for layer in (1, 2, 3):
        source = source.replace(
            rf"U(\theta_{{{layer}}})",
            rf"\mathcal{{U}}_{{{layer}}}(\mathbf{{s}};\Theta_{{{layer}}})",
        )
    source = source.replace("R_z(", "R_Z(")
    if n == 10:
        source = source.replace("\\begin{document}", "\\begin{document}\n\\scriptsize")
    header = tex_header(
        title,
        source_cells,
        "qasp.circuits.tape_to_tikz(QASP-traced tape, compress=True, explicit layer metadata)",
    )
    header += (
        "% Depth reading: L=1 stops after U_1; L=2 after U_2; "
        "L=3 uses the complete displayed circuit.\n"
    )
    path = save_tex(filename, source, header)
    return {
        "filename": path.name,
        "title": title,
        "n_qubits": n,
        "represented_depths": [1, 2, 3],
        "edges": edges,
        "initializer": initializer,
        "measurement_wires": list(measured),
        "operation_counts": operation_counts,
        "first_state_conditioned_program_operations": first_program_count,
        "sources": source_cells,
        "depth_convention": "L=1/2/3 are prefixes through U_1/U_2/U_3",
    }


def compile_and_render(tex_paths: list[Path]) -> None:
    pdftoppm = subprocess.check_output(
        ["bash", "-lc", "command -v pdftoppm"], text=True
    ).strip()
    for tex_path in tex_paths:
        log_path = tex_path.with_suffix(".compile.log")
        completed = subprocess.run(
            [
                "latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                "-outdir=" + str(OUTPUT), tex_path.name,
            ],
            cwd=OUTPUT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        log_path.write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"LaTeX compilation failed for {tex_path}: {completed.stdout[-4000:]}")
        pdf_path = tex_path.with_suffix(".pdf")
        png_prefix = tex_path.with_suffix("")
        subprocess.run(
            [pdftoppm, "-png", "-singlefile", "-r", "300", str(pdf_path), str(png_prefix)],
            check=True,
            cwd=OUTPUT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )


def main():
    records = export_variance_circuits()
    records.append(export_repeated_iqp())
    records.append(export_cartpole_template(
        filename="cartpole_graph_iqp_ring5_depths_L1_L2_L3.tex",
        title="CartPole graph-IQP template: 5-qubit ring, L=1,2,3",
        n=5,
        edges=ring_edges(5),
        initializer="normal_0_1",
        measured=list(range(5)),
        source_cells=["tests/test_policy_training_efficiency.ipynb cells 8-16"],
    ))
    records.append(export_cartpole_template(
        filename="cartpole_graph_iqp_expander10_depths_L1_L2_L3.tex",
        title="CartPole graph-IQP template: 10-qubit cubic expander, L=1,2,3",
        n=10,
        edges=EXPANDER_EDGES,
        initializer="uniform",
        measured=list(range(9)),
        source_cells=["tests/test_policy_training_efficiency.ipynb cells 17-26"],
    ))

    tex_paths = [OUTPUT / record["filename"] for record in records]
    compile_and_render(tex_paths)

    for record in records:
        stem = Path(record["filename"]).stem
        for suffix in (".tex", ".pdf", ".png"):
            artifact = OUTPUT / f"{stem}{suffix}"
            record[suffix[1:] + "_sha256"] = hashlib.sha256(
                artifact.read_bytes()
            ).hexdigest()
            record[suffix[1:] + "_bytes"] = artifact.stat().st_size

    notebook_hashes = {
        name: hashlib.sha256((TESTS / name).read_bytes()).hexdigest()
        for name in (
            "iqp_variance_experiments.ipynb",
            "test_policy_training_efficiency.ipynb",
        )
    }
    manifest = {
        "generator": str(Path(__file__).resolve()),
        "qasp_api_calls": [
            "qasp.circuits.circuit_to_tikz",
            "qasp.circuits.tape_to_tikz",
            "qasp.rl.model.create_model",
            "CircuitBuilder.set_measurement('probs')",
        ],
        "source_notebook_sha256": notebook_hashes,
        "artifacts": records,
        "exporter_limitations": [
            "The exporter emits LaTeX but does not compile it; this build uses latexmk.",
            "Automatic builder compression rejects reuploading=True; CartPole circuits are traced from QASP builders and passed to tape_to_tikz with verified layer boundaries.",
            "Compressed later programs are intentionally represented by full-register U_l blocks; the first program is gate-explicit.",
            "NeuralEmbedding contractions need parameter_formatter for paper symbols.",
            "Computational-basis probability meters are unlabeled by Quantikz; measured wires are explicit in the manifest.",
            "For the 10-qubit CartPole circuit, q_10 participates in the unitary but is deliberately unmeasured, matching the notebook.",
            "The 10-qubit diagram keeps every traced edge endpoint but shortens all edge-angle labels to beta for legibility at two-column width; the indexed edge list is stored in this manifest.",
        ],
    }
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    readme = """# QASP TikZ circuit exports

Every gate/wire layout in this directory is generated by QASP's own
`circuit_to_tikz` or `tape_to_tikz` path from the same circuit objects and
configurations used by the current executed notebooks. The CartPole display
builders change only the readout to full computational-basis probabilities;
the instantiated unitary is unchanged.

## Files

- `variance_pure_rxx_ring_n4`: policy-variance pure-RXX periodic ring, shown in
  its native H-D_Z-H realization.
- `variance_graph_iqp_path_n4`: local-plus-edge path graph-IQP, native H-D_Z-H.
- `variance_repeated_iqp_n5_L2`: initial trainability embedding followed by the
  canonical RZ/RZZ-ring/RX/RXX-chain alternating programs; layer 1 is explicit
  and layer 2 is the QASP-generated compressed block.
- `cartpole_graph_iqp_ring5_depths_L1_L2_L3`: state-conditioned reuploaded
  5-qubit ring. L=1,2,3 are prefixes ending after U_1,U_2,U_3.
- `cartpole_graph_iqp_expander10_depths_L1_L2_L3`: the exact connected cubic
  10-qubit graph and the same prefix convention. Wires q_1,...,q_9 are measured;
  q_10 remains active but unmeasured, matching the notebook.

Each base name has standalone `.tex`, compiled `.pdf`, and 300-DPI `.png`.
Compile a source with:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error FILE.tex
```

Dependencies: the QASP environment (PennyLane, PyTorch, and QASP itself) is
needed only to regenerate the sources. Standalone compilation needs a TeX
distribution containing `standalone`, `tikz`/`xcolor`, and `quantikz`;
`latexmk` drives `pdflatex`. PNG previews are rendered with Poppler's
`pdftoppm -png -r 300`.

## API calls and limitations

The variance circuits call `qasp.circuits.circuit_to_tikz` on their actual
`CircuitBuilder` instances. The CartPole circuits are created through
`qasp.rl.model.create_model`; a copy of each model's QASP builder is given a
`probs` readout, traced, and exported with `qasp.circuits.tape_to_tikz`.
The exact export calls are:

```python
circuit_to_tikz(builder, circuit_kwargs=kwargs, standalone=True,
                parameter_formatter=formatter, gate_colors=palette)
circuit_to_tikz(builder, circuit_kwargs=kwargs, standalone=True,
                compress=True, parameter_formatter=formatter,
                gate_colors=palette)
tape_to_tikz(tape, wire_order=range(n), standalone=True,
             parameter_formatter=formatter, gate_colors=palette,
             compress=True, n_layers=3, first_layer_start=n,
             first_layer_operation_count=3*n + len(edges),
             suffix_operation_count=0, max_explicit_layers=3)
```

QASP cannot automatically compress a builder with `reuploading=True`. For the
CartPole diagrams, verified QASP tape boundaries are supplied explicitly:
5-qubit prefix 5 and program length 20; 10-qubit prefix 10 and program length
45. Hidden programs are QASP's full-register U_l blocks. Only block text is
changed to U_l(s;Theta_l); no gate or wire layout is hand drawn.
For the dense 10-qubit drawing, QASP still generates all gate/wire geometry;
only the repeated edge-angle text is shortened to beta. Its indexed edge list
is retained in `manifest.json`.
"""
    (OUTPUT / "README.md").write_text(readme, encoding="utf-8")
    print(OUTPUT)
    for record in records:
        print(record["filename"])


if __name__ == "__main__":
    main()
