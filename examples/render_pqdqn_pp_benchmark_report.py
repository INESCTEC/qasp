"""Render the saved benchmark into paper figures and a standalone PDF report.

Requires pandas, matplotlib and reportlab. Run after the benchmark script and
its notebook drawing cell. This reads saved measurements; it never retrains.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/results/pqdqn_pp_10qubit_benchmark"
PDF_DIR = ROOT / "output/pdf"
LABELS = {"lightning_qubit": "Lightning adjoint", "pp_k2": "PP k=2", "pp_k3": "PP k=3"}
ORDER = list(LABELS)
PALETTE = ["#284b63", "#2a9d8f", "#e09f3e"]


def main():
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert "completed_utc" in manifest
    raw = pd.read_csv(DATA / "timings.csv")
    accuracy = pd.read_csv(DATA / "accuracy.csv")
    medians = raw.groupby(
        ["seed", "batch_size", "backend", "workload"], as_index=False
    ).milliseconds.median()
    summary = (
        medians.groupby(["batch_size", "backend", "workload"])
        .milliseconds.agg(["mean", "std"])
        .reset_index()
    )
    accuracy_summary = (
        accuracy.groupby(["batch_size", "backend"])
        .agg(
            feature_rmse=("features_rmse", "mean"),
            q_rmse=("values_rmse", "mean"),
            feature_relative_l2=("features_relative_l2", "mean"),
            quantum_gradient_relative_l2=("quantum_gradients_relative_l2", "mean"),
            quantum_gradient_cosine=("quantum_gradients_cosine", "mean"),
        )
        .reset_index()
    )
    accuracy_summary.to_csv(DATA / "accuracy_summary.csv", index=False)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.35), layout="constrained")
    selected = (
        summary.query("batch_size == 8 and workload == 'forward_backward'")
        .set_index("backend")
        .loc[ORDER]
    )
    axes[0].bar(
        range(3), selected["mean"], yerr=selected["std"], color=PALETTE, capsize=4
    )
    axes[0].set(
        xticks=range(3),
        xticklabels=[LABELS[b] for b in ORDER],
        ylabel="Forward + backward time (ms)",
        title="A. Batch of eight observations",
    )
    for i, v in enumerate(selected["mean"]):
        axes[0].text(
            i, v + selected["std"].iloc[i] + 3, f"{v:.1f}", ha="center", fontsize=9
        )
    axes[0].set_ylim(0, float((selected["mean"] + selected["std"]).max()) * 1.2)
    for i, b in enumerate(ORDER[1:]):
        a = accuracy.query("batch_size == 8 and backend == @b")
        axes[1].scatter(
            np.full(len(a), i) + np.linspace(-0.09, 0.09, len(a)),
            a.quantum_gradients_relative_l2,
            color=PALETTE[i + 1],
            s=32,
        )
        axes[1].plot(
            [i - 0.18, i + 0.18],
            [a.quantum_gradients_relative_l2.mean()] * 2,
            color="#333333",
            lw=1.8,
        )
    axes[1].axhline(
        1, color="#888888", ls="--", lw=0.8, label="Error norm = reference norm"
    )
    axes[1].set(
        xticks=[0, 1],
        xticklabels=["PP k=2", "PP k=3"],
        ylabel="Relative L2 quantum-gradient error",
        title="B. Five matched parameter seeds",
    )
    axes[1].legend(fontsize=7, loc="lower left")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    fig.savefig(DATA / "timing_accuracy.pdf", bbox_inches="tight")
    fig.savefig(DATA / "timing_accuracy.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BodyCustom",
            fontName="Helvetica",
            fontSize=9.4,
            leading=13,
            spaceAfter=8,
            textColor=colors.HexColor("#243746"),
        )
    )
    styles.add(
        ParagraphStyle(
            name="SmallCustom",
            fontName="Helvetica",
            fontSize=8,
            leading=10.5,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(name="CodeCustom", fontName="Courier", fontSize=7.2, leading=9)
    )
    styles["Title"].textColor = colors.HexColor("#284b63")
    styles["Heading2"].textColor = colors.HexColor("#284b63")
    story = []

    def p(text, style="BodyCustom"):
        story.append(Paragraph(text, styles[style]))

    def table(rows, widths):
        t = Table(
            [[Paragraph(str(c), styles["SmallCustom"]) for c in row] for row in rows],
            colWidths=widths,
            repeatRows=1,
            hAlign="LEFT",
        )
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eff3")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.7, colors.HexColor("#7793a4")),
                    ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#7793a4")),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story.extend([t, Spacer(1, 10)])

    def picture(path, width, max_height=None):
        w, h = ImageReader(str(path)).getSize()
        scale = min(width / w, max_height / h if max_height else width / w)
        story.append(Image(str(path), width=w * scale, height=h * scale))

    p("10-qubit PQDQN simulator benchmark", "Title")
    p(
        "Differentiable Pauli propagation with low Pauli-weight cutoffs versus Lightning statevector adjoint differentiation. Measured 14 October 2026.",
        "BodyCustom",
    )
    p(
        "<b>Finding.</b> PP at k=2 has a lower gradient-workload time in this experiment; k=3 is slower than Lightning. Both low cutoffs poorly approximate the exact quantum gradients. This is evidence of a computational trade-off, not evidence of faster or equally effective reinforcement learning."
    )
    p("Matched model and measurement protocol", "Heading2")
    p(
        "Ten qubits; four YZ AngleAnsatz layers; trainable Y angle encoding with tanh input and data reuploading; CNOT ring in every layer, including the final layer. Ten shared features z<sub>i</sub>=&lt;Z<sub>i</sub>&gt; feed Q=Wz+b with two action outputs. There are "
        + str(manifest["parameter_count"])
        + " trainable parameters. All simulators receive identical parameters and observations for seeds 7-11. The QASP initializer normal_0_1 uses N(0,1), not standard deviation 0.1."
    )
    p(
        "PP uses k=2 or k=3 with coefficient/top-M pruning disabled: pp_prune=False, pp_tau=0, pp_topM=None. Lightning uses analytic lightning.qubit with adjoint differentiation. Both use CPU float64 and sequential sample evaluation. Two warmups precede seven timed repeats per seed, workload and batch size; backend order is shuffled within each repeat."
    )
    rows = [["Batch", "Backend", "Inference (ms)", "Forward + backward (ms)"]]
    for batch in [1, 8]:
        for b in ORDER:
            s = summary.query("batch_size == @batch and backend == @b").set_index(
                "workload"
            )
            rows.append(
                [
                    batch,
                    LABELS[b],
                    f"{s.loc['inference', 'mean']:.2f} +/- {s.loc['inference', 'std']:.2f}",
                    f"{s.loc['forward_backward', 'mean']:.2f} +/- {s.loc['forward_backward', 'std']:.2f}",
                ]
            )
    table(rows, [40, 145, 140, 170])
    p(
        "Table 1. Mean +/- sample SD of the five within-seed median times; each median uses seven measurements. Times are per whole batch. SD is descriptive variability, not a confidence interval. Forward + backward includes zero_grad, PQDQN evaluation, mean Huber loss and backward; it excludes optimizer, replay, target-network and environment work.",
        "SmallCustom",
    )
    picture(DATA / "timing_accuracy.png", 495)
    p(
        "Figure 1. Left: batch-eight gradient-workload timing with sample SD across seed medians. Right: quantum-parameter gradient errors for each of five matched seeds; horizontal solid marks are means.",
        "SmallCustom",
    )

    story.append(PageBreak())
    p("Accuracy and interpretation", "Title")
    rows = [
        [
            "Batch",
            "Approximation",
            "Feature RMSE",
            "Q RMSE",
            "Quantum grad. relative L2",
            "Grad. cosine",
        ]
    ]
    for batch in [1, 8]:
        for b in ORDER[1:]:
            a = accuracy_summary.query("batch_size == @batch and backend == @b").iloc[0]
            rows.append(
                [
                    batch,
                    LABELS[b],
                    f"{a.feature_rmse:.4f}",
                    f"{a.q_rmse:.4f}",
                    f"{a.quantum_gradient_relative_l2:.3f}",
                    f"{a.quantum_gradient_cosine:.3f}",
                ]
            )
    table(rows, [35, 95, 85, 70, 125, 85])
    p(
        "Table 2. Arithmetic means across five model seeds. Each seed metric is computed over the entire indicated batch. Quantum gradients include all variational rotation and input-scaling parameters and exclude the classical head. Relative L2 error is ||g_PP-g_exact|| / ||g_exact||. Cosine similarity is g_PP dot g_exact divided by the product of their norms. Raw per-seed absolute errors and norms are also saved.",
        "SmallCustom",
    )
    p("Why the timing alone is insufficient", "Heading2")
    p(
        "A relative gradient error near one means the error norm is comparable with the entire exact gradient. The small absolute Q errors do not imply accurate quantum propagation: the shared classical bias is identical between models and the affine head can attenuate feature differences. Low cutoffs discard propagated Pauli terms and can remove paths that later contribute to a low-weight observable. Raising k is not guaranteed to improve every error metric monotonically."
    )
    p(
        "The measured workload is a fixed-parameter Huber-loss derivative, with alternating selected actions and fixed targets from 0.25 to 1.75. It is representative of the differentiable part of a Q-learning update, but is not a full DQN update and does not demonstrate convergence, equal learning quality, or reduced time to solve an environment. No CartPole data were added to or replaced by this benchmark."
    )
    p("Correctness checks", "Heading2")
    checks = json.loads((DATA / "verification.json").read_text())
    exact = checks[0]
    p(
        f"Lightning adjoint outputs and all parameter gradients were checked against default.qubit/backprop for seed 7, batch eight (maximum gradient discrepancy {exact['max_gradient_error']:.2e}). Directional finite differences independently checked the gradients of both truncated PP models at the same inputs. These checks establish differentiation correctness for the respective truncated functions; they do not make truncation exact."
    )
    p("Hardware and reproducibility limits", "Heading2")
    versions = manifest["versions"]
    p(
        f"Apple M1 CPU, 16 GiB memory, eight hardware cores; macOS {manifest['platform']}. Torch intra/inter-op threads and OMP/OpenBLAS/MKL thread settings are one. Python {manifest['python']}; Torch {versions['torch']}; PennyLane {versions['pennylane']}; PennyLane-Lightning {versions['pennylane-lightning']}. Analytic execution, no finite shots, GPU or MPI. Timings include the current QASP wrappers and per-call PP tape construction."
    )
    p(
        "The five model seeds probe parameter dependence; they are not five independent machine sessions. Measurements were collected in one local session, without core pinning, thermal control or a dedicated idle machine. First construction timings are saved separately and excluded from Table 1. Results apply to this circuit, readout, initialization and small batches; they do not establish general backend scaling or a universal speedup."
    )
    p("Use in the QASP paper", "Heading2")
    p(
        "Present this as a simulator microbenchmark accompanying PQDQN differentiability support. Keep it separate from the completed four-qubit CartPole learning experiment. State both the runtime advantage observed here and the large quantum-gradient approximation error; do not describe small k as an accuracy-preserving replacement for adjoint simulation."
    )

    story.append(PageBreak())
    p("Circuit and reproducible example", "Title")
    drawing = DATA / "drawings/circuit.png"
    if drawing.exists():
        picture(drawing, 495, max_height=285)
        p(
            "Figure 2. QASP Quantikz circuit for all three backend settings. Subsequent blocks denote the complete reuploaded circuit layers. Meters specify local Z measurement bases; ten separate single-Z expectations form the readout vector. The affine two-output head Q=Wz+b is classical and lies outside the diagram. Expanded vector and TeX versions are supplied separately.",
            "SmallCustom",
        )
    p("Instantiate the truncated model", "Heading2")
    code = """from pqdqn_pauli_propagation_benchmark import model_config
from qasp.rl import PQDQN

model = PQDQN(
    **model_config(),            # ten-qubit, four-layer model
    device="lightning.pp",
    diff_method="backprop",
    pp_truncation=2,             # also benchmarked at k=3
    pp_prune=False,
    pp_topM=None,
    pp_tau=0.0,
).double()
# lightning.pp is QASP's differentiable Pauli engine.
# It is not a separate PennyLane Lightning device plugin."""
    story.append(Preformatted(code, styles["CodeCustom"]))
    p("Reproduce and inspect", "Heading2")
    p(
        "Open examples/pqdqn_pauli_propagation_benchmark.ipynb for the executed example, circuit drawing and measured tables. To rerun, invoke examples/pqdqn_pauli_propagation_benchmark.py from the repository root with PYTHONPATH=. and OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1; pass --output with a new directory. The script refuses to overwrite nonempty results."
    )
    p(
        "The results directory examples/results/pqdqn_pp_10qubit_benchmark contains manifest.json, timings.csv, seed_timing_medians.csv, timing_summary.csv, accuracy.csv, accuracy_summary.csv, setup_times.csv, verification.json, five seed checkpoints and their input tensors, and the vector timing_accuracy.pdf figure. The drawings subdirectory contains compact/full circuit.pdf and .tex exports and circuit_readout.json. Source fingerprints and verification details are retained with the report."
    )
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#b9cad4"))
        canvas.line(50, 42, 545, 42)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#526775"))
        canvas.drawString(
            50, 29, "QASP | 10-qubit PQDQN simulator benchmark | 14 October 2026"
        )
        canvas.drawRightString(545, 29, str(doc.page))
        canvas.restoreState()

    SimpleDocTemplate(
        str(PDF_DIR / "pqdqn_10qubit_backend_report.pdf"),
        pagesize=(595.28, 841.89),
        rightMargin=50,
        leftMargin=50,
        topMargin=42,
        bottomMargin=55,
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    print(PDF_DIR / "pqdqn_10qubit_backend_report.pdf")


if __name__ == "__main__":
    main()
