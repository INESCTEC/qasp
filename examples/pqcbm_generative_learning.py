"""Reproducible three-qubit PQCBM experiment with exact MMD and PP validation.

Run from the repository root with:
    python examples/pqcbm_generative_learning.py
Outputs are written to examples/results/pqcbm_small_experiment by default.
"""

import json
import platform
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import torch

from qasp import create_qcbm, draw_qcbm
from qasp.genlearn.losses import basis_indices

CONFIG = {
    "n_qubits": 3,
    "ansatz": "AngleAnsatz",
    "ansatz_config": {
        "n_layers": 2,
        "rotations": "YZ",
        "include_imprimitive": True,
        "imprimitive": "CNOT",
        "imprimitive_pattern": "chain",
    },
    "loss": "mmd",
    "mmd_config": {"mode": "exact", "bandwidths": [0.5, 1.0, 2.0]},
    "device": "default.qubit",
    "seed": 7,
    "lr": 0.05,
}
OUTPUT = Path(__file__).resolve().parent / "results" / "pqcbm_small_experiment"


def run_experiment(output=OUTPUT, steps=120, *, export_tikz=True):
    """Train matched statevector and unpruned-PP models; save portable artifacts."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = np.array([[0, 0, 0], [1, 1, 1]] * 128)
    statevector = create_qcbm(CONFIG, data=data)
    pp = create_qcbm({**CONFIG, "device": "lightning.pp", "pp_prune": False}, data=data)
    initial_probs = statevector.get_probs().detach().numpy()
    initial = float(statevector.loss().detach())
    a, b = statevector.loss(), pp.loss()
    ga = torch.autograd.grad(a, statevector.params)[0]
    gb = torch.autograd.grad(b, pp.params)[0]
    gradient_error = float((ga - gb).abs().max())
    assert gradient_error < 1e-10
    start = time.perf_counter()
    _, sv_history = statevector.train(steps)
    sv_seconds = time.perf_counter() - start
    start = time.perf_counter()
    _, pp_history = pp.train(steps)
    pp_seconds = time.perf_counter() - start
    final = float(statevector.loss().detach())
    final_pp = float(pp.loss().detach())
    probs = statevector.get_probs().detach().numpy()
    pp_probs = pp.get_probs(device="default.qubit").detach().numpy()
    np.testing.assert_allclose(probs, pp_probs, atol=1e-9, rtol=1e-9)
    assert final < initial * 0.1
    bits = pp.sample(2000, device="default.qubit", device_kwargs={"seed": 19})
    histogram = np.bincount(basis_indices(bits), minlength=8) / len(bits)
    target = np.zeros(8)
    target[[0, 7]] = 0.5
    summary = {
        "config": CONFIG,
        "steps": steps,
        "training_rows": len(data),
        "initial_mmd2": initial,
        "final_mmd2": final,
        "final_pp_mmd2": final_pp,
        "initial_gradient_max_error": gradient_error,
        "final_probability_max_error": float(np.max(np.abs(probs - pp_probs))),
        "target_support_probability": float(probs[[0, 7]].sum()),
        "total_variation_to_target": float(np.abs(probs - target).sum() / 2),
        "statevector_seconds": sv_seconds,
        "pp_seconds": pp_seconds,
        "python": platform.python_version(),
        "pennylane": qml.__version__,
        "torch": torch.__version__,
        "validation_scope": "Three qubits, one seed; unpruned PP; not a scaling benchmark or MPI run.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "trained_parameters.json").write_text(
        json.dumps(
            {"config": CONFIG, "params": statevector.params.detach().tolist()}, indent=2
        )
        + "\n"
    )
    np.savez(
        output / "results.npz",
        target=target,
        initial_probs=initial_probs,
        final_probs=probs,
        pp_probs=pp_probs,
        sampled_frequencies=histogram,
        statevector_losses=sv_history,
        pp_losses=pp_history,
        samples=bits,
    )
    if export_tikz:
        (output / "pqcbm.tex").write_text(
            draw_qcbm(
                statevector,
                output="tikz",
                tikz_options={
                    "standalone": True,
                    "symbolic_parameters": True,
                    "compress": True,
                    "gate_colors": False,
                },
            )
        )
    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    axes[0].semilogy(
        np.arange(steps + 1), [*sv_history, final], color="#225D96", label="Statevector"
    )
    axes[0].semilogy(
        np.arange(steps + 1),
        [*pp_history, final_pp],
        "--",
        color="#C36124",
        label="Pauli propagation",
    )
    axes[0].set(
        xlabel="Optimizer update",
        ylabel="Exact Gaussian MMD squared",
        title="Learning correlated bitstrings",
    )
    axes[0].legend(frameon=False)
    x = np.arange(8)
    axes[1].bar(x - 0.24, target, width=0.24, color="#283B49", label="Target")
    axes[1].bar(x, probs, width=0.24, color="#225D96", label="Trained model")
    axes[1].bar(x + 0.24, histogram, width=0.24, color="#55A49F", label="2,000 samples")
    axes[1].set(
        xticks=x,
        xticklabels=[format(i, "03b") for i in x],
        xlabel="Bitstring",
        ylabel="Probability",
        title="Distribution after training",
    )
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend(frameon=False)
    fig.savefig(output / "learning.png", dpi=180)
    plt.close(fig)
    return summary


if __name__ == "__main__":
    print(json.dumps(run_experiment(), indent=2))
