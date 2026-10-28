"""Render publication figures from the completed PQDQN CartPole records.

This script reads the preserved seed-level CSV files only; it never trains or
modifies a model.  It complements the vector moving-return plot exported by
the executed notebook with a vector held-out-return figure.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "examples/results/pqdqn_cartpole_pauli_features_bias_v2_400ep_5seeds"
FIGURES = Path(__file__).resolve().parent / "figures"
READOUTS = ["shared", "per_action", "parity_a2", "majority_a2"]
LABELS = {
    "shared": "Shared\nZ features",
    "per_action": "Per-action\nsums",
    "parity_a2": "Parity\na = 2",
    "majority_a2": "Majority\na = 2",
}


def main():
    frame = pd.read_csv(RESULTS / "run_summaries.csv")
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    for index, name in enumerate(READOUTS):
        scores = (
            frame.loc[frame["readout"] == name]
            .sort_values("seed")["held_out_mean"]
            .to_numpy()
        )
        jitter = np.linspace(-0.12, 0.12, len(scores))
        ax.scatter(index + jitter, scores, s=34, alpha=0.78, zorder=3)
        ax.errorbar(
            index,
            scores.mean(),
            yerr=scores.std(ddof=1),
            fmt="k_",
            markersize=13,
            capsize=5,
            linewidth=1.2,
            zorder=4,
        )
    ax.set_xticks(range(len(READOUTS)), [LABELS[name] for name in READOUTS])
    ax.set_ylabel("Mean held-out return per trained seed")
    ax.set_ylim(0, 510)
    ax.axhline(475, color="0.45", linestyle="--", linewidth=1.2,
               label="475 reference")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False, loc="upper right")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "pqdqn_held_out_returns.pdf")
    fig.savefig(FIGURES / "pqdqn_held_out_returns.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
