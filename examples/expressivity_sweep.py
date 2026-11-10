"""Compare the Fourier spectrum of local and parity readouts as the width grows."""

import matplotlib.pyplot as plt
import numpy as np

from qasp import create_classifier, expressivity_analyzer


def make_classifier(n):
    # One shared input x at every width. Each qubit prepares RY(x)|0>.
    config = {
        "feature_map": "NeuralEmbedding",
        "feature_map_config": {"features": 1, "rotations": "Y"},
        "ansatz": "AngleAnsatz",
        "ansatz_config": {"n_layers": 1, "rotations": "Z"},
        "init": "zeros",
        "input_scaling_init": "ones",
        "readout": "expectation",
    }
    return create_classifier(config, n_qubits=n, n_classes=2)


def compare_widths(widths=(2, 4, 6, 8)):
    return expressivity_analyzer(
        model_factory=make_classifier,
        qubit_list=widths,
        observables=lambda n: ["Z 0", "Z" * n],
    )


def plot_sweep(sweep):
    """Plot counts and both coefficient spectra, using common magnitude limits."""
    widths = sweep.qubit_list
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
    active = np.stack([s.active_counts for s in sweep.results_by_qubit.values()])
    for values, label, marker in [
        (sweep.frequency_counts, "Encoding allows", "o"),
        (active[:, 0], "Local Z", "s"),
        (active[:, 1], "Parity", "^"),
    ]:
        axes[0].plot(widths, values, marker=marker, label=label)
    axes[0].set(
        xlabel="Data qubits", ylabel="Signed frequencies", title="Frequency counts", xticks=widths
    )
    axes[0].legend(frameon=False)

    # This family has integer frequencies, so all widths share one horizontal axis.
    largest = max(widths)
    for output, title in enumerate(["Local Z", "Parity"]):
        grid = np.zeros((len(widths), 2 * largest + 1))
        for row, spectrum in enumerate(sweep.results_by_qubit.values()):
            columns = spectrum.frequencies[:, 0].astype(int) + largest
            grid[row, columns] = spectrum.magnitudes[:, output]
        ax = axes[output + 1]
        heatmap = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=0,
            vmax=0.5,
            interpolation="nearest",
        )
        ticks = np.arange(-largest, largest + 1, max(1, largest // 2))
        ax.set(
            xticks=ticks + largest,
            xticklabels=ticks,
            yticks=np.arange(len(widths)),
            yticklabels=widths,
            xlabel=r"Frequency $\omega$",
            ylabel="Data qubits",
            title=title,
        )
    fig.colorbar(heatmap, ax=axes[1:], label=r"$|c_\omega|$", shrink=0.8)
    return fig


if __name__ == "__main__":
    result = compare_widths()
    for row in result.summary():
        print(row)
    plot_sweep(result)
    plt.show()
