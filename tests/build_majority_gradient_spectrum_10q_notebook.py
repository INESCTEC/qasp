"""Build the 10-qubit L=1 majority-gradient spectrum notebook.

The five-qubit builder remains the audited source template.  This variant
applies explicit, asserted substitutions for the existing 10-qubit connected
cubic graph used by the decoder-ablation study.  The generated notebook is
self-contained and does not import either builder at execution time.
"""

from pathlib import Path


ROOT = Path("/Users/andresequeira/Desktop/QASP")
SOURCE_BUILDER = ROOT / "tests" / "build_majority_gradient_spectrum_notebook.py"
source = SOURCE_BUILDER.read_text()


def replace_once(old: str, new: str) -> None:
    global source
    count = source.count(old)
    if count != 1:
        raise RuntimeError(f"Expected one source-template match, found {count}: {old[:80]!r}")
    source = source.replace(old, new, 1)


replace_once(
    'NOTEBOOK = ROOT / "tests" / "majority_gradient_spectrum_during_training.ipynb"',
    'NOTEBOOK = ROOT / "tests" / "majority_gradient_spectrum_during_training_10q.ipynb"',
)
replace_once(
    '# Majority Pauli-gradient spectrum during CartPole-v1/200 training',
    '# Ten-qubit majority Pauli-gradient spectrum during CartPole-v1/200 training',
)
replace_once(
    'full deterministic five-qubit majority policy** become more important during',
    'full deterministic nine-measured-bit majority policy** become more important during',
)
replace_once(
    'and 500-episode analytic-gradient protocol match the five-qubit ring, L=3\n'
    'experiment in `test_policy_training_efficiency.ipynb` and',
    'and 500-episode analytic-gradient protocol match the ten-qubit connected cubic\n'
    'graph, L=1 experiment in `test_policy_training_efficiency.ipynb` and',
)
replace_once(
    'RESULTS_DIR = ROOT / "tests" / "results" / "majority_gradient_spectrum_during_training"',
    'RESULTS_DIR = ROOT / "tests" / "results" / "majority_gradient_spectrum_during_training_10q"',
)
replace_once(
    'import matplotlib.pyplot as plt\nimport numpy as np',
    'import matplotlib.pyplot as plt\nimport networkx as nx\nimport numpy as np',
)
replace_once(
    'PROTOCOL_VERSION = "source_action_rng_reset_after_model_init_v2"',
    'PROTOCOL_VERSION = "source_action_rng_reset_after_model_init_v2_exact_sector_vjp"',
)

old_spec = '''SPEC = {
    "label": "5-qubit ring, L=3",
    "n_qubits": 5,
    "n_layers": 3,
    "edges": [(i, (i + 1) % 5) for i in range(5)],
    "measurement_wires": list(range(5)),
    "initializer": "normal_0_1",
    "expected_quantum_parameters": 90,
}'''
new_spec = '''EXPANDER_EDGES = [
    (0, 5), (0, 6), (0, 7), (1, 2), (1, 4),
    (1, 7), (2, 6), (2, 8), (3, 4), (3, 7),
    (3, 9), (4, 9), (5, 6), (5, 8), (8, 9),
]

SPEC = {
    "label": "10-qubit connected cubic graph, L=1 (9 measured)",
    "n_qubits": 10,
    "n_layers": 1,
    "edges": EXPANDER_EDGES,
    "measurement_wires": list(range(9)),
    "initializer": "uniform",
    "expected_quantum_parameters": 65,
    "accepted_networkx_seed": 1039606416,
}
ODD_WEIGHTS = tuple(range(1, len(SPEC["measurement_wires"]) + 1, 2))
graph = nx.Graph()
graph.add_nodes_from(range(SPEC["n_qubits"]))
graph.add_edges_from(SPEC["edges"])
assert nx.is_connected(graph)
assert all(degree == 3 for _, degree in graph.degree())
assert SPEC["measurement_wires"] == list(range(9)) and 9 not in SPEC["measurement_wires"]'''
replace_once(old_spec, new_spec)

old_theory = r'''For five measured qubits, the deterministic majority observable is

\[
M_5=\frac{3}{8}\sum_{|S|=1}Z_S
-\frac{1}{8}\sum_{|S|=3}Z_S
+\frac{3}{8}Z_1Z_2Z_3Z_4Z_5,
\qquad
\Pi_0=\frac{I+M_5}{2},\quad \Pi_1=\frac{I-M_5}{2}.
\]

Weights two and four vanish exactly.'''
new_theory = r'''For the nine measured qubits of the ten-qubit circuit, the deterministic
majority observable is

\[
M_9=\frac{35}{128}\sum_{|S|=1}Z_S
-\frac{5}{128}\sum_{|S|=3}Z_S
+\frac{3}{128}\sum_{|S|=5}Z_S
-\frac{5}{128}\sum_{|S|=7}Z_S
+\frac{35}{128}Z_1\cdots Z_9,
\qquad
\Pi_0=\frac{I+M_9}{2},\quad \Pi_1=\frac{I-M_9}{2}.
\]

Even weights vanish exactly.  Qubit 10 remains coupled through the cubic graph
but is traced out by the decoder.'''
replace_once(old_theory, new_theory)
source = source.replace(r'M_5(x_t)', r'M_9(x_t)')

old_table = '''def build_pauli_table(n=5):
    rows = []
    coefficients = {1: 3.0 / 8.0, 3: -1.0 / 8.0, 5: 3.0 / 8.0}
    string_index = 0
    for weight in (1, 3, 5):
        for support in itertools.combinations(range(n), weight):
            label = "".join("Z" if wire in support else "I" for wire in range(n))
            rows.append({
                "string_index": string_index,
                "weight": weight,
                "support_zero_based": " ".join(map(str, support)),
                "pauli_string": label,
                "coefficient": coefficients[weight],
            })
            string_index += 1
    table = pd.DataFrame(rows)
    assert len(table) == 16
    assert table.groupby("weight").size().to_dict() == {1: 5, 3: 10, 5: 1}
    return table'''
new_table = '''def build_pauli_table(n=9):
    bitstrings = np.asarray(list(itertools.product((0, 1), repeat=n)), dtype=int)
    majority = np.where(bitstrings.sum(axis=1) <= n // 2, 1.0, -1.0)
    rows = []
    string_index = 0
    for weight in ODD_WEIGHTS:
        for support in itertools.combinations(range(n), weight):
            character = (-1.0) ** bitstrings[:, support].sum(axis=1)
            coefficient = float(np.mean(majority * character))
            label = "".join("Z" if wire in support else "I" for wire in range(n))
            rows.append({
                "string_index": string_index,
                "weight": weight,
                "support_zero_based": " ".join(map(str, support)),
                "pauli_string": label,
                "coefficient": coefficient,
            })
            string_index += 1
    table = pd.DataFrame(rows)
    assert len(table) == 2 ** (n - 1)
    assert table.groupby("weight").size().to_dict() == {
        weight: math.comb(n, weight) for weight in ODD_WEIGHTS
    }
    expected = {1: 35/128, 3: -5/128, 5: 3/128, 7: -5/128, 9: 35/128}
    for weight, coefficient in expected.items():
        np.testing.assert_allclose(
            table.loc[table.weight == weight, "coefficient"], coefficient, atol=1e-12
        )
    return table'''
replace_once(old_table, new_table)

replace_once(
    'majority_diag = (bits.sum(dim=1) <= n // 2).to(torch.float32)',
    'majority_diag = (bits[:, SPEC["measurement_wires"]].sum(dim=1) <= '
    'len(SPEC["measurement_wires"]) // 2).to(torch.float32)',
)
source = source.replace('(1, 3, 5)', 'ODD_WEIGHTS')
source = source.replace(
    'len(SEEDS) * N_UPDATES * 3',
    'len(SEEDS) * N_UPDATES * len(ODD_WEIGHTS)',
)
source = source.replace('len(SEEDS) * N_UPDATES * 16', 'len(SEEDS) * N_UPDATES * len(PAULI_TABLE)')
source = source.replace('16-string batched VJP', '256-string batched VJP')

replace_once(
    '''    identity = torch.eye(len(PAULI_TABLE), dtype=string_objectives.dtype)
    string_grads = torch.autograd.grad(
        string_objectives,
        parameters,
        grad_outputs=identity,
        is_grads_batched=True,
    )
    string_vectors = torch.cat(
        [gradient.reshape(len(PAULI_TABLE), -1) for gradient in string_grads], dim=1
    )''',
    '''    weights_array = PAULI_TABLE.weight.to_numpy()
    sector_objectives = torch.stack([
        string_objectives[torch.tensor(weights_array == weight)].sum()
        for weight in ODD_WEIGHTS
    ])
    identity = torch.eye(len(ODD_WEIGHTS), dtype=sector_objectives.dtype)
    sector_grads = torch.autograd.grad(
        sector_objectives,
        parameters,
        grad_outputs=identity,
        is_grads_batched=True,
    )
    sector_vectors = torch.cat(
        [gradient.reshape(len(ODD_WEIGHTS), -1) for gradient in sector_grads], dim=1
    )''',
)
source = source.replace('"string_vectors": string_vectors,',
                        '"sector_vectors": sector_vectors,')
source = source.replace('attribution["string_vectors"].sum(dim=0)',
                        'attribution["sector_vectors"].sum(dim=0)')
source = source.replace('"Batched VJP strings:", attribution["string_vectors"].shape',
                        '"Exact sector VJPs:", attribution["sector_vectors"].shape')

replace_once(
    '''        strings_pre = attribution["string_vectors"].detach()
        reconstruction_pre = torch.linalg.vector_norm(
            full_pre - strings_pre.sum(dim=0)
        ) / torch.linalg.vector_norm(full_pre).clamp_min(EPS)''',
    '''        sectors_pre = attribution["sector_vectors"].detach()
        reconstruction_pre = torch.linalg.vector_norm(
            full_pre - sectors_pre.sum(dim=0)
        ) / torch.linalg.vector_norm(full_pre).clamp_min(EPS)''',
)
replace_once(
    '''        strings_used = strings_pre * clip_scale

        reconstruction_used = torch.linalg.vector_norm(
            full_used - strings_used.sum(dim=0)
        ) / torch.linalg.vector_norm(full_used).clamp_min(EPS)

        sector_vectors = torch.stack([
            strings_used[torch.tensor(weights_array == weight)].sum(dim=0)
            for weight in ODD_WEIGHTS
        ])''',
    '''        sector_vectors = sectors_pre * clip_scale

        reconstruction_used = torch.linalg.vector_norm(
            full_used - sector_vectors.sum(dim=0)
        ) / torch.linalg.vector_norm(full_used).clamp_min(EPS)''',
)

old_string_and_sector_rows = '''        string_norms = torch.linalg.vector_norm(strings_used, dim=1)
        for string_index, row in enumerate(PAULI_TABLE.itertuples(index=False)):
            string_vector = strings_used[string_index]
            alignment = (
                torch.dot(string_vector, full_used) / full_norm_sq.clamp_min(EPS)
            )
            string_rows.append({
                "seed": seed,
                "update": update,
                "episode_end": episode,
                "string_index": string_index,
                "weight": int(row.weight),
                "pauli_string": row.pauli_string,
                "coefficient": float(row.coefficient),
                "vector_norm": float(string_norms[string_index].item()),
                "mean_absolute_component": float(string_vector.abs().mean().item()),
                "signed_alignment_share": float(alignment.item()),
            })

        for sector_index, weight in enumerate(ODD_WEIGHTS):
            mask = torch.tensor(weights_array == weight)
            vector = sector_vectors[sector_index]
            incoherent = torch.sqrt(torch.sum(string_norms[mask] ** 2))
            sector_rows.append({
                "seed": seed,
                "update": update,
                "episode_end": episode,
                "weight": weight,
                "coherent_vector_norm": float(sector_norms[sector_index].item()),
                "incoherent_sector_gradient_norm": float(incoherent.item()),
                "mean_absolute_component": float(vector.abs().mean().item()),
                "signed_alignment_share": float(alignment_shares[sector_index].item()),
                "nonnegative_norm_share": float((sector_norms[sector_index] / norm_denom).item()),
            })'''
new_sector_rows = '''        for sector_index, weight in enumerate(ODD_WEIGHTS):
            vector = sector_vectors[sector_index]
            sector_rows.append({
                "seed": seed,
                "update": update,
                "episode_end": episode,
                "weight": weight,
                "coherent_vector_norm": float(sector_norms[sector_index].item()),
                "incoherent_sector_gradient_norm": float("nan"),
                "mean_absolute_component": float(vector.abs().mean().item()),
                "signed_alignment_share": float(alignment_shares[sector_index].item()),
                "nonnegative_norm_share": float((sector_norms[sector_index] / norm_denom).item()),
            })'''
replace_once(old_string_and_sector_rows, new_sector_rows)
source = source.replace(
    '        string_vector_blocks.append(strings_used.cpu().numpy().astype(np.float32))\n',
    '',
)
replace_once(
    '''        "episode_metrics.csv", "update_metrics.csv", "sector_metrics.csv",
        "string_metrics.csv", "step_metrics.parquet",
        "pauli_string_gradient_vectors.parquet",
        "sector_gradient_vectors.parquet", "final_state.pt",''',
    '''        "episode_metrics.csv", "update_metrics.csv", "sector_metrics.csv",
        "step_metrics.parquet", "sector_gradient_vectors.parquet", "final_state.pt",''',
)
replace_once(
    '''    string_df = pd.DataFrame(string_rows)
    string_df["episode_bin"] = ((string_df.episode_end - 1) // BIN_WIDTH + 1).astype(int)
''',
    '',
)
source = source.replace(
    '    string_df.to_csv(seed_dir / "string_metrics.csv", index=False)\n',
    '',
)
old_vector_writes = '''    string_vectors = np.stack(string_vector_blocks)
    sector_vectors = np.stack(sector_vector_blocks)
    string_item_table = PAULI_TABLE[[
        "string_index", "weight", "pauli_string", "coefficient"
    ]]
    write_vector_parquet(
        seed_dir / "pauli_string_gradient_vectors.parquet",
        string_vectors, np.asarray(update_numbers), string_item_table, "string_index"
    )
    sector_item_table = pd.DataFrame({
        "sector_index": [0, 1, 2], "weight": [1, 3, 5]
    })'''
new_vector_writes = '''    sector_vectors = np.stack(sector_vector_blocks)
    sector_item_table = pd.DataFrame({
        "sector_index": np.arange(len(ODD_WEIGHTS)), "weight": list(ODD_WEIGHTS)
    })'''
replace_once(old_vector_writes, new_vector_writes)

source = source.replace('string_all = []\n', '')
source = source.replace('    string_all.append(pd.read_csv(seed_dir / "string_metrics.csv"))\n', '')
replace_once(
    '''    for name in (
        "step_metrics.parquet", "pauli_string_gradient_vectors.parquet",
        "sector_gradient_vectors.parquet",
    ):''',
    '''    for name in (
        "step_metrics.parquet", "sector_gradient_vectors.parquet",
    ):''',
)
source = source.replace('string_df = pd.concat(string_all, ignore_index=True)\n', '')
source = source.replace(
    'string_df.to_parquet(RAW_DIR / "string_metrics.parquet", index=False, compression="zstd")\n',
    '',
)
source = source.replace(
    'assert len(string_df) == len(SEEDS) * N_UPDATES * len(PAULI_TABLE)\n',
    '',
)
source = source.replace(', "strings": len(string_df)', ', "strings": 0')
source = source.replace('"n_string_rows": len(string_df),', '"n_string_rows": 0,')

replace_once(
    'the primary signed alignment share, which sums to one within each nonzero\n'
    'update and may be negative.  The bottom panel shows the incoherent sector\n'
    'gradient norm',
    'the primary signed alignment share, which sums to one within each nonzero\n'
    'update and may be negative.  The bottom panel shows the coherent sector\n'
    'gradient norm',
)
replace_once(
    'G_w^{\\mathrm{inc}}=\\sqrt{\\sum_{S:\\,|S|=w}\\lVert g_S\\rVert_2^2}',
    'G_w^{\\mathrm{coh}}=\\left\\lVert\\sum_{S:\\,|S|=w} g_S\\right\\rVert_2',
)
replace_once(
    'on a logarithmic scale.  This is a root-sum-square over string-gradient norms,\n'
    'not a per-string RMS.',
    'on a logarithmic scale.  For computational tractability at nine measured\n'
    'bits, the 256 Pauli strings are summed into their five exact weight sectors\n'
    'before differentiation; per-string incoherent norms are therefore not reported.',
)
source = source.replace(
    'plot_metric("incoherent_sector_gradient_norm", ax_norm, log_scale=True)',
    'plot_metric("coherent_vector_norm", ax_norm, log_scale=True)',
)
source = source.replace(
    'ax_norm.set_ylabel("Incoherent sector gradient norm")',
    'ax_norm.set_ylabel("Coherent sector gradient norm")',
)
source = source.replace(
    '                   "incoherent_sector_gradient_norm"):',
    '                   "coherent_vector_norm"):',
)
source = source.replace(
    'rows.incoherent_sector_gradient_norm_mean.to_numpy()',
    'rows.coherent_vector_norm_mean.to_numpy()',
)
source = source.replace(
    'rows.incoherent_sector_gradient_norm_sem.to_numpy()',
    'rows.coherent_vector_norm_sem.to_numpy()',
)
source = source.replace(
    'ax_rms.set_ylabel("sector-gradient RSS norm", fontsize=8)',
    'ax_rms.set_ylabel("coherent sector-gradient norm", fontsize=8)',
)
source = source.replace(
    'ax_rms.set_title("(C) Incoherent sector-gradient norm (log scale)")',
    'ax_rms.set_title("(C) Coherent sector-gradient norm (log scale)")',
)
source = source.replace('    "incoherent_sector_gradient_norm",\n', '')

replace_once(
    '''fig = plt.figure(figsize=(7.2, 6.1))
grid = fig.add_gridspec(3, 1, height_ratios=[1.05, 1.15, 1.0], hspace=0.52)''',
    '''fig = plt.figure(figsize=(7.2, 4.8))
grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15], hspace=0.38)''',
)

old_alignment_panel = '''ax_align = fig.add_subplot(grid[1, 0], sharex=ax_return)
centers = np.arange(25, 501, 50)
bar_width = 12
offsets = {1: -bar_width, 3: 0, 5: bar_width}
for weight in ODD_WEIGHTS:
    rows = bin_summary_df[bin_summary_df.weight == weight].sort_values("episode_bin")
    ax_align.bar(
        centers + offsets[weight], rows.signed_alignment_share_mean,
        width=bar_width * 0.9, color=colors[weight], alpha=0.88,
        yerr=rows.signed_alignment_share_sem, capsize=2,
        label=f"weight {weight}", linewidth=0,
    )
ax_align.axhline(0, color="0.25", linewidth=0.8)
for boundary in range(50, 500, 50):
    ax_align.axvline(boundary, color="0.88", linewidth=0.6)
ax_align.set_ylabel("signed alignment share")
ax_align.set_title("(B) Exact contribution to the optimizer-gradient direction")
ax_align.legend(ncol=3, frameon=False, loc="upper center")
ax_align.grid(axis="y", alpha=0.2)

'''
replace_once(old_alignment_panel, 'centers = np.arange(25, 501, 50)\n\n')

replace_once(
    'ax_rms = fig.add_subplot(grid[2, 0], sharex=ax_return)',
    'ax_rms = fig.add_subplot(grid[1, 0], sharex=ax_return)',
)
replace_once(
    '''    y = rows.coherent_vector_norm_mean.to_numpy()
    yerr = rows.coherent_vector_norm_sem.to_numpy()''',
    '''    y = rows.nonnegative_norm_share_mean.to_numpy()
    yerr = rows.nonnegative_norm_share_sem.to_numpy()''',
)
source = source.replace('ax_rms.set_yscale("log")\n', '')
replace_once(
    'ax_rms.set_ylabel("coherent sector-gradient norm", fontsize=8)',
    'ax_rms.set_ylabel("mean gradient-norm share")',
)
replace_once(
    'ax_rms.set_title("(C) Coherent sector-gradient norm (log scale)")',
    'ax_rms.set_title("(B) Relative gradient magnitude by Pauli locality")',
)
replace_once(
    'ax_rms.grid(which="both", axis="y", alpha=0.2)',
    'ax_rms.set_ylim(0, 0.82)\nax_rms.legend(ncol=5, frameon=False, loc="upper center")\n'
    'ax_rms.grid(axis="y", alpha=0.2)',
)
replace_once(
    '''fig.suptitle(
    "Five-qubit majority policy: Pauli-gradient spectrum during training\\n"
    "weights 2 and 4 vanish exactly; shading/error bars show SE across 10 seeds",
    y=0.995, fontsize=11,
)''',
    '''fig.suptitle(
    "Ten-qubit majority policy: gradient participation by locality\\n"
    "9 measured bits; lines show seed means and shading shows SE",
    y=0.995, fontsize=11,
)''',
)
replace_once(
    'fig.subplots_adjust(top=0.86, bottom=0.11, left=0.12, right=0.98)',
    'fig.subplots_adjust(top=0.82, bottom=0.14, left=0.12, right=0.98)',
)
replace_once(
    'FIGURES_DIR / f"majority_gradient_spectrum_during_training.{suffix}"',
    'FIGURES_DIR / f"majority_gradient_norm_share_during_training_10q.{suffix}"',
)
replace_once(
    '''The return panel uses the existing moving-20 convention.  The middle panel is
the primary signed alignment share, which sums to one within each nonzero
update and may be negative.  The bottom panel shows the coherent sector
gradient norm

\\[
G_w^{\\mathrm{coh}}=\\left\\lVert\\sum_{S:\\,|S|=w} g_S\\right\\rVert_2
\\]

on a logarithmic scale.  For computational tractability at nine measured
bits, the 256 Pauli strings are summed into their five exact weight sectors
before differentiation; per-string incoherent norms are therefore not reported.  Error bars and bands are standard errors across ten
independent seeds after within-seed binning.''',
    '''The top panel uses the existing moving-20 return convention.  The bottom
panel reports, in 50-episode bins, the nonnegative coherent gradient-norm
share for each Pauli locality,

\\[
q_w=\\frac{\\lVert g_w\\rVert_2}{\\sum_{v\\in\\{1,3,5,7,9\\}}\\lVert g_v\\rVert_2}.
\\]

The shares sum to one within every nonzero update, so the plot directly shows
which locality sectors gain or lose relative gradient magnitude during
training.  For computational tractability at nine measured bits, the 256
Pauli strings are summed into their five exact weight sectors before
differentiation; per-string incoherent norms are not reported.  Lines show
across-seed means after within-seed binning and shading shows standard errors.''',
)
replace_once(
    '"attribution_scope": "quantum policy parameters only",',
    '"attribution_scope": "quantum policy parameters only; exact coherent weight-sector VJPs",\n'
    '    "per_string_incoherent_norms_reported": False,',
)

replace_once(
    'bootstrap_df.to_csv(RESULTS_DIR / "paired_early_late_bootstrap.csv", index=False)',
    '''bootstrap_df.to_csv(RESULTS_DIR / "paired_early_late_bootstrap.csv", index=False)

# Paired aggregate of every higher-weight sector (w >= 3).  The signed shares
# add algebraically; the nonnegative norm shares add by construction.
combined_rows = []
for metric in ("signed_alignment_share", "nonnegative_norm_share"):
    higher = seed_bin_sector[seed_bin_sector.weight >= 3]
    early = higher[higher.episode_bin == 1].groupby("seed")[metric].sum()
    late = higher[higher.episode_bin == 10].groupby("seed")[metric].sum()
    assert early.index.tolist() == late.index.tolist() == list(SEEDS)
    differences = late.to_numpy() - early.to_numpy()
    mean, low, high = bootstrap_mean_interval(differences, rng)
    combined_rows.append({
        "comparison": "episodes_1_50_vs_451_500",
        "sector": "combined_weights_3_5_7_9",
        "metric": metric,
        "n_pairs": len(differences),
        "early_mean": float(early.mean()),
        "late_mean": float(late.mean()),
        "mean_late_minus_early": mean,
        "ci95_low": low,
        "ci95_high": high,
    })
combined_higher_df = pd.DataFrame(combined_rows)
combined_higher_df.to_csv(
    RESULTS_DIR / "paired_combined_higher_weight_bootstrap.csv", index=False
)''',
)

replace_once(
    'colors = {1: "#4C78A8", 3: "#F58518", 5: "#E45756"}',
    'colors = {1: "#4C78A8", 3: "#F58518", 5: "#E45756", '
    '7: "#72B7B2", 9: "#B279A2"}',
)
source = source.replace(
    'bar_width = 12\noffsets = {1: -bar_width, 3: 0, 5: bar_width}',
    'bar_width = 7.5\noffsets = {weight: (index - 2) * bar_width '
    'for index, weight in enumerate(ODD_WEIGHTS)}',
)
source = source.replace('ax_align.legend(ncol=3, frameon=False, loc="upper center")',
                        'ax_align.legend(ncol=5, frameon=False, loc="upper center")')
source = source.replace(
    '"Five-qubit majority policy: Pauli-gradient spectrum during training\\n"\n'
    '    "weights 2 and 4 vanish exactly; shading/error bars show SE across 10 seeds"',
    '"Ten-qubit majority policy: Pauli-gradient spectrum during training\\n"\n'
    '    "9 measured bits; even weights vanish; shading/error bars show SE across 10 seeds"',
)
source = source.replace(
    'FIGURES_DIR / f"majority_gradient_spectrum_during_training.{suffix}"',
    'FIGURES_DIR / f"majority_gradient_spectrum_during_training_10q.{suffix}"',
)
replace_once(
    '"measurement_effects": "Pi0=(I+M5)/2, Pi1=(I-M5)/2",',
    '"measurement_effects": "Pi0=(I+M9)/2, Pi1=(I-M9)/2",',
)
replace_once(
    '"pauli_weights_present": [1, 3, 5],\n'
    '    "pauli_weights_absent": [2, 4],',
    '"pauli_weights_present": list(ODD_WEIGHTS),\n'
    '    "pauli_weights_absent": [2, 4, 6, 8],',
)

old_conclusion = '''## Scientific conclusion

The corrected direct optimizer-gradient attribution does **not** support a
blanket claim that all higher-weight sectors become more prominent late in
training.  Between episodes 1--50 and 451--500, the mean signed alignment share
of weight 1 rises from approximately 0.815 to 0.928, while weight 3 falls from
approximately 0.129 to 0.039 and weight 5 changes from approximately 0.057 to
0.034.  The absolute incoherent sector-gradient norm declines in every sector.

The norm-share diagnostic is more nuanced: weight 3 increases from
approximately 0.226 to 0.265, weight 5 decreases from approximately 0.173 to
0.149, and their combined share changes from approximately 0.399 to 0.414.
That combined change is not statistically resolved by the ten matched seeds.
Late gradients are therefore more aligned with the singleton direction, but
higher-weight directions still account for about 41% of the summed sector
norms.  These are optimizer-gradient diagnostics, not a counterfactual local
policy and not a classical-simulability test.  The next-bin correlations are
descriptive only and do not establish causality.'''
new_conclusion = '''## Scientific conclusion

For the ten-qubit circuit (nine measured bits), the higher-weight sectors become
**larger in nonnegative norm share** but **less aligned with the actual update**.
From episodes 1--50 to 451--500, the combined weights 3, 5, 7, and 9 rise from
approximately 0.292 to 0.436 of the summed coherent sector norms (paired change
0.145, 95% bootstrap CI [0.115, 0.172]).  Their combined signed alignment falls
from approximately 0.055 to -0.042 (change -0.097, CI [-0.161, -0.038]).

Thus the singleton sector remains the direction most aligned with the optimizer
gradient, while the higher-weight sector gradients grow in relative magnitude
and increasingly cancel or oppose one another.  Signed alignment and
nonnegative norm share answer different questions and must be reported
separately.

This experiment diagnoses the optimizer gradient of the **full majority
policy** along its own training trajectory.  It is not a counterfactual
local-only policy and not a classical-simulability test.  The next-bin
correlations are descriptive only and do not establish causality.'''
replace_once(old_conclusion, new_conclusion)

replace_once('"supports_blanket_late_higher_weight_prominence": False,',
             '"supports_blanket_late_higher_weight_prominence": False,')
replace_once(
    '"Weight-1 signed alignment increases late and weight-3 signed "\n'
    '            "alignment decreases, while weight-3 norm share increases and "\n'
    '            "weight-5 norm share decreases. The combined higher-weight norm "\n'
    '            "share is statistically unresolved; all absolute sector-gradient "\n'
    '            "norms decrease."',
    '"Combined weights 3,5,7,9 increase from 0.292 to 0.436 in "\n'
    '            "nonnegative coherent norm share, but their signed alignment "\n'
    '            "falls from 0.055 to -0.042. Higher-weight magnitude and update "\n'
    '            "alignment therefore move in opposite directions."',
)

source = source.replace('"label": "5-qubit ring, L=3"',
                        '"label": "10-qubit connected cubic graph, L=1 (9 measured)"')

namespace = {"__name__": "__main__", "__file__": str(SOURCE_BUILDER)}
exec(compile(source, str(SOURCE_BUILDER), "exec"), namespace)
