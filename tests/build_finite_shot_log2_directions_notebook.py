from pathlib import Path
import nbformat as nbf


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "test_policy_training_efficiency_finite_shots.ipynb"
TARGET = ROOT / "test_policy_training_efficiency_finite_shots_log2_directions.ipynb"


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"Expected one occurrence, found {text.count(old)}: {old[:80]!r}")
    return text.replace(old, new, 1)


nb = nbf.read(SOURCE, as_version=4)
for cell in nb.cells:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None

nb.cells[0].source = r"""# Base-2 log-direction finite-shot SPSA for graph-IQP CartPole policies

This is a separate confirmatory study. It preserves the validated finite-shot
notebook and its results, reuses only the verified exact-analytic baselines,
and reruns the 4,096-shot condition with

\[
K=\left\lceil\log_2 p\right\rceil
\]

independent Rademacher directions per update, where \(p\) is the number of
trainable scalar policy weights. All labels use **CartPole-v1/200** because
every episode is explicitly capped at 200 steps.

The notebook computes \(p\) from the actual QASP state tensors and checks it
against the model and architecture formulas. This gives \(p=90\) for the
5-qubit \(L=3\) policy and \(p=65\) for the 10-qubit \(L=1\) policy, hence
\(K=7\) for both. The earlier 45/35 figures inferred from native
parameter-shift counts are effective parameterized gate arguments, not the
optimizer-vector dimension perturbed by SPSA.

Finite action probabilities come from fresh multinomial circuit counts.
Jeffreys smoothing is applied symmetrically:

\[
\widehat p_a=\frac{c_a+0.5}{N+1},\qquad N=4096.
\]

No exact-statevector probability or gradient is used in the finite-shot
losses. Exact Born probabilities are evaluated only as the categorical
distribution from which finite circuit counts are sampled.
"""

cell1 = nb.cells[1].source
cell1 = replace_once(
    cell1,
    'os.environ.setdefault("MPLCONFIGDIR", "/tmp/qasp-finite-shot-mpl")',
    'os.environ.setdefault("MPLCONFIGDIR", "/tmp/qasp-log2-directions-mpl")',
)
cell1 = replace_once(
    cell1,
    'RESULTS_DIR = ROOT / "tests" / "finite_shot_cartpole_results"',
    'BASELINE_RESULTS_DIR = ROOT / "tests" / "finite_shot_cartpole_results"\n'
    'RESULTS_DIR = ROOT / "tests" / "finite_shot_cartpole_log2_directions_results"',
)
old_conditions = '''CONDITIONS = (
    {"mode": "exact_analytic", "shots": None, "gradient_estimator": "analytic"},
    {"mode": "exact_spsa", "shots": None, "gradient_estimator": "spsa"},
    {"mode": "shots_1024", "shots": 1024, "gradient_estimator": "spsa"},
    {"mode": "shots_4096", "shots": 4096, "gradient_estimator": "spsa"},
)
PILOT_SHOTS = (256, 1024, 4096)'''
new_conditions = '''CONDITIONS = (
    {"mode": "shots_4096_log2K", "shots": 4096, "gradient_estimator": "spsa"},
)'''
cell1 = replace_once(cell1, old_conditions, new_conditions)
cell1 = replace_once(
    cell1,
    '    "spsa_directions": 2,',
    '    "spsa_directions_rule": "ceil(log2(total trainable scalar parameters))",',
)
cell1 = replace_once(
    cell1,
    'print("Results:", RESULTS_DIR)',
    'print("Preserved baseline results:", BASELINE_RESULTS_DIR)\nprint("New results:", RESULTS_DIR)',
)
nb.cells[1].source = cell1

nb.cells[2].source += r"""

# Compute the literal optimizer dimension and K from the actual QASP tensors.
parameter_audit_df = pd.DataFrame(initialization_audit)
baseline_pilot_df = pd.read_csv(BASELINE_RESULTS_DIR / "pilot_results.csv")
parameter_rows = []
for architecture, spec in ARCHITECTURES.items():
    rows = parameter_audit_df[parameter_audit_df.architecture == architecture]
    p_values = (rows.scaling_entries + rows.circuit_entries).unique()
    assert len(p_values) == 1
    p = int(p_values[0])
    formula_p = (
        spec["n_layers"] * spec["n_qubits"] * 4
        + spec["n_layers"] * (spec["n_qubits"] + len(spec["edges"]))
    )
    assert p == formula_p
    K = int(math.ceil(math.log2(p)))
    pilot_row = baseline_pilot_df[
        (baseline_pilot_df.architecture == architecture)
        & (baseline_pilot_df.shots == 4096)
    ].iloc[0]
    gate_arguments = int(
        (pilot_row.native_parameter_shift_circuit_executions - 1) // 2
    )
    assert int(pilot_row.native_parameter_shift_circuit_executions) == 2 * gate_arguments + 1
    spec["parameter_count"] = p
    spec["spsa_directions"] = K
    parameter_rows.append({
        "architecture": architecture,
        "scaling_weights": int(rows.scaling_entries.iloc[0]),
        "circuit_weights": int(rows.circuit_entries.iloc[0]),
        "trainable_scalar_parameters_p": p,
        "effective_gate_arguments": gate_arguments,
        "native_parameter_shift_executions": int(
            pilot_row.native_parameter_shift_circuit_executions
        ),
        "K_ceil_log2_p": K,
    })
parameter_count_df = pd.DataFrame(parameter_rows)
assert parameter_count_df.K_ceil_log2_p.eq(
    np.ceil(np.log2(parameter_count_df.trainable_scalar_parameters_p)).astype(int)
).all()
display(parameter_count_df)

# Read-only integrity and protocol checks for the preserved baseline study.
BASELINE_FILE_HASHES = {
    "experiment_config.json": "aa45b82af8b62855329fd68a8df97975c51b702b076f1c3a27db906da7d9cbe2",
    "run_metadata.csv": "4bbfc6c4b4f939ef2ac72e5fc888e1f26f4b9aaa89ae231df62b0573cec4fd8d",
    "episode_returns.csv": "3a2bd4ea1c4b8dd46f8a98d6a874ce04cc1161dd82cc2b8a721fa3321601fca4",
}
for filename, expected in BASELINE_FILE_HASHES.items():
    actual = hashlib.sha256((BASELINE_RESULTS_DIR / filename).read_bytes()).hexdigest()
    assert actual == expected, (filename, actual, expected)

with open(BASELINE_RESULTS_DIR / "experiment_config.json") as handle:
    baseline_config = json.load(handle)
baseline_metadata_all = pd.read_csv(BASELINE_RESULTS_DIR / "run_metadata.csv")
baseline_returns_all = pd.read_csv(BASELINE_RESULTS_DIR / "episode_returns.csv")

assert baseline_config["environment_label"] == "CartPole-v1/200"
assert baseline_config["gymnasium_environment_id"] == "CartPole-v1"
assert baseline_config["episodes"] == N_EPISODES == 500
assert baseline_config["max_steps"] == MAX_STEPS == 200
assert baseline_config["terminal_metric"] == "mean return over final 50 episodes"
assert baseline_config["solved_metric"] == "first full 100-episode rolling mean >= 195.0"

protocol_keys = (
    "gamma", "lr", "lr_schedule", "optimizer", "batch_episodes",
    "normalize_returns", "baseline_type", "baseline_momentum",
    "entropy_coef", "max_grad_norm", "spsa_c0", "spsa_gamma",
    "pseudocount_alpha",
)
for key in protocol_keys:
    assert baseline_config["training"][key] == TRAINING[key]

for architecture, spec in ARCHITECTURES.items():
    old = baseline_config["architectures"][architecture]
    assert old["n_qubits"] == spec["n_qubits"]
    assert old["n_layers"] == spec["n_layers"]
    assert [tuple(edge) for edge in old["edges"]] == [tuple(edge) for edge in spec["edges"]]
    assert old["measurement_wires"] == spec["measurement_wires"]
    assert old["initializer"] == spec["initializer"]
    assert old["seeds"] == list(spec["seeds"])

exact_baseline_metadata = baseline_metadata_all[
    baseline_metadata_all["mode"] == "exact_analytic"
].copy()
exact_baseline_returns = baseline_returns_all[
    baseline_returns_all["mode"] == "exact_analytic"
].copy()
assert len(exact_baseline_metadata) == 20
assert len(exact_baseline_returns) == 20 * N_EPISODES
assert exact_baseline_metadata.gradient_estimator.eq("analytic").all()
assert exact_baseline_metadata.shots_per_evaluation.eq(0).all()

hash_lookup = parameter_audit_df.set_index(["architecture", "seed"])["sha256"]
for architecture, spec in ARCHITECTURES.items():
    for seed in spec["seeds"]:
        expected_hash = hash_lookup.loc[(architecture, seed)]
        rows = exact_baseline_metadata[
            (exact_baseline_metadata.architecture == architecture)
            & (exact_baseline_metadata.seed == seed)
        ]
        assert len(rows) == 2
        assert rows.initial_parameter_sha256.nunique() == 1
        assert rows.initial_parameter_sha256.iloc[0] == expected_hash
"""

cell4 = nb.cells[4].source
cell4 = replace_once(
    cell4,
    '        self.params = torch.nn.Parameter(initial_state["params"].clone())',
    '''        self.params = torch.nn.Parameter(initial_state["params"].clone())
        self.parameter_count = sum(
            tensor.numel() for tensor in self.parameters() if tensor.requires_grad
        )
        formula_p = (
            spec["n_layers"] * spec["n_qubits"] * 4
            + spec["n_layers"] * (spec["n_qubits"] + len(spec["edges"]))
        )
        assert self.parameter_count == formula_p == spec["parameter_count"]
        self.spsa_directions = int(math.ceil(math.log2(self.parameter_count)))
        assert self.spsa_directions == spec["spsa_directions"]''',
)
nb.cells[4].source = cell4

cell5 = nb.cells[5].source
cell5 = replace_once(
    cell5,
    "'''Same CartPole REINFORCE objective, with exact backprop or finite-shot SPSA.''",
    "'''Matched CartPole REINFORCE with finite-shot log-direction SPSA.''",
)
cell5 = replace_once(
    cell5,
    '        directions = TRAINING["spsa_directions"]',
    '        directions = self.policy.spsa_directions',
)
cell5 = replace_once(
    cell5,
    '            "strict_solved_episode": strict_solved_episode(returns),',
    '''            "strict_solved_episode": strict_solved_episode(returns),
            "parameter_count": self.policy.parameter_count,
            "spsa_directions": self.policy.spsa_directions,
            "independent_plus_minus_shot_streams": True,''',
)
nb.cells[5].source = cell5

nb.cells[6].source = r"""## Verified reuse of exact-analytic baselines

The exact-analytic trajectories below are copied read-only from the preserved
validated study. Reuse is allowed only after checking file hashes, architecture,
seeds, initial-parameter hashes, preprocessing/training protocol, checkpoint
contents, final-50 values, and strict solved episodes. No analytic run is
retrained here.
"""

nb.cells[7].source = r"""# The preprocessing code is inherited unchanged from the validated notebook.
assert torch.equal(
    preprocess_cartpole(torch.tensor([2.4, 3.0, 0.2095, 3.5])),
    torch.ones(4),
)
assert torch.equal(
    preprocess_cartpole(torch.tensor([24.0, -30.0, 2.095, -35.0])),
    torch.tensor([3.0, -3.0, 3.0, -3.0]),
)

baseline_validation_rows = []
for row in exact_baseline_metadata.itertuples(index=False):
    checkpoint_path = BASELINE_RESULTS_DIR / "runs" / f"{row.run_id}.npz"
    assert checkpoint_path.exists()
    with np.load(checkpoint_path) as checkpoint:
        checkpoint_returns = checkpoint["returns"].astype(float)
        assert str(checkpoint["architecture"]) == row.architecture
        assert str(checkpoint["policy"]) == row.policy
        assert str(checkpoint["mode"]) == "exact_analytic"
        assert int(checkpoint["seed"]) == int(row.seed)
    csv_returns = exact_baseline_returns[
        (exact_baseline_returns.architecture == row.architecture)
        & (exact_baseline_returns.policy == row.policy)
        & (exact_baseline_returns.seed == row.seed)
    ].sort_values("episode")["return"].to_numpy(float)
    assert len(checkpoint_returns) == len(csv_returns) == N_EPISODES
    np.testing.assert_array_equal(checkpoint_returns, csv_returns)
    assert float(checkpoint_returns[-FINAL_WINDOW:].mean()) == float(row.final_50)
    solved = strict_solved_episode(checkpoint_returns)
    if np.isnan(row.strict_solved_episode):
        assert np.isnan(solved)
    else:
        assert solved == row.strict_solved_episode
    baseline_validation_rows.append({
        "architecture": row.architecture,
        "policy": row.policy,
        "seed": row.seed,
        "initial_parameter_sha256": row.initial_parameter_sha256,
        "final_50": row.final_50,
        "strict_solved_episode": row.strict_solved_episode,
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    })

baseline_validation_df = pd.DataFrame(baseline_validation_rows)
baseline_validation_df.to_csv(RESULTS_DIR / "reused_exact_analytic_metadata.csv", index=False)
exact_baseline_returns.to_csv(
    RESULTS_DIR / "reused_exact_analytic_episode_returns.csv", index=False
)
display(baseline_validation_df.head())
print("Verified and copied exact-analytic runs:", len(baseline_validation_df))
"""

nb.cells[8].source = r"""## Confirmatory 4,096-shot experiment with \(K=\lceil\log_2 p\rceil\)

For every update, each of the \(K\) directions uses a fresh Rademacher vector
and two independent 4,096-shot multinomial evaluations at \(+c_k\Delta\) and
\(-c_k\Delta\). The sampler's generator advances continuously and is never
reset between signs or directions. The only newly trained condition is
4,096-shot SPSA with the dynamically computed \(K\).
"""

nb.cells[9].source = r"""all_returns = []
run_metadata = []
existing_metadata_path = RESULTS_DIR / "run_metadata.csv"
existing_metadata_df = (
    pd.read_csv(existing_metadata_path)
    if existing_metadata_path.exists()
    else pd.DataFrame()
)

for architecture, spec in ARCHITECTURES.items():
    for seed in spec["seeds"]:
        initial_state = qasp_initial_state(spec, seed)
        expected_hash = hashlib.sha256(
            torch.cat([initial_state["input_scaling_params"].flatten(),
                       initial_state["params"].flatten()]).numpy().tobytes()
        ).hexdigest()
        baseline_hashes = exact_baseline_metadata[
            (exact_baseline_metadata.architecture == architecture)
            & (exact_baseline_metadata.seed == seed)
        ].initial_parameter_sha256.unique()
        assert baseline_hashes.tolist() == [expected_hash]

        condition = CONDITIONS[0]
        mode = condition["mode"]
        shots = condition["shots"]
        gradient_estimator = condition["gradient_estimator"]
        shot_seed = 1_000_003 + 10_000 * seed + shots
        for policy_name in POLICIES:
            run_id = f"{architecture}__{policy_name}__{mode}__seed_{seed}"
            checkpoint_path = RUNS_DIR / f"{run_id}.npz"
            previous = (
                existing_metadata_df[existing_metadata_df.run_id == run_id]
                if not existing_metadata_df.empty
                else pd.DataFrame()
            )
            if checkpoint_path.exists() and len(previous) == 1:
                with np.load(checkpoint_path) as checkpoint:
                    returns = checkpoint["returns"].astype(float)
                if len(returns) == N_EPISODES:
                    print(f"CHECKPOINT {run_id}", flush=True)
                    for episode, value in enumerate(returns, start=1):
                        all_returns.append({
                            "architecture": architecture,
                            "architecture_label": spec["label"],
                            "n_qubits": spec["n_qubits"],
                            "depth": spec["n_layers"],
                            "policy": policy_name,
                            "mode": mode,
                            "shots_per_evaluation": shots,
                            "parameter_count": spec["parameter_count"],
                            "spsa_directions": spec["spsa_directions"],
                            "seed": seed,
                            "episode": episode,
                            "return": value,
                        })
                    run_metadata.append(previous.iloc[0].to_dict())
                    continue

            policy = IQPExperimentPolicy(
                spec, policy_name, initial_state, shots, shot_seed
            )
            assert policy.flat_parameters_numpy().size == spec["parameter_count"]
            assert policy.spsa_directions == math.ceil(
                math.log2(policy.flat_parameters_numpy().size)
            )
            actual_hash = hashlib.sha256(
                torch.cat([
                    policy.input_scaling_params.detach().flatten(),
                    policy.params.detach().flatten(),
                ]).cpu().numpy().tobytes()
            ).hexdigest()
            assert actual_hash == expected_hash

            print(
                f"\n=== {spec['label']} | {policy_name} | {mode} | seed={seed} "
                f"| p={policy.parameter_count} | K={policy.spsa_directions} ===",
                flush=True,
            )
            trainer = MatchedFiniteShotREINFORCE(
                policy, seed, gradient_estimator=gradient_estimator
            )
            returns, metadata = trainer.train()
            final_50 = float(returns[-FINAL_WINDOW:].mean())
            print(
                f"DONE final50={final_50:.2f} seconds={metadata['seconds']:.1f} "
                f"evals={metadata['total_circuit_evaluations']:,} "
                f"shots={metadata['total_circuit_shots']:,}",
                flush=True,
            )

            np.savez_compressed(
                checkpoint_path,
                returns=returns,
                architecture=architecture,
                policy=policy_name,
                mode=mode,
                shots=shots,
                seed=seed,
                parameter_count=policy.parameter_count,
                spsa_directions=policy.spsa_directions,
            )
            for episode, value in enumerate(returns, start=1):
                all_returns.append({
                    "architecture": architecture,
                    "architecture_label": spec["label"],
                    "n_qubits": spec["n_qubits"],
                    "depth": spec["n_layers"],
                    "policy": policy_name,
                    "mode": mode,
                    "shots_per_evaluation": shots,
                    "parameter_count": policy.parameter_count,
                    "spsa_directions": policy.spsa_directions,
                    "seed": seed,
                    "episode": episode,
                    "return": value,
                })
            run_metadata.append({
                "run_id": run_id,
                "architecture": architecture,
                "architecture_label": spec["label"],
                "n_qubits": spec["n_qubits"],
                "depth": spec["n_layers"],
                "policy": policy_name,
                "mode": mode,
                "gradient_estimator": gradient_estimator,
                "shots_per_evaluation": shots,
                "seed": seed,
                "initial_parameter_sha256": expected_hash,
                "final_50": final_50,
                **metadata,
            })
            del policy, trainer
            gc.collect()

new_returns_df = pd.DataFrame(all_returns)
new_run_metadata_df = pd.DataFrame(run_metadata)
assert len(new_returns_df) == 20 * N_EPISODES
assert len(new_run_metadata_df) == 20
assert new_run_metadata_df.spsa_directions.eq(7).all()
new_returns_df.to_csv(RESULTS_DIR / "episode_returns.csv", index=False)
new_run_metadata_df.to_csv(RESULTS_DIR / "run_metadata.csv", index=False)

# Import the preserved K=2 seed-level trajectories for publication plotting.
preserved_k2_returns = baseline_returns_all[
    baseline_returns_all["mode"].isin(["shots_1024", "shots_4096"])
].copy()
assert len(preserved_k2_returns) == 40 * N_EPISODES
assert preserved_k2_returns.groupby(
    ["architecture", "policy", "mode", "seed"]
).size().eq(N_EPISODES).all()
preserved_k2_returns.to_csv(
    RESULTS_DIR / "preserved_k2_1024_4096_episode_returns.csv", index=False
)

# Combined in-memory frame drives the four-condition publication plot only.
returns_df = pd.concat(
    [exact_baseline_returns, preserved_k2_returns, new_returns_df],
    ignore_index=True,
)
run_metadata_df = pd.concat(
    [exact_baseline_metadata, new_run_metadata_df], ignore_index=True, sort=False
)
print("New finite-shot runs:", len(new_run_metadata_df))
print("Reused exact-analytic runs:", len(exact_baseline_metadata))
print("Imported preserved K=2 trajectory rows:", len(preserved_k2_returns))
display(new_run_metadata_df.head())
"""

nb.cells[10].source = r"""def paired_bootstrap(values_a, values_b, rng, repetitions=20_000):
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("Paired samples must have equal length")
    if len(a) != 5:
        raise ValueError(f"Expected five matched seeds, received {len(a)}")
    difference = a - b
    indices = rng.integers(0, len(difference), size=(repetitions, len(difference)))
    boot = difference[indices].mean(axis=1)
    mean_difference = float(difference.mean())
    sd_difference = float(difference.std(ddof=1)) if len(difference) > 1 else np.nan
    effect_dz = mean_difference / sd_difference if sd_difference > 0 else np.nan
    return mean_difference, np.quantile(boot, [0.025, 0.975]), effect_dz


summary_rows = []
for keys, group in run_metadata_df.groupby(
    ["architecture", "architecture_label", "n_qubits", "depth", "policy", "mode", "shots_per_evaluation"],
    sort=False,
):
    architecture, label, n_qubits, depth, policy, mode, shots = keys
    final = group["final_50"].to_numpy(float)
    summary_rows.append({
        "architecture": architecture,
        "architecture_label": label,
        "n_qubits": n_qubits,
        "depth": depth,
        "policy": policy,
        "mode": mode,
        "shots_per_evaluation": shots,
        "n_seeds": len(group),
        "parameter_count": ARCHITECTURES[architecture]["parameter_count"],
        "spsa_directions": (
            0 if mode == "exact_analytic" else ARCHITECTURES[architecture]["spsa_directions"]
        ),
        "mean_final_50": final.mean(),
        "sem_final_50": final.std(ddof=1) / np.sqrt(len(final)),
        "solved_seed_fraction": np.isfinite(group["strict_solved_episode"]).mean(),
        "mean_runtime_seconds": group["seconds"].mean(),
        "total_circuit_evaluations": int(group["total_circuit_evaluations"].sum()),
        "total_circuit_shots": int(group["total_circuit_shots"].sum()),
    })
summary_df = pd.DataFrame(summary_rows)
summary_df["condition_label"] = np.where(
    summary_df["mode"].eq("exact_analytic"),
    "exact + analytic gradient (K=N/A)",
    "4,096 shots + SPSA (K=7)",
)

preserved_k2_metadata = baseline_metadata_all[
    baseline_metadata_all["mode"].isin(["shots_1024", "shots_4096"])
].copy()
assert len(preserved_k2_metadata) == 40
preserved_k2_metadata["spsa_directions"] = 2
preserved_k2_metadata.to_csv(
    RESULTS_DIR / "preserved_k2_1024_4096_metadata.csv", index=False
)

preserved_summary_rows = []
for keys, group in preserved_k2_metadata.groupby(
    ["architecture", "architecture_label", "n_qubits", "depth", "policy", "mode", "shots_per_evaluation"],
    sort=False,
):
    architecture, label, n_qubits, depth, policy, mode, shots = keys
    final = group["final_50"].to_numpy(float)
    preserved_summary_rows.append({
        "architecture": architecture,
        "architecture_label": label,
        "n_qubits": n_qubits,
        "depth": depth,
        "policy": policy,
        "mode": mode,
        "shots_per_evaluation": shots,
        "n_seeds": len(group),
        "parameter_count": ARCHITECTURES[architecture]["parameter_count"],
        "spsa_directions": 2,
        "mean_final_50": final.mean(),
        "sem_final_50": final.std(ddof=1) / np.sqrt(len(final)),
        "solved_seed_fraction": np.isfinite(group["strict_solved_episode"]).mean(),
        "mean_runtime_seconds": group["seconds"].mean(),
        "total_circuit_evaluations": int(group["total_circuit_evaluations"].sum()),
        "total_circuit_shots": int(group["total_circuit_shots"].sum()),
        "condition_label": f"{int(shots):,} shots + SPSA (K=2)",
    })
preserved_k2_summary_df = pd.DataFrame(preserved_summary_rows)
preserved_k2_summary_df.to_csv(
    RESULTS_DIR / "preserved_k2_1024_4096_summary_metrics.csv", index=False
)
summary_df = pd.concat([summary_df, preserved_k2_summary_df], ignore_index=True)

previous_k2_metadata = preserved_k2_metadata[
    preserved_k2_metadata["mode"] == "shots_4096"
].copy()
assert len(previous_k2_metadata) == 20
previous_k2_metadata.to_csv(
    RESULTS_DIR / "previous_k2_4096_metadata.csv", index=False
)

rng = np.random.default_rng(20260917)
comparison_rows = []

def paired_rows(left, right, family, architecture, label, contrast):
    left = left.sort_values("seed")
    right = right.sort_values("seed")
    assert left.seed.tolist() == right.seed.tolist()
    diff, ci, dz = paired_bootstrap(left.final_50, right.final_50, rng)
    comparison_rows.append({
        "comparison_family": family,
        "architecture": architecture,
        "policy_or_mode": label,
        "contrast": contrast,
        "n_pairs": len(left),
        "mean_paired_difference_final_50": diff,
        "ci95_low": ci[0],
        "ci95_high": ci[1],
        "paired_effect_size_dz": dz,
    })

for architecture in ARCHITECTURES:
    for policy_name in POLICIES:
        new = new_run_metadata_df[
            (new_run_metadata_df.architecture == architecture)
            & (new_run_metadata_df.policy == policy_name)
        ]
        exact = exact_baseline_metadata[
            (exact_baseline_metadata.architecture == architecture)
            & (exact_baseline_metadata.policy == policy_name)
        ]
        old_k2 = previous_k2_metadata[
            (previous_k2_metadata.architecture == architecture)
            & (previous_k2_metadata.policy == policy_name)
        ]
        paired_rows(
            new, exact, "log2K_minus_exact_analytic", architecture, policy_name,
            "shots_4096_log2K - exact_analytic",
        )
        paired_rows(
            new, old_k2, "log2K_minus_previous_K2", architecture, policy_name,
            "shots_4096_log2K - shots_4096_K2",
        )

# Preserved K=2 comparison: increasing shots from 1,024 to 4,096.
for architecture in ARCHITECTURES:
    for policy_name in POLICIES:
        old_1024 = preserved_k2_metadata[
            (preserved_k2_metadata.architecture == architecture)
            & (preserved_k2_metadata.policy == policy_name)
            & (preserved_k2_metadata["mode"] == "shots_1024")
        ]
        old_4096 = preserved_k2_metadata[
            (preserved_k2_metadata.architecture == architecture)
            & (preserved_k2_metadata.policy == policy_name)
            & (preserved_k2_metadata["mode"] == "shots_4096")
        ]
        paired_rows(
            old_4096, old_1024, "K2_4096_minus_K2_1024", architecture,
            policy_name, "shots_4096_K2 - shots_1024_K2",
        )

for architecture in ARCHITECTURES:
    for mode, frame in (
        ("exact_analytic", exact_baseline_metadata),
        ("shots_4096_log2K", new_run_metadata_df),
    ):
        parity = frame[
            (frame.architecture == architecture) & (frame.policy == "parity")
        ]
        majority = frame[
            (frame.architecture == architecture) & (frame.policy == "majority")
        ]
        paired_rows(
            majority, parity, "majority_minus_parity", architecture, mode,
            "majority - parity",
        )

comparisons_df = pd.DataFrame(comparison_rows)
assert len(summary_df) == 16
assert len(comparisons_df) == 16
assert comparisons_df.n_pairs.eq(5).all()
assert comparisons_df[[
    "mean_paired_difference_final_50", "ci95_low", "ci95_high"
]].notna().all().all()
summary_df.to_csv(RESULTS_DIR / "summary_metrics.csv", index=False)
comparisons_df.to_csv(RESULTS_DIR / "paired_comparisons.csv", index=False)
display(summary_df.round(3))
display(comparisons_df.round(3))
"""

cell11 = nb.cells[11].source
old_mode_block = '''mode_order = ["exact_analytic", "exact_spsa", "shots_1024", "shots_4096"]
mode_labels = {
    "exact_analytic": "exact + analytic gradient",
    "exact_spsa": "exact + SPSA",
    "shots_1024": "1,024 shots + SPSA",
    "shots_4096": "4,096 shots + SPSA",
}
terminal_mode_labels = {
    "exact_analytic": "exact +\\nanalytic",
    "exact_spsa": "exact +\\nSPSA",
    "shots_1024": "1,024 shots +\\nSPSA",
    "shots_4096": "4,096 shots +\\nSPSA",
}
mode_colors = {
    "exact_analytic": "#222222",
    "exact_spsa": "#6A3D9A",
    "shots_1024": "#2878B5",
    "shots_4096": "#D95319",
}'''
new_mode_block = '''learning_mode_order = [
    "exact_analytic", "shots_1024", "shots_4096", "shots_4096_log2K"
]
terminal_mode_order = ["exact_analytic", "shots_4096_log2K"]
mode_labels = {
    "exact_analytic": "exact + analytic gradient (K=N/A)",
    "shots_1024": "1,024 shots + SPSA (K=2)",
    "shots_4096": "4,096 shots + SPSA (K=2)",
    "shots_4096_log2K": "4,096 shots + SPSA (K=7)",
}
terminal_mode_labels = {
    "exact_analytic": "exact + analytic\\n(K=N/A)",
    "shots_4096_log2K": "4,096 shots + SPSA\\n(K=7)",
}
mode_colors = {
    "exact_analytic": "#222222",
    "shots_1024": "#2878B5",
    "shots_4096": "#6A3D9A",
    "shots_4096_log2K": "#D95319",
}
mode_linestyles = {
    "exact_analytic": "-",
    "shots_1024": "--",
    "shots_4096": "-.",
    "shots_4096_log2K": "-",
}'''
cell11 = replace_once(cell11, old_mode_block, new_mode_block)
cell11 = replace_once(cell11, 'for mode in mode_order:', 'for mode in learning_mode_order:')
cell11 = replace_once(
    cell11,
    'ax.plot(episodes, seed_curve, color=color, alpha=0.08, linewidth=0.55)',
    'ax.plot(episodes, seed_curve, color=color, linestyle=mode_linestyles[mode], alpha=0.08, linewidth=0.55)',
)
cell11 = replace_once(
    cell11,
    'ax.plot(episodes, mean, color=color, label=mode_labels[mode])',
    'ax.plot(episodes, mean, color=color, linestyle=mode_linestyles[mode], label=mode_labels[mode])',
)
cell11 = replace_once(cell11, 'figsize=(11.0, 7.2)', 'figsize=(11.0, 7.6)')
cell11 = replace_once(cell11, 'bbox_to_anchor=(0.5, 1.01)', 'bbox_to_anchor=(0.5, 1.005)')
cell11 = replace_once(cell11, 'y=1.015', 'y=1.045')
cell11 = replace_once(cell11, 'len(mode_order)', 'len(terminal_mode_order)')
cell11 = replace_once(
    cell11,
    'zip(x_positions, mode_order)',
    'zip(x_positions, terminal_mode_order)',
)
cell11 = replace_once(
    cell11,
    '[terminal_mode_labels[m] for m in mode_order]',
    '[terminal_mode_labels[m] for m in terminal_mode_order]',
)
cell11 = replace_once(cell11, 'ncol=4', 'ncol=2')
cell11 = replace_once(
    cell11,
    'CartPole-v1/200: exact and finite-shot training',
    'CartPole-v1/200: analytic and finite-shot training',
)
cell11 = replace_once(
    cell11,
    'finite_shot_learning_curves.png',
    'log2_direction_finite_shot_learning_curves.png',
)
cell11 = replace_once(
    cell11,
    'finite_shot_learning_curves.pdf',
    'log2_direction_finite_shot_learning_curves.pdf',
)
cell11 = replace_once(
    cell11,
    'finite_shot_final50_paired.png',
    'log2_direction_final50_paired.png',
)
cell11 = replace_once(
    cell11,
    'finite_shot_final50_paired.pdf',
    'log2_direction_final50_paired.pdf',
)
nb.cells[11].source = cell11

nb.cells[12].source = r"""try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    git_dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip())
except Exception:
    git_commit, git_dirty = None, None

# Prove the preserved source data stayed byte-for-byte unchanged.
for filename, expected in BASELINE_FILE_HASHES.items():
    actual = hashlib.sha256((BASELINE_RESULTS_DIR / filename).read_bytes()).hexdigest()
    assert actual == expected

config_metadata = {
    "created_utc": pd.Timestamp.utcnow().isoformat(),
    "source_notebook": str(ROOT / "tests" / "test_policy_training_efficiency_finite_shots.ipynb"),
    "notebook": str(ROOT / "tests" / "test_policy_training_efficiency_finite_shots_log2_directions.ipynb"),
    "preserved_baseline_results": str(BASELINE_RESULTS_DIR),
    "preserved_baseline_file_hashes": BASELINE_FILE_HASHES,
    "preserved_reporting_conditions": [
        "1,024 shots + SPSA (K=2)",
        "4,096 shots + SPSA (K=2)",
    ],
    "publication_curve_conditions": [
        "exact + analytic gradient (K=N/A)",
        "4,096 shots + SPSA (K=7)",
    ],
    "results_directory": str(RESULTS_DIR),
    "environment_label": "CartPole-v1/200",
    "gymnasium_environment_id": "CartPole-v1",
    "architectures": ARCHITECTURES,
    "parameter_count_audit": parameter_count_df.to_dict(orient="records"),
    "conditions": CONDITIONS,
    "policies": list(POLICIES),
    "episodes": N_EPISODES,
    "max_steps": MAX_STEPS,
    "terminal_metric": f"mean return over final {FINAL_WINDOW} episodes",
    "solved_metric": f"first full {SOLVED_WINDOW}-episode rolling mean >= {SOLVED_THRESHOLD}",
    "optimizer_freeze_rule": "at 5-episode boundary, mean of most recent up-to-100 returns >=195; interaction continues to episode 500",
    "preprocessing": "state / [2.4, 3.0, 0.2095, 3.5], clipped to [-3,3]",
    "finite_probability_rule": "Jeffreys alpha=0.5 per action: (count+0.5)/(shots+1)",
    "finite_gradient_estimator": "two-sided SPSA with K=ceil(log2 p) independent Rademacher directions/update and independent 4096-shot +/- losses",
    "parameter_count_definition": "all trainable scalar entries in input_scaling_params and circuit params",
    "training": TRAINING,
    "versions": {
        "python": platform.python_version(),
        "qasp": getattr(qasp, "__version__", "local checkout"),
        "pennylane": qml.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "gymnasium": gym.__version__,
    },
    "git_commit": git_commit,
    "git_dirty": git_dirty,
}

with open(RESULTS_DIR / "experiment_config.json", "w") as handle:
    json.dump(config_metadata, handle, indent=2, default=str)

manifest_rows = []
for path in sorted(RESULTS_DIR.rglob("*")):
    if path.is_file():
        manifest_rows.append({
            "path": str(path.relative_to(RESULTS_DIR)),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
manifest_df = pd.DataFrame(manifest_rows)
manifest_df.to_csv(RESULTS_DIR / "manifest.csv", index=False)

total_runtime = new_run_metadata_df.seconds.sum()
total_evaluations = int(new_run_metadata_df.total_circuit_evaluations.sum())
total_shots = int(new_run_metadata_df.total_circuit_shots.sum())
print(f"Aggregate serial training runtime: {total_runtime/3600:.3f} h")
print(f"Total NEW logical circuit evaluations: {total_evaluations:,}")
print(f"Total NEW finite circuit shots: {total_shots:,}")
print("Machine-readable files:")
display(manifest_df)
"""

nb.cells[13].source = r"""## Interpretation guardrails

- The preregistered terminal statistic remains final-50-episode return.
- “Solved” means a full rolling 100-episode mean of at least 195; the separate
  optimizer-freeze episode retains the source implementation's up-to-100 rule.
- The publication plot contains only the verified exact-analytic baseline and
  the new 4,096-shot \(K=\lceil\log_2 p\rceil\) condition.
- Bands are 95% normal intervals across five matched seeds and are clipped to
  the physical return range [0,200]. Paired bootstrap intervals resample seeds.
- The direct K-new minus K=2 comparison uses the preserved 4,096-shot results
  but does not add their curves to the publication plot.
- Preserved 1,024- and 4,096-shot K=2 summaries are retained with explicit K
  labels to quantify the effect of increasing shots alone; neither is added to
  the publication learning-curve plot.
- Here \(p\) is the full trainable optimizer dimension (90/65), not the number
  of effective gate arguments (45/35). Consequently, \(K=7\) for both models.
- Exact Born probabilities are used only to sample multinomial counts. No
  exact probability enters a finite-shot policy loss or gradient estimate.
"""

nb.metadata["kernelspec"] = {
    "display_name": "Python 3 (ipykernel)",
    "language": "python",
    "name": "python3",
}
nb.metadata["language_info"] = {"name": "python", "version": "3.10"}
nbf.write(nb, TARGET)
print(TARGET)
