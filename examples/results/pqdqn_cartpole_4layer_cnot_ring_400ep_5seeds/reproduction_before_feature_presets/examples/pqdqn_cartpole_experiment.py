"""Reproducible, resumable multi-seed CartPole experiments used by the notebooks.

A completed run is reused only when its full experiment specification matches.
Incomplete runs start afresh; model checkpoints do not claim trainer resumability.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml
import torch
from pqdqn_paper_figures import save_circuit_drawing

from qasp.rl import DeepQLearning, create_qfunction

BASE_CONFIG = {
    "ansatz": "AngleAnsatz",
    "ansatz_config": {
        "n_layers": 4,
        "rotations": "YZ",
        "reuploading": True,
        "include_imprimitive": True,
        "imprimitive": "CNOT",
        "imprimitive_pattern": "ring",
        "include_final_imprimitive": True,
    },
    "feature_map": "AngleEmbedding",
    "feature_map_config": {"features": 4, "rotations": "Y", "input_scaling": True},
    "fc_layer": True,
    "device": "default.qubit",
    "diff_method": "backprop",
    "batch_mode": "broadcast",
}
READOUTS = {
    "shared": {"observables": ["I 0", "Z 0", "Z 1", "Z 2", "Z 3"]},
    "per_action": {"observables": [["I 0", "Z 0", "Z 2"], ["I 0", "Z 1", "Z 3"]]},
    "parity": {"observables": "parity-like"},
    "majority": {"observables": "majority-modulo", "majority_blocks": [[0, 1, 2, 3]]},
}
TRAINING = {
    "lr": 0.01,
    "gamma": 0.99,
    "batch_size": 64,
    "learning_starts": 256,
    "train_frequency": 4,
    "target_update_interval": 100,
    "epsilon_decay_steps": 8000,
}


def _fingerprint(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def _without_backend(spec):
    """Compare experiment settings while allowing a simulator-only change."""
    from copy import deepcopy

    result = deepcopy(spec)
    for key in ("device", "diff_method"):
        result["model_config"].pop(key, None)
    return result


def _retain_completed_specs(specs, previous, output_root):
    """Keep completed results verbatim; only unfinished runs change backend."""
    old = {(s["readout"], s["seed"]): s for s in previous["specs"]}
    if set(old) != {(s["readout"], s["seed"]) for s in specs}:
        raise ValueError(
            "Preserving completed runs requires the same readouts and seeds."
        )
    retained = []
    for spec in specs:
        prior = old[(spec["readout"], spec["seed"])]
        if _without_backend(prior) != _without_backend(spec):
            raise ValueError(
                "Only device and diff_method may change when preserving completed runs."
            )
        marker = (
            Path(output_root)
            / spec["readout"]
            / f"seed_{spec['seed']}"
            / "summary.json"
        )
        if marker.exists():
            saved = json.loads(marker.read_text())
            if saved["fingerprint"] != _fingerprint(prior):
                raise ValueError(
                    f"Completion marker differs from the saved manifest: {marker}"
                )
            retained.append(prior)
        else:
            retained.append(spec)
    return retained


def run_one(spec, output_root):
    """Train one independent seed and write a completion marker last."""
    torch.set_num_threads(1)
    name, seed = spec["readout"], spec["seed"]
    run_dir = Path(output_root) / name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = _fingerprint(spec)
    marker = run_dir / "summary.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved["fingerprint"] != fingerprint:
            raise ValueError(
                f"Saved experiment differs at {run_dir}; choose a new output directory."
            )
        required = [
            "training.csv",
            "evaluation.csv",
            "model.pt",
            "circuit.tex",
            "circuit.png",
            "circuit.pdf",
            "circuit_full.tex",
            "circuit_full.pdf",
            "circuit_readout.json",
        ]
        if all((run_dir / name).exists() for name in required):
            return saved
    torch.manual_seed(seed)
    model = create_qfunction(spec["model_config"], n_qubits=4, n_actions=2)
    save_circuit_drawing(model, run_dir)
    start = time.perf_counter()
    with DeepQLearning(
        model, "CartPole-v1", seed=seed, **spec["training_config"]
    ) as trainer:
        baseline = trainer.evaluate(spec["eval_episodes"], seed=spec["eval_seed"])
        for start_episode in range(0, spec["episodes"], 50):
            trainer.train(min(50, spec["episodes"] - start_episode), log_every=0)
            history = pd.DataFrame(trainer.history)
            history.to_csv(run_dir / "training.csv", index=False)
            progress = {
                "readout": name,
                "seed": seed,
                "episode": trainer.episode_count,
                "frames": trainer.frame_count,
                "updates": trainer.update_count,
                "last_20_mean": float(history["return"].tail(20).mean()),
            }
            (run_dir / "progress.json").write_text(json.dumps(progress, indent=2))
        evaluation = trainer.evaluate(spec["eval_episodes"], seed=spec["eval_seed"])
        held_out = trainer.evaluate(
            spec["held_out_episodes"], seed=spec["held_out_seed"]
        )
        summary = {
            "readout": name,
            "seed": seed,
            "device": spec["model_config"]["device"],
            "diff_method": spec["model_config"]["diff_method"],
            "episodes": spec["episodes"],
            "frames": trainer.frame_count,
            "updates": trainer.update_count,
            "initial_greedy_mean": float(np.mean(baseline)),
            "final_greedy_mean": float(np.mean(evaluation)),
            "held_out_mean": float(np.mean(held_out)),
            "last_20_training_mean": float(history["return"].tail(20).mean()),
            "elapsed_seconds": time.perf_counter() - start,
            "fingerprint": fingerprint,
            "spec": spec,
        }
    evaluation_rows = [
        {"phase": phase, "episode": i + 1, "return": value}
        for phase, returns in [
            ("initial", baseline),
            ("final", evaluation),
            ("held_out", held_out),
        ]
        for i, value in enumerate(returns)
    ]
    pd.DataFrame(evaluation_rows).to_csv(run_dir / "evaluation.csv", index=False)
    torch.save(model.state_dict(), run_dir / "model.pt")
    # Verify saved parameters on the existing model without creating an extra model.
    probe = torch.tensor([[0.1, -0.2, 0.05, 0.3]])
    expected = model(probe).detach()
    model.load_state_dict(torch.load(run_dir / "model.pt", weights_only=True))
    torch.testing.assert_close(model(probe), expected)
    marker.write_text(json.dumps(summary, indent=2))
    return summary


def run_experiments(
    output_root,
    *,
    base_config=None,
    readouts=None,
    training=None,
    seeds=tuple(range(7, 12)),
    episodes=400,
    max_workers=4,
    eval_episodes=20,
    held_out_episodes=100,
    preserve_completed=False,
):
    """Run each readout for each seed, retaining all completed runs on restart."""
    from copy import deepcopy

    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    base_config = deepcopy(BASE_CONFIG if base_config is None else base_config)
    readouts = deepcopy(READOUTS if readouts is None else readouts)
    training = deepcopy(TRAINING if training is None else training)
    specs = [
        {
            "readout": name,
            "seed": seed,
            "episodes": episodes,
            "model_config": dict(base_config, **readout),
            "training_config": training,
            "eval_episodes": eval_episodes,
            "eval_seed": 10000,
            "held_out_episodes": held_out_episodes,
            "held_out_seed": 30000,
            "torch": torch.__version__,
            "pennylane": qml.__version__,
        }
        for seed in seeds
        for name, readout in readouts.items()
    ]
    manifest_path = output_root / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if preserve_completed:
            specs = _retain_completed_specs(specs, previous, output_root)
        elif previous["specs"] != specs:
            raise ValueError(
                "Experiment specification changed; choose a new output directory."
            )
    manifest = {
        "seeds": list(seeds),
        "readouts": list(readouts),
        "episodes": episodes,
        "specs": specs,
        "max_workers": max_workers,
        "preserve_completed": preserve_completed,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    summaries = []
    # Spawned workers keep PyTorch/RNG state separate and work in Jupyter too.
    import multiprocessing as mp

    with ProcessPoolExecutor(
        max_workers=max_workers, mp_context=mp.get_context("spawn")
    ) as pool:
        futures = {pool.submit(run_one, spec, output_root): spec for spec in specs}
        for future in as_completed(futures):
            summary = future.result()
            summaries.append(summary)
            print(
                f"Completed {len(summaries)}/{len(specs)}: {summary['readout']} "
                f"seed {summary['seed']} | held-out return {summary['held_out_mean']:.1f}",
                flush=True,
            )
    return pd.DataFrame(
        [
            dict(
                {k: v for k, v in s.items() if k != "spec"},
                device=s["spec"]["model_config"]["device"],
                diff_method=s["spec"]["model_config"]["diff_method"],
            )
            for s in summaries
        ]
    )


def aggregate_results(output_root, *, window=20):
    """Smooth each independent run, then compute mean/std across seeds per episode."""
    output_root = Path(output_root)
    manifest = json.loads((output_root / "manifest.json").read_text())
    frames, summaries = [], []
    for spec in manifest["specs"]:
        run_dir = output_root / spec["readout"] / f"seed_{spec['seed']}"
        summary = json.loads((run_dir / "summary.json").read_text())
        if summary["fingerprint"] != _fingerprint(spec):
            raise ValueError(f"Mismatched run: {run_dir}")
        frame = pd.read_csv(run_dir / "training.csv")
        if frame["episode"].tolist() != list(range(1, spec["episodes"] + 1)):
            raise ValueError(f"Incomplete episode history: {run_dir}")
        frame["readout"], frame["seed"] = spec["readout"], spec["seed"]
        frame["device"] = spec["model_config"]["device"]
        frame["diff_method"] = spec["model_config"]["diff_method"]
        frame["moving_return"] = (
            frame["return"].rolling(window, min_periods=window).mean()
        )
        frames.append(frame)
        summaries.append(
            dict(
                {k: v for k, v in summary.items() if k != "spec"},
                device=spec["model_config"]["device"],
                diff_method=spec["model_config"]["diff_method"],
            )
        )
    history = pd.concat(frames, ignore_index=True)
    moving = (
        history.groupby(["readout", "episode"])["moving_return"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    run_summary = pd.DataFrame(summaries)
    summary = (
        run_summary.groupby("readout")
        .agg(
            runs=("seed", "nunique"),
            backends=("device", lambda values: ", ".join(sorted(set(values)))),
            initial_greedy_mean=("initial_greedy_mean", "mean"),
            final_greedy_mean=("final_greedy_mean", "mean"),
            held_out_mean=("held_out_mean", "mean"),
            held_out_std_across_seeds=("held_out_mean", "std"),
            last_20_training_mean=("last_20_training_mean", "mean"),
        )
        .reindex(manifest["readouts"])
        .reset_index()
    )
    history.to_csv(output_root / "all_training.csv", index=False)
    moving.to_csv(output_root / "moving_returns.csv", index=False)
    run_summary.to_csv(output_root / "run_summaries.csv", index=False)
    summary.to_csv(output_root / "readout_summary.csv", index=False)
    return history, moving, run_summary, summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--preserve-completed", action="store_true")
    args = parser.parse_args()
    run_experiments(
        args.output_root,
        max_workers=args.workers,
        preserve_completed=args.preserve_completed,
    )
    aggregate_results(args.output_root)
