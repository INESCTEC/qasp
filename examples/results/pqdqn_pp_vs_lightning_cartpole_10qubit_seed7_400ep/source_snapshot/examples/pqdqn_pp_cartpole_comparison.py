"""Matched one-seed, 400-episode CartPole study of the ten-qubit PQDQN.

Both backends load the same microbenchmark seed-7 checkpoint. CartPole's four
observations are zero-padded to ten before the model's tanh activation.
Run from the repository root with PYTHONPATH=.: the default starts two workers.
Interrupted runs resume at an episode boundary with replay, optimizer, target,
trainer RNG and Torch RNG restored. A process lock prevents duplicate runners.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pqdqn_pauli_propagation_benchmark import OUTPUT as BENCHMARK
from pqdqn_pauli_propagation_benchmark import make_model, model_config

from qasp.rl import DeepQLearning

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "examples/results/pqdqn_pp_vs_lightning_cartpole_10qubit_seed7_400ep"
BACKENDS = ["pp_k2", "lightning_qubit"]
TRAINING = {
    "lr": 0.01,
    "gamma": 0.99,
    "batch_size": 64,
    "replay_capacity": 10000,
    "learning_starts": 256,
    "train_frequency": 4,
    "gradient_steps": 1,
    "target_update_interval": 100,
    "double_dqn": True,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay_steps": 8000,
    "max_steps": 500,
    "max_grad_norm": 10.0,
}


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def write_csv(path, frame):
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def pad_observation(observation):
    values = np.asarray(observation, dtype=np.float64)
    if values.shape != (4,):
        raise ValueError("This comparison expects four CartPole observation values.")
    return np.pad(values, (0, 6))


def checkpoint(trainer, path, fingerprint, elapsed, baseline):
    payload = {
        "fingerprint": fingerprint,
        "model": trainer.qfunction.state_dict(),
        "target": trainer.target_network.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "replay": list(trainer.replay_buffer),
        "trainer_rng": trainer._rng.getstate(),
        "torch_rng": torch.get_rng_state(),
        "frames": trainer.frame_count,
        "updates": trainer.update_count,
        "episodes": trainer.episode_count,
        "history": trainer.history,
        "training_seconds": elapsed,
        "baseline": baseline,
    }
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(path)


def restore(trainer, path, fingerprint):
    saved = torch.load(path, weights_only=True)
    if saved["fingerprint"] != fingerprint:
        raise ValueError("Checkpoint configuration does not match this run.")
    trainer.qfunction.load_state_dict(saved["model"])
    trainer.target_network.load_state_dict(saved["target"])
    trainer.optimizer.load_state_dict(saved["optimizer"])
    trainer.replay_buffer.extend(saved["replay"])
    trainer._rng.setstate(saved["trainer_rng"])
    torch.set_rng_state(saved["torch_rng"])
    trainer.frame_count = saved["frames"]
    trainer.update_count = saved["updates"]
    trainer.episode_count = saved["episodes"]
    trainer.history = saved["history"]
    return saved["training_seconds"], saved["baseline"]


def run_backend(backend, output, spec):
    folder = Path(output) / backend
    folder.mkdir(exist_ok=True)
    with (folder / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            return _run_backend(backend, output, spec)
        except BaseException:
            write_json(
                folder / "status.json",
                {
                    "status": "failed",
                    "pid": os.getpid(),
                    "updated_utc": now(),
                    "traceback": traceback.format_exc(),
                },
            )
            raise


def _run_backend(backend, output, spec):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(spec["seed"])
    folder = Path(output) / backend
    folder.mkdir(exist_ok=True)
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    completed = folder / "summary.json"
    if completed.exists():
        result = json.loads(completed.read_text())
        if result["fingerprint"] != fingerprint:
            raise ValueError("Completed run differs from requested configuration.")
        return result
    write_json(
        folder / "status.json",
        {"status": "initializing", "pid": os.getpid(), "updated_utc": now()},
    )
    model = make_model(backend)
    initial = torch.load(Path(output) / "initial_state.pt", weights_only=True)
    model.load_state_dict(initial)
    train_seconds = 0.0
    with DeepQLearning(
        model,
        "CartPole-v1",
        seed=spec["seed"],
        observation_transform=pad_observation,
        **spec["training"],
    ) as trainer:
        saved = folder / "checkpoint.pt"
        if saved.exists():
            train_seconds, baseline = restore(trainer, saved, fingerprint)
        else:
            baseline = trainer.evaluate(spec["initial_eval_episodes"], seed=20000)
            checkpoint(trainer, saved, fingerprint, train_seconds, baseline)
        write_csv(
            folder / "initial_evaluation.csv",
            pd.DataFrame(
                {"reset_seed": range(20000, 20000 + len(baseline)), "return": baseline}
            ),
        )
        while trainer.episode_count < spec["episodes"]:
            started = time.perf_counter()
            trainer.train(1, log_every=0)
            train_seconds += time.perf_counter() - started
            checkpoint(trainer, saved, fingerprint, train_seconds, baseline)
            history = pd.DataFrame(trainer.history)
            write_csv(folder / "training.csv", history)
            write_json(
                folder / "status.json",
                {
                    "status": "training",
                    "pid": os.getpid(),
                    "updated_utc": now(),
                    "episode": trainer.episode_count,
                    "frames": trainer.frame_count,
                    "updates": trainer.update_count,
                    "training_seconds": train_seconds,
                    "last_20_mean": float(history["return"].tail(20).mean()),
                },
            )
            if trainer.episode_count % 10 == 0:
                print(
                    f"{backend}: episode {trainer.episode_count}/{spec['episodes']}; last20={history['return'].tail(20).mean():.1f}; training={train_seconds / 60:.1f}min",
                    flush=True,
                )
        write_json(
            folder / "status.json",
            {
                "status": "evaluating",
                "pid": os.getpid(),
                "updated_utc": now(),
                "episode": trainer.episode_count,
            },
        )
        started = time.perf_counter()
        final = trainer.evaluate(spec["initial_eval_episodes"], seed=20000)
        held_out = trainer.evaluate(spec["held_out_episodes"], seed=30000)
        eval_seconds = time.perf_counter() - started
        rows = [
            dict(phase=phase, episode=i + 1, reset_seed=base + i, **{"return": value})
            for phase, base, returns in [
                ("initial", 20000, baseline),
                ("final", 20000, final),
                ("held_out", 30000, held_out),
            ]
            for i, value in enumerate(returns)
        ]
        write_csv(folder / "evaluation.csv", pd.DataFrame(rows))
        torch.save(model.state_dict(), folder / "model.pt")
        changes = {
            name: float(torch.linalg.vector_norm(p.detach() - initial[name]))
            for name, p in model.named_parameters()
        }
        summary = {
            "backend": backend,
            "seed": spec["seed"],
            "episodes": trainer.episode_count,
            "frames": trainer.frame_count,
            "updates": trainer.update_count,
            "initial_greedy_mean": float(np.mean(baseline)),
            "final_greedy_mean": float(np.mean(final)),
            "held_out_mean": float(np.mean(held_out)),
            "held_out_episode_sd": float(np.std(held_out, ddof=1)),
            "last_20_training_mean": float(
                np.mean([h["return"] for h in trainer.history[-20:]])
            ),
            "training_seconds": train_seconds,
            "final_evaluation_seconds": eval_seconds,
            "parameter_change_l2": changes,
            "fingerprint": fingerprint,
            "completed_utc": now(),
        }
        write_json(completed, summary)
        write_json(
            folder / "status.json",
            {
                "status": "complete",
                "pid": os.getpid(),
                "updated_utc": now(),
                "episode": trainer.episode_count,
            },
        )
        return summary


def collect(output, spec):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    histories = []
    for b in BACKENDS:
        folder = output / b
        summary = json.loads((folder / "summary.json").read_text())
        history = pd.read_csv(folder / "training.csv")
        evaluation = pd.read_csv(folder / "evaluation.csv")
        assert len(history) == spec["episodes"]
        assert history.episode.tolist() == list(range(1, spec["episodes"] + 1))
        assert evaluation.groupby("phase").size().to_dict() == {
            "initial": spec["initial_eval_episodes"],
            "final": spec["initial_eval_episodes"],
            "held_out": spec["held_out_episodes"],
        }
        assert np.isfinite(history["return"]).all()
        held = evaluation.query("phase == 'held_out'")["return"]
        assert np.isclose(held.mean(), summary["held_out_mean"])
        history["backend"] = b
        history["moving_return"] = history["return"].rolling(20, min_periods=20).mean()
        histories.append(history)
        rows.append({k: v for k, v in summary.items() if not isinstance(v, dict)})
    write_csv(output / "summary.csv", pd.DataFrame(rows))
    write_csv(
        output / "training_comparison.csv", pd.concat(histories, ignore_index=True)
    )
    fig, ax = plt.subplots(figsize=(8, 4.2), layout="constrained")
    for h in histories:
        ax.plot(h.episode, h.moving_return, label=h.backend.iloc[0])
    ax.set(
        xlabel="Training episode",
        ylabel="Trailing 20-episode return",
        title=f"10-qubit CartPole: one matched seed, {spec['episodes']} episodes per backend",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(output / "learning_curves.png", dpi=180)
    plt.close(fig)
    write_json(
        output / "finalization_status.json",
        {
            "status": "complete",
            "updated_utc": now(),
            "verification": "Both training histories and evaluation counts checked; held-out means recomputed.",
            "independent_seeds_per_backend": 1,
        },
    )
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        training = TRAINING.copy()
        if args.smoke:
            training.update(
                batch_size=2, learning_starts=2, train_frequency=1, max_steps=4
            )
        source = BENCHMARK / "seed_7.pt"
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        spec = {
            "seed": 7,
            "episodes": 2 if args.smoke else 400,
            "model_config": model_config(),
            "training": training,
            "initial_eval_episodes": 2 if args.smoke else 20,
            "held_out_episodes": 2 if args.smoke else 100,
            "backend_settings": {
                "pp_k2": {
                    "device": "lightning.pp",
                    "diff_method": "backprop",
                    "pp_truncation": 2,
                    "pp_prune": False,
                    "pp_topM": None,
                    "pp_tau": 0.0,
                },
                "lightning_qubit": {
                    "device": "lightning.qubit",
                    "diff_method": "adjoint",
                },
            },
            "initial_checkpoint_source": str(source),
            "initial_checkpoint_sha256": source_hash,
            "observation_transform": "Append six zeros to CartPole [x,x_dot,theta,theta_dot]; tanh inside PQDQN",
            "dtype": "float64",
            "workers": 2,
            "torch_threads_per_worker": 1,
            "timing_note": "Runs execute concurrently in separate processes. Training wall time excludes evaluation and checkpoint I/O; contention prevents a controlled speed comparison.",
            "versions": {
                n: importlib.metadata.version(n)
                for n in ["torch", "pennylane", "pennylane-lightning", "gymnasium"]
            },
        }
        manifest = output / "manifest.json"
        if manifest.exists() and json.loads(manifest.read_text()) != spec:
            raise ValueError(
                "Existing manifest differs; choose a new output directory."
            )
        write_json(manifest, spec)
        if args.collect_only:
            collect(output, spec)
            return
        if not (output / "initial_state.pt").exists():
            torch.save(
                torch.load(source, weights_only=True)["state_dict"],
                output / "initial_state.pt",
            )
            drawing = BENCHMARK / "drawings"
            if drawing.exists():
                shutil.copytree(drawing, output / "drawings", dirs_exist_ok=True)
        for path in [
            "qasp/rl/qfunctions/models.py",
            "qasp/rl/differentiable_pp_sim.py",
            "qasp/rl/torchrl.py",
        ]:
            target = output / "source_snapshot" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copyfile(ROOT / path, target)
        write_json(
            output / "finalization_status.json",
            {
                "status": "running",
                "pid": os.getpid(),
                "updated_utc": now(),
                "script": str(Path(__file__).resolve()),
            },
        )
        try:
            with ProcessPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(run_backend, b, str(output), spec) for b in BACKENDS
                ]
                for future in as_completed(futures):
                    future.result()
            collect(output, spec)
        except BaseException:
            write_json(
                output / "finalization_status.json",
                {
                    "status": "failed",
                    "pid": os.getpid(),
                    "updated_utc": now(),
                    "traceback": traceback.format_exc(),
                },
            )
            raise


if __name__ == "__main__":
    main()
