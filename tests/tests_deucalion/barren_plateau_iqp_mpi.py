#!/usr/bin/env python3
"""Run one distributed QASP IQP barren-plateau diagnostic on Deucalion.

All MPI ranks execute the same circuit calls. Only rank zero writes results.
The companion Slurm array selects one of 28, 29, 30, or 31 qubits per job.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import sys
import time
import traceback
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import pennylane as qml
import torch
from mpi4py import MPI

import qasp
from qasp import barren_plateau_diagnoser, create_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose an IQP expectation classifier with a distributed "
            "lightning.kokkos statevector."
        )
    )
    parser.add_argument(
        "--n-qubits",
        type=int,
        default=int(os.environ.get("N_QUBITS", "28")),
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=int(os.environ.get("N_SAMPLES", "32")),
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=int(os.environ.get("N_LAYERS", "2")),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("SEED", "19")),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("OUTPUT_DIR", "results/barren_plateau_iqp_mpi")),
    )
    return parser.parse_args()


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def global_z_observable(n_qubits: int) -> str:
    paulis = "Z" * n_qubits
    wires = " ".join(str(wire) for wire in range(n_qubits))
    return f"{paulis} {wires}"


def create_iqp_classifier(n_qubits: int, n_layers: int):
    """Create the same ring-IQP model family at any requested width."""
    return create_classifier(
        {
            "ansatz": "IQPAnsatz",
            "ansatz_config": {
                "n_layers": n_layers,
                "imprimitive_pattern": "ring",
                "reuploading": False,
                "include_single_qubit": True,
            },
            "feature_map": "AngleEmbedding",
            "feature_map_config": {
                "features": n_qubits,
                "rotations": "Y",
                "imprimitive_pattern": "ring",
            },
            "readout": "expectation",
            "observables": [global_z_observable(n_qubits)],
            "init": "zeros",
            "device": "lightning.kokkos",
            "diff_method": "adjoint",
            "shots": None,
            "enable_mpi": True,
        },
        n_qubits=n_qubits,
        n_classes=2,
    )


def validate(args: argparse.Namespace, comm: MPI.Comm) -> None:
    if args.n_qubits not in {28, 29, 30, 31}:
        raise ValueError("--n-qubits must be one of 28, 29, 30, or 31.")
    if args.n_samples < 2:
        raise ValueError("--n-samples must be at least 2.")
    if args.n_layers < 1:
        raise ValueError("--n-layers must be positive.")
    if comm.size < 2:
        raise RuntimeError(
            "This experiment requires at least two MPI ranks so the statevector "
            "is genuinely distributed. Launch it with srun via the Slurm file."
        )
    if comm.size & (comm.size - 1):
        raise RuntimeError(
            f"Lightning-Kokkos requires a power-of-two MPI size; received {comm.size}."
        )
    distributed_bits = int(math.log2(comm.size))
    if args.n_qubits <= distributed_bits:
        raise RuntimeError(
            f"{args.n_qubits} qubits are insufficient for {comm.size} MPI ranks."
        )


def run(args: argparse.Namespace, comm: MPI.Comm) -> None:
    validate(args, comm)
    rank = comm.rank
    hosts = comm.gather(socket.gethostname(), root=0)

    if rank == 0:
        raw_gib = (2**args.n_qubits * np.dtype(np.complex128).itemsize) / 2**30
        local_gib = raw_gib / comm.size
        print(
            f"[INFO] n_qubits={args.n_qubits} n_samples={args.n_samples} "
            f"n_layers={args.n_layers} mpi_ranks={comm.size}",
            flush=True,
        )
        print(
            f"[INFO] raw_statevector={raw_gib:.2f} GiB total, "
            f"{local_gib:.2f} GiB/rank; hosts={hosts}",
            flush=True,
        )

    comm.Barrier()
    started = time.perf_counter()
    model = create_iqp_classifier(args.n_qubits, args.n_layers)

    # A fixed zero input isolates variance over the sampled trainable IQP
    # parameters. The cost is the positive-class probability derived from the
    # global Z expectation, while the diagnoser also records the raw value.
    result = barren_plateau_diagnoser(
        model,
        n_samples=args.n_samples,
        initialization="random_-pi_pi",
        state=torch.zeros(args.n_qubits),
        output_index=1,
        parameter_scope="quantum",
        seed=args.seed,
        return_samples=True,
    )
    local_elapsed = time.perf_counter() - started
    elapsed = comm.reduce(local_elapsed, op=MPI.MAX, root=0)

    if rank == 0:
        metadata = {
            "n_qubits": args.n_qubits,
            "n_samples": args.n_samples,
            "n_layers": args.n_layers,
            "seed": args.seed,
            "mpi_ranks": comm.size,
            "hosts": hosts,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "elapsed_seconds": elapsed,
            "python": sys.version,
            "platform": platform.platform(),
            "qasp_module": str(Path(qasp.__file__).resolve()),
            "qasp_version": package_version("qasp"),
            "pennylane_version": qml.__version__,
            "pennylane_lightning_version": package_version("pennylane-lightning"),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "mpi_library": MPI.Get_library_version(),
        }
        payload = {"metadata": metadata, "diagnosis": result}

        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = (
            f"iqp_barren_plateau_n{args.n_qubits}_"
            f"samples{args.n_samples}_seed{args.seed}"
        )
        json_path = args.output_dir / f"{stem}.json"
        npz_path = args.output_dir / f"{stem}.npz"
        json_path.write_text(
            json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        np.savez_compressed(
            npz_path,
            cost_samples=result["cost_samples"],
            expectation_samples=result["expectation_samples"],
            expectation_value_samples=result["expectation_value_samples"],
            sample_counts=result["sample_counts"],
            cost_variance_by_sample=result["cost_variance_by_sample"],
            expectation_variance_by_sample=result[
                "expectation_variance_by_sample"
            ],
            expectation_value_variance_by_sample=result[
                "expectation_value_variance_by_sample"
            ],
        )
        print(
            f"[RESULT] cost_variance={result['cost_variance']:.12e} "
            f"expectation_variance={result['expectation_value_variance']:.12e}",
            flush=True,
        )
        print(f"[RESULT] wrote {json_path} and {npz_path}", flush=True)

    comm.Barrier()


def main() -> None:
    comm = MPI.COMM_WORLD
    try:
        run(parse_args(), comm)
    except BaseException:
        print(f"[ERROR rank={comm.rank}]", file=sys.stderr, flush=True)
        traceback.print_exc()
        comm.Abort(1)


if __name__ == "__main__":
    main()
