"""Collective PQCBM check: mpiexec -n 2 python tests/tests_deucalion/pqcbm_mpi_smoke.py.

Requires mpi4py and an MPI-enabled Lightning build on the cluster. Compare
MMD values, gradients and three training updates against a serial reference.
"""

import argparse

import numpy as np
import torch
from mpi4py import MPI

from qasp.genlearn import create_qcbm


def main():
    """Run all circuit calls on every rank; restrict output to rank zero."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="lightning.kokkos")
    parser.add_argument("--qubits", type=int, default=6)
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    if comm.size < 2 or comm.size & (comm.size - 1):
        raise ValueError("Use a power of two MPI ranks, at least two.")
    if args.qubits < (comm.size - 1).bit_length() + 2:
        raise ValueError("Use at least log2(ranks)+2 qubits.")
    config = {
        "n_qubits": args.qubits,
        "seed": 71,
        "loss": "mmd",
        "lr": 0.02,
        "ansatz_config": {"rotations": "YZ", "n_layers": 2},
        "mmd_config": {"bandwidths": [1.0, 2.0], "n_observables": 16},
        "batch_size": 4,
    }
    data = np.repeat(np.array([[0], [1], [0], [1], [0], [1]]), args.qubits, axis=1)
    serial = create_qcbm(config, data=data)
    distributed = create_qcbm(
        {**config, "device": args.device, "diff_method": "adjoint", "mpi": True},
        data=data,
    )

    def check(actual, expected):
        """Compare reference numerics and verify rank-identical state."""
        actual = np.asarray(actual)
        np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-6)
        for result in comm.allgather(actual):
            np.testing.assert_allclose(result, actual, atol=3e-6, rtol=3e-6)

    a, b = distributed.loss(), serial.loss()
    check(a.detach().numpy(), b.detach().numpy())
    check(
        torch.autograd.grad(a, distributed.params)[0].numpy(),
        torch.autograd.grad(b, serial.params)[0].numpy(),
    )
    check(distributed.train(3)[1], serial.train(3)[1])
    check(distributed.params.detach().numpy(), serial.params.detach().numpy())
    if comm.rank == 0:
        print("PASS PQCBM MPI: loss, gradient, training, and rank agreement")


if __name__ == "__main__":
    main()
