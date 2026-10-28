"""Run on every rank: mpiexec -n 2 python tests/tests_deucalion/mpi_modules_smoke.py.

Requires the cluster's MPI-enabled PennyLane plugin, mpi4py, and installed QASP.
All circuits and gradients are evaluated collectively; only rank zero prints.
"""

import argparse
import math

import numpy as np
import torch
from mpi4py import MPI

from qasp import QAOA, QUBO, PCE, WarmStartQAOA, LRQAOA, RecursiveQAOA
from qasp import create_classifier, create_kernel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="lightning.kokkos")
    parser.add_argument("--qubits", type=int, default=6)
    args = parser.parse_args()
    comm = MPI.COMM_WORLD
    n = args.qubits
    if comm.size < 2 or comm.size & (comm.size - 1):
        raise ValueError("Run with a power-of-two number of MPI ranks, at least two.")
    if n < math.ceil(math.log2(comm.size)) + 2:
        raise ValueError("Use at least log2(ranks) + 2 qubits for this smoke test.")

    np.random.seed(17)
    torch.manual_seed(17)
    problem = QUBO.maxcut(n, [(i, i + 1) for i in range(n - 1)])

    def check(label, actual, expected):
        if isinstance(actual, torch.Tensor):
            actual = actual.detach().cpu().numpy()
        if isinstance(expected, torch.Tensor):
            expected = expected.detach().cpu().numpy()
        np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
        for rank_result in comm.allgather(np.asarray(actual)):
            np.testing.assert_allclose(actual, rank_result, atol=2e-6, rtol=2e-6)
        if comm.rank == 0:
            print(f"PASS {label}", flush=True)

    for backend in ("numpy", "torch"):
        for variant in (QAOA, WarmStartQAOA):
            for mode in ("standard", "multi_angle"):
                extra = {"warm_start": [0.3] * n} if variant is WarmStartQAOA else {}
                options = dict(backend=backend, seed=17, parametrization=mode, **extra)
                local = variant(problem, **options)
                distributed = variant(problem, device=args.device, mpi=True, **options)
                parameters = local.initial_parameters()
                label = f"{variant.__name__}/{mode}/{backend}"
                check(
                    label + "/energy",
                    distributed.expectation(parameters),
                    local.expectation(parameters),
                )
                check(
                    label + "/gradient",
                    distributed.gradient(parameters),
                    local.gradient(parameters),
                )
                expected = local.solve(
                    parameters, optimizer="Adam", optimizer_options={"max_steps": 1}
                )
                actual = distributed.solve(
                    parameters, optimizer="Adam", optimizer_options={"max_steps": 1}
                )
                check(label + "/train", actual.parameters, expected.parameters)

        local = PCE(problem, n_qubits=n, backend=backend, seed=17)
        distributed = PCE(
            problem, n_qubits=n, backend=backend, seed=17, device=args.device, mpi=True
        )
        parameters = local.initial_parameters()
        check(
            f"PCE/{backend}/loss", distributed.loss(parameters), local.loss(parameters)
        )
        check(
            f"PCE/{backend}/gradient",
            distributed.gradient(parameters),
            local.gradient(parameters),
        )

    check(
        "LRQAOA",
        LRQAOA(problem, device=args.device, mpi=True).solve().objective,
        LRQAOA(problem).solve().objective,
    )
    recursive = RecursiveQAOA(
        problem,
        device=args.device,
        mpi=True,
        cutoff=n - 1,
        optimizer_options={"maxiter": 3},
    )
    recursive.solve(result_shots=50, include_probabilities=False)
    sampled = QAOA(problem, device=args.device, mpi=True, seed=17)
    counts = sampled.solution_counts(sampled.initial_parameters(), shots=50)
    assert sum(counts.values()) == 50

    base = {
        "ansatz": "AngleAnsatz",
        "ansatz_config": {"n_layers": 1, "rotations": "Y"},
        "feature_map": "AngleEmbedding",
        "feature_map_config": {"rotations": "Y"},
        "init": "zeros",
        "diff_method": "adjoint",
    }
    x = torch.linspace(0.1, 0.4, n)
    cfg = {**base, "readout": "expectation", "observables": ["Z 0", "Z 1"]}
    local = create_classifier(cfg, n, 2)
    distributed = create_classifier({**cfg, "device": args.device, "mpi": True}, n, 2)
    check("classifier/output", distributed(x), local(x))
    distributed(x)[0].backward()
    local(x)[0].backward()
    for a, b in zip(distributed.parameters(), local.parameters()):
        if a.grad is not None:
            check("classifier/gradient", a.grad, b.grad)

    data = np.stack([x.numpy(), x.numpy() + 0.2])
    for kind in ("fidelity", "projected"):
        cfg = {**base, "kernel_type": kind}
        local = create_kernel(cfg, n)
        distributed = create_kernel({**cfg, "device": args.device, "mpi": True}, n)
        actual = distributed.gram(data, return_tensor=True)
        expected = local.gram(data, return_tensor=True)
        check(f"{kind}/gram", actual, expected)
        actual.sum().backward()
        expected.sum().backward()
        for a, b in zip(distributed.parameters(), local.parameters()):
            check(f"{kind}/gradient", a.grad, b.grad)
    if comm.rank == 0:
        print(f"All MPI module checks passed on {comm.size} ranks.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        MPI.COMM_WORLD.Abort(1)
