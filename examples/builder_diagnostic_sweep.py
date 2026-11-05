"""Run a cost-concentration sweep directly on compiled building blocks."""

import numpy as np
import torch

from qasp import AngleAnsatz, AngleEmbedding, CircuitBuilder, Ones, Uniform
from qasp import barren_plateau_diagnoser


def build_circuit(n):
    embedding = AngleEmbedding(n, features=n, rotations="Y", input_scaling=True)
    ansatz = AngleAnsatz(n, feature_map=embedding, rotations="Y",
                         include_imprimitive=True, imprimitive="CZ", imprimitive_pattern="chain")
    builder = CircuitBuilder(n)
    builder.set_ansatz(ansatz)
    parity = "Z" * n + " " + " ".join(map(str, range(n)))
    builder.set_measurement("expectation", observables=[parity])
    builder.build()
    return builder.compile(device="default.qubit")


if __name__ == "__main__":
    sweep = barren_plateau_diagnoser(
        model_factory=build_circuit,
        qubit_list=[4, 6, 8, 10, 12, 14, 16],
        state_factory=lambda n: torch.zeros(n, dtype=torch.float64),
        initialization=Uniform(-np.pi, np.pi),
        input_scaling_init=Ones(),
        n_samples=128,
        seed=19,
    )
    for width, variance in zip(sweep["qubit_list"], sweep["cost_variance"]):
        print(f"{width:2d} qubits   Var[global Z parity] = {variance:.6g}")
