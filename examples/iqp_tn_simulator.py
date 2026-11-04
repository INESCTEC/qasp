"""Run from the repository root: python examples/iqp_tn_simulator.py.

Exact majority contraction, deliberately over-budget automatic fallback, and
forced exact execution of the same small circuit. No large forced allocation.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # before Torch for installations sharing OpenMP runtimes
import torch
import pennylane as qml
from dataclasses import asdict
import json
from qasp import IQPTNSimulator, DecoderSpec, action_probs
from qasp.devices import create_device


def quantum_policy(theta):
    for wire in range(5):
        qml.Hadamard(wire)
        qml.RZ(theta[wire], wire)
    qml.IsingZZ(theta[5], [0, 3])
    qml.MultiRZ(theta[6], [1, 2, 4])
    qml.MultiRZ(theta[5], [0, 1, 2, 3])  # a shared trainable angle
    for wire in range(5):
        qml.Hadamard(wire)
    return action_probs(range(5), DecoderSpec("majority"))


def main():
    theta = torch.linspace(-.7, .8, 7, dtype=torch.float64, requires_grad=True)
    cases = {
        "automatic": create_device("auto", wires=5),
        "automatic_fallback": create_device("auto", wires=5, device_kwargs={"auto_max_flops": 1}),
        "forced": IQPTNSimulator(5, auto_max_flops=1),
    }
    outputs = []
    for name, device in cases.items():
        node = qml.QNode(quantum_policy, device, interface="torch", diff_method="backprop")
        probabilities = node(theta)
        gradient = torch.autograd.grad(-probabilities[0].log(), theta)[0]
        outputs.append((probabilities, gradient))
        print(json.dumps({"case": name, "probabilities": probabilities.detach().tolist(),
                          "gradient": gradient.tolist(), "report": asdict(device.last_report)}, indent=2))
    for probabilities, gradient in outputs[1:]:
        torch.testing.assert_close(probabilities, outputs[0][0])
        torch.testing.assert_close(gradient, outputs[0][1])
    print("All three routes agree on probabilities and gradients.")


if __name__ == "__main__":
    main()
