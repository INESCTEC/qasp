"""Finite Fourier transform in the independent encoding angles of a QASP circuit."""

from __future__ import annotations

from copy import copy
from itertools import product
from math import prod

import numpy as np
import pennylane as qml
import torch

from qasp.circuits import AngleAnsatz, IQPAnsatz
from qasp.feature_maps import AngleEmbedding, NeuralEmbedding
from qasp.qkernels.quantum_kernels import EncodingAnsatz


def _primitive_ops(operations):
    """Expose one scalar angle per Pauli rotation, including in U(y)†."""
    for op in operations:
        if isinstance(op, qml.AngleEmbedding):
            yield from op.decomposition()
        elif isinstance(op, qml.ops.Adjoint):
            for base in reversed(list(_primitive_ops([op.base]))):
                yield base.adjoint()
        elif isinstance(op, qml.WireCut):
            continue  # ideal uncut circuit
        elif isinstance(op, qml.PauliRot) and set(op.hyperparameters["pauli_word"]) == {"I"}:
            continue  # a global phase cancels from every expectation
        elif isinstance(op, (qml.PauliRot, qml.MultiRZ)) and len(op.wires) > 2:
            # Avoid forming a dense 2**n by 2**n matrix for a long Pauli word.
            yield from _primitive_ops(op.decomposition())
        else:
            yield op


def _weights(model):
    if model.qlayer is None:  # encoding-only kernel
        return {"input_scaling_params": model._fixed_isp, "params": None}
    return {
        name: value.detach().cpu().double().numpy().copy()
        for name, value in model.qlayer.named_parameters()
    }


def feature_count(builder):
    """Validate the exact circuit family and size before preparing measurements."""
    if builder.n_qubits + builder.n_ancillas > 21:
        raise ValueError("The exact expressivity analyzer supports up to 21 total circuit wires.")
    # The built-in templates use inputs only in these linear feature maps.
    # A derivative at zero would be insufficient to certify an arbitrary encoder.
    ansatz = builder.ansatz
    if type(ansatz) not in (AngleAnsatz, IQPAnsatz, EncodingAnsatz):
        raise ValueError(
            "Fourier analysis requires AngleAnsatz, IQPAnsatz or an encoding-only kernel."
        )
    if type(ansatz.feature_map) not in (AngleEmbedding, NeuralEmbedding):
        raise ValueError("Fourier analysis requires QASP AngleEmbedding or NeuralEmbedding.")
    count = ansatz.feature_map.features
    if count is None:
        return builder.n_qubits
    if not isinstance(count, (int, np.integer)) or count < 1:
        raise ValueError("The feature map must have a positive feature count.")
    return int(count)


def _phase_tape(model, builder, n_features, overlap, reference, measurements):
    ansatz = copy(builder.ansatz)
    ansatz.feature_map = copy(ansatz.feature_map)
    weights = _weights(model)
    if any(not np.isfinite(w).all() for w in weights.values() if w is not None):
        raise ValueError("Circuit weights must be finite.")
    joint = overlap and reference is None
    dimension = n_features * (2 if joint else 1)
    with torch.inference_mode(False), torch.enable_grad():
        if weights.get("input_scaling_params") is not None:
            weights["input_scaling_params"] = torch.as_tensor(
                weights["input_scaling_params"], dtype=torch.float64
            )
        inputs = torch.zeros(dimension, dtype=torch.float64, requires_grad=True)
        with qml.queuing.AnnotatedQueue() as queue:
            ansatz(inputs=inputs[:n_features], **weights)
            if overlap:
                y = inputs[n_features:] if joint else torch.tensor(reference, dtype=torch.float64)
                qml.adjoint(ansatz)(inputs=y, **weights)
        source = qml.tape.QuantumScript.from_queue(queue)
        directions = {}
        degrees, bindings, gates = [], [], []
        parameter_index = 0
        for gate in _primitive_ops(source.operations):
            data = []
            for value in gate.data:
                if torch.is_tensor(value):
                    if value.requires_grad:
                        (slope,) = torch.autograd.grad(value, inputs, retain_graph=True)
                        vector = slope.tolist()
                        first = next((x for x in vector if x != 0), 0)
                        if first:
                            sign = 1 if first > 0 else -1
                            direction = tuple(sign * x for x in vector)
                            axis = directions.setdefault(direction, len(directions))
                            if axis == len(degrees):
                                degrees.append(0)
                            degrees[axis] += 1
                            bindings.append((parameter_index, axis, sign, value.item()))
                    value = value.detach().numpy()
                data.append(value)
                parameter_index += 1
            gates.append(qml.ops.functions.bind_new_parameters(gate, data) if data else gate)

    return (
        qml.tape.QuantumScript(gates, measurements),
        list(directions),
        degrees,
        bindings,
        dimension,
    )


def _physical_spectrum(coefficients, directions, degrees, dimension):
    """Sum colliding harmonics on the exact binary-rational lattice of the weights."""
    if not directions:
        return np.zeros((1, dimension)), coefficients.reshape(1, -1)
    ratios = [[x.as_integer_ratio() for x in direction] for direction in directions]
    denominators = [max(den for _, den in column) for column in zip(*ratios)]
    lattice = [
        [num * (scale // den) for (num, den), scale in zip(row, denominators)] for row in ratios
    ]
    # Integer keys avoid merging close but distinct learned frequencies, and
    # also avoid splitting a collision due to floating-point summation order.
    harmonics = product(*(range(-r, r + 1) for r in degrees))
    merged = {}
    for harmonic, row in zip(harmonics, coefficients.reshape(-1, coefficients.shape[-1])):
        key = tuple(sum(h * x for h, x in zip(harmonic, column)) for column in zip(*lattice))
        if key in merged:
            merged[key] += row
        else:
            merged[key] = row.copy()
    keys = sorted(merged)
    frequencies = [[x / scale for x, scale in zip(key, denominators)] for key in keys]
    return np.asarray(frequencies), np.stack([merged[key] for key in keys])


def transform(
    model,
    builder,
    n_features,
    overlap,
    reference,
    measurements,
    readout,
    n_outputs,
    max_evaluations,
):
    """Interpolate the ideal readout on the alias-free grid of Pauli generator phases."""
    n_wires = builder.n_qubits + builder.n_ancillas
    # Bound the analytic statevector and reduce the broadcast batch at larger widths.
    amplitude_budget = 1 << 21
    tape, directions, degrees, bindings, dimension = _phase_tape(
        model, builder, n_features, overlap, reference, measurements
    )
    shape = tuple(2 * r + 1 for r in degrees)
    count = prod(shape)
    if count > max_evaluations:
        raise ValueError(
            f"Exact phase grid requires {count} evaluations, exceeding "
            f"max_evaluations={max_evaluations}. Reduce the encoding or raise the limit."
        )
    if count * n_outputs > 4_000_000:
        raise ValueError(
            "Fourier coefficient array exceeds the size limit. Select fewer observables."
        )
    device = qml.device("default.qubit", wires=n_wires, seed=0)
    batch = min(64, amplitude_budget // (2**n_wires))
    values = np.empty((count, n_outputs))
    params = [index for index, _, _, _ in bindings]
    for start in range(0, count, batch):
        stop = min(start + batch, count)
        bound = tape
        if bindings:
            indices = np.unravel_index(np.arange(start, stop), shape)
            angles = [2 * np.pi * indices[i] / size for i, size in enumerate(shape)]
            bound = tape.bind_new_parameters(
                [offset + sign * angles[axis] for _, axis, sign, offset in bindings], params
            )
        (raw,) = qml.execute((bound,), device, diff_method=None, interface="numpy")
        # PennyLane returns a tuple for multiple expvals, an array for a single
        # measurement. A probability vector keeps its entire final basis axis.
        columns = raw if isinstance(raw, tuple) else (raw,)
        measured = np.concatenate(
            [np.asarray(c).reshape(stop - start, -1) for c in columns], axis=1
        )
        values[start:stop] = readout(measured)

    if not np.isfinite(values).all():
        raise ValueError("Circuit readout produced non-finite values.")
    coefficients = values.astype(complex).reshape((*shape, n_outputs))
    if shape:
        axes = tuple(range(len(shape)))
        # ifft uses the positive exponent to extract coefficients for exp(-i ω·x).
        coefficients = np.fft.fftshift(np.fft.ifftn(coefficients, axes=axes), axes=axes)
    frequencies, coefficients = _physical_spectrum(coefficients, directions, degrees, dimension)
    return frequencies, coefficients, count
