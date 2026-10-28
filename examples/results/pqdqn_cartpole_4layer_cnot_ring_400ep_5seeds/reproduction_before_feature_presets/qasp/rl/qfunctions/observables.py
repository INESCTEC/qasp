"""Pauli readouts, including exact Z decompositions of QASP action projectors."""

from __future__ import annotations

import math

import numpy as np
import pennylane as qml

from qasp.rl.policies import PQCBornVariance, _PolicyDecoder


def parse_pauli(spec: str, n_qubits: int):
    """Accept QASP notation (``'-XZ 0 2'``) or a full word (``'XIZ'``)."""
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("Each observable must be a non-empty Pauli string.")
    parts = spec.split()
    word = parts[0]
    sign = -1 if word.startswith("-") else 1
    word = word[1:] if word.startswith(("+", "-")) else word
    if not word or any(p not in "IXYZ" for p in word):
        raise ValueError(f"Invalid Pauli word: {spec!r}.")
    if len(parts) == 1:
        if len(word) != n_qubits:
            raise ValueError("A Pauli word without wire labels must span n_qubits.")
        wires = list(range(n_qubits))
    else:
        try:
            wires = [int(w) for w in parts[1:]]
        except ValueError as error:
            raise ValueError(f"Invalid wire labels: {spec!r}.") from error
    if (
        len(word) != len(wires)
        or len(set(wires)) != len(wires)
        or any(w < 0 or w >= n_qubits for w in wires)
    ):
        raise ValueError(
            f"Pauli letters require distinct, valid wire labels: {spec!r}."
        )
    return sign * qml.pauli.string_to_pauli_word(
        word, wire_map={wire: i for i, wire in enumerate(wires)}
    )


def predefined_observables(
    name, n_qubits, n_actions, measurement_qubits=None, majority_blocks=None
):
    """Return strings and coefficients C with P(action) = <Paulis> @ C.T.

    Uses an O(m 2**m) Walsh-Hadamard transform, never dense operator matrices.
    The identity term is retained. Even majority blocks resolve ties to zero.
    """
    if not isinstance(n_qubits, int) or n_qubits < 1:
        raise ValueError("n_qubits must be a positive integer.")
    if not isinstance(n_actions, int) or n_actions < 2:
        raise ValueError("n_actions must be an integer >= 2.")
    aliases = {
        "parity": "parity-like",
        "parity-like": "parity-like",
        "majority": "majority-modulo",
        "majority_modulo": "majority-modulo",
        "majority-modulo": "majority-modulo",
    }
    if name not in aliases:
        raise ValueError(f"Unknown predefined observable {name!r}.")
    name = aliases[name]
    wires = (
        list(range(n_qubits))
        if measurement_qubits is None
        else list(measurement_qubits)
    )
    if (
        not wires
        or len(set(wires)) != len(wires)
        or any(not isinstance(w, int) or w < 0 or w >= n_qubits for w in wires)
    ):
        raise ValueError("measurement_qubits must contain distinct valid wires.")
    # Exact majority decomposition is exponential; fail before a huge allocation.
    if len(wires) > 16:
        raise ValueError(
            "Exact predefined decompositions support at most 16 measured qubits."
        )
    if n_actions * (1 << len(wires)) > (1 << 22):
        raise ValueError(
            "Predefined decomposition is too large; supply explicit Pauli strings."
        )
    positions = None
    if name == "majority-modulo":
        k = math.ceil(math.log2(n_actions))
        if majority_blocks is None:
            sizes, _ = PQCBornVariance._balanced_odd_block_sizes(len(wires), k)
            majority_blocks, start = [], 0
            for size in sizes:
                majority_blocks.append(wires[start : start + size])
                start += size
        if len(majority_blocks) != k or any(not b for b in majority_blocks):
            raise ValueError(f"majority_blocks must contain {k} non-empty blocks.")
        flattened = [w for block in majority_blocks for w in block]
        if len(set(flattened)) != len(flattened) or any(
            w not in wires for w in flattened
        ):
            raise ValueError(
                "majority_blocks must be disjoint subsets of measurement_qubits."
            )
        positions = [[wires.index(w) for w in block] for block in majority_blocks]
    elif majority_blocks is not None:
        raise ValueError("majority_blocks is only valid for majority-modulo.")
    decoder = _PolicyDecoder(name, n_actions, len(wires), positions)
    indices = decoder.action_index.numpy()
    coeffs = np.stack([(indices == a).astype(float) for a in range(n_actions)])
    size = 1 << len(wires)
    stride = 1
    while stride < size:
        view = coeffs.reshape(n_actions, -1, 2 * stride)
        left, right = view[..., :stride].copy(), view[..., stride:].copy()
        view[..., :stride], view[..., stride:] = left + right, left - right
        stride *= 2
    coeffs /= size
    masks = np.flatnonzero(np.any(np.abs(coeffs) > 1e-12, axis=0))
    strings = []
    for mask in masks:
        selected = [
            w for i, w in enumerate(wires) if mask & (1 << (len(wires) - 1 - i))
        ]
        strings.append(
            "Z" * len(selected) + " " + " ".join(map(str, selected))
            if selected
            else f"I {wires[0]}"
        )
    return strings, coeffs[:, masks]


def resolve_observables(
    observables, n_qubits, n_actions, measurement_qubits=None, majority_blocks=None
):
    if isinstance(observables, str):
        strings, coefficients = predefined_observables(
            observables, n_qubits, n_actions, measurement_qubits, majority_blocks
        )
    else:
        if measurement_qubits is not None or majority_blocks is not None:
            raise ValueError(
                "measurement_qubits/majority_blocks require a named observable."
            )
        if not isinstance(observables, (list, tuple)) or not observables:
            raise ValueError(
                "Provide Pauli strings, per-action lists, or a named observable."
            )
        if all(isinstance(s, str) for s in observables):
            strings = list(observables)
            coefficients = np.eye(len(strings))
        else:
            if len(observables) != n_actions or any(
                not isinstance(group, (list, tuple)) or not group
                for group in observables
            ):
                raise ValueError(
                    "Nested observables require one non-empty list per action."
                )
            strings = [s for group in observables for s in group]
            coefficients = np.zeros((n_actions, len(strings)))
            start = 0
            for action, group in enumerate(observables):
                coefficients[action, start : start + len(group)] = 1
                start += len(group)
    operators = [parse_pauli(s, n_qubits) for s in strings]
    return strings, operators, coefficients
