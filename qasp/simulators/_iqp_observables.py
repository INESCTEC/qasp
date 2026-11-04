"""Compact diagonal observables and unbiased Pauli-term proposals.

A component is a real coefficient times a Z word and zero or more basis-state
projectors. Projector expansions are sampled by independent uniform subsets;
neither a Pauli sentence nor an observable matrix is constructed here.
"""
from dataclasses import dataclass
import math

import numpy as np
import pennylane as qml
from pennylane.ops.qubit.observables import BasisStateProjector
import torch

from .iqp_tn import UnsupportedIQP


@dataclass(frozen=True)
class DiagonalComponent:
    coefficient: object
    z_indices: tuple
    projectors: tuple  # (device-wire indices, basis bits)

    @property
    def expansion_bits(self):
        return sum(len(wires) for wires, _ in self.projectors)


@dataclass(frozen=True)
class DiagonalObservable:
    components: tuple

    @property
    def log2_term_count(self):
        # Count expansion paths before cancellations, without making 2**n terms.
        sizes = [c.expansion_bits for c in self.components]
        largest = max(sizes, default=0)
        return largest + math.log2(sum(2.**(n-largest) for n in sizes))

    @property
    def description_size(self):
        return sum(1 + len(c.z_indices) + 2*c.expansion_bits for c in self.components)


def real_coefficient_value(c):
    value = c.detach().cpu().numpy() if isinstance(c, torch.Tensor) else np.asarray(c)
    if value.size != 1 or not np.isfinite(value).all() or np.any(np.imag(value) != 0):
        raise UnsupportedIQP("Z-observable coefficients must be finite real scalars.")
    return float(np.real(value).item())


def compile_diagonal_observable(obs, wires, max_components=100_000):
    """Read sums of products of I/Z and basis projectors without expansion.

    Scalar coefficients remain live Torch tensors, including zero coefficients.
    Products containing sums are deliberately not distributed automatically.
    """
    index = {w: i for i, w in enumerate(wires)}
    if not set(obs.wires) <= set(index):
        raise UnsupportedIQP("Observable addresses wires outside the device.")
    pending, components = [(1., obs)], []
    while pending:
        coefficient, op = pending.pop()
        if isinstance(op, qml.ops.op_math.SProd):
            pending.append((coefficient * op.scalar, op.base))
            continue
        if isinstance(op, qml.ops.op_math.LinearCombination):
            coeffs, terms = op.terms()
            pending.extend((coefficient*c, term) for c, term in zip(coeffs, terms))
            if len(pending) + len(components) > max_components:
                raise UnsupportedIQP("Observable exceeds vn_max_observable_components.")
            continue
        if isinstance(op, qml.ops.op_math.Sum):
            pending.extend((coefficient, term) for term in op.operands)
            if len(pending) + len(components) > max_components:
                raise UnsupportedIQP("Observable exceeds vn_max_observable_components.")
            continue
        factors, z, projectors = [op], set(), []
        while factors:
            factor = factors.pop()
            if isinstance(factor, qml.ops.op_math.SProd):
                coefficient = coefficient * factor.scalar
                factors.append(factor.base)
            elif isinstance(factor, qml.ops.op_math.Prod):
                factors.extend(factor.operands)
            elif isinstance(factor, qml.Identity):
                pass
            elif isinstance(factor, qml.PauliZ):
                z.symmetric_difference_update((index[factor.wires[0]],))
            elif isinstance(factor, BasisStateProjector):
                state = factor.data[0]
                if isinstance(state, torch.Tensor) and state.requires_grad:
                    raise UnsupportedIQP("Projector basis bits are discrete, not trainable parameters.")
                bits = tuple(int(v) for v in state)
                projectors.append((tuple(index[w] for w in factor.wires), bits))
            else:
                raise UnsupportedIQP(
                    "Importance sampling supports sums of products of I/Z words and "
                    "computational-basis Projectors; distribute products of sums explicitly. "
                    "General statevector projectors and majority decoders are not supported.")
        real_coefficient_value(coefficient)
        components.append(DiagonalComponent(coefficient, tuple(sorted(z)), tuple(projectors)))
        if len(components) > max_components:
            raise UnsupportedIQP("Observable exceeds vn_max_observable_components.")
    if not components:
        raise UnsupportedIQP("An importance observable needs at least one component.")
    return DiagonalObservable(tuple(components))


@dataclass(frozen=True)
class TermProposal:
    observable: DiagonalObservable
    probabilities: np.ndarray
    cdf: np.ndarray
    l1_bound: float
    weight_bound: float


def prepare_proposal(observable, uniform_mix):
    """Detached full-support proposal; live c/q preserves coefficient gradients.

    The positive uniform component is essential at trainable c=0: dropping such
    a term would omit its generally nonzero coefficient derivative.
    """
    magnitudes = np.array([abs(real_coefficient_value(c.coefficient))
                           for c in observable.components], dtype=np.float64)
    largest = float(magnitudes.max())
    if largest:
        scaled = magnitudes / largest
        base = scaled / scaled.sum()
        l1 = largest * float(scaled.sum())
    else:
        base = np.full(len(magnitudes), 1/len(magnitudes))
        l1 = 0.
    probabilities = (1-uniform_mix)*base + uniform_mix/len(base)
    probabilities /= probabilities.sum()
    bound = float(np.max(magnitudes / probabilities))
    if not math.isfinite(l1) or not math.isfinite(bound):
        raise UnsupportedIQP("Importance coefficient norm/weight bound must be finite.")
    cdf = np.cumsum(probabilities)
    cdf[-1] = 1.
    return TermProposal(observable, probabilities, cdf, l1, bound)


def sample_terms(proposal, rng, count, n):
    """Return component indices, Z masks and projector signs for one chunk."""
    selected = np.searchsorted(proposal.cdf, rng.random(count), side="right")
    masks = np.zeros((count, n), dtype=np.uint8)
    signs = np.ones(count, dtype=np.int8)
    for i in np.unique(selected):
        rows = np.flatnonzero(selected == i)
        component = proposal.observable.components[i]
        if component.z_indices:
            masks[np.ix_(rows, component.z_indices)] ^= 1
        for wires, bits in component.projectors:
            subsets = rng.integers(0, 2, size=(len(rows), len(wires)), dtype=np.uint8)
            masks[np.ix_(rows, wires)] ^= subsets
            parity = (subsets[:, np.flatnonzero(bits)].sum(axis=1) % 2).astype(np.int8)
            signs[rows] *= 1-2*parity
    return selected, masks, signs
