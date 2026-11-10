"""Observable-resolved Fourier expressivity of Hamiltonian-encoded QASP models."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pennylane as qml
import torch

from qasp.classifiers import PQCClassifier
from qasp.qkernels import FidelityKernel, ProjectedKernel
from qasp.rl.policies import LearnablePQCBorn, PQCBorn, PQCBornVariance, PQCSoftmax
from qasp.rl.qfunctions import PQDQN
from qasp.rl.qfunctions.observables import parse_pauli

from ._fourier import feature_count, transform


@dataclass
class FourierSpectrum:
    """Rows are signed frequencies (including DC), columns are labeled outputs.

    coefficients[j, a] multiplies exp(-1j * frequencies[j] @ x) in output a.
    x is the input of the feature map, before any learned scaling. For joint
    fidelity kernels the frequency columns refer to the concatenation of (x, y).
    """

    n_qubits: int
    frequencies: np.ndarray
    coefficients: np.ndarray
    output_labels: tuple[str, ...]
    n_features: int
    joint_kernel: bool
    reference: np.ndarray | None
    input_activation: str | None
    atol: float
    n_evaluations: int

    @property
    def n_frequencies(self):
        """Number of distinct generator-allowed frequencies, including inactive ones."""
        return len(self.frequencies)

    @property
    def magnitudes(self):
        return np.abs(self.coefficients)

    @property
    def active_counts(self):
        """Number of coefficients above the absolute tolerance, per output."""
        return (self.magnitudes > self.atol).sum(axis=0)

    def evaluate(self, x, y=None):
        """Reconstruct all columns at feature-map inputs, with optional batch axes.

        Supply both x and y for a joint kernel, or just x for a fixed-reference
        kernel. For a Q-function with input_activation='tanh', pass tanh(state).
        """
        x = np.asarray(_array(x), dtype=float)
        if x.ndim == 0 or x.shape[-1] != self.n_features:
            raise ValueError(f"x must have a final axis of length {self.n_features}.")
        if self.joint_kernel:
            if y is None:
                raise ValueError("A joint fidelity spectrum requires both x and y.")
            y = np.asarray(_array(y), dtype=float)
            if y.ndim == 0 or y.shape[-1] != self.n_features:
                raise ValueError(f"y must have a final axis of length {self.n_features}.")
            x = np.concatenate(np.broadcast_arrays(x, y), axis=-1)
        elif y is not None:
            raise ValueError("Supply y only for a joint fidelity spectrum.")
        if not np.isfinite(x).all():
            raise ValueError("Evaluation inputs must be finite.")
        return (np.exp(-1j * (x @ self.frequencies.T)) @ self.coefficients).real


@dataclass
class ExpressivitySweep:
    """Spectra indexed by data-qubit count, in the requested order."""

    results_by_qubit: dict[int, FourierSpectrum]

    @property
    def qubit_list(self):
        return list(self.results_by_qubit)

    @property
    def frequency_counts(self):
        return np.array([s.n_frequencies for s in self.results_by_qubit.values()])

    def summary(self):
        """One plot/table-ready record per width and output, excluding DC from power."""
        rows = []
        for n, s in self.results_by_qubit.items():
            non_dc = np.any(s.frequencies != 0, axis=1)
            magnitudes = s.magnitudes
            counts = s.active_counts
            for i, label in enumerate(s.output_labels):
                rows.append(
                    dict(
                        n_qubits=n,
                        output=label,
                        n_frequencies=s.n_frequencies,
                        active_count=int(counts[i]),
                        max_magnitude=float(magnitudes[:, i].max()),
                        non_dc_power=float(np.sum(magnitudes[non_dc, i] ** 2)),
                    )
                )
        return rows


def _identity(values):
    return values


def _labels(prefix, count):
    return tuple(f"{prefix}[{i}]" for i in range(count))


def _array(value):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.is_floating_point():
            value = value.double()
        return value.numpy()
    return np.asarray(value)


def _fixed_observable(op):
    if not isinstance(op, qml.operation.Operator):
        raise ValueError("Supply Hermitian PennyLane observables or Pauli strings.")
    if any(torch.is_tensor(value) for value in op.data):
        data = [_array(value).copy() if torch.is_tensor(value) else value for value in op.data]
        return qml.ops.functions.bind_new_parameters(op, data)
    return op


def _readout(model, builder, observables):
    """Return terminal measurements, reported labels and the finite readout."""
    n_wires = builder.n_qubits + builder.n_ancillas
    if observables is not None:
        if isinstance(observables, (str, qml.operation.Operator)):
            observables = [observables]
        specs = list(observables)
        if not specs:
            raise ValueError("observables must contain at least one observable.")
        ops = [
            parse_pauli(s, n_wires) if isinstance(s, str) else _fixed_observable(s) for s in specs
        ]
        for spec, op in zip(specs, ops):
            if not set(op.wires).issubset(range(n_wires)):
                raise ValueError("Observable addresses a wire outside the circuit.")
            if not isinstance(spec, str) and not qml.is_hermitian(op):
                raise ValueError("Supply a Hermitian observable.")
        return [qml.expval(op) for op in ops], tuple(map(str, specs)), _identity

    if isinstance(model, LearnablePQCBorn):
        raise ValueError("Adaptive measurement policies require explicit fixed observables.")
    kind, wires = builder.measurement
    if kind == "expectation":
        measurements = [qml.expval(_fixed_observable(op)) for op in builder.obs]
        labels = tuple(map(str, builder.obs))
    elif kind == "probs":
        measurements = [qml.probs(wires=sorted(wires))]
    else:
        raise ValueError("Select fixed observables for this circuit readout.")

    if isinstance(model, FidelityKernel):
        return measurements, ("fidelity",), lambda v: v.mean(axis=1, keepdims=True)
    if isinstance(model, ProjectedKernel):
        return measurements, tuple(model.observable_labels), _identity

    if isinstance(model, (PQDQN, PQCSoftmax)):

        def scores(v):
            if isinstance(model, PQDQN):
                v = v @ _array(model.readout_coefficients).T
            elif model.output_scaling is not None:
                scale = (
                    model.output_scaling_weight
                    if model.output_scaling is True
                    else model.output_scaling
                )
                v = v * _array(scale)
            head = model.fc_layer
            if head is not None:
                v = v @ _array(head.weight).T
                if head.bias is not None:
                    v = v + _array(head.bias)
            return v

        prefix = "Q" if isinstance(model, PQDQN) else "logit"
        return measurements, _labels(prefix, model.n_actions), scores

    if isinstance(model, PQCClassifier):
        if model.readout == "expectation":
            sizes = [len(group) for group in model.observable_groups]

            def class_scores(v):
                groups = np.split(v, np.cumsum(sizes)[:-1], axis=1)
                scores = np.column_stack([g.mean(axis=1) for g in groups])
                return scores if model.output_scale is None else scores * _array(model.output_scale)

            return measurements, _labels("score", model.n_outputs), class_scores

        def class_probabilities(v):
            return _array(
                model.probability_decoder(
                    torch.as_tensor(v, device=model.probability_decoder.class_index.device)
                )
            )

        return measurements, _labels("class_probability", model.n_classes), class_probabilities

    if isinstance(model, PQCBornVariance) and model.training_method != "local_pauli_decomposition":

        def action_probabilities(v):
            values = torch.as_tensor(v, device=model.qlayer.params.device)
            if kind == "probs":
                return _array(model.policy_decoder(values))
            moments = values.T if len(measurements) > 1 else values
            return _array(model.post_process_policy(moments)).reshape(len(v), model.n_actions)

        return measurements, _labels("action_probability", model.n_actions), action_probabilities

    # Legacy sampling policies expose their underlying fixed quantum readout.
    if kind == "probs":
        labels = _labels("basis_probability", 2 ** len(wires))
    return measurements, labels, _identity


def _analyze(model, observables, reference, atol, max_evaluations):
    if isinstance(model, (FidelityKernel, ProjectedKernel)):
        builder = model.builder
    elif isinstance(
        model, (PQCClassifier, PQDQN, PQCSoftmax, PQCBornVariance, PQCBorn, LearnablePQCBorn)
    ):
        builder = model.circuit_builder
    else:
        raise TypeError(
            "Expected a QASP RL policy, Q-function, classifier, fidelity or projected kernel."
        )
    if builder.n_qubits != model.n_qubits:
        raise ValueError("The model and its circuit builder disagree on the data-qubit count.")
    n_features = feature_count(builder)
    overlap = isinstance(model, FidelityKernel)
    if reference is not None:
        if not overlap:
            raise ValueError("reference applies only to fidelity kernels.")
        reference = np.array(_array(reference), dtype=float, copy=True)
        if reference.shape != (n_features,) or not np.isfinite(reference).all():
            raise ValueError(f"reference must be a finite vector of length {n_features}.")
    measurements, labels, readout = _readout(model, builder, observables)
    frequencies, coefficients, count = transform(
        model,
        builder,
        n_features,
        overlap,
        reference,
        measurements,
        readout,
        len(labels),
        max_evaluations,
    )
    return FourierSpectrum(
        builder.n_qubits,
        frequencies,
        coefficients,
        labels,
        n_features,
        overlap and reference is None,
        reference,
        getattr(model, "input_activation", None),
        atol,
        count,
    )


def expressivity_analyzer(
    model=None,
    *,
    model_factory=None,
    qubit_list=None,
    observables=None,
    reference=None,
    atol=1e-10,
    max_evaluations=65_536,
):
    """Compute a FourierSpectrum or a width-indexed ExpressivitySweep.

    Supply a constructed model, or model_factory(n_qubits) and a qubit_list.
    observables overrides the model readout with individual Pauli strings or
    Hermitian PennyLane operators. In a sweep it may also be a function of width.
    Fixed ansatz parameters and encoder scales are read from each model.
    atol controls active counts only. max_evaluations bounds the exact phase
    grid per width; a larger grid raises ValueError before simulation.

    Default outputs are policy logits or pre-softmax Born weights, affine
    Q-values, classifier scores or pre-calibration class probabilities, the
    fidelity kernel, or the measured features of a projected kernel. A fidelity
    kernel has a joint (x, y) spectrum unless a fixed reference y is supplied.
    The transform uses ideal analytic measurements in feature-map coordinates.
    """
    if (
        not isinstance(max_evaluations, Integral)
        or isinstance(max_evaluations, bool)
        or max_evaluations < 1
    ):
        raise ValueError("max_evaluations must be a positive integer.")
    if not np.isfinite(atol) or atol < 0:
        raise ValueError("atol must be a finite nonnegative magnitude.")
    if qubit_list is None:
        if model is None or model_factory is not None:
            raise ValueError("Supply a model or both model_factory and qubit_list.")
        return _analyze(model, observables, reference, atol, max_evaluations)
    if model is not None or not callable(model_factory):
        raise ValueError("A qubit sweep requires model_factory(n_qubits) and qubit_list.")
    widths = list(qubit_list)
    if (
        not widths
        or any(not isinstance(n, Integral) or isinstance(n, bool) or n < 1 for n in widths)
        or len(set(widths)) != len(widths)
    ):
        raise ValueError("qubit_list must contain distinct positive integers.")
    results = {}
    # A generator of observables should apply equally to every width.
    factory = callable(observables) and not isinstance(observables, qml.operation.Operator)
    if (
        not factory
        and observables is not None
        and not isinstance(observables, (str, qml.operation.Operator))
    ):
        observables = tuple(observables)
    for n in widths:
        model = model_factory(int(n))
        if getattr(model, "n_qubits", None) != n:
            raise ValueError(f"model_factory must return a model with {n} data qubits.")
        specs = observables(n) if factory else observables
        results[n] = _analyze(model, specs, reference, atol, max_evaluations)
    return ExpressivitySweep(results)
