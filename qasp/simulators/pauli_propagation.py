"""Shared analytic, Torch-differentiable Heisenberg simulator.

Exact by default. Truncation and detached coefficient pruning are explicit
approximations. General noncommuting circuits are supported through gate
decomposition; this is not restricted to IQP circuits. No statevector fallback
is used. Full probability readout expands 2**m parity moments for m readout wires.
"""
from dataclasses import dataclass, replace
from numbers import Integral
import math

import numpy as np
import pennylane as qml
import torch
from pennylane.pauli import PauliSentence, PauliWord

from ._pauli_propagation import DifferentiablePauliProp
from .readout import ActionProbsMP


@dataclass(frozen=True)
class PauliPropagationOptions:
    """Configure exact propagation or explicit approximation budgets.

    Attributes:
        k: Maximum number of non-identity factors in a Pauli word. Applies to
            the initial observable and during propagation, independently of
            ``prune``. None retains every weight; this is not a gate-arity limit.
        prune: Enable coefficient selection using ``tau`` and ``topM``.
        tau: Discard merged coefficients with magnitude strictly below this
            nonnegative threshold when pruning. Not an output-error tolerance.
        topM: Retain at most this positive number of largest-magnitude terms
            after thresholding at pruning points; None disables the count limit.
            Intermediate expansion and the autograd graph can exceed this size.
        prune_every: Coefficient-selection interval in reversed gates, plus a
            final selection. Zero means final-only. Does not control ``k``.
        max_probability_wires: Maximum measured wires in each dense probability
            or computational-projector expansion, not the total circuit width.

    Exact defaults are ``k=None`` and ``prune=False``. Pruning decisions use
    detached magnitudes; autograd differentiates the currently retained paths.
    Neither approximation has a general error bound or guarantees nonnegative
    probabilities/variances. The device does not clip or renormalize results.
    Readout limits and topM are not global memory caps.
    """
    k: int | None = None
    prune: bool = False
    topM: int | None = None
    tau: float = 0.0
    prune_every: int = 5
    max_probability_wires: int = 16

    def __post_init__(self):
        for name in ("k", "topM", "prune_every", "max_probability_wires"):
            value = getattr(self, name)
            if value is None and name in {"k", "topM"}:
                continue
            if isinstance(value, bool) or not isinstance(value, Integral) or value < (1 if name == "topM" else 0):
                raise ValueError(f"{name} must be a {'positive' if name == 'topM' else 'nonnegative'} integer.")
        if not isinstance(self.prune, bool):
            raise ValueError("prune must be a boolean.")
        if not math.isfinite(self.tau) or self.tau < 0:
            raise ValueError("tau must be finite and nonnegative.")


def _sentence(obs, max_wires):
    """Convert Pauli arithmetic and computational-basis projectors without matrices."""
    if obs.pauli_rep is not None:
        return obs.pauli_rep
    if isinstance(obs, qml.Projector):
        bits = qml.math.toarray(obs.data[0])
        if bits.shape != (len(obs.wires),) or not np.all((bits == 0) | (bits == 1)):
            raise ValueError("PauliPropagationSimulator supports computational-basis projectors only.")
        if len(obs.wires) > max_wires:
            raise MemoryError("Projector expansion exceeds max_probability_wires.")
        result = PauliSentence({PauliWord({}): 1.0})
        for w, bit in zip(obs.wires, bits):
            result = result @ PauliSentence({PauliWord({}): .5, PauliWord({w: "Z"}): .5 * (1 - 2 * int(bit))})
        return result
    if isinstance(obs, qml.ops.op_math.SProd):
        return obs.scalar * _sentence(obs.base, max_wires)
    if isinstance(obs, qml.ops.op_math.Sum):
        result = PauliSentence()
        for term in obs:
            result = result + _sentence(term, max_wires)
        return result
    if isinstance(obs, qml.ops.op_math.Prod):
        result = PauliSentence({PauliWord({}): 1.0})
        for term in obs:
            result = result @ _sentence(term, max_wires)
        return result
    if isinstance(obs, qml.ops.op_math.LinearCombination):
        result = PauliSentence()
        for c, term in zip(*obs.terms()):
            result = result + c * _sentence(term, max_wires)
        return result
    raise ValueError("PauliPropagationSimulator requires Pauli observables or computational-basis projectors; "
                     f"received {obs.name}.")


def _walsh_probabilities(moments):
    """p(b)=2**(-m) sum_S (-1)**(b.S) <Z_S>, in measurement wire order."""
    result = moments
    width = 1
    while width < len(moments):
        pairs = result.reshape(-1, 2, width)
        result = torch.stack((pairs[:, 0] + pairs[:, 1], pairs[:, 0] - pairs[:, 1]), dim=1).reshape(-1)
        width *= 2
    return result / len(moments)


class PauliPropagationSimulator(qml.devices.Device):
    """Analytic Pauli propagation usable in every QASP device configuration.

    Use device="PauliPropagationSimulator", shots=None, diff_method="backprop"
    with Torch models. This device supports Pauli expectations/variances and
    computational-basis probabilities. It expands decomposable unitary gates to
    Clifford gates and Pauli rotations, starting from |0...0>. BasisState is
    decomposed to gates. State/density outputs, noise, mid-circuit measurements,
    finite shots and non-Torch backpropagation are deliberately unsupported.

    Both the device class and an instance can be passed to QASP. Legacy
    DifferentiablePauliProp and lightning.pp interfaces remain available.
    """
    name = "PauliPropagationSimulator"

    def __init__(self, wires=None, shots=None, *, options=None, **kwargs):
        if shots is not None:
            raise ValueError("PauliPropagationSimulator is analytic; use shots=None.")
        if options is not None and kwargs:
            raise ValueError("Supply options or keyword options, not both.")
        self.options = options if isinstance(options, PauliPropagationOptions) else PauliPropagationOptions(**(options or kwargs))
        super().__init__(wires=wires, shots=None)

    def supports_derivatives(self, execution_config=None, circuit=None):
        if circuit is not None and circuit.shots:
            return False
        if execution_config is None:
            return True
        interface = getattr(execution_config.interface, "value", execution_config.interface)
        return (execution_config.gradient_method in {None, "best", "backprop"}
                and interface in {None, "auto", "torch", "numpy"})

    def preprocess(self, execution_config=None):
        from pennylane.devices import ExecutionConfig
        from pennylane.transforms.core import TransformProgram
        config = execution_config or ExecutionConfig()
        if config.gradient_method == "best":
            config = replace(config, gradient_method="backprop")
        interface = getattr(config.interface, "value", config.interface)
        if config.gradient_method == "backprop" and interface not in {None, "auto", "torch", "numpy"}:
            raise ValueError("PauliPropagationSimulator backpropagation requires interface='torch'.")
        config = replace(config, use_device_gradient=False, grad_on_execution=False,
                         use_device_jacobian_product=False)
        program = TransformProgram()
        program.add_transform(qml.transforms.broadcast_expand)
        return program, config

    @qml.QueuingManager.stop_recording()
    def _execute_one(self, tape):
        if tape.shots:
            raise ValueError("PauliPropagationSimulator is analytic; use shots=None.")
        wires = tuple(self.wires) if self.wires is not None else tuple(tape.wires)
        if not set(tape.wires) <= set(wires):
            raise ValueError("Circuit wires are not present on PauliPropagationSimulator.")
        if not wires:
            raise ValueError("PauliPropagationSimulator requires at least one wire.")
        tape = qml.map_wires(tape, {w: i for i, w in enumerate(wires)})[0][0]
        for op in tape.operations:
            if isinstance(op, (qml.operation.Channel, qml.measurements.MidMeasureMP)) or op.name in {"StatePrep", "QubitDensityMatrix"}:
                raise ValueError(f"PauliPropagationSimulator does not support {op.name}; use unitary gate preparation.")
        # Build all required moments together; the engine compiles the gates once.
        measurements, processors = [], []
        def add(sentence):
            if self.options.k is not None:
                sentence = PauliSentence({word: c for word, c in sentence.items()
                                          if len(word) <= self.options.k})
            index = len(measurements)
            measurements.append(qml.expval(sentence.operation(wire_order=range(len(wires)))))
            return index
        for meas in tape.measurements:
            if isinstance(meas, (qml.measurements.ExpectationMP, qml.measurements.VarianceMP)):
                sentence = _sentence(meas.obs, self.options.max_probability_wires)
                first = add(sentence)
                if isinstance(meas, qml.measurements.VarianceMP):
                    second = add(sentence @ sentence)
                    processors.append(lambda v, a=first, b=second: v[b] - v[a] ** 2)
                else:
                    processors.append(lambda v, a=first: v[a])
            elif isinstance(meas, qml.measurements.ProbabilityMP) and meas.obs is None:
                measured = tuple(meas.wires) or tuple(range(len(wires)))
                n = len(measured)
                if n > self.options.max_probability_wires:
                    raise MemoryError(f"Probability readout on {n} wires exceeds max_probability_wires="
                                      f"{self.options.max_probability_wires}; use sparse expectations or raise the explicit limit.")
                start = len(measurements)
                for mask in range(2**n):
                    word = PauliWord({w: "Z" for j, w in enumerate(measured) if mask & (1 << (n - 1 - j))})
                    add(PauliSentence({word: 1.0}))
                decoder = meas.decoder if isinstance(meas, ActionProbsMP) else None
                def probabilities(v, start=start, size=2**n, decoder=decoder):
                    p = _walsh_probabilities(v[start:start+size])
                    return decoder.aggregate(p) if decoder else p
                processors.append(probabilities)
            else:
                raise ValueError("PauliPropagationSimulator supports analytic Pauli expval/var and "
                                 "computational-basis probs; state, sample and other measurements are unsupported.")
        if not measurements:
            raise ValueError("PauliPropagationSimulator requires a measurement.")
        expanded = qml.tape.QuantumScript(tape.operations, measurements)
        engine = DifferentiablePauliProp(topM=self.options.topM, tau=self.options.tau,
                                        prune_every=self.options.prune_every)
        values = engine.execute_from_tape(expanded, k=self.options.k, prune=self.options.prune)
        # Initial weight projection can remove all dependence on a trainable
        # observable coefficient. Keep its zero derivative attached as well.
        if self.options.k is not None:
            for p in tape.get_parameters(trainable_only=False):
                if isinstance(p, torch.Tensor) and p.requires_grad:
                    values = values + 0 * p.real.sin().sum()
        result = tuple(fn(values) for fn in processors)
        if not any(isinstance(p, torch.Tensor) for p in tape.get_parameters(trainable_only=False)):
            result = tuple(v.detach().cpu().numpy() for v in result)
        return result[0] if len(result) == 1 else result

    def execute(self, circuits, execution_config=None):
        single = isinstance(circuits, qml.tape.QuantumScript)
        tapes = (circuits,) if single else circuits
        result = tuple(self._execute_one(tape) for tape in tapes)
        return result[0] if single else result
