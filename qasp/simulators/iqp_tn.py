"""Exact doubled tensor networks for locally dressed product-basis IQP circuits.

Structural paths and optional inference kernels/constants are cached. Every
execution rebuilds all parameter-dependent tensor values, so
input encoding, shared angles, zero angles and higher Torch derivatives survive.
No truncation, parameter pruning, sampling approximation or statevector is used
on the selected TN route. See docs/iqp_tn.md for the precise contract and costs.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
import math
import warnings

import numpy as np
import pennylane as qml
import torch
from .readout import ActionProbsMP
from ._inference import InferenceCache


class UnsupportedIQP(ValueError):
    """The actual tape/readout is outside the supported commuting-layer contract."""


@dataclass(frozen=True)
class IQPTNOptions:
    auto_max_flops: int = 100_000_000
    auto_max_memory_bytes: int = 256 * 1024**2
    hard_max_flops: int | None = None
    hard_max_memory_bytes: int | None = None
    fallback_max_wires: int = 24
    optimize: str = "greedy"
    precision: str = "complex128"
    warn_expensive: bool = True
    inference_execution: str = "eager"
    inference_cache_size: int = 32
    inference_constant_bytes: int = 16 * 1024**2

    def __post_init__(self):
        if self.inference_execution not in {"eager", "cached", "torchscript"}:
            raise ValueError("inference_execution must be eager, cached or torchscript.")
        for name in ("inference_cache_size", "inference_constant_bytes"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        for name in ("auto_max_flops", "auto_max_memory_bytes", "hard_max_flops", "hard_max_memory_bytes"):
            value = getattr(self, name)
            if value is None and name.startswith("hard_"):
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
        if not isinstance(self.fallback_max_wires, int) or self.fallback_max_wires < 1:
            raise ValueError("fallback_max_wires must be a positive integer.")
        if self.precision not in {"complex64", "complex128"}:
            raise ValueError("precision must be complex64 or complex128.")
        if self.optimize not in {"greedy", "auto", "auto-hq"}:
            raise ValueError("Use a bounded heuristic planner: greedy, auto or auto-hq.")


@dataclass(frozen=True)
class IQPTNReport:
    mode: str
    selected: str
    eligible: bool
    reason: str
    estimated_flops: int = 0
    largest_intermediate_bytes: int = 0
    estimated_memory_bytes: int = 0
    batch_size: int = 1
    requires_grad: bool = False
    precision: str = "complex128"
    networks: int = 0
    output_elements: int = 0
    estimate_scope: str = ("Conservative tensor-storage proxy including inputs, saved intermediates, "
                           "batch and backward; excludes framework/planner/allocator overhead. "
                           "Not an OS memory cap or an asymptotic complexity certificate.")


@lru_cache(maxsize=512)
def _path(structure, output, optimize):
    try:
        import opt_einsum as oe
    except ImportError as exc:
        raise UnsupportedIQP("IQPTNSimulator requires opt_einsum>=3.3; install QASP's dependencies.") from exc
    labels = list(dict.fromkeys(i for _, ids in structure for i in ids))
    symbols = {label: oe.get_symbol(j) for j, label in enumerate(labels)}
    equation = ",".join("".join(symbols[i] for i in ids) for _, ids in structure)
    equation += "->" + "".join(symbols[i] for i in output)
    shapes = [shape for shape, _ in structure]
    path, info = oe.contract_path(equation, *shapes, shapes=True, optimize=optimize)
    expr = oe.contract_expression(equation, *shapes, optimize=path)
    return expr, int(info.opt_cost), int(info.largest_intermediate), sum(int(s) for s in info.size_list)


class _Network:
    def __init__(self, ctx):
        self.ctx = ctx
        self.terms = []
        self.output = ()
        self.serial = 0

    def label(self):
        self.serial += 1
        return ("aux", self.serial)

    def add(self, shape, labels, value):
        self.terms.append((tuple(shape), tuple(labels), value))

    def constant(self, value, labels):
        array = np.asarray(value)
        self.add(array.shape, labels, lambda a=array: self.ctx.constant(a))

    def plan(self, options):
        return _path(tuple((shape, labels) for shape, labels, _ in self.terms), tuple(self.output), options.optimize)

    def evaluate(self, options):
        expr, _, _, _ = self.plan(options)
        operands = tuple(factory() for _, _, factory in self.terms)
        if self.ctx.cache is not None and options.inference_execution == "torchscript":
            return self.ctx.cache.contract(expr, operands,
                tuple((shape, labels) for shape, labels, _ in self.terms), tuple(self.output), options.optimize)
        return expr(*operands, backend="torch").real


class _Context:
    def __init__(self, tape, options, cache=None):
        self.cache = cache
        params = [p for p in tape.get_parameters(trainable_only=False) if isinstance(p, torch.Tensor)]
        self.torch_result = bool(params)
        self.device = params[0].device if params else torch.device("cpu")
        if any(p.device != self.device for p in params):
            raise UnsupportedIQP("All quantum parameters must share a Torch device.")
        self.dtype = getattr(torch, options.precision)

    def tensor(self, value):
        return torch.as_tensor(value, dtype=self.dtype, device=self.device)

    def static(self, key, factory, key_bytes=0):
        if self.cache is None:
            return factory()
        return self.cache.constant((key, self.dtype, self.device, torch.is_inference_mode_enabled()), factory, key_bytes)

    def constant(self, array):
        if self.cache is None:
            return self.tensor(array)
        key = ("array", array.dtype.str, array.shape, array.tobytes())
        return self.static(key, lambda: self.tensor(array), len(key[-1]))

    def local(self, ops):
        result = self.static(("identity",), lambda: torch.eye(2, dtype=self.dtype, device=self.device))
        for op in ops:
            result = self.tensor(qml.matrix(op)) @ result
        return result


_LOCAL = {"Hadamard", "PauliX", "PauliY", "PauliZ", "Identity", "S", "T", "SX",
          "RX", "RY", "RZ", "Rot", "PhaseShift", "QubitUnitary", "PauliRot", "MultiRZ"}
_LOCAL_DIAG = {"RZ", "PhaseShift", "PauliZ", "S", "T", "Identity"}


def _diagonal(op):
    if op.name in {"IsingZZ", "MultiRZ", "CZ", "ControlledPhaseShift"}:
        return True
    return op.name == "PauliRot" and set(op.hyperparameters["pauli_word"]) <= {"I", "Z"}


@qml.QueuingManager.stop_recording()
def _operations(tape):
    """Expand known embeddings and normalize inverses without detaching angles."""
    result = []
    def append(op):
        if isinstance(op, qml.ops.op_math.Adjoint):
            base = op.base
            if isinstance(base, qml.ops.op_math.Adjoint):
                append(base.base)
                return
            if base.name in {"AngleEmbedding", "IQPEmbedding"}:
                for child in reversed(base.decomposition()):
                    append(qml.adjoint(child))
                return
            if base.has_adjoint:
                inverse = base.adjoint()
                if not isinstance(inverse, qml.ops.op_math.Adjoint):
                    append(inverse)
                    return
        if op.name in {"AngleEmbedding", "IQPEmbedding"}:
            for child in op.decomposition():
                append(child)
        elif op.name in {"Barrier", "GlobalPhase"}:
            return
        else:
            result.append(op)
    for op in tape.operations:
        append(op)
    # In an IQP fidelity overlap the closing H layer meets the opening H
    # layer of the inverse. Cancel only these structural identities, never
    # rotations whose current trainable angles merely happen to be equal.
    last, removed = {}, set()
    for pos, op in enumerate(result):
        if op.name == "Hadamard":
            wire = op.wires[0]
            previous = last.get(wire)
            if previous is not None and result[previous].name == "Hadamard":
                removed.update((previous, pos))
                last[wire] = None
                continue
        for wire in op.wires:
            last[wire] = pos
    return [op for pos, op in enumerate(result) if pos not in removed]


def _pauli_axes(op):
    """Pauli-rotation generator, without inspecting or detaching its angle."""
    if op.name == "PauliRot":
        return tuple(zip(op.wires, op.hyperparameters["pauli_word"]))
    if op.name in {"IsingXX", "IsingYY", "IsingZZ", "MultiRZ"}:
        axis = {"IsingXX": "X", "IsingYY": "Y"}.get(op.name, "Z")
        return tuple((w, axis) for w in op.wires)
    return None


@qml.QueuingManager.stop_recording()
def _product_basis_canonical(ops, wires):
    """Reduce interactions sharing a per-wire Pauli basis to the Z network.

    With B_w P_w B_w^dagger = Z, absorb B_w into the local preparation
    and B_w^dagger into the readout dressing. Only local 2x2 basis changes
    are added; all current rotation parameters pass unchanged to Z rotations.
    Pairwise commutation alone is insufficient (for example XX and YY).
    """
    axes, first, last, interactions = {}, {}, {}, {}
    for pos, op in enumerate(ops):
        if not set(op.wires) <= set(wires):
            raise UnsupportedIQP("Operation addresses wires outside the device.")
        if len(op.wires) == 1:
            continue
        generator = _pauli_axes(op)
        if generator is None:
            if op.name not in {"CZ", "ControlledPhaseShift"}:
                raise UnsupportedIQP(f"Unsupported operation {op.name}; require interactions diagonal in one product Pauli basis.")
            generator = tuple((w, "Z") for w in op.wires)
        active = [(w, axis) for w, axis in generator if axis != "I"]
        for w, axis in active:
            if w in axes and axes[w] != axis:
                raise UnsupportedIQP(f"Interactions require incompatible Pauli axes on wire {w}; no shared product Pauli basis. Pairwise commutation alone is insufficient.")
            axes[w] = axis
            first.setdefault(w, pos)
            last[w] = pos
        if op.name in {"CZ", "ControlledPhaseShift"}:
            interactions[pos] = op
        elif active:
            interactions[pos] = qml.MultiRZ(op.data[0], wires=[w for w, _ in active])
        else:
            interactions[pos] = None  # identity generator is a global phase

    pre, post, phases = {w: [] for w in wires}, {w: [] for w in wires}, []
    for pos, op in enumerate(ops):
        if pos in interactions:
            if interactions[pos] is not None:
                phases.append(interactions[pos])
            continue
        base = op.base if isinstance(op, qml.ops.op_math.Adjoint) else op
        if base.name not in _LOCAL or len(op.wires) != 1:
            raise UnsupportedIQP(f"Unsupported local operation {op.name}.")
        w = op.wires[0]
        if w not in first or pos < first[w]:
            pre[w].append(op)
        elif pos > last[w]:
            post[w].append(op)
        elif axes[w] == "Z" and (base.name in _LOCAL_DIAG or _diagonal(op)):
            phases.append(op)
        else:
            # Inside the interaction interval a single-wire gate must commute
            # with that wire's fixed generator, independent of parameter values.
            axis = {"RX": "X", "RY": "Y", "RZ": "Z",
                    "PauliX": "X", "PauliY": "Y", "PauliZ": "Z",
                    "SX": "X", "MultiRZ": "Z"}.get(base.name)
            if base.name == "PauliRot":
                axis = base.hyperparameters["pauli_word"]
            if base.name == "Identity" or axis == "I":
                continue
            if axis != axes[w]:
                raise UnsupportedIQP(f"Noncommuting {op.name} between interactions on wire {w} in the {axes[w]} basis.")
            if base.name.startswith("Pauli") and base.name != "PauliRot":
                phases.append(qml.PauliZ(w))
            else:
                angle = math.pi/2 if base.name == "SX" else base.data[0]
                if isinstance(op, qml.ops.op_math.Adjoint):
                    angle = -angle
                phases.append(qml.RZ(angle, w))
    for w, axis in axes.items():
        if axis == "X":
            pre[w].append(qml.Hadamard(w))
            post[w].insert(0, qml.Hadamard(w))
        elif axis == "Y":
            pre[w].append(qml.RX(math.pi/2, w))
            post[w].insert(0, qml.RX(-math.pi/2, w))
    return pre, post, phases


def _canonical(tape, wires):
    ops = _operations(tape)
    # Keep the established Z-diagonal route unchanged. Other product Pauli
    # bases use an exact local conversion, never a full-state simulation.
    if any(len(op.wires) > 1 and (generator := _pauli_axes(op)) is not None
           and any(axis in {"X", "Y"} for _, axis in generator) for op in ops):
        return _product_basis_canonical(ops, wires)
    first, last = {}, {}
    for pos, op in enumerate(ops):
        if _diagonal(op) and len(op.wires) > 1:
            for w in op.wires:
                first.setdefault(w, pos)
                last[w] = pos
        elif len(op.wires) != 1:
            raise UnsupportedIQP(f"Unsupported operation {op.name} on {list(op.wires)}; require diagonal Z interactions and local unitaries.")
    pre = {w: [] for w in wires}
    post = {w: [] for w in wires}
    phases = []
    for pos, op in enumerate(ops):
        if not set(op.wires) <= set(wires):
            raise UnsupportedIQP("Operation addresses wires outside the device.")
        if _diagonal(op) and len(op.wires) > 1:
            phases.append(op)
            continue
        base = op.base if isinstance(op, qml.ops.op_math.Adjoint) else op
        if base.name not in _LOCAL or len(op.wires) != 1:
            raise UnsupportedIQP(f"Unsupported operation {op.name}; no implicit large decomposition is attempted.")
        w = op.wires[0]
        if w not in first or pos < first[w]:
            pre[w].append(op)
        elif pos > last[w]:
            post[w].append(op)
        elif base.name in _LOCAL_DIAG or _diagonal(op):
            phases.append(op)
        else:
            raise UnsupportedIQP(f"Noncommuting {op.name} between diagonal interactions on wire {w}; repeated IQP layers are not a single supported layer.")
    return pre, post, phases


def _base_network(ctx, wires, canonical, measured=(), pauli=None, conditions=None):
    net = _Network(ctx)
    pre, post, phases = canonical
    measured = tuple(measured)
    conditions = conditions or {}
    for w in wires:
        ket, bra, bit = ("ket", w), ("bra", w), ("bit", w)
        def density(w=w):
            v = ctx.local(pre[w])[:, 0]
            return v[:, None] * v.conj()[None, :]
        net.add((2, 2), (ket, bra), density)
        if w in measured:
            def readout(w=w):
                b = ctx.local(post[w])
                return torch.einsum("yx,yz->xzy", b, b.conj())
            net.add((2, 2, 2), (ket, bra, bit), readout)
        else:
            def observable(w=w):
                b = ctx.local(post[w])
                char = (pauli or {}).get(w, "I")
                matrices = {"I": [[1, 0], [0, 1]], "X": [[0, 1], [1, 0]],
                            "Y": [[0, -1j], [1j, 0]], "Z": [[1, 0], [0, -1]]}
                return (b.conj().T @ ctx.tensor(matrices[char]) @ b).T
            net.add((2, 2), (ket, bra), observable)
        if w in conditions:
            net.constant([1-conditions[w], conditions[w]], (bit,))
    parity = np.zeros((2, 2, 2))
    for a in range(2):
        for b in range(2):
            parity[a, b, a ^ b] = 1
    for op in phases:
        for side in ("ket", "bra"):
            conj = side == "bra"
            if op.name in {"IsingZZ", "MultiRZ", "PauliRot"}:
                support = list(op.wires)
                if op.name == "PauliRot":
                    support = [w for w, c in zip(op.wires, op.hyperparameters["pauli_word"]) if c == "Z"]
                if not support:  # pure global phase cancels exactly, including its derivative
                    continue
                acc = (side, support[0])
                for w in support[1:]:
                    nxt = net.label()
                    net.constant(parity, (acc, (side, w), nxt))
                    acc = nxt
                def phase(op=op, conj=conj):
                    values = torch.exp(-0.5j * ctx.tensor(op.data[0]) * ctx.tensor([1., -1.]))
                    return values.conj() if conj else values
                net.add((2,), (acc,), phase)
            else:
                def phase(op=op, conj=conj):
                    values = torch.diagonal(ctx.tensor(qml.matrix(op))).reshape((2,)*len(op.wires))
                    return values.conj() if conj else values
                net.add((2,)*len(op.wires), tuple((side, w) for w in op.wires), phase)
    return net


def _count_bit(net, bits, majority):
    """Count automaton with absorbing strict-majority state, or a two-state XOR."""
    cap = len(bits)//2+1 if majority else 1
    left = net.label()
    net.constant([1.], (left,))
    size = 1
    for j, bit in enumerate(bits):
        right = net.label()
        new_size = min(j+2, cap+1) if majority else 2
        def transition(size=size, new_size=new_size):
            def make():
                a = torch.arange(size, device=net.ctx.device)[:, None]
                b = torch.arange(2, device=net.ctx.device)[None, :]
                dest = (a+b).clamp_max(cap) if majority else (a+b).remainder(2)
                return torch.nn.functional.one_hot(dest, new_size).to(net.ctx.dtype)
            return net.ctx.static(("count", size, new_size, cap, majority), make)
        net.add((size, 2, new_size), (left, bit, right), transition)
        left, size = right, new_size
    logical = net.label()
    values = np.zeros((size, 2))
    for count in range(size):
        values[count, int(count == cap) if majority else count] = 1
    net.constant(values, (left, logical))
    return logical


def _decoder_network(net, wires, decoder):
    decoder.validate(len(wires))
    bits = [("bit", w) for w in wires]
    if decoder.policy == "majority":
        blocks = (tuple(range(len(bits))),)
    elif decoder.policy == "majority-modulo":
        blocks = decoder.blocks
    else:
        blocks = tuple(tuple(range(len(bits)-j)) for j in range(math.ceil(math.log2(decoder.n_actions))))
    logical = [_count_bit(net, [bits[p] for p in block], decoder.policy != "parity-like") for block in blocks]
    left = net.label()
    net.constant([1.], (left,))
    size = 1
    for bit in logical:
        right = net.label()
        def modulo(size=size):
            def make():
                a = torch.arange(size, device=net.ctx.device)[:, None]
                b = torch.arange(2, device=net.ctx.device)[None, :]
                return torch.nn.functional.one_hot((2*a+b) % decoder.n_actions, decoder.n_actions).to(net.ctx.dtype)
            return net.ctx.static(("modulo", size, decoder.n_actions), make)
        net.add((size, 2, decoder.n_actions), (left, bit, right), modulo)
        left, size = right, decoder.n_actions
    # Readout wires absent from all blocks are marginalized by summing their bit index.
    net.output = (left,)
    return net


def _less_than(net, wires, bound):
    left = net.label()
    net.constant([0., 1., 0.], (left,))  # less, equal, greater
    for pos, w in enumerate(wires):
        right = net.label()
        bit = (bound >> (len(wires)-pos-1)) & 1
        t = np.zeros((3, 2, 3))
        for state in range(3):
            for value in range(2):
                dest = state if state != 1 else (0 if value < bit else 2 if value > bit else 1)
                t[state, value, dest] = 1
        net.constant(t, (left, ("bit", w), right))
        left = right
    net.constant([1., 0., 0.], (left,))
    return net


class IQPTNSimulator(qml.devices.Device):
    """PennyLane-compatible exact TN device, accepted by all QASP device factories.

    ``mode='auto'`` selects the TN per actual tape and readout or uses default.qubit
    with an inspectable reason. The default explicit mode bypasses auto budgets.
    Only optional ``hard_max_*`` options reject an expensive supported forced plan.
    Analytic Torch backprop is supported; finite shots are nondifferentiable.
    """
    name = "IQPTNSimulator"

    def __init__(self, wires=None, shots=None, *, mode="explicit", options=None, seed=None, **kwargs):
        if mode not in {"auto", "explicit"}:
            raise ValueError("mode must be auto or explicit.")
        if options is not None and kwargs:
            raise ValueError("Supply either options=IQPTNOptions(...) or option keywords.")
        self.options = options if isinstance(options, IQPTNOptions) else IQPTNOptions(**(options or kwargs))
        self.mode = mode
        self.last_report = None
        self.last_reports = ()
        self._rng = np.random.default_rng(seed)
        self._inference_cache = InferenceCache()
        super().__init__(wires=wires, shots=shots)

    def clear_inference_cache(self):
        """Release cached inference kernels/constants and reset their diagnostics."""
        self._inference_cache.clear()

    def inference_cache_info(self):
        """Return cache counters/storage, without exposing cached tensors."""
        return self._inference_cache.info()

    def supports_derivatives(self, execution_config=None, circuit=None):
        if execution_config is None:
            return True
        if self.mode == "auto" and execution_config.gradient_method == "adjoint":
            return qml.devices.DefaultQubit(wires=self.wires).supports_derivatives(execution_config, circuit)
        return execution_config.gradient_method in {"best", "backprop", None} and not (circuit is not None and circuit.shots)

    def preprocess(self, execution_config=None):
        from pennylane.devices import ExecutionConfig
        from pennylane.transforms.core import TransformProgram
        config = execution_config or ExecutionConfig()
        if self.mode == "auto" and config.gradient_method == "adjoint":
            return qml.devices.DefaultQubit(wires=self.wires).preprocess(config)
        if config.gradient_method == "best":
            config = replace(config, gradient_method="backprop")
        config = replace(config, use_device_gradient=False, grad_on_execution=False, use_device_jacobian_product=False)
        program = TransformProgram()
        program.add_transform(qml.transforms.broadcast_expand)
        return program, config

    def _compile(self, tape):
        if tape.shots.has_partitioned_shots:
            raise UnsupportedIQP("Shot vectors are not supported by IQPTNSimulator.")
        wires = tuple(self.wires) if self.wires is not None else tuple(tape.wires)
        cache = None
        if not torch.is_grad_enabled() and self.options.inference_execution != "eager":
            self._inference_cache.configure(self.options.inference_cache_size, self.options.inference_constant_bytes)
            cache = self._inference_cache
        ctx = _Context(tape, self.options, cache)
        canonical = _canonical(tape, wires)
        networks, outputs = [], []
        def add(net):
            networks.append(net)
            return len(networks)-1
        def base(measured=(), pauli=None, conditions=None):
            return _base_network(ctx, wires, canonical, measured, pauli, conditions)
        for m in tape.measurements:
            mw = tuple(m.wires) or wires
            if not set(mw) <= set(wires):
                raise UnsupportedIQP("Readout wires are not present on the device.")
            if isinstance(m, ActionProbsMP):
                if m.decoder.policy == "contiguous-like":
                    nstates = 2**len(mw)
                    block, remainder = divmod(nstates, m.decoder.n_actions)
                    indices = [add(_less_than(base(mw), mw, j*block+min(j,remainder)))
                               for j in range(1, m.decoder.n_actions)]
                    outputs.append(("contiguous", indices, m.decoder.n_actions))
                else:
                    outputs.append(("direct", [add(_decoder_network(base(mw), mw, m.decoder))], m.decoder.n_actions))
            elif isinstance(m, qml.measurements.ProbabilityMP) and m.obs is None:
                net = base(mw)
                net.output = tuple(("bit", w) for w in mw)
                outputs.append(("prob", [add(net)], 2**len(mw)))
            elif isinstance(m, (qml.measurements.SampleMP, qml.measurements.CountsMP)) and m.obs is None:
                if not tape.shots:
                    raise UnsupportedIQP("Samples/counts require a positive shot count.")
                if isinstance(m, qml.measurements.CountsMP) and m.all_outcomes:
                    raise UnsupportedIQP("counts(all_outcomes=True) has exponential output; use all_outcomes=False or a dense probs request.")
                # Prefix constraints have the same shapes for every sampled value.
                indices = []
                for j, w in enumerate(mw):
                    net = base(mw[:j+1], conditions={v: 0 for v in mw[:j]})
                    net.output = (("bit", w),)
                    indices.append(add(net))
                outputs.append(("counts" if isinstance(m, qml.measurements.CountsMP) else "samples", indices, (mw, m.all_outcomes if hasattr(m, "all_outcomes") else False)))
            elif isinstance(m, (qml.measurements.ExpectationMP, qml.measurements.VarianceMP)):
                if tape.shots:
                    raise UnsupportedIQP("Finite-shot observables are not supported; use analytic expectations or explicit samples.")
                sentence = m.obs.pauli_rep
                if sentence is None:
                    if not isinstance(m, qml.measurements.ExpectationMP):
                        raise UnsupportedIQP("Non-Pauli variance observables are not supported.")
                    # Basis projectors have an exponential Pauli expansion but
                    # compact Born boundary conditions. Share the VN parser;
                    # never expand their Pauli sentence or construct a matrix.
                    from ._iqp_observables import compile_diagonal_observable
                    observable = compile_diagonal_observable(
                        m.obs, wires, getattr(self.options, "vn_max_observable_components", 100_000))
                    group = []
                    for component in observable.components:
                        conditions, conflict = {}, False
                        for indices, bits in component.projectors:
                            for i, bit in zip(indices, bits):
                                wire = wires[i]
                                conflict |= wire in conditions and conditions[wire] != bit
                                conditions[wire] = bit
                        sign = (-1)**sum(conditions.get(wires[i], 0) for i in component.z_indices)
                        pauli = {wires[i]: "Z" for i in component.z_indices if wires[i] not in conditions}
                        coefficient = component.coefficient * (0 if conflict else sign)
                        group.append((coefficient, add(base(tuple(conditions), pauli=pauli, conditions=conditions))))
                    outputs.append(("expectation", [group], 1))
                    continue
                groups = []
                for terms in ([sentence, sentence @ sentence] if isinstance(m, qml.measurements.VarianceMP) else [sentence]):
                    groups.append([(coefficient, add(base(pauli=dict(word)))) for word, coefficient in terms.items()])
                outputs.append(("variance" if len(groups) == 2 else "expectation", groups, 1))
            else:
                raise UnsupportedIQP(f"Unsupported measurement {type(m).__name__}; use action probabilities, basis probabilities/marginals, Pauli moments, samples or counts.")
        return ctx, canonical, networks, outputs, wires

    def _report(self, tape, compiled, batch_size, requires_grad):
        _, _, nets, outputs, _ = compiled
        flops = storage = largest = 0
        for net in nets:
            _, cost, peak, retained = net.plan(self.options)
            flops += cost
            storage += sum(math.prod(shape) for shape, _, _ in net.terms) + retained
            largest = max(largest, peak)
        shot_factor = tape.shots.total_shots if any(o[0] in {"samples", "counts"} for o in outputs) else 1
        flops *= shot_factor
        bytes_per = 16 if self.options.precision == "complex128" else 8
        # Complex multiplication is costed at four real planner operations.
        flops *= 4 * batch_size * (3 if requires_grad else 1)
        memory = (storage + largest) * bytes_per * batch_size * (3 if requires_grad else 1)
        elems = sum((len(o[2][0]) * tape.shots.total_shots if o[0] in {"samples", "counts"} else o[2]) for o in outputs)
        memory += elems * bytes_per * batch_size * (3 if requires_grad else 1)
        over = flops > self.options.auto_max_flops or memory > self.options.auto_max_memory_bytes
        selected = "default.qubit" if self.mode == "auto" and over else self.name
        reason = "Automatic contraction budget exceeded." if over else "Supported circuit and readout within automatic budgets."
        if self.mode == "explicit":
            reason = "Explicit exact contraction; automatic budgets bypassed." + (" Estimated cost exceeds automatic budgets." if over else "")
        return IQPTNReport(self.mode, selected, True, reason, flops, largest*bytes_per, memory,
                           batch_size, requires_grad, self.options.precision, len(nets), elems)

    def plan(self, tape, *, batch_size=1, requires_grad=False, interface="torch"):
        """Inspect shapes/costs without executing or allocating numerical network tensors."""
        try:
            if interface not in {"torch", "numpy", None, "auto"}:
                raise UnsupportedIQP(f"TN differentiation supports Torch; interface={interface!r} requires fallback.")
            if tape.batch_size is not None:
                tapes, _ = qml.transforms.broadcast_expand(tape)
                tape = tapes[0]
                batch_size *= len(tapes)
            wires = tuple(self.wires) if self.wires is not None else tuple(tape.wires)
            _canonical(tape, wires)
            nw = len(wires)
            output_size = sum(m.decoder.n_actions if isinstance(m, ActionProbsMP)
                              else 2**(len(m.wires) or nw) if isinstance(m, qml.measurements.ProbabilityMP)
                              else (len(m.wires) or nw)*(tape.shots.total_shots or 0)
                              if isinstance(m, (qml.measurements.SampleMP, qml.measurements.CountsMP))
                              else 1 for m in tape.measurements)
            output_bytes = output_size * (16 if self.options.precision == "complex128" else 8) * batch_size * (3 if requires_grad else 1)
            if self.mode == "auto" and output_bytes > self.options.auto_max_memory_bytes:
                return IQPTNReport(self.mode, "default.qubit", True,
                    "Requested output alone exceeds the automatic memory budget.",
                    estimated_memory_bytes=output_bytes, batch_size=batch_size, requires_grad=requires_grad,
                    precision=self.options.precision, output_elements=output_size)
            if self.options.hard_max_memory_bytes is not None and output_bytes > self.options.hard_max_memory_bytes:
                raise MemoryError("Requested output exceeds explicit hard limit before planning.")
            return self._report(tape, self._compile(tape), batch_size, requires_grad)
        except UnsupportedIQP as exc:
            return IQPTNReport(self.mode, "default.qubit" if self.mode == "auto" else "error", False,
                               str(exc), batch_size=batch_size, requires_grad=requires_grad, precision=self.options.precision)

    def _fallback(self, tape, config, reason):
        n = len(self.wires) if self.wires is not None else len(tape.wires)
        if n > self.options.fallback_max_wires:
            raise RuntimeError(f"Auto fallback requires a {n}-wire statevector, exceeding fallback_max_wires={self.options.fallback_max_wires}. {reason}")
        measurements = [qml.probs(wires=m.wires) if isinstance(m, ActionProbsMP) else m for m in tape.measurements]
        raw = tape.copy(measurements=measurements)
        dev = qml.device("default.qubit", wires=self.wires, shots=tape.shots, seed=int(self._rng.integers(2**31)))
        program, cfg = dev.preprocess(config)
        tapes, fn = program((raw,))
        result = fn(dev.execute(tapes, cfg))[0]
        values = (result,) if len(measurements) == 1 else result
        decoded = []
        for m, value in zip(tape.measurements, values):
            if isinstance(m, ActionProbsMP):
                if not isinstance(value, torch.Tensor):
                    # Differentiable non-Torch interfaces cannot pass through torch.
                    states = np.arange(2**len(m.wires))
                    bits = (states[:, None] >> np.arange(len(m.wires)-1, -1, -1)) & 1
                    indices = m.decoder.decode(bits).numpy()
                    value = qml.math.stack([qml.math.sum(value[..., indices == a], axis=-1) for a in range(m.decoder.n_actions)], axis=-1)
                else:
                    value = m.decoder.aggregate(value)
            decoded.append(value)
        return decoded[0] if len(decoded) == 1 else tuple(decoded)

    def _evaluate(self, tape, compiled):
        ctx, canonical, nets, outputs, wires = compiled
        # Sample-prefix networks are evaluated only inside the conditional sampler.
        sample_ids = {i for kind, ids, _ in outputs if kind in {"samples", "counts"} for i in ids}
        values = {i: net.evaluate(self.options) for i, net in enumerate(nets) if i not in sample_ids}
        results = []
        for kind, ids, extra in outputs:
            if kind in {"direct", "prob"}:
                value = values[ids[0]].reshape(-1)
            elif kind == "contiguous":
                cdf = torch.stack([ctx.tensor(0.).real, *[values[i] for i in ids], ctx.tensor(1.).real])
                value = cdf[1:] - cdf[:-1]
            elif kind in {"expectation", "variance"}:
                moments = [sum((ctx.tensor(c).real * values[i] for c, i in group), ctx.tensor(0.).real) for group in ids]
                value = moments[0] if kind == "expectation" else moments[1]-moments[0]**2
            else:
                mw, all_outcomes = extra
                if all_outcomes:
                    raise UnsupportedIQP("counts(all_outcomes=True) has exponential output; use counts with all_outcomes=False or a costed dense probs request.")
                rows = []
                with torch.no_grad():
                    for _ in range(tape.shots.total_shots):
                        conditions = {}
                        row = []
                        for j, w in enumerate(mw):
                            net = _base_network(ctx, wires, canonical, mw[:j+1], conditions=conditions)
                            net.output = (("bit", w),)
                            p = self._distribution(net.evaluate(self.options))
                            bit = int(self._rng.choice(2, p=p))
                            conditions[w] = bit
                            row.append(bit)
                        rows.append(row)
                value = torch.tensor(rows, dtype=torch.long, device=ctx.device)
                if kind == "counts":
                    labels, counts = np.unique(np.array(["".join(map(str, r)) for r in rows]), return_counts=True)
                    value = dict(zip(labels, counts))
            if tape.shots and kind in {"direct", "prob", "contiguous"}:
                counts = self._rng.multinomial(tape.shots.total_shots, self._distribution(value))
                value = torch.as_tensor(counts, device=ctx.device, dtype=value.dtype)/tape.shots.total_shots
            if not tape.shots and isinstance(value, torch.Tensor):
                for parameter in tape.get_parameters(trainable_only=False):
                    if isinstance(parameter, torch.Tensor) and parameter.requires_grad:
                        value = value + 0 * parameter.real.sum()
            if not ctx.torch_result and isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            results.append(value)
        return results[0] if len(results) == 1 else tuple(results)

    @staticmethod
    def _distribution(value):
        p = value.detach().cpu().double().numpy()
        total = p.sum()
        if not np.all(np.isfinite(p)) or total <= 0 or p.min() < -1e-6 * total:
            raise RuntimeError("Invalid conditional probability from contraction; no significant negative probabilities are clipped.")
        p = np.maximum(p, 0)
        return p/p.sum()

    def _adjoint_fallback(self, circuits, config, method):
        tapes = (circuits,) if isinstance(circuits, qml.tape.QuantumScript) else circuits
        reports = []
        for tape in tapes:
            n = len(self.wires) if self.wires is not None else len(tape.wires)
            if n > self.options.fallback_max_wires:
                raise RuntimeError("Adjoint auto fallback exceeds fallback_max_wires.")
            reports.append(IQPTNReport(self.mode, "default.qubit", False,
                "Requested adjoint differentiation uses default.qubit; TN supports Torch backprop.",
                batch_size=len(tapes), requires_grad=True, precision=self.options.precision))
        self.last_reports = tuple(reports)
        self.last_report = reports[-1]
        dev = qml.devices.DefaultQubit(wires=self.wires)
        return getattr(dev, method)(circuits, config)

    def compute_derivatives(self, circuits, execution_config=None):
        if self.mode != "auto":
            raise UnsupportedIQP("Explicit IQPTNSimulator uses Torch backprop, not device adjoint derivatives.")
        return self._adjoint_fallback(circuits, execution_config, "compute_derivatives")

    def execute_and_compute_derivatives(self, circuits, execution_config=None):
        if self.mode != "auto":
            raise UnsupportedIQP("Explicit IQPTNSimulator uses Torch backprop, not device adjoint derivatives.")
        return self._adjoint_fallback(circuits, execution_config, "execute_and_compute_derivatives")

    def execute(self, circuits, execution_config=None):
        from pennylane.devices import ExecutionConfig
        config = execution_config or ExecutionConfig()
        if self.mode == "auto" and config.gradient_method == "adjoint":
            return self._adjoint_fallback(circuits, config, "execute")
        single = isinstance(circuits, qml.tape.QuantumScript)
        tapes = (circuits,) if single else circuits
        results, reports = [], []
        for tape in tapes:
            interface = getattr(config.interface, "value", config.interface)
            grad = any(isinstance(p, torch.Tensor) and p.requires_grad for p in tape.get_parameters(trainable_only=False)) and torch.is_grad_enabled()
            report = self.plan(tape, batch_size=len(tapes), requires_grad=grad, interface=interface)
            reports.append(report)
            self.last_report, self.last_reports = report, tuple(reports)
            if not report.eligible and self.mode == "explicit":
                raise UnsupportedIQP(report.reason)
            if report.selected == "default.qubit":
                results.append(self._fallback(tape, config, report.reason))
                continue
            for estimate, limit, label in ((report.estimated_flops, self.options.hard_max_flops, "operations"),
                                            (report.estimated_memory_bytes, self.options.hard_max_memory_bytes, "tensor storage")):
                if limit is not None and estimate > limit:
                    raise MemoryError(f"IQPTNSimulator estimated {label} {estimate} exceeds explicit hard limit {limit}; no fallback or truncation performed.")
            if self.mode == "explicit" and self.options.warn_expensive and (report.estimated_flops > self.options.auto_max_flops or report.estimated_memory_bytes > self.options.auto_max_memory_bytes):
                warnings.warn(report.reason, RuntimeWarning, stacklevel=2)
            results.append(self._evaluate(tape, self._compile(tape)))
        return results[0] if single else tuple(results)
