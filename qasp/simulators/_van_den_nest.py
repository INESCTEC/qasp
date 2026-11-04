"""IQP Z-correlator estimator (Van den Nest; IQPopt, arXiv:2501.04776).

This is a Torch implementation of the published estimator, not a JAX wrapper.
For physical rotation angles phi, the integrand is
cos(sum_j phi_j (-1)**(g_j.z) [g_j.a == 1 mod 2]).
No quantum-output samples or statevector are constructed.
"""
from dataclasses import dataclass
import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pennylane as qml
import torch
from torch.utils.checkpoint import checkpoint

from .iqp_tn import UnsupportedIQP, _Context, _operations, _pauli_axes
from ._iqp_observables import (compile_diagonal_observable, prepare_proposal,
                               real_coefficient_value, sample_terms)


def _parity_blocks(supports, batch_size):
    """Group equal-arity supports, bounding each gather to ~256 Ki elements.

    One exceptionally wide support may require batch_size * arity elements.
    No dense n-by-g incidence matrix or exponential lookup is constructed.
    """
    groups = {}
    for column, support in enumerate(supports):
        groups.setdefault(len(support), []).append((column, support))
    blocks = []
    for arity, group in groups.items():
        width = max(1, 262144 // max(1, batch_size * arity))
        for start in range(0, len(group), width):
            part = group[start:start+width]
            blocks.append((np.array([c for c, _ in part]),
                           np.array([w for _, w in part], dtype=np.intp)))
    return blocks


def _parities(bits, blocks, gates, dtype):
    out = np.empty((len(bits), gates), dtype=dtype)
    for columns, indices in blocks:
        # Arrange gather as (support, arity, sample): reduce contiguous samples
        # across arity without one Python call per gate.
        gathered = bits.T[indices]
        out[:, columns] = np.bitwise_xor.reduce(gathered, axis=1).T
    return out


def _chunk_results(run, samples, options, seed):
    """Bounded, ordered CPU work; thread count never changes the random bank.

    Torch grad/inference modes are thread-local. Preserve both from the caller;
    checkpoint closures retain their own seed for subsequent backward calls.
    The executor belongs to this execution, never to a global device pool.
    """
    rng = np.random.default_rng(seed)
    grad_enabled, inference = torch.is_grad_enabled(), torch.is_inference_mode_enabled()

    def tasks():
        for start in range(0, samples, options.vn_batch_size):
            yield start, min(options.vn_batch_size, samples-start), int(rng.integers(0, 2**63-1))

    def invoke(task):
        with torch.inference_mode(inference), torch.set_grad_enabled(grad_enabled):
            return run(*task)

    workers = min(options.vn_workers, math.ceil(samples/options.vn_batch_size))
    if workers == 1:
        for task in tasks():
            yield run(*task)
        return
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="qasp-vn") as pool:
        iterator = iter(tasks())
        pending = deque(pool.submit(invoke, task) for _, task in zip(range(workers), iterator))
        while pending:
            yield pending.popleft().result()
            task = next(iterator, None)
            if task is not None:
                pending.append(pool.submit(invoke, task))


class _VNChunkFunction(torch.autograd.Function):
    """A deterministic sum with bounded, explicitly parallel chunk VJPs.

    No worker writes .grad or traverses the encoder graph. Workers return local
    angle/coefficient sensitivities, which the caller reduces before Torch
    continues through the original inputs. Higher-order calls replay with the
    original inputs so their derivative graphs remain connected.
    """

    @staticmethod
    def forward(ctx, angles, coefficients, kernel, options, seed, samples, outputs):
        ctx.save_for_backward(angles, coefficients)
        ctx.kernel, ctx.options, ctx.seed, ctx.samples = kernel, options, seed, samples
        ctx.cached = [] if not options.vn_checkpoint else None
        needs = (angles.requires_grad, coefficients.requires_grad)

        def run(start, count, chunk_seed):
            if options.vn_checkpoint:
                value, squares = kernel(angles, coefficients, start, count, chunk_seed)
                return value, squares, None
            # Retain independent local graphs when checkpointing is disabled.
            # Detaching the boundary prevents concurrent traversal of encoders.
            with torch.enable_grad():
                local = tuple(x.detach().requires_grad_(need)
                              for x, need in zip((angles, coefficients), needs))
                value, squares = kernel(*local, start, count, chunk_seed)
            return value.detach(), squares, (local, value)

        total = angles.new_zeros(outputs)
        square_total = torch.zeros_like(total)
        for value, squares, cached in _chunk_results(run, samples, options, seed):
            total = total + value
            square_total = square_total + squares
            if ctx.cached is not None:
                ctx.cached.append(cached)
        ctx.mark_non_differentiable(square_total)
        ctx.set_materialize_grads(False)
        return total, square_total

    @staticmethod
    def backward(ctx, upstream, ignored_squares):
        if upstream is None:
            return (None,) * 7
        inputs = ctx.saved_tensors
        needs = ctx.needs_input_grad[:2]
        higher_order = torch.is_grad_enabled()

        def run(start, count, chunk_seed):
            with torch.enable_grad():
                cached = ctx.cached is not None and not higher_order
                if cached:
                    local, value = ctx.cached[start // ctx.options.vn_batch_size]
                else:
                    # Preserve dependence on both original inputs and upstream
                    # for mixed derivatives and nonlinear outer losses.
                    local = inputs if higher_order else tuple(
                        x.detach().requires_grad_(need) for x, need in zip(inputs, needs))
                    value, _ = ctx.kernel(*local, start, count, chunk_seed)
                targets = tuple(x for x, need in zip(local, needs) if need)
                if value.requires_grad:
                    gradients = iter(torch.autograd.grad(
                        value, targets, grad_outputs=upstream, create_graph=higher_order,
                        retain_graph=cached or higher_order, allow_unused=True))
                else:
                    gradients = iter(None for _ in targets)
                result = []
                for x, need in zip(local, needs):
                    gradient = next(gradients) if need else None
                    if need:
                        if gradient is None:
                            gradient = x * 0 if higher_order else torch.zeros_like(x)
                        elif higher_order:
                            gradient = gradient + x * 0
                    result.append(gradient)
                return tuple(result)

        totals = [torch.zeros_like(x) if need else None for x, need in zip(inputs, needs)]
        for partial in _chunk_results(run, ctx.samples, ctx.options, ctx.seed):
            for i, need in enumerate(needs):
                if need:
                    totals[i] = totals[i] + partial[i]
        return *totals, None, None, None, None, None


def _reduce_chunks(angles, coefficients, kernel, options, seed, samples, outputs):
    """Keep eager autograd as an explicit comparison and accelerator fallback."""
    trainable = torch.is_grad_enabled() and (angles.requires_grad or coefficients.requires_grad)
    if options.vn_parallel_backward and angles.device.type == "cpu" and trainable:
        return _VNChunkFunction.apply(angles, coefficients, kernel, options, seed, samples, outputs)

    def run(start, count, chunk_seed):
        def chunk(a, c):
            return kernel(a, c, start, count, chunk_seed)
        if options.vn_checkpoint and trainable:
            return checkpoint(chunk, angles, coefficients, use_reentrant=False, preserve_rng_state=False)
        return chunk(angles, coefficients)

    total = angles.new_zeros(outputs)
    squares = torch.zeros_like(total)
    for value, square in _chunk_results(run, samples, options, seed):
        total = total + value
        squares = squares + square
    return total, squares


@dataclass
class VNProgram:
    wires: tuple
    supports: tuple
    angles: tuple
    masks: tuple
    groups: tuple
    sampled_observables: tuple = ()


def compile_vn(tape, wires, options=None):
    """Recognize X rotations, or their local-H-frame Z representation.

    Checks depend on gate structure, never on a trainable angle being zero.
    Hadamards are tracked symbolically; every nontrivial rotation must become
    X-only and the final frame must be identity. No arbitrary local dressing.
    """
    if tape.shots:
        raise UnsupportedIQP("van_den_nest uses classical estimator samples with shots=None; it does not sample quantum outputs.")
    index = {w: i for i, w in enumerate(wires)}
    frame = {w: False for w in wires}
    supports, angles = [], []

    def add(generator, angle):
        support = []
        for w, axis in generator:
            if axis == "I":
                continue
            mapped = {"X": "Z", "Z": "X", "Y": "Y"}[axis] if frame[w] else axis
            if mapped != "X":
                raise UnsupportedIQP("van_den_nest requires X-only rotations on |0>, or an equivalent H-D(Z)-H circuit; use tn for local dressing.")
            support.append(index[w])
        if support:
            supports.append(tuple(support))
            angles.append(angle)

    for op in _operations(tape):
        if not set(op.wires) <= set(wires):
            raise UnsupportedIQP("Operation addresses wires outside the device.")
        sign = 1
        while isinstance(op, qml.ops.op_math.Adjoint):
            sign = -sign
            op = op.base
        if op.name == "Hadamard":
            frame[op.wires[0]] = not frame[op.wires[0]]
        elif op.name in {"Identity", "GlobalPhase"}:
            continue
        elif op.name in {"CZ", "ControlledPhaseShift"}:
            a, b = op.wires
            angle = sign * (math.pi if op.name == "CZ" else op.data[0])
            add(((a, "Z"),), angle / 2)
            add(((b, "Z"),), angle / 2)
            add(((a, "Z"), (b, "Z")), -angle / 2)
        else:
            generator = _pauli_axes(op)
            if generator is not None:
                add(generator, sign * op.data[0])
            elif op.name in {"RX", "RY", "RZ", "PhaseShift", "PauliX", "PauliY", "PauliZ", "S", "T", "SX"}:
                axis = {"RX": "X", "RY": "Y", "RZ": "Z", "PhaseShift": "Z",
                        "PauliX": "X", "PauliY": "Y", "PauliZ": "Z",
                        "S": "Z", "T": "Z", "SX": "X"}[op.name]
                fixed = {"PauliX": math.pi, "PauliY": math.pi, "PauliZ": math.pi,
                         "S": math.pi/2, "T": math.pi/4, "SX": math.pi/2}
                add(((op.wires[0], axis),), sign * (fixed[op.name] if op.name in fixed else op.data[0]))
            else:
                raise UnsupportedIQP(f"van_den_nest does not recognize {op.name}; use tn or rewrite as commuting Pauli rotations.")
    if any(frame.values()):
        raise UnsupportedIQP("van_den_nest requires a closed Hadamard frame (H-D-H), with Z-string readout.")

    if options is not None and options.vn_observable_sampling == "importance":
        if options.vn_sampling == "enumerate":
            raise UnsupportedIQP("vn_sampling='enumerate' does not enumerate sampled observable terms; use vn_observable_sampling='all_terms' for exact validation.")
        observables = []
        for measurement in tape.measurements:
            if not isinstance(measurement, qml.measurements.ExpectationMP):
                raise UnsupportedIQP("Importance sampling requires expval observables; action probabilities, variances and quantum samples use tn.")
            observables.append(compile_diagonal_observable(
                measurement.obs, wires, options.vn_max_observable_components))
        if not observables:
            raise UnsupportedIQP("At least one expectation value is required.")
        return VNProgram(tuple(wires), tuple(supports), tuple(angles), (), (), tuple(observables))

    masks, groups = [], []
    for measurement in tape.measurements:
        if not isinstance(measurement, qml.measurements.ExpectationMP):
            raise UnsupportedIQP("van_den_nest supports expectations of Z strings and their real weighted sums; probabilities, variances and samples use tn.")
        sentence = measurement.obs.pauli_rep
        if sentence is None:
            raise UnsupportedIQP("van_den_nest requires a compact Pauli representation.")
        group = []
        for word, coefficient in sentence.items():
            if any(axis != "Z" for axis in word.values()) or not set(word) <= set(wires):
                raise UnsupportedIQP("van_den_nest observables must be real sums of I/Z strings on device wires.")
            mask = sum(1 << index[w] for w in word)
            if mask not in masks:
                masks.append(mask)
            group.append((masks.index(mask), coefficient))
        groups.append(tuple(group))
    if not groups:
        raise UnsupportedIQP("At least one expectation value is required.")
    return VNProgram(tuple(wires), tuple(supports), tuple(angles), tuple(masks), tuple(groups))


_coefficient_value = real_coefficient_value


def _sample_budget(bounds, options, n):
    """Bound EACH returned mean, simultaneously at the requested confidence."""
    log_delta = math.log(2 * max(1, len(bounds)) / (1-options.vn_confidence))
    if options.vn_sampling == "enumerate":
        if n > options.vn_max_exact_wires:
            raise UnsupportedIQP(f"Exact estimator enumeration is limited to {options.vn_max_exact_wires} wires.")
        return 1 << n, log_delta
    samples = options.vn_samples
    if options.vn_error is not None:
        largest = max(bounds, default=0.)
        if largest and (math.log(2*log_delta) + 2*math.log(largest)
                        - 2*math.log(options.vn_error) > math.log(options.vn_max_samples)):
            raise UnsupportedIQP(f"Requested additive error needs more than vn_max_samples={options.vn_max_samples}.")
        required = math.ceil(2 * (largest/options.vn_error)**2 * log_delta) if largest else 0
        samples = max(samples, required)
    if samples > options.vn_max_samples:
        raise UnsupportedIQP(f"Requested additive error needs {samples} samples, exceeding vn_max_samples={options.vn_max_samples}.")
    return samples, log_delta


def estimate_vn(program, options, batch_size=1, requires_grad=False):
    """Conservative arithmetic/storage proxies, not wall-time or OS limits."""
    if program.sampled_observables:
        return _estimate_importance(program, options, batch_size, requires_grad)
    n, g, k, m = len(program.wires), len(program.angles), len(program.masks), len(program.groups)
    l1 = tuple(sum(abs(_coefficient_value(c)) for _, c in group) for group in program.groups)
    samples, log_delta = _sample_budget(l1, options, n)
    b = min(samples, options.vn_batch_size)
    o = min(k, options.vn_observable_batch_size)
    sparse_size = sum(map(len, program.supports))
    flops = samples * (n + sparse_size + 2*g*k + 20*k + 2*k*m) + k*g*max(1, (n+63)//64)
    # Include chunk temporaries, coefficient map, parity mask and RNG bitstrings.
    real_bytes = 8 if options.precision == "complex128" else 4
    storage = real_bytes * (4*b*g + 4*b*o + 4*b*m + k*g + k*m + g + m) + b*n + 8*sparse_size
    workers = min(options.vn_workers, math.ceil(samples/options.vn_batch_size))
    storage = (storage + max(262144, b*max(map(len, program.supports), default=0))) * workers
    if requires_grad:
        if options.vn_checkpoint:
            storage *= 3
        else:
            storage += real_bytes * samples * (2*g + 2*k + m)
    bounds = tuple(0. if options.vn_sampling == "enumerate" else v * math.sqrt(2*log_delta/samples) for v in l1)
    return dict(samples=samples, cpu_workers=workers, estimated_flops=flops*batch_size*(3 if requires_grad else 1),
                estimated_memory_bytes=storage*batch_size, hoeffding_half_widths=bounds,
                observable_sampling="all_terms", coefficient_l1_bounds=l1)


def evaluate_vn(tape, program, options, seed, samples):
    """Differentiable empirical mean; checkpointing regenerates fixed chunks.

    Statistics are for each returned weighted sum, so shared-sample covariance
    is included. Resampling occurs outside this function, never in backward.
    """
    if program.sampled_observables:
        return _evaluate_importance(tape, program, options, seed, samples)
    ctx = _Context(tape, options)
    dtype = torch.float64 if options.precision == "complex128" else torch.float32
    device = ctx.device
    g, k, m, n = len(program.angles), len(program.masks), len(program.groups), len(program.wires)

    def real(value):
        if isinstance(value, torch.Tensor):
            return value.real.to(device=device, dtype=dtype)
        return torch.as_tensor(np.real(value), device=device, dtype=dtype)

    phi = torch.stack([real(p) for p in program.angles]) if g else torch.empty(0, dtype=dtype, device=device)
    rows = []
    for group in program.groups:
        row = [real(0.) for _ in range(k)]
        for j, coefficient in group:
            row[j] = row[j] + real(coefficient)
        rows.append(torch.stack(row) if k else torch.empty(0, dtype=dtype, device=device))
    weights = torch.stack(rows).T
    generator_masks = [sum(1 << w for w in support) for support in program.supports]
    anticommutes = torch.tensor([[(mask & gm).bit_count() % 2 for gm in generator_masks]
                                 for mask in program.masks], dtype=dtype, device=device).reshape(k, g)
    if options.vn_workers > 1 and device.type != "cpu":
        raise UnsupportedIQP("vn_workers > 1 requires CPU tensors; use one worker for accelerators.")
    blocks = _parity_blocks(program.supports, min(samples, options.vn_batch_size))
    numpy_dtype = np.float64 if dtype == torch.float64 else np.float32

    def kernel(angles, coefficients, start, count, chunk_seed):
        if options.vn_sampling == "enumerate":
            bits = ((np.arange(start, start+count, dtype=np.uint64)[:, None]
                     >> np.arange(n, dtype=np.uint64)) & 1).astype(np.uint8)
        else:
            bits = np.random.default_rng(chunk_seed).integers(0, 2, size=(count, n), dtype=np.uint8)
        signs = _parities(bits, blocks, g, numpy_dtype)
        signs *= -2
        signs += 1
        signed_angles = torch.as_tensor(signs, device=device) * angles
        values = torch.zeros((count, m), dtype=dtype, device=device)
        for lo in range(0, k, options.vn_observable_batch_size):
            hi = min(k, lo+options.vn_observable_batch_size)
            terms = torch.cos(signed_angles @ anticommutes[lo:hi].T)
            values = values + terms @ coefficients[lo:hi]
        return values.sum(0), values.detach().square().sum(0)

    total, square_total = _reduce_chunks(phi, weights, kernel, options, seed, samples, m)
    return _finish_vn(tape, options, ctx, total, square_total, samples)


def _finish_vn(tape, options, ctx, total, square_total, samples):
    value = total / samples
    # Retain connected zero derivatives, including identity/global-phase gates.
    for parameter in tape.get_parameters(trainable_only=False):
        if isinstance(parameter, torch.Tensor) and parameter.requires_grad:
            value = value + 0 * parameter.real.sum()
    if options.vn_sampling == "enumerate":
        standard_errors = tuple(0. for _ in range(len(total)))
    else:
        variance = ((square_total - total.detach().square()/samples) / (samples-1)).clamp_min(0)
        standard_errors = tuple(torch.sqrt(variance/samples).cpu().tolist())
    results = tuple(v if ctx.torch_result else v.detach().cpu().numpy() for v in value)
    return (results[0] if len(results) == 1 else results), standard_errors


def _estimate_importance(program, options, batch_size, requires_grad):
    proposals = [prepare_proposal(o, options.vn_term_uniform_mix) for o in program.sampled_observables]
    bounds = tuple(p.weight_bound for p in proposals)
    samples, log_delta = _sample_budget(bounds, options, len(program.wires))
    n, g, m = len(program.wires), len(program.angles), len(proposals)
    b = min(samples, options.vn_batch_size)
    description = sum(p.observable.description_size for p in proposals)
    components = tuple(len(p.observable.components) for p in proposals)
    total_components = sum(components)
    support_size = sum(map(len, program.supports))
    # Worst component size per output; no factor equal to expanded term count.
    draw_size = sum(max(len(c.z_indices) + c.expansion_bits for c in p.observable.components)
                    for p in proposals)
    flops = description + samples * (n + support_size + draw_size + m*(n+support_size+4*g+24)
                                    + sum(max(1, math.ceil(math.log2(c))) for c in components))
    real_bytes = 8 if options.precision == "complex128" else 4
    storage = (real_bytes*(8*b*g + 4*b*m + 4*total_components + g + m)
               + 4*b*n + 16*description + 16*support_size)
    workers = min(options.vn_workers, math.ceil(samples/options.vn_batch_size))
    storage = (storage + max(262144, b*max(map(len, program.supports), default=0))) * workers
    if requires_grad:
        if options.vn_checkpoint:
            storage *= 3
        else:
            storage += real_bytes*samples*(m*(3*g+4)+g)
    return dict(samples=samples, cpu_workers=workers, estimated_flops=flops*batch_size*(3 if requires_grad else 1),
                estimated_memory_bytes=storage*batch_size,
                hoeffding_half_widths=tuple(v*math.sqrt(2*log_delta/samples) for v in bounds),
                observable_sampling="importance", observable_components=components,
                observable_log2_term_counts=tuple(p.observable.log2_term_count for p in proposals),
                coefficient_l1_bounds=tuple(p.l1_bound for p in proposals),
                importance_weight_bounds=bounds)


def _evaluate_importance(tape, program, options, seed, samples):
    ctx = _Context(tape, options)
    dtype = torch.float64 if options.precision == "complex128" else torch.float32
    device = ctx.device
    n, g, m = len(program.wires), len(program.angles), len(program.sampled_observables)
    proposals = tuple(prepare_proposal(o, options.vn_term_uniform_mix) for o in program.sampled_observables)

    def real(value):
        if isinstance(value, torch.Tensor):
            return value.real.to(device=device, dtype=dtype)
        return torch.as_tensor(np.real(value), device=device, dtype=dtype)

    phi = torch.stack([real(p) for p in program.angles]) if g else torch.empty(0, dtype=dtype, device=device)
    coefficients = torch.stack([real(c.coefficient) for p in proposals for c in p.observable.components])
    if options.vn_workers > 1 and device.type != "cpu":
        raise UnsupportedIQP("vn_workers > 1 requires CPU tensors; use one worker for accelerators.")
    blocks = _parity_blocks(program.supports, min(samples, options.vn_batch_size))
    numpy_dtype = np.float64 if dtype == torch.float64 else np.float32

    def kernel(angles, live_coefficients, start, count, chunk_seed):
        rng = np.random.default_rng(chunk_seed)
        bits = rng.integers(0, 2, size=(count, n), dtype=np.uint8)
        signs = _parities(bits, blocks, g, numpy_dtype)
        signs *= -2
        signs += 1
        signed_angles = torch.as_tensor(signs, device=device)*angles
        values, offset = [], 0
        for proposal in proposals:
            selected, masks, projector_signs = sample_terms(proposal, rng, count, n)
            anti = _parities(masks, blocks, g, numpy_dtype)
            phase = (signed_angles*torch.as_tensor(anti, device=device)).sum(1)
            indices = torch.as_tensor(selected+offset, dtype=torch.long, device=device)
            # q is detached and strictly positive, including when c is zero.
            inverse_q = torch.as_tensor(projector_signs/proposal.probabilities[selected],
                                        dtype=dtype, device=device)
            values.append(torch.cos(phase)*live_coefficients[indices]*inverse_q)
            offset += len(proposal.observable.components)
        values = torch.stack(values, dim=1)
        return values.sum(0), values.detach().square().sum(0)

    total, square_total = _reduce_chunks(phi, coefficients, kernel, options, seed, samples, m)
    return _finish_vn(tape, options, ctx, total, square_total, samples)
