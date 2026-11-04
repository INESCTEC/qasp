"""Bounded inference caches; no circuit parameters or autograd graphs are retained."""
from collections import OrderedDict
import warnings

import torch


class InferenceCache:
    """Cache static tensors and traced *contractions*, with every operand explicit.

    Callers supply only parameter-independent constants to ``constant``. Tracing
    sees an opt_einsum expression, never gate factories, tapes, or decoder Python.
    Exact shapes/strides and execution context are guarded instead of assuming
    that traces support polymorphic batches. Training always bypasses this class.
    """

    def __init__(self):
        self.limits = None
        self.clear()

    def clear(self):
        self.kernels = OrderedDict()
        self.constants = OrderedDict()
        self.constant_bytes = 0
        self.hits = self.traces = self.fallbacks = self.constant_hits = 0
        self.last_failure = None

    def configure(self, entries, constant_bytes):
        limits = (entries, constant_bytes)
        if limits != self.limits:
            self.clear()
            self.limits = limits

    def info(self):
        return {"kernel_entries": len(self.kernels), "constant_entries": len(self.constants),
                "constant_bytes": self.constant_bytes, "kernel_hits": self.hits,
                "traces": self.traces, "fallbacks": self.fallbacks,
                "constant_hits": self.constant_hits, "last_failure": self.last_failure}

    def constant(self, key, factory, key_bytes=0):
        if key in self.constants:
            self.constant_hits += 1
            value, _ = self.constants[key]
            self.constants.move_to_end(key)
            return value
        value = factory()
        # Tensor payload and byte-string keys count against this storage budget.
        size = value.numel() * value.element_size() + key_bytes
        if size <= self.limits[1] and self.limits[1] > 0:
            while self.constants and (self.constant_bytes + size > self.limits[1]
                                      or len(self.constants) >= 512):
                _, (_, old_size) = self.constants.popitem(last=False)
                self.constant_bytes -= old_size
            self.constants[key] = (value, size)
            self.constant_bytes += size
        return value

    def contract(self, expression, operands, structure, output, optimize):
        # This guard also protects direct/internal callers and nested tracing.
        if torch.is_grad_enabled() or torch.jit.is_tracing() or torch.jit.is_scripting():
            return expression(*operands, backend="torch").real
        if self.limits[0] == 0:
            return expression(*operands, backend="torch").real
        # Autocast is left to the eager implementation; no global Torch settings
        # are changed to make a trace acceptable.
        if any(torch.is_autocast_enabled(kind) for kind in {"cpu", operands[0].device.type}):
            return expression(*operands, backend="torch").real
        # no_grad disables reverse-mode recording, but forward-mode duals can
        # still carry tangents. Preserve the reference execution for those too.
        if any(torch.autograd.forward_ad.unpack_dual(t).tangent is not None for t in operands):
            return expression(*operands, backend="torch").real
        signatures = tuple((tuple(t.shape), tuple(t.stride()), t.dtype, t.device,
                            t.is_conj(), t.is_neg(), t.requires_grad) for t in operands)
        key = (structure, output, optimize, signatures, "no_grad", torch.is_inference_mode_enabled())
        if key in self.kernels:
            self.hits += 1
            kernel = self.kernels[key]
            self.kernels.move_to_end(key)
            if kernel is None:
                self.fallbacks += 1
                return expression(*operands, backend="torch").real
            try:
                return kernel(*operands)
            except (RuntimeError, TypeError, ValueError, NotImplementedError) as exc:
                self._failure(key, exc)
                return expression(*operands, backend="torch").real

        expected = expression(*operands, backend="torch").real
        try:
            # Closure contains only the structural opt_einsum expression.
            def contraction(*current):
                return expression(*current, backend="torch").real
            kernel = torch.jit.trace(contraction, operands, check_trace=True)
            # Check both first and warmed execution before admitting the cache.
            if not all(torch.equal(kernel(*operands), expected) for _ in range(2)):
                raise ValueError("Traced contraction differs from eager arithmetic; keeping eager execution.")
            self.traces += 1
            self._put(key, kernel)
        except (RuntimeError, TypeError, ValueError, NotImplementedError, torch.jit.TracingCheckError) as exc:
            self._failure(key, exc)
        # The first call returns the already evaluated reference result.
        return expected

    def _put(self, key, kernel):
        if key not in self.kernels and len(self.kernels) >= self.limits[0]:
            self.kernels.popitem(last=False)
        self.kernels[key] = kernel

    def _failure(self, key, exc):
        self.fallbacks += 1
        self.last_failure = f"{type(exc).__name__}: {exc}"
        self._put(key, None)
        warnings.warn(f"IQP inference kernel unavailable; using eager contraction. {self.last_failure}",
                      RuntimeWarning, stacklevel=3)
