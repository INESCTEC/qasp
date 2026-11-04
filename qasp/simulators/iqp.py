"""Unified IQP device: exact tensor contraction or sampled Z correlators."""
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
import math

import numpy as np
import pennylane as qml
import torch

from .iqp_tn import IQPTNOptions, IQPTNReport, IQPTNSimulator, UnsupportedIQP
from ._van_den_nest import compile_vn, estimate_vn, evaluate_vn


@dataclass(frozen=True)
class IQPOptions(IQPTNOptions):
    """TN options plus explicit accuracy, sampling and planning controls.

    vn_error is an absolute error target for EACH returned weighted expectation,
    using a simultaneous Hoeffding bound at vn_confidence. It controls values,
    not nonlinear losses, relative error, or gradient error.
    """
    vn_samples: int = 4096
    vn_batch_size: int = 1024
    vn_observable_batch_size: int = 64
    vn_sampling: str = "resample"
    vn_checkpoint: bool = True
    vn_workers: int = 1
    vn_parallel_backward: bool = True
    vn_observable_sampling: str = "all_terms"
    vn_term_uniform_mix: float = 0.05
    vn_max_observable_components: int = 100_000
    vn_error: float | None = None
    vn_confidence: float = 0.99
    vn_max_samples: int = 1_000_000
    vn_max_exact_wires: int = 16
    allow_approximate: bool = True
    auto_tn_max_wires: int = 64
    auto_tn_max_gates: int = 512
    auto_tn_operation_overhead: int = 20_000

    def __post_init__(self):
        super().__post_init__()
        for key in ("vn_samples", "vn_batch_size", "vn_observable_batch_size",
                    "vn_max_samples", "vn_max_exact_wires", "auto_tn_max_wires", "auto_tn_max_gates",
                    "vn_max_observable_components", "vn_workers"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{key} must be a positive integer.")
        if (not isinstance(self.auto_tn_operation_overhead, int)
                or isinstance(self.auto_tn_operation_overhead, bool) or self.auto_tn_operation_overhead < 0):
            raise ValueError("auto_tn_operation_overhead must be a nonnegative integer.")
        if self.vn_samples < 2 or self.vn_samples > self.vn_max_samples:
            raise ValueError("vn_samples must be at least 2 and at most vn_max_samples.")
        if self.vn_max_exact_wires > 24:
            raise ValueError("vn_max_exact_wires must not exceed 24; enumeration is for small validation circuits.")
        if self.vn_observable_sampling not in {"all_terms", "importance"}:
            raise ValueError("vn_observable_sampling must be all_terms or importance.")
        if (isinstance(self.vn_term_uniform_mix, bool) or not isinstance(self.vn_term_uniform_mix, (int, float))
                or not math.isfinite(self.vn_term_uniform_mix) or not 1e-6 <= self.vn_term_uniform_mix <= 1):
            raise ValueError("vn_term_uniform_mix must be finite and between 1e-6 and 1.")
        if self.vn_observable_sampling == "importance" and self.vn_sampling == "enumerate":
            raise ValueError("enumerate requires vn_observable_sampling='all_terms'; observable terms would otherwise remain sampled.")
        if self.vn_sampling not in {"resample", "fixed", "enumerate"}:
            raise ValueError("vn_sampling must be resample, fixed or enumerate.")
        if not math.isfinite(self.vn_confidence) or not 0 < self.vn_confidence < 1:
            raise ValueError("vn_confidence must be strictly between zero and one.")
        if self.vn_error is not None and (not math.isfinite(self.vn_error) or self.vn_error <= 0):
            raise ValueError("vn_error must be a positive finite absolute error.")
        if not isinstance(self.vn_parallel_backward, bool):
            raise ValueError("vn_parallel_backward must be a boolean.")
        if not isinstance(self.vn_checkpoint, bool) or not isinstance(self.allow_approximate, bool):
            raise ValueError("vn_checkpoint and allow_approximate must be booleans.")


@dataclass(frozen=True)
class IQPReport(IQPTNReport):
    backend: str = "auto"
    approximate: bool = False
    samples: int = 0
    sampling: str = ""
    standard_errors: tuple = ()
    hoeffding_half_widths: tuple = ()
    confidence: float = 0.
    cpu_workers: int = 1
    observable_sampling: str = ""
    observable_components: tuple = ()
    observable_log2_term_counts: tuple = ()
    coefficient_l1_bounds: tuple = ()
    importance_weight_bounds: tuple = ()
    tn_estimated_flops: int | None = None
    vn_estimated_flops: int | None = None
    tn_planning_skipped: bool = False
    tn_selection_score: int | None = None
    vn_selection_score: int | None = None


class IQPSimulator(IQPTNSimulator):
    """IQP simulator with backend='auto', 'tn' or 'van_den_nest'.

    auto compares bounded heuristic cost estimates, not measured optimal runtime.
    vn_observable_sampling="importance" samples observable terms as well as bits.
    It may select a stochastic approximation; set allow_approximate=False for
    exact execution. Probability/sampling/variance readouts remain on TN.
    mode='auto' retains the legacy size-limited default.qubit fallback for
    unsupported programs. mode='explicit' disables that fallback. Forcing either
    backend always disables fallback. IQPTNSimulator remains unchanged.
    """
    name = "IQPSimulator"

    def __init__(self, wires=None, shots=None, *, backend="auto", mode="auto", options=None, seed=None, **kwargs):
        if mode not in {"auto", "explicit"}:
            raise ValueError("mode must be auto or explicit.")
        if backend not in {"auto", "tn", "van_den_nest"}:
            raise ValueError("backend must be auto, tn or van_den_nest.")
        if options is not None and kwargs:
            raise ValueError("Supply either options=IQPOptions(...) or option keywords.")
        if isinstance(options, IQPOptions):
            configured = options
        elif isinstance(options, IQPTNOptions):
            configured = IQPOptions(**asdict(options))
        else:
            configured = IQPOptions(**(options or kwargs))
        super().__init__(wires, shots, mode=mode if backend == "auto" else "explicit",
                         options=configured, seed=seed)
        self.backend = backend
        self._independent_scopes = []
        self._tn_plan_cache = {}
        self._vn_rng = np.random.default_rng(seed)
        self._vn_fixed_seed = int(self._vn_rng.integers(0, 2**63-1))

    def resample(self, seed=None):
        """Replace the fixed sample bank, or reset the resampling stream.

        Never call during a backward pass. Existing graphs retain their own
        chunk seeds, so subsequent executions do not change their gradients.
        """
        if seed is not None:
            self._vn_rng = np.random.default_rng(seed)
        self._vn_fixed_seed = int(self._vn_rng.integers(0, 2**63-1))

    @contextmanager
    def independent_estimates(self):
        """Fresh samples per execution, including when vn_sampling='fixed'.

        The yielded mapping records whether any sampled estimate was used.
        QCBM uses this to form an unbiased two-replica model-square in MMD.
        """
        scope = {"approximate": False}
        self._independent_scopes.append(scope)
        try:
            yield scope
        finally:
            self._independent_scopes.pop()

    def clear_plan_cache(self):
        """Drop the bounded cache of parameter-independent TN planning reports."""
        self._tn_plan_cache.clear()

    def _tn_plan(self, tape, program, batch_size, requires_grad, interface):
        # Cache only recognized estimator circuits/readouts: all topology checks
        # are structural, including at zero angles; coefficients affect estimator
        # accuracy separately. Cache no numerical tensors or gate parameters.
        key = None
        if program is not None and not program.sampled_observables:
            key = (tuple((op.name, tuple(op.wires), repr(op.hyperparameters),
                          tuple(qml.math.shape(p) for p in op.data)) for op in tape.operations),
                   program.wires, program.masks,
                   tuple(tuple(j for j, _ in group) for group in program.groups),
                   self.options, batch_size, requires_grad, interface)
            if key in self._tn_plan_cache:
                return self._tn_plan_cache[key]
        report = super().plan(tape, batch_size=batch_size, requires_grad=requires_grad, interface=interface)
        if key is not None:
            if len(self._tn_plan_cache) >= 64:
                self._tn_plan_cache.pop(next(iter(self._tn_plan_cache)))
            self._tn_plan_cache[key] = report
        return report

    def _within_hard_limits(self, report):
        return all(limit is None or value <= limit for value, limit in (
            (report.estimated_flops, self.options.hard_max_flops),
            (report.estimated_memory_bytes, self.options.hard_max_memory_bytes)))

    def plan(self, tape, *, batch_size=1, requires_grad=False, interface="torch"):
        """Inspect both methods without drawing samples or evaluating a circuit."""
        if tape.batch_size is not None:
            tapes, _ = qml.transforms.broadcast_expand(tape)
            tape, batch_size = tapes[0], batch_size*len(tapes)
        common = dict(mode=self.mode, backend=self.backend, batch_size=batch_size,
                      requires_grad=requires_grad, precision=self.options.precision)
        vn, program = None, None
        vn_reason = ""
        if self.backend != "tn":
            try:
                if interface not in {"torch", "numpy", None, "auto"}:
                    raise UnsupportedIQP("van_den_nest differentiation supports Torch.")
                if not self.options.allow_approximate and self.options.vn_sampling != "enumerate":
                    raise UnsupportedIQP("Sampled estimation disabled by allow_approximate=False.")
                if self.options.vn_workers > 1 and any(
                    isinstance(p, torch.Tensor) and p.device.type != "cpu"
                    for p in tape.get_parameters(trainable_only=False)
                ):
                    raise UnsupportedIQP("vn_workers > 1 requires CPU tensors; use one worker for accelerators.")
                wires = tuple(self.wires) if self.wires is not None else tuple(tape.wires)
                program = compile_vn(tape, wires, self.options)
                estimates = estimate_vn(program, self.options, batch_size, requires_grad)
                vn = IQPReport(**common, selected="van_den_nest", eligible=True,
                    reason=("Joint observable-term and uniform-bitstring estimation."
                            if program.sampled_observables else "Uniform-bitstring Z-correlator estimation."),
                    approximate=self.options.vn_sampling != "enumerate",
                    sampling=self.options.vn_sampling, confidence=self.options.vn_confidence,
                    output_elements=len(tape.measurements), **estimates)
            except UnsupportedIQP as exc:
                vn_reason = str(exc)
        if self.backend == "van_den_nest":
            return vn if vn is not None else IQPReport(**common, selected="error", eligible=False, reason=vn_reason)

        usable_vn = (vn is not None and self._within_hard_limits(vn)
                     and vn.estimated_memory_bytes <= self.options.auto_max_memory_bytes)
        skip_tn = self.backend == "auto" and usable_vn and (
            len(program.wires) > self.options.auto_tn_max_wires
            or len(tape.operations) > self.options.auto_tn_max_gates)
        tn = None
        if not skip_tn:
            try:
                legacy = self._tn_plan(tape, program, batch_size, requires_grad, interface)
                tn = IQPReport(**asdict(legacy), backend=self.backend)
                if tn.selected == self.name:
                    tn = replace(tn, selected="tn")
            except MemoryError as exc:
                tn = IQPReport(**common, selected="error", eligible=False, reason=str(exc))
        if self.backend == "tn":
            return tn

        usable_tn = tn is not None and tn.eligible and tn.selected == "tn" and self._within_hard_limits(tn)
        # TN builds/conjugates many small operands per observable in Python.
        # A configurable equivalent-work penalty accounts for that overhead;
        # it is a heuristic, not a calibrated hardware timing guarantee.
        tn_score = (tn.estimated_flops + self.options.auto_tn_operation_overhead
                    * (len(tape.operations)+len(tuple(self.wires or tape.wires)))
                    * tn.networks * batch_size * (3 if requires_grad else 1)) if tn else None
        cost_fields = dict(tn_selection_score=tn_score,
                           vn_selection_score=vn.estimated_flops if vn else None,
tn_estimated_flops=tn.estimated_flops if tn and tn.eligible else None,
                           vn_estimated_flops=vn.estimated_flops if vn else None, tn_planning_skipped=skip_tn)
        if usable_vn and (not usable_tn or vn.estimated_flops < tn_score):
            reason = ("TN planning skipped at the configured structural limit; polynomial estimator selected."
                      if skip_tn else "Estimator selected by resource budgets and heuristic arithmetic and small-operand overhead; runtime optimality is not guaranteed.")
            return replace(vn, reason=reason, **cost_fields)
        if usable_tn:
            reason = ("Exact TN selected by heuristic cost; estimator cost is higher."
                      if usable_vn else f"Exact TN selected. Estimator unavailable: {vn_reason or 'resource budget exceeded'}.")
            return replace(tn, reason=reason, **cost_fields)
        reason = f"No IQP backend fits the supported structure and budgets. Estimator: {vn_reason or 'resource budget exceeded'}. TN: {tn.reason if tn else 'planning skipped'}"
        return IQPReport(**common, selected="default.qubit" if self.mode == "auto" else "error",
                         eligible=False, reason=reason, **cost_fields)

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
            grad = torch.is_grad_enabled() and any(isinstance(p, torch.Tensor) and p.requires_grad
                for p in tape.get_parameters(trainable_only=False))
            report = self.plan(tape, batch_size=len(tapes), requires_grad=grad, interface=interface)
            self.last_report = report
            if report.selected == "error":
                raise UnsupportedIQP(report.reason)
            if report.selected == "default.qubit":
                value = self._fallback(tape, config, report.reason)
            else:
                if not self._within_hard_limits(report):
                    raise MemoryError("IQPSimulator explicit hard limit exceeded; no truncation or fallback performed.")
                if report.selected == "tn":
                    value = self._evaluate(tape, self._compile(tape))
                else:
                    wires = tuple(self.wires) if self.wires is not None else tuple(tape.wires)
                    program = compile_vn(tape, wires, self.options)
                    seed = self._vn_fixed_seed if self.options.vn_sampling == "fixed" and not self._independent_scopes else int(self._vn_rng.integers(0, 2**63-1))
                    value, errors = evaluate_vn(tape, program, self.options, seed, report.samples)
                    report = replace(report, standard_errors=errors)
                    if report.approximate:
                        for scope in self._independent_scopes:
                            scope["approximate"] = True
                        if self._independent_scopes:
                            report = replace(report, sampling="resample")
            results.append(value)
            reports.append(report)
            self.last_report, self.last_reports = report, tuple(reports)
        return results[0] if single else tuple(results)
