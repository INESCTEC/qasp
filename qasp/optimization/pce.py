"""Pauli Correlation Encoding with QASP parameterized circuits."""

from __future__ import annotations

import math
from copy import deepcopy
from itertools import combinations

import numpy as np
import pennylane as qml

from qasp.circuits import AngleAnsatz, CircuitBuilder, IQPAnsatz
from qasp.devices import create_device

from .optimizers import (
    is_pennylane_optimizer,
    minimize_objective,
    minimize_pennylane,
    minimize_torch,
    normalize_backend,
)
from .problem import PauliTerm, QUBO
from .results import OptimizationResult


_ANSATZES = {
    "AngleAnsatz": AngleAnsatz,
    "angle": AngleAnsatz,
    "IQPAnsatz": IQPAnsatz,
    "iqp": IQPAnsatz,
}


def build_pauli_correlation_encoding(
    n_variables: int,
    *,
    n_qubits: int | None = None,
    correlation_order: int = 2,
    paulis: str = "XYZ",
) -> tuple[int, list[PauliTerm]]:
    """Assign each variable to an equal-Pauli correlation observable.

    The default order-two encoding provides ``3 * binomial(n_qubits, 2)``
    observables, following Sciorilli et al., Nature Communications 16, 476
    (2025), https://doi.org/10.1038/s41467-024-55346-z.
    """
    n_variables = int(n_variables)
    correlation_order = int(correlation_order)
    if n_variables < 1:
        raise ValueError("n_variables must be positive.")
    if correlation_order < 1:
        raise ValueError("correlation_order must be positive.")
    paulis = paulis.upper()
    if not paulis or any(pauli not in "XYZ" for pauli in paulis):
        raise ValueError("paulis must be a nonempty string containing X, Y, or Z.")
    if len(set(paulis)) != len(paulis):
        raise ValueError("paulis cannot contain duplicates.")
    if n_qubits is None:
        n_qubits = correlation_order
        while len(paulis) * math.comb(n_qubits, correlation_order) < n_variables:
            n_qubits += 1
    n_qubits = int(n_qubits)
    if n_qubits < correlation_order:
        raise ValueError("n_qubits must be at least correlation_order.")
    capacity = len(paulis) * math.comb(n_qubits, correlation_order)
    if capacity < n_variables:
        raise ValueError(
            f"The requested PCE has capacity {capacity}, below {n_variables} variables."
        )
    terms = []
    for pauli in paulis:
        for wires in combinations(range(n_qubits), correlation_order):
            terms.append(PauliTerm(1.0, pauli * correlation_order, wires))
            if len(terms) == n_variables:
                return n_qubits, terms
    raise RuntimeError("Failed to construct the requested PCE.")


def _observable(term: PauliTerm):
    """Build the tensor-product PennyLane Pauli observable for an encoding term."""
    operation = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}
    factors = [operation[pauli](wire) for pauli, wire in zip(term.pauli, term.wires)]
    result = factors[0]
    for factor in factors[1:]:
        result = result @ factor
    return result


def _positive_graph_scale(n_variables, couplings):
    """Paper's weighted scale, componentwise for disconnected positive graphs.

    MaxCut weights are twice the Ising couplings. Kruskal's algorithm computes
    a minimum spanning forest; isolated vertices contribute zero. Taking the
    positive part for signed graphs follows the authors' reference code, but
    the positive-weight lower-bound interpretation does not extend to them.
    """
    edges = sorted((2.0 * value, i, j) for (i, j), value in couplings.items() if value > 0)
    parent = list(range(n_variables))
    size = [1] * n_variables

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    forest_weight = 0.0
    for weight, i, j in edges:
        a, b = find(i), find(j)
        if a != b:
            if size[a] < size[b]:
                a, b = b, a
            parent[b] = a
            size[a] += size[b]
            forest_weight += weight
    return math.fsum(weight for weight, _, _ in edges) / 2.0 + forest_weight / 4.0


class PauliCorrelationEncoding:
    """Solve a QUBO using PCE and a dictionary-defined QASP circuit.

    ``circuit`` uses the same class-and-config convention as other QASP
    factories::

        {
            "ansatz": "AngleAnsatz",
            "ansatz_config": {
                "n_layers": 2,
                "rotations": ["Y 0", "Y 1", "ZZ 0 1"],
            },
            "device": "default.qubit",
        }

    A preconfigured :class:`qasp.circuits.CircuitBuilder` may instead be passed
    under the ``builder`` key.

    For ``QUBO.maxcut``, the loss follows Eqs. (2) and (5) of Sciorilli et al.,
    https://doi.org/10.1038/s41467-024-55346-z. ``solve`` includes their final
    single-bit improvement sweep; the ansatz and optimizer remain configurable.
    """

    def __init__(
        self,
        problem: QUBO | np.ndarray,
        *,
        circuit: dict | None = None,
        device: str | qml.devices.Device | None = None,
        mpi: bool | None = None,
        device_kwargs: dict | None = None,
        n_qubits: int | None = None,
        correlation_order: int = 2,
        paulis: str = "XYZ",
        alpha: float | None = None,
        regularization: float = 0.5,
        regularization_scale: float | None = None,
        shots: int | None = None,
        diff_method: str | None = "auto",
        backend: str = "numpy",
        seed: int | None = None,
        init=None,
    ):
        """Build a Pauli-correlation surrogate for a binary quadratic problem.

        Args:
            problem: QUBO instance or coefficient matrix in the QUBO convention.
            circuit: QASP ansatz dictionary or dictionary containing a builder;
                defaults to two layers of local Y/Z and neighboring ZZ rotations.
                Feature-map inputs are unsupported: one parameter tensor is required.
            device: PennyLane device name/instance, overriding the circuit setting.
            mpi: Override distributed-device selection; all ranks must cooperate.
            device_kwargs: Additional plugin options, merged with circuit options.
            n_qubits: Encoding register size, or None for the minimum sufficient size.
            correlation_order: Number of wires in each same-axis Pauli product.
            paulis: Axis families used to encode the original binary variables.
            alpha: Finite positive tanh slope; defaults to
                n_qubits ** floor(correlation_order / 2).
            regularization: Finite nonnegative beta in the paper; defaults to 0.5.
            regularization_scale: Optional finite nonnegative override of nu.
                By default, half the positive MaxCut edge weight plus one quarter
                of its minimum spanning forest weight. For general QUBOs, add
                sum(abs(h_i)) for Ising fields as a scale-homogeneous extension;
                this is not a MaxCut bound for arbitrary QUBOs or signed graphs.
            shots: Analytic execution when None; otherwise finite-shot training.
            diff_method: Auto chooses backprop for QASP Torch simulators, otherwise adjoint analytically; parameter-shift with shots.
            backend: NumPy/Autograd or Torch optimization interface.
            seed: Seed for built-in parameter initialization, not a device-shot RNG guarantee.
            init: Shared initializer, legacy mode or explicit angle array. Overrides
                circuit['init']; absent settings retain uniform [-pi, pi).

        The optimized surrogate differs from the decoded QUBO energy. Finite shots
        are incompatible with adjoint/backprop. Invalid encodings, parameter layouts
        or circuit register sizes raise ValueError/TypeError during construction."""
        self.problem = problem if isinstance(problem, QUBO) else QUBO(problem)
        self.n_qubits, self.encoding = build_pauli_correlation_encoding(
            self.problem.n_variables,
            n_qubits=n_qubits,
            correlation_order=correlation_order,
            paulis=paulis,
        )
        self.correlation_order = int(correlation_order)
        self.alpha = float(self.n_qubits ** (self.correlation_order // 2) if alpha is None else alpha)
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive.")
        self.regularization = float(regularization)
        if not np.isfinite(self.regularization) or self.regularization < 0:
            raise ValueError("regularization must be finite and nonnegative.")
        if regularization_scale is not None:
            regularization_scale = float(regularization_scale)
            if not np.isfinite(regularization_scale) or regularization_scale < 0:
                raise ValueError("regularization_scale must be finite and nonnegative.")
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.backend = normalize_backend(backend)
        self.interface = "torch" if self.backend == "torch" else "autograd"
        self.circuit_config = deepcopy(circuit or self._default_circuit())
        self.init = self.circuit_config.get("init", "random") if init is None else init
        if self.init is None:
            self.init = "random"
        self.builder, self.parameter_shape = self._build_qasp_circuit(
            self.circuit_config
        )
        self.n_parameters = int(np.prod(self.parameter_shape))
        observables = [_observable(term) for term in self.encoding]
        self.builder.set_measurement("expectation", observables=observables)
        circuit_fn = self.builder.build()
        device_spec = device if device is not None else self.circuit_config.get("device", "default.qubit")
        requested_mpi = self.circuit_config.get("mpi", False) if mpi is None else mpi
        self.mpi = bool(requested_mpi or getattr(device_spec, "_mpi", False))
        options = dict(self.circuit_config.get("device_kwargs", {}) or {})
        options.update(device_kwargs or {})
        self.device = create_device(
            device_spec, wires=self.n_qubits, shots=shots,
            mpi=requested_mpi, device_kwargs=options,
        )
        device_shots = getattr(getattr(self.device, "shots", None), "total_shots", None)
        self.training_shots = shots if shots is not None else device_shots
        self.shots = self.training_shots
        if diff_method == "auto":
            from qasp.simulators import IQPTNSimulator, PauliPropagationSimulator
            diff_method = (
                ("backprop" if isinstance(self.device, (IQPTNSimulator, PauliPropagationSimulator)) else "adjoint")
                if self.training_shots is None else "parameter-shift"
            )
        if self.training_shots is not None and diff_method in {"adjoint", "backprop"}:
            raise ValueError(
                f"diff_method={diff_method!r} requires analytic execution; use "
                "'parameter-shift' or an SPSA-style optimizer with finite shots."
            )
        self.diff_method = diff_method
        self.qnode = qml.QNode(
            circuit_fn,
            self.device,
            interface=self.interface,
            diff_method=self.diff_method,
        )
        self.ising_offset, self.ising_linear, self.ising_quadratic = (
            self.problem.to_ising()
        )
        self.regularization_scale = (
            _positive_graph_scale(self.problem.n_variables, self.ising_quadratic)
            + math.fsum(abs(value) for value in self.ising_linear.values())
            if regularization_scale is None else regularization_scale
        )
        # Construct once from the original binary cost, including small terms.
        # A sweep then visits each variable and each adjacency list only once.
        matrix = self.problem.matrix
        self._binary_linear = np.diag(matrix).copy()
        self._neighbors = [[] for _ in range(self.problem.n_variables)]
        for i, j in combinations(range(self.problem.n_variables), 2):
            coefficient = float(matrix[i, j] + matrix[j, i])
            if coefficient != 0.0:
                self._neighbors[i].append((j, coefficient))
                self._neighbors[j].append((i, coefficient))

    def _default_circuit(self):
        """Return the two-layer Hadamard-initialized Y/Z/neighbor-ZZ ansatz dictionary used when no circuit is supplied."""
        rotations = []
        for wire in range(self.n_qubits):
            rotations.extend((f"Y {wire}", f"Z {wire}"))
        for wire in range(self.n_qubits - 1):
            rotations.append(f"ZZ {wire} {wire + 1}")
        return {
            "ansatz": "AngleAnsatz",
            "ansatz_config": {
                "n_layers": 2,
                "rotations": rotations,
                "include_hadamards_initial": True,
            },
            "device": "default.qubit",
        }

    def _build_qasp_circuit(self, config):
        """Resolve the configured builder/ansatz and its single trainable tensor shape; reject feature-input circuits and register mismatches."""
        builder = config.get("builder")
        if builder is not None:
            if not isinstance(builder, CircuitBuilder):
                raise TypeError("circuit['builder'] must be a CircuitBuilder.")
            if builder.n_qubits != self.n_qubits:
                raise ValueError("The PCE register size does not match the builder.")
            ansatz = builder.ansatz
        else:
            ansatz_spec = config.get("ansatz", "AngleAnsatz")
            if isinstance(ansatz_spec, str):
                try:
                    ansatz_class = _ANSATZES[ansatz_spec]
                except KeyError as exc:
                    raise ValueError(
                        f"Unknown QASP ansatz {ansatz_spec!r}; choose from {sorted(_ANSATZES)}."
                    ) from exc
                ansatz_config = dict(config.get("ansatz_config", {}) or {})
                ansatz_config["n_qubits"] = self.n_qubits
                ansatz = ansatz_class(**ansatz_config)
            elif isinstance(ansatz_spec, type):
                ansatz_config = dict(config.get("ansatz_config", {}) or {})
                ansatz_config["n_qubits"] = self.n_qubits
                ansatz = ansatz_spec(**ansatz_config)
            else:
                ansatz = ansatz_spec
            builder = CircuitBuilder(n_qubits=self.n_qubits)
            builder.set_ansatz(ansatz)
        if ansatz is None or not hasattr(ansatz, "get_param_shapes"):
            raise TypeError("The PCE ansatz must expose get_param_shapes().")
        shapes = list(ansatz.get_param_shapes())
        if len(shapes) != 1:
            raise ValueError(
                "PCE circuits cannot require feature-map inputs; use an ansatz "
                "with one trainable parameter tensor."
            )
        return builder, tuple(shapes[0])

    @property
    def pauli_labels(self) -> list[str]:
        """Return encoding Pauli strings in original binary-variable order."""
        return [
            f"{term.pauli} {' '.join(map(str, term.wires))}" for term in self.encoding
        ]

    def initial_parameters(self, initialization=None) -> np.ndarray:
        """Sample init or an override using the instance RNG; explicit arrays win.

        The default 'random' remains uniform [-pi, pi). The constructor accepts
        init directly or in its circuit dictionary (a direct value takes priority).
        """
        from qasp.initialization import sample_parameters, Uniform

        initialization = self.init if initialization is None else initialization
        if not isinstance(initialization, str) and not callable(initialization):
            values = np.asarray(initialization, dtype=float)
            if values.size != self.n_parameters or not np.isfinite(values).all():
                raise ValueError(f"Expected {self.n_parameters} finite parameters, got {values.size}.")
            return values.reshape(-1).copy()
        if isinstance(initialization, str) and initialization.lower() == "random":
            initialization = Uniform(-np.pi, np.pi)
        return sample_parameters((self.n_parameters,), initialization, rng=self.rng)

    def expectations_tensor(self, parameters):
        """Return differentiable encoded Pauli expectations."""
        if qml.math.size(parameters) != self.n_parameters:
            raise ValueError(
                f"Expected {self.n_parameters} parameters, "
                f"got {qml.math.size(parameters)}."
            )
        values = qml.math.reshape(parameters, self.parameter_shape)
        measured = self.qnode(params=values)
        return qml.math.stack(measured) if isinstance(measured, tuple) else measured

    def _backend_parameters(self, parameters, *, trainable=False):
        """Convert to the configured numerical interface; trainable Torch requests receive a detached float64 leaf copy."""
        if self.backend == "torch":
            import torch

            values = torch.as_tensor(parameters, dtype=torch.float64)
            if trainable:
                values = values.detach().clone().requires_grad_(True)
            return values
        return qml.numpy.array(
            np.asarray(parameters, dtype=float), requires_grad=trainable
        )

    def expectations(self, parameters) -> np.ndarray:
        """Return a detached flat NumPy vector of encoding expectation values; use expectations_tensor to preserve gradients."""
        values = self._backend_parameters(parameters)
        measured = self.expectations_tensor(values)
        if self.backend == "torch":
            measured = measured.detach().cpu()
        return np.asarray(measured, dtype=float).reshape(-1)

    def soft_spins(self, parameters) -> np.ndarray:
        """Return detached surrogate spins tanh(alpha * expectation), one per original binary variable."""
        return np.tanh(self.alpha * self.expectations(parameters))

    def loss_tensor(self, parameters):
        """Return the paper-normalized differentiable PCE loss.

        With soft spins s, the data term is 2*(E_Ising(s)-offset). For MaxCut
        this is exactly sum_(i,j) w_ij*s_i*s_j, each edge counted once. Add
        beta*nu*mean(s**2)**2. Constants in the original QUBO do not enter the
        training objective; decoded result.energy retains their original scale.
        """
        spins = qml.math.tanh(self.alpha * self.expectations_tensor(parameters))
        # The zero term keeps constant problems connected to autodiff.
        value = 0.0 * qml.math.sum(spins)
        value += sum(
            2.0 * coefficient * spins[index]
            for index, coefficient in self.ising_linear.items()
        )
        value += sum(
            2.0 * coefficient * spins[i] * spins[j]
            for (i, j), coefficient in self.ising_quadratic.items()
        )
        if self.regularization:
            value += self.regularization * self.regularization_scale * qml.math.mean(spins**2) ** 2
        return value

    def loss(self, parameters) -> float:
        """Return the surrogate objective as a Python float; use loss_tensor for differentiable optimization."""
        values = self._backend_parameters(parameters)
        result = self.loss_tensor(values)
        if self.backend == "torch":
            result = result.detach().cpu()
        return float(result)

    def gradient(self, parameters) -> np.ndarray:
        """Differentiate the PCE surrogate with the configured interface."""
        if self.diff_method is None:
            raise RuntimeError("This PCE instance was configured without gradients.")
        values = self._backend_parameters(parameters, trainable=True)
        if self.backend == "torch":
            self.loss_tensor(values).backward()
            return values.grad.detach().cpu().numpy().astype(float, copy=True)
        gradient = qml.grad(self.loss_tensor)(values)
        return np.asarray(gradient, dtype=float)

    def decode(self, parameters) -> str:
        """Decode surrogate-spin signs into an original-variable bit string; nonnegative values, including zero, map to bit 0."""
        spins = self.soft_spins(parameters)
        # x=(1-Z)/2; zero expectations deterministically map to x=0.
        return "".join("0" if spin >= 0.0 else "1" for spin in spins)

    def improve_bitstring(self, bitstring: str) -> str:
        """Apply one sequential single-bit improvement sweep to the original QUBO.

        Visit variables in index order and retain only strictly improving flips.
        This is the paper's final classical step, not an iteration to a local
        optimum. It is independent of the circuit and can also be applied to a
        QAOA candidate for a comparison with identical postprocessing. Runtime
        is O(N+E) with the adjacency lists constructed during initialization.
        """
        if not isinstance(bitstring, str) or len(bitstring) != self.problem.n_variables or any(bit not in "01" for bit in bitstring):
            raise ValueError("bitstring must contain one binary digit per QUBO variable.")
        bits = np.fromiter((int(bit) for bit in bitstring), dtype=int)
        fields = self._binary_linear.copy()
        for i, neighbors in enumerate(self._neighbors):
            fields[i] += math.fsum(weight * bits[j] for j, weight in neighbors)
        for i, neighbors in enumerate(self._neighbors):
            step = 1 - 2 * bits[i]
            if step * fields[i] < 0.0:
                bits[i] += step
                for j, weight in neighbors:
                    fields[j] += weight * step
        return "".join(map(str, bits))

    def draw(self, parameters=None, **drawing_options):
        """Draw the PCE ansatz as Matplotlib, text, or Quantikz/TikZ."""
        from .drawing import draw_pce

        return draw_pce(self, parameters=parameters, **drawing_options)

    def solve(
        self,
        initial_parameters=None,
        *,
        optimizer=None,
        optimizer_options: dict | None = None,
        postprocess: bool = True,
    ) -> OptimizationResult:
        """Optimize the surrogate and return an OptimizationResult.

        initial_parameters accepts the initial_parameters method's modes/arrays.
        optimizer defaults to Adam for Torch and COBYLA otherwise; optimizer_options
        are forwarded with maxiter=100 unless overridden. The result records decoded
        QUBO energy separately from surrogate objective, flat parameters, history,
        backend/measurement metadata and the underlying optimizer result. No exact
        optimality guarantee follows from optimizer success.

        postprocess=True applies one classical single-bit improvement sweep.
        Metadata always retains raw_bitstring and raw_energy before this step.
        decode(parameters) is the raw circuit candidate, not necessarily the
        returned postprocessed bitstring. The sweep is outside autodiff."""
        if not isinstance(postprocess, (bool, np.bool_)):
            raise TypeError("postprocess must be a boolean.")
        x0 = self.initial_parameters(initial_parameters)
        optimizer = optimizer or ("Adam" if self.backend == "torch" else "COBYLA")
        options = {"maxiter": 100}
        options.update(optimizer_options or {})
        if self.backend == "torch":
            raw_result = minimize_torch(
                self.loss_tensor,
                x0,
                optimizer=optimizer,
                optimizer_options=options,
            )
            history = list(raw_result.history)
        elif is_pennylane_optimizer(optimizer):
            raw_result = minimize_pennylane(
                self.loss_tensor,
                x0,
                optimizer=optimizer,
                optimizer_options=options,
            )
            history = list(raw_result.history)
        else:
            history = []

            def objective(values):
                """Evaluate the detached surrogate and append each objective evaluation to history."""
                value = self.loss(values)
                history.append(value)
                return value

            raw_result = minimize_objective(
                objective,
                x0,
                optimizer=optimizer,
                optimizer_options=options,
            )
        parameters = np.asarray(raw_result.x, dtype=float).reshape(-1)
        expectations = self.expectations(parameters)
        raw_bitstring = "".join("0" if value >= 0.0 else "1" for value in expectations)
        bitstring = self.improve_bitstring(raw_bitstring) if postprocess else raw_bitstring
        return OptimizationResult(
            algorithm="pauli_correlation_encoding",
            parameters=parameters,
            bitstring=bitstring,
            energy=self.problem.energy(bitstring),
            objective=self.loss(parameters),
            probabilities=None,
            history=history,
            success=bool(getattr(raw_result, "success", True)),
            message=str(getattr(raw_result, "message", "")),
            metadata={
                "n_qubits": self.n_qubits,
                "alpha": self.alpha,
                "correlation_order": self.correlation_order,
                "regularization": self.regularization,
                "regularization_scale": self.regularization_scale,
                "raw_bitstring": raw_bitstring,
                "raw_energy": self.problem.energy(raw_bitstring),
                "postprocess": bool(postprocess),
                "backend": self.backend,
                "mpi": self.mpi,
                "training_shots": self.training_shots,
                "diff_method": self.diff_method,
                "optimizer": getattr(raw_result, "optimizer", None),
                "pauli_labels": self.pauli_labels,
                "expectations": expectations,
            },
            raw_result=raw_result,
        )


PCE = PauliCorrelationEncoding


__all__ = [
    "PCE",
    "PauliCorrelationEncoding",
    "build_pauli_correlation_encoding",
]
