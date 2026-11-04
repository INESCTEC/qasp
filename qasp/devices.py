"""Shared device creation for QASP's circuit and algorithm modules."""

import pennylane as qml


def resolve_mpi(mpi=None, enable_mpi=None):
    """Resolve mpi and the legacy enable_mpi alias to one boolean.

    None means unspecified. If both values are supplied their truth values
    must agree, otherwise ValueError is raised; both omitted resolves False."""
    if mpi is not None and enable_mpi is not None and bool(mpi) != bool(enable_mpi):
        raise ValueError("mpi and enable_mpi must agree when both are supplied.")
    return bool(enable_mpi if mpi is None else mpi)


def create_device(device, *, wires, shots=None, mpi=False, device_kwargs=None):
    """Create or reuse a PennyLane device without silently discarding MPI requests.

    Args:
        device: Plugin name, an existing PennyLane Device, or a QASP simulator class,
            or "auto" for circuit/readout-aware IQP tensor-network selection.
        wires, shots: Passed when creating a new plugin device. An existing
            device is reused as-is; these arguments do not reconfigure it.
        mpi: Forward mpi=True to a new plugin, or require an existing device
            to report MPI enabled. Unsupported requests propagate plugin errors.
        device_kwargs: Additional plugin options, e.g. batch_obs or c_dtype.
            wires/shots/mpi must use the explicit arguments; options cannot
            reconfigure an existing device.

    Explicit plugin and MPI requests never silently fall back. Serial "auto"
    uses IQPTNSimulator's costed selector with default.qubit fallback. QASP's
    internal lightning.pp adapter is handled by CircuitBuilder. "IQPSimulator"
    or "iqp" selects the unified device; device_kwargs["backend"] chooses
    "auto", "tn" or "van_den_nest". The legacy "auto" alias stays exact.
    "PauliPropagationSimulator" (alias "pauli.propagation") selects the shared
    analytic Torch Heisenberg device with optional explicit pruning."""
    options = dict(device_kwargs or {})
    reserved = {"wires", "shots", "mpi"}.intersection(options)
    if reserved:
        raise ValueError(
            f"Use the explicit arguments for {sorted(reserved)}, not device_kwargs."
        )
    from qasp.simulators import IQPTNSimulator, IQPSimulator, PauliPropagationSimulator
    if device is PauliPropagationSimulator or (isinstance(device, str) and device in {"PauliPropagationSimulator", "pauli.propagation"}):
        if mpi:
            raise ValueError("PauliPropagationSimulator does not support mpi=True.")
        return PauliPropagationSimulator(wires=wires, shots=shots, **options)
    if device is IQPSimulator or (isinstance(device, str) and device in {"IQPSimulator", "iqp"}):
        if mpi:
            raise ValueError("IQPSimulator is serial and does not support mpi=True.")
        return IQPSimulator(wires=wires, shots=shots, **options)
    if device is IQPTNSimulator or (isinstance(device, str) and device in {"IQPTNSimulator", "iqp.tn", "auto"}):
        if mpi:
            if device == "auto":
                return qml.device("lightning.kokkos", wires=wires, shots=shots, mpi=True, **options)
            raise ValueError("IQPTNSimulator is serial and does not support mpi=True.")
        return IQPTNSimulator(wires=wires, shots=shots,
                              mode="auto" if device == "auto" else "explicit", **options)
    if isinstance(device, qml.devices.Device):
        if options:
            raise ValueError("device_kwargs cannot configure an existing device.")
        if mpi and not getattr(device, "_mpi", False):
            raise ValueError(
                "mpi=True requires an existing device constructed with mpi=True; "
                "alternatively pass its device name to QASP."
            )
        return device
    if mpi:
        options["mpi"] = True
    return qml.device(device, wires=wires, shots=shots, **options)
