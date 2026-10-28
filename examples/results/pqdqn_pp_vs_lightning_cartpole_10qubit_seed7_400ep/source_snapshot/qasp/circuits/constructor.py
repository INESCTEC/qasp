"""
constructor.py
==============
Flexible Pennylane circuit constructor with customizable measurements.
"""

import functools
from functools import partial

import pennylane as qml
import torch
from pennylane import numpy as np, qnode

import os
from qasp.devices import create_device

class CircuitBuilder:
    """
    Builds parameterized circuits with pluggable measurement types.

    Example:
        builder = CircuitBuilder(n_qubits=4, dev_name="lightning.qubit")
        builder.set_ansatz(ansatz)
        builder.set_measurement("probs")
        qnode = builder.build()
        # now qnode(params) returns probabilities
    """

    def __init__(
        self,
        n_qubits: int,
        n_ancillas: int = 0,
        dev_kwargs: dict = None,
    ):
        self.n_qubits = n_qubits
        self.dev_kwargs = dict(dev_kwargs or {})
        self.ansatz = None
        self.feature_map = None
        self.measurement = ("probs", list(range(n_qubits)))
        self.obs = None
        self.is_compiled = False
        self.z_gates = None
        self.CNOT_Gates = None
        self.n_ancillas = n_ancillas

    def set_ansatz(self, ansatz_fn):
        """
        ansatz_fn(params) -> None: user-defined gate sequence.
        Should apply gates on wires 0..n_qubits-1.
        """
        self.ansatz = ansatz_fn

    def set_CNOT_gates(self, CNOT_Gates: list):
        """
        CNOT_Gates (list[tuple[int, int]]): list of (control, target) qubit pairs for CNOT gates.
        """
        self.CNOT_Gates = CNOT_Gates
        self.ansatz.CNOT_Gates = CNOT_Gates

    def set_parity_check(self):
        parity_gates = []
        for i in range(self.n_ancillas):
            for j in range(self.n_qubits - i):
                parity_gates.append((j, self.n_qubits + self.n_ancillas - 1 - i))

        self.ansatz.parity_gates = parity_gates

    def set_z_gates(self, z_gates: list):
        """
        z_gates (list[int]): list of qubits to apply Z gates before measurement.
        """
        self.z_gates = z_gates

    def set_feature_map(self, feature_map_fn):
        """
        feature_map_fn(params) -> None: user-defined feature map.
        Should apply gates on wires 0..n_qubits-1.
        """
        self.feature_map = feature_map_fn

    def set_measurement(
        self,
        meas_type: str,
        wires=None,
        observables=None,
    ):
        """
        Define measurement for the circuit.

        Args:
            meas_type (str): one of 'probs', 'state', 'expectation', 'var', 'sample'.
            wires (list[int], optional): wires for probs/state/sample.
            observables (list[qml.operation], optional): list of observables for expectation/var.
        """
        valid = {"probs", "state", "expectation", "variance", "sample"}
        if meas_type not in valid:
            raise ValueError(f"Invalid measurement '{meas_type}'. Choose from {valid}.")

        if wires is None:
            wires = range(self.n_qubits)
        self.measurement = (meas_type, wires)
        if meas_type in {"expectation", "variance"}:
            if observables is None:
                raise ValueError(
                    "observables must be provided for expectation/variance."
                )
            self.obs = []
            for item in observables:
                # Accept already constructed PennyLane observables, including
                # SparseHamiltonian action projectors.
                if isinstance(item, qml.operation.Operator):
                    self.obs.append(item)
                    continue
                if not isinstance(item, str):
                    raise TypeError(
                        "Expectation observables must be PennyLane operators or "
                        f"Pauli-string specifications; received {type(item)}."
                    )

                signal = 1
                parts = item.split()
                pauli_string = parts[0]
                if pauli_string.startswith("-"):
                    signal = -1
                    pauli_string = pauli_string[1:]
                wire_indices = list(map(int, parts[1:]))
                wire_map = {wire: idx for idx, wire in enumerate(wire_indices)}
                #print(f"Parsed observable: signal={signal}, pauli_string='{pauli_string}', wires={wire_indices} , wire_map={wire_map}", flush=True)
                # Use PennyLane's string_to_pauli_word
                pauli_word = qml.pauli.string_to_pauli_word(
                    pauli_string, wire_map=wire_map
                )
                #self.obs.append(qml.Hamiltonian([signal * 1.0], [pauli_word]))
                self.obs.append(signal * pauli_word)
                #print(f"Constructed observable: {self.obs[-1]}", flush=True)

    def build(self):
        """Construct and return the QNode with the chosen ansatz and measurement."""
        if self.ansatz is None:
            raise RuntimeError("Ansatz function not set. Call set_ansatz() first.")

        meas_type, wires = self.measurement
        # @qml.compile
        wires = sorted(wires)  # Ensure wires are in a consistent order
        if self.ansatz.feature_map is not None:
            if self.ansatz.feature_map.input_scaling:

                def circuit_fm_is(inputs=None, input_scaling_params=None, params=None):

                    self.ansatz(
                        inputs=inputs,
                        input_scaling_params=input_scaling_params,
                        params=params,
                    )

                    if meas_type == "probs":
                        return qml.probs(wires=wires)
                    if meas_type == "state":
                        return qml.state()
                    if meas_type == "sample":
                        return qml.sample(wires=wires)
                    if meas_type == "expectation":
                        return tuple(qml.expval(o) for o in self.obs)
                    if meas_type == "variance":
                        return tuple(qml.var(o) for o in self.obs)
                    # unreachable
                    raise RuntimeError("Unknown measurement type")
                    # wrap in QNode

                self.circuit = circuit_fm_is

            else:

                def circuit_fm(inputs=None, params=None):
                    self.ansatz(inputs=inputs, input_scaling_params=None, params=params)

                    if meas_type == "probs":
                        return qml.probs(wires=wires)
                    if meas_type == "state":
                        return qml.state()
                    if meas_type == "sample":
                        return qml.sample(wires=wires)
                    if meas_type == "expectation":
                        return tuple(qml.expval(o) for o in self.obs)
                    if meas_type == "variance":
                        return tuple(qml.var(o) for o in self.obs)
                    # unreachable
                    raise RuntimeError("Unknown measurement type")
                    # wrap in QNode

                self.circuit = circuit_fm
        else:

            def circuit(params=None):
                self.ansatz(params=params)

                if meas_type == "probs":
                    return qml.probs(wires=wires)
                if meas_type == "state":
                    return qml.state()
                if meas_type == "sample":
                    return qml.sample(wires=wires)
                if meas_type == "expectation":
                    return tuple(qml.expval(o) for o in self.obs)
                if meas_type == "variance":
                    return tuple(qml.var(o) for o in self.obs)
                # unreachable
                raise RuntimeError("Unknown measurement type")
                # wrap in QNode

            self.circuit = circuit

        return self.circuit

    def build_adaptive(self, for_torchlayer=False):
        """Construct and return the QNode with the chosen ansatz and measurement that changes adaptively
        Input of the model now has two dimensions (features, qubits or observables to measure)."""
        if self.ansatz is None:
            raise RuntimeError("Ansatz function not set. Call set_ansatz() first.")

        self.for_torchlayer = for_torchlayer
        meas_type, wires = self.measurement
        # @qml.compile

        if self.ansatz.feature_map is not None:
            if self.ansatz.feature_map.input_scaling:

                def circuit_fm_is(inputs=None, input_scaling_params=None, params=None):
                    if self.for_torchlayer:
                        inpt = inputs[: self.ansatz.feature_map.features]
                        inpt_ = inputs[self.ansatz.feature_map.features :]
                        if type(inpt_) == torch.Tensor:
                            inpt_measure = inpt_.int().numpy()
                        else:
                            inpt_measure = inpt_
                    else:
                        inpt = inputs[0]
                        inpt_measure = inputs[1]

                    self.ansatz(
                        inputs=inpt,
                        input_scaling_params=input_scaling_params,
                        params=params,
                    )

                    if meas_type == "probs":
                        return qml.probs(wires=inpt_measure)
                    if meas_type == "state":
                        return qml.state()
                    if meas_type == "sample":
                        return qml.sample(wires=inpt_measure)
                    if meas_type == "expectation":
                        return [qml.expval(o) for o in inpt_measure]
                    if meas_type == "variance":
                        return [qml.var(o) for o in inpt_measure]
                    # unreachable
                    raise RuntimeError("Unknown measurement type")
                    # wrap in QNode

                self.circuit = circuit_fm_is

            else:

                def circuit_fm(inputs=None, params=None):
                    if self.for_torchlayer:
                        inpt = inputs[: self.ansatz.feature_map.features]
                        inpt_ = inputs[self.ansatz.feature_map.features :]
                        if type(inpt_) == torch.Tensor:
                            inpt_measure = inpt_.int().numpy()
                        else:
                            inpt_measure = inpt_
                    else:
                        inpt = inputs[0]
                        inpt_measure = inputs[1]

                    self.ansatz(inputs=inpt, input_scaling_params=None, params=params)

                    if meas_type == "probs":
                        return qml.probs(wires=inpt_measure)
                    if meas_type == "state":
                        return qml.state()
                    if meas_type == "sample":
                        return qml.sample(wires=inpt_measure)
                    if meas_type == "expectation":
                        return [qml.expval(o) for o in inpt_measure]
                    if meas_type == "variance":
                        return [qml.var(o) for o in inpt_measure]
                    # unreachable
                    raise RuntimeError("Unknown measurement type")
                    # wrap in QNode

                self.circuit = circuit_fm
        else:
            # no inputs for this model, so receives the adaptive measurement as a new argument "measurement",
            # but it does not work with torch since it interprets as a trainable parameter which is not
            def circuit(params=None, measurement=None):
                self.ansatz(params=params)

                if meas_type == "probs":
                    return qml.probs(wires=measurement)
                if meas_type == "state":
                    return qml.state()
                if meas_type == "sample":
                    return qml.sample(wires=measurement)
                if meas_type == "expectation":
                    return [qml.expval(o) for o in measurement]
                if meas_type == "variance":
                    return [qml.var(o) for o in measurement]
                # unreachable
                raise RuntimeError("Unknown measurement type")
                # wrap in QNode

            self.circuit = circuit

        return self.circuit

    def build_kernel(self, for_torchlayer=False):
        """Construct an overlap circuit for fidelity-style kernels.

        The circuit applies ``U(x)`` followed by ``U(x')^\u2020`` while sharing
        the same variational parameters. For a TorchLayer, the two examples
        are supplied as one concatenated input; otherwise they are supplied as
        a two-row array.
        """
        if self.ansatz is None:
            raise RuntimeError("Ansatz function not set. Call set_ansatz() first.")

        self.for_torchlayer = for_torchlayer
        meas_type, wires = self.measurement
        wires = sorted(wires)
        feature_map = self.ansatz.feature_map
        n_features = self.n_qubits
        if feature_map is not None and feature_map.features is not None:
            n_features = feature_map.features

        def split_inputs(inputs):
            if for_torchlayer:
                return inputs[:n_features], inputs[n_features : 2 * n_features]
            return inputs[0], inputs[1]

        def measure():
            if meas_type == "probs":
                return qml.probs(wires=wires)
            if meas_type == "state":
                return qml.state()
            if meas_type == "sample":
                return qml.sample(wires=wires)
            if meas_type == "expectation":
                return tuple(qml.expval(o) for o in self.obs)
            if meas_type == "variance":
                return tuple(qml.var(o) for o in self.obs)
            raise RuntimeError("Unknown measurement type")

        if feature_map is not None and feature_map.input_scaling:

            def circuit(inputs=None, input_scaling_params=None, params=None):
                x, x_prime = split_inputs(inputs)
                self.ansatz(
                    inputs=x,
                    input_scaling_params=input_scaling_params,
                    params=params,
                )
                qml.adjoint(self.ansatz)(
                    inputs=x_prime,
                    input_scaling_params=input_scaling_params,
                    params=params,
                )
                return measure()

        else:

            def circuit(inputs=None, params=None):
                x, x_prime = split_inputs(inputs)
                self.ansatz(inputs=x, input_scaling_params=None, params=params)
                qml.adjoint(self.ansatz)(
                    inputs=x_prime,
                    input_scaling_params=None,
                    params=params,
                )
                return measure()

        self.circuit = circuit
        return self.circuit

    def compile(
        self,
        device="auto",
        diff_method="best",
        method="mps",
        shots=None,
        mpi=False,
        cut_circuit=False,
        dev_wires=10,
        spsa_num_directions=1,
        device_vjp=False,
        batch_obs=True,
        device_kwargs=None,
    ):
        """
        Compile the circuit for a specific device and differentiation method.

        Args:
            device (str): device name or 'auto' for automatic selection.
            diff_method (str): differentiation method ('backprop', 'parameter-shift', etc.).
            mpi (bool): forward MPI execution to the selected plugin; auto
                selects lightning.kokkos when enabled.
            device_kwargs (dict): additional PennyLane device options.
        """
        options = dict(self.dev_kwargs)
        options.update(device_kwargs or {})
        if mpi and device == "lightning.pp":
            raise ValueError("lightning.pp does not support mpi=True.")
        # Auto selection must not silently select a serial simulator for
        # probability or sample measurements when MPI was requested.
        if mpi and device == "auto":
            device = "lightning.kokkos"
        if isinstance(device, qml.devices.Device):
            self.device = create_device(
                device, wires=self.n_qubits + self.n_ancillas,
                shots=shots, mpi=mpi, device_kwargs=options,
            )
        elif device == "auto":
            if self.measurement[0] == "probs":
                self.device = create_device(
                    "default.qubit", wires=self.n_qubits + self.n_ancillas + 1,
                    shots=shots, device_kwargs=options,
                )
            elif self.measurement[0] == "sample":
                self.device = create_device(
                    "lightning.qubit",
                    wires=self.n_qubits + self.n_ancillas + 1,
                    shots=1 if shots is None else shots, device_kwargs=options,
                )
            elif self.measurement[0] == "expectation":
                print("Using Kokkos device for expectation measurements.", flush=True)
                options.setdefault("batch_obs", batch_obs)
                self.device = create_device(
                    "lightning.kokkos",
                    wires=self.n_qubits + self.n_ancillas + 1,
                    shots=shots, device_kwargs=options,
                )

        elif device == "default.tensor":
            options.setdefault("method", method)
            self.device = create_device(
                "default.tensor",
                wires=self.n_qubits + self.n_ancillas + 1,
                shots=shots, mpi=mpi, device_kwargs=options,
            )
        
        elif device == "lightning.pp":
            # Parameter-owning placeholder only; PP executes the traced tape.
            # Do not require an optional Lightning/Kokkos installation here.
            self.device = create_device(
                "default.qubit", wires=self.n_qubits + self.n_ancillas + 1,
                shots=shots, device_kwargs=options,
            )

        elif device in {"lightning.kokkos", "lightning.gpu", "lightning.qubit"}:
            options.setdefault("batch_obs", batch_obs)
            self.device = create_device(
                device, wires=self.n_qubits + self.n_ancillas,
                mpi=mpi, shots=shots, device_kwargs=options,
            )
        elif device == "default.clifford":      
            options.setdefault("max_workers", None)
            self.device = create_device(
                "default.clifford", wires=self.n_qubits + self.n_ancillas,
                shots=shots, mpi=mpi, device_kwargs=options,
            )
        else:
            self.device = create_device(
                device, wires=self.n_qubits + self.n_ancillas,
                shots=shots, mpi=mpi, device_kwargs=options,
            )
         
        self.diff_method = diff_method


        #n_threads = str(os.cpu_count())

        #os.environ["OMP_NUM_THREADS"] = n_threads
        #os.environ["OMP_NUM_THREADS"] = "1"
        #os.environ["MKL_NUM_THREADS"] = "1"
        #os.environ["OPENBLAS_NUM_THREADS"] = "1"
        if cut_circuit:
            #self.qnode = qml.QNode(
                #self.circuit, device=qml.device(
                 #       "lightning.kokkos", wires=12, mpi=mpi,batch_obs=True
                  #  ), diff_method=self.diff_method
            #)
            #self.qnode = qml.transforms.make_tape(self.circuit)
            #make a class that returns a new object , the qtape and has a method called forward 
            class CutQNode:
                def __init__(self, circuit, device, diff_method):
                    self.qtape = qml.tape.make_qscript(circuit)
                    self.small_dev = create_device(
                        "lightning.kokkos", wires=dev_wires, shots=shots,
                        mpi=mpi, device_kwargs={"batch_obs": True, **options},
                    )
                    self.diff_method = diff_method

                def forward(self, *args, **kwargs):
                    
                    fragment_tapes, processing_fn = qml.cut_circuit(

                        self.qtape(*args, **kwargs),

                        device_wires=self.small_dev.wires,

                    )

                    fragment_results = qml.execute(fragment_tapes, self.small_dev, diff_method="adjoint", interface="torch")

                    return processing_fn(fragment_results)
            self.qnode = CutQNode(self.circuit, device=self.device, diff_method=self.diff_method)
            return self.qnode
        else:
            if device == "default.clifford":
                self.qnode = qml.QNode(
                    self.circuit, device=self.device, interface="torch" , diff_method=self.diff_method, gradient_kwargs={"num_directions": spsa_num_directions} if self.diff_method == "spsa" else None 
                )
                self.qnode = qml.transforms.clifford_t_decomposition(self.qnode)
                #self.qnode = qml.transforms.decompose(self.qnode, gate_set={"H", "X", "Y" , "Z" , "CNOT"})
                #self.qnode = qml.transforms.decompose(self.qnode , gate_set={qml.Hadamard , qml.PauliX , qml.PauliY , qml.PauliZ , qml.CNOT , qml.CZ})
            else:
                # ``lightning.pp`` is a QASP logical backend. Its QNode is
                # retained only so TorchLayer can own and initialize the
                # circuit parameters; execution and autograd are handled by
                # DifferentiablePauliProp from a traced tape. Asking the
                # placeholder for a device-specific derivative is unnecessary:
                # that QNode is never used for PP evaluation.
                qnode_diff_method = (
                    "best" if device == "lightning.pp" else self.diff_method
                )
                self.qnode = qml.QNode(
                    self.circuit,
                    device=self.device,
                    diff_method=qnode_diff_method,
                    device_vjp=device_vjp,
                    gradient_kwargs={"num_directions": spsa_num_directions}
                    if self.diff_method == "spsa"
                    else None,
                )
        #if shots is not None:
            #self.qnode = qml.set_shots(self.qnode, shots=shots)

        #self.qnode, proc_fn = qml.cut_circuit(self.qnode, device_wires=range(30),auto_cutter=True)

        self.is_compiled = True



        if device == "lightning.pp":
            self.ansatz_tape = qml.transforms.make_tape(self.circuit)
            """
            self.ansatz_tape_compiled = qml.compile(
                self.ansatz_tape,
                basis_set=["CNOT", "RX", "RY", "RZ"],
                num_passes=2
            )
            """

            return self.qnode, self.ansatz_tape  # _compiled
        else:
            return self.qnode

    def to_tikz(self, *args, **kwargs):
        """Return this builder's circuit as Quantikz/TikZ LaTeX source.

        The circuit is traced with ``args`` and circuit keyword arguments. See
        :func:`qasp.circuits.circuit_to_tikz` for formatting options.
        """
        from .tikz import circuit_to_tikz

        return circuit_to_tikz(self, *args, **kwargs)

    def save_tikz(self, path, *args, **kwargs):
        """Save this builder's Quantikz/TikZ source to a ``.tex`` file."""
        from .tikz import save_tikz

        return save_tikz(path, self, *args, **kwargs)

    def set_weights(self, params=None, input_scaling_params=None) -> qml.qnn.TorchLayer:
        if params is not None and input_scaling_params is not None:
            if self.ansatz.feature_map.input_scaling:
                self.qlayer.load_state_dict(
                    {"params": params, "input_scaling_params": input_scaling_params}
                )

            else:
                self.qlayer.load_state_dict({"params": params})

        else:
            self.qlayer.load_state_dict({"params": params})

    def create_torchlayer(
        self, params=None, input_scaling_params=None, initialization=None
    ) -> qml.qnn.TorchLayer:
        """
        Create a TorchLayer for the ansatz.

        Args:
            params (np.ndarray): initial parameters for the layer.
            initialization (str): initialization method for the parameters.

        Returns:
            qml.qnn.TorchLayer: TorchLayer object.
        """

        import torch
        import functools
        from pennylane import numpy as np

        def init_binary_0_pi_over_2(tensor: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                # Sample 0 or 1 with equal probability for each entry
                bits = torch.randint(
                    low=0,
                    high=2,
                    size=tensor.shape,
                    device=tensor.device,
                )
                tensor.copy_(bits * (np.pi))
            return tensor
        
        self.initialization = initialization

        if self.initialization == "random_0_2pi":
            self.init_method = functools.partial(
                torch.nn.init.uniform_, a=0, b=2 * np.pi
            )
        elif self.initialization == "random_-1_1":
            self.init_method = functools.partial(torch.nn.init.uniform_, a=-1, b=1)
        elif self.initialization == "random_0_1":
            self.init_method = functools.partial(torch.nn.init.uniform_, a=0, b=1)
        elif self.initialization == "random_half_1":
            self.init_method = functools.partial(torch.nn.init.uniform_, a=0.5, b=1)
        elif self.initialization == "random_-pi_pi":
            self.init_method = functools.partial(
                torch.nn.init.uniform_, a=-np.pi, b=np.pi
            )
        elif self.initialization == "zeros":
            self.init_method = functools.partial(torch.nn.init.zeros_)
        elif self.initialization == "ones":
            self.init_method = functools.partial(torch.nn.init.ones_)
        elif self.initialization == "constant":
            self.init_method = functools.partial(torch.nn.init.constant_, val=np.pi)
        elif self.initialization == "normal_0_1":
            self.init_method = functools.partial(torch.nn.init.normal_, mean=0.0, std=1)
        elif self.initialization == "normal_half_0_15":
            self.init_method = functools.partial(torch.nn.init.normal_, mean=0.5, std=0.15)
        elif self.initialization == "small_normal":
            self.init_method = functools.partial(torch.nn.init.normal_, mean=0.0, std=0.1)
        elif self.initialization == "discrete_0_pi_over_2":
            self.init_method = init_binary_0_pi_over_2
        else:
            self.init_method = None

        self.weight_shapes = {}
        shapes = self.ansatz.get_param_shapes()
        if (
            self.ansatz.feature_map is not None
            and self.ansatz.feature_map.input_scaling
        ):
            self.weight_shapes["input_scaling_params"] = shapes[0]

            self.weight_shapes["params"] = shapes[1]
        else:
            self.weight_shapes["params"] = shapes[0]

        self.qlayer = qml.qnn.TorchLayer(
            self.qnode, self.weight_shapes, init_method=self.init_method
        )

        if params is not None or input_scaling_params is not None:
            self.set_weights(params=params, input_scaling_params=input_scaling_params)

        return self.qlayer
