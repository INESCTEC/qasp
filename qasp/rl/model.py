"""Dictionary-based RL policy creation and circuit drawing adapters.

create_policy selects the current Born decoder, softmax policy or PQDQN.
Configuration dictionaries and supplied builders are copied before construction."""
import numpy as np
import pennylane as qml

from copy import deepcopy


def create_policy(policy_class, policy_config_kwargs, n_qubits):
    """Create an RL policy from a selector, component dictionary and wire count.

    Selectors are PQCBorn (constructs PQCBornVariance), PQCSoftmax and PQDQN.
    For policies, resolve registered ansatz/feature-map names and callable
    n_actions, n_layers and features using n_qubits. Optional parity CNOTs
    configure ancilla readout. The PQDQN route delegates to create_qfunction.

    Returns the constructed Torch module without mutating caller configuration.
    A circuit_builder bypasses component construction. An ansatz without a
    feature map creates an input-independent policy; input placeholders retain
    their batch axes. Omitted ansatz_config uses the template defaults;
    feature-map configurations must specify their encoding rotations."""
    if policy_class == "PQDQN":
        from qasp.rl.qfunctions import create_qfunction
        return create_qfunction(policy_config_kwargs, n_qubits=n_qubits)
    if policy_class == "PQCBorn":
        from qasp.rl.policies import PQCBornVariance
        policy = PQCBornVariance
    elif policy_class == "PQCSoftmax":
        from qasp.rl.policies import PQCSoftmax
        policy = PQCSoftmax


    else:
        raise ValueError("policy_class must be 'PQCBorn', 'PQCSoftmax', or 'PQDQN'.")

    # Device instances are caller-owned execution resources (and expose live
    # reports/caches). Copy config data, while retaining that device identity.
    device = policy_config_kwargs.get("device")
    resolved_config = deepcopy(policy_config_kwargs, memo={id(device): device})
    if callable(resolved_config.get("n_actions")):
        resolved_config["n_actions"] = resolved_config["n_actions"](n_qubits)
    if resolved_config.get("circuit_builder") is None:
        if resolved_config.get("ansatz") is None:
            raise ValueError("Provide an ansatz or circuit_builder.")
        resolved_config["ansatz_config"] = dict(resolved_config.get("ansatz_config") or {})
        resolved_config["ansatz_config"].setdefault("n_layers", 1)
        resolved_config["feature_map_config"] = dict(resolved_config.get("feature_map_config") or {})
        if resolved_config.get("feature_map") is None:
            from qasp.circuits import CircuitBuilder, AngleAnsatz, IQPAnsatz
            cls = {"AngleAnsatz": AngleAnsatz, "IQPAnsatz": IQPAnsatz}.get(
                resolved_config["ansatz"], resolved_config["ansatz"])
            ac = dict(resolved_config["ansatz_config"] or {})
            ac["n_qubits"] = n_qubits
            if callable(ac.get("n_layers")):
                ac["n_layers"] = ac["n_layers"](n_qubits)
            builder = CircuitBuilder(n_qubits)
            builder.set_ansatz(cls(**ac))
            resolved_config["circuit_builder"] = builder
        else:
            if callable(resolved_config["ansatz_config"].get("n_layers")):
                resolved_config["ansatz_config"]["n_layers"] = resolved_config["ansatz_config"]["n_layers"](n_qubits)
            if callable(resolved_config["feature_map_config"].get("features")):
                resolved_config["feature_map_config"]["features"] = resolved_config["feature_map_config"]["features"](n_qubits)
    
            if resolved_config["ansatz"] == "IQPAnsatz":
                from qasp.circuits.templates import IQPAnsatz
                resolved_config["ansatz"] = IQPAnsatz
            elif resolved_config["ansatz"] == "AngleAnsatz":
                from qasp.circuits.templates import AngleAnsatz
                resolved_config["ansatz"] = AngleAnsatz

            if resolved_config["feature_map"] == "AngleEmbedding":
                from qasp.feature_maps.templates import AngleEmbedding
                resolved_config["feature_map"] = AngleEmbedding
            elif resolved_config["feature_map"] == "NeuralEmbedding":
                from qasp.feature_maps.templates import NeuralEmbedding
                resolved_config["feature_map"] = NeuralEmbedding
    


            resolved_config["feature_map_config"]["n_qubits"] = n_qubits
            resolved_config["feature_map_config"].setdefault("features", n_qubits)
            resolved_config["ansatz_config"]["n_qubits"] = n_qubits


    if resolved_config.get("policy") is not None:
        if (resolved_config["policy"] == "parity-like" and policy_class == "PQCBorn"):
            if resolved_config.get("parity_cnot_decomposition", False):
                n_ancillas = int(np.ceil(np.log2(resolved_config["n_actions"])))
                gates = []
                if (resolved_config.get("measurement_qubits") is not None):
                    qubits_to_use = resolved_config["measurement_qubits"]

                    for i in range(n_ancillas):
                        list_qubits  = qubits_to_use[:-i] if i  !=0 else qubits_to_use
                        for value in list_qubits:
                            gates.append((value,n_qubits+n_ancillas-1-i))

                
                else:
                    for value in range(n_ancillas):
                        for i in range(n_qubits - value):
                            gates.append((i,n_qubits+n_ancillas-1-value))


                resolved_config["n_ancillas"] = n_ancillas
                resolved_config["CNOT_gates"] = gates
                measure_qubits = [n_qubits+n_ancillas-1-i for i in range(n_ancillas)]
                resolved_config["measurement_qubits"] = measure_qubits

            pqc_model = policy(**resolved_config, n_qubits=n_qubits)
            return pqc_model
        

    if policy_class == "PQCBorn":
        pqc_model = policy(**resolved_config, n_qubits=n_qubits)
    elif policy_class == "PQCSoftmax":
        pqc_model = policy(**resolved_config, n_qubits=n_qubits)

    return pqc_model


def draw_model(
    num_inputs=1,
    policy_class=None,
    policy_config_kwargs=None,
    n_qubits=None,
    model=None,
    inputs=None,
    output="matplotlib",
    tikz_options=None,
    return_tex: bool = True,
):
    """Draw a QASP policy or PQDQN as Matplotlib (default) or Quantikz/TikZ.

    Set ``output="tikz"`` to return LaTeX source. ``tikz_options`` is passed
    to :func:`qasp.circuits.circuit_to_tikz` and can contain options such as
    ``standalone``, ``wire_labels``, and ``symbolic_parameters``.
    For PQDQN, omitted inputs use the configured feature count and zero values.

    In Jupyter, TikZ output previews the generated figure. By default the
    LaTeX string is also returned; return_tex=False displays only.
    """

    from qasp.rl.qfunctions import PQDQN, draw_qfunction

    if isinstance(model, PQDQN) or (model is None and policy_class == "PQDQN"):
        return draw_qfunction(
            model,
            config=policy_config_kwargs,
            n_qubits=n_qubits,
            inputs=inputs,
            output=output,
            tikz_options=tikz_options,
            return_tex=return_tex,
        )

    if model is None:
        if policy_class is None or policy_config_kwargs is None or n_qubits is None:
            raise ValueError("Provide model or policy_class, policy_config_kwargs and n_qubits.")
        pqc_model = create_policy(policy_class, policy_config_kwargs, n_qubits)
    else:
        pqc_model = model

    if output not in {"matplotlib", "mpl", "tikz", "tex", "latex"}:
        raise ValueError("output must be 'matplotlib' or 'tikz'")

    draw_inputs = inputs if inputs is not None else np.random.random(num_inputs)
    circuit_kwargs = {
        "inputs": draw_inputs,
        "params": pqc_model.qlayer.state_dict()["params"],
    }
    builder = getattr(pqc_model, "circuit_builder", None)
    feature_map = getattr(getattr(builder, "ansatz", None), "feature_map", getattr(pqc_model, "fm", None))
    if feature_map is None:
        circuit_kwargs.pop("inputs", None)
    if getattr(feature_map, "input_scaling", False):
        circuit_kwargs["input_scaling_params"] = pqc_model.qlayer.state_dict()[
            "input_scaling_params"
        ]

    if output in {"tikz", "tex", "latex"}:
        from qasp.circuits import circuit_to_tikz
        from qasp.circuits.preview import _tikz_result

        circuit_source = getattr(pqc_model, "circuit_builder", pqc_model.circuit)
        tex = circuit_to_tikz(
            circuit_source,
            circuit_kwargs=circuit_kwargs,
            **(tikz_options or {}),
        )
        return _tikz_result(tex, return_tex)

    fig, ax = qml.draw_mpl(
        pqc_model.circuit, decimals=2, style="sketch"
    )(**circuit_kwargs)

    return fig, ax
