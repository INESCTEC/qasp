from types import SimpleNamespace

import pytest

from qasp.circuits.tikz import circuit_to_tikz, tape_to_tikz


class Operation:
    def __init__(self, name, wires, data=(), **attributes):
        self.name = name
        self.wires = wires
        self.data = data
        for key, value in attributes.items():
            setattr(self, key, value)


class ProbabilityMP:
    def __init__(self, wires=()):
        self.wires = wires
        self.obs = None


class ExpectationMP:
    def __init__(self, observable):
        self.obs = observable
        self.wires = observable.wires


def make_tape(operations, measurements, wires):
    return SimpleNamespace(operations=operations, measurements=measurements, wires=wires)


def test_exports_common_gates_controls_and_measurements():
    tape = make_tape(
        [
            Operation("Hadamard", [0]),
            Operation("RX", [1], [0.125]),
            Operation("CNOT", [0, 1]),
            Operation("PauliRot", [0, 1], [0.5], hyperparameters={"pauli_word": "ZZ"}),
        ],
        [ProbabilityMP([0, 1])],
        [0, 1],
    )

    result = tape_to_tikz(tape, decimals=2)

    assert r"\begin{quantikz}" in result
    assert r"\gate[style={fill=qaspSingleGate}]{H}" in result
    assert r"\gate[style={fill=qaspSingleGate}]{R_x(0.12)}" in result
    assert r"\ctrl{1}" in result
    assert r"\targ{}" in result
    assert r"\gate[wires=2, style={fill=qaspTwoQubitGate}]{R_{ZZ}(0.5)}" in result
    assert result.count(r"\meter[style={fill=qaspMeasurement}]{}") == 2


def test_symbolic_parameters_and_custom_wire_labels():
    tape = make_tape(
        [Operation("RY", ["top"], [1.2]), Operation("RZ", ["bottom"], [2.4])],
        [],
        ["top", "bottom"],
    )

    result = tape_to_tikz(
        tape,
        symbolic_parameters=True,
        wire_labels={"top": r"q_{\mathrm{data}}", "bottom": "a"},
    )

    assert r"R_y(\theta_{1})" in result
    assert r"R_z(\theta_{2})" in result
    assert r"\lstick{$q_{\mathrm{data}}$}" in result


def test_angle_embedding_expands_to_feature_labelled_rotations():
    class RY:
        pass

    embedding = Operation(
        "AngleEmbedding",
        [0, 1],
        ([0.1, 0.2],),
        hyperparameters={"rotation": RY},
    )
    tape = make_tape(
        [embedding, Operation("RX", [0], [0.3])],
        [],
        [0, 1],
    )

    result = tape_to_tikz(tape, symbolic_parameters=True)

    assert r"R_y(f_{1})" in result
    assert r"R_y(f_{2})" in result
    assert r"R_x(\theta_{1})" in result
    assert "AngleEmbedding" not in result


def test_scaled_and_adjoint_angle_embedding_uses_feature_scaling_labels():
    class RZ:
        pass

    embedding = Operation(
        "AngleEmbedding",
        [0, 1],
        ([0.1, 0.2],),
        hyperparameters={"rotation": RZ},
    )
    adjoint_embedding = Operation(
        "Adjoint(AngleEmbedding)",
        [0, 1],
        base=embedding,
    )
    tape = make_tape([embedding, adjoint_embedding], [], [0, 1])

    result = tape_to_tikz(
        tape,
        symbolic_parameters=True,
        feature_map_input_scaling=True,
    )

    assert result.count(r"R_z(f_{1}\lambda_{1})") == 1
    assert result.count(r"R_z(f_{2}\lambda_{2})") == 1
    assert r"R_z^{\dagger}(f_{1}\lambda_{1})" in result
    assert r"R_z^{\dagger}(f_{2}\lambda_{2})" in result


def test_expectation_and_standalone_document():
    observable = Operation("PauliZ", [0])
    tape = make_tape([], [ExpectationMP(observable)], [0])

    result = tape_to_tikz(tape, standalone=True)

    assert result.startswith(r"\documentclass[tikz,border=4pt]{standalone}")
    assert r"\usepackage{quantikz}" in result
    assert (
        r"\meter[style={fill=qaspMeasurement}]{\langle Z\rangle}" in result
    )
    assert result.rstrip().endswith(r"\end{document}")


def test_parameter_labels_must_match_parameter_count():
    tape = make_tape([Operation("RX", [0], [0.1])], [], [0])

    with pytest.raises(ValueError, match="exactly one entry"):
        tape_to_tikz(tape, parameter_labels=[r"\alpha", r"\beta"])


def test_qnode_like_objects_are_traced_with_circuit_arguments():
    tape = make_tape([Operation("RY", [0], [0.25])], [], [0])

    class QNodeLike:
        def construct(self, args, kwargs):
            assert args == ("positional",)
            assert kwargs == {"inputs": "features"}
            return tape

    result = circuit_to_tikz(
        QNodeLike(),
        "positional",
        inputs="features",
        parameter_labels=[r"\phi"],
    )

    assert r"\gate[style={fill=qaspSingleGate}]{R_y(\phi)}" in result


def test_generic_controlled_gate_and_zero_control():
    base = Operation("RY", [2], [0.4])
    controlled = Operation(
        "C(RY)",
        [0, 1, 2],
        [0.4],
        control_wires=[0, 1],
        control_values=[True, False],
        base=base,
    )
    tape = make_tape([controlled], [], [0, 1, 2])

    result = tape_to_tikz(tape, parameter_labels=[r"\gamma"])

    assert r"\ctrl{2}" in result
    assert r"\octrl{1}" in result
    assert r"\gate[style={fill=qaspTwoQubitGate}]{R_y(\gamma)}" in result


@pytest.mark.parametrize(
    ("wires", "control_command"),
    [([0, 1], r"\ctrl{1}"), ([1, 0], r"\ctrl{-1}")],
)
def test_adjoint_cnot_uses_the_standard_cnot_symbol(wires, control_command):
    target = Operation("PauliX", [wires[1]])
    cnot = Operation("CNOT", wires, base=target)
    adjoint_cnot = Operation("Adjoint(CNOT)", wires, base=cnot)
    tape = make_tape([adjoint_cnot], [], [0, 1])

    result = tape_to_tikz(tape)

    assert control_command in result
    assert r"\targ{}" in result
    assert r"\operatorname{CNOT}" not in result


def test_independent_gates_are_packed_into_one_column():
    tape = make_tape(
        [Operation("Hadamard", [0]), Operation("PauliX", [1])], [], [0, 1]
    )

    rows = [line for line in tape_to_tikz(tape).splitlines() if "lstick" in line]

    assert rows[0].count(" & ") == 2
    assert rows[1].count(" & ") == 2
    assert r"\gate[style={fill=qaspSingleGate}]{H}" in rows[0]
    assert r"\gate[style={fill=qaspSingleGate}]{X}" in rows[1]


def test_non_adjacent_gate_does_not_span_unused_wire():
    tape = make_tape(
        [Operation("PauliRot", [0, 2], [0.5], hyperparameters={"pauli_word": "ZZ"})],
        [],
        [0, 1, 2],
    )

    result = tape_to_tikz(tape)

    paired_gate = r"\gate[style={fill=qaspTwoQubitGate}]{R_{ZZ}(0.5)}"
    assert paired_gate + r"\vqw{2}" in result
    assert result.count(paired_gate) == 2
    assert r"\gate[wires=3]" not in result


def test_pauli_product_measurement_appears_on_every_wire():
    operands = [Operation("PauliZ", [wire]) for wire in range(3)]
    observable = Operation("Prod", [0, 1, 2], operands=operands)
    tape = make_tape([], [ExpectationMP(observable)], [0, 1, 2])

    result = tape_to_tikz(tape)

    assert result.count(r"\meter[style={fill=qaspMeasurement}]{Z}") == 3


def test_palette_can_be_customized_or_disabled():
    tape = make_tape([Operation("RY", [0], [0.2])], [], [0])

    custom = tape_to_tikz(tape, gate_colors={"single": "ABCDEF"})
    plain = tape_to_tikz(tape, gate_colors=False)

    assert r"\definecolor{qaspSingleGate}{HTML}{ABCDEF}" in custom
    assert r"\gate[style={fill=qaspSingleGate}]{R_y(0.2)}" in custom
    assert r"\gate{R_y(0.2)}" in plain


def test_compression_expands_first_layer_and_abstracts_later_layers():
    operations = [Operation("Hadamard", [0])]
    operations.extend(
        Operation("RY", [wire], [0.1 * (index + 1)])
        for index, wire in enumerate([0, 1, 0, 1, 0, 1])
    )
    tape = make_tape(operations, [ProbabilityMP([0, 1])], [0, 1])

    result = tape_to_tikz(
        tape,
        compress=True,
        n_layers=3,
        first_layer_start=1,
        first_layer_operation_count=2,
        symbolic_parameters=True,
    )

    assert "execute at end picture={" in result
    assert r"\draw[dashed,rounded corners]" in result
    assert r"{$U(\theta_{1})$}" in result
    assert r"{U(\theta_{2})}" in result
    assert r"{U(\theta_{3})}" in result
    assert r"R_y(\theta_{3})" not in result


def test_large_compressed_circuit_uses_ellipsis():
    tape = make_tape(
        [Operation("RY", [0], [float(index)]) for index in range(8)], [], [0]
    )

    result = tape_to_tikz(
        tape,
        compress=True,
        n_layers=8,
        first_layer_operation_count=1,
        max_explicit_layers=4,
    )

    assert r"U(\theta_{2})" in result
    assert r"\push{\cdots}" in result
    assert r"U(\theta_{8})" in result
    assert r"U(\theta_{3})" not in result


def test_builder_compression_metadata_is_inferred_automatically():
    class AngleAnsatz:
        n_layers = 3
        n_qubits = 2
        _block_size = 2
        reuploading = False
        feature_map = None
        include_imprimitive = False
        include_final_imprimitive = False
        imprimitive_edges = []
        z_gates = None
        parity_gates = None
        CNOT_Gates = None

    tape = make_tape(
        [Operation("Hadamard", [0])]
        + [Operation("RY", [index % 2], [0.1]) for index in range(6)],
        [],
        [0, 1],
    )

    class QNodeLike:
        def construct(self, args, kwargs):
            return tape

    class BuilderLike:
        n_qubits = 2
        n_ancillas = 0
        ansatz = AngleAnsatz()
        circuit = QNodeLike()

        def build(self):
            return self.circuit

    result = circuit_to_tikz(BuilderLike(), compress=True)

    assert r"{$U(\theta_{1})$}" in result
    assert r"U(\theta_{2})" in result
    assert r"U(\theta_{3})" in result


def test_identity_feature_does_not_overwrite_a_physical_measurement():
    z = ExpectationMP(Operation("PauliZ", [0]))
    identity = ExpectationMP(Operation("Identity", [0]))
    for measurements in ([z, identity], [identity, z]):
        result = tape_to_tikz(make_tape([], measurements, [0]))
        assert r"\langle Z\rangle" in result
        assert r"\langle I\rangle" not in result
