"""Independent finite-difference references for Fisher score kernels.

Fisher kernel inputs are outcomes, not continuous features. Encoding-only
circuits below differentiate input scales at a fixed encoded feature vector.
"""
import numpy as np
import pennylane as qml
import pytest
import torch
from qasp import AngleAnsatz, AngleEmbedding, CircuitBuilder
from qasp.qkernels import QFKernel
from qasp.simulators import UnsupportedIQP
from qasp.qkernels.quantum_kernels import EncodingAnsatz
from configuration_helpers import execution

OUTCOMES = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])


def fisher_node(layout, backend, interface, measurement='probs'):
    b = CircuitBuilder(2, n_ancillas=1)
    if layout == 'encoding':
        fm = AngleEmbedding(2, features=2, rotations='X', input_scaling=True)
        b.set_ansatz(EncodingAnsatz(2, feature_map=fm))
    else:
        fm = AngleEmbedding(2, features=2, rotations='X') if layout == 'both' else None
        b.set_ansatz(AngleAnsatz(2, n_layers=1, feature_map=fm,
            rotations=['X 0', 'X 1', 'XX 0 1'], include_imprimitive=False))
    b.set_measurement(measurement, wires=[0, 1]); b.build()
    options = execution(backend)
    if backend == 'lightning':
        options['diff_method'] = 'parameter-shift'  # probability readout
    compiled = b.compile(**options)
    def circuit(weights):
        features = (torch.tensor([.3, .7], dtype=weights.dtype, device=weights.device)
                    if isinstance(weights, torch.Tensor) else qml.numpy.array([.3, .7], requires_grad=False))
        if layout == 'encoding':
            return b.circuit(inputs=features, input_scaling_params=weights)
        if layout == 'both':
            return b.circuit(inputs=features, params=weights)
        return b.circuit(params=weights)
    return qml.QNode(circuit, compiled.device, interface=interface, diff_method=options['diff_method'])


@pytest.mark.parametrize('backend', ['statevector', 'lightning', 'mixed', 'iqp_tn', 'iqp_auto', 'pp'])
@pytest.mark.parametrize('layout', ['encoding', 'ansatz', 'both'])
@pytest.mark.parametrize('interface', ['autograd', 'torch'])
def test_classical_fisher_matches_probability_finite_difference(backend, layout, interface):
    node = fisher_node(layout, backend, interface)
    theta = np.array([.6, .9] if layout == 'encoding' else [.3, .5, .7])
    params = (torch.tensor(theta, requires_grad=True) if interface == 'torch'
              else qml.numpy.array(theta, requires_grad=True))
    if interface == 'autograd' and backend in {'iqp_tn', 'pp'}:
        # QASP's differentiable engines require Torch; verify the rejection.
        with pytest.raises((qml.exceptions.QuantumFunctionError, UnsupportedIQP), match='Torch|backprop|autograd'):
            QFKernel(node, params, fisher='classical', lambda_reg=.02)
        return
    kernel = QFKernel(node, params, fisher='classical', lambda_reg=.02)
    reference = fisher_node(layout, 'statevector', 'numpy')
    p = reference(theta)
    eps = 1e-5
    jac = np.column_stack([(reference(theta+eps*v)-reference(theta-eps*v))/(2*eps)
                          for v in np.eye(len(theta))])
    scores = jac / p[:, None]
    fim = jac.T @ (jac / p[:, None]) + .02*np.eye(len(theta))
    expected = scores @ np.linalg.solve(fim, scores.T)
    np.testing.assert_allclose(kernel.gram(OUTCOMES, normalize=False), expected, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(kernel.gram(OUTCOMES[:2], OUTCOMES[2:], normalize=False), expected[:2, 2:], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('layout', ['encoding', 'ansatz', 'both'])
@pytest.mark.parametrize('interface', ['autograd', 'torch'])
@pytest.mark.parametrize('backend', ['statevector', 'lightning', 'mixed', 'iqp_tn', 'iqp_auto', 'pp'])
def test_quantum_fisher_matches_state_finite_difference(layout, interface, backend):
    theta = np.array([.6, .9] if layout == 'encoding' else [.3, .5, .7])
    params = (torch.tensor(theta, requires_grad=True) if interface == 'torch'
              else qml.numpy.array(theta, requires_grad=True))
    node = fisher_node(layout, backend, interface)
    if interface == 'autograd' and backend in {'iqp_tn', 'pp'}:
        with pytest.raises((qml.exceptions.QuantumFunctionError, UnsupportedIQP), match='Torch|backprop|autograd'):
            QFKernel(node, params, lambda_reg=.02)
        return
    if backend == 'iqp_tn' and layout != 'encoding':
        with pytest.raises(UnsupportedIQP, match='CNOT|Noncommuting'):
            QFKernel(node, params, lambda_reg=.02)
        return
    kernel = QFKernel(node, params, lambda_reg=.02)
    state = fisher_node(layout, 'statevector', 'numpy', 'state')
    psi = state(theta)
    eps = 1e-5
    jac = np.column_stack([(state(theta+eps*v)-state(theta-eps*v))/(2*eps)
                          for v in np.eye(len(theta))])
    tangent = jac.conj().T @ psi
    fim = 4*np.real(jac.conj().T@jac - np.outer(tangent, tangent.conj()))
    np.testing.assert_allclose(kernel.Freg, fim + .02*np.eye(len(theta)), atol=1e-6, rtol=1e-6)
    gram = kernel.gram(OUTCOMES, normalize=False)
    assert np.isfinite(gram).all()
    np.testing.assert_allclose(gram, gram.T, atol=1e-8)
    assert np.linalg.eigvalsh(gram).min() > -1e-8


def test_fisher_normalization_validation_and_drawing():
    from qasp import draw_kernel
    import matplotlib.pyplot as plt
    node = fisher_node('ansatz', 'statevector', 'torch')
    params = torch.tensor([.3, .5, .7], dtype=torch.float64, requires_grad=True)
    kernel = QFKernel(node, params, fisher='classical')
    raw = kernel.gram(OUTCOMES, normalize=False)
    normalized = kernel.gram(OUTCOMES)
    np.testing.assert_allclose(normalized, raw/np.sqrt(np.outer(raw.diagonal(), raw.diagonal())), atol=1e-10)
    for invalid in [[[.1, 1]], [[1, 0, 1]], [[0, 2]]]:
        with pytest.raises(ValueError, match='binary'):kernel.gram(invalid)
    with pytest.raises(ValueError, match='fisher'):QFKernel(node, params, fisher='typo')
    with pytest.raises(ValueError, match='lambda_reg'):QFKernel(node, params, lambda_reg=-1)
    assert 'quantikz' in draw_kernel(kernel, output='tikz')
    figure, _ = draw_kernel(kernel)
    plt.close(figure)


def test_fisher_zero_probability_outcomes_are_explicitly_undefined():
    @qml.qnode(qml.device('default.qubit', wires=2), interface='autograd')
    def node(theta):
        qml.RY(theta[0], 0)
        return qml.probs(wires=[0, 1])
    kernel = QFKernel(node, qml.numpy.array([.4], requires_grad=True), fisher='classical')
    assert np.isfinite(kernel.gram([[0, 0], [1, 0]])).all()
    with pytest.raises(ValueError, match='zero-probability'):kernel.gram([[0, 1]])


@pytest.mark.parametrize('backend', ['mps', 'tn', 'iqp_vn'])
def test_fisher_requires_probability_capable_backend(backend):
    node = fisher_node('ansatz', backend, 'torch')
    params = torch.tensor([.3, .5, .7], dtype=torch.float64, requires_grad=True)
    with pytest.raises((NotImplementedError, UnsupportedIQP), match='probs|probabilities'):
        QFKernel(node, params, fisher='classical')


@pytest.mark.parametrize('parameter_type', ['list', 'numpy'])
def test_fisher_accepts_plain_parameters_and_keeps_a_snapshot(parameter_type):
    original = [.3, .5, .7]
    params = np.array(original) if parameter_type == 'numpy' else original.copy()
    kernel = QFKernel(fisher_node('ansatz', 'statevector', 'autograd'), params, fisher='classical')
    before = kernel.gram(OUTCOMES)
    params[0] = .8
    np.testing.assert_array_equal(kernel.params, original)
    np.testing.assert_array_equal(kernel.gram(OUTCOMES), before)
