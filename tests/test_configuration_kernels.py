"""Fidelity/projected kernels with encoding-only and trainable configurations."""
from copy import deepcopy
import numpy as np
import pytest
import torch
from qasp import create_kernel, draw_kernel
from qasp.initialization import Normal
from qasp.simulators import UnsupportedIQP
from configuration_helpers import BACKENDS, execution, components, compare_tensors

X = np.array([[.2, -.3], [.6, .4], [-.7, .8]])
Y = np.array([[.9, -.2], [.3, .7]])


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('layout', ['encoding', 'angle', 'iqp'])
@pytest.mark.parametrize('embedding', ['angle', 'scaled', 'words', 'neural'])
@pytest.mark.parametrize('kind', ['fidelity', 'projected'])
def test_kernel_configurations_against_statevector(backend, layout, embedding, kind):
    cfg = dict(components(embedding, layout), kernel_type=kind)
    if kind == 'projected':
        cfg['observables'] = ['Z 0', 'Z 1', 'ZZ 0 1']
    if layout == 'encoding' and embedding in {'scaled', 'neural'}:
        cfg['input_scaling_params'] = np.linspace(.7, 1.1, 4 if embedding == 'neural' else 2)
    if layout != 'encoding':
        cfg['init'] = Normal(.2, .1)
    torch.manual_seed(32)
    actual = create_kernel({**deepcopy(cfg), **execution(backend)}, n_qubits=2)
    reference = create_kernel({**deepcopy(cfg), **execution('statevector')}, n_qubits=2)
    if kind == 'fidelity' and backend == 'iqp_vn':
        # A projector is not a compact Pauli sentence; the all-terms VN
        # estimator must reject it. Importance mode is tested separately.
        with pytest.raises(UnsupportedIQP, match='compact Pauli'):
            actual.gram(X)
        return
    if actual.trainable:
        reference.qlayer.load_state_dict(actual.qlayer.state_dict())
        a = actual.gram(X, Y, return_tensor=True)
        b = reference.gram(X, Y, return_tensor=True)
        compare_tensors(a, b, tuple(actual.parameters()), tuple(reference.parameters()))
    else:
        assert actual.parameters() == []
    gram = actual.gram(X)
    np.testing.assert_allclose(gram, reference.gram(X), atol=5e-6, rtol=5e-6)
    np.testing.assert_allclose(actual.gram(X, Y), reference.gram(X, Y), atol=5e-6, rtol=5e-6)
    np.testing.assert_allclose(gram, gram.T, atol=2e-6)
    np.testing.assert_allclose(np.diag(gram), 1, atol=2e-6)
    assert np.linalg.eigvalsh(gram).min() > -5e-6


@pytest.mark.parametrize('kind', ['fidelity', 'projected'])
@pytest.mark.parametrize('layout', ['encoding', 'angle', 'iqp'])
def test_kernel_minimal_optional_configuration_and_drawing(kind, layout):
    cfg = dict(components('angle', layout), kernel_type=kind)
    if layout != 'encoding':
        # An ansatz name alone is allowed: its defaults must be executable.
        cfg.pop('ansatz_config')
    kernel = create_kernel(cfg, 2)
    assert np.isfinite(kernel.gram(X)).all()
    tex = draw_kernel(kernel, x=X[0], y=X[1], output='tikz')
    assert 'quantikz' in tex
    figure, _ = draw_kernel(kernel, x=X[0], y=X[1])
    import matplotlib.pyplot as plt
    plt.close(figure)


@pytest.mark.parametrize('layout', ['encoding', 'angle', 'iqp'])
@pytest.mark.parametrize('embedding', ['angle', 'scaled', 'words', 'neural'])
def test_fidelity_vn_importance_estimates_values_and_gradients(layout, embedding):
    cfg = dict(components(embedding, layout), kernel_type='fidelity')
    if layout == 'encoding' and embedding in {'scaled', 'neural'}:
        cfg['input_scaling_params'] = np.linspace(.7, 1.1, 4 if embedding == 'neural' else 2)
    torch.manual_seed(51)
    actual = create_kernel({**deepcopy(cfg), 'device': 'IQPSimulator', 'diff_method': 'backprop',
        'device_kwargs': {'backend': 'van_den_nest', 'vn_observable_sampling': 'importance',
                          'vn_samples': 32768, 'vn_sampling': 'fixed', 'seed': 51}}, 2)
    reference = create_kernel({**deepcopy(cfg), **execution('statevector')}, 2)
    if actual.trainable:
        reference.qlayer.load_state_dict(actual.qlayer.state_dict())
        a, b = actual.gram(X[:1], Y[:1], return_tensor=True), reference.gram(X[:1], Y[:1], return_tensor=True)
        # This is a sampled estimator. Tolerances are statistical, not exact.
        compare_tensors(a, b, tuple(actual.parameters()), tuple(reference.parameters()), atol=.04)
    else:
        np.testing.assert_allclose(actual.gram(X[:1], Y[:1]), reference.gram(X[:1], Y[:1]), atol=.025, rtol=0)
