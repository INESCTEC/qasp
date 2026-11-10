"""Configuration families across module factories and simulator backends."""
from copy import deepcopy
import numpy as np
import pytest
import torch
from qasp import AngleAnsatz, IQPAnsatz, AngleEmbedding, NeuralEmbedding, CircuitBuilder
from qasp import create_classifier, create_policy, create_qfunction, create_qcbm
from qasp.qkernels.quantum_kernels import EncodingAnsatz
from configuration_helpers import BACKENDS, execution, components, compare_tensors


@pytest.fixture(autouse=True)
def double_precision():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)

def bare_builder(kind='angle', embedding=None):
    fm = None
    if embedding:
        fm = (NeuralEmbedding if embedding == 'neural' else AngleEmbedding)(2, features=2, rotations='Z' if kind == 'iqp' else 'X')
    b = CircuitBuilder(2)
    if kind == 'encoding':
        ans = EncodingAnsatz(2, feature_map=fm)
    elif kind == 'iqp':
        ans = IQPAnsatz(2, n_layers=1, feature_map=fm)
    else:
        ans = AngleAnsatz(2, n_layers=1, rotations=['X 0', 'X 1', 'XX 0 1'], feature_map=fm, include_imprimitive=False)
    b.set_ansatz(ans)
    return b


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('kind', ['angle', 'iqp'])
@pytest.mark.parametrize('measurement', ['expectation', 'probs'])
def test_ansatz_only_direct_node_and_torch_layer(backend, kind, measurement):
    b = bare_builder(kind); reference = bare_builder(kind)
    for builder in (b, reference):
        builder.set_measurement(measurement, observables=['Z 0', 'ZZ 0 1'] if measurement == 'expectation' else None)
        builder.build()
    if measurement == 'probs' and backend == 'iqp_vn':
        from qasp.simulators import UnsupportedIQP
        with pytest.raises(UnsupportedIQP, match='probabilities'):
            b.compile(**execution(backend))(torch.tensor([.2, .3, .4], requires_grad=True))
        return
    if measurement == 'probs' and backend in {'mps', 'tn'}:
        with pytest.raises(NotImplementedError, match='probs'):
            b.compile(**execution(backend))(torch.tensor([.2, .3, .4], requires_grad=True))
        return
    opts = execution(backend)
    if backend == 'lightning' and measurement == 'probs':opts['diff_method'] = 'parameter-shift'
    node = b.compile(**opts); ref = reference.compile(**execution('statevector'))
    count = b.ansatz.get_param_shapes()[-1][0]
    theta = torch.linspace(.2, .7, count, dtype=torch.float64, requires_grad=True)
    a, r = node(params=theta), ref(params=theta)
    if isinstance(a, tuple):a, r = torch.stack(a), torch.stack(r)
    compare_tensors(a, r, (theta,), (theta,), atol=2e-8)
    layer = b.create_torchlayer(params=theta.detach()).double()
    # No feature map: inputs are placeholders; all samples share the circuit.
    one = layer(torch.zeros(2, dtype=torch.float64))
    batch = layer(torch.zeros((3, 2), dtype=torch.float64))
    expected = one[:, None, :].expand(-1, 3, -1) if measurement == 'expectation' else one[None, :].expand(3, -1)
    torch.testing.assert_close(batch, expected)
    batch.sum().backward()
    assert torch.isfinite(layer.params.grad).all()


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('layout', ['angle', 'iqp', 'bare', 'builder_encoding'])
@pytest.mark.parametrize('embedding', ['angle', 'neural'])
@pytest.mark.parametrize('module', ['classifier', 'softmax', 'born', 'qfunction'])
def test_module_factory_values_gradients_and_batches(backend, layout, embedding, module):
    # Every chosen readout is an expectation or compact decoder compatible
    # with the IQP engines; explicit VN cannot execute full probabilities.
    if layout in {'bare', 'builder_encoding'}:
        cfg = {'circuit_builder': bare_builder('angle' if layout == 'bare' else 'encoding',
                                             embedding if layout == 'builder_encoding' else None)}
    else:
        cfg = components(embedding, layout)
    def make(which):
        options = {**deepcopy(cfg), **execution(which)}
        if module == 'classifier':
            return create_classifier({**options, 'readout': 'expectation', 'observables': ['Z 0', 'Z 1']}, 2, 2)
        if module == 'qfunction':
            return create_qfunction({**options, 'n_actions': 2, 'observables': ['Z 0', 'Z 1'], 'input_activation': None}, n_qubits=2)
        if module == 'softmax':
            return create_policy('PQCSoftmax', {**options, 'n_actions': 2, 'observables': ['Z 0', 'Z 1'], 'fc_layer': None}, 2)
        if which == 'lightning':options['diff_method'] = 'parameter-shift'
        return create_policy('PQCBorn', {**options, 'n_actions': 2, 'policy': 'parity-like',
                            'parity_cnot_decomposition': False, 'training_method': 'expectation'}, 2)
    torch.manual_seed(31)
    actual, reference = make(backend).double(), make('statevector').double()
    reference.load_state_dict(actual.state_dict())
    x = torch.tensor([[.2, -.3], [-.4, .7]], dtype=torch.float64)
    a, b = actual(x), reference(x)
    compare_tensors(a, b, tuple(actual.parameters()), tuple(reference.parameters()), atol=1e-7)
    assert a.shape[0] == 2
    if layout == 'bare':torch.testing.assert_close(a[0], a[1])
    if backend == 'statevector':
        from qasp import draw_classifier, draw_policy, draw_qfunction
        import matplotlib.pyplot as plt
        draw = (lambda **kw: draw_classifier(actual, inputs=x[0], **kw)) if module == 'classifier' else (
            (lambda **kw: draw_qfunction(actual, inputs=x[0], **kw)) if module == 'qfunction' else
            (lambda **kw: draw_policy(model=actual, inputs=x[0], **kw)))
        assert 'quantikz' in draw(output='tikz', tikz_options={'symbolic_parameters': True})
        figure, _ = draw()
        plt.close(figure)


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('layout', ['angle', 'iqp', 'bare'])
@pytest.mark.parametrize('loss', ['nll', 'mmd'])
def test_born_machine_loss_gradients_and_update(backend, layout, loss):
    if backend == 'iqp_vn' and loss == 'nll':
        from qasp.simulators import UnsupportedIQP
        with pytest.raises(UnsupportedIQP, match='probabilities'):
            create_qcbm({**components('angle', 'angle'), 'inputs': [.2, .3], 'loss': 'nll', **execution(backend)}, 2, [[0, 0]]).loss()
        return
    if backend in {'mps', 'tn'} and loss == 'nll':
        with pytest.raises(NotImplementedError, match='probs'):
            create_qcbm({**components('angle', 'angle'), 'inputs': [.2, .3], 'loss': 'nll', **execution(backend)}, 2, [[0, 0]]).loss()
        return
    cfg = components('scaled', layout if layout != 'bare' else 'angle')
    if layout == 'bare':cfg.pop('feature_map');cfg.pop('feature_map_config')
    else:cfg['inputs'] = np.array([.3, .7])
    cfg.update(loss=loss, seed=31)
    if loss == 'mmd':cfg['mmd_config'] = {'mode': 'exact'}
    options = execution(backend)
    if backend == 'lightning' and loss == 'nll':options['diff_method'] = 'parameter-shift'
    data = [[0, 0], [1, 1], [0, 1]]
    actual = create_qcbm({**deepcopy(cfg), **options}, 2, data)
    reference = create_qcbm({**deepcopy(cfg), **execution('statevector')}, 2, data)
    compare_tensors(actual.loss(), reference.loss(), tuple(actual.parameters()), tuple(reference.parameters()), atol=2e-7)
    optimizer = torch.optim.Adam(actual.parameters(), lr=.01)
    before = actual.params.detach().clone()
    optimizer.zero_grad(); actual.loss().backward(); optimizer.step()
    assert torch.isfinite(actual.params).all()
    assert not torch.equal(before, actual.params)


@pytest.mark.parametrize('kind', ['PQCBorn', 'PQCSoftmax'])
@pytest.mark.parametrize('optional_configs', ['omitted', 'none'])
def test_policy_optional_configs_are_not_mutated(kind, optional_configs):
    cfg = {'ansatz': 'AngleAnsatz', 'feature_map': 'NeuralEmbedding', 'n_actions': 2,
           'feature_map_config': {'rotations': 'Y'},
           'device': 'default.qubit', 'diff_method': 'backprop'}
    if optional_configs == 'none':cfg.update(ansatz_config=None)
    if kind == 'PQCSoftmax':cfg.update(observables=['Z 0', 'Z 1'], fc_layer=None)
    original = deepcopy(cfg)
    policy = create_policy(kind, cfg, 2)
    assert torch.isfinite(policy(torch.tensor([.2, .4]))).all()
    assert cfg == original


def test_softmax_probabilities_use_head_scale_and_simulator():
    cfg = {**components('neural', 'angle'), 'n_actions': 3, 'observables': ['Z 0', 'Z 1'],
           'fc_layer': True, 'output_scaling': True, **execution('pp')}
    model = create_policy('PQCSoftmax', cfg, 2).double()
    x = torch.tensor([[.2, -.3], [.4, .6]], dtype=torch.float64, requires_grad=True)
    probabilities = model.action_probs(x)
    torch.testing.assert_close(probabilities, model(torch.tanh(x)).softmax(-1))
    assert probabilities.shape == (2, 3)
    (-probabilities[0, 0].log()).backward()
    assert torch.isfinite(x.grad).all()
    assert model.fc_layer.weight.grad is not None
    assert model.output_scaling_weight.grad is not None
