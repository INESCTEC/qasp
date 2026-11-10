"""Shared, explicit simulator choices for the configuration regression matrix."""
import torch

BACKENDS = ['statevector', 'lightning', 'mixed', 'mps', 'tn', 'iqp_tn', 'iqp_vn', 'iqp_auto', 'pp']


def execution(backend):
    if backend == 'statevector':
        return dict(device='default.qubit', diff_method='backprop')
    if backend == 'lightning':
        return dict(device='lightning.qubit', diff_method='adjoint')
    if backend == 'mixed':
        return dict(device='default.mixed', diff_method='backprop')
    if backend in {'mps', 'tn'}:
        import pytest
        pytest.importorskip('quimb', reason='PennyLane tensor tests require optional quimb.')
        pytest.importorskip('cotengra', reason='PennyLane tensor tests require optional cotengra.')
        return dict(device='default.tensor', diff_method='parameter-shift', device_kwargs={'method': backend})
    if backend.startswith('iqp_'):
        return dict(device='IQPSimulator', diff_method='backprop', device_kwargs={
            'backend': {'iqp_tn': 'tn', 'iqp_vn': 'van_den_nest', 'iqp_auto': 'auto'}[backend],
            'vn_sampling': 'enumerate', 'seed': 31})
    return dict(device='PauliPropagationSimulator', diff_method='backprop', device_kwargs={'k': None, 'prune': False})


def components(embedding='angle', layout='angle'):
    neural = embedding == 'neural'
    axis = 'Z' if layout == 'iqp' else 'X'
    cfg = {
        'feature_map': 'NeuralEmbedding' if neural else 'AngleEmbedding',
        'feature_map_config': {'features': 2, 'rotations': axis},
    }
    if embedding == 'scaled':
        cfg['feature_map_config']['input_scaling'] = True
    if embedding == 'words':
        cfg['feature_map_config']['rotations'] = [f'{axis} 0', f'{axis*2} 0 1']
    if layout != 'encoding':
        cfg['ansatz'] = 'IQPAnsatz' if layout == 'iqp' else 'AngleAnsatz'
        cfg['ansatz_config'] = {'n_layers': 1}
        if layout == 'angle':
            cfg['ansatz_config'].update(rotations=['X 0', 'X 1', 'XX 0 1'], include_imprimitive=False)
    return cfg


def compare_tensors(a, b, params_a=(), params_b=(), atol=5e-6):
    torch.testing.assert_close(a, b, atol=atol, rtol=atol)
    params_a = tuple(p for p in params_a if p.numel())
    params_b = tuple(p for p in params_b if p.numel())
    if params_a:
        weight = torch.linspace(.2, 1.1, a.numel(), dtype=a.dtype).reshape(a.shape)
        ga = torch.autograd.grad((a*weight).sum(), params_a, allow_unused=True)
        gb = torch.autograd.grad((b*weight).sum(), params_b, allow_unused=True)
        for x, y in zip(ga, gb):
            assert (x is None) == (y is None)
            if x is not None:
                assert torch.isfinite(x).all()
                torch.testing.assert_close(x, y, atol=atol, rtol=atol)
