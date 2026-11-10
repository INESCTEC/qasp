"""Optimization objectives/gradients on CPU simulation families."""
import numpy as np
import pytest
import torch
from qasp import QAOA, WarmStartQAOA, LRQAOA, PCE, QUBO
from qasp.simulators import UnsupportedIQP
from configuration_helpers import BACKENDS, execution, compare_tensors


@pytest.fixture(autouse=True)
def double_precision():
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)

@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('algorithm', ['qaoa', 'warm', 'ramp', 'pce'])
def test_objectives_gradients_and_optimization(backend, algorithm):
    problem = QUBO.maxcut(3, [(0, 1), (1, 2), (0, 2)])
    kwargs = dict(backend='torch', seed=31)
    cls = {'qaoa': QAOA, 'warm': WarmStartQAOA, 'ramp': LRQAOA, 'pce': PCE}[algorithm]
    if algorithm == 'warm':kwargs['warm_start'] = [.2, .5, .8]
    if algorithm == 'pce':
        kwargs.update(n_qubits=3, paulis='Z', circuit={'ansatz': 'AngleAnsatz',
            'ansatz_config': {'n_layers': 1, 'rotations': ['X 0', 'X 1', 'X 2', 'XX 0 1'], 'include_imprimitive': False}})
    actual = cls(problem, **kwargs, **execution(backend))
    reference = cls(problem, **kwargs, **execution('statevector'))
    theta = torch.tensor(actual.initial_parameters(), dtype=torch.float64, requires_grad=True)
    fa = actual.loss_tensor if algorithm == 'pce' else actual.expectation_tensor
    fb = reference.loss_tensor if algorithm == 'pce' else reference.expectation_tensor
    if backend == 'iqp_vn' and algorithm != 'pce':
        with pytest.raises(UnsupportedIQP, match='X-only|Hadamard|frame|recognize'):
            fa(theta)
        return
    compare_tensors(fa(theta), fb(theta), (theta,), (theta,), atol=2e-7)
    if backend in {'mps', 'tn'} and algorithm != 'pce':
        # The objective is supported; these PennyLane devices cannot fetch
        # full output probabilities for QAOA's solution-decoding stage.
        optimizer = torch.optim.Adam([theta], lr=.03)
        for _ in range(2):
            optimizer.zero_grad(); fa(theta).backward(); optimizer.step()
        assert torch.isfinite(theta).all()
        with pytest.raises(NotImplementedError, match='probs'):
            if algorithm == 'ramp':actual.solve()
            else:actual.solve(optimizer='Adam', optimizer_options={'max_steps': 1})
        return
    if algorithm == 'pce':
        result = actual.solve(optimizer='Adam', optimizer_options={'max_steps': 2, 'lr': .03})
    elif algorithm == 'ramp':
        result = actual.solve()
    else:
        result = actual.solve(optimizer='Adam', optimizer_options={'max_steps': 2, 'lr': .03})
    assert np.isfinite(result.objective)
