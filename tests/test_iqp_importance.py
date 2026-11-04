"""Importance estimator and deterministic single-node parallel execution."""
import numpy as np
import pennylane as qml
import pytest
import torch

from qasp import IQPSimulator, IQPOptions
from qasp.simulators._van_den_nest import _parity_blocks, _parities


def circuit(theta, c):
    qml.RX(theta[0], 'a')
    qml.IsingXX(theta[1], ['a', 'b'])
    qml.PauliRot(theta[2], 'XXX', ['a', 'b', 'c'])
    return qml.expval(qml.dot(c, [qml.Z('a'), qml.Z('b') @ qml.Z('c'), qml.I('a')])), qml.expval(qml.Projector([1, 0], wires=['c', 'a']))


def evaluate(device, theta, c, hessian=False):
    node = qml.QNode(circuit, device, interface='torch', diff_method='backprop')
    value = torch.stack(node(theta, c))
    grad = torch.autograd.grad(value.sum(), (theta, c), create_graph=hessian)
    second = torch.autograd.grad(grad[0].sum(), (theta, c)) if hessian else ()
    return value, grad, second


def inputs():
    return (torch.tensor([.5, -.7, .8], dtype=torch.float64, requires_grad=True),
            torch.tensor([.7, -.3, 0.], dtype=torch.float64, requires_grad=True))


@pytest.mark.parametrize('checkpoint', [False, True])
def test_importance_values_gradients_and_zero_coefficient(checkpoint):
    theta, c = inputs()
    device = IQPSimulator(['a','b','c'], backend='van_den_nest', seed=91,
        vn_observable_sampling='importance', vn_sampling='fixed', vn_samples=65536,
        vn_term_uniform_mix=.3, vn_workers=2, vn_checkpoint=checkpoint)
    value, grad, _ = evaluate(device, theta, c)
    reference, refgrad, _ = evaluate(qml.device('default.qubit', wires=['a','b','c']), theta, c)
    torch.testing.assert_close(value, reference, atol=.025, rtol=0)
    for x, y in zip(grad, refgrad):
        torch.testing.assert_close(x, y, atol=.04, rtol=0)
    assert abs(grad[1][2].item()-1) < .04  # derivative of a zero identity coefficient
    assert device.last_report.observable_log2_term_counts == (np.log2(3), 2.)
    assert device.last_report.cpu_workers == 2


@pytest.mark.parametrize('checkpoint', [False, True])
def test_workers_preserve_values_gradients_hessians_and_modes(checkpoint):
    theta, c = inputs()
    before = torch.get_num_threads()
    results = []
    for workers in (1, 2, 4):
        device = IQPSimulator(['a','b','c'], backend='van_den_nest', seed=12,
            vn_observable_sampling='importance', vn_sampling='fixed', vn_samples=2051,
            vn_batch_size=256, vn_workers=workers, vn_checkpoint=checkpoint)
        results.append(evaluate(device, theta, c, hessian=True))
        node = qml.QNode(circuit, device, interface='torch')
        for mode in (torch.no_grad, torch.inference_mode):
            with mode():
                actual = torch.stack(node(theta, c))
                assert not actual.requires_grad
                torch.testing.assert_close(actual, results[-1][0], atol=0, rtol=0)
    for actual in results[1:]:
        torch.testing.assert_close(actual, results[0], atol=1e-13, rtol=1e-13)
    assert torch.get_num_threads() == before


@pytest.mark.parametrize('sampling', ['fixed', 'enumerate'])
def test_all_terms_workers_and_finite_difference(sampling):
    theta = inputs()[0]
    def f(t):
        qml.RX(t[0], 0); qml.IsingXX(t[1], [0,1]); qml.PauliRot(t[2], 'XXX', [0,1,2])
        return qml.expval(qml.Z(0)+.4*qml.Z(1)@qml.Z(2))
    values, grads = [], []
    for workers in (1, 4):
        d = IQPSimulator(3, backend='van_den_nest', seed=1, vn_sampling=sampling,
                         vn_samples=1024, vn_batch_size=3, vn_workers=workers)
        node = qml.QNode(f, d, interface='torch')
        v = node(theta); grad = torch.autograd.grad(v, theta)[0]
        values.append(v); grads.append(grad)
        delta = torch.tensor([1e-5, 0., 0.], dtype=theta.dtype)
        torch.testing.assert_close((node(theta+delta)-node(theta-delta))/2e-5, grad[0], atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(*values, atol=0, rtol=0)
    torch.testing.assert_close(*grads, atol=1e-13, rtol=1e-13)


def test_importance_checkpoint_keeps_bank_after_later_forward_and_angle_finite_difference():
    theta, c = inputs()
    d = IQPSimulator(['a','b','c'], backend='van_den_nest', seed=3, vn_sampling='fixed',
        vn_observable_sampling='importance', vn_samples=2048, vn_batch_size=256, vn_workers=4)
    node = qml.QNode(circuit, d, interface='torch')
    v = sum(node(theta, c)); expected = torch.autograd.grad(sum(node(theta, c)), theta)[0]
    delta = torch.tensor([1e-5,0,0],dtype=theta.dtype)
    fd = (sum(node(theta+delta,c))-sum(node(theta-delta,c)))/2e-5
    torch.testing.assert_close(fd, expected[0], atol=1e-8, rtol=1e-8)
    d.resample(41); node(theta,c)
    torch.testing.assert_close(torch.autograd.grad(v,theta)[0], expected, atol=0, rtol=0)


@pytest.mark.parametrize('conflict', [False, True])
def test_exact_compact_projector_products(conflict):
    theta = inputs()[0]
    def f(t):
        qml.RX(t[0],0); qml.IsingXX(t[1],[0,1]); qml.PauliRot(t[2],'XXX',[0,1,2])
        obs = qml.prod(qml.Projector([1,0], wires=[2,0]),
                       qml.Projector([int(conflict)], wires=[0]), qml.Z(2), qml.Z(1))
        return qml.expval(.4*obs + .7*qml.Projector([0],wires=[1]))
    out=[]
    for device in (IQPSimulator(3,backend='tn'),qml.device('default.qubit',wires=3)):
        value=qml.QNode(f,device,interface='torch')(theta)
        # PennyLane 0.45 returns complex dtype for this overlapping product,
        # although the commuting diagonal observable is Hermitian. Check that
        # its imaginary part vanishes before comparing the real expectation.
        if value.is_complex():
            assert not isinstance(device, IQPSimulator)
            torch.testing.assert_close(value.imag, torch.zeros_like(value.imag), atol=1e-12, rtol=0)
            value = value.real
        out.append((value,torch.autograd.grad(value,theta)[0]))
    torch.testing.assert_close(*out,atol=1e-12,rtol=1e-12)


def test_large_projector_has_no_exponential_expansion(monkeypatch):
    n=100
    d=IQPSimulator(n,seed=4,vn_observable_sampling='importance',vn_samples=2048,vn_workers=2)
    def forbidden(*a,**k): raise AssertionError('TN planning is unnecessary')
    monkeypatch.setattr(d,'_tn_plan',forbidden)
    def f(t):
        for i in range(n): qml.RX(t,i)
        return qml.expval(qml.Projector([0]*n,wires=range(n)))
    t=torch.tensor(.1,dtype=torch.float64,requires_grad=True)
    v=qml.QNode(f,d,interface='torch')(t)
    assert abs(v.item()-np.cos(.05)**(2*n))<.04
    assert torch.isfinite(torch.autograd.grad(v,t)[0])
    assert d.last_report.observable_components==(1,)
    assert d.last_report.observable_log2_term_counts==(100.,)
    assert d.last_report.tn_planning_skipped
    assert d.last_report.estimated_memory_bytes<100_000_000


def test_grouped_sparse_parity():
    rng=np.random.default_rng(4)
    bits=rng.integers(0,2,(17,300),dtype=np.uint8)
    supports=((2,), (0,1,8), tuple(range(300)), (1,7), (3,), (4,9,12))
    blocks=_parity_blocks(supports,17)
    got=_parities(bits,blocks,len(supports),np.float64)
    expected=np.stack([bits[:,s].sum(1)%2 for s in supports],axis=1)
    np.testing.assert_array_equal(got,expected)


@pytest.mark.parametrize('kwargs', [dict(vn_workers=0),dict(vn_workers=True),
    dict(vn_term_uniform_mix=True),dict(vn_term_uniform_mix=0.),dict(vn_term_uniform_mix='x'),
    dict(vn_observable_sampling='importance',vn_sampling='enumerate'),
    dict(vn_max_observable_components=0)])
def test_options(kwargs):
    with pytest.raises(ValueError): IQPOptions(**kwargs)


def test_memory_budget_accounts_for_concurrent_chunks():
    with qml.tape.QuantumTape() as tape:
        qml.RX(.3,0); qml.expval(qml.Z(0))
    reports=[IQPSimulator(2,backend='van_den_nest',vn_workers=w).plan(tape) for w in (1,4)]
    assert reports[1].estimated_memory_bytes == 4*reports[0].estimated_memory_bytes
    assert reports[1].estimated_flops == reports[0].estimated_flops
