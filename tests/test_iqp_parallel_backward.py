"""Chunk VJP correctness, higher derivatives, retained graphs and concurrency."""
from dataclasses import replace
import threading
import numpy as np
import pennylane as qml
import pytest
import torch
from qasp import IQPSimulator, IQPOptions
from qasp.simulators._van_den_nest import _reduce_chunks


def circuit(t, c, w, x):
    encoded = w @ x
    qml.RX(torch.sin(t[0]) + encoded[0], 0)
    qml.IsingXX(t[1] * encoded[1], [0, 1])
    qml.PauliRot(t[2] + encoded[0], 'XXX', [0, 1, 2])
    return qml.expval(qml.dot(c, [qml.Z(0), qml.Z(1) @ qml.Z(2), qml.I(0)])), qml.expval(qml.Z(2))


@pytest.mark.parametrize('checkpoint', [True, False])
@pytest.mark.parametrize('sampling', ['all_terms', 'importance'])
def test_nonlinear_loss_shared_encoding_mixed_and_third_derivatives(checkpoint, sampling):
    args=(torch.tensor([.3,.6,-.2],dtype=torch.float64,requires_grad=True),
          torch.tensor([.7,-.3,0.],dtype=torch.float64,requires_grad=True),
          torch.tensor([[.5,.8],[-.4,.2]],dtype=torch.float64,requires_grad=True),
          torch.tensor([.6,-.3],dtype=torch.float64,requires_grad=True))
    results=[]
    for parallel in (False,True):
        dev=IQPSimulator(3,backend='van_den_nest',vn_workers=4,vn_samples=67,
            vn_batch_size=16,vn_sampling='fixed',vn_observable_sampling=sampling,
            vn_term_uniform_mix=1.,vn_checkpoint=checkpoint,vn_parallel_backward=parallel,seed=9)
        node=qml.QNode(circuit,dev,interface='torch',diff_method='backprop')
        value=torch.stack(node(*args))
        loss=torch.logsumexp(value.square(),0)
        first=torch.autograd.grad(loss,args,create_graph=True)
        second=torch.autograd.grad(sum(g.square().sum() for g in first),args,create_graph=True)
        third=torch.autograd.grad(sum(g.sum() for g in second),args)
        results.append((value,first,second,third))
    torch.testing.assert_close(*results,atol=3e-12,rtol=3e-12)


@pytest.mark.parametrize('checkpoint', [True, False])
@pytest.mark.parametrize('importance', [True, False])
def test_gradcheck_and_gradgradcheck(checkpoint, importance):
    def f(t,c):
        qml.RX(t[0],0);qml.IsingXX(t[1],[0,1])
        return qml.expval(c[0]*qml.Z(0)+c[1]*qml.Z(1))
    d=IQPSimulator(2,backend='van_den_nest',vn_workers=2,vn_samples=8,vn_batch_size=2,
        vn_sampling='fixed' if importance else 'enumerate',
        vn_observable_sampling='importance' if importance else 'all_terms',
        vn_term_uniform_mix=1.,vn_checkpoint=checkpoint,seed=3)
    node=qml.QNode(f,d,interface='torch',diff_method='backprop')
    args=(torch.tensor([.5,.4],dtype=torch.float64,requires_grad=True),
          torch.tensor([.8,-.2],dtype=torch.float64,requires_grad=True))
    # Uniform proposals keep coefficient finite differences on the same samples.
    assert torch.autograd.gradcheck(node,args,fast_mode=True)
    assert torch.autograd.gradgradcheck(node,args,fast_mode=True)


@pytest.mark.parametrize('checkpoint', [True, False])
def test_repeated_backward_accumulation_and_later_forward(checkpoint):
    options=dict(backend='van_den_nest',vn_workers=2,vn_samples=65,vn_batch_size=16,
                 vn_sampling='fixed',vn_checkpoint=checkpoint,seed=17)
    d=IQPSimulator(2,**options)
    def f(t):
        qml.RX(t[0],0);qml.IsingXX(t[1],[0,1]);return qml.expval(qml.Z(0))
    node=qml.QNode(f,d,interface='torch')
    t=torch.tensor([.8,.9],dtype=torch.float64,requires_grad=True)
    value=node(t)
    expected=torch.autograd.grad(value,t,retain_graph=True)[0]
    d.resample(6);node(t)
    torch.testing.assert_close(torch.autograd.grad(value,t,retain_graph=True)[0],expected)
    value.backward(retain_graph=True);value.backward()
    torch.testing.assert_close(t.grad,2*expected)


def test_backward_chunks_actually_run_concurrently():
    barrier=threading.Barrier(2)
    identifiers=set()
    lock=threading.Lock()
    class Probe(torch.autograd.Function):
        @staticmethod
        def forward(ctx,x):
            ctx.save_for_backward(x);return x.square().sum()
        @staticmethod
        def backward(ctx,g):
            with lock: identifiers.add(threading.get_ident())
            barrier.wait(timeout=10)
            return 2*ctx.saved_tensors[0]*g
    def kernel(a,c,start,count,seed):
        v=(count*Probe.apply(a)*c).reshape(1)
        return v,v.detach().square()
    x=torch.tensor([.2,.7],requires_grad=True)
    options=IQPOptions(vn_samples=4,vn_batch_size=2,vn_workers=2)
    y,_=_reduce_chunks(x,torch.tensor(1.),kernel,options,0,4,1)
    gradient=torch.autograd.grad(y.sum(),x)[0]
    torch.testing.assert_close(gradient,8*x)
    assert len(identifiers)==2


@pytest.mark.parametrize('checkpoint',[True,False])
def test_unused_angle_and_zero_coefficient_gradients(checkpoint):
    x=torch.tensor([.4,.3],dtype=torch.float64,requires_grad=True)
    c=torch.tensor([0.,.2],dtype=torch.float64,requires_grad=True)
    def kernel(a,c,start,count,seed):
        # The first angle is absent; the zero coefficient has nonzero derivative.
        value=count*(c[0]*torch.cos(a[1])+c[1])
        return value.reshape(1),value.detach().square().reshape(1)
    options=IQPOptions(vn_samples=4,vn_batch_size=2,vn_workers=2,vn_checkpoint=checkpoint)
    total,_=_reduce_chunks(x,c,kernel,options,0,4,1)
    g=torch.autograd.grad(total.sum(),(x,c),create_graph=True)
    torch.testing.assert_close(g[0],torch.zeros_like(x))
    torch.testing.assert_close(g[1],4*torch.stack((torch.cos(x[1]),x.new_tensor(1.))))
    assert torch.isfinite(torch.autograd.grad(sum(v.sum() for v in g),x)[0]).all()


def test_parallel_backward_option_validation():
    with pytest.raises(ValueError): IQPOptions(vn_parallel_backward='yes')
