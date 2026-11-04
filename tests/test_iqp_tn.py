"""Numerical and resource-contract tests for exact IQP tensor-network execution."""
import numpy as np  # import before Torch for this numerical installation
import pennylane as qml
import pytest
import torch

from qasp.simulators import IQPTNSimulator, IQPTNOptions, DecoderSpec, action_probs, UnsupportedIQP
from qasp.devices import create_device
from qasp.circuits import CircuitBuilder, IQPAnsatz


def circuit(theta, inputs, edges, n=6, readout=None):
    for w in range(n):
        qml.Hadamard(w)
        qml.RZ(theta[w] + inputs[w % len(inputs)] * theta[-1], w)
    for j, support in enumerate(edges):
        qml.MultiRZ(theta[n+j % (len(theta)-n)], wires=support)
    for w in range(n):
        qml.Hadamard(w)
    return readout() if readout else qml.probs(wires=range(n))


GRAPHS = {
    'chain': [(i,i+1) for i in range(5)],
    'star': [(0,i) for i in range(1,6)],
    'cycle': [(i,(i+1)%6) for i in range(6)],
    'cubic': [(i,j) for i in range(3) for j in range(3,6)],
    'hyper': [(0,1,2),(2,3,4),(0,4,5),(1,2,4,5)],
}
DECODERS = [DecoderSpec(), DecoderSpec('majority-modulo',3,((0,2),(1,3,4))),
            DecoderSpec('majority-modulo',4,((0,1,2),(2,3,4))),
            DecoderSpec('parity-like',3), DecoderSpec('contiguous-like',3)]


@pytest.mark.parametrize('seed',[2,17,41])
@pytest.mark.parametrize('graph',GRAPHS)
@pytest.mark.parametrize('decoder',DECODERS)
def test_values_parameter_input_gradients(seed, graph, decoder):
    gen=torch.Generator().manual_seed(seed)
    theta=torch.randn(11,generator=gen,dtype=torch.float64,requires_grad=True)
    x=torch.randn(3,generator=gen,dtype=torch.float64,requires_grad=True)
    wires=[4,1,5,0,2]  # partial readout, explicit noncanonical ordering
    def tn(t,x): return circuit(t,x,GRAPHS[graph],readout=lambda:action_probs(wires,decoder))
    def sv(t,x): return circuit(t,x,GRAPHS[graph],readout=lambda:qml.probs(wires=wires))
    node=qml.QNode(tn,IQPTNSimulator(6),interface='torch',diff_method='backprop')
    ref=qml.QNode(sv,qml.device('default.qubit',wires=6),interface='torch',diff_method='backprop')
    actual=node(theta,x); expected=decoder.aggregate(ref(theta,x))
    torch.testing.assert_close(actual,expected,atol=2e-11,rtol=2e-11)
    weights=torch.arange(decoder.n_actions,dtype=torch.float64)
    ga=torch.autograd.grad(actual@weights,(theta,x));ge=torch.autograd.grad(expected@weights,(theta,x))
    for a,e in zip(ga,ge): torch.testing.assert_close(a,e,atol=3e-11,rtol=3e-11)


@pytest.mark.parametrize('angle',[0.,np.pi/2,np.pi,-np.pi/2])
def test_zero_special_angles_shared_parameters_and_hessian(angle):
    def f(t):
        for w in range(4): qml.Hadamard(w);qml.RZ(t,w)
        qml.MultiRZ(t,[0,1,2,3]);qml.IsingZZ(t,[1,2])
        for w in range(4): qml.Hadamard(w)
        return qml.expval(qml.Y(0)@qml.Y(2)),qml.var(qml.X(1)+.3*qml.Z(3))
    t=torch.tensor(angle,dtype=torch.float64,requires_grad=True)
    nodes=[qml.QNode(f,d,interface='torch',diff_method='backprop') for d in [IQPTNSimulator(4),qml.device('default.qubit',wires=4)]]
    result=[]
    for node in nodes:
        y=sum(node(t));g=torch.autograd.grad(y,t,create_graph=True)[0];h=torch.autograd.grad(g,t)[0];result.append(torch.stack([y,g,h]))
    torch.testing.assert_close(*result,atol=2e-11,rtol=2e-11)


def test_local_dressing_cz_and_wire_labels():
    def f(t):
        qml.RY(t,'b');qml.RX(0.4,'a');qml.CZ(['b','a']);qml.RZ(t,'b')
        qml.ControlledPhaseShift(t,['a','b']);qml.Rot(t,t/2,.3,'a');qml.Hadamard('b')
        return qml.probs(wires=['a','b']),qml.expval(qml.Y('a'))
    x=torch.tensor(.3,dtype=torch.float64,requires_grad=True)
    results=[qml.QNode(f,d,interface='torch')(x) for d in [IQPTNSimulator(['b','a']),qml.device('default.qubit',wires=['b','a'])]]
    for a,b in zip(*results): torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)


def test_auto_forced_hard_limit_and_path_cache():
    def f(x): return circuit(x,torch.tensor([.1,.3],dtype=torch.float64),GRAPHS['chain'],readout=lambda:action_probs(range(6)))
    theta=torch.randn(11,dtype=torch.float64,requires_grad=True)
    auto=create_device('auto',wires=6,device_kwargs={'auto_max_flops':0})
    forced=create_device('IQPTNSimulator',wires=6,device_kwargs={'auto_max_flops':0})
    na=qml.QNode(f,auto,interface='torch');nf=qml.QNode(f,forced,interface='torch')
    a=na(theta)
    with pytest.warns(RuntimeWarning,match='bypassed'): b=nf(theta)
    torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
    assert auto.last_report.selected=='default.qubit'
    assert forced.last_report.selected=='IQPTNSimulator'
    with pytest.raises(MemoryError,match='hard limit'):
        qml.QNode(f,IQPTNSimulator(6,hard_max_flops=0),interface='torch')(theta)
    from qasp.simulators.iqp_tn import _path
    hits=_path.cache_info().hits
    nf.device.options=IQPTNOptions()
    changed=nf(theta+.23)
    assert _path.cache_info().hits>hits
    assert not torch.allclose(changed,b)


@pytest.mark.parametrize('unsupported',['CNOT','repeated','state','channel'])
def test_unsupported_and_auto_fallback(unsupported):
    def f(t):
        qml.Hadamard(0);qml.Hadamard(1);qml.IsingZZ(t,[0,1])
        if unsupported=='CNOT': qml.CNOT([0,1])
        if unsupported=='repeated': qml.Hadamard(0);qml.IsingZZ(t,[0,1])
        if unsupported=='channel': qml.DepolarizingChannel(.1,0)
        return qml.state() if unsupported=='state' else qml.probs(wires=[0,1])
    t=torch.tensor(.4,dtype=torch.float64,requires_grad=True)
    with pytest.raises(UnsupportedIQP): qml.QNode(f,IQPTNSimulator(2),interface='torch')(t)
    if unsupported!='channel':
        auto=IQPTNSimulator(2,mode='auto');out=qml.QNode(f,auto,interface='torch')(t)
        ref=qml.QNode(f,qml.device('default.qubit',wires=2),interface='torch')(t)
        torch.testing.assert_close(out,ref)
        assert not auto.last_report.eligible


def test_batch_optimizer_topology_and_neural_encoding():
    weights=torch.nn.Parameter(torch.randn(3,2,dtype=torch.float64))
    params=torch.nn.Parameter(torch.randn(3,dtype=torch.float64))
    x=torch.randn(4,2,dtype=torch.float64,requires_grad=True)
    def f(encoded,t,edge):
        for w in range(3): qml.Hadamard(w);qml.RZ(encoded[...,w]+t[w],w)
        qml.MultiRZ(t[0],edge)
        for w in range(3): qml.Hadamard(w)
        return qml.probs(wires=[2,0])
    tn=qml.QNode(f,IQPTNSimulator(3,mode='auto'),interface='torch');sv=qml.QNode(f,qml.device('default.qubit',wires=3),interface='torch')
    for edge in ([0,1],[0,1,2],[1,2]):
        a=tn(x@weights.T,params,edge);b=sv(x@weights.T,params,edge)
        torch.testing.assert_close(a,b,atol=1e-11,rtol=1e-11)
        gs=[torch.autograd.grad(o[:,0].sum(),(weights,params,x)) for o in (a,b)]
        for u,v in zip(*gs):torch.testing.assert_close(u,v,atol=1e-11,rtol=1e-11)
    assert tn.device.last_report.batch_size==4
    optimizer=torch.optim.SGD([weights,params],lr=.1);before=params.detach().clone()
    loss=-tn(x@weights.T,params,[0,1])[:,0].mean();loss.backward();optimizer.step()
    assert not torch.equal(params,before)


@pytest.mark.parametrize('kind',['probs','samples','counts','action'])
def test_shots(kind):
    def f():
        qml.Hadamard(0);qml.Hadamard(1);qml.RZ(.9,0);qml.IsingZZ(.6,[0,1]);qml.Hadamard(0);qml.Hadamard(1)
        if kind=='samples': return qml.sample(wires=[1,0])
        if kind=='counts': return qml.counts(wires=[1,0])
        if kind=='action': return action_probs([1,0])
        return qml.probs(wires=[1,0])
    d=IQPTNSimulator(2,shots=800,seed=12)
    out=qml.QNode(f,d,diff_method=None)()
    if kind=='counts':
        assert sum(out.values())==800
        observed=np.array([out.get(f'{i:02b}',0)/800 for i in range(4)])
    elif kind=='samples':
        assert out.shape==(800,2)
        observed=np.bincount(out@np.array([2,1]),minlength=4)/800
    else: observed=np.asarray(out)
    def ref():
        qml.Hadamard(0);qml.Hadamard(1);qml.RZ(.9,0);qml.IsingZZ(.6,[0,1]);qml.Hadamard(0);qml.Hadamard(1)
        return qml.probs(wires=[1,0])
    expected=qml.QNode(ref,qml.device('default.qubit',wires=2))()
    if kind=='action':expected=DecoderSpec().aggregate(torch.tensor(expected)).numpy()
    np.testing.assert_allclose(observed,expected,atol=.055)


def test_large_public_born_policy_has_no_dense_decoder(monkeypatch):
    from qasp.rl.policies import PQCBornVariance,_PolicyDecoder
    from qasp.feature_maps import AngleEmbedding
    def forbidden(*a,**k):raise AssertionError('Exponential decoder table or fallback was used')
    monkeypatch.setattr(_PolicyDecoder,'_build_action_index',forbidden)
    monkeypatch.setattr(IQPTNSimulator,'_fallback',forbidden)
    model=PQCBornVariance(n_qubits=45,n_actions=2,policy='majority-modulo',device='auto',
        ansatz=IQPAnsatz,ansatz_config={'n_qubits':45,'n_layers':1,'imprimitive_pattern':'chain'},
        feature_map=AngleEmbedding,feature_map_config={'n_qubits':45,'features':1,'rotations':'Z'},diff_method='backprop')
    p=model.action_probs(torch.tensor([.23]));assert p.shape[-1]==2
    torch.testing.assert_close(p.sum(-1),torch.ones_like(p.sum(-1)),atol=2e-6,rtol=2e-6)
    (-p[...,0].log()).sum().backward()
    assert model.policy_decoder.action_index.numel()==0
    assert model.circuit.device.last_report.output_elements==2
    assert model.circuit.device.last_report.selected=='IQPTNSimulator'


def test_decoder_direct_samples_wide():
    for n in [4,5,70]:
        bits=torch.randint(2,(6,n))
        for decoder in [DecoderSpec(),DecoderSpec('contiguous',3),DecoderSpec('parity',3)]:
            out=decoder.decode(bits)
            assert out.shape==(6,) and out.min()>=0 and out.max()<decoder.n_actions
    for n in [4,5]:
        bits=((torch.arange(2**n)[:,None]>>torch.arange(n-1,-1,-1))&1)
        np.testing.assert_array_equal(DecoderSpec().decode(bits),bits.sum(-1)>n//2)


def test_explicit_other_device_and_mpi_preserved():
    explicit=qml.device('default.qubit',wires=2)
    assert create_device(explicit,wires=2) is explicit
    assert type(create_device('default.qubit',wires=2)) is type(explicit)
    assert isinstance(create_device(IQPTNSimulator,wires=2),IQPTNSimulator)
    with pytest.raises(ValueError,match='mpi'):create_device('IQPTNSimulator',wires=2,mpi=True)

@pytest.mark.parametrize('gate',[qml.RZ,lambda t,w:qml.PauliRot(t,'Z',wires=w),qml.MultiRZ])
def test_single_diagonal_rotations_keep_local_order(gate):
    def f(t):
        gate(t,0);qml.Hadamard(0);gate(t,0);qml.IsingZZ(t,[0,1]);qml.Hadamard(0);gate(t,0)
        return qml.expval(qml.X(0))
    t=torch.tensor(.6,dtype=torch.float64,requires_grad=True)
    values=[qml.QNode(f,d,interface='torch')(t) for d in [IQPTNSimulator(2),qml.device('default.qubit',wires=2)]]
    torch.testing.assert_close(*values,atol=1e-12,rtol=1e-12)
    grads=[torch.autograd.grad(v,t)[0] for v in values];torch.testing.assert_close(*grads)


def test_auto_adjoint_and_autograd_fallback():
    def f(t):
        qml.Hadamard(0);qml.IsingZZ(t,[0,1]);qml.Hadamard(0)
        return qml.expval(qml.Z(0))
    for interface,diff in [('torch','adjoint'),('autograd','backprop')]:
        d=IQPTNSimulator(2,mode='auto')
        node=qml.QNode(f,d,interface=interface,diff_method=diff)
        if interface=='torch':
            t=torch.tensor(.7,dtype=torch.float64,requires_grad=True);g=torch.autograd.grad(node(t),t)[0].item()
        else:
            t=qml.numpy.array(.7,requires_grad=True);g=qml.grad(node)(t)
        np.testing.assert_allclose(g,-np.sin(.7),atol=1e-12)
        assert d.last_report.selected=='default.qubit'


def test_dense_output_budget_is_checked_without_allocating():
    tape=qml.tape.QuantumScript([qml.Hadamard(0)],[qml.probs(wires=range(50))])
    dev=IQPTNSimulator(50,mode='auto')
    report=dev.plan(tape)
    assert report.selected=='default.qubit' and report.output_elements==2**50
    with pytest.raises(RuntimeError,match='fallback_max_wires'):dev.execute(tape)


@pytest.mark.parametrize('precision,tolerance',[('complex64',100*np.finfo(np.float32).eps),('complex128',2e-12)])
def test_precision_and_compact_fifty_body_rotation(precision,tolerance,monkeypatch):
    original=qml.matrix
    def no_large_matrix(op,*args,**kwargs):
        if isinstance(op,qml.operation.Operator) and len(op.wires)>1:
            raise AssertionError('Compact Pauli rotations must not become dense gate matrices')
        return original(op,*args,**kwargs)
    monkeypatch.setattr(qml,'matrix',no_large_matrix)
    def f(t):
        for w in range(50):qml.Hadamard(w)
        qml.PauliRot(t,'Z'*50,wires=range(50))
        for w in range(50):qml.Hadamard(w)
        return qml.expval(qml.Z(0))
    t=torch.tensor(.63,dtype=torch.float64,requires_grad=True)
    d=IQPTNSimulator(50,precision=precision);y=qml.QNode(f,d,interface='torch')(t)
    np.testing.assert_allclose(y.detach(),np.cos(.63),atol=tolerance)
    np.testing.assert_allclose(torch.autograd.grad(y,t)[0],-np.sin(.63),atol=tolerance)
    assert d.last_report.precision==precision


@pytest.mark.parametrize('seed',[3,29,61])
def test_random_cubic_graphs(seed):
    import networkx as nx
    edges=list(nx.random_regular_graph(3,8,seed=seed).edges())
    def f(t):
        for w in range(8):qml.Hadamard(w);qml.RZ(t[w],w)
        for i,edge in enumerate(edges):qml.IsingZZ(t[i % 8],edge)
        for w in range(8):qml.Hadamard(w)
        return qml.expval(qml.X(1)@qml.Z(5)),qml.probs(wires=[7,2])
    gen=torch.Generator().manual_seed(seed);t=torch.randn(8,generator=gen,dtype=torch.float64,requires_grad=True)
    a=qml.QNode(f,IQPTNSimulator(8),interface='torch')(t)
    b=qml.QNode(f,qml.device('default.qubit',wires=8),interface='torch')(t)
    for u,v in zip(a,b):torch.testing.assert_close(u,v,atol=1e-11,rtol=1e-11)
    ga=torch.autograd.grad(a[0]+a[1][0],t)[0];gb=torch.autograd.grad(b[0]+b[1][0],t)[0]
    torch.testing.assert_close(ga,gb,atol=1e-11,rtol=1e-11)


def test_compact_decoder_validates_sample_width():
    from qasp.rl.policies import _PolicyDecoder
    d=_PolicyDecoder('contiguous-like',3,4,compact=True)
    with pytest.raises(ValueError,match='final dimension 4'):d.decode_samples(torch.zeros(2,3))


def test_action_measurement_cannot_silently_return_dense_probabilities():
    def f():
        qml.Hadamard(0)
        return action_probs([0,1])
    with pytest.raises(NotImplementedError,match='IQPTNSimulator'):
        qml.QNode(f,qml.device('default.qubit',wires=2),diff_method=None)()


def test_sample_output_memory_is_included_before_drawing():
    tape=qml.tape.QuantumScript([qml.Hadamard(0)],[qml.sample(wires=[0,1])],shots=10_000_000)
    d=IQPTNSimulator(2,mode='auto',auto_max_memory_bytes=100_000)
    report=d.plan(tape)
    assert report.selected=='default.qubit' and report.estimated_memory_bytes>=160_000_000
    assert report.output_elements==20_000_000
