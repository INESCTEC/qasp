"""Product-basis normalization and public dictionary-configured RL policies."""
from copy import deepcopy
import numpy as np
import pennylane as qml
import pytest
import torch

from qasp import IQPTNSimulator, DecoderSpec, action_probs
from qasp.rl.model import create_policy
from qasp.simulators import UnsupportedIQP


@pytest.mark.parametrize('axes', ['XXXX', 'YYYY', 'XYZX'])
@pytest.mark.parametrize('precision', ['complex64', 'complex128'])
@pytest.mark.parametrize('readout', ['probs', 'actions', 'expectation', 'variance'])
def test_product_basis_values_gradients_and_cache(axes, precision, readout):
    dtype = torch.float32 if precision == 'complex64' else torch.float64
    tol = 8e-6 if dtype == torch.float32 else 5e-11
    theta = torch.zeros(5, dtype=dtype, requires_grad=True)
    inputs = torch.tensor([.2, -.3], dtype=dtype, requires_grad=True)
    decoder = DecoderSpec('majority-modulo', 3, ((0,1),(1,2)))
    wires = ['b', 7, 'a', 2]
    measured = [2, 'b', 'a']
    def gates(t, x):
        for j,w in enumerate(wires):
            qml.RY(x[j%2],w); qml.RZ(.13,w)
        qml.PauliRot(t[0]+x[0], axes[:2], wires[:2])
        qml.adjoint(qml.PauliRot(t[1],axes[1],wires[1]))
        qml.PauliRot(t[2], axes[1:3], wires[1:3])
        qml.PauliRot(t[3], axes[0]+'I'+axes[2:], wires)
        qml.PauliRot(t[4], axes, wires)
        for w in wires: qml.RX(.31,w)
    def circuit(t,x,reference=False):
        gates(t,x)
        if readout=='actions':
            return qml.probs(wires=measured) if reference else action_probs(measured,decoder)
        if readout=='probs': return qml.probs(wires=measured)
        obs=qml.X('b')@qml.Y(2)+.23*qml.Z(7)
        return qml.var(obs) if readout=='variance' else qml.expval(obs)
    dev=IQPTNSimulator(wires, precision=precision,inference_execution='torchscript')
    node=qml.QNode(circuit,dev,interface='torch',diff_method='backprop')
    ref=qml.QNode(lambda t,x:circuit(t,x,True),qml.device('default.qubit',wires=wires),interface='torch',diff_method='backprop')
    def reference():
        y=ref(theta,inputs)
        return decoder.aggregate(y) if readout=='actions' else y
    with torch.no_grad():
        node(theta,inputs)
        traces=dev.inference_cache_info()['traces']
        theta.copy_(torch.linspace(-.7,.8,5,dtype=dtype))
        torch.testing.assert_close(node(theta,inputs),reference().to(dtype),atol=tol,rtol=tol)
        assert dev.inference_cache_info()['traces']==traces>0
        assert dev.inference_cache_info()['fallbacks']==0
    cache=dev.inference_cache_info()
    actual=node(theta,inputs); expected=reference().to(dtype)
    torch.testing.assert_close(actual,expected,atol=tol,rtol=tol)
    weights=torch.linspace(.2,.9,actual.numel(),dtype=dtype).reshape(actual.shape)
    ga=torch.autograd.grad((actual*weights).sum(),(theta,inputs))
    gb=torch.autograd.grad((expected*weights).sum(),(theta,inputs))
    for a,b in zip(ga,gb): torch.testing.assert_close(a,b,atol=tol,rtol=tol)
    assert dev.inference_cache_info()==cache


@pytest.mark.parametrize('axis', ['X','Y'])
@pytest.mark.parametrize('angle', [0., .37, np.pi/2])
def test_native_ising_local_gates_shared_parameters_and_hessian(axis,angle):
    rotation=getattr(qml,'R'+axis)
    ising=getattr(qml,'Ising'+axis*2)
    pauli=getattr(qml,'Pauli'+axis)
    def circuit(t):
        qml.RY(.2,0)
        ising(t,[0,1])
        rotation(t,0); pauli(1)
        qml.adjoint(rotation(t,1))
        if axis=='X': qml.SX(0); qml.adjoint(qml.SX(1))
        ising(t,[0,1])
        return qml.expval(qml.Z(0)@qml.Z(1))
    t=torch.tensor(angle,dtype=torch.float64,requires_grad=True)
    values=[]
    for device in [IQPTNSimulator(2),qml.device('default.qubit',wires=2)]:
        node=qml.QNode(circuit,device,interface='torch',diff_method='backprop')
        value=node(t); grad=torch.autograd.grad(value,t,create_graph=True)[0]
        hessian=torch.autograd.grad(grad,t)[0]
        values.append(torch.stack([value,grad,hessian]))
    torch.testing.assert_close(*values,atol=3e-11,rtol=3e-11)


@pytest.mark.parametrize('middle', ['YY','RZ','CNOT'])
def test_incompatible_basis_rejects_or_falls_back(middle):
    def circuit(t):
        qml.IsingXX(t,[0,1])
        if middle=='YY': qml.IsingYY(t,[0,1])  # commuting, but not a product basis
        elif middle=='RZ': qml.RZ(t,0)
        else: qml.CNOT([0,1])
        qml.IsingXX(t,[0,1])
        return qml.probs(wires=[0,1])
    theta=torch.tensor(.3,dtype=torch.float64,requires_grad=True)
    explicit=qml.QNode(circuit,IQPTNSimulator(2),interface='torch',diff_method='backprop')
    with pytest.raises(UnsupportedIQP): explicit(theta)
    automatic=qml.QNode(circuit,IQPTNSimulator(2,mode='auto'),interface='torch',diff_method='backprop')
    reference=qml.QNode(circuit,qml.device('default.qubit',wires=2),interface='torch',diff_method='backprop')
    torch.testing.assert_close(automatic(theta),reference(theta),atol=2e-11,rtol=2e-11)
    assert automatic.device.last_report.selected=='default.qubit'


@pytest.mark.parametrize('ansatz', ['IQPAnsatz','AngleAnsatz'])
@pytest.mark.parametrize('decoder', ['majority-modulo','parity-like','contiguous-like'])
@pytest.mark.parametrize('device', ['IQPTNSimulator','auto'])
def test_rl_dictionary_factory_backprop_and_adam(ansatz,decoder,device):
    config={
        'ansatz':ansatz,
        'ansatz_config':{'n_layers':1,'imprimitive_pattern':'chain'} if ansatz=='IQPAnsatz' else {
            'n_layers':2,'rotations':['X 0','X 1','X 2','XX 0 1','XXX 0 1 2'],
            'include_imprimitive':False,'reuploading':True},
        'feature_map':'AngleEmbedding',
        'feature_map_config':{'rotations':'Z' if ansatz=='IQPAnsatz' else 'X','features':3,'input_scaling':True},
        'n_actions':3,'policy':decoder,'parity_cnot_decomposition':False,
        'device':device,'diff_method':'backprop','training_method':'probs','shots':None,
    }
    if decoder=='majority-modulo': config['majority_blocks']=[[0],[1,2]]
    torch.manual_seed(29)
    model=create_policy('PQCBorn',deepcopy(config),n_qubits=3).double()
    ref=create_policy('PQCBorn',{**deepcopy(config),'device':'default.qubit'},n_qubits=3).double()
    ref.load_state_dict(model.state_dict())
    # Avoid magnifying round-off in mathematically zero gradients into
    # backend-dependent updates; retain the strict value/gradient tolerances.
    optimizers=[torch.optim.Adam(m.parameters(),lr=.01,eps=1e-6) for m in (model,ref)]
    for _ in range(2):
        outputs=[]; grads=[]
        for m,opt in zip((model,ref),optimizers):
            x=torch.tensor([.2,-.3,.5],dtype=torch.float64,requires_grad=True)
            opt.zero_grad(); p=m(x); (-p[1].log()).backward()
            assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in m.parameters())
            outputs.append(p.detach()); grads.append([t.grad.clone() for t in m.parameters()]+[x.grad])
            opt.step()
        torch.testing.assert_close(*outputs,atol=2e-10,rtol=2e-10)
        for a,b in zip(*grads): torch.testing.assert_close(a,b,atol=2e-9,rtol=2e-9)
        for a,b in zip(model.parameters(),ref.parameters()): torch.testing.assert_close(a,b,atol=2e-9,rtol=2e-9)
    assert model.circuit.device.last_report.selected=='IQPTNSimulator'


@pytest.mark.parametrize('readout', ['born', 'softmax'])
def test_neural_commuting_policy_encoder_input_and_adam(readout):
    """Neural weights and inputs stay live across cached rollouts and updates."""
    n = 4
    config = {
        'ansatz': 'AngleAnsatz',
        'ansatz_config': {
            'n_layers': 2,
            'rotations': ([f'X {w}' for w in range(n)]
                          + [f'XX {w} {w+1}' for w in range(n-1)]
                          + [f'XXX {w} {w+1} {w+2}' for w in range(n-2)]),
            'include_imprimitive': False, 'include_hadamards_initial': False,
            'reuploading': True,
        },
        'feature_map': 'NeuralEmbedding',
        'feature_map_config': {'features': 4, 'rotations': 'X'},
        'n_actions': 2, 'diff_method': 'backprop', 'shots': None,
    }
    if readout == 'born':
        config.update(policy='majority-modulo', majority_blocks=[list(range(n))],
                      training_method='probs', parity_cnot_decomposition=False)
    else:
        config.update(observables=[qml.Z(0), qml.Z(n-1)], fc_layer=None)
    selector = 'PQCBorn' if readout == 'born' else 'PQCSoftmax'
    torch.manual_seed(2026)
    device = IQPTNSimulator(n, inference_execution='torchscript')
    actual = create_policy(selector, {**deepcopy(config), 'device': device}, n_qubits=n).double()
    reference = create_policy(selector, {**deepcopy(config), 'device': 'default.qubit'}, n_qubits=n).double()
    reference.load_state_dict(actual.state_dict())
    optimizers = [torch.optim.Adam(m.parameters(), lr=.01) for m in (actual, reference)]
    def probs(model, x):
        value = model(x)
        return value if readout == 'born' else torch.softmax(value, dim=-1)
    x0 = torch.tensor([.2, -.4, .3, .1], dtype=torch.float64)
    traces = None
    for step in range(3):
        with torch.no_grad():
            torch.testing.assert_close(probs(actual, x0), probs(reference, x0), atol=2e-11, rtol=2e-11)
        info = device.inference_cache_info()
        if traces is not None:
            assert info['traces'] == traces and info['kernel_hits'] > 0
        traces = info['traces']
        values, gradients = [], []
        for model, optimizer in zip((actual, reference), optimizers):
            x = x0.clone().requires_grad_()
            optimizer.zero_grad()
            p = probs(model, x)
            (-p[1].log()).backward()
            encoder_grad = dict(model.named_parameters())['qlayer.input_scaling_params'].grad
            assert encoder_grad is not None and torch.isfinite(encoder_grad).all()
            assert encoder_grad.abs().max() > 1e-8
            assert x.grad is not None and torch.isfinite(x.grad).all()
            values.append(p.detach())
            gradients.append([p.grad.clone() for p in model.parameters()] + [x.grad.clone()])
            optimizer.step()
        torch.testing.assert_close(*values, atol=2e-11, rtol=2e-11)
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, atol=2e-10, rtol=2e-10)
        for a, b in zip(actual.parameters(), reference.parameters()):
            torch.testing.assert_close(a, b, atol=2e-10, rtol=2e-10)
        assert device.inference_cache_info() == info
        assert device.last_report.selected == 'IQPTNSimulator'
    assert traces > 0 and device.inference_cache_info()['fallbacks'] == 0
