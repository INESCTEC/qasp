"""Public QASP integration: supported programs select TN, other programs fall back."""
import numpy as np
import torch
import pennylane as qml
import pytest
from qasp import CircuitBuilder, IQPAnsatz
from qasp.feature_maps import AngleEmbedding, NeuralEmbedding
from qasp.classifiers import PQCClassifier
from qasp.genlearn import create_qcbm
from qasp.optimization import QAOA, QUBO, PCE
from qasp.qkernels import create_kernel
from qasp.rl.policies import PQCBornVariance
from qasp.simulators import IQPTNSimulator, DecoderSpec, UnsupportedIQP


def builder():
    b=CircuitBuilder(3)
    b.set_ansatz(IQPAnsatz(3,imprimitive_pattern='chain',feature_map=AngleEmbedding(3,rotations='Z',features=3)))
    return b


@pytest.mark.parametrize('device',['auto','IQPTNSimulator'])
def test_classifier_and_born_machine(device):
    a=PQCClassifier(circuit_builder=builder(),n_qubits=3,n_classes=3,device=device,init='zeros')
    b=PQCClassifier(circuit_builder=builder(),n_qubits=3,n_classes=3,device='default.qubit',init='zeros')
    b.load_state_dict(a.state_dict());x=torch.tensor([[.2,.3,.7],[-.5,.2,.1]],requires_grad=True)
    torch.testing.assert_close(a(x),b(x),atol=2e-6,rtol=2e-6)
    a(x).sum().backward()
    assert a.qnode.device.last_report.selected=='IQPTNSimulator'
    config={'ansatz':'IQPAnsatz','ansatz_config':{'n_layers':1,'imprimitive_pattern':'chain'},'loss':'mmd','mmd_config':{'mode':'exact'},'seed':41}
    a=create_qcbm({**config,'device':device},3,[[0,0,0],[1,1,1]])
    b=create_qcbm({**config,'device':'default.qubit'},3,[[0,0,0],[1,1,1]])
    torch.testing.assert_close(a.loss(),b.loss(),atol=1e-11,rtol=1e-11)
    ga=torch.autograd.grad(a.loss(),a.params)[0];gb=torch.autograd.grad(b.loss(),b.params)[0]
    torch.testing.assert_close(ga,gb,atol=1e-11,rtol=1e-11)


@pytest.mark.parametrize('policy,actions,blocks',[('majority-modulo',3,[[0,1],[2,3,4]]),('parity-like',3,None),('contiguous-like',3,None)])
def test_born_policy_neural_batch(policy,actions,blocks):
    kwargs=dict(n_qubits=5,n_actions=actions,policy=policy,majority_blocks=blocks,
        ansatz=IQPAnsatz,ansatz_config={'n_qubits':5,'n_layers':1,'imprimitive_pattern':'chain'},
        feature_map=NeuralEmbedding,feature_map_config={'n_qubits':5,'features':2,'rotations':'Z'})
    torch.manual_seed(8);a=PQCBornVariance(**kwargs,device='auto')
    torch.manual_seed(8);b=PQCBornVariance(**kwargs,device='default.qubit')
    b.load_state_dict(a.state_dict())
    a.load_state_dict(b.state_dict())
    x=torch.tensor([[.2,.4],[-.1,.3]],requires_grad=True)
    pa=torch.stack([a.action_probs(row) for row in x]);pb=torch.stack([b.action_probs(row) for row in x])
    torch.testing.assert_close(pa,pb,atol=2e-6,rtol=2e-6)
    ga=torch.autograd.grad(pa[:,0].sum(),(*a.qlayer.parameters(),x))
    gb=torch.autograd.grad(pb[:,0].sum(),(*b.qlayer.parameters(),x))
    for u,v in zip(ga,gb):torch.testing.assert_close(u,v,atol=3e-6,rtol=3e-6)
    assert a.circuit.device.last_report.selected=='IQPTNSimulator'


@pytest.mark.parametrize('layers',[1,2])
def test_optimization_auto_and_forced(layers):
    problem=QUBO.maxcut(3,[(0,1),(1,2),(0,2)])
    a=QAOA(problem,n_layers=layers,device='auto',backend='torch',seed=3)
    b=QAOA(problem,n_layers=layers,device='default.qubit',backend='torch',diff_method='backprop',seed=3)
    t=torch.tensor(a.initial_parameters(),dtype=torch.float64,requires_grad=True)
    ya=a.expectation_tensor(t);yb=b.expectation_tensor(t)
    torch.testing.assert_close(ya,yb,atol=1e-11,rtol=1e-11)
    ga=torch.autograd.grad(ya,t)[0];gb=torch.autograd.grad(yb,t)[0]
    torch.testing.assert_close(ga,gb,atol=1e-11,rtol=1e-11)
    assert a.device.last_report.selected==('IQPTNSimulator' if layers==1 else 'default.qubit')
    forced=QAOA(problem,n_layers=layers,device='IQPTNSimulator',backend='torch')
    if layers==2:
        with pytest.raises(UnsupportedIQP):forced.expectation_tensor(t)
    else:torch.testing.assert_close(forced.expectation_tensor(t),yb)


@pytest.mark.parametrize('kind',['projected','fidelity'])
def test_kernel_automatic_selection_and_fallback(kind):
    config={'kernel_type':kind,'feature_map':'AngleEmbedding','feature_map_config':{'rotations':'Z'},
            'ansatz':'IQPAnsatz','ansatz_config':{'n_layers':1,'imprimitive_pattern':'chain'},'init':'zeros'}
    a=create_kernel({**config,'device':'auto'},n_qubits=3);b=create_kernel({**config,'device':'default.qubit'},n_qubits=3)
    b.qlayer.load_state_dict(a.qlayer.state_dict());x=np.array([[.2,.5,.1],[.3,-.4,.6]])
    ga=a.gram(x,return_tensor=True);gb=b.gram(x,return_tensor=True)
    torch.testing.assert_close(ga,gb,atol=3e-6,rtol=3e-6)
    gradients=torch.autograd.grad(ga.sum(),tuple(a.parameters()))
    assert all(torch.isfinite(g).all() for g in gradients)


def test_pce_iqp_readout():
    problem=QUBO.maxcut(3,[(0,1),(1,2),(0,2)])
    config={'ansatz':'IQPAnsatz','ansatz_config':{'n_layers':1,'imprimitive_pattern':'chain'}}
    a=PCE(problem,circuit=config,device='IQPTNSimulator',backend='torch',seed=31)
    b=PCE(problem,circuit=config,device='default.qubit',backend='torch',diff_method='backprop',seed=31)
    t=torch.tensor(a.initial_parameters(),dtype=torch.float64,requires_grad=True)
    ya=a.loss_tensor(t);yb=b.loss_tensor(t)
    torch.testing.assert_close(ya,yb,atol=1e-11,rtol=1e-11)
    torch.testing.assert_close(torch.autograd.grad(ya,t)[0],torch.autograd.grad(yb,t)[0],atol=1e-11,rtol=1e-11)
