"""Public QASP modules share the unified device and retain Torch gradients."""
from copy import deepcopy
import numpy as np
import pennylane as qml
import pytest
import torch

from qasp import IQPSimulator, CircuitBuilder, IQPAnsatz
from qasp.feature_maps import AngleEmbedding
from qasp.rl.model import create_model
from qasp.classifiers import PQCClassifier
from qasp.qkernels import create_kernel
from qasp.genlearn import create_qcbm
from qasp.optimization import PCE, QUBO

EXACT = {"backend":"van_den_nest", "vn_sampling":"enumerate", "vn_batch_size":3}


@pytest.fixture(params=[1, 2], autouse=True)
def cpu_workers(request, monkeypatch):
    # Exact enumeration isolates integration errors from Monte Carlo noise.
    monkeypatch.setitem(EXACT, "vn_workers", request.param)
    return request.param


def builder():
    b=CircuitBuilder(3)
    b.set_ansatz(IQPAnsatz(3,imprimitive_pattern="chain",
        feature_map=AngleEmbedding(3,rotations="Z",features=3)))
    return b


@pytest.mark.parametrize("backend",["auto","tn","van_den_nest"])
def test_rl_neural_public_dictionary_adam_and_input_gradients(backend):
    config={
        "ansatz":"AngleAnsatz",
        "ansatz_config":{"n_layers":2,"rotations":["X 0","X 1","X 2","XX 0 1","XXX 0 1 2"],
                        "include_imprimitive":False,"include_hadamards_initial":False,"reuploading":True},
        "feature_map":"NeuralEmbedding","feature_map_config":{"features":2,"rotations":"X"},
        "n_actions":2,"observables":[qml.Z(0),qml.Z(1)@qml.Z(2)],
        "fc_layer":None,"device":"IQPSimulator","device_kwargs":{**EXACT,"backend":backend},
        "shots":None,"diff_method":"backprop",
    }
    torch.manual_seed(21)
    model=create_model("PQCSoftmax",deepcopy(config),n_qubits=3).double()
    reference=create_model("PQCSoftmax",{**deepcopy(config),"device":"default.qubit","device_kwargs":None},n_qubits=3).double()
    reference.load_state_dict(model.state_dict())
    optimizers=[torch.optim.Adam(m.parameters(),lr=.01) for m in (model,reference)]
    for step in range(2):
        values,grads=[],[]
        for m,opt in zip((model,reference),optimizers):
            x=torch.tensor([.4,-.3],dtype=torch.float64,requires_grad=True)
            opt.zero_grad()
            y=m(x); loss=-torch.log_softmax(y,-1)[0]
            loss.backward()
            values.append(y.detach())
            grads.append([p.grad.clone() for p in m.parameters()]+[x.grad.clone()])
            opt.step()
        torch.testing.assert_close(*values,atol=2e-11,rtol=2e-11)
        for a,b in zip(*grads):torch.testing.assert_close(a,b,atol=3e-10,rtol=3e-10)
        for a,b in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(a,b,atol=3e-10,rtol=3e-10)


@pytest.mark.parametrize("readout",["expectation","probabilities"])
def test_classifiers_batched_values_and_gradients(readout):
    kwargs=dict(circuit_builder=builder(),n_qubits=3,n_classes=2,readout=readout,init="normal_0_1")
    opts=EXACT if readout=="expectation" else {"backend":"auto"}
    if readout=="expectation":kwargs["observables"]=[qml.Z(0),qml.Z(1)@qml.Z(2)]
    model=PQCClassifier(**kwargs,device="iqp",device_kwargs=opts).double()
    ref=PQCClassifier(**kwargs,device="default.qubit").double()
    ref.load_state_dict(model.state_dict())
    x=torch.tensor([[.2,.3,.7],[-.5,.2,.1]],dtype=torch.float64,requires_grad=True)
    a,b=model(x),ref(x)
    torch.testing.assert_close(a,b,atol=2e-11,rtol=2e-11)
    ga=torch.autograd.grad(a.sum(),(*model.parameters(),x))
    gb=torch.autograd.grad(b.sum(),(*ref.parameters(),x))
    for u,v in zip(ga,gb):torch.testing.assert_close(u,v,atol=2e-10,rtol=2e-10)


def test_projected_z_kernel_values_and_gradients():
    config={"kernel_type":"projected","feature_map":"AngleEmbedding","feature_map_config":{"rotations":"Z"},
            "ansatz":"IQPAnsatz","ansatz_config":{"n_layers":1,"imprimitive_pattern":"chain"},
            "observables":["Z 0","ZZ 1 2"],"init":"normal_0_1","diff_method":"backprop"}
    a=create_kernel({**deepcopy(config),"device":"iqp","device_kwargs":EXACT},n_qubits=3)
    b=create_kernel({**deepcopy(config),"device":"default.qubit"},n_qubits=3)
    b.qlayer.load_state_dict(a.qlayer.state_dict())
    x=np.array([[.2,.5,.1],[.3,-.4,.6]])
    ya,yb=a.gram(x,return_tensor=True),b.gram(x,return_tensor=True)
    torch.testing.assert_close(ya,yb,atol=3e-6,rtol=3e-6)
    ga=torch.autograd.grad(ya.sum(),tuple(a.parameters()))
    gb=torch.autograd.grad(yb.sum(),tuple(b.parameters()))
    for u,v in zip(ga,gb):torch.testing.assert_close(u,v,atol=3e-6,rtol=3e-6)


def test_pce_z_iqp_objective():
    problem=QUBO.maxcut(3,[(0,1),(1,2),(0,2)])
    cfg={"ansatz":"IQPAnsatz","ansatz_config":{"n_layers":1,"imprimitive_pattern":"chain"}}
    a=PCE(problem,circuit=deepcopy(cfg),device="iqp",device_kwargs=EXACT,paulis="Z",backend="torch",seed=31)
    b=PCE(problem,circuit=deepcopy(cfg),device="default.qubit",paulis="Z",backend="torch",diff_method="backprop",seed=31)
    t=torch.tensor(a.initial_parameters(),dtype=torch.float64,requires_grad=True)
    ya,yb=a.loss_tensor(t),b.loss_tensor(t)
    torch.testing.assert_close(ya,yb,atol=2e-11,rtol=2e-11)
    torch.testing.assert_close(torch.autograd.grad(ya,t)[0],torch.autograd.grad(yb,t)[0],atol=2e-10,rtol=2e-10)


def gen_config():
    return {"ansatz":"IQPAnsatz","ansatz_config":{"n_layers":1,"imprimitive_pattern":"chain"},
            "loss":"mmd","mmd_config":{"mode":"exact"},"seed":41,"diff_method":"backprop"}


def test_genlearn_exact_mmd_and_gradients():
    cfg=gen_config()
    a=create_qcbm({**deepcopy(cfg),"device":"iqp","device_kwargs":EXACT},3,[[0,0,0],[1,1,1]])
    b=create_qcbm({**deepcopy(cfg),"device":"default.qubit"},3,[[0,0,0],[1,1,1]])
    ya,yb=a.loss(),b.loss()
    torch.testing.assert_close(ya,yb,atol=2e-11,rtol=2e-11)
    torch.testing.assert_close(torch.autograd.grad(ya,a.params)[0],torch.autograd.grad(yb,b.params)[0],atol=2e-10,rtol=2e-10)


@pytest.mark.parametrize("observable_sampling", ["all_terms", "importance"])
def test_genlearn_sampled_mmd_uses_independent_replicas(monkeypatch, cpu_workers, observable_sampling):
    cfg=gen_config()
    model=create_qcbm({**cfg,"device":"iqp","device_kwargs":{"backend":"van_den_nest","vn_samples":128,
        "vn_sampling":"fixed","seed":12,"vn_workers":cpu_workers,"vn_batch_size":32,
        "vn_observable_sampling":observable_sampling}},3,[[0,0,0],[1,1,1]])
    moments=model.moments
    calls=[]
    def record(*args,**kwargs):
        result=moments(*args,**kwargs);calls.append(result.detach().clone());return result
    monkeypatch.setattr(model,"moments",record)
    # Make sampled variance non-negligible.
    with torch.no_grad():model.params.fill_(.8)
    loss=model.loss()
    assert len(calls)==2
    assert not torch.equal(calls[0],calls[1])
    assert model.device.last_report.sampling=="resample"
    assert torch.isfinite(torch.autograd.grad(loss,model.params)[0]).all()
    assert model.device._independent_scopes==[]


def test_born_policy_compact_readout_stays_exact():
    config={"ansatz":"IQPAnsatz","ansatz_config":{"n_layers":1,"imprimitive_pattern":"chain"},
            "feature_map":"AngleEmbedding","feature_map_config":{"features":3,"rotations":"Z"},
            "n_actions":2,"policy":"majority-modulo","training_method":"probs",
            "parity_cnot_decomposition":False,"device":"IQPSimulator","device_kwargs":{"backend":"auto"},
            "diff_method":"backprop","shots":None}
    a=create_model("PQCBorn",deepcopy(config),n_qubits=3)
    b=create_model("PQCBorn",{**deepcopy(config),"device":"default.qubit","device_kwargs":None},n_qubits=3)
    b.load_state_dict(a.state_dict())
    x=torch.tensor([.2,.3,-.1])
    torch.testing.assert_close(a(x),b(x),atol=2e-6,rtol=2e-6)
    a(x)[0].backward()
    assert a._tn_action_readout and a.circuit.device.last_report.selected=="tn"
