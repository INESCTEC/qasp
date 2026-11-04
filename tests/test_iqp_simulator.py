"""Published correlator identity, stochastic semantics, and unified dispatch."""
from copy import deepcopy
import math
import numpy as np
import pennylane as qml
import pytest
import torch

from qasp import IQPSimulator, IQPOptions, IQPTNSimulator
from qasp.devices import create_device
from qasp.simulators import UnsupportedIQP


@pytest.mark.parametrize("picture", ["x", "z"])
@pytest.mark.parametrize("chunk", [3, 64])
@pytest.mark.parametrize("checkpoint", [True, False])
def test_enumerated_identity_values_gradients_hessian(picture, chunk, checkpoint):
    wires = ["q", 4, "r", 1]
    def circuit(t, x, w, c):
        a = w @ x
        if picture == "z":
            for wire in wires: qml.Hadamard(wire)
        axis = "X" if picture == "x" else "Z"
        qml.PauliRot(a[0]+t[0], axis, wires[0])
        qml.PauliRot(t[1], axis*2, wires[:2])
        qml.adjoint(qml.PauliRot(t[2], axis*3, wires[1:]))
        qml.PauliRot(t[3]+a[1], axis+"I"+axis*2, wires)
        qml.PauliRot(t[4], "IIII", wires)
        if picture == "z":
            for wire in wires: qml.Hadamard(wire)
        return qml.expval(c[0]*qml.Z("q") + c[1]*qml.Z(4)@qml.Z(1) + c[2]*qml.I("r")), qml.expval(qml.Z("r"))
    args = (torch.tensor([0.,.2,-.4,.6,.8],dtype=torch.float64,requires_grad=True),
            torch.tensor([.2,-.3],dtype=torch.float64,requires_grad=True),
            torch.tensor([[.7,.4],[-.2,.9]],dtype=torch.float64,requires_grad=True),
            torch.tensor([.2,-.6,1.],dtype=torch.float64,requires_grad=True))
    devices = [IQPSimulator(wires, backend="van_den_nest", vn_sampling="enumerate",
                vn_batch_size=chunk, vn_observable_batch_size=1, vn_checkpoint=checkpoint),
               qml.device("default.qubit",wires=wires)]
    values, gradients, hessians = [], [], []
    for device in devices:
        node=qml.QNode(circuit,device,interface="torch",diff_method="backprop")
        value=torch.stack(node(*args))
        grad=torch.autograd.grad(value.sum(),args,create_graph=True)
        hessian=torch.autograd.grad(sum(g.sum() for g in grad),args,allow_unused=True)
        values.append(value); gradients.append(grad); hessians.append(hessian)
    torch.testing.assert_close(*values,atol=2e-12,rtol=2e-12)
    for a,b in zip(*gradients): torch.testing.assert_close(a,b,atol=2e-12,rtol=2e-12)
    for a,b in zip(*hessians):
        if a is not None and b is not None: torch.testing.assert_close(a,b,atol=3e-12,rtol=3e-12)
    assert devices[0].last_report.standard_errors == (0.,0.)


def test_diagonal_controlled_phases_and_adjoint():
    def f(t):
        for w in range(3): qml.Hadamard(w)
        qml.RZ(t,0); qml.CZ([0,1]); qml.ControlledPhaseShift(t,[1,2])
        qml.S(1); qml.adjoint(qml.T(0)); qml.adjoint(qml.MultiRZ(t,[0,1,2]))
        for w in range(3): qml.Hadamard(w)
        return qml.expval(qml.Z(0)@qml.Z(1)),qml.expval(qml.Z(2))
    t=torch.tensor(.7,dtype=torch.float64,requires_grad=True)
    outputs=[]
    for d in [IQPSimulator(3,backend="van_den_nest",vn_sampling="enumerate"),qml.device("default.qubit",wires=3)]:
        y=torch.stack(qml.QNode(f,d,interface="torch")(t)); g=torch.autograd.grad(y.sum(),t)[0]
        outputs.append((y,g))
    for a,b in zip(*outputs):torch.testing.assert_close(a,b,atol=2e-12,rtol=2e-12)


def noisy_circuit(t):
    qml.RX(t[0],0);qml.RX(t[1],1);qml.IsingXX(t[2],[0,1])
    return qml.expval(qml.Z(0)+2*qml.Z(1)),qml.expval(qml.Z(0)@qml.Z(1))


def test_monte_carlo_uncertainty_reproducibility_and_live_gradients():
    t=torch.tensor([.8,.7,1.1],dtype=torch.float64,requires_grad=True)
    d=IQPSimulator(2,backend="van_den_nest",seed=71,vn_samples=12000,vn_sampling="fixed",vn_batch_size=1000)
    node=qml.QNode(noisy_circuit,d,interface="torch",diff_method="backprop")
    y=torch.stack(node(t)); report=d.last_report
    torch.testing.assert_close(y,torch.stack(node(t)),atol=0,rtol=0)
    ref=torch.stack(qml.QNode(noisy_circuit,qml.device("default.qubit",wires=2),interface="torch")(t))
    assert torch.all((y-ref).abs() < torch.tensor(report.hoeffding_half_widths))
    assert all(0 < s < .04 for s in report.standard_errors)
    g=torch.autograd.grad(y.sum(),t)[0]
    gref=torch.autograd.grad(ref.sum(),t)[0]
    torch.testing.assert_close(g,gref,atol=.08,rtol=0)
    torch.testing.assert_close(g,torch.autograd.functional.jacobian(lambda x: torch.stack(node(x)).sum(),t),atol=1e-12,rtol=1e-12)
    # Compare differentiable empirical objective to finite differences of SAME bank.
    delta=1e-5
    for i in range(3):
        shift=torch.eye(3,dtype=t.dtype)[i]*delta
        fd=(sum(node(t+shift))-sum(node(t-shift)))/(2*delta)
        torch.testing.assert_close(g[i],fd,atol=1e-8,rtol=1e-8)
    d.resample(72)
    assert not torch.equal(y,torch.stack(node(t)))
    d.resample(71)
    torch.testing.assert_close(y,torch.stack(node(t)),atol=0,rtol=0)
    other=IQPSimulator(2,backend="van_den_nest",seed=71,vn_samples=12000,vn_sampling="fixed",vn_batch_size=1000)
    torch.testing.assert_close(y,torch.stack(qml.QNode(noisy_circuit,other,interface="torch")(t)),atol=0,rtol=0)
    # Later forward calls cannot change samples used by checkpointed backward.
    before=torch.stack(node(t)).sum()
    d.resample(5); node(t)
    torch.testing.assert_close(torch.autograd.grad(before,t)[0],g,atol=1e-12,rtol=1e-12)


def test_confidence_budget_is_for_weighted_sum_and_plan_does_not_sample():
    d=IQPSimulator(2,backend="van_den_nest",vn_samples=2,vn_error=.1,vn_confidence=.95)
    tape=qml.tape.make_qscript(noisy_circuit)([.8,.7,1.1])
    before=deepcopy(d._vn_rng.bit_generator.state)
    r=d.plan(tape)
    assert d._vn_rng.bit_generator.state==before
    assert r.samples==math.ceil(2*9*math.log(4/.05)/.1**2)
    assert max(r.hoeffding_half_widths)<=.1
    limited=IQPSimulator(2,backend="van_den_nest",vn_error=1e-6)
    with pytest.raises(UnsupportedIQP,match="vn_max_samples"): limited.execute(tape)


@pytest.mark.parametrize("kind",["probs","var","xobs","preparation","open_h","noncommuting","shots"])
def test_ineligible_estimator_never_silently_changes_measurement(kind):
    def f(t):
        if kind=="preparation":qml.RY(t,0)
        qml.IsingXX(t,[0,1])
        if kind=="noncommuting":qml.RZ(t,0);qml.IsingXX(t,[0,1])
        if kind=="open_h":qml.Hadamard(0)
        if kind=="probs":return qml.probs(wires=[0,1])
        if kind=="var":return qml.var(qml.Z(0))
        return qml.expval(qml.X(0) if kind=="xobs" else qml.Z(0))
    d=IQPSimulator(2,backend="van_den_nest",shots=10 if kind=="shots" else None)
    with pytest.raises(UnsupportedIQP):qml.QNode(f,d)(.3)
    if kind not in {"shots","noncommuting"}:
        auto=IQPSimulator(2)
        qml.QNode(f,auto)(.3)
        assert auto.last_report.selected=="tn"


def test_selector_and_compatibility():
    tape=qml.tape.make_qscript(noisy_circuit)([.8,.7,1.1])
    auto=IQPSimulator(2)
    assert auto.plan(tape).selected=="tn"
    small_mc=IQPSimulator(2,vn_samples=2)
    assert small_mc.plan(tape).selected=="van_den_nest"
    exact=IQPSimulator(2,allow_approximate=False)
    assert exact.plan(tape).selected=="tn"
    limited=IQPSimulator(2,auto_max_flops=0)
    assert limited.plan(tape).selected=="van_den_nest"
    skip=IQPSimulator(2,auto_tn_max_wires=1)
    assert skip.plan(tape).tn_planning_skipped
    for alias in ("IQPSimulator","iqp",IQPSimulator):
        assert isinstance(create_device(alias,wires=2),IQPSimulator)
    assert type(create_device("auto",wires=2)) is IQPTNSimulator
    with pytest.raises(ValueError,match="mpi"):create_device("iqp",wires=2,mpi=True)
    with pytest.raises(MemoryError): IQPSimulator(2,backend="van_den_nest",hard_max_flops=0).execute(tape)


def test_broadcast_numpy_identity_and_resampled_scope():
    d=IQPSimulator(2,backend="van_den_nest",vn_sampling="enumerate",vn_batch_size=2)
    def f(t):qml.IsingXX(t,[0,1]);return qml.expval(qml.Z(0))
    t=torch.tensor([.2,.3],dtype=torch.float64,requires_grad=True)
    y=qml.QNode(f,d,interface="torch")(t)
    torch.testing.assert_close(y,t.cos())
    torch.testing.assert_close(torch.autograd.grad(y.sum(),t)[0],-t.sin())
    assert float(d.execute(qml.tape.make_qscript(f)(.2)))==pytest.approx(math.cos(.2))
    mc=IQPSimulator(2,backend="van_den_nest",vn_sampling="fixed",vn_samples=100,seed=7)
    node=qml.QNode(noisy_circuit,mc,interface="torch")
    x=torch.tensor([.8,.7,1.1],dtype=torch.float64)
    with mc.independent_estimates() as scope:
        a=torch.stack(node(x));b=torch.stack(node(x))
    assert scope["approximate"] and not torch.equal(a,b)
    assert mc.last_report.sampling=="resample"
    identity=qml.tape.QuantumScript([qml.PauliRot(x[0],"II",[0,1])],[qml.expval(qml.I(0))])
    assert float(mc.execute(identity))==1.


@pytest.mark.parametrize("kwargs",[{"vn_samples":1},{"vn_sampling":"bad"},{"vn_confidence":1},{"vn_error":0},{"vn_max_exact_wires":25}])
def test_options_validation(kwargs):
    with pytest.raises(ValueError): IQPOptions(**kwargs)


def test_thousand_wire_high_order_rotation_never_builds_tn(monkeypatch):
    n=1000
    device=IQPSimulator(n,vn_samples=64,seed=1)
    def forbidden(*args,**kwargs):raise AssertionError("large estimator case must not build a TN")
    monkeypatch.setattr(device,"_compile",forbidden)
    def circuit(t):
        qml.PauliRot(t,"X"*n,range(n))
        return qml.expval(qml.Z(0))
    theta=torch.tensor(.43,dtype=torch.float64,requires_grad=True)
    y=qml.QNode(circuit,device,interface="torch")(theta)
    torch.testing.assert_close(y,theta.cos())
    torch.testing.assert_close(torch.autograd.grad(y,theta)[0],-theta.sin())
    assert device.last_report.tn_planning_skipped


def test_cached_auto_plan_tracks_changed_structure_and_live_values(monkeypatch):
    d=IQPSimulator(2,vn_samples=2,vn_sampling="fixed")
    compile_original=d._compile
    calls=[]
    def record(tape):
        calls.append(1);return compile_original(tape)
    monkeypatch.setattr(d,"_compile",record)
    node=qml.QNode(noisy_circuit,d,interface="torch")
    x=torch.tensor([.8,.7,1.1],dtype=torch.float64,requires_grad=True)
    first=torch.stack(node(x))
    second=torch.stack(node(x+.1))
    assert len(calls)==1 and not torch.equal(first,second)
    d.clear_plan_cache()
    node(x)
    assert len(calls)==2
    # A different observable on the same circuit invalidates the plan.
    ops=qml.tape.make_qscript(noisy_circuit)(x).operations
    d.plan(qml.tape.QuantumScript(ops,[qml.expval(qml.Z(0))]),requires_grad=True)
    assert len(calls)==3


def test_generic_auto_fallback_is_bounded_and_force_tn_is_strict():
    def f(t):
        qml.IsingXX(t,[0,1]);qml.RZ(t,0);qml.IsingXX(t,[0,1])
        return qml.expval(qml.Z(0))
    theta=torch.tensor(.5,dtype=torch.float64,requires_grad=True)
    d=IQPSimulator(2)
    actual=qml.QNode(f,d,interface="torch")(theta)
    ref=qml.QNode(f,qml.device("default.qubit",wires=2),interface="torch")(theta)
    torch.testing.assert_close(actual,ref)
    assert d.last_report.selected=="default.qubit"
    with pytest.raises(UnsupportedIQP):qml.QNode(f,IQPSimulator(2,backend="tn"),interface="torch")(theta)
    with pytest.raises(RuntimeError,match="fallback_max_wires"):qml.QNode(f,IQPSimulator(2,fallback_max_wires=1),interface="torch")(theta)


def test_real_complex_typed_coefficients_and_tiny_error_limit():
    def circuit(t):
        qml.RX(t,0)
        return qml.expval((1+0j)*qml.Z(0))
    d=IQPSimulator(1,backend="van_den_nest",vn_sampling="enumerate")
    theta=torch.tensor(.2,dtype=torch.float64,requires_grad=True)
    y=qml.QNode(circuit,d,interface="torch")(theta)
    torch.testing.assert_close(y,theta.cos())
    tape=qml.tape.make_qscript(circuit)(theta)
    tiny=IQPSimulator(1,backend="van_den_nest",vn_error=1e-300)
    with pytest.raises(UnsupportedIQP,match="vn_max_samples"):tiny.execute(tape)
