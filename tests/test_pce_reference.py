"""PCE paper equations, dense-state oracles, and classical postprocessing.

Reference: Sciorilli et al., https://doi.org/10.1038/s41467-024-55346-z.
"""
import math
from itertools import combinations, product
import numpy as np
import pytest
import pennylane as qml
import torch
from qasp import PCE, QUBO
from qasp.optimization import build_pauli_correlation_encoding


@pytest.fixture(autouse=True)
def double_precision():
    # Lightning returns Python scalar values through PennyLane's Torch bridge,
    # which uses Torch's default dtype even with float64 circuit parameters.
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(previous)


MATRICES={
    'I':np.eye(2,dtype=complex),
    'X':np.array([[0,1],[1,0]],dtype=complex),
    'Y':np.array([[0,-1j],[1j,0]],dtype=complex),
    'Z':np.diag([1,-1]).astype(complex),
}
H=np.array([[1,1],[1,-1]])/np.sqrt(2)

def tensor(matrices):
    value=np.array([[1]],dtype=complex)
    for matrix in matrices:value=np.kron(value,matrix)
    return value

def pauli(n,word,wires):
    factors=['I']*n
    for axis,wire in zip(word,wires):factors[wire]=axis
    return tensor([MATRICES[p] for p in factors])

def layout(n,commuting=False):
    if commuting:return [f'X {j}' for j in range(n)]+[f'XX {j} {j+1}' for j in range(n-1)]
    return [f'{p} {j}' for j in range(n) for p in 'YZ']+[f'ZZ {j} {j+1}' for j in range(n-1)]

def config(n,commuting=False):
    return {'ansatz':'AngleAnsatz','ansatz_config':{
        'n_layers':2,'rotations':layout(n,commuting),'include_imprimitive':False,
        'include_hadamards_initial':not commuting}}

def dense_correlations(theta,n,m,order,axes,commuting=False):
    state=np.zeros(2**n,dtype=complex);state[0]=1
    if not commuting:state=tensor([H]*n)@state
    words=layout(n,commuting)*2
    for angle,spec in zip(theta,words):
        axis,*wires=spec.split();P=pauli(n,axis,list(map(int,wires)))
        state=np.cos(angle/2)*state-1j*np.sin(angle/2)*(P@state)
    observables=[pauli(n,p*order,w) for p in axes for w in combinations(range(n),order)][:m]
    return np.array([np.vdot(state,P@state).real for P in observables])

def relaxed_binary_energy(problem,spins):
    # E[E(x)] for independent bits with E[x_i]=(1-spin_i)/2.
    p=(1-np.asarray(spins))/2
    result=problem.offset+np.dot(np.diag(problem.matrix),p)
    for i,j in combinations(range(len(p)),2):
        result+=(problem.matrix[i,j]+problem.matrix[j,i])*p[i]*p[j]
    return result

def independent_loss(theta,solver,order,axes,commuting=False):
    c=dense_correlations(theta,solver.n_qubits,solver.problem.n_variables,order,axes,commuting)
    s=np.tanh(solver.alpha*c)
    return 2*(relaxed_binary_energy(solver.problem,s)-relaxed_binary_energy(solver.problem,np.zeros_like(s)))+solver.regularization*reference_scale(solver.problem)*np.mean(s*s)**2

def reference_scale(problem):
    # Independent weighted minimum spanning forest via SciPy, not QASP's helper.
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import minimum_spanning_tree
    Q=problem.matrix; pairs=Q+Q.T; np.fill_diagonal(pairs,0)
    weights=np.maximum(pairs/2,0)
    h=-np.diag(Q)/2-pairs.sum(axis=1)/4
    forest=minimum_spanning_tree(csr_matrix(weights)).sum()
    return weights.sum()/4+float(forest)/4+np.abs(h).sum()

def finite_difference(fn,x):
    eps=1e-6;gradient=np.empty_like(x)
    for i in range(len(x)):
        direction=np.zeros_like(x);direction[i]=eps
        gradient[i]=(fn(x+direction)-fn(x-direction))/(2*eps)
    return gradient

@pytest.mark.parametrize('order',[1,2,3,4])
@pytest.mark.parametrize('axes',['XYZ','Z','XY'])
def test_encoding_capacity_unique_words_and_commuting_families(order,axes):
    for m in [1,3,7,12]:
        n,terms=build_pauli_correlation_encoding(m,correlation_order=order,paulis=axes)
        assert len(axes)*math.comb(n,order)>=m
        assert n==order or len(axes)*math.comb(n-1,order)<m
        expected=[(p*order,w) for p in axes for w in combinations(range(n),order)][:m]
        assert [(t.pauli,t.wires) for t in terms]==expected
        assert len(set(expected))==m
        for t in terms:assert len(t.wires)==order and len(set(t.wires))==order

@pytest.mark.parametrize('seed',[2,13,31])
def test_binary_to_ising_and_soft_relaxation(seed):
    rng=np.random.default_rng(seed);Q=rng.normal(size=(5,5));problem=QUBO(Q,offset=.73)
    c,h,J=problem.to_ising()
    for bits in product([0,1],repeat=5):
        s=1-2*np.array(bits)
        reference=.73+np.array(bits)@Q@np.array(bits)
        actual=c+sum(v*s[i] for i,v in h.items())+sum(v*s[i]*s[j] for (i,j),v in J.items())
        assert actual==pytest.approx(reference,abs=1e-11)
    s=rng.uniform(-.8,.8,5)
    oracle=0.
    for bits in product([0,1],repeat=5):
        p=(1-s)/2;b=np.array(bits)
        oracle+=np.prod(np.where(b,p,1-p))*problem.energy(bits)
    assert relaxed_binary_energy(problem,s)==pytest.approx(oracle,abs=1e-11)

@pytest.mark.parametrize('backend,device,diff,options',[
    ('numpy','default.qubit','backprop',{}),
    ('numpy','lightning.qubit','adjoint',{}),
    ('torch','default.qubit','backprop',{}),
    ('torch','lightning.qubit','adjoint',{}),
    ('torch','PauliPropagationSimulator','backprop',{'k':None,'prune':False}),
])
@pytest.mark.parametrize('n,m,order',[(2,3,2),(3,7,2),(3,3,3)])
def test_dense_oracle_values_loss_and_gradients(backend,device,diff,options,n,m,order):
    rng=np.random.default_rng(101+m);problem=QUBO(rng.normal(size=(m,m)),offset=.27)
    solver=PCE(problem,n_qubits=n,correlation_order=order,circuit=config(n),backend=backend,
               device=device,diff_method=diff,device_kwargs=options,alpha=1.7,regularization=.31)
    theta=np.linspace(-.53,.71,solver.n_parameters)
    np.testing.assert_allclose(solver.expectations(theta),dense_correlations(theta,n,m,order,'XYZ'),atol=3e-11)
    fn=lambda x:independent_loss(x,solver,order,'XYZ')
    assert solver.loss(theta)==pytest.approx(fn(theta),abs=3e-11)
    np.testing.assert_allclose(solver.gradient(theta),finite_difference(fn,theta),atol=3e-6,rtol=3e-6)
    expected=''.join('0' if c>=0 else '1' for c in dense_correlations(theta,n,m,order,'XYZ'))
    assert solver.decode(theta)==expected

@pytest.mark.parametrize('backend',['tn','van_den_nest','auto'])
def test_iqp_values_and_gradients_against_dense_oracle(backend):
    problem=QUBO.maxcut(3,[(0,1),(0,2),(1,2)])
    solver=PCE(problem,n_qubits=3,paulis='Z',circuit=config(3,True),backend='torch',
               device='IQPSimulator',diff_method='backprop',device_kwargs={'backend':backend,'vn_sampling':'enumerate'})
    theta=np.linspace(-.67,.83,solver.n_parameters)
    fn=lambda x:independent_loss(x,solver,2,'Z',True)
    np.testing.assert_allclose(solver.expectations(theta),dense_correlations(theta,3,3,2,'Z',True),atol=3e-11)
    assert solver.loss(theta)==pytest.approx(fn(theta),abs=3e-11)
    np.testing.assert_allclose(solver.gradient(theta),finite_difference(fn,theta),atol=3e-6,rtol=3e-6)

def test_torch_training_result_and_energy_contract():
    problem=QUBO.maxcut(3,[(0,1),(0,2),(1,2)])
    solver=PCE(problem,circuit=config(2),backend='torch',diff_method='backprop')
    theta=np.linspace(-.53,.71,solver.n_parameters);before=solver.loss(theta)
    result=solver.solve(theta,optimizer='Adam',optimizer_options={'max_steps':30,'lr':.03})
    assert result.objective<before-1e-5
    assert result.objective==pytest.approx(solver.loss(result.parameters))
    assert result.energy==pytest.approx(problem.energy(result.bitstring))
    assert result.metadata['raw_bitstring']==solver.decode(result.parameters)
    assert result.energy<=result.metadata['raw_energy']+1e-12

def test_paper_unweighted_loss_exact_normalization():
    problem=QUBO.maxcut(3,[(0,1),(0,2),(1,2)])
    solver=PCE(problem,circuit=config(2),backend='torch',diff_method='backprop',regularization=.5)
    theta=np.linspace(-.53,.71,solver.n_parameters);s=np.tanh(solver.alpha*dense_correlations(theta,2,3,2,'XYZ'))
    data=sum(s[i]*s[j] for i,j in [(0,1),(0,2),(1,2)])
    nu=3/2+(3-1)/4;paper=data+.5*nu*np.mean(s*s)**2
    assert solver.loss(theta)==pytest.approx(paper,abs=1e-11)

def test_paper_positive_weight_scale_covariance():
    edges=[(0,1,1.),(0,2,2.),(1,2,3.)]
    a=PCE(QUBO.maxcut(3,edges),circuit=config(2),backend='torch')
    b=PCE(QUBO.maxcut(3,[(i,j,10*w) for i,j,w in edges]),circuit=config(2),backend='torch')
    theta=np.linspace(-.53,.71,a.n_parameters)
    assert b.loss(theta)==pytest.approx(10*a.loss(theta),abs=1e-11)

def test_order_four_default_scaling():
    solver=PCE(np.eye(3),n_qubits=4,correlation_order=4)
    assert solver.alpha==4**(4//2)

def test_paper_final_local_improvement_step():
    problem=QUBO.maxcut(3,[(0,1),(0,2),(1,2)])
    solver=PCE(problem,circuit={'ansatz':'AngleAnsatz','ansatz_config':{'n_layers':1,'rotations':'Z'}},
               backend='torch',diff_method='backprop',init='zeros')
    result=solver.solve(optimizer='Adam',optimizer_options={'max_steps':1})
    bits=list(result.bitstring)
    for i in range(len(bits)):
        old=problem.energy(''.join(bits));bits[i]='1' if bits[i]=='0' else '0'
        if problem.energy(''.join(bits))>=old:bits[i]='1' if bits[i]=='0' else '0'
    assert result.energy<=problem.energy(''.join(bits))+1e-12

@pytest.mark.parametrize('name',['alpha','regularization','regularization_scale'])
@pytest.mark.parametrize('value',[float('nan'),float('inf')])
def test_nonfinite_hyperparameters_rejected(name,value):
    with pytest.raises(ValueError):PCE(np.eye(3),**{name:value})


@pytest.mark.parametrize('method', ['mps', 'tn'])
@pytest.mark.parametrize('backend', ['numpy', 'torch'])
def test_pennylane_tensor_values_and_parameter_shift_gradients(method, backend):
    pytest.importorskip('quimb')
    problem = QUBO.maxcut(3, [(0, 1, 1.), (1, 2, 2.), (0, 2, 3.)])
    solver = PCE(problem, circuit=config(2), backend=backend,
                 device='default.tensor', device_kwargs={'method': method},
                 diff_method='parameter-shift')
    theta = np.linspace(-.53, .71, solver.n_parameters)
    fn = lambda x: independent_loss(x, solver, 2, 'XYZ')
    np.testing.assert_allclose(solver.expectations(theta), dense_correlations(theta, 2, 3, 2, 'XYZ'), atol=2e-10)
    assert solver.loss(theta) == pytest.approx(fn(theta), abs=2e-10)
    np.testing.assert_allclose(solver.gradient(theta), finite_difference(fn, theta), atol=3e-6, rtol=3e-6)


def test_sampled_vn_empirical_loss_gradient_and_workers():
    problem = QUBO.maxcut(3, [(0, 1), (1, 2), (0, 2)])
    outputs = []
    for workers in (1, 4):
        solver = PCE(problem, n_qubits=3, paulis='Z', circuit=config(3, True),
                     backend='torch', device='IQPSimulator',
                     device_kwargs={'backend': 'van_den_nest', 'vn_sampling': 'fixed',
                                    'vn_samples': 257, 'vn_batch_size': 64,
                                    'vn_workers': workers, 'vn_parallel_backward': True, 'seed': 19})
        theta = np.linspace(-.67, .83, solver.n_parameters)
        gradient = solver.gradient(theta)
        # A fixed sample bank makes this the derivative of the same empirical loss.
        np.testing.assert_allclose(gradient, finite_difference(solver.loss, theta), atol=3e-6, rtol=3e-6)
        outputs.append((solver.loss(theta), gradient))
    assert outputs[0][0] == pytest.approx(outputs[1][0], abs=1e-12)
    np.testing.assert_allclose(outputs[0][1], outputs[1][1], atol=1e-12)


@pytest.mark.parametrize('n,edges,nu', [
    (3, [(0, 1, 1.), (1, 2, 2.), (0, 2, 3.)], 3.75),
    (5, [(0, 1, 2.), (2, 3, 4.)], 4.5),  # disconnected, plus isolated vertex
    (3, [], 0.),
    (3, [(0, 1, -2.), (1, 2, 4.)], 3.),  # reference-code positive-part extension
])
def test_weighted_regularization_scale(n, edges, nu):
    solver = PCE(QUBO.maxcut(n, edges))
    assert solver.regularization_scale == pytest.approx(nu)


@pytest.mark.parametrize('order', [1, 2, 3, 4, 5])
def test_default_alpha_depends_on_order_and_accepts_override(order):
    solver = PCE(np.eye(3), correlation_order=order)
    assert solver.alpha == solver.n_qubits ** (order // 2)
    assert PCE(np.eye(3), correlation_order=order, alpha=1.25).alpha == 1.25


def test_explicit_scale_offset_invariance_and_zero_problem_gradient():
    p = QUBO.maxcut(3, [(0, 1), (1, 2), (0, 2)])
    a = PCE(p, backend='torch', regularization_scale=7)
    b = PCE(QUBO(p.matrix, offset=99), backend='torch', regularization_scale=7)
    theta = np.linspace(-.53, .71, a.n_parameters)
    assert a.regularization_scale == 7
    assert a.loss(theta) == pytest.approx(b.loss(theta), abs=1e-12)
    np.testing.assert_allclose(a.gradient(theta), b.gradient(theta), atol=1e-12)
    zero = PCE(np.zeros((3, 3)), backend='torch')
    assert zero.loss(theta) == 0
    np.testing.assert_array_equal(zero.gradient(theta), np.zeros_like(theta))


@pytest.mark.parametrize('seed', [3, 17, 41])
def test_single_sweep_matches_naive_full_qubo_recomputation(seed):
    Q = np.random.default_rng(seed).normal(size=(5, 5))
    problem = QUBO(Q, offset=7.2)
    solver = PCE(problem)
    for bits in product('01', repeat=5):
        raw = ''.join(bits)
        reference = raw
        for i in range(5):
            flipped = reference[:i] + ('1' if reference[i] == '0' else '0') + reference[i+1:]
            if problem.energy(flipped) < problem.energy(reference):
                reference = flipped
        actual = solver.improve_bitstring(raw)
        assert actual == reference
        assert problem.energy(actual) <= problem.energy(raw) + 1e-12


def test_postprocessing_can_be_disabled_and_preserves_raw_candidate():
    problem = QUBO.maxcut(3, [(0, 1), (0, 2), (1, 2)])
    solver = PCE(problem, circuit={'ansatz': 'AngleAnsatz', 'ansatz_config': {
        'n_layers': 1, 'rotations': 'Z'}}, backend='torch', init='zeros')
    raw = solver.solve(postprocess=False, optimizer_options={'max_steps': 1})
    improved = solver.solve(postprocess=True, optimizer_options={'max_steps': 1})
    assert raw.bitstring == improved.metadata['raw_bitstring'] == '000'
    assert raw.energy == improved.metadata['raw_energy'] == 0
    assert improved.energy == -2
    assert improved.bitstring == solver.improve_bitstring(raw.bitstring)
    assert raw.objective == improved.objective
    assert not raw.metadata['postprocess'] and improved.metadata['postprocess']
    for invalid in ('0', 'abc', '0000', None):
        with pytest.raises(ValueError): solver.improve_bitstring(invalid)
    with pytest.raises(TypeError, match='boolean'): solver.solve(postprocess='yes')


def test_finite_shot_decode_metadata_uses_the_same_measurements():
    problem = QUBO.maxcut(3, [(0, 1), (0, 2), (1, 2)])
    solver = PCE(problem, backend='torch', shots=500, device_kwargs={'seed': 11}, seed=4)
    assert solver.diff_method == 'parameter-shift'
    theta = solver.initial_parameters()
    assert np.isfinite(solver.gradient(theta)).all()
    result = solver.solve(theta, optimizer_options={'max_steps': 1})
    expected = ''.join('0' if c >= 0 else '1' for c in result.metadata['expectations'])
    assert result.metadata['raw_bitstring'] == expected
    assert result.bitstring == solver.improve_bitstring(expected)
    assert result.energy == problem.energy(result.bitstring)


def test_tikz_retains_all_pce_readout_bases():
    from qasp import draw_pce
    model = PCE(QUBO.maxcut(3, [(0, 1), (0, 2), (1, 2)]))
    tex = draw_pce(model, parameters=np.zeros(model.n_parameters), output='tikz')
    # XX, YY and ZZ require alternative settings; none may overwrite another.
    assert tex.count(r'X\,/\,Y\,/\,Z') == 2
