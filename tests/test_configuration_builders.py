"""Cross-backend value, input-gradient and parameter-gradient audit.

Small exact circuits are intentional: they make independent statevector checks
possible for every tested configuration. No stochastic backend is treated as exact.
"""
import itertools
import pennylane as qml
import pytest
import torch
from qasp.simulators import UnsupportedIQP
from qasp import CircuitBuilder, AngleAnsatz, IQPAnsatz, AngleEmbedding, NeuralEmbedding

DTYPE=torch.float64


@pytest.mark.parametrize('diff_method', ['best', 'adjoint'])
@pytest.mark.parametrize('device_instance', [False, True])
def test_lightning_compilation_preserves_device_and_batched_gradients(diff_method, device_instance):
    builder, features = make_builder('neural_words', True, 'expval')
    reference, _ = make_builder('neural_words', True, 'expval')
    args = arguments(builder, features, True)
    device = qml.device('lightning.qubit', wires=3) if device_instance else 'lightning.qubit'
    node = builder.compile(device=device, diff_method=diff_method)
    if device_instance:
        assert node.device is device
    actual = value(node, args)
    expected = value(reference.compile(device='default.qubit', diff_method='backprop'), args)
    torch.testing.assert_close(actual, expected, atol=3e-9, rtol=3e-9)
    weight = torch.arange(1, actual.numel()+1, dtype=DTYPE).reshape(actual.shape)
    gradients = torch.autograd.grad((actual*weight).sum(), list(args.values()))
    baseline = torch.autograd.grad((expected*weight).sum(), list(args.values()))
    for a, b in zip(gradients, baseline):
        torch.testing.assert_close(a, b, atol=3e-8, rtol=3e-8)

@pytest.fixture(autouse=True)
def double_precision():
    previous=torch.get_default_dtype()
    torch.set_default_dtype(DTYPE)
    yield
    torch.set_default_dtype(previous)

VARIANTS=['angle_y','angle_scaled','angle_words','neural_y','neural_words']

def make_builder(variant, reupload, measurement):
    n=3
    neural=variant.startswith('neural')
    commuting=variant.endswith('_x')
    iqp=variant.startswith('iqp')
    if iqp: neural='neural' in variant
    features=2 if neural else 3
    axis='Z' if iqp else 'X' if commuting else 'Y'
    rotations=[f'{axis} 0',f'{axis*2} 0 1',f'{axis*3} 0 1 2'] if variant.endswith('words') else axis
    fm=(NeuralEmbedding(n,features=features,rotations=rotations) if neural else
        AngleEmbedding(n,features=features,rotations=rotations,input_scaling=variant!='angle_y'))
    if iqp:
        ansatz=IQPAnsatz(n,n_layers=1,feature_map=fm,imprimitive_pattern='all_to_all')
    else:
        ansatz=AngleAnsatz(n,n_layers=2,feature_map=fm,reuploading=reupload,
            rotations=['X 0','X 1','X 2','XX 0 1','XXX 0 1 2'] if commuting else 'ZY',
            include_imprimitive=not commuting,imprimitive='CZ',imprimitive_pattern='all_to_all',
            include_final_imprimitive=False)
    b=CircuitBuilder(n);b.set_ansatz(ansatz)
    if measurement=='probs':b.set_measurement('probs',wires=[0,2])
    else:b.set_measurement('expectation',observables=['Z 0','ZZZ 0 1 2'])
    b.build()
    return b,features

def arguments(builder,features,batch):
    x=torch.linspace(-.43,.61,features*(3 if batch else 1),dtype=DTYPE)
    if batch:x=x.reshape(3,features)
    args={'inputs':x.requires_grad_()}
    shapes=builder.ansatz.get_param_shapes()
    args['params']=torch.linspace(.13,.72,shapes[-1][0],dtype=DTYPE,requires_grad=True)
    if len(shapes)==2:args['input_scaling_params']=torch.linspace(.7,1.1,shapes[0][0],dtype=DTYPE,requires_grad=True)
    return args

def compile_node(builder,backend):
    if backend=='mixed':return builder.compile(device='default.mixed',diff_method='backprop')
    if backend=='pp':return builder.compile(device='PauliPropagationSimulator',diff_method='backprop',device_kwargs={'k':None,'prune':False})
    if backend in ['mps','tn']:
        pytest.importorskip('quimb')
        pytest.importorskip('cotengra')
        return builder.compile(device='default.tensor',method=backend,diff_method='parameter-shift')
    if backend in ['iqp_tn','iqp_vn','iqp_auto']:
        return builder.compile(device='IQPSimulator',diff_method='backprop',device_kwargs={
            'backend':{'iqp_tn':'tn','iqp_vn':'van_den_nest','iqp_auto':'auto'}[backend],
            'vn_sampling':'enumerate','seed':37,'vn_workers':1})
    return builder.compile(device='lightning.qubit' if backend=='lightning' else 'default.qubit',diff_method='adjoint')

def value(node,args):
    y=node(**args)
    return torch.stack(list(y),dim=-1) if isinstance(y,(tuple,list)) else y

CASES=[]
for variant,reupload,measurement,batch in itertools.product(VARIANTS,[False,True],['expval','probs'],[False,True]):
    CASES.extend((variant,reupload,measurement,batch,b) for b in ['pp','mixed'])
    if measurement=='expval':
        CASES.extend((variant,reupload,measurement,batch,b) for b in ['adjoint','lightning'])
        if not batch:CASES.extend((variant,reupload,measurement,batch,b) for b in ['mps','tn'])
for variant,measurement,batch,backend in itertools.product(['angle_x','neural_x','iqp_angle','iqp_neural'],['expval','probs'],[False,True],['iqp_tn','iqp_vn','iqp_auto']):
    if not (measurement=='probs' and backend=='iqp_vn'):
        CASES.append((variant,False,measurement,batch,backend))

@pytest.mark.parametrize('variant,reupload,measurement,batch,backend',CASES)
def test_build_compile_values_and_all_gradients(variant,reupload,measurement,batch,backend):
    builder,features=make_builder(variant,reupload,measurement)
    ref,_=make_builder(variant,reupload,measurement)
    args=arguments(builder,features,batch)
    expected=value(ref.compile(device='default.qubit',diff_method='backprop'),args)
    actual=value(compile_node(builder,backend),args)
    assert actual.shape==expected.shape, f'{backend}: {actual.shape} != {expected.shape}'
    torch.testing.assert_close(actual,expected,atol=3e-9,rtol=3e-9)
    if measurement=='probs':
        torch.testing.assert_close(actual.sum(-1),torch.ones_like(actual.sum(-1)),atol=3e-9,rtol=0)
        assert actual.min()>=-1e-10
    weight=torch.linspace(.3,1.1,actual.numel(),dtype=DTYPE).reshape(actual.shape)
    g=torch.autograd.grad((actual*weight).sum(),list(args.values()))
    gr=torch.autograd.grad((expected*weight).sum(),list(args.values()))
    for a,b in zip(g,gr):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a,b,atol=3e-8,rtol=3e-8)

@pytest.mark.parametrize('variant',VARIANTS+['angle_x','neural_x','iqp_angle','iqp_neural'])
def test_building_block_reference_gradient_finite_difference(variant):
    b,f=make_builder(variant,False,'expval');args=arguments(b,f,False)
    node=b.compile(device='default.qubit',diff_method='backprop')
    actual=torch.autograd.grad(value(node,args).sum(),tuple(args.values()))
    for (name,tensor),grad in zip(args.items(),actual):
        direction=torch.linspace(.3,.6,tensor.numel(),dtype=DTYPE).reshape(tensor.shape)
        plus={**args,name:tensor+1e-5*direction};minus={**args,name:tensor-1e-5*direction}
        fd=(value(node,plus).sum()-value(node,minus).sum())/2e-5
        torch.testing.assert_close(fd,(grad*direction).sum(),atol=2e-8,rtol=2e-8)

@pytest.mark.parametrize('variant,batch',itertools.product(['angle_x','neural_x','iqp_angle','iqp_neural'],[False,True]))
def test_vn_rejects_probability_readouts_explicitly(variant,batch):
    builder,features=make_builder(variant,False,'probs')
    node=compile_node(builder,'iqp_vn')
    with pytest.raises(UnsupportedIQP,match='probabilities'):
        value(node,arguments(builder,features,batch))
