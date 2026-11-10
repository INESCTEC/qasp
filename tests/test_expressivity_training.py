"""Live Fourier spectra before/after optimization and independent width laws."""
import json
import math
import os
from pathlib import Path
import numpy as np
import pytest
import torch
from qasp import create_classifier, create_model, expressivity_analyzer, Normal
from configuration_helpers import BACKENDS, execution


def make_model(kind, embedding, layout, n, backend='statevector', seed=31):
    torch.manual_seed(seed)
    axis = 'Z' if layout == 'iqp' else 'X'
    config = dict(feature_map='NeuralEmbedding' if embedding == 'neural' else 'AngleEmbedding',
        feature_map_config=dict(features=1, rotations=[f'{axis} 0']),
        ansatz='IQPAnsatz' if layout == 'iqp' else 'AngleAnsatz',
        ansatz_config=dict(n_layers=2, reuploading=True), init=Normal(0, .3),
        input_scaling_init='ones', **execution(backend))
    if embedding == 'angle': config.pop('input_scaling_init')
    if embedding == 'scaled': config['feature_map_config']['input_scaling'] = True
    if layout != 'iqp':
        config['ansatz_config'].update(rotations=[f'X {i}' for i in range(n)]+['XX 0 1'], include_imprimitive=False)
    if kind.startswith('classifier'):
        config.update(readout='probabilities' if kind == 'classifier_probs' else 'expectation', observables=['Z 0'])
        if kind == 'classifier_probs': config.pop('observables')
        model = create_classifier(config, n_qubits=n, n_classes=2)
    else:
        config.update(n_actions=2, observables=['Z 0', 'Z 1'], fc_layer=True)
        if kind == 'born':
            config.pop('observables'); config.pop('fc_layer')
            config.update(policy='parity-like', training_method='expectation', parity_cnot_decomposition=False)
        model = create_model('PQCBorn' if kind == 'born' else 'PQCSoftmax', config, n_qubits=n)
    return model.double()


def output(model, x):
    return torch.stack([model(row).reshape(-1) for row in x])


def checked_spectrum(model, x):
    state = {k:v.clone() for k,v in model.state_dict().items()}
    gradients = [None if p.grad is None else p.grad.clone() for p in model.parameters()]
    mode = model.training
    spectrum = expressivity_analyzer(model)
    assert model.training == mode
    for k,v in model.state_dict().items(): torch.testing.assert_close(v, state[k])
    for p,g in zip(model.parameters(), gradients):
        if g is None: assert p.grad is None
        else: torch.testing.assert_close(p.grad, g)
    expected = output(model, x).detach().numpy()
    np.testing.assert_allclose(spectrum.evaluate(x), expected, atol=2e-8, rtol=2e-8)
    assert np.isfinite(spectrum.coefficients).all()
    # Reality requires conjugate symmetry, independently of the chosen readout.
    for f,c in zip(spectrum.frequencies, spectrum.coefficients):
        opposite = np.argmin(np.linalg.norm(spectrum.frequencies + f, axis=1))
        np.testing.assert_allclose(spectrum.frequencies[opposite], -f, atol=1e-12)
        np.testing.assert_allclose(spectrum.coefficients[opposite], c.conj(), atol=2e-9)
    return spectrum


CASES = [(kind, emb, layout, n, 'statevector', seed)
         for kind in ['classifier', 'classifier_probs', 'softmax', 'born']
         for emb in ['angle', 'scaled', 'neural'] for layout in ['angle', 'iqp']
         for n in [2, 4] for seed in [17, 31]]
CASES += [(kind, emb, 'angle', 2, backend, 31)
          for kind in ['classifier', 'softmax', 'born']
          for emb in ['angle', 'scaled', 'neural'] for backend in BACKENDS[1:]]


@pytest.mark.parametrize('kind,embedding,layout,n,backend,seed', CASES)
def test_training_snapshots(kind, embedding, layout, n, backend, seed):
    model = make_model(kind, embedding, layout, n, backend, seed)
    x = torch.tensor([[-.81], [-.19], [.37], [.92]], dtype=torch.float64)
    labels = torch.tensor([0, 0, 1, 1])
    before = checked_spectrum(model, x)
    saved = before.coefficients.copy()
    old_predictions = before.evaluate(x).copy()
    initial_weights = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.Adam(model.parameters(), lr=.035)
    losses=[]
    for _ in range(6):
        optimizer.zero_grad()
        values = output(model, x)
        if kind == 'classifier':
            loss = torch.nn.functional.softplus(-(2*labels-1)*values[:,0]).mean()
        elif kind == 'softmax':
            # Exact expected contextual-bandit reward: correct action receives 1.
            loss = -values.softmax(-1)[torch.arange(len(x)), labels].mean()
        elif kind == 'born':
            loss = -values[torch.arange(len(x)), labels].mean()
        else:
            loss = -values[torch.arange(len(x)), labels].clamp_min(1e-12).log().mean()
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step(); losses.append(float(loss.detach()))
    model.eval()
    after = checked_spectrum(model, x)
    assert any(not torch.equal(a,b) for a,b in zip(initial_weights,model.parameters()))
    assert np.max(abs(after.evaluate(x)-old_predictions)) > 1e-7
    np.testing.assert_array_equal(before.coefficients, saved)  # immutable training snapshot
    if embedding == 'angle': np.testing.assert_array_equal(before.frequencies, after.frequencies)
    else: assert not np.array_equal(before.frequencies, after.frequencies)
    assert np.isfinite(losses).all()
    artifact = os.environ.get('QASP_EXPRESSIVITY_RESULTS')
    if artifact:
        target = Path(artifact); target.mkdir(parents=True, exist_ok=True)
        name=f'{kind}-{embedding}-{layout}-{n}-{backend}-{seed}'
        np.savez_compressed(target/f'{name}.npz', before_frequencies=before.frequencies,
            before_coefficients=before.coefficients, after_frequencies=after.frequencies,
            after_coefficients=after.coefficients, inputs=x.numpy(), predictions_before=old_predictions,
            predictions_after=after.evaluate(x), losses=losses)
        def stats(s):
            mask=np.any(s.frequencies!=0,axis=1)
            return dict(allowed=s.n_frequencies, active=s.active_counts.tolist(),
                        non_dc_power=(abs(s.coefficients[mask])**2).sum(0).tolist())
        (target/f'{name}.json').write_text(json.dumps(dict(kind=kind,embedding=embedding,
            layout=layout,n=n,backend=backend,seed=seed,before=stats(before),after=stats(after),losses=losses)))


def test_width_sweep_closed_form_through_16_qubits():
    def make_classifier(n):
        return create_classifier({
            'feature_map': 'NeuralEmbedding',
            'feature_map_config': {'features': 1, 'rotations': 'Y'},
            'ansatz': 'AngleAnsatz',
            'ansatz_config': {'n_layers': 1, 'rotations': 'Z'},
            'init': 'zeros', 'input_scaling_init': 'ones', 'readout': 'expectation'
        }, n_qubits=n, n_classes=2)
    widths=[2,4,6,8,10,12,14,16]
    sweep=expressivity_analyzer(model_factory=make_classifier, qubit_list=widths,
                               observables=lambda n:['Z 0','Z'*n])
    for n,s in sweep.results_by_qubit.items():
        assert s.n_frequencies==2*n+1
        np.testing.assert_array_equal(s.active_counts,[2,n+1])
        expected={n-2*k:math.comb(n,k)/2**n for k in range(n+1)}
        for f,c in zip(s.frequencies[:,0],s.coefficients):
            np.testing.assert_allclose(c,[.5 if abs(f)==1 else 0,expected.get(round(f),0)],atol=1e-11)
        x=np.array([[-.37],[.81],[1.42]])
        np.testing.assert_allclose(s.evaluate(x),np.column_stack([np.cos(x[:,0]),np.cos(x[:,0])**n]),atol=1e-11)
    artifact=os.environ.get('QASP_EXPRESSIVITY_RESULTS')
    if artifact:
        Path(artifact,'widths.json').write_text(json.dumps(sweep.summary()))
