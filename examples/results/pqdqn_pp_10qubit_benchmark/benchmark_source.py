"""Benchmark a 10-qubit PQDQN: weight-truncated PP versus Lightning adjoint.

Run from the repository root with PYTHONPATH=. and OMP_NUM_THREADS=1.
This measures fixed-parameter workloads, not an RL learning curve. Raw timings,
matched model states, input batches, and errors are saved for reproduction.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from qasp.rl import PQDQN

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'examples/results/pqdqn_pp_10qubit_benchmark'
BACKENDS = ('lightning_qubit', 'pp_k2', 'pp_k3')


def model_config(n_qubits=10):
    return dict(
        n_qubits=n_qubits, n_actions=2, ansatz='AngleAnsatz',
        ansatz_config=dict(n_layers=4, rotations='YZ', reuploading=True,
            include_imprimitive=True, imprimitive='CNOT',
            imprimitive_pattern='ring', include_final_imprimitive=True),
        feature_map='AngleEmbedding',
        feature_map_config=dict(features=n_qubits, rotations='Y', input_scaling=True),
        observables=[f'Z {i}' for i in range(n_qubits)], fc_layer=True,
        fc_bias=True, batch_mode='sequential', input_activation='tanh',
        init='normal_0_1', shots=None,
    )


def make_model(backend, config=None):
    config = model_config() if config is None else config
    if backend == 'lightning_qubit':
        return PQDQN(**config, device='lightning.qubit', diff_method='adjoint').double()
    return PQDQN(**config, device='lightning.pp', diff_method='backprop',
        pp_truncation=int(backend[-1]), pp_topM=None, pp_tau=0.0,
        pp_prune=False).double()


def loss_fn(model, x, actions, targets):
    chosen = model(x).gather(1, actions[:, None]).squeeze(1)
    return F.smooth_l1_loss(chosen, targets)


def flatten_grads(model, prefix=''):
    return torch.cat([
        (p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
        for n, p in model.named_parameters() if n.startswith(prefix)
    ]).detach().clone()


def evaluate(model, x, actions, targets):
    with torch.no_grad():
        features, values = model.quantum_features(x), model(x)
    model.zero_grad(set_to_none=True)
    loss = loss_fn(model, x, actions, targets)
    loss.backward()
    result = dict(features=features, values=values, loss=float(loss.detach()),
        gradients=flatten_grads(model), quantum_gradients=flatten_grads(model, 'qlayer.'))
    assert all(torch.isfinite(v).all() for v in result.values() if isinstance(v, torch.Tensor))
    return result


def deviation(a, b):
    delta = a - b
    norm_b = float(torch.linalg.vector_norm(b))
    norm_a = float(torch.linalg.vector_norm(a))
    return dict(rmse=float(delta.square().mean().sqrt()),
        max_abs=float(delta.abs().max()),
        relative_l2=float(torch.linalg.vector_norm(delta)) / norm_b if norm_b else None,
        reference_norm=norm_b, approximate_norm=norm_a,
        cosine=float(torch.dot(a.flatten(), b.flatten()) / (norm_a * norm_b))
        if norm_a and norm_b else None)


def workload(model, x, actions, targets, kind):
    if kind == 'inference':
        with torch.no_grad():
            model(x)
    else:
        model.zero_grad(set_to_none=True)
        loss_fn(model, x, actions, targets).backward()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--seeds', nargs='+', type=int, default=list(range(7, 12)))
    parser.add_argument('--batches', nargs='+', type=int, default=[1, 8])
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--warmups', type=int, default=2)
    args = parser.parse_args(argv)
    if min(args.batches) < 1 or args.repeats < 2 or args.warmups < 1:
        parser.error('Positive batches, at least two repeats, and one warmup are required.')
    if any(args.output.iterdir()) if args.output.exists() else False:
        parser.error('Output directory is not empty; choose a new directory to preserve results.')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    versions = {name: importlib.metadata.version(name) for name in
                ('torch', 'pennylane', 'pennylane-lightning', 'numpy')}
    try:
        cpu = subprocess.check_output(['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        cpu = platform.processor()
    manifest = dict(config=model_config(), seeds=args.seeds, batches=args.batches,
        repeats=args.repeats, warmups=args.warmups, versions=versions,
        python=platform.python_version(), platform=platform.platform(), cpu=cpu,
        torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
        environment={k: os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        dtype='float64', backend_options={
            'lightning_qubit': dict(device='lightning.qubit', diff_method='adjoint'),
            'pp_k2': dict(device='lightning.pp', diff_method='backprop', pp_truncation=2, pp_prune=False, pp_topM=None, pp_tau=0.0),
            'pp_k3': dict(device='lightning.pp', diff_method='backprop', pp_truncation=3, pp_prune=False, pp_topM=None, pp_tau=0.0)},
        timing='perf_counter_ns; whole PQDQN calls; shuffled backend order within every repeat; no optimizer/environment/target-network work',
        initialization='QASP normal_0_1 = N(0,1) quantum weights/scales; default Torch linear head, then cast to float64',
        inputs='CPU generator seeded 10000+model seed; normal(0,1); tanh inside model; batch 1 is first row of batch 8',
        targets='actions alternate 0,1; targets linearly spaced 0.25 to 1.75; mean SmoothL1Loss(beta=1)',
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    timings, errors, checks, setup = [], [], [], []
    for seed in args.seeds:
        torch.manual_seed(seed)
        models = {}
        for backend in BACKENDS:
            started = time.perf_counter_ns()
            models[backend] = make_model(backend)
            setup.append(dict(seed=seed, backend=backend, milliseconds=(time.perf_counter_ns()-started)/1e6))
        exact = models['lightning_qubit']
        for model in models.values():
            model.load_state_dict(exact.state_dict())
        generator = torch.Generator().manual_seed(10000+seed)
        inputs = torch.randn(max(args.batches), 10, generator=generator, dtype=torch.float64)
        torch.save(dict(state_dict=exact.state_dict(), inputs=inputs), args.output/f'seed_{seed}.pt')
        for batch in args.batches:
            x = inputs[:batch]
            actions = torch.arange(batch) % 2
            targets = torch.linspace(.25, 1.75, batch, dtype=torch.float64)
            values = {b: evaluate(m, x, actions, targets) for b,m in models.items()}
            ref = values['lightning_qubit']
            if seed == args.seeds[0] and batch == args.batches[-1]:
                reference = PQDQN(**model_config(), device='default.qubit', diff_method='backprop').double()
                reference.load_state_dict(exact.state_dict())
                control = evaluate(reference, x, actions, targets)
                for key in ('features','values','gradients'):
                    torch.testing.assert_close(ref[key], control[key], atol=1e-9, rtol=1e-7)
                checks.append(dict(test='lightning_adjoint_vs_default_backprop', passed=True,
                    seed=seed, batch=batch, max_gradient_error=float((ref['gradients']-control['gradients']).abs().max())))
                for b in ('pp_k2','pp_k3'):
                    m = models[b]
                    direction = torch.linspace(-.5,.5,m.qlayer.params.numel(),dtype=torch.float64).reshape_as(m.qlayer.params)
                    original = m.qlayer.params.detach().clone()
                    m.zero_grad(set_to_none=True)
                    loss_fn(m,x,actions,targets).backward()
                    autodiff = float((m.qlayer.params.grad*direction).sum())
                    estimates=[]
                    for sign in (1,-1):
                        with torch.no_grad():
                            m.qlayer.params.copy_(original+sign*1e-5*direction)
                            estimates.append(float(loss_fn(m,x,actions,targets)))
                    with torch.no_grad():
                        m.qlayer.params.copy_(original)
                    fd=(estimates[0]-estimates[1])/2e-5
                    assert abs(fd-autodiff) < 1e-7
                    checks.append(dict(test='truncated_gradient_directional_finite_difference',backend=b,
                        passed=True,autodiff=autodiff,finite_difference=fd,absolute_error=abs(fd-autodiff)))
            for backend in BACKENDS:
                row = dict(seed=seed,batch_size=batch,backend=backend,loss=values[backend]['loss'])
                for key in ('features','values','gradients','quantum_gradients'):
                    row.update({f'{key}_{k}':v for k,v in deviation(values[backend][key],ref[key]).items()})
                errors.append(row)
            rng = random.Random(seed*100+batch)
            for kind in ('inference','forward_backward'):
                for _ in range(args.warmups):
                    for backend in BACKENDS:
                        workload(models[backend],x,actions,targets,kind)
                for repeat in range(args.repeats):
                    order = list(BACKENDS)
                    rng.shuffle(order)
                    for backend in order:
                        started = time.perf_counter_ns()
                        workload(models[backend],x,actions,targets,kind)
                        timings.append(dict(seed=seed,batch_size=batch,backend=backend,
                            workload=kind,repeat=repeat,milliseconds=(time.perf_counter_ns()-started)/1e6))
            pd.DataFrame(timings).to_csv(args.output/'timings.csv',index=False)
            pd.DataFrame(errors).to_csv(args.output/'accuracy.csv',index=False)
            print(f'Completed seed {seed}, batch {batch}',flush=True)
    pd.DataFrame(setup).to_csv(args.output/'setup_times.csv',index=False)
    frame = pd.DataFrame(timings)
    medians = frame.groupby(['seed','batch_size','backend','workload'],as_index=False).milliseconds.median()
    medians.to_csv(args.output/'seed_timing_medians.csv',index=False)
    stats = medians.groupby(['batch_size','backend','workload']).milliseconds.agg(['mean','std','min','max']).reset_index()
    stats.to_csv(args.output/'timing_summary.csv',index=False)
    (args.output/'verification.json').write_text(json.dumps(checks,indent=2))
    manifest['parameter_count'] = sum(p.numel() for p in exact.parameters())
    manifest['parameter_shapes'] = {n:list(p.shape) for n,p in exact.named_parameters()}
    manifest['completed_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(stats.to_string(index=False),flush=True)


if __name__ == '__main__':
    main()
