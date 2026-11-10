"""Distribution moments, bounds, RNG isolation and model compatibility."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import math

import numpy as np
import pennylane as qml
import pytest
import torch

import qasp
from qasp.initialization import (
    Beta, TruncatedNormal, LogNormal, Gamma, Exponential, Laplace, Rademacher,
)


SPECS = [Beta(2, 5, -.4, .6), TruncatedNormal(.1, .3, -.4, .6),
         LogNormal(-.02, .2), Gamma(4, 4), Exponential(2),
         Laplace(.1, .2), Rademacher(.2)]


def draw(spec, backend, size=80_000):
    if backend == "numpy":
        return spec.sample(size, rng=np.random.default_rng(13))
    tensor = torch.empty(size, dtype=torch.float64)
    return spec(tensor, generator=torch.Generator().manual_seed(13)).numpy()


@pytest.mark.parametrize("spec", SPECS)
def test_public_exports_and_immutable_reusable_specs(spec):
    name = type(spec).__name__
    assert getattr(qasp, name) is type(spec) and name in qasp.__all__
    assert deepcopy(spec) == spec
    field = next(iter(spec.__dataclass_fields__))
    with pytest.raises(FrozenInstanceError):
        setattr(spec, field, 1.5)


@pytest.mark.parametrize("spec", SPECS)
def test_seeded_sampling_preserves_global_rngs_and_tensor_registration(spec):
    np.random.seed(82)
    expected_numpy = np.random.random(4)
    np.random.seed(82)
    torch.manual_seed(82)
    torch_state = torch.get_rng_state().clone()
    a = spec.sample((4, 3), rng=np.random.default_rng(15))
    b = spec.sample((4, 3), rng=np.random.default_rng(15))
    np.testing.assert_array_equal(a, b)
    # A noncontiguous view tests in-place semantics, not just a fresh allocation.
    base = torch.zeros((3, 4), dtype=torch.float64)
    tensor = base.T.requires_grad_()
    result = spec(tensor, generator=torch.Generator().manual_seed(15))
    assert result is tensor and tensor.is_leaf and tensor.requires_grad
    assert tensor.dtype == torch.float64 and tensor.device.type == "cpu"
    other = spec(torch.empty_like(tensor), generator=torch.Generator().manual_seed(15))
    torch.testing.assert_close(tensor, other)
    assert torch.isfinite(tensor).all() and np.isfinite(a).all()
    assert torch.equal(torch.get_rng_state(), torch_state)
    np.testing.assert_array_equal(np.random.random(4), expected_numpy)


@pytest.mark.parametrize("spec", SPECS)
def test_global_seeds_scalar_empty_and_low_precision_shapes(spec):
    np.random.seed(22)
    a = spec.sample((8,))
    np.random.seed(22)
    np.testing.assert_array_equal(a, spec.sample((8,)))
    torch.manual_seed(22)
    a = spec(torch.empty(8))
    torch.manual_seed(22)
    torch.testing.assert_close(a, spec(torch.empty(8)))
    assert spec.sample(()).shape == ()
    assert spec.sample((0, 2)).shape == (0, 2)
    for dtype in (torch.float16, torch.float32, torch.float64):
        assert spec(torch.empty((), dtype=dtype)).shape == ()
        assert spec(torch.empty((0, 2), dtype=dtype)).shape == (0, 2)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_beta_moments_skew_and_rescaling(backend):
    values = draw(Beta(2, 5, -2, 3), backend)
    assert values.min() >= -2 and values.max() <= 3
    assert abs(values.mean() - (-2 + 5 * 2 / 7)) < .015
    assert abs(values.var() - 25 * 2 * 5 / (7**2 * 8)) < .015
    endpoints = draw(Beta(.3, .3), backend)
    centered = draw(Beta(5, 5), backend)
    assert np.mean((endpoints < .1) | (endpoints > .9)) > .5
    assert np.mean((centered < .1) | (centered > .9)) < .01


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_truncated_normal_conditions_instead_of_clipping_and_handles_far_tails(backend):
    values = draw(TruncatedNormal(0, 1, -1, 1), backend)
    assert values.min() > -1 and values.max() < 1
    assert abs(values.mean()) < .01
    assert abs(values.var() - .291125) < .01
    assert not np.any((values == -1) | (values == 1))
    tail = draw(TruncatedNormal(0, 1, 9, 10), backend, size=2000)
    assert np.isfinite(tail).all() and np.all((tail > 9) & (tail < 10))
    assert np.std(tail) > .03


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("spec,mean,variance", [
    (LogNormal(-.02, .2), 1., math.expm1(.04)),
    (Gamma(4, 2), 2., 1.), (Exponential(2), .5, .25),
    (Laplace(.3, .2), .3, .08), (Rademacher(.2), 0., .04),
])
def test_positive_and_symmetric_distribution_moments(spec, mean, variance, backend):
    values = draw(spec, backend)
    assert abs(values.mean() - mean) < .015
    assert abs(values.var() - variance) < .025
    if isinstance(spec, (LogNormal, Gamma, Exponential)):
        assert values.min() > 0
    if isinstance(spec, Rademacher):
        assert set(values) == {-.2, .2}


@pytest.mark.parametrize("factory", [
    lambda: Beta(0, 1), lambda: Beta(1, -1), lambda: Beta(1, 1, 2, 2),
    lambda: Beta(float("nan"), 1), lambda: Beta(1, 1, 0, float("inf")),
    lambda: TruncatedNormal(0, 0), lambda: TruncatedNormal(0, -1),
    lambda: TruncatedNormal(0, 1, 2, 1), lambda: TruncatedNormal(float("inf")),
    lambda: LogNormal(0, -1), lambda: Gamma(0, 1), lambda: Gamma(1, 0),
    lambda: Exponential(-1), lambda: Laplace(0, 0), lambda: Rademacher(0),
    lambda: Gamma(True, 1), lambda: Laplace(0, float("nan")),
])
def test_invalid_distribution_parameters_are_rejected_early(factory):
    with pytest.raises(ValueError):
        factory()


def configuration():
    return dict(
        ansatz="AngleAnsatz",
        ansatz_config=dict(n_layers=2, rotations="YZ", reuploading=True),
        feature_map="AngleEmbedding",
        feature_map_config=dict(features=2, rotations="Y", input_scaling=True),
        init=TruncatedNormal(.2, .1, .05, .4),
        input_scaling_init=Beta(3, 3, .8, 1.2),
        device="default.qubit", diff_method="backprop",
    )


@pytest.mark.parametrize("kind", ["classifier", "policy", "pqdqn", "kernel", "pqcbm"])
def test_bounded_initializers_work_in_both_parameter_groups_with_gradients(kind):
    from qasp.rl import create_policy, create_qfunction

    cfg = configuration()
    if kind == "classifier":
        model = qasp.create_classifier(cfg, 2, 2)
    elif kind == "policy":
        cfg.update(n_actions=2, observables=["Z 0", "Z 1"])
        model = create_policy("PQCSoftmax", cfg, 2)
    elif kind == "pqdqn":
        model = create_qfunction({**cfg, "observables": ["Z 0", "Z 1"]}, 2, 2)
    elif kind == "kernel":
        model = qasp.create_kernel(cfg, 2)
    else:
        model = qasp.create_qcbm({**cfg, "inputs": [.2, -.3], "seed": 7}, 2, [[0, 0], [1, 1]])
    layer = model if kind == "pqcbm" else model.qlayer
    assert torch.all((layer.params >= .05) & (layer.params <= .4))
    assert torch.all((layer.input_scaling_params >= .8) & (layer.input_scaling_params <= 1.2))
    if kind == "pqcbm":
        loss = model.loss()
    elif kind == "kernel":
        loss = model.gram([[.2, -.3], [.7, .4]], return_tensor=True)[0, 1]
    else:
        loss = model(torch.tensor([.2, -.3])).reshape(-1)[0]
    gradients = torch.autograd.grad(loss, [layer.params, layer.input_scaling_params])
    assert all(torch.isfinite(g).all() and g.abs().sum() > 1e-9 for g in gradients)


@pytest.mark.parametrize("spec", SPECS)
def test_new_initializers_reach_seeded_numpy_models_and_local_diagnostics(spec):
    problem = qasp.QUBO.maxcut(2, [(0, 1)])
    for family in (qasp.QAOA, qasp.PCE):
        a, b = family(problem, seed=4, init=spec), family(problem, seed=4, init=spec)
        np.testing.assert_array_equal(a.initial_parameters(), b.initial_parameters())
        result = a.solve(optimizer=lambda objective, x, **kwargs: x)
        assert np.isfinite(result.energy)
    cfg = configuration()
    cfg.update(interface="autograd", init=spec, input_scaling_init=spec, inputs=[.2, -.3], seed=5)
    a = qasp.create_qcbm(cfg, 2, [[0, 0], [1, 1]])
    b = qasp.create_qcbm(cfg, 2, [[0, 0], [1, 1]])
    np.testing.assert_array_equal(a.params, b.params)
    np.testing.assert_array_equal(a.input_scaling_params, b.input_scaling_params)
    assert np.isfinite(a.loss())
    model = qasp.create_classifier(configuration(), 2, 2)
    snapshot = {k: v.clone() for k, v in model.qlayer.state_dict().items()}
    kwargs = dict(n_samples=3, initialization=spec, input_scaling_init=spec,
                  state=[.2, -.3], seed=8, return_samples=True)
    a = qasp.barren_plateau_diagnoser(model, **kwargs)
    b = qasp.barren_plateau_diagnoser(model, **kwargs)
    assert a["variance"] == b["variance"] and np.isfinite(a["variance"])
    for key, value in snapshot.items():
        torch.testing.assert_close(model.qlayer.state_dict()[key], value)


def test_custom_numpy_only_spec_uses_generator_aware_torch_bridge():
    class Custom(qasp.Initializer):
        def _sample(self, shape, rng):
            return rng.uniform(.3, .4, size=shape)

    tensor = Custom()(torch.empty(10), generator=torch.Generator().manual_seed(1))
    assert torch.all((tensor >= .3) & (tensor <= .4))
