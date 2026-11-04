"""Initialization distributions, precedence, reproducibility and model integration."""
from copy import deepcopy
import math

import numpy as np
import pennylane  # Load the simulator runtime before Torch on macOS.
import pytest
import torch

from qasp import (
    AngleAnsatz, AngleEmbedding, CircuitBuilder, IQPAnsatz, NeuralEmbedding,
    PCE, QAOA, QUBO, LRQAOA, RecursiveQAOA, WarmStartQAOA,
    barren_plateau_diagnoser, create_classifier, create_kernel, create_qcbm,
)
from qasp.initialization import (
    Choice, Constant, Normal, Ones, Uniform, Zeros,
    initialize_tensor, resolve_initializer, sample_parameters,
)
from qasp.rl import create_model, create_qfunction


def config(ansatz="AngleAnsatz", feature_map="AngleEmbedding"):
    ac = {"n_layers": 2, "reuploading": True}
    if ansatz == "AngleAnsatz":
        ac.update(rotations="YZ", include_imprimitive=True, imprimitive="CNOT",
                  imprimitive_pattern="chain")
    else:
        ac["imprimitive_pattern"] = "chain"
    fc = {"features": 2, "rotations": "Y"}
    if feature_map == "AngleEmbedding":
        fc["input_scaling"] = True
    return dict(ansatz=ansatz, ansatz_config=ac, feature_map=feature_map,
                feature_map_config=fc, init=Normal(0.25, 0),
                input_scaling_init=Uniform(0.8, 1.2),
                device="default.qubit", diff_method="backprop")


def builder(feature_map=AngleEmbedding, scaled=True, ansatz=AngleAnsatz):
    fm = feature_map(n_qubits=2, features=2, rotations="Y",
                     **({"input_scaling": scaled} if feature_map is AngleEmbedding else {}))
    ac = dict(n_qubits=2, n_layers=2, feature_map=fm, reuploading=True)
    if ansatz is AngleAnsatz:
        ac["rotations"] = "YZ"
    b = CircuitBuilder(n_qubits=2)
    b.set_ansatz(ansatz(**ac))
    b.set_measurement("expectation", observables=["Z 0", "Z 1"])
    b.build()
    b.compile(device="default.qubit", diff_method="backprop")
    return b


def assert_groups(params, scales):
    torch.testing.assert_close(params, torch.full_like(params, 0.25))
    assert torch.all((scales >= 0.8) & (scales <= 1.2))
    for p in (params, scales):
        assert p.requires_grad and p.is_leaf


def assert_gradients(loss, parameters):
    params = list(parameters)
    gradients = torch.autograd.grad(loss, params)
    assert all(torch.isfinite(g).all() for g in gradients)
    assert all(g.abs().sum() > 1e-9 for g in gradients)


@pytest.mark.parametrize("spec", [Normal(1.7, 0.3), Uniform(-2.3, 0.7), Constant(-0.4),
                                   Zeros(), Ones(), Choice([-1, 2, 4])])
def test_rng_shape_dtype_and_finite_values(spec):
    a = spec.sample((3, 4), rng=np.random.default_rng(17))
    b = spec.sample((3, 4), rng=np.random.default_rng(17))
    np.testing.assert_array_equal(a, b)
    tensor = torch.empty((3, 4), dtype=torch.float64, requires_grad=True)
    result = spec(tensor, generator=torch.Generator().manual_seed(17))
    assert result is tensor and tensor.dtype == torch.float64 and tensor.is_leaf
    duplicate = spec(torch.empty_like(tensor), generator=torch.Generator().manual_seed(17))
    torch.testing.assert_close(tensor, duplicate)
    assert a.shape == (3, 4) and np.isfinite(a).all()
    assert spec.sample((0, 3)).shape == (0, 3)
    assert deepcopy(spec) == spec


def test_distribution_parameters_have_the_requested_mean_spread_and_support():
    spec = Normal(1.3, 0.7)
    values = spec.sample(100_000, rng=np.random.default_rng(4))
    assert abs(values.mean() - 1.3) < 0.01
    assert abs(values.std() - 0.7) < 0.01
    tensor = spec(torch.empty(100_000), generator=torch.Generator().manual_seed(4))
    assert abs(tensor.mean().item() - 1.3) < 0.01
    assert abs(tensor.std().item() - 0.7) < 0.01
    values = Uniform(-0.7, 1.9).sample(100_000, rng=np.random.default_rng(5))
    assert values.min() >= -0.7 and values.max() < 1.9
    assert abs(values.mean() - 0.6) < 0.01
    assert set(Choice([-2, 3]).sample(1000)) == {-2, 3}


@pytest.mark.parametrize("factory", [lambda: Normal(0, -1), lambda: Normal(float('nan'), 1),
                                      lambda: Uniform(2, 1), lambda: Uniform(0, float('inf')),
                                      lambda: Constant(float('nan')), lambda: Choice([])])
def test_invalid_distributions_fail_at_configuration_time(factory):
    with pytest.raises(ValueError):
        factory()


def test_custom_callables_and_validation():
    def fill(tensor):
        tensor.fill_(0.4)
    assert np.all(sample_parameters((2, 3), fill) == 0.4)
    with pytest.raises(ValueError, match="shape"):
        initialize_tensor(torch.empty(3), lambda t: torch.ones(2))
    with pytest.raises(ValueError, match="finite"):
        initialize_tensor(torch.empty(3), lambda t: t.fill_(float('nan')))
    with pytest.raises(ValueError, match="Unknown initialization"):
        resolve_initializer("normla_0_1")
    with pytest.raises(TypeError):
        Normal()(torch.empty(3, dtype=torch.int64))
    with pytest.raises(ValueError, match="shape"):
        Uniform().sample((-1,))


@pytest.mark.parametrize("name,spec", [
    ("normal_0_1", Normal()), ("small_normal", Normal(0, .1)),
    ("normal_half_0_15", Normal(.5, .15)), ("random_-pi_pi", Uniform(-math.pi, math.pi)),
    ("random_0_2pi", Uniform()), ("uniform", Uniform()),
    ("random_-1_1", Uniform(-1, 1)), ("random_0_1", Uniform(0, 1)),
    ("random_half_1", Uniform(.5, 1)), ("random_0_pi", Uniform(0, math.pi)),
    ("constant", Constant(math.pi)), ("zeros", Zeros()), ("ones", Ones()),
])
def test_legacy_aliases_match_explicit_distributions(name, spec):
    assert resolve_initializer(name) == spec
    torch.manual_seed(22)
    a = builder().create_torchlayer(initialization=name)
    torch.manual_seed(22)
    b = builder().create_torchlayer(initialization=spec)
    for key in a.state_dict():
        torch.testing.assert_close(a.state_dict()[key], b.state_dict()[key])


def test_legacy_discrete_alias_keeps_historical_builder_and_diagnostic_meanings():
    assert resolve_initializer("discrete_0_pi_over_2", legacy_discrete_pi=True) == Choice((0, math.pi))
    assert resolve_initializer("discrete_0_pi_over_2") == Choice((0, math.pi / 2))


@pytest.mark.parametrize("fm", [AngleEmbedding, NeuralEmbedding])
@pytest.mark.parametrize("ansatz", [AngleAnsatz, IQPAnsatz])
def test_manual_builder_separates_all_reuploading_scale_parameters(fm, ansatz):
    b = builder(fm, ansatz=ansatz)
    layer = b.create_torchlayer(initialization=Normal(.25, 0), input_scaling_init=Uniform(.8, 1.2))
    assert_groups(layer.params, layer.input_scaling_params)
    assert layer.input_scaling_params.numel() == (4 if fm is AngleEmbedding else 8)
    out = layer(torch.tensor([.3, -.4]))
    assert_gradients(out.square().sum(), layer.parameters())


def test_independent_explicit_weights_override_only_their_group():
    b = builder()
    n_scales, n_params = b.ansatz.get_param_shapes()
    layer = b.create_torchlayer(params=torch.full(n_params, .6),
                                initialization=Zeros(), input_scaling_init=Constant(1.5))
    assert torch.all(layer.params == .6) and torch.all(layer.input_scaling_params == 1.5)
    layer = b.create_torchlayer(input_scaling_params=torch.full(n_scales, 2.),
                                initialization=Constant(.3), input_scaling_init=Zeros())
    assert torch.all(layer.params == .3) and torch.all(layer.input_scaling_params == 2)
    identity = id(layer.params)
    b.set_weights(input_scaling_params=torch.ones(n_scales))
    assert id(layer.params) == identity and torch.all(layer.params == .3)
    with pytest.raises(ValueError, match="shape"):
        b.create_torchlayer(params=[0])
    with pytest.raises(ValueError, match="input-scaling"):
        builder(scaled=False).create_torchlayer(input_scaling_init=Ones())


@pytest.mark.parametrize("kind", ["classifier_born", "classifier_expectation", "born_policy",
                                  "softmax_policy", "pqdqn", "fidelity", "localized", "projected"])
def test_public_factories_use_both_initializers_and_preserve_gradients(kind):
    cfg = config()
    original = deepcopy(cfg)
    if kind.startswith("classifier"):
        cfg.update(readout="probabilities" if kind.endswith("born") else "expectation")
        if kind.endswith("expectation"):
            cfg["observables"] = ["Z 0", "Z 1"]
        model = create_classifier(cfg, 2, 2)
    elif kind in {"born_policy", "softmax_policy"}:
        cfg["n_actions"] = 2
        if kind == "born_policy":
            cfg.update(policy="parity-like", parity_cnot_decomposition=False)
        else:
            cfg.update(observables=["Z 0", "Z 1"], fc_layer=True)
        model = create_model("PQCBorn" if kind == "born_policy" else "PQCSoftmax", cfg, 2)
    elif kind == "pqdqn":
        model = create_qfunction({**cfg, "observables": ["Z 0", "Z 1"]}, n_qubits=2, n_actions=2)
    else:
        cfg["kernel_type"] = "projected" if kind == "projected" else "fidelity"
        if kind == "localized":
            cfg["projectors"] = ["P 0"]
        model = create_kernel(cfg, 2)
    assert cfg["init"] == original["init"] and cfg["input_scaling_init"] == original["input_scaling_init"]
    assert_groups(model.qlayer.params, model.qlayer.input_scaling_params)
    if kind in {"fidelity", "localized", "projected"}:
        output = model.gram([[.2, -.3], [.7, .4]], return_tensor=True)[0, 1]
    else:
        output = model(torch.tensor([.2, -.3])).reshape(-1)[0]
    assert_gradients(output, model.qlayer.parameters())


@pytest.mark.parametrize("interface", ["torch", "autograd"])
def test_born_machine_seeded_distributions_explicit_precedence_and_training(interface):
    cfg = {**config(), "interface": interface, "inputs": [.2, -.3], "seed": 11,
           "init": Normal(.2, .1), "lr": .01}
    data = np.array([[0, 0], [1, 1]])
    a = create_qcbm(cfg, 2, data)
    b = create_qcbm(cfg, 2, data)
    def array(v):
        return v.detach().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
    np.testing.assert_array_equal(array(a.params), array(b.params))
    np.testing.assert_array_equal(array(a.input_scaling_params), array(b.input_scaling_params))
    before = array(a.params).copy()
    _, history = a.train(1)
    assert np.isfinite(history).all() and not np.array_equal(array(a.params), before)
    override = create_qcbm({**cfg, "init_params": np.full(before.shape, .7),
                           "input_scaling_params": np.full(array(b.input_scaling_params).shape, 1.1)}, 2, data)
    np.testing.assert_allclose(array(override.params), .7)
    np.testing.assert_allclose(array(override.input_scaling_params), 1.1)
    default_scales = create_qcbm({k: v for k, v in cfg.items() if k != "input_scaling_init"}, 2, data)
    np.testing.assert_allclose(array(default_scales.input_scaling_params), 1.)


@pytest.mark.parametrize("feature_map", ["AngleEmbedding", "NeuralEmbedding"])
def test_encoding_only_kernel_can_sample_fixed_scales_and_explicit_values_win(feature_map):
    cfg = config(feature_map=feature_map)
    cfg.pop("ansatz"); cfg.pop("ansatz_config")
    cfg["input_scaling_init"] = Constant(1.3)
    kernel = create_kernel(cfg, 2)
    np.testing.assert_allclose(kernel._fixed_isp, 1.3)
    assert kernel.parameters() == []
    assert np.isfinite(kernel.gram([[.2, -.3], [.5, .8]])).all()
    cfg["input_scaling_params"] = np.full(kernel._fixed_isp.shape, .9)
    explicit = create_kernel(cfg, 2)
    np.testing.assert_allclose(explicit._fixed_isp, .9)


@pytest.mark.parametrize("explicit", [False, True])
def test_encoding_kernel_preserves_inferred_feature_count_for_map_instances(explicit):
    fm = AngleEmbedding(n_qubits=2, rotations="Y", input_scaling=True)
    cfg = dict(feature_map=fm, input_scaling_init=Constant(1.3), device="default.qubit")
    if explicit:
        cfg["input_scaling_params"] = [.8, 1.2]
    kernel = create_kernel(cfg, 2)
    assert kernel._fixed_isp.shape == (2,)
    np.testing.assert_allclose(np.diag(kernel.gram([[.2, -.3], [.5, .8]])), 1.)


@pytest.mark.parametrize("family", [QAOA, WarmStartQAOA, PCE])
@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_optimization_creation_seeding_and_solve_override(family, backend):
    problem = QUBO.maxcut(3, [(0, 1), (1, 2), (0, 2)])
    kwargs = {"backend": backend, "seed": 7, "init": Uniform(.15, .35)}
    if family is WarmStartQAOA:
        kwargs["warm_start"] = [.2, .6, .7]
    a, b = family(problem, **kwargs), family(problem, **kwargs)
    initial = a.initial_parameters()
    np.testing.assert_array_equal(initial, b.initial_parameters())
    assert np.all((initial >= .15) & (initial <= .35))
    np.testing.assert_allclose(a.initial_parameters(Constant(.4)), .4)
    if backend == "numpy":
        result = a.solve(initial_parameters=Constant(.3), optimizer=lambda objective, x, **kw: x)
        np.testing.assert_allclose(result.parameters, .3)
    else:
        result = a.solve(initial_parameters=Normal(.3, 0), optimizer="Adam",
                         optimizer_options={"max_steps": 1, "lr": .01})
    assert np.isfinite(result.parameters).all() and np.isfinite(result.energy)


def test_recursive_qaoa_and_fixed_ramp_contracts():
    problem = QUBO.maxcut(3, [(0, 1), (1, 2)])
    solver = RecursiveQAOA(problem, cutoff=2, init=Constant(.2),
                           optimizer=lambda objective, x, **kw: x)
    result = solver.solve()
    assert len(solver.step_results) == 1 and np.isfinite(result.energy)
    np.testing.assert_allclose(solver.step_results[0].parameters, .2)
    ramp = LRQAOA(problem, n_layers=2)
    np.testing.assert_allclose(ramp.solve().parameters.reshape(2, 2), ramp.ramp_parameters)
    with pytest.raises(ValueError, match="fixed ramp"):
        LRQAOA(problem, init=Normal())


def test_diagnostic_uses_local_seed_separate_scales_and_restores_original_parameters():
    model = create_classifier(config(), 2, 2)
    snapshot = {k: v.clone() for k, v in model.qlayer.state_dict().items()}
    seen = []
    def cost(model, state):
        assert torch.all(model.qlayer.input_scaling_params == 1.7)
        seen.append(model.qlayer.params.detach().clone())
        return model(state).reshape(-1)[0]
    kwargs = dict(n_samples=4, initialization=Normal(.3, .2), input_scaling_init=Constant(1.7),
                  state=torch.tensor([.2, .4]), cost_function=cost, seed=21, return_samples=True)
    rng = torch.get_rng_state().clone()
    a = barren_plateau_diagnoser(model, **kwargs)
    b = barren_plateau_diagnoser(model, **kwargs)
    assert a["variance"] == b["variance"]
    assert not torch.equal(seen[0], seen[1])
    torch.testing.assert_close(seen[0], seen[4])
    assert torch.equal(torch.get_rng_state(), rng)
    for key, value in snapshot.items():
        torch.testing.assert_close(model.qlayer.state_dict()[key], value)


@pytest.mark.parametrize("kind", ["classifier", "policy", "qfunction", "born_machine"])
def test_shared_initializers_with_differentiable_pauli_propagation(kind):
    cfg = config()
    cfg.update(device="lightning.pp", pp_truncation=None, pp_topM=None, pp_tau=0, pp_prune=False)
    if kind == "born_machine":
        cfg.update(inputs=[.2, -.3], loss="mmd", mmd_config={"mode": "exact"})
        model = create_qcbm(cfg, 2, [[0, 0], [1, 1]])
        assert_groups(model.params, model.input_scaling_params)
        assert_gradients(model.loss(), model.parameters())
        return
    cfg["observables"] = ["Z 0", "Z 1"]
    if kind == "classifier":
        cfg["readout"] = "expectation"
        model = create_classifier(cfg, 2, 2)
    elif kind == "policy":
        cfg.update(n_actions=2, fc_layer=True)
        model = create_model("PQCSoftmax", cfg, 2)
    else:
        model = create_qfunction(cfg, n_qubits=2, n_actions=2)
    assert_groups(model.qlayer.params, model.qlayer.input_scaling_params)
    assert_gradients(model(torch.tensor([.2, -.3])).reshape(-1)[0], model.qlayer.parameters())


def test_legacy_variance_worker_reinitializes_from_shared_objects():
    from qasp.rl.metrics import _sample_worker_batch
    cfg = config()
    cfg.update(n_actions=2, policy="parity-like", parity_cnot_decomposition=False,
               init=Constant(.3), input_scaling_init=Constant(1.4))
    values = _sample_worker_batch("PQCBorn", cfg, 2, action=0, batch_size=3,
                                  state=torch.tensor([.2, -.3]))
    assert len(values) == 3 and np.isfinite(values).all()
    np.testing.assert_allclose(values, values[0])


def test_default_distributions_and_pce_dictionary_override():
    b = builder()
    torch.manual_seed(3)
    layer = b.create_torchlayer()
    assert torch.all((layer.params >= 0) & (layer.params < 2 * math.pi))
    problem = QUBO.maxcut(3, [(0, 1), (1, 2)])
    qaoa = QAOA(problem, seed=9)
    np.testing.assert_array_equal(qaoa.initial_parameters(),
                                  np.random.default_rng(9).uniform(-.1, .1, qaoa.n_parameters))
    pce = PCE(problem, seed=9)
    np.testing.assert_array_equal(pce.initial_parameters(),
                                  np.random.default_rng(9).uniform(-math.pi, math.pi, pce.n_parameters))
    cfg = pce._default_circuit()
    cfg["init"] = Constant(.7)
    np.testing.assert_allclose(PCE(problem, circuit=cfg).initial_parameters(), .7)
    np.testing.assert_allclose(PCE(problem, circuit=cfg, init=Constant(.4)).initial_parameters(), .4)
    cfg["init"] = None
    np.testing.assert_array_equal(
        PCE(problem, circuit=cfg, seed=9).initial_parameters(),
        np.random.default_rng(9).uniform(-math.pi, math.pi, pce.n_parameters),
    )


def test_torch_global_seeding_and_state_randomization_with_distribution_objects():
    cfg = config()
    cfg["init"] = Normal(.2, .1)
    torch.manual_seed(19)
    a = create_classifier(cfg, 2, 2)
    torch.manual_seed(19)
    b = create_classifier(cfg, 2, 2)
    for name in a.qlayer.state_dict():
        torch.testing.assert_close(a.qlayer.state_dict()[name], b.qlayer.state_dict()[name])
    result = barren_plateau_diagnoser(a, n_samples=3, initialization=Uniform(-.4, .5),
                                      input_scaling_init=Ones(), state_initialization=Normal(.2, .1), seed=8)
    assert np.isfinite(result["variance"])
    assert result["initialization"] == "Uniform(low=-0.4, high=0.5)"
    assert result["state_initialization"] == "Normal(mu=0.2, std=0.1)"
