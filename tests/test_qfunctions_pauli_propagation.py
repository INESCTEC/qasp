"""Verify real PP execution, gradients, readouts and DQN integration."""

from copy import deepcopy

import numpy as np
import pytest
import torch

from qasp.rl import PQDQN, DeepQLearning


def pp_config(**overrides):
    return {
        "n_qubits": 3,
        "n_actions": 2,
        "ansatz": "AngleAnsatz",
        "ansatz_config": {
            "n_layers": 2,
            "rotations": "YZ",
            "reuploading": True,
            "include_imprimitive": True,
            "imprimitive": "CNOT",
            "imprimitive_pattern": "ring",
            "include_final_imprimitive": True,
        },
        "feature_map": "AngleEmbedding",
        "feature_map_config": {"features": 3, "rotations": "Y", "input_scaling": True},
        "observables": ["Z 0", "X 1", "-YZ 1 2", "I 0"],
        "device": "lightning.pp",
        "diff_method": "backprop",
        "batch_mode": "broadcast",
        "pp_truncation": None,
        "pp_topM": None,
        "pp_tau": 0.0,
        "pp_prune": False,
    } | overrides


@pytest.mark.parametrize(
    "readout",
    [
        {"observables": ["Z 0", "X 1", "-YZ 1 2", "I 0"]},
        {"observables": [["I 0", "Z 0"], ["-Z 1", "ZZ 0 2"]]},
        {"observables": "parity", "preset_actions": 2},
        {"observables": "parity", "preset_actions": 4},
        {"observables": "majority", "preset_actions": 2},
        {
            "observables": "majority",
            "preset_actions": 4,
            "majority_blocks": [[0, 1], [2]],
        },
    ],
)
def test_pp_values_and_all_gradients_match_statevector(readout, monkeypatch):
    torch.manual_seed(19)
    pp = PQDQN(**pp_config(**readout)).double()
    exact = PQDQN(**pp_config(**readout, device="default.qubit")).double()
    exact.load_state_dict(pp.state_dict())

    # Never execute the parameter-owning QNode as a hidden fallback.
    def forbidden(*args, **kwargs):
        raise AssertionError("PP must not execute the statevector TorchLayer")

    monkeypatch.setattr(pp.qlayer, "forward", forbidden)
    x = torch.tensor(
        [[[0.2, -0.4, 0.1], [0.3, 0.1, -0.2]]], dtype=torch.float64, requires_grad=True
    )
    y = x.detach().clone().requires_grad_(True)
    a, b = pp(x), exact(y)
    assert a.shape == (1, 2, 2)
    torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-9)
    a.square().sum().backward()
    b.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=1e-7, atol=1e-9)
    for (name, p), (_, q) in zip(pp.named_parameters(), exact.named_parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-7, atol=1e-9)
    assert pp.qlayer.params.grad.abs().sum() > 0
    assert pp.qlayer.input_scaling_params.grad.abs().sum() > 0
    assert pp(torch.empty(0, 3, dtype=torch.float64)).shape == (0, 2)
    assert pp(torch.zeros(3, dtype=torch.float64)).shape == (2,)


@pytest.mark.parametrize("kind", ["neural", "iqp"])
def test_pp_existing_feature_maps_and_ansatz(kind):
    cfg = pp_config(
        n_qubits=2,
        observables=["Z 0", "ZZ 0 1"],
        feature_map_config={"features": 2, "rotations": "Y", "input_scaling": True},
        ansatz_config={"n_layers": 1},
    )
    if kind == "neural":
        cfg["feature_map"] = "NeuralEmbedding"
        cfg["feature_map_config"].pop("input_scaling")
    else:
        cfg["ansatz"] = "IQPAnsatz"
    pp = PQDQN(**cfg).double()
    exact = PQDQN(
        **(cfg | {"device": "default.qubit", "batch_mode": "sequential"})
    ).double()
    exact.load_state_dict(pp.state_dict())
    x = torch.tensor([[0.23, -0.41]], dtype=torch.float64)
    a, b = pp(x), exact(x)
    torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-8)
    a.sum().backward()
    b.sum().backward()
    for p, q in zip(pp.parameters(), exact.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-8, rtol=1e-7)


def test_pp_truncation_controls_and_cartpole_training(tmp_path):
    torch.manual_seed(3)
    model = PQDQN(
        **pp_config(
            n_qubits=4,
            feature_map_config={"features": 4, "rotations": "Y", "input_scaling": True},
            ansatz_config={"n_layers": 1, "rotations": "Y"},
            observables=["Z 0", "Z 1"],
            pp_truncation=2,
            pp_topM=8,
            pp_tau=1e-10,
            pp_prune_every=1,
            pp_prune=True,
        )
    )
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    assert model.pp_engine.topM == 8 and model.pp_engine.prune_every == 1
    with DeepQLearning(
        model,
        "CartPole-v1",
        seed=3,
        batch_size=2,
        learning_starts=2,
        max_steps=4,
        target_update_interval=1000,
    ) as trainer:
        target = deepcopy(trainer.target_network.state_dict())
        history = trainer.train(2, log_every=0)
        assert trainer.update_count > 0
        assert all(np.isfinite(row["loss"]) for row in history)
        for n, p in model.named_parameters():
            assert torch.isfinite(p.grad).all()
            assert not torch.equal(before[n], p), n
        for n, p in trainer.target_network.state_dict().items():
            torch.testing.assert_close(p, target[n])
        assert all(p.grad is None for p in trainer.target_network.parameters())
        trainer.sync_target()
        probe = torch.tensor([[0.2, 0.1, -0.2, 0.3]])
        torch.testing.assert_close(trainer.target_network(probe), model(probe))
        assert len(trainer.evaluate(2, seed=1000)) == 2
        checkpoint = tmp_path / "model.pt"
        torch.save(model.state_dict(), checkpoint)
        restored = deepcopy(model)
        restored.load_state_dict(torch.load(checkpoint, weights_only=True))
        torch.testing.assert_close(restored(probe), model(probe))
    assert r"\begin{quantikz}" in model.draw(output="tikz")
    assert "RY" in model.draw(output="text")


@pytest.mark.parametrize(
    "options",
    [
        {"shots": 10},
        {"mpi": True},
        {"diff_method": "adjoint"},
        {"pp_truncation": 0},
        {"pp_truncation": 1.5},
        {"pp_topM": 0},
        {"pp_tau": -1},
        {"pp_tau": float("nan")},
        {"pp_prune_every": -1},
        {"pp_prune": "yes"},
    ],
)
def test_pp_invalid_configuration(options):
    with pytest.raises(ValueError):
        PQDQN(**pp_config(**options))


def test_pp_four_layer_cartpole_circuit_and_default_pruning():
    torch.manual_seed(7)
    cfg = pp_config(
        n_qubits=4,
        observables="parity",
        preset_actions=4,
        feature_map_config={"features": 4, "rotations": "Y", "input_scaling": True},
    )
    cfg["ansatz_config"]["n_layers"] = 4
    pp = PQDQN(**cfg).double()
    exact = PQDQN(**(cfg | {"device": "default.qubit"})).double()
    exact.load_state_dict(pp.state_dict())
    x = torch.tensor([0.21, -0.37, 0.12, 0.31], dtype=torch.float64)
    a, b = pp(x), exact(x)
    torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-9)
    a.sum().backward()
    b.sum().backward()
    for p, q in zip(pp.parameters(), exact.parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-7, atol=1e-9)
    default_cfg = {k: v for k, v in cfg.items() if not k.startswith("pp_")}
    pruned = PQDQN(**default_cfg).double()
    pruned.load_state_dict(pp.state_dict())
    assert pruned.pp_prune and pruned.pp_engine.topM == 20000
    torch.testing.assert_close(pruned(x), exact(x), rtol=1e-6, atol=1e-7)
