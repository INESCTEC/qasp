import gymnasium as gym
import numpy as np
import pytest
import torch
from test_qfunctions import config

from qasp.rl import PQDQN, DeepQLearning


class ConstantQ(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.values = torch.nn.Parameter(torch.tensor([2.0, 4.0]))

    def forward(self, x):
        return self.values.expand(*x.shape[:-1], 2)


@pytest.mark.parametrize("double", [True, False])
def test_bellman_termination_truncation_and_double_selection(double):
    with DeepQLearning(ConstantQ(), gamma=0.5, double_dqn=double) as trainer:
        with torch.no_grad():
            trainer.target_network.values.copy_(torch.tensor([10.0, 6.0]))
        target = trainer._bellman_targets(
            torch.ones(2), torch.zeros(2, 4), torch.tensor([True, False])
        )
        # Online selects action 1, vanilla target-max selects action 0.
        torch.testing.assert_close(target, torch.tensor([1.0, 4.0 if double else 6.0]))
        assert not target.requires_grad


def test_quantum_cartpole_training_target_isolation_and_evaluation():
    torch.manual_seed(3)
    model = PQDQN(**config(batch_mode="broadcast"))
    with DeepQLearning(
        model,
        batch_size=4,
        learning_starts=4,
        target_update_interval=1000,
        max_steps=8,
        seed=3,
    ) as trainer:
        original = {
            k: v.clone() for k, v in trainer.target_network.state_dict().items()
        }
        before = model.qlayer.params.detach().clone()
        rows = trainer.train(2, log_every=0)
        assert trainer.frame_count == 16 and trainer.update_count == 13
        assert all(np.isfinite(r["loss"]) for r in rows)
        assert not torch.equal(before, model.qlayer.params)
        for name, value in trainer.target_network.state_dict().items():
            torch.testing.assert_close(value, original[name])
        assert all(p.grad is None for p in trainer.target_network.parameters())
        trainer.sync_target()
        x = torch.randn(3, 4)
        torch.testing.assert_close(trainer.target_network(x), model(x))
        counts = (trainer.frame_count, trainer.update_count, len(trainer.replay_buffer))
        first = trainer.evaluate(2)
        assert first == trainer.evaluate(2)
        assert counts == (
            trainer.frame_count,
            trainer.update_count,
            len(trainer.replay_buffer),
        )
        assert trainer.epsilon < trainer.epsilon_start


class OneStepEnv(gym.Env):
    action_space = gym.spaces.Discrete(2)
    observation_space = gym.spaces.Box(-1, 1, shape=(4,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        return np.ones(4, dtype=np.float32), 1.0, False, True, {}


def test_time_limit_is_stored_as_nonterminal_and_target_syncs():
    trainer = DeepQLearning(
        ConstantQ(),
        OneStepEnv(),
        batch_size=1,
        learning_starts=0,
        target_update_interval=1,
        epsilon_decay_steps=1,
    )
    trainer.train(1, log_every=0)
    assert trainer.replay_buffer[0][-1] is False
    assert trainer.epsilon == pytest.approx(trainer.epsilon_end)
    torch.testing.assert_close(trainer.qfunction.values, trainer.target_network.values)
    with pytest.raises(ValueError, match="separate"):
        trainer.evaluate()


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(batch_size=0),
        dict(epsilon_start=2),
        dict(gamma=-1),
        dict(replay_capacity=1),
        dict(learning_starts=-1),
    ],
)
def test_bad_config(kwargs):
    with pytest.raises(ValueError):
        DeepQLearning(ConstantQ(), **kwargs)
