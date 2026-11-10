"""Small real learning loops across the three supported general execution paths.

These verify collection, finite statistics and actual weight updates, not solved
CartPole or a learning advantage.
"""
import numpy as np
import pytest
import torch
from qasp import create_model
from qasp.initialization import Normal
from configuration_helpers import BACKENDS, execution
from qasp.rl.torchrl import (
    PPO, Reinforce, ReinforceConfig,
    ContextualBanditTrainer, ContextualBanditEnv, BanditAWRConfig,
)


def configuration(device):
    config = {
        'ansatz': 'AngleAnsatz',
        'ansatz_config': {'n_layers': 1, 'rotations': ['X 0', 'X 1', 'XX 0 1'],
                          'include_imprimitive': False},
        'feature_map': 'NeuralEmbedding',
        'feature_map_config': {'features': 4, 'rotations': 'X'},
        'n_actions': 2, 'policy': 'parity-like', 'training_method': 'expectation',
        'parity_cnot_decomposition': False,
        'device': device, 'diff_method': 'backprop', 'shots': None,
        'init': Normal(0.0, 0.3),
        'input_scaling_init': Normal(0.5, 0.1),
    }
    if device == 'IQPSimulator':
        config['device_kwargs'] = {'backend': 'tn', 'mode': 'explicit'}
    if device == 'PauliPropagationSimulator':
        config['device_kwargs'] = {'k': None, 'prune': False}
    return config


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('algorithm', ['ppo', 'reinforce', 'bandit'])
def test_training_loops_update_quantum_weights(backend, algorithm):
    torch.manual_seed(404)
    np.random.seed(404)
    policy = create_model('PQCBorn', {**configuration('default.qubit'), **execution(backend)}, n_qubits=2)
    before = [p.detach().clone() for p in policy.parameters()]
    trainer = None
    try:
        if algorithm == 'ppo':
            trainer = PPO(policy, steps_per_rollout=8, epochs=1, minibatch_size=4,
                          logits=False, device='cpu')
            result = trainer.train(total_updates=2, log_every=1)
            stats = result['update_returns']
        elif algorithm == 'reinforce':
            cfg = ReinforceConfig(n_episodes=2, n_batch=1, max_episode_steps=8,
                                  log_every=1, eval_every=100)
            trainer = Reinforce(policy, config=cfg)
            result = trainer.train()
            stats = result['episode_returns']
        else:
            env = ContextualBanditEnv(n_features=4, n_actions=2, n_contexts=16, seed=404)
            cfg = BanditAWRConfig(batch_size=4, updates_per_epoch=2, device='cpu', seed=404)
            trainer = ContextualBanditTrainer(env, policy, n_actions=2, cfg=cfg)
            stats = trainer.train(epochs=2, episodes_per_epoch=8, log_every=1)
        assert len(stats) > 0 and np.isfinite(stats).all()
        assert all(torch.isfinite(p).all() for p in policy.parameters())
        assert any(not torch.equal(p, old) for p, old in zip(policy.parameters(), before))
        gradients = [p.grad for p in policy.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
    finally:
        if trainer is not None and hasattr(trainer, 'env') and hasattr(trainer.env, 'close'):
            trainer.env.close()
