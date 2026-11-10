"""Exercise the QASP/TensorDict/TorchRL boundary with real quantum policies."""
import pennylane  # Load the numerical runtime before Torch on macOS.
import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.modules import ProbabilisticActor

from qasp.rl.model import create_model
from qasp.rl.torchrl import PQCBornWrapper


def test_public_wildcard_import_resolves_lazy_rl_exports():
    import qasp
    from qasp.rl import torchrl

    namespace = {}
    exec('from qasp import *', namespace)
    assert set(qasp.__all__) <= namespace.keys()
    assert namespace['REINFORCE'] is torchrl.REINFORCE
    assert namespace['PQCBornWrapper'] is torchrl.PQCBornWrapper
    assert 'TorchRLTrainer' not in namespace
    assert 'PQCWrapper' not in namespace
    assert not hasattr(torchrl, 'TorchRLTrainer')


@pytest.mark.parametrize('device', ['default.qubit', 'IQPSimulator', 'PauliPropagationSimulator'])
def test_torchrl_actor_preserves_quantum_policy_gradients(device):
    torch.manual_seed(17)
    config = {
        'ansatz': 'AngleAnsatz',
        'ansatz_config': {'n_layers': 1, 'rotations': ['X 0', 'X 1', 'XX 0 1'],
                          'include_imprimitive': False},
        'feature_map': 'AngleEmbedding',
        'feature_map_config': {'features': 2, 'rotations': 'X'},
        'n_actions': 2, 'policy': 'parity-like', 'training_method': 'probs',
        'parity_cnot_decomposition': False,
        'device': device, 'diff_method': 'backprop', 'shots': None,
    }
    if device == 'IQPSimulator':
        config['device_kwargs'] = {'backend': 'tn', 'mode': 'explicit'}
    policy = create_model('PQCBorn', config, n_qubits=2).double()
    actor = ProbabilisticActor(
        module=TensorDictModule(PQCBornWrapper(policy, obs_dim=2),
                               in_keys=['observation'], out_keys=['probs']),
        in_keys=['probs'], distribution_class=torch.distributions.Categorical,
    )
    obs = torch.tensor([0.3, -0.2], dtype=torch.float64, requires_grad=True)
    td = TensorDict({'observation': obs}, batch_size=[])
    distribution = actor.get_dist(td)
    torch.testing.assert_close(distribution.probs, policy.action_probs(obs))
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    before = [p.detach().clone() for p in policy.parameters()]
    loss = -distribution.log_prob(torch.tensor(0))
    loss.backward()
    assert obs.grad is not None and torch.isfinite(obs.grad).all()
    grads = [p.grad for p in policy.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)
    optimizer.step()
    assert any(not torch.equal(p, old) for p, old in zip(policy.parameters(), before))
