"""Optional adapters must preserve the existing model APIs and behavior."""
from copy import deepcopy
from functools import partial
import subprocess
import sys

import gymnasium as gym
import pytest
import torch
from torch import nn
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.learner.utils import update_target_network

from qasp.circuits import AngleAnsatz
from qasp.feature_maps import AngleEmbedding, NeuralEmbedding
from qasp.rl import PQCSoftmax, PQCBornVariance, PQCBorn, PQDQN
from qasp.rl.rllib import ppo_module_spec, dqn_module_spec


pytestmark = pytest.mark.filterwarnings("error:The 'id' argument is deprecated")

def model_kwargs():
    return dict(n_qubits=4, n_actions=2, ansatz=AngleAnsatz,
                ansatz_config=dict(n_qubits=4, n_layers=2, rotations="YZ", reuploading=True),
                feature_map=AngleEmbedding,
                feature_map_config=dict(n_qubits=4, features=4, rotations="Y", input_scaling=True),
                device="default.qubit", diff_method="backprop")


def make_policy(kind="softmax"):
    if kind == "softmax":
        return PQCSoftmax(**model_kwargs(), observables=["Z 0", "Z 1"], fc_layer=True)
    if kind == "legacy":
        return PQCBorn(**model_kwargs(), measurement_qubits=[0, 1, 2, 3])
    return PQCBornVariance(**model_kwargs(), policy="parity-like", measurement_qubits=[0, 1, 2, 3])


def make_q_model():
    return PQDQN(**model_kwargs(), observables=["Z 0", "Z 1"], fc_layer=True)


def build(spec):
    spec.observation_space = gym.spaces.Box(-float("inf"), float("inf"), (4,))
    spec.action_space = gym.spaces.Discrete(2)
    return spec.build()


def test_existing_usage_does_not_import_rllib():
    # Start fresh so no adapter/test imports have loaded Ray. Block Ray entirely.
    result = subprocess.run([sys.executable, "-I", "-c", '''
import importlib.abc, sys
class BlockRay(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "ray" or fullname.startswith("ray."):
            raise AssertionError("Existing QASP usage tried to import Ray")
sys.meta_path.insert(0, BlockRay())
from qasp.rl import PQDQN, PQCSoftmax, PQCBornVariance, DeepQLearning, REINFORCE
from qasp.circuits import AngleAnsatz
from qasp.feature_maps import AngleEmbedding
import torch
model = PQDQN(n_qubits=2, n_actions=2, ansatz=AngleAnsatz,
    ansatz_config={"n_layers":1}, feature_map=AngleEmbedding,
    feature_map_config={"features":2,"rotations":"Y"}, observables=["Z 0","Z 1"])
model(torch.randn(2)).sum().backward()
assert "qasp.rl.rllib" not in sys.modules
print("Existing models and trainers work without importing Ray")
'''], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("kind", ["softmax", "born", "legacy"])
def test_policy_semantics_and_independent_training(kind):
    torch.manual_seed(11)
    policy = make_policy(kind)
    original = deepcopy(policy.state_dict())
    x = torch.randn(2, 3, 4)
    before_x = x.clone()
    method = policy.action_probs if kind == "legacy" else policy
    expected = torch.stack([method(row) for row in x.reshape(-1, 4)]).reshape(2, 3, 2)
    expected = expected.softmax(-1) if kind == "softmax" else expected / expected.sum(-1, keepdim=True)
    module = build(ppo_module_spec(policy))
    logits = module.forward_train({Columns.OBS: x})[Columns.ACTION_DIST_INPUTS]
    torch.testing.assert_close(logits.softmax(-1), expected)
    torch.testing.assert_close(x, before_x)
    values = module.compute_values({Columns.OBS: x})
    assert values.shape == (2, 3)
    loss = -logits.log_softmax(-1)[..., 0].mean() + (values - 2).square().mean()
    loss.backward()
    assert module.policy.qlayer.params.grad.abs().sum() > 0
    assert sum(p.grad.abs().sum() for p in module.critic.parameters()) > 0
    torch.optim.Adam(module.parameters(), lr=.01).step()
    assert any(not torch.equal(original[k], v) for k, v in module.policy.state_dict().items())
    for key, value in policy.state_dict().items():
        torch.testing.assert_close(original[key], value)
    assert all(p.grad is None for p in policy.parameters())


def test_hybrid_policy_custom_critic_and_factory():
    hybrid = nn.Sequential(nn.Linear(4, 4), nn.Tanh(), make_policy())
    critic = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 1))
    module = build(ppo_module_spec(hybrid, critic=critic, policy_output="logits"))
    obs = torch.randn(3, 4)
    actual = module.forward_train({Columns.OBS: obs})[Columns.ACTION_DIST_INPUTS]
    torch.testing.assert_close(actual, torch.stack([hybrid(row) for row in obs]))
    (actual.square().sum() + module.compute_values({Columns.OBS: obs}).square().sum()).backward()
    assert module.policy[0].weight.grad.abs().sum() > 0
    assert module.policy[2].qlayer.params.grad.abs().sum() > 0
    factory_module = build(ppo_module_spec(partial(make_policy, "born")))
    assert factory_module.policy_output == "probs"


def test_q_model_outputs_targets_and_gradients():
    model = make_q_model()
    module = build(dqn_module_spec(model))
    x = torch.randn(6, 4)
    output = module.forward_train({Columns.OBS: x, Columns.NEXT_OBS: x + .2})
    torch.testing.assert_close(output["qf_preds"], model(x))
    output["qf_preds"].square().sum().backward()
    assert module.q_network.qlayer.params.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())
    assert all(p.grad is None for p in module.target_q_network.parameters())
    old_target = deepcopy(module.target_q_network.state_dict())
    with torch.no_grad():
        next(module.q_network.parameters()).add_(1)
    assert all(torch.equal(v, old_target[k]) for k, v in module.target_q_network.state_dict().items())
    update_target_network(main_net=module.q_network, target_net=module.target_q_network, tau=1)
    for key, value in module.q_network.state_dict().items():
        torch.testing.assert_close(value, module.target_q_network.state_dict()[key])


@pytest.mark.parametrize("algorithm", ["ppo", "dqn"])
def test_snapshot_and_checkpoint(algorithm, tmp_path):
    model = make_policy() if algorithm == "ppo" else make_q_model()
    original = deepcopy(model.state_dict())
    spec = ppo_module_spec(model) if algorithm == "ppo" else dqn_module_spec(model)
    with torch.no_grad():
        next(model.parameters()).add_(10)
    module = build(spec)
    restored_model = module.policy if algorithm == "ppo" else module.q_network
    for key, value in restored_model.state_dict().items():
        torch.testing.assert_close(value, original[key])
    restored = RLModule.from_checkpoint(module.save_to_path(str(tmp_path / "module")))
    for key, value in module.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key])


def test_invalid_options():
    with pytest.raises(ValueError, match="policy_output"):
        build(ppo_module_spec(nn.Linear(4, 2)))
    spec = dqn_module_spec(make_q_model())
    spec.model_config["dueling"] = True
    with pytest.raises(ValueError, match="dueling"):
        build(spec)
    policy = make_policy()
    policy.mpi = True
    with pytest.raises(ValueError, match="MPI"):
        ppo_module_spec(policy)


@pytest.mark.parametrize("kind", ["softmax", "born", "neural"])
def test_vmap_matches_sequential_outputs_and_gradients(kind):
    torch.manual_seed(12)
    if kind == "neural":
        kwargs = model_kwargs()
        kwargs.update(feature_map=NeuralEmbedding,
                      feature_map_config=dict(n_qubits=4, features=4, rotations="Y"))
        policy = PQCSoftmax(**kwargs, observables=["Z 0", "Z 1"], fc_layer=True)
    else:
        policy = make_policy(kind)
    sequential = build(ppo_module_spec(policy))
    batched = build(ppo_module_spec(policy, policy_batch_mode="vmap"))
    obs = torch.randn(2, 8, 4)
    a = sequential.forward_train({Columns.OBS: obs})[Columns.ACTION_DIST_INPUTS]
    b = batched.forward_train({Columns.OBS: obs})[Columns.ACTION_DIST_INPUTS]
    torch.testing.assert_close(a, b)
    a.log_softmax(-1)[..., 0].sum().backward()
    b.log_softmax(-1)[..., 0].sum().backward()
    for p, q in zip(sequential.policy.parameters(), batched.policy.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=1e-4)


def test_notebook_vector_evaluation_matches_serial():
    import nbformat
    import numpy as np
    from pathlib import Path
    notebook = nbformat.read(Path(__file__).resolve().parents[2] / "examples/cartpole_rllib.ipynb", as_version=4)
    source = next(c.source for c in notebook.cells if c.cell_type == "code" and c.source.startswith("def evaluate"))
    namespace = {"gym": gym, "np": np, "torch": torch, "Columns": Columns}
    exec(source.split("def train_and_evaluate")[0], namespace)
    # Test the evaluation machinery on long episodes, independently of learning.
    controller = nn.Linear(4, 2, bias=False)
    with torch.no_grad():
        controller.weight.copy_(torch.tensor([[0., -.1, -1., -.2], [0., .1, 1., .2]]))
    module = build(ppo_module_spec(controller, policy_output="logits"))
    env = gym.make("CartPole-v1")
    expected = []
    for seed in range(30000, 30005):
        observation, _ = env.reset(seed=seed)
        total = 0
        while True:
            out = module.forward_inference({Columns.OBS: torch.as_tensor(observation).unsqueeze(0)})
            observation, reward, terminated, truncated, _ = env.step(int(out[Columns.ACTION_DIST_INPUTS].argmax(-1).item()))
            total += reward
            if terminated or truncated:
                break
        expected.append(total)
    env.close()
    assert namespace["evaluate"](module, 5, 30000) == expected
    assert max(expected) == 500  # truncation at the actual CartPole-v1 horizon
