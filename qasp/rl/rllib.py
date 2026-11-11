"""Optional RLlib adapters for already-constructed QASP/PyTorch models.

Nothing imports this module automatically: existing qasp.rl models and trainers
retain their original behavior and do not acquire an RLlib dependency on import.

    policy = PQCSoftmax(...)  # the usual QASP constructor
    spec = ppo_module_spec(policy)
    config = PPOConfig().rl_module(rl_module_spec=spec)

The helpers snapshot the supplied model, including its current weights. RLlib
trains independent copies; retrieve the trained model via ``module.policy`` or
``module.q_network``. A zero-argument factory can instead construct a model inside
each worker (useful for process-local simulator resources). Tested with Ray 2.60.
"""

from collections.abc import Callable

import cloudpickle
import gymnasium as gym
import torch
from torch import nn

from ray.rllib.algorithms.dqn.default_dqn_rl_module import (
    QF_NEXT_PREDS, QF_PREDS, QF_TARGET_NEXT_PREDS,
)
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis import InferenceOnlyAPI, QNetAPI, TargetNetworkAPI
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.schedules.scheduler import Scheduler

__all__ = ["ppo_module_spec", "dqn_module_spec", "PQCPolicyRLModule", "PQDQNRLModule"]


def _snapshot(model):
    if not isinstance(model, nn.Module) and not callable(model):
        raise TypeError("Pass an nn.Module or a zero-argument model factory.")
    if isinstance(model, nn.Module):
        _validate_model(model)
    try:
        return cloudpickle.dumps(model)
    except Exception as exc:
        raise TypeError("This model cannot be copied to RLlib workers. Pass a "
                        "zero-argument factory that constructs it inside each worker.") from exc


def _validate_model(model):
    if not isinstance(model, nn.Module):
        raise TypeError("The model factory must return torch.nn.Module.")
    for child in model.modules():
        if getattr(child, "mpi", False) or getattr(child, "enable_mpi", False):
            raise ValueError("These adapters distribute independent rollouts; "
                             "MPI-distributed quantum circuits are not supported.")


def _restore_model(snapshot):
    source = cloudpickle.loads(snapshot)
    model = source if isinstance(source, nn.Module) else source()
    _validate_model(model)
    return model


def ppo_module_spec(
    policy: nn.Module | Callable[[], nn.Module],
    *,
    critic: nn.Module | Callable[[], nn.Module] | None = None,
    policy_output: str | None = None,
    policy_method: str | None = None,
    critic_hidden: tuple[int, ...] = (64, 64),
    probability_floor: float = 1e-8,
    policy_batch_mode: str = "sequential",
) -> RLModuleSpec:
    """Adapt an existing policy to native RLlib PPO with a classical critic.

    ``policy_output`` is inferred for QASP's PQCSoftmax (logits) and Born policies
    (probabilities). Specify 'logits' or 'probs' for other hybrid/PyTorch models.
    ``policy_method`` defaults to forward, except legacy PQCBorn and
    LearnablePQCBorn, whose probability entrypoint is action_probs.
    Defaults do not preprocess observations or alter the supplied model.

    Policy evaluation is per observation to accommodate published QASP policies
    that flatten batch axes. Gradients remain connected to all trainable layers.
    Opt into policy_batch_mode='vmap' for torch.vmap-compatible deterministic
    policies (e.g. analytic default.qubit/backprop). The default is unchanged.
    A supplied critic must return shape (..., 1) or (...) for (..., features).
    Factories must create identical architectures on all workers; RLlib syncs
    weights. With a model instance, the current weights are snapshotted now.
    """
    if policy_output not in {None, "logits", "probs"}:
        raise ValueError("policy_output must be 'logits', 'probs', or None.")
    if policy_batch_mode not in {"sequential", "vmap"}:
        raise ValueError("policy_batch_mode must be 'sequential' or 'vmap'.")
    if not 0 < probability_floor < 1:
        raise ValueError("probability_floor must lie between zero and one.")
    if not all(isinstance(width, int) and width > 0 for width in critic_hidden):
        raise ValueError("critic_hidden must contain positive integers.")
    return RLModuleSpec(
        module_class=PQCPolicyRLModule,
        model_config={
            "policy_snapshot": _snapshot(policy),
            "critic_snapshot": None if critic is None else _snapshot(critic),
            "policy_output": policy_output,
            "policy_method": policy_method,
            "critic_hidden": list(critic_hidden),
            "probability_floor": probability_floor,
            "policy_batch_mode": policy_batch_mode,
        },
    )


def dqn_module_spec(q_model: nn.Module | Callable[[], nn.Module]) -> RLModuleSpec:
    """Adapt an existing PQDQN/hybrid Q-network to RLlib DQN or Double DQN.

    Configure RLlib with ``.training(dueling=False, num_atoms=1)``. RLlib owns
    the loss, replay, optimizer and target updates. The model must return raw
    Q-values of shape (..., n_actions) for inputs (..., features).
    """
    return RLModuleSpec(
        module_class=PQDQNRLModule,
        model_config={"q_model_snapshot": _snapshot(q_model)},
    )


def _validate_spaces(module):
    if not isinstance(module.action_space, gym.spaces.Discrete) or module.action_space.start != 0:
        raise ValueError("The adapters require zero-based Discrete actions.")
    if not isinstance(module.observation_space, gym.spaces.Box) or len(module.observation_space.shape) != 1:
        raise ValueError("The adapters require a one-dimensional Box observation space.")


def _observations(batch, model):
    reference = next(model.parameters(), None)
    if reference is None:
        reference = next(model.buffers(), None)
    kwargs = {} if reference is None else {"dtype": reference.dtype, "device": reference.device}
    return torch.as_tensor(batch[Columns.OBS], **kwargs)


class PQCPolicyRLModule(TorchRLModule, ValueFunctionAPI):
    """Optional PPO wrapper; ``policy`` is an independent copy of the user model."""

    def setup(self):
        _validate_spaces(self)
        self.policy = _restore_model(self.model_config["policy_snapshot"])
        from .policies import LearnablePQCBorn, PQCBorn, PQCBornVariance, PQCSoftmax
        legacy_born = isinstance(self.policy, (PQCBorn, LearnablePQCBorn))
        self.policy_method = self.model_config["policy_method"] or (
            "action_probs" if legacy_born else "forward"
        )
        if not callable(getattr(self.policy, self.policy_method, None)):
            raise ValueError(f"Policy has no callable {self.policy_method!r} method.")
        self.policy_output = self.model_config["policy_output"]
        if self.policy_output is None:
            if self.policy_method == "forward" and isinstance(self.policy, PQCSoftmax):
                self.policy_output = "logits"
            elif isinstance(self.policy, (PQCBorn, PQCBornVariance, LearnablePQCBorn)):
                self.policy_output = "probs"
            else:
                raise ValueError("Specify policy_output='logits' or 'probs' for this policy.")
        self.probability_floor = self.model_config["probability_floor"]
        self.policy_batch_mode = self.model_config.get("policy_batch_mode", "sequential")
        if self.model_config["critic_snapshot"] is not None:
            self.critic = _restore_model(self.model_config["critic_snapshot"])
        else:
            dims = [self.observation_space.shape[0], *self.model_config["critic_hidden"]]
            layers = []
            for left, right in zip(dims, dims[1:]):
                layers.extend([nn.Linear(left, right), nn.Tanh()])
            self.critic = nn.Sequential(*layers, nn.Linear(dims[-1], 1))

    def _forward(self, batch, **kwargs):
        x = _observations(batch, self.policy)
        flat = x.reshape(-1, x.shape[-1])
        evaluate = getattr(self.policy, self.policy_method)
        # Legacy models can preprocess their argument in place. Protect RLlib's
        # observations without detaching the gradient graph.
        if not len(flat):
            scores = x.new_empty((0, self.action_space.n))
        elif self.policy_batch_mode == "vmap":
            scores = torch.vmap(lambda row: evaluate(row.clone()).reshape(-1))(flat)
        else:
            scores = torch.stack([evaluate(row.clone()).reshape(-1) for row in flat])
        if scores.shape[-1] != self.action_space.n:
            raise ValueError("Policy must return one value per action.")
        if not torch.isfinite(scores).all():
            raise ValueError("Policy returned nonfinite values.")
        if self.policy_output == "probs":
            if (scores < 0).any() or (scores.sum(-1) <= 0).any():
                raise ValueError("Probabilities must be nonnegative with positive mass.")
            scores = scores / scores.sum(-1, keepdim=True)
            # Explicit smoothing of zero Born probabilities before taking logs.
            scores = scores.clamp_min(self.probability_floor).log()
        return {Columns.ACTION_DIST_INPUTS: scores.reshape(*x.shape[:-1], self.action_space.n)}

    def compute_values(self, batch, embeddings=None):
        x = _observations(batch, self.critic)
        values = self.critic(x)
        if values.shape == (*x.shape[:-1], 1):
            values = values.squeeze(-1)
        if values.shape != x.shape[:-1]:
            raise ValueError("Critic must return one scalar value per observation.")
        return values


class PQDQNRLModule(TorchRLModule, QNetAPI, TargetNetworkAPI, InferenceOnlyAPI):
    """Optional DQN wrapper with independent online and frozen target models."""

    def setup(self):
        _validate_spaces(self)
        if self.model_config.get("num_atoms", 1) != 1 or self.model_config.get("dueling", False):
            raise ValueError("Configure DQN with dueling=False and num_atoms=1.")
        self.q_network = _restore_model(self.model_config["q_model_snapshot"])
        self.epsilon_schedule = Scheduler(
            fixed_value_or_schedule=self.model_config.get("epsilon", [[0, 1.0], [20000, 0.02]]),
            framework="torch",
        )
        if not self.inference_only:
            self.make_target_networks()

    def get_non_inference_attributes(self):
        return ["target_q_network"]

    def make_target_networks(self):
        if not hasattr(self, "target_q_network"):
            self.target_q_network = _restore_model(self.model_config["q_model_snapshot"])
            parameter = next(self.q_network.parameters())
            self.target_q_network.to(device=parameter.device, dtype=parameter.dtype)
            self.target_q_network.load_state_dict(self.q_network.state_dict())
            self.target_q_network.requires_grad_(False)

    def get_target_network_pairs(self):
        return [(self.q_network, self.target_q_network)]

    def _q_values(self, batch, model):
        x = _observations(batch, model)
        values = model(x)
        if values.shape != (*x.shape[:-1], self.action_space.n):
            raise ValueError("Q model must preserve batch axes and return one value per action.")
        return values

    def compute_q_values(self, batch):
        return {QF_PREDS: self._q_values(batch, self.q_network)}

    def forward_target(self, batch):
        with torch.no_grad():
            return {QF_PREDS: self._q_values(batch, self.target_q_network)}

    def _forward(self, batch, **kwargs):
        return {Columns.ACTIONS: self.compute_q_values(batch)[QF_PREDS].argmax(-1)}

    def _forward_exploration(self, batch, t=0, **kwargs):
        with torch.no_grad():
            greedy = self._forward(batch)[Columns.ACTIONS]
            self.epsilon_schedule.update(t)
            random_actions = torch.randint(self.action_space.n, greedy.shape, device=greedy.device)
            explore = torch.rand(greedy.shape, device=greedy.device) < self.epsilon_schedule.get_current_value()
            return {Columns.ACTIONS: torch.where(explore, random_actions, greedy)}

    def _forward_train(self, batch, **kwargs):
        output = self.compute_q_values(batch)
        next_batch = {Columns.OBS: batch[Columns.NEXT_OBS]}
        with torch.no_grad():
            output[QF_NEXT_PREDS] = self.compute_q_values(next_batch)[QF_PREDS]
            output[QF_TARGET_NEXT_PREDS] = self.forward_target(next_batch)[QF_PREDS]
        return output
