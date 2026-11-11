"""Native RLlib training with user-created QASP model instances."""
from copy import deepcopy
from pathlib import Path
import importlib.util

import numpy as np
import pytest
import qasp
import ray
import torch
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.train._internal.storage import StorageContext

# Reuse the runnable example; this path is not shipped to workers.
source = Path(__file__).resolve().parents[2] / "examples/rllib_cartpole.py"
spec = importlib.util.spec_from_file_location("cartpole_example", source)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


@pytest.mark.parametrize("algorithm", ["ppo", "dqn"])
def test_distributed_rollouts_checkpoint_and_resume(algorithm, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    torch.set_num_threads(1)
    ray.init(num_cpus=3, include_dashboard=False, log_to_driver=False)
    algo = restored = None
    try:
        config = example.make_config(algorithm, workers=2, smoke=True)
        storage = StorageContext(storage_path=str(tmp_path), experiment_dir_name="test", trial_dir_name=algorithm)
        algo = config.algo_class(config=config, storage=storage)
        module = algo.learner_group._learner.module["default_policy"]
        model = module.policy if algorithm == "ppo" else module.q_network
        before = deepcopy(model.qlayer.state_dict())
        result = algo.train()
        assert any(not torch.equal(v, before[k]) for k, v in model.qlayer.state_dict().items())
        workers = algo.env_runner_group.foreach_env_runner(
            lambda r: {"source": __import__("qasp").__file__, "state": r.module.get_state()},
            local_env_runner=False,
        )
        assert len(workers) == 2
        state = module.get_state(inference_only=True)
        for worker in workers:
            assert Path(worker["source"]).resolve() == Path(qasp.__file__).resolve()
            for key, value in worker["state"].items():
                np.testing.assert_allclose(value, state[key], atol=1e-6)
        assert 1 <= example.evaluate(algo.get_module(), 2)["mean"] <= 500
        checkpoint = algo.save_to_path(str(tmp_path / "checkpoint"))
        before_restore = module.get_state()
        selected = RLModule.from_checkpoint(str(Path(checkpoint) /
            "learner_group/learner/rl_module/default_policy"))
        for key, value in before_restore.items():
            np.testing.assert_allclose(value, selected.get_state()[key])
        algo.stop()
        algo = None
        restored = Algorithm.from_checkpoint(checkpoint)
        module = restored.learner_group._learner.module["default_policy"]
        for key, value in before_restore.items():
            np.testing.assert_allclose(value, module.get_state()[key])
        assert restored.train()["num_env_steps_sampled_lifetime"] > result["num_env_steps_sampled_lifetime"]
    finally:
        if algo is not None:
            algo.stop()
        if restored is not None:
            restored.stop()
        ray.shutdown()


@pytest.mark.parametrize("algorithm", ["ppo", "dqn"])
@pytest.mark.parametrize("workers", [0, 2, 7])
def test_example_topologies(algorithm, workers):
    example.make_config(algorithm, workers=workers).validate()
