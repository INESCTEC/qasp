import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from qasp.rl.envs import WindyTargetNavigationEnv


@pytest.mark.parametrize("n_actions", [3, 4])
def test_windy_navigation_gymnasium_api(n_actions):
    env = WindyTargetNavigationEnv(
        n_actions=n_actions, max_episode_steps=8, n_waypoints=2
    )
    check_env(env, skip_render_check=True)
    observation, info = env.reset(seed=5)
    assert env.action_space.n == n_actions
    assert observation.shape == (6,)
    assert observation.dtype == np.float32
    assert env.observation_space.contains(observation)
    assert set(info) == {"distance", "waypoints_reached", "success"}

    terminated = truncated = False
    for step in range(8):
        observation, reward, terminated, truncated, info = env.step(
            step % n_actions
        )
        assert env.observation_space.contains(observation)
        assert np.isfinite(reward)
        if terminated or truncated:
            break
    assert terminated or truncated


@pytest.mark.parametrize("n_actions", [3, 4])
def test_windy_navigation_seed_reproduces_trajectory(n_actions):
    left = WindyTargetNavigationEnv(n_actions=n_actions)
    right = WindyTargetNavigationEnv(n_actions=n_actions)
    left_observation, left_info = left.reset(seed=17)
    right_observation, right_info = right.reset(seed=17)
    np.testing.assert_array_equal(left_observation, right_observation)
    assert left_info == right_info

    for action in [0, 1, 2, 0, 1]:
        left_step = left.step(action)
        right_step = right.step(action)
        np.testing.assert_array_equal(left_step[0], right_step[0])
        assert left_step[1:] == right_step[1:]


def test_windy_navigation_rejects_other_action_counts():
    with pytest.raises(ValueError, match="supports 3 or 4 actions"):
        WindyTargetNavigationEnv(n_actions=2)
