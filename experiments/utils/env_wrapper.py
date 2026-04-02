"""Environment utilities for CartPole-v1."""
import gymnasium as gym
import numpy as np


def make_env(config, seed=None):
    """Create CartPole-v1 environment with configured max steps."""
    env = gym.make(config.env.env_id, max_episode_steps=config.env.max_steps)
    if seed is not None:
        env.reset(seed=seed)
    return env


def get_env_state(env) -> np.ndarray:
    """Get CartPole internal state for save/restore."""
    return np.array(env.unwrapped.state, dtype=np.float32)


def set_env_state(env, state: np.ndarray):
    """Restore CartPole to a specific state."""
    env.unwrapped.state = tuple(state.tolist())
    return np.array(env.unwrapped.state, dtype=np.float32)
