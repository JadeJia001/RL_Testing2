"""Episode data structure and replay buffer."""
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import List


@dataclass
class Episode:
    """Recorded trajectory from one environment run."""
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    total_reward: float = 0.0
    length: int = 0
    done: bool = False


class ReplayBuffer:
    """Simple replay buffer storing individual transitions."""

    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self._obs: List[np.ndarray] = []
        self._actions: List[int] = []
        self._rewards: List[float] = []
        self._next_obs: List[np.ndarray] = []
        self._dones: List[bool] = []

    def push(self, obs: np.ndarray, action: int, reward: float,
             next_obs: np.ndarray, done: bool):
        if len(self._obs) >= self.max_size:
            self._obs.pop(0)
            self._actions.pop(0)
            self._rewards.pop(0)
            self._next_obs.pop(0)
            self._dones.pop(0)
        self._obs.append(obs.copy())
        self._actions.append(action)
        self._rewards.append(reward)
        self._next_obs.append(next_obs.copy())
        self._dones.append(done)

    def push_episode(self, episode: Episode):
        for i in range(episode.length):
            self.push(
                episode.observations[i],
                episode.actions[i],
                episode.rewards[i],
                episode.observations[i + 1],
                (i == episode.length - 1) and episode.done,
            )

    def get_all_tensors(self) -> dict:
        """Return all transitions as a dict of tensors."""
        return {
            'obs': torch.FloatTensor(np.array(self._obs)),
            'action': torch.LongTensor(self._actions),
            'reward': torch.FloatTensor(self._rewards),
            'next_obs': torch.FloatTensor(np.array(self._next_obs)),
            'done': torch.BoolTensor(self._dones),
        }

    def __len__(self) -> int:
        return len(self._obs)
