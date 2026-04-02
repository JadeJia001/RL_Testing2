"""PETS Agent — Probabilistic Ensemble Trajectory Sampling.

Main training loop:
  1. Collect seed episodes with random actions
  2. Repeat:
     a. Train ensemble dynamics model on replay buffer
     b. Run episode(s) using CEM planner
     c. Store transitions in buffer
"""
import numpy as np
import torch
from tqdm import tqdm

from experiments.agents.ensemble_model import EnsembleDynamics
from experiments.agents.cem_planner import CEMPlanner
from experiments.utils.episode import Episode, ReplayBuffer


class PETSAgent:
    """PETS: pure planning agent with no policy network."""

    def __init__(self, config):
        self.cfg = config
        self.dynamics = EnsembleDynamics(config)
        self.planner = CEMPlanner(config, self.dynamics)
        self.buffer = ReplayBuffer(max_size=config.buffer.max_size)

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------
    def collect_random_episodes(self, env, n_episodes: int) -> list:
        """Collect episodes with uniformly random actions."""
        episodes = []
        for _ in range(n_episodes):
            ep = self._run_episode(env, random=True)
            self.buffer.push_episode(ep)
            episodes.append(ep)
        return episodes

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def select_action(self, obs: np.ndarray) -> int:
        """Use CEM planner to pick the best action."""
        return self.planner.plan(obs)

    # ------------------------------------------------------------------
    # Episode execution
    # ------------------------------------------------------------------
    def rollout_episode(self, env) -> Episode:
        """Run one episode using CEM planning, store in buffer."""
        ep = self._run_episode(env, random=False)
        self.buffer.push_episode(ep)
        return ep

    def _run_episode(self, env, random: bool = False) -> Episode:
        obs, _ = env.reset()
        self.planner.reset()  # clear CEM warm-start
        ep = Episode()
        ep.observations.append(obs.copy())
        done = False
        truncated = False

        while not (done or truncated):
            if random:
                action = env.action_space.sample()
            else:
                action = self.select_action(obs)

            next_obs, reward, done, truncated, _info = env.step(action)
            ep.actions.append(action)
            ep.rewards.append(reward)
            ep.observations.append(next_obs.copy())
            ep.total_reward += reward
            obs = next_obs

        ep.length = len(ep.actions)
        ep.done = done  # True if terminated (pole fell), False if truncated (time limit)
        return ep

    # ------------------------------------------------------------------
    # Model training
    # ------------------------------------------------------------------
    def train_model(self) -> dict:
        """Train ensemble dynamics model on current buffer."""
        return self.dynamics.train(self.buffer)

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------
    def train(self, env, total_iterations: int) -> list:
        """PETS main training loop.

        Returns:
            reward_history: list of episode rewards (seed + planned)
        """
        reward_history = []

        # Phase 1: random seed data
        print(f'Collecting {self.cfg.training.num_seed_episodes} random seed episodes...')
        seed_eps = self.collect_random_episodes(
            env, self.cfg.training.num_seed_episodes)
        for ep in seed_eps:
            reward_history.append(ep.total_reward)
        print(f'  Buffer size: {len(self.buffer)}, '
              f'avg reward: {np.mean([e.total_reward for e in seed_eps]):.1f}')

        # Phase 2: train model → plan → collect → repeat
        pbar = tqdm(range(1, total_iterations + 1), desc='PETS training')
        for iteration in pbar:
            # Train model
            metrics = self.train_model()

            # Collect episodes with CEM planner
            for _ in range(self.cfg.training.rollout_episodes_per_iter):
                ep = self.rollout_episode(env)
                reward_history.append(ep.total_reward)

            # Logging
            recent = reward_history[-20:] if len(reward_history) >= 20 else reward_history
            avg = np.mean(recent)
            pbar.set_postfix({
                'ep_rew': f'{ep.total_reward:.0f}',
                'avg20': f'{avg:.1f}',
                'buf': len(self.buffer),
                'mse': f'{metrics["holdout_mse"]:.4f}',
                'ep_tr': metrics['epochs'],
            })

            # Early success
            if len(reward_history) >= 20 and avg >= 195:
                print(f'\nSolved at iteration {iteration}! '
                      f'Last-20 avg: {avg:.1f}')
                break

        return reward_history

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str):
        torch.save({
            'dynamics': self.dynamics.state_dict(),
            'config': dict(self.cfg),
        }, path)
        print(f'Checkpoint saved to {path}')

    def load(self, path: str):
        ckpt = torch.load(path, weights_only=False)
        self.dynamics.load_state_dict(ckpt['dynamics'])
        print(f'Checkpoint loaded from {path}')
