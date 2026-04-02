#!/usr/bin/env python3
"""Train PETS on CartPole-v1 and save checkpoint + training curve."""
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Allow running from project root: python -m experiments.train_pets
# or from experiments/: python train_pets.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.configs.pets_cartpole import config
from experiments.agents.pets_agent import PETSAgent
from experiments.utils.env_wrapper import make_env


def main():
    # --- Seed ---
    seed = config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # --- Environment ---
    env = make_env(config, seed=seed)
    print(f'Environment: {config.env.env_id}  '
          f'(state_dim={config.env.state_dim}, '
          f'actions={config.env.num_actions}, '
          f'max_steps={config.env.max_steps})')

    # --- Agent ---
    agent = PETSAgent(config)

    # --- Train ---
    reward_history = agent.train(env, config.training.total_iterations)

    # --- Save checkpoint ---
    ckpt_dir = os.path.join(os.path.dirname(__file__), 'data', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'pets_cartpole.pt')
    agent.save(ckpt_path)

    # --- Plot training curve ---
    out_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    episodes = np.arange(1, len(reward_history) + 1)
    ax.plot(episodes, reward_history, alpha=0.4, label='Episode reward')

    # Moving average (window=20)
    if len(reward_history) >= 20:
        ma = np.convolve(reward_history, np.ones(20) / 20, mode='valid')
        ax.plot(np.arange(20, 20 + len(ma)), ma, linewidth=2,
                label='20-episode moving avg')

    ax.axhline(y=195, color='r', linestyle='--', alpha=0.5, label='Solved (195)')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('PETS on CartPole-v1')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(out_dir, 'pets_training_curve.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Training curve saved to {plot_path}')

    # --- Verification ---
    last_20 = reward_history[-20:]
    avg = np.mean(last_20)
    print(f'\n=== Results ===')
    print(f'Total episodes: {len(reward_history)}')
    print(f'Last 20 avg reward: {avg:.1f}')
    print(f'Last 20 min/max: {min(last_20):.0f} / {max(last_20):.0f}')
    if avg >= 195:
        print('PASSED: Solved criterion met (avg >= 195)')
    else:
        print(f'NOT YET SOLVED: avg={avg:.1f} < 195')

    env.close()
    return avg >= 195


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
