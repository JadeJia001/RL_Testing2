#!/usr/bin/env python3
"""Collect episodes from trained PETS agent.

Runs >=500 episodes with 20% random-action perturbation to increase
failure diversity.  Saves full episode data + reference training states.

Uses a lightweight planner (fewer candidates, shorter horizon) for
collection speed while preserving representative agent behavior.
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.configs.pets_cartpole import config
from experiments.agents.pets_agent import PETSAgent
from experiments.agents.cem_planner import CEMPlanner
from experiments.utils.env_wrapper import make_env
from copy import deepcopy


# --------------- configuration ---------------
N_EPISODES = 600          # target total episodes
FAILURE_THRESHOLD = 195   # reward < this → failure
RANDOM_ACTION_PROB = 0.20 # probability of taking a random action
N_REF_RANDOM = 20         # random episodes for training-state reference
N_REF_PLANNED = 10        # CEM episodes for training-state reference
SEED = 42
CHECKPOINT = 'experiments/data/checkpoints/pets_cartpole.pt'
OUT_DIR = 'experiments/data/episodes'


def make_fast_planner(agent):
    """Create a lightweight CEM planner for faster collection.

    Uses 50 candidates × horizon 5 instead of 500 × 10 → ~20x faster.
    Still produces good policies for CartPole (2 discrete actions).
    """
    fast_cfg = deepcopy(config)
    fast_cfg.planner.num_candidates = 100
    fast_cfg.planner.horizon = 8
    fast_cfg.planner.num_elites = 20
    fast_cfg.planner.num_iterations = 1
    return CEMPlanner(fast_cfg, agent.dynamics)


def run_episode_fast(planner, env, random_prob: float) -> dict:
    """Run one episode with lightweight planner + random perturbation."""
    obs, _ = env.reset()
    planner.reset()

    states, actions, rewards, next_states = [obs.copy()], [], [], []
    done, truncated = False, False

    while not (done or truncated):
        if np.random.random() < random_prob:
            action = env.action_space.sample()
        else:
            action = planner.plan(obs)
        next_obs, reward, done, truncated, _ = env.step(action)
        actions.append(int(action))
        rewards.append(float(reward))
        next_states.append(next_obs.copy())
        states.append(next_obs.copy())
        obs = next_obs

    total_reward = sum(rewards)
    return {
        'states': np.array(states, dtype=np.float32),          # [T+1, 4]
        'actions': np.array(actions, dtype=np.int64),           # [T]
        'rewards': np.array(rewards, dtype=np.float32),         # [T]
        'next_states': np.array(next_states, dtype=np.float32), # [T, 4]
        'total_reward': float(total_reward),
        'is_failure': total_reward < FAILURE_THRESHOLD,
        'length': len(actions),
    }


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Load trained agent ---
    agent = PETSAgent(config)
    agent.load(CHECKPOINT)
    env = make_env(config, seed=SEED)

    # Create fast planner for collection
    fast_planner = make_fast_planner(agent)

    # ------------------------------------------------------------------
    # Step 1: Collect reference states (training distribution proxy)
    # ------------------------------------------------------------------
    print('=== Collecting reference states for F5 ===')
    ref_states = []

    # Random episodes (represents early training data)
    for _ in range(N_REF_RANDOM):
        ep = run_episode_fast(fast_planner, env, random_prob=1.0)
        ref_states.append(ep['states'][:-1])
    n_random = sum(s.shape[0] for s in ref_states)
    print(f'  Random: {N_REF_RANDOM} episodes, {n_random} states')

    # CEM episodes (represents later training data)
    for _ in range(N_REF_PLANNED):
        ep = run_episode_fast(fast_planner, env, random_prob=0.0)
        ref_states.append(ep['states'][:-1])
    all_ref_states = np.concatenate(ref_states, axis=0)
    print(f'  Total reference states: {all_ref_states.shape[0]}')
    np.save(os.path.join(OUT_DIR, 'training_states.npy'), all_ref_states)

    # ------------------------------------------------------------------
    # Step 2: Collect main episodes
    # ------------------------------------------------------------------
    print(f'\n=== Collecting {N_EPISODES} episodes '
          f'(random_prob={RANDOM_ACTION_PROB}) ===')
    t0 = time.time()

    episodes = []
    n_fail = 0
    for i in range(N_EPISODES):
        ep = run_episode_fast(fast_planner, env, random_prob=RANDOM_ACTION_PROB)
        episodes.append(ep)
        if ep['is_failure']:
            n_fail += 1
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (N_EPISODES - i - 1) / rate
            print(f'  [{i+1}/{N_EPISODES}] failures={n_fail}, '
                  f'{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining')

    # ------------------------------------------------------------------
    # Step 3: Top up failures if needed
    # ------------------------------------------------------------------
    min_failures = 50
    extra = 0
    while n_fail < min_failures:
        ep = run_episode_fast(fast_planner, env, random_prob=0.5)
        episodes.append(ep)
        if ep['is_failure']:
            n_fail += 1
        extra += 1
    if extra:
        print(f'  Collected {extra} extra episodes to reach {min_failures} failures')

    elapsed = time.time() - t0
    print(f'  Collection done in {elapsed:.0f}s')

    # ------------------------------------------------------------------
    # Step 4: Save
    # ------------------------------------------------------------------
    total = len(episodes)
    n_success = sum(1 for e in episodes if not e['is_failure'])
    n_fail = sum(1 for e in episodes if e['is_failure'])
    rewards = [e['total_reward'] for e in episodes]
    lengths = [e['length'] for e in episodes]

    print(f'\n=== Summary ===')
    print(f'Total episodes: {total}')
    print(f'  Success (≥{FAILURE_THRESHOLD}): {n_success}')
    print(f'  Failure (<{FAILURE_THRESHOLD}): {n_fail}')
    print(f'Reward: mean={np.mean(rewards):.1f}, '
          f'min={np.min(rewards):.0f}, max={np.max(rewards):.0f}')
    print(f'Length: mean={np.mean(lengths):.1f}')

    np.savez_compressed(
        os.path.join(OUT_DIR, 'episodes.npz'),
        total_rewards=np.array(rewards, dtype=np.float32),
        is_failure=np.array([e['is_failure'] for e in episodes]),
        lengths=np.array(lengths, dtype=np.int32),
        all_states=np.concatenate(
            [e['states'][:-1] for e in episodes], axis=0),
        all_actions=np.concatenate(
            [e['actions'] for e in episodes], axis=0),
        all_rewards=np.concatenate(
            [e['rewards'] for e in episodes], axis=0),
        all_next_states=np.concatenate(
            [e['next_states'] for e in episodes], axis=0),
        episode_ids=np.concatenate(
            [np.full(e['length'], i, dtype=np.int32)
             for i, e in enumerate(episodes)]),
        step_ids=np.concatenate(
            [np.arange(e['length'], dtype=np.int32)
             for e in episodes]),
    )
    print(f'\nSaved to {os.path.join(OUT_DIR, "episodes.npz")}')
    env.close()


if __name__ == '__main__':
    main()
