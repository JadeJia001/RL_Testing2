#!/usr/bin/env python3
"""Extract 6 features from collected episodes.

Loads episodes from data/episodes/, runs FeatureExtractor on each,
and assembles the full feature matrix X [n_steps, 6] with labels Y.
"""
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.configs.pets_cartpole import config
from experiments.agents.ensemble_model import EnsembleDynamics
from experiments.features.feature_extractor import FeatureExtractor

CHECKPOINT = 'experiments/data/checkpoints/pets_cartpole.pt'
EPISODE_DIR = 'experiments/data/episodes'
OUT_DIR = 'experiments/data/features'
FAILURE_THRESHOLD = 195

FEATURE_NAMES = [
    'F1_advantage',
    'F2_dV',
    'F3_prediction_error',
    'F4_uncertainty_accel',
    'F5_density',
    'F6_snr',
]


def load_episodes(episode_dir):
    """Load episodes from the concatenated npz file.

    Returns list of episode dicts (same schema as collect_episodes).
    """
    path = os.path.join(episode_dir, 'episodes.npz')
    data = np.load(path, allow_pickle=True)

    total_rewards = data['total_rewards']
    is_failure = data['is_failure']
    lengths = data['lengths']
    all_states = data['all_states']
    all_actions = data['all_actions']
    all_next_states = data['all_next_states']
    episode_ids = data['episode_ids']

    n_episodes = len(total_rewards)
    episodes = []
    offset = 0
    for i in range(n_episodes):
        L = int(lengths[i])
        s = all_states[offset:offset + L]            # [L, 4]
        a = all_actions[offset:offset + L]            # [L]
        ns = all_next_states[offset:offset + L]       # [L, 4]
        # Reconstruct full states array [L+1, 4]: s_0..s_L
        full_states = np.concatenate([s, ns[-1:]], axis=0)
        episodes.append({
            'states': full_states,
            'actions': a,
            'next_states': ns,
            'total_reward': float(total_rewards[i]),
            'is_failure': bool(is_failure[i]),
            'length': L,
        })
        offset += L
    return episodes


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Load dynamics model ---
    print('Loading ensemble dynamics model...')
    dynamics = EnsembleDynamics(config)
    ckpt = torch.load(CHECKPOINT, weights_only=False)
    dynamics.load_state_dict(ckpt['dynamics'])
    print(f'  Elite indices: {dynamics.elite_indices}')

    # --- Load training states for F5 ---
    ref_path = os.path.join(EPISODE_DIR, 'training_states.npy')
    training_states = np.load(ref_path)
    print(f'  Training reference states: {training_states.shape[0]}')

    # --- Create feature extractor ---
    extractor = FeatureExtractor(config, dynamics, training_states)

    # --- Load episodes ---
    print('Loading episodes...')
    episodes = load_episodes(EPISODE_DIR)
    n_ep = len(episodes)
    n_fail = sum(1 for e in episodes if e['is_failure'])
    print(f'  {n_ep} episodes ({n_fail} failures)')

    # --- Extract features ---
    print(f'\nExtracting 6 features for {n_ep} episodes...')
    all_X = []
    all_Y = []
    all_ep_ids = []
    all_step_ids = []

    t0 = time.time()
    for i, ep in enumerate(episodes):
        feats = extractor.extract(ep)    # [T, 6]
        T = feats.shape[0]

        label = 1 if ep['is_failure'] else 0
        all_X.append(feats)
        all_Y.append(np.full(T, label, dtype=np.int32))
        all_ep_ids.append(np.full(T, i, dtype=np.int32))
        all_step_ids.append(np.arange(T, dtype=np.int32))

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_ep - i - 1) / rate
            print(f'  [{i+1}/{n_ep}]  {elapsed:.0f}s elapsed, '
                  f'~{eta:.0f}s remaining')

    elapsed = time.time() - t0
    print(f'  Done in {elapsed:.1f}s')

    # --- Assemble matrices ---
    X = np.concatenate(all_X, axis=0).astype(np.float32)   # [N, 6]
    Y = np.concatenate(all_Y, axis=0)                       # [N]
    ep_ids = np.concatenate(all_ep_ids, axis=0)
    step_ids = np.concatenate(all_step_ids, axis=0)

    # --- Sanity checks ---
    print(f'\n=== Feature matrix ===')
    print(f'Shape: X={X.shape}, Y={Y.shape}')
    print(f'Labels: 0(success)={np.sum(Y == 0)}, 1(failure)={np.sum(Y == 1)}')

    has_nan = np.isnan(X).any()
    has_inf = np.isinf(X).any()
    print(f'NaN: {has_nan}  Inf: {has_inf}')
    if has_nan or has_inf:
        # Replace any NaN/Inf with 0 (should not happen)
        print('  WARNING: replacing NaN/Inf with 0')
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Per-feature statistics ---
    print(f'\n=== Per-feature statistics ===')
    print(f'{"Feature":<25s} {"mean":>10s} {"std":>10s} '
          f'{"min":>10s} {"max":>10s}')
    print('-' * 68)
    for j, name in enumerate(FEATURE_NAMES):
        col = X[:, j]
        print(f'{name:<25s} {col.mean():10.4f} {col.std():10.4f} '
              f'{col.min():10.4f} {col.max():10.4f}')

    # --- Failure vs success comparison ---
    print(f'\n=== Failure vs Success means ===')
    print(f'{"Feature":<25s} {"Success":>10s} {"Failure":>10s} {"Diff":>10s}')
    print('-' * 58)
    for j, name in enumerate(FEATURE_NAMES):
        s_mean = X[Y == 0, j].mean() if (Y == 0).any() else 0
        f_mean = X[Y == 1, j].mean() if (Y == 1).any() else 0
        print(f'{name:<25s} {s_mean:10.4f} {f_mean:10.4f} '
              f'{f_mean - s_mean:10.4f}')

    # --- Save ---
    np.savez_compressed(
        os.path.join(OUT_DIR, 'features.npz'),
        X=X,
        Y=Y,
        episode_ids=ep_ids,
        step_ids=step_ids,
        feature_names=FEATURE_NAMES,
    )
    print(f'\nSaved to {os.path.join(OUT_DIR, "features.npz")}')


if __name__ == '__main__':
    main()
