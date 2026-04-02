#!/usr/bin/env python3
"""Phase 4c — Cross-validate EBM / CORELS consistency."""
import os, sys, pickle
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

MODEL_DIR    = 'experiments/data/models'
FEATURE_PATH = 'experiments/data/features/features.npz'


def main():
    # --- Load models ---
    with open(os.path.join(MODEL_DIR, 'ebm_model.pkl'), 'rb') as f:
        ebm = pickle.load(f)
    with open(os.path.join(MODEL_DIR, 'corels_model.pkl'), 'rb') as f:
        corels_data = pickle.load(f)
    corels_model = corels_data['model']
    thresholds = corels_data['thresholds']
    directions = corels_data['directions']

    # --- Load test data ---
    data = np.load(FEATURE_PATH, allow_pickle=True)
    X = data['X'].astype(np.float64)
    Y = data['Y'].astype(int)

    # --- EBM predictions ---
    ebm_pred = ebm.predict(X)
    ebm_proba = ebm.predict_proba(X)[:, 1]

    # --- CORELS predictions ---
    from experiments.models.train_corels import binarize
    Xb = binarize(X, thresholds, directions)
    corels_pred = corels_model.predict(Xb)

    # --- Consistency ---
    agree = np.mean(ebm_pred == corels_pred)
    print(f'=== EBM vs CORELS consistency ===')
    print(f'Overall agreement: {agree:.1%}')

    # Agreement on failure predictions specifically
    ebm_fail = (ebm_pred == 1)
    corels_fail = (corels_pred == 1)
    both_fail = ebm_fail & corels_fail
    either_fail = ebm_fail | corels_fail
    if either_fail.sum() > 0:
        jaccard = both_fail.sum() / either_fail.sum()
        print(f'Failure Jaccard:  {jaccard:.1%}')
    if ebm_fail.sum() > 0:
        overlap = both_fail.sum() / ebm_fail.sum()
        print(f'CORELS covers {overlap:.1%} of EBM failure predictions')

    # --- Per-class agreement ---
    for label, name in [(0, 'success'), (1, 'failure')]:
        mask = (Y == label)
        a = np.mean(ebm_pred[mask] == corels_pred[mask])
        print(f'  Agreement on true {name} steps: {a:.1%}')

    # --- High-danger EBM episodes ---
    episode_ids = data['episode_ids']
    unique_eps = np.unique(episode_ids)
    n_agree_eps = 0
    n_disagree_eps = 0
    disagree_details = []

    for ep_id in unique_eps:
        mask = (episode_ids == ep_id)
        ep_ebm_score = ebm_proba[mask].max()
        ep_ebm_high = ep_ebm_score > 0.5
        ep_corels_high = corels_pred[mask].max() > 0.5
        if ep_ebm_high == ep_corels_high:
            n_agree_eps += 1
        else:
            n_disagree_eps += 1
            if len(disagree_details) < 5:
                disagree_details.append(
                    f'  ep={ep_id}: EBM_high={ep_ebm_high} '
                    f'(score={ep_ebm_score:.3f})  '
                    f'CORELS_high={ep_corels_high}')

    ep_agree = n_agree_eps / len(unique_eps)
    print(f'\nEpisode-level agreement: {ep_agree:.1%} '
          f'({n_agree_eps}/{len(unique_eps)})')
    if disagree_details:
        print('Sample disagreements:')
        for d in disagree_details:
            print(d)

    if agree < 0.70:
        print('\n⚠  WARNING: Step-level agreement < 70%')
    else:
        print(f'\nPASSED: consistency = {agree:.1%} >= 70%')

    return agree


if __name__ == '__main__':
    main()
