#!/usr/bin/env python3
"""Phase 4a — Train ExplainableBoostingClassifier (EBM) on 6 features.

Splits by episode (no data leakage), trains EBM, evaluates AUC/F1,
saves model + shape-function plots + feature importances.
"""
import os, sys, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
from interpret.glassbox import ExplainableBoostingClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

FEATURE_PATH = 'experiments/data/features/features.npz'
MODEL_DIR    = 'experiments/data/models'
OUTPUT_DIR   = 'experiments/outputs'
SEED = 42

FEATURE_NAMES = ['F1_advantage', 'F2_dV', 'F3_PE', 'F4_d2U', 'F5_rho', 'F6_SNR']


def episode_train_test_split(X, Y, episode_ids, test_ratio=0.2, seed=42):
    """Split by episode: all steps of an episode go to train OR test."""
    rng = np.random.RandomState(seed)
    unique_eps = np.unique(episode_ids)
    rng.shuffle(unique_eps)
    n_test = max(1, int(len(unique_eps) * test_ratio))
    test_eps = set(unique_eps[:n_test])

    test_mask = np.isin(episode_ids, list(test_eps))
    train_mask = ~test_mask
    return (X[train_mask], Y[train_mask],
            X[test_mask],  Y[test_mask],
            train_mask, test_mask)


def train_ebm():
    np.random.seed(SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load data ---
    data = np.load(FEATURE_PATH, allow_pickle=True)
    X = data['X'].astype(np.float64)
    Y = data['Y'].astype(int)
    episode_ids = data['episode_ids']
    print(f'Loaded features: X={X.shape}, Y class balance: '
          f'0={np.sum(Y==0)} 1={np.sum(Y==1)}')

    # --- Episode-level train/test split ---
    X_tr, Y_tr, X_te, Y_te, _, _ = episode_train_test_split(
        X, Y, episode_ids, test_ratio=0.2, seed=SEED)
    print(f'Train: {X_tr.shape[0]} steps  Test: {X_te.shape[0]} steps')

    # --- Train EBM ---
    print('\nTraining EBM...')
    ebm = ExplainableBoostingClassifier(
        feature_names=FEATURE_NAMES,
        random_state=SEED,
        max_bins=256,
        max_rounds=5000,
        learning_rate=0.01,
    )
    ebm.fit(X_tr, Y_tr)

    # Also compute episode-level AUC (more relevant for search)
    data_ep_ids = data['episode_ids']
    unique_eps = np.unique(data_ep_ids)
    ep_proba = ebm.predict_proba(X)[:, 1]
    ep_scores = []
    ep_labels = []
    for eid in unique_eps:
        mask = data_ep_ids == eid
        ep_scores.append(ep_proba[mask].max())
        ep_labels.append(int(Y[mask].max()))  # 1 if any step is failure
    ep_auc = roc_auc_score(ep_labels, ep_scores)
    print(f'  Episode-level AUC: {ep_auc:.4f}')

    # --- Evaluate ---
    proba_te = ebm.predict_proba(X_te)[:, 1]
    pred_te  = ebm.predict(X_te)
    auc  = roc_auc_score(Y_te, proba_te)
    acc  = accuracy_score(Y_te, pred_te)
    f1   = f1_score(Y_te, pred_te)

    print(f'\n=== Test metrics ===')
    print(f'  AUC:      {auc:.4f}')
    print(f'  Accuracy: {acc:.4f}')
    print(f'  F1:       {f1:.4f}')
    print(classification_report(Y_te, pred_te, target_names=['success', 'failure']))

    # --- Feature importances ---
    importances = dict(zip(FEATURE_NAMES, ebm.term_importances()))
    print('Feature importances:')
    for name in sorted(importances, key=importances.get, reverse=True):
        print(f'  {name:25s}  {importances[name]:.4f}')

    # --- Save model ---
    model_path = os.path.join(MODEL_DIR, 'ebm_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(ebm, f)
    print(f'\nModel saved to {model_path}')

    # --- Shape function plots ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    ebm_global = ebm.explain_global()
    for idx, ax in enumerate(axes.flat):
        if idx >= len(FEATURE_NAMES):
            ax.set_visible(False)
            continue
        data_dict = ebm_global.data(idx)
        xs = np.array(data_dict['names'])
        ys = np.array(data_dict['scores'])
        upper = data_dict.get('upper_bounds')
        lower = data_dict.get('lower_bounds')

        # Align lengths (interpret may return len(xs)==len(ys)+1 for bin edges)
        n = min(len(xs), len(ys))
        xs, ys = xs[:n], ys[:n]
        ax.plot(xs, ys, 'b-', linewidth=1.5)
        if upper is not None and lower is not None:
            upper = np.array(upper)[:n]
            lower = np.array(lower)[:n]
            ax.fill_between(xs, lower, upper, alpha=0.15, color='b')
        ax.set_xlabel(FEATURE_NAMES[idx], fontsize=10)
        ax.set_ylabel('Score contribution', fontsize=9)
        ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
        ax.grid(True, alpha=0.3)

    fig.suptitle('EBM Shape Functions — Feature Contributions to danger_score',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    shape_path = os.path.join(OUTPUT_DIR, 'ebm_shape_functions.png')
    fig.savefig(shape_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Shape functions saved to {shape_path}')

    # --- Save split info for later use ---
    np.savez_compressed(
        os.path.join(MODEL_DIR, 'ebm_split.npz'),
        X_train=X_tr, Y_train=Y_tr,
        X_test=X_te, Y_test=Y_te,
        auc=auc, accuracy=acc, f1=f1,
    )

    return ebm, auc


if __name__ == '__main__':
    ebm, auc = train_ebm()
    if auc >= 0.70:
        print(f'\nPASSED: AUC={auc:.4f} >= 0.70')
    else:
        print(f'\nWARNING: AUC={auc:.4f} < 0.70')
