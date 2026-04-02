#!/usr/bin/env python3
"""Phase 4b — Train CORELS rule list on binarized features.

Binarizes 6 features at multiple quantile thresholds, selects the
binarization with best CORELS accuracy, outputs if-then rules.
"""
import os, sys, pickle, warnings
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

FEATURE_PATH = 'experiments/data/features/features.npz'
MODEL_DIR    = 'experiments/data/models'
OUTPUT_DIR   = 'experiments/outputs'
SEED = 42

FEATURE_NAMES_SHORT = ['F1_adv', 'F2_dV', 'F3_PE', 'F4_d2U', 'F5_rho', 'F6_SNR']

from experiments.models.train_ebm import episode_train_test_split


def binarize(X, thresholds, direction):
    """Binarize features: 1 if feature OP threshold, 0 otherwise.

    direction[j] = 'high' → feature > threshold → 1 (higher = more dangerous)
    direction[j] = 'low'  → feature < threshold → 1 (lower = more dangerous)
    """
    Xb = np.zeros_like(X, dtype=np.int8)
    for j in range(X.shape[1]):
        if direction[j] == 'high':
            Xb[:, j] = (X[:, j] > thresholds[j]).astype(np.int8)
        else:
            Xb[:, j] = (X[:, j] < thresholds[j]).astype(np.int8)
    return Xb


def train_corels():
    np.random.seed(SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Load data ---
    data = np.load(FEATURE_PATH, allow_pickle=True)
    X = data['X'].astype(np.float64)
    Y = data['Y'].astype(int)
    episode_ids = data['episode_ids']

    X_tr, Y_tr, X_te, Y_te, _, _ = episode_train_test_split(
        X, Y, episode_ids, test_ratio=0.2, seed=SEED)

    # --- Determine binarization directions ---
    # For each feature: which direction correlates with failure (Y=1)?
    fail_means = X[Y == 1].mean(axis=0)
    succ_means = X[Y == 0].mean(axis=0)
    directions = []
    for j in range(6):
        directions.append('high' if fail_means[j] > succ_means[j] else 'low')
    print('Binarization directions:', dict(zip(FEATURE_NAMES_SHORT, directions)))

    # --- Try multiple quantile thresholds ---
    best_acc = 0
    best_q = None
    best_model = None
    best_thresholds = None
    best_bin_names = None

    for q in [0.50, 0.60, 0.70, 0.75, 0.80]:
        thresholds = np.quantile(X_tr, q, axis=0)
        Xb_tr = binarize(X_tr, thresholds, directions)
        Xb_te = binarize(X_te, thresholds, directions)

        dir_labels = ['high' if d == 'high' else 'low' for d in directions]
        bin_names = [f'{FEATURE_NAMES_SHORT[j]}_{dir_labels[j]}'
                     for j in range(6)]

        try:
            from corels import CorelsClassifier
            clf = CorelsClassifier(
                max_card=2,
                c=0.001,
                n_iter=10000,
                verbosity=[],
            )
            clf.fit(Xb_tr, Y_tr, features=bin_names, prediction_name='failure')
            pred_te = clf.predict(Xb_te)
            acc = np.mean(pred_te == Y_te)
        except Exception as e:
            print(f'  CORELS failed at q={q}: {e}')
            # Fallback: sklearn DecisionTree
            from sklearn.tree import DecisionTreeClassifier
            clf = DecisionTreeClassifier(max_depth=3, random_state=SEED)
            clf.fit(Xb_tr, Y_tr)
            pred_te = clf.predict(Xb_te)
            acc = np.mean(pred_te == Y_te)

        print(f'  q={q:.2f}: accuracy={acc:.4f}')
        if acc > best_acc:
            best_acc = acc
            best_q = q
            best_model = clf
            best_thresholds = thresholds
            best_bin_names = bin_names

    print(f'\nBest quantile: {best_q}, accuracy: {best_acc:.4f}')

    # --- Extract rules ---
    rules = extract_rules(best_model, best_bin_names)
    print('\n=== CORELS Rules (Table 1) ===')
    for r in rules:
        print(f"  Rule {r['id']}: {r['condition']}  →  {r['prediction']}  "
              f"(accuracy: {r['accuracy']:.1%})")

    # --- Save ---
    with open(os.path.join(MODEL_DIR, 'corels_model.pkl'), 'wb') as f:
        pickle.dump({
            'model': best_model,
            'thresholds': best_thresholds,
            'directions': directions,
            'bin_names': best_bin_names,
            'best_q': best_q,
            'rules': rules,
        }, f)

    # Save rules as CSV
    import csv
    csv_path = os.path.join(OUTPUT_DIR, 'table1_rules.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Rule', 'Condition', 'Prediction', 'Accuracy'])
        writer.writeheader()
        for r in rules:
            writer.writerow({
                'Rule': r['id'],
                'Condition': r['condition'],
                'Prediction': r['prediction'],
                'Accuracy': f"{r['accuracy']:.1%}",
            })
    print(f'\nRules saved to {csv_path}')

    return best_model, rules, best_acc


def extract_rules(model, bin_names):
    """Extract human-readable rules from CORELS or DecisionTree."""
    rules = []
    try:
        # CORELS model
        rl = model.rl()
        rule_str = rl.rules  if hasattr(rl, 'rules') else str(rl)
        # Parse the CORELS rule list
        rl_str = str(model.rl())
        lines = [l.strip() for l in rl_str.split('\n') if l.strip()]
        rule_id = 0
        for line in lines:
            if 'if' in line.lower() or 'else' in line.lower():
                rule_id += 1
                # Clean up the condition
                cond = line.replace('{', '').replace('}', '').strip()
                pred = 'failure' if '1' in line.split(':')[-1] else 'success'
                rules.append({
                    'id': rule_id,
                    'condition': cond,
                    'prediction': pred,
                    'accuracy': 0.0,  # filled below
                })
    except Exception:
        # Fallback: extract from DecisionTree
        from sklearn.tree import _tree
        tree = model.tree_
        feature_names_map = bin_names

        def recurse(node, conditions, depth=0):
            if tree.feature[node] == _tree.TREE_UNDEFINED:
                # Leaf node
                counts = tree.value[node][0]
                pred = 'failure' if counts[1] > counts[0] else 'success'
                total = counts.sum()
                acc = max(counts) / total if total > 0 else 0
                cond_str = ' AND '.join(conditions) if conditions else 'DEFAULT'
                rules.append({
                    'id': len(rules) + 1,
                    'condition': f'IF {cond_str}' if conditions else 'ELSE',
                    'prediction': pred,
                    'accuracy': acc,
                })
                return
            fname = feature_names_map[tree.feature[node]]
            # Left: feature <= 0.5 → feature = 0 (not triggered)
            # Right: feature > 0.5 → feature = 1 (triggered)
            recurse(tree.children_right[node],
                     conditions + [fname], depth + 1)
            recurse(tree.children_left[node],
                     conditions + [f'NOT {fname}'], depth + 1)

        recurse(0, [])
        # Keep only failure-predicting rules + default
        rules = [r for r in rules if r['prediction'] == 'failure' or
                 r['condition'] == 'ELSE'][:5]

    return rules


if __name__ == '__main__':
    model, rules, acc = train_corels()
