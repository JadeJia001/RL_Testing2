#!/usr/bin/env python3
"""Generate Table 1 — CORELS danger-boundary rules."""
import os, sys, pickle
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

MODEL_DIR  = 'experiments/data/models'
OUTPUT_DIR = 'experiments/outputs'


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(MODEL_DIR, 'corels_model.pkl'), 'rb') as f:
        corels_data = pickle.load(f)

    rules = corels_data['rules']
    thresholds = corels_data['thresholds']
    directions = corels_data['directions']
    best_q = corels_data['best_q']

    print('=== Table 1: Danger Boundary Rules (CORELS) ===\n')
    print(f'Binarization quantile: {best_q}')
    print(f'Thresholds: {np.round(thresholds, 4)}')
    print(f'Directions: {directions}\n')

    # --- CSV ---
    import csv
    csv_path = os.path.join(OUTPUT_DIR, 'table1_rules.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['Rule #', 'Condition', 'Prediction', 'Accuracy'])
        for r in rules:
            w.writerow([r['id'], r['condition'], r['prediction'],
                        f"{r['accuracy']:.1%}"])
    print(f'CSV saved to {csv_path}')

    # --- LaTeX ---
    tex_path = os.path.join(OUTPUT_DIR, 'table1_rules.tex')
    with open(tex_path, 'w') as f:
        f.write('\\begin{table}[h]\n')
        f.write('\\centering\n')
        f.write('\\caption{Danger Boundary Rules extracted by CORELS}\n')
        f.write('\\label{tab:rules}\n')
        f.write('\\begin{tabular}{clcc}\n')
        f.write('\\toprule\n')
        f.write('Rule \\# & Condition & Prediction & Accuracy \\\\\n')
        f.write('\\midrule\n')
        for r in rules:
            cond = r['condition'].replace('_', '\\_').replace('&', '\\&')
            f.write(f"{r['id']} & {cond} & {r['prediction']} & "
                    f"{r['accuracy']:.1\\%} \\\\\n")
        f.write('\\bottomrule\n')
        f.write('\\end{tabular}\n')
        f.write('\\end{table}\n')
    print(f'LaTeX saved to {tex_path}')

    # --- Print ---
    print(f'\n{"Rule #":<8} {"Condition":<50} {"Prediction":<12} {"Accuracy":<10}')
    print('-' * 82)
    for r in rules:
        print(f"{r['id']:<8} {r['condition']:<50} {r['prediction']:<12} "
              f"{r['accuracy']:.1%}")


if __name__ == '__main__':
    main()
