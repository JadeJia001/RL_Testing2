#!/usr/bin/env python3
"""Generate Figure 1 — G0 vs G1 cumulative fault-rate efficiency curve."""
import os, sys, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

RESULT_PATH = 'experiments/data/results/goal1_results.pkl'
OUTPUT_DIR  = 'experiments/outputs'


def interpolate_curves(all_curves, x_common):
    """Interpolate multiple (x, y) curves onto common x grid."""
    ys = []
    for xs, faults in all_curves:
        rates = faults / xs  # cumulative fault rate
        y_interp = np.interp(x_common, xs, rates,
                             left=rates[0], right=rates[-1])
        ys.append(y_interp)
    return np.array(ys)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(RESULT_PATH, 'rb') as f:
        results = pickle.load(f)

    all_g0 = results['g0']
    all_g1 = results['g1']
    n_total = results['n_total_episodes']

    # Common x-axis
    x_common = np.arange(20, n_total + 1, 20)

    g0_rates = interpolate_curves(all_g0, x_common)
    g1_rates = interpolate_curves(all_g1, x_common)

    g0_mean = g0_rates.mean(axis=0)
    g0_std  = g0_rates.std(axis=0)
    g1_mean = g1_rates.mean(axis=0)
    g1_std  = g1_rates.std(axis=0)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(x_common, g0_mean, 'b-', linewidth=2, label='G0: Random testing')
    ax.fill_between(x_common, g0_mean - g0_std, g0_mean + g0_std,
                    alpha=0.15, color='b')

    ax.plot(x_common, g1_mean, 'r-', linewidth=2,
            label='G1: danger_score guided')
    ax.fill_between(x_common, g1_mean - g1_std, g1_mean + g1_std,
                    alpha=0.15, color='r')

    ax.set_xlabel('Cumulative episodes', fontsize=12)
    ax.set_ylabel('Cumulative fault rate', fontsize=12)
    ax.set_title('Goal 1: Efficiency of danger_score-guided search\n'
                 'vs random testing', fontsize=13)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_total)
    ax.set_ylim(0, None)

    fig_path = os.path.join(OUTPUT_DIR, 'figure1_efficiency_curve.png')
    fig.savefig(fig_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f'Figure 1 saved to {fig_path}')

    # Print final numbers
    print(f'Final fault rates (at {n_total} episodes):')
    print(f'  G0: {g0_mean[-1]:.3f} ± {g0_std[-1]:.3f}')
    print(f'  G1: {g1_mean[-1]:.3f} ± {g1_std[-1]:.3f}')


if __name__ == '__main__':
    main()
