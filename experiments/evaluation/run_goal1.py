#!/usr/bin/env python3
"""Goal 1 experiment: G0 (random) vs G1 (danger-guided) search.

Both methods use the SAME planner (CEM) with NO action noise.
The only variable is the initial-state perturbation:
  G0: randomly sampled each episode (no search guidance)
  G1: evolved by a genetic algorithm to maximize danger_score

Runs multiple trials for statistical comparison.
"""
import os, sys, time, pickle
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from copy import deepcopy
from experiments.configs.pets_cartpole import config
from experiments.agents.ensemble_model import EnsembleDynamics
from experiments.agents.cem_planner import CEMPlanner
from experiments.features.feature_extractor import FeatureExtractor
from experiments.models.danger_scorer import DangerScorer
from experiments.search.genetic_search import GeneticSearch
from experiments.search.random_baseline import random_baseline
from experiments.utils.env_wrapper import make_env

CHECKPOINT   = 'experiments/data/checkpoints/pets_cartpole.pt'
EBM_MODEL    = 'experiments/data/models/ebm_model.pkl'
REF_STATES   = 'experiments/data/episodes/training_states.npy'
RESULT_DIR   = 'experiments/data/results'

POP_SIZE = 20
N_GENERATIONS = 20
N_REPEATS = 5
PERTURB_SIGMA = 0.05   # same for both G0 and G1


def make_fast_planner(dynamics):
    fast_cfg = deepcopy(config)
    fast_cfg.planner.num_candidates = 100
    fast_cfg.planner.horizon = 8
    fast_cfg.planner.num_elites = 20
    fast_cfg.planner.num_iterations = 1
    return CEMPlanner(fast_cfg, dynamics)


def run_g1(dynamics, planner, feature_extractor, danger_scorer,
           env, pop_size, n_gen, seed):
    ga = GeneticSearch(
        pop_size=pop_size, n_generations=n_gen,
        crossover_prob=0.8, mutation_prob=0.4,
        tournament_size=5, perturb_sigma=PERTURB_SIGMA,
        mutation_sigma=0.03, seed=seed,
    )

    def cb(gen, pop, stats):
        cum_fr = stats['cum_faults'] / stats['cum_episodes']
        print(f'    G1 gen {gen:2d}: gen_fr={stats["fault_rate"]:.2f} '
              f'cum_fr={cum_fr:.3f} best_fit={stats["best_fitness"]:.3f} '
              f'mean_fit={stats["mean_fitness"]:.3f}')

    return ga.search(env, planner, feature_extractor, danger_scorer,
                     callback=cb)


def run_g0(planner, env, n_episodes, seed):
    return random_baseline(env, planner, n_episodes,
                           perturb_sigma=PERTURB_SIGMA, seed=seed)


def compile_g1(history, pop_size):
    """Convert G1 history → (cum_episodes[], cum_faults[])."""
    eps = [h['cum_episodes'] for h in history]
    faults = [h['cum_faults'] for h in history]
    return np.array(eps), np.array(faults)


def compile_g0(results, batch_size):
    """Batch G0 flat results → (cum_episodes[], cum_faults[])."""
    eps, faults = [], []
    running = 0
    for i, r in enumerate(results):
        running += int(r['is_fault'])
        if (i + 1) % batch_size == 0 or i == len(results) - 1:
            eps.append(i + 1)
            faults.append(running)
    return np.array(eps), np.array(faults)


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    os.makedirs(RESULT_DIR, exist_ok=True)

    print('Loading agent and models...')
    dynamics = EnsembleDynamics(config)
    ckpt = torch.load(CHECKPOINT, weights_only=False)
    dynamics.load_state_dict(ckpt['dynamics'])

    planner = make_fast_planner(dynamics)
    training_states = np.load(REF_STATES)
    feature_extractor = FeatureExtractor(config, dynamics, training_states)
    danger_scorer = DangerScorer(EBM_MODEL)
    env = make_env(config)

    n_total = POP_SIZE * (N_GENERATIONS + 1)
    print(f'Budget per trial: {n_total} episodes  '
          f'(perturbation σ={PERTURB_SIGMA})')

    all_g0, all_g1 = [], []

    for trial in range(N_REPEATS):
        tseed = 42 + trial * 1000
        print(f'\n=== Trial {trial+1}/{N_REPEATS} (seed={tseed}) ===')

        t0 = time.time()
        print('  Running G1 (danger-guided)...')
        g1_hist = run_g1(dynamics, planner, feature_extractor, danger_scorer,
                         env, POP_SIZE, N_GENERATIONS, tseed)
        g1_eps, g1_faults = compile_g1(g1_hist, POP_SIZE)
        all_g1.append((g1_eps, g1_faults))
        g1t = time.time() - t0
        print(f'  G1: {g1_faults[-1]} faults / {g1_eps[-1]} eps ({g1t:.0f}s)')

        t0 = time.time()
        print('  Running G0 (random)...')
        g0_res = run_g0(planner, env, n_total, tseed + 500)
        g0_eps, g0_faults = compile_g0(g0_res, POP_SIZE)
        all_g0.append((g0_eps, g0_faults))
        g0t = time.time() - t0
        print(f'  G0: {g0_faults[-1]} faults / {g0_eps[-1]} eps ({g0t:.0f}s)')

    # Save
    results = {
        'g0': all_g0, 'g1': all_g1,
        'pop_size': POP_SIZE, 'n_generations': N_GENERATIONS,
        'n_repeats': N_REPEATS, 'n_total_episodes': n_total,
    }
    result_path = os.path.join(RESULT_DIR, 'goal1_results.pkl')
    with open(result_path, 'wb') as f:
        pickle.dump(results, f)
    print(f'\nResults saved to {result_path}')

    # Summary
    print('\n=== Summary ===')
    g0_fr = [g0f[-1] / g0e[-1] for g0e, g0f in all_g0]
    g1_fr = [g1f[-1] / g1e[-1] for g1e, g1f in all_g1]
    print(f'G0 final fault rate: {np.mean(g0_fr):.3f} ± {np.std(g0_fr):.3f}')
    print(f'G1 final fault rate: {np.mean(g1_fr):.3f} ± {np.std(g1_fr):.3f}')
    if np.mean(g1_fr) > np.mean(g0_fr):
        improvement = (np.mean(g1_fr) - np.mean(g0_fr)) / np.mean(g0_fr) * 100
        print(f'✓ G1 > G0 by {improvement:.1f}%: '
              f'danger_score guidance finds more faults!')
    else:
        print('✗ G1 ≤ G0')

    env.close()


if __name__ == '__main__':
    main()
