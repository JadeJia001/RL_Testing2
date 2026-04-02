"""STARLA-adapted genetic search guided by danger_score.

Each individual = state_perturbation vector [4] applied to the
CartPole initial state.  The CEM planner runs WITHOUT action noise —
the only source of variation is the initial condition.

Fitness = danger_score (maximize — higher = more dangerous episode).
Single-objective: tournament selection, uniform crossover, Gaussian mutation.
"""
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Optional
from copy import deepcopy


@dataclass
class Individual:
    """One member of the GA population."""
    state_perturb: np.ndarray        # [4] additive initial-state noise
    fitness: float = 0.0             # danger_score (higher = better)
    is_fault: bool = False           # actual episode failure
    episode_reward: float = 0.0
    episode_length: int = 0


class GeneticSearch:
    """Single-objective GA guided by danger_score.

    Adapted from STARLA: keeps tournament selection, crossover, mutation.
    Removes: multi-objective preference sort, abstract-state crossover,
    perturbation channels E1–E4, action noise.
    """

    def __init__(self, pop_size=20, n_generations=20,
                 crossover_prob=0.8, mutation_prob=0.4,
                 tournament_size=5, perturb_sigma=0.05,
                 mutation_sigma=0.03, seed=42):
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.perturb_sigma = perturb_sigma
        self.mutation_sigma = mutation_sigma
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------
    def init_population(self) -> List[Individual]:
        return [Individual(
            state_perturb=self.rng.normal(0, self.perturb_sigma, size=4
                                          ).astype(np.float32))
                for _ in range(self.pop_size)]

    # ------------------------------------------------------------------
    # Genetic operators
    # ------------------------------------------------------------------
    def tournament_select(self, pop: List[Individual]) -> Individual:
        idxs = self.rng.choice(len(pop),
                               size=min(self.tournament_size, len(pop)),
                               replace=False)
        return deepcopy(max((pop[i] for i in idxs), key=lambda x: x.fitness))

    def crossover(self, p1: Individual, p2: Individual) -> Individual:
        mask = self.rng.randint(0, 2, size=4).astype(bool)
        return Individual(
            state_perturb=np.where(mask, p1.state_perturb, p2.state_perturb))

    def mutate(self, ind: Individual) -> Individual:
        child = deepcopy(ind)
        child.state_perturb += self.rng.normal(
            0, self.mutation_sigma, size=4).astype(np.float32)
        child.state_perturb = np.clip(child.state_perturb, -0.3, 0.3)
        return child

    def generate_offspring(self, pop: List[Individual]) -> List[Individual]:
        offspring = []
        while len(offspring) < self.pop_size:
            if self.rng.random() < self.crossover_prob:
                p1 = self.tournament_select(pop)
                p2 = self.tournament_select(pop)
                child = self.crossover(p1, p2)
            else:
                child = self.tournament_select(pop)
            if self.rng.random() < self.mutation_prob:
                child = self.mutate(child)
            offspring.append(child)
        return offspring[:self.pop_size]

    # ------------------------------------------------------------------
    # Episode execution (NO action noise — pure CEM planner)
    # ------------------------------------------------------------------
    @staticmethod
    def run_individual(ind: Individual, env, planner, feature_extractor,
                       danger_scorer, max_steps=200):
        """Run episode with perturbed initial state, pure CEM actions."""
        obs, _ = env.reset()
        try:
            base_state = np.array(env.unwrapped.state, dtype=np.float32)
            perturbed = base_state + ind.state_perturb
            env.unwrapped.state = tuple(perturbed.tolist())
            obs = np.array(env.unwrapped.state, dtype=np.float32)
        except Exception:
            pass

        planner.reset()
        states = [obs.copy()]
        actions, rewards, next_states = [], [], []
        done, truncated = False, False
        step = 0

        while not (done or truncated) and step < max_steps:
            action = planner.plan(obs)
            next_obs, reward, done, truncated, _ = env.step(int(action))
            actions.append(int(action))
            rewards.append(float(reward))
            next_states.append(next_obs.copy())
            states.append(next_obs.copy())
            obs = next_obs
            step += 1

        total_reward = sum(rewards)
        length = len(actions)

        # Compute danger_score (NO reward information used)
        if length > 0:
            ep_data = {
                'states': np.array(states, dtype=np.float32),
                'actions': np.array(actions, dtype=np.int64),
                'next_states': np.array(next_states, dtype=np.float32),
            }
            features = feature_extractor.extract(ep_data)
            fitness = danger_scorer.score_episode(features)
        else:
            fitness = 0.0

        ind.fitness = fitness
        ind.is_fault = total_reward < 195
        ind.episode_reward = total_reward
        ind.episode_length = length
        return ind

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def search(self, env, planner, feature_extractor, danger_scorer,
               callback=None):
        history = []
        pop = self.init_population()

        # Evaluate initial population
        for ind in pop:
            self.run_individual(ind, env, planner, feature_extractor,
                                danger_scorer)
        init_faults = sum(1 for ind in pop if ind.is_fault)

        for gen in range(self.n_generations):
            offspring = self.generate_offspring(pop)
            for ind in offspring:
                self.run_individual(ind, env, planner, feature_extractor,
                                    danger_scorer)

            # (μ+λ) selection
            combined = pop + offspring
            combined.sort(key=lambda x: x.fitness, reverse=True)
            pop = combined[:self.pop_size]

            n_faults = sum(1 for ind in offspring if ind.is_fault)
            prev_cum = history[-1]['cum_faults'] if history else init_faults
            prev_eps = history[-1]['cum_episodes'] if history else self.pop_size
            stats = {
                'generation': gen,
                'n_episodes': self.pop_size,
                'n_faults': n_faults,
                'fault_rate': n_faults / self.pop_size,
                'best_fitness': max(ind.fitness for ind in pop),
                'mean_fitness': np.mean([ind.fitness for ind in pop]),
                'cum_episodes': prev_eps + self.pop_size,
                'cum_faults': prev_cum + n_faults,
            }
            history.append(stats)
            if callback:
                callback(gen, pop, stats)

        return history
