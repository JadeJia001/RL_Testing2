"""G0 — Random baseline: run episodes with random initial perturbations.

Same budget as G1, same planner, same perturbation range — but NO
evolutionary optimization of the perturbation.  Pure random sampling.
"""
import numpy as np


def random_baseline(env, planner, n_episodes, perturb_sigma=0.05,
                    max_steps=200, seed=42):
    """Run n_episodes with random initial perturbations, pure CEM actions.

    Returns list of per-episode result dicts.
    """
    rng = np.random.RandomState(seed)
    results = []

    for i in range(n_episodes):
        obs, _ = env.reset()
        try:
            base_state = np.array(env.unwrapped.state, dtype=np.float32)
            perturbed = base_state + rng.normal(
                0, perturb_sigma, size=4).astype(np.float32)
            env.unwrapped.state = tuple(perturbed.tolist())
            obs = np.array(env.unwrapped.state, dtype=np.float32)
        except Exception:
            pass

        planner.reset()
        total_reward = 0
        done, truncated = False, False
        step = 0

        while not (done or truncated) and step < max_steps:
            action = planner.plan(obs)
            obs, reward, done, truncated, _ = env.step(int(action))
            total_reward += reward
            step += 1

        results.append({
            'episode': i,
            'reward': total_reward,
            'length': step,
            'is_fault': total_reward < 195,
        })

    return results
