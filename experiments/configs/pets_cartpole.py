"""PETS configuration for CartPole-v1.

CartPole has 4D state and 2 discrete actions — its dynamics are nearly
linear, so a 2-layer 64-wide ensemble is more than sufficient.  We use
random shooting (CEM with 1 iteration) since the binary action space
makes refinement less impactful than more samples.
"""
from easydict import EasyDict

config = EasyDict(dict(
    seed=42,
    env=dict(
        env_id='CartPole-v1',
        max_steps=200,
        state_dim=4,
        action_dim=2,       # one-hot encoded dimension
        num_actions=2,       # discrete action count
        # CartPole-v1 termination thresholds
        x_threshold=2.4,
        theta_threshold=0.20943951,  # 12 degrees in radians
    ),
    ensemble=dict(
        ensemble_size=5,
        elite_size=3,
        # 2×64 is ample for CartPole's near-linear dynamics.
        # Original PETS/MBPO used 4×200 for complex Mujoco envs.
        hidden_dims=[64, 64],
        learning_rate=1e-3,
        reward_size=1,
        holdout_ratio=0.2,
        max_epochs_since_update=5,
    ),
    planner=dict(
        type='cem',
        horizon=10,
        num_candidates=500,  # more samples, single pass (random shooting)
        num_elites=50,
        num_iterations=1,    # =1 → pure random shooting (fastest for binary actions)
    ),
    training=dict(
        num_seed_episodes=5,
        total_iterations=100,
        rollout_episodes_per_iter=1,
        model_train_epochs=50,
        batch_size=256,
    ),
    buffer=dict(
        max_size=100000,
    ),
))
