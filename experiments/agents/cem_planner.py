"""CEM Planner for PETS with discrete action spaces.

For CartPole's 2-action space, CEM maintains per-timestep Bernoulli
probabilities p_t = P(action=1 at step t).  With num_iterations=1 this
reduces to pure random shooting — sample many candidates, pick the best.

Supports warm-starting between consecutive planning calls.
"""
import torch
import torch.nn.functional as F


class CEMPlanner:
    """Cross-Entropy Method planner adapted for discrete actions."""

    def __init__(self, config, dynamics):
        self.horizon = config.planner.horizon
        self.num_candidates = config.planner.num_candidates
        self.num_elites = config.planner.num_elites
        self.num_iterations = config.planner.num_iterations
        self.num_actions = config.env.num_actions
        self.ensemble_size = config.ensemble.ensemble_size
        self.x_threshold = config.env.x_threshold
        self.theta_threshold = config.env.theta_threshold
        self.dynamics = dynamics
        self._prev_probs = None
        # Pre-allocate reusable index tensor
        self._arange_N = torch.arange(self.num_candidates)

    def reset(self):
        """Reset warm-start state at the beginning of an episode."""
        self._prev_probs = None

    @torch.no_grad()
    def plan(self, obs) -> int:
        """Select the best action for the current observation."""
        obs_t = torch.FloatTensor(obs)

        # Warm-start from previous step or uniform
        if self._prev_probs is not None:
            probs = torch.cat([self._prev_probs[1:], torch.tensor([0.5])])
        else:
            probs = torch.full((self.horizon,), 0.5)

        best_action = 0
        best_reward = -float('inf')

        for _ in range(self.num_iterations):
            # Sample action sequences [N, H]
            action_seqs = torch.bernoulli(
                probs.unsqueeze(0).expand(self.num_candidates, -1)
            ).long()

            # TS-1: assign each candidate a random ensemble member
            model_idx = torch.randint(0, self.ensemble_size,
                                      (self.num_candidates,))

            # Evaluate rollouts
            rewards = self._evaluate(obs_t, action_seqs, model_idx)

            # Select elites
            elite_vals, elite_idx = rewards.topk(self.num_elites)
            elite_actions = action_seqs[elite_idx]

            # Update probabilities
            probs = elite_actions.float().mean(dim=0).clamp(0.05, 0.95)

            if elite_vals[0].item() > best_reward:
                best_reward = elite_vals[0].item()
                best_action = action_seqs[elite_idx[0], 0].item()

        self._prev_probs = probs.clone()
        return best_action

    def _evaluate(self, obs: torch.Tensor, action_seqs: torch.Tensor,
                  model_idx: torch.Tensor) -> torch.Tensor:
        """Rollout action sequences through ensemble (optimized inner loop).

        Minimizes per-step Python overhead by pre-computing one-hot actions
        and calling model.forward() directly.
        """
        N, H = action_seqs.shape
        E = self.dynamics.model.ensemble_size
        model = self.dynamics.model

        # Pre-compute all one-hot actions at once
        act_oh_all = F.one_hot(action_seqs, self.num_actions).float()  # [N, H, 2]

        current = obs.unsqueeze(0).expand(N, -1).clone()  # [N, 4]
        total_rewards = torch.zeros(N)
        alive = torch.ones(N, dtype=torch.bool)
        arange_N = self._arange_N

        for t in range(H):
            if not alive.any():
                break

            total_rewards += alive.float()

            # Build input and normalize
            inp = torch.cat([current, act_oh_all[:, t]], dim=1)  # [N, 6]
            inp = model.normalize(inp)
            inp_e = inp.unsqueeze(0).expand(E, -1, -1)  # [E, N, 6]

            # Forward through ensemble
            mean, _logvar = model(inp_e)  # [E, N, 5]

            # TS-1: select each candidate's model
            selected = mean[model_idx, arange_N]  # [N, 5]

            # Update state: next = current + delta (cols 1:)
            current = current + selected[:, 1:]

            # CartPole termination check
            terminated = ((current[:, 0].abs() > self.x_threshold) |
                          (current[:, 2].abs() > self.theta_threshold))
            alive = alive & ~terminated

        return total_rewards
