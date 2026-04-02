"""Six-feature extractor for PETS episodes (Figure 4).

Features per timestep:
  F1: Advantage  A_t = Q(s_t,a_t) - V(s_t)          (policy quality)
  F2: dV_t = (V_t - V_{t-k}) / k,  k=5              (value trend)
  F3: Prediction error  PE_t = ||ŝ_{t+1} - s_{t+1}|| (model reliability)
  F4: Uncertainty accel  d²U_t                        (worsening speed)
  F5: Training density   ρ_t = mean 5-NN dist         (distribution shift)
  F6: SNR  |V_t| / (U_t + ε)                         (decision confidence)
"""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors


class FeatureExtractor:
    """Extracts 6 danger-relevant features from episode trajectories."""

    def __init__(self, config, dynamics, training_states=None):
        """
        Args:
            config:  EasyDict (pets_cartpole config)
            dynamics: trained EnsembleDynamics instance
            training_states: np.array [N, 4] reference states for F5
        """
        self.config = config
        self.dynamics = dynamics
        self.model = dynamics.model
        self.E = config.ensemble.ensemble_size
        self.x_thresh = config.env.x_threshold
        self.theta_thresh = config.env.theta_threshold

        # F1 params
        self.n_rollout = 20        # candidates per rollout
        self.value_horizon = 5     # lookahead steps for value estimation

        # F2 params
        self.dv_window = 5

        # F4 params (ε for F6)
        self.snr_eps = 0.01

        # F5: build nearest-neighbor index
        self._nn = None
        if training_states is not None:
            self.set_training_states(training_states)

    def set_training_states(self, states: np.ndarray):
        """Build BallTree index from training states for F5."""
        self._nn = NearestNeighbors(n_neighbors=5, algorithm='ball_tree')
        self._nn.fit(states)

    # ------------------------------------------------------------------
    # F1: Per-action value estimation (batched over all timesteps)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_values(self, states: np.ndarray,
                       n_rollout: int = None, horizon: int = None):
        """Compute Q(s,a) for each action and V(s) for all states.

        For each action a ∈ {0,1}:
          1. One-step ensemble-mean prediction  ŝ' = s + E[Δs | s,a]
          2. From ŝ', random-shoot n_rollout trajectories of length `horizon`
          3. Q(s,a) ≈ 1 + mean(rollout returns from ŝ')

        Args:
            states: [T, 4]
        Returns:
            V: [T]  max_a Q(s,a)
            Q: {0: [T], 1: [T]}
        """
        n_rollout = n_rollout or self.n_rollout
        horizon = horizon or self.value_horizon
        T = states.shape[0]
        obs = torch.FloatTensor(states)
        model = self.model
        E = self.E

        Q = {}
        for a in range(2):
            # --- one-step prediction for action a ---
            act_oh = torch.zeros(T, 2)
            act_oh[:, a] = 1.0
            mean_all, _ = self.dynamics.predict(obs, act_oh)  # [E,T,5]
            # ensemble-mean delta → next state
            next_obs = obs + mean_all[:, :, 1:].mean(dim=0)  # [T,4]

            # check if first step already terminates
            term0 = ((next_obs[:, 0].abs() > self.x_thresh) |
                     (next_obs[:, 2].abs() > self.theta_thresh))

            # --- random rollout from next_obs ---
            TN = T * n_rollout
            cur = (next_obs.unsqueeze(1)
                   .expand(T, n_rollout, -1)
                   .reshape(TN, 4).clone())
            model_idx = torch.randint(0, E, (TN,))
            arange_TN = torch.arange(TN)

            alive = (~term0.unsqueeze(1)
                     .expand(T, n_rollout)
                     .reshape(TN))
            total = torch.ones(TN)  # reward 1 for the initial action

            for _ in range(horizon):
                if not alive.any():
                    break
                total += alive.float()
                rand_a = torch.randint(0, 2, (TN,))
                a_oh = F.one_hot(rand_a, 2).float()
                inp = model.normalize(torch.cat([cur, a_oh], 1))
                inp_e = inp.unsqueeze(0).expand(E, -1, -1)
                m, _ = model(inp_e)
                sel = m[model_idx, arange_TN]
                cur = cur + sel[:, 1:]
                term = ((cur[:, 0].abs() > self.x_thresh) |
                        (cur[:, 2].abs() > self.theta_thresh))
                alive = alive & ~term

            Q[a] = total.reshape(T, n_rollout).mean(dim=1).numpy()

        V = np.maximum(Q[0], Q[1])
        return V, Q

    # ------------------------------------------------------------------
    # F3 + F4-base: prediction error & uncertainty (batched)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_prediction_error_and_uncertainty(
            self, states: np.ndarray, actions: np.ndarray,
            next_states: np.ndarray):
        """Compute PE_t (F3) and U_t (base for F4/F6) for all steps.

        Args:
            states:      [T, 4]
            actions:     [T]
            next_states: [T, 4]
        Returns:
            PE: [T]  L2 prediction error
            U:  [T]  ensemble reward-prediction variance
        """
        obs = torch.FloatTensor(states)
        act = torch.LongTensor(actions)
        next_obs = torch.FloatTensor(next_states)

        act_oh = F.one_hot(act, 2).float()
        mean, _var = self.dynamics.predict(obs, act_oh)  # [E,T,5]

        # F3: ensemble-mean delta → predicted next state
        delta_pred = mean[:, :, 1:].mean(dim=0)          # [T,4]
        predicted = obs + delta_pred
        PE = (predicted - next_obs).pow(2).sum(dim=1).sqrt().numpy()

        # F4 base: reward-prediction variance across members
        reward_preds = mean[:, :, 0]                      # [E,T]
        U = reward_preds.var(dim=0).numpy()               # [T]

        return PE, U

    # ------------------------------------------------------------------
    # F5: training-data density
    # ------------------------------------------------------------------
    def compute_density(self, states: np.ndarray) -> np.ndarray:
        """Mean 5-NN distance to training states.

        Args:  states [T, 4]
        Returns: rho [T]
        """
        if self._nn is None:
            return np.zeros(states.shape[0])
        dists, _ = self._nn.kneighbors(states)   # [T, 5]
        return dists.mean(axis=1)

    # ------------------------------------------------------------------
    # Full extraction for one episode
    # ------------------------------------------------------------------
    def extract(self, episode_data: dict) -> np.ndarray:
        """Extract 6 features for every timestep of one episode.

        Args:
            episode_data: dict with keys
                'states'      np.array [T+1, 4]   (obs_0 … obs_T)
                'actions'     np.array [T]
                'next_states' np.array [T, 4]      (= states[1:])
        Returns:
            features: np.array [T, 6]
        """
        states = episode_data['states'][:-1]     # [T, 4]
        actions = episode_data['actions']         # [T]
        next_states = episode_data['next_states'] # [T, 4]
        T = len(actions)

        feats = np.zeros((T, 6), dtype=np.float32)

        # --- F1 (advantage) + values for F2/F6 ---
        V, Q = self.compute_values(states)
        Q_actual = np.where(actions == 0, Q[0], Q[1])
        feats[:, 0] = Q_actual - V                      # F1: A_t

        # --- F2: value change rate ---
        k = self.dv_window
        for t in range(k, T):
            feats[t, 1] = (V[t] - V[t - k]) / k        # F2: dV_t

        # --- F3 (prediction error) + F4-base (uncertainty) ---
        PE, U = self.compute_prediction_error_and_uncertainty(
            states, actions, next_states)
        feats[:, 2] = PE                                 # F3: PE_t

        # --- F4: uncertainty acceleration (2nd-order difference) ---
        for t in range(2, T):
            feats[t, 3] = U[t] - 2 * U[t - 1] + U[t - 2]

        # --- F5: training-data density ---
        feats[:, 4] = self.compute_density(states)       # F5: ρ_t

        # --- F6: signal-to-noise ratio ---
        feats[:, 5] = np.abs(V) / (U + self.snr_eps)    # F6: SNR_t

        return feats
