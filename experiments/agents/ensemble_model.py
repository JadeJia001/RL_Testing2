"""Probabilistic Ensemble Dynamics Model for PETS.

Each ensemble member is a probabilistic neural network:
    input:  (state, one_hot_action)
    output: (mean, logvar) of Gaussian over [reward, delta_state]

Training uses Gaussian NLL with logvar clamping for numerical stability.
Follows DI-engine's EnsembleModel patterns: parallel EnsembleFC layers,
Swish activation, truncated-normal init, holdout-based early stopping.
"""
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsembleLinear(nn.Module):
    """Fully-connected layer replicated across ensemble members.

    Weight shape: [ensemble_size, in_features, out_features]
    Forward via batched matmul — all members computed in one call.
    """

    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.in_features = in_features
        self.ensemble_size = ensemble_size
        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))
        self._init_weights()

    def _init_weights(self):
        for i in range(self.ensemble_size):
            std = 1.0 / (2.0 * math.sqrt(self.in_features))
            nn.init.trunc_normal_(self.weight[i], std=std)
            nn.init.zeros_(self.bias[i])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [E, B, in] -> [E, B, out]
        return torch.bmm(x, self.weight) + self.bias


class EnsembleDynamicsModel(nn.Module):
    """Probabilistic ensemble network.

    Predicts [reward, delta_state] with Gaussian output (mean + logvar).
    """

    def __init__(self, state_dim: int, action_dim: int, ensemble_size: int = 5,
                 hidden_dims: tuple = (200, 200, 200, 200), lr: float = 1e-3):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size
        self.output_dim = state_dim + 1  # reward + delta_state

        # Build layers
        in_dim = state_dim + action_dim
        layers = []
        for h_dim in hidden_dims:
            layers.append(EnsembleLinear(in_dim, h_dim, ensemble_size))
            in_dim = h_dim
        layers.append(EnsembleLinear(in_dim, self.output_dim * 2, ensemble_size))
        self.layers = nn.ModuleList(layers)

        # Logvar bounds (following DI-engine)
        self.max_logvar = nn.Parameter(
            0.5 * torch.ones(1, self.output_dim), requires_grad=False)
        self.min_logvar = nn.Parameter(
            -10.0 * torch.ones(1, self.output_dim), requires_grad=False)

        # Input normalization buffers
        self.register_buffer('input_mu', torch.zeros(1, state_dim + action_dim))
        self.register_buffer('input_std', torch.ones(1, state_dim + action_dim))

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def fit_scaler(self, data: torch.Tensor):
        """Fit input normalization statistics. data: [N, input_dim]."""
        mu = data.mean(dim=0, keepdim=True)
        std = data.std(dim=0, keepdim=True)
        std[std < 1e-12] = 1.0
        self.input_mu.copy_(mu)
        self.input_std.copy_(std)

    def normalize(self, data: torch.Tensor) -> torch.Tensor:
        return (data - self.input_mu) / self.input_std

    def forward(self, x: torch.Tensor):
        """Forward pass.
        Args:
            x: [ensemble_size, batch, input_dim] — pre-normalized.
        Returns:
            mean:   [ensemble_size, batch, output_dim]
            logvar: [ensemble_size, batch, output_dim] (clamped)
        """
        for layer in self.layers[:-1]:
            x = F.silu(layer(x))
        x = self.layers[-1](x)

        mean, logvar = x.chunk(2, dim=-1)
        # Soft-clamp logvar (DI-engine pattern)
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    def compute_loss(self, mean, logvar, labels):
        """Gaussian NLL loss across ensemble.
        Returns:
            total_loss: scalar (for backprop)
            mse_per_member: [ensemble_size] (for diagnostics / elite selection)
        """
        inv_var = torch.exp(-logvar)
        mse_inv = (torch.pow(mean - labels, 2) * inv_var).mean(dim=(1, 2))
        var_loss = logvar.mean(dim=(1, 2))
        total_loss = (mse_inv + var_loss).sum()
        # Regularize logvar bounds
        total_loss += 0.01 * self.max_logvar.sum() - 0.01 * self.min_logvar.sum()

        with torch.no_grad():
            mse = torch.pow(mean - labels, 2).mean(dim=(1, 2))
        return total_loss, mse


class EnsembleDynamics:
    """High-level wrapper: training loop, prediction, elite tracking."""

    def __init__(self, config):
        self.cfg = config
        E = config.ensemble.ensemble_size
        self.model = EnsembleDynamicsModel(
            state_dim=config.env.state_dim,
            action_dim=config.env.action_dim,
            ensemble_size=E,
            hidden_dims=tuple(config.ensemble.hidden_dims),
            lr=config.ensemble.learning_rate,
        )
        self.elite_indices = list(range(config.ensemble.elite_size))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, buffer) -> dict:
        """Train ensemble on all data in buffer.

        Uses holdout split, per-member shuffling, and early stopping.
        Returns dict with training metrics.
        """
        data = buffer.get_all_tensors()
        obs = data['obs']
        action = data['action']
        reward = data['reward']
        next_obs = data['next_obs']

        # Format inputs / labels
        action_oh = F.one_hot(action, self.cfg.env.action_dim).float()
        inputs = torch.cat([obs, action_oh], dim=1)
        delta_obs = next_obs - obs
        labels = torch.cat([reward.unsqueeze(1), delta_obs], dim=1)

        # Shuffle then split holdout
        N = inputs.shape[0]
        perm = torch.randperm(N)
        num_holdout = max(int(N * self.cfg.ensemble.holdout_ratio), 1)
        train_inputs = inputs[perm[num_holdout:]]
        train_labels = labels[perm[num_holdout:]]
        holdout_inputs = inputs[perm[:num_holdout]]
        holdout_labels = labels[perm[:num_holdout]]

        # Fit scaler on training data
        self.model.fit_scaler(train_inputs)
        train_inputs = self.model.normalize(train_inputs)
        holdout_inputs = self.model.normalize(holdout_inputs)

        E = self.cfg.ensemble.ensemble_size
        holdout_inputs_e = holdout_inputs.unsqueeze(0).expand(E, -1, -1)
        holdout_labels_e = holdout_labels.unsqueeze(0).expand(E, -1, -1)

        # Early-stopping bookkeeping
        best_holdout = [float('inf')] * E
        best_state = copy.deepcopy(self.model.state_dict())
        epochs_no_improve = 0

        N_train = train_inputs.shape[0]
        bs = min(self.cfg.training.batch_size, N_train)

        for epoch in range(self.cfg.training.model_train_epochs):
            # Each member sees differently shuffled data
            idx = torch.stack([torch.randperm(N_train) for _ in range(E)])

            for start in range(0, N_train, bs):
                b_idx = idx[:, start:start + bs]
                b_in = train_inputs[b_idx]
                b_lab = train_labels[b_idx]

                mean, logvar = self.model(b_in)
                loss, _ = self.model.compute_loss(mean, logvar, b_lab)
                self.model.optimizer.zero_grad()
                loss.backward()
                self.model.optimizer.step()

            # Holdout evaluation
            with torch.no_grad():
                h_mean, h_logvar = self.model(holdout_inputs_e)
                _, h_mse = self.model.compute_loss(h_mean, h_logvar, holdout_labels_e)

            improved = False
            for i in range(E):
                val = h_mse[i].item()
                if val < best_holdout[i]:
                    best_holdout[i] = val
                    improved = True
            if improved:
                best_state = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.cfg.ensemble.max_epochs_since_update:
                break

        # Restore best and select elites
        self.model.load_state_dict(best_state)
        with torch.no_grad():
            h_mean, h_logvar = self.model(holdout_inputs_e)
            _, h_mse = self.model.compute_loss(h_mean, h_logvar, holdout_labels_e)
        self.elite_indices = h_mse.argsort().tolist()[:self.cfg.ensemble.elite_size]

        return {'holdout_mse': h_mse.mean().item(), 'epochs': epoch + 1}

    # ------------------------------------------------------------------
    # Prediction (used by planner)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, obs: torch.Tensor, action_oh: torch.Tensor):
        """Predict next state for ALL ensemble members.
        Args:
            obs:       [B, state_dim]
            action_oh: [B, action_dim]
        Returns:
            mean: [E, B, output_dim]   (output_dim = reward + state_dim)
            var:  [E, B, output_dim]
        """
        inp = torch.cat([obs, action_oh], dim=1)
        inp = self.model.normalize(inp)
        inp = inp.unsqueeze(0).expand(self.model.ensemble_size, -1, -1)
        mean, logvar = self.model(inp)
        return mean, torch.exp(logvar)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def state_dict(self):
        return {
            'model': self.model.state_dict(),
            'elite_indices': self.elite_indices,
        }

    def load_state_dict(self, d):
        self.model.load_state_dict(d['model'])
        self.elite_indices = d['elite_indices']
