"""DangerScorer — standalone module for Phase 5 search.

Wraps the trained EBM to provide danger_score(features) → float.
Uses mean per-step score with logit transform to amplify differences
in the high-probability regime, giving the GA a stronger gradient.
"""
import pickle
import numpy as np


class DangerScorer:
    """EBM-based danger score predictor."""

    def __init__(self, ebm_model_path: str):
        with open(ebm_model_path, 'rb') as f:
            self.ebm = pickle.load(f)

    def score(self, features: np.ndarray) -> float:
        """Single-step danger score ∈ [0, 1]."""
        x = np.asarray(features, dtype=np.float64).reshape(1, -1)
        return float(self.ebm.predict_proba(x)[0, 1])

    def score_batch(self, features: np.ndarray) -> np.ndarray:
        """Batch danger scores ∈ [0, 1]."""
        return self.ebm.predict_proba(
            np.asarray(features, dtype=np.float64))[:, 1]

    def score_episode(self, episode_features: np.ndarray) -> float:
        """Episode-level danger score using mean + logit transform.

        The logit transform maps p ∈ (0,1) → ℝ, amplifying small
        differences near p≈0.8 where success/failure episodes cluster.
        This gives the GA a much stronger selection gradient than raw
        probabilities.

        Returns:
            float — higher = more dangerous (unbounded above 0)
        """
        scores = self.score_batch(episode_features)
        mean_p = float(np.clip(scores.mean(), 1e-6, 1 - 1e-6))
        return np.log(mean_p / (1.0 - mean_p))  # logit
