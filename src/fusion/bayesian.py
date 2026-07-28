"""Bayesian multi-modal fusion.

Combines per-sensor probability distributions using a weighted geometric
mean, weighted by each sensor's known reliability for each gesture class
and the current per-sensor confidence scores.
"""

from __future__ import annotations

import numpy as np

from src.models.imu_model import NUM_CLASSES, GESTURE_CLASSES, INDEX_TO_GESTURE


# ── Sensor reliability matrix ────────────────────────────────────────────
# R[gesture_idx, sensor] = expected reliability for that sensor on that gesture
# derived from physics of each gesture (see Docs/05-classification-and-fusion.md)

_SENSOR_RELIABILITY = np.array([
    # IMU, mmWave
    [0.7,  0.9 ],  # pull
    [0.7,  0.9 ],  # push
    [0.95, 0.6 ],  # clockwise
    [0.95, 0.6 ],  # anti_clockwise
    [0.6,  0.9 ],  # left
    [0.6,  0.9 ],  # right
    [0.85, 0.85],  # bye_bye
    [0.8,  0.8 ],  # one_arm_boxing
    [0.5,  0.9 ],  # clapping
    [0.3,  0.85],  # two_arm_boxing
    [0.5,  0.8 ],  # t_arms
    [0.8,  0.85],  # raise_arms
    [0.2,  0.85],  # soli
    [0.55, 0.4 ],  # opening_closing_fist
    [0.9,  0.3 ],  # palm_up_down
], dtype=np.float64)


class BayesianFusion:
    """Fuses IMU and mmWave predictions using weighted geometric mean.

    The fusion equation:
      P_fused(g) ∝ P_IMU(g)^w_IMU(g) × P_mmW(g)^w_mmW(g)

    where w_IMU(g) = R[g, IMU] × c_IMU
          w_mmW(g) = R[g, mmW] × c_mmW

    Usage::

        fuser = BayesianFusion()
        fused_probs, contributions = fuser.fuse(
            imu_probs, imu_confidence,
            mmwave_probs, mmwave_confidence,
        )
    """

    def __init__(self, reliability_matrix: np.ndarray | None = None):
        """
        Args:
            reliability_matrix: Custom (15, 2) reliability matrix.
                                 Uses built-in defaults if None.
        """
        if reliability_matrix is not None:
            if reliability_matrix.shape != (NUM_CLASSES, 2):
                raise ValueError(
                    f"Reliability matrix must be ({NUM_CLASSES}, 2), "
                    f"got {reliability_matrix.shape}"
                )
            self._reliability = reliability_matrix.astype(np.float64)
        else:
            self._reliability = _SENSOR_RELIABILITY.copy()

    def fuse(
        self,
        imu_probs: np.ndarray,
        imu_confidence: float,
        mmwave_probs: np.ndarray,
        mmwave_confidence: float,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Fuse two sensor probability distributions.

        Args:
            imu_probs: (15,) probability vector from IMU classifier.
            imu_confidence: IMU confidence scalar in [0, 1].
            mmwave_probs: (15,) probability vector from mmWave classifier.
            mmwave_confidence: mmWave confidence scalar in [0, 1].

        Returns:
            (fused_probs, contributions)
            fused_probs: (15,) fused probability vector.
            contributions: Dict with per-sensor weighted probabilities
                           {'imu': (15,), 'mmwave': (15,)}.
        """
        # Compute per-gesture weights
        imu_weights = self._reliability[:, 0] * imu_confidence
        mmwave_weights = self._reliability[:, 1] * mmwave_confidence

        # Clamp weights to avoid numerical issues
        eps = 1e-9
        imu_weights = np.clip(imu_weights, eps, 5.0)
        mmwave_weights = np.clip(mmwave_weights, eps, 5.0)

        # Weighted geometric mean
        # P_fused ∝ P_IMU^w_IMU × P_mmW^w_mmW
        # In log space: log(P_fused) ∝ w_IMU*log(P_IMU) + w_mmW*log(P_mmW)
        imu_probs_safe = np.clip(imu_probs, eps, 1.0)
        mmwave_probs_safe = np.clip(mmwave_probs, eps, 1.0)

        log_fused = (
            imu_weights * np.log(imu_probs_safe)
            + mmwave_weights * np.log(mmwave_probs_safe)
        )

        # Normalize (softmax in log space)
        log_fused -= np.max(log_fused)  # For numerical stability
        fused_probs = np.exp(log_fused)
        fused_probs /= fused_probs.sum()

        # Per-sensor contributions (weighted probabilities)
        contributions = {
            "imu": imu_probs * imu_weights.reshape(-1, 1).repeat(1, 1)[:, 0]
            if imu_probs.ndim == 1 else imu_probs,
            "mmwave": mmwave_probs * mmwave_weights.reshape(-1, 1).repeat(1, 1)[:, 0]
            if mmwave_probs.ndim == 1 else mmwave_probs,
        }

        # Normalize contributions for interpretability
        for key in contributions:
            total = contributions[key].sum()
            if total > 0:
                contributions[key] = contributions[key] / total

        return fused_probs, contributions

    def fuse_with_primitive_bias(
        self,
        imu_probs: np.ndarray,
        imu_confidence: float,
        mmwave_probs: np.ndarray,
        mmwave_confidence: float,
        primitive_scores: dict[str, float] | None = None,
        primitive_weight: float = 0.15,
    ) -> np.ndarray:
        """Fuse with additional bias from primitive detection scores.

        Args:
            imu_probs, imu_confidence, mmwave_probs, mmwave_confidence:
                Standard fusion inputs.
            primitive_scores: Dict mapping gesture_name → primitive_match_score.
                              Should be in [0, 1] per gesture.
            primitive_weight: How much to weight primitive evidence (0 = ignore).

        Returns:
            (15,) fused + primitive-biased probability vector.
        """
        fused_probs, _ = self.fuse(
            imu_probs, imu_confidence,
            mmwave_probs, mmwave_confidence,
        )

        if primitive_scores is None or primitive_weight <= 0:
            return fused_probs

        # Build primitive bias vector
        primitive_bias = np.ones(NUM_CLASSES, dtype=np.float64)
        for gesture_name, score in primitive_scores.items():
            if gesture_name in GESTURE_CLASSES:
                idx = GESTURE_CLASSES.index(gesture_name)
                # Boost gesture proportionally to its primitive match
                primitive_bias[idx] = 1.0 + score * primitive_weight

        # Apply bias and renormalize
        biased = fused_probs * primitive_bias
        biased /= biased.sum()
        return biased

    @staticmethod
    def top_gesture(probs: np.ndarray) -> str:
        """Return the gesture label with highest probability."""
        idx = int(np.argmax(probs))
        return INDEX_TO_GESTURE.get(idx, "unknown")

    @staticmethod
    def confidence(probs: np.ndarray) -> float:
        """Compute normalized confidence from a probability distribution."""
        max_entropy = np.log(NUM_CLASSES)
        eps = 1e-12
        entropy = -np.sum(probs * np.log(probs + eps))
        return float(1.0 - entropy / max_entropy)
