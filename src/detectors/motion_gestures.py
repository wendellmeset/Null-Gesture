"""Motion Gesture Detector: IMU-driven classification of dynamic hand motions.

Covers: Clockwise, Anti-Clockwise, Left, Right, Bye-Bye, One Arm Boxing, Two-Arm Boxing.

Uses three complementary methods:
1. DTW for template-matching gestures (CW, ACW, Left, Right)
2. HMM for sequential/periodic gestures (Bye-Bye, Boxing)
3. Random Forest as a catch-all statistical classifier
"""

from __future__ import annotations

import math
import pickle
from collections import deque
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from src.features.imu_features import IMUWindow

# ── DTW (Dynamic Time Warping) ───────────────────────────────────────────


def _dtw_distance(seq_a: np.ndarray, seq_b: np.ndarray) -> float:
    """Compute DTW distance between two 1D sequences."""
    n, m = len(seq_a), len(seq_b)
    if n == 0 or m == 0:
        return float("inf")

    dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dtw[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(seq_a[i - 1] - seq_b[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])

    return float(dtw[n, m] / (n + m))


def _normalize_seq(seq: np.ndarray) -> np.ndarray:
    """Z-score normalize a sequence."""
    std = np.std(seq)
    if std < 1e-10:
        return seq - np.mean(seq)
    return (seq - np.mean(seq)) / std


# ── DTW Templates ────────────────────────────────────────────────────────

# Reference templates: normalized gyro_z trajectory for circles,
# accel_x trajectory for left/right.
# Values are approximate — can be replaced with recorded templates.


def _default_templates() -> dict[str, np.ndarray]:
    """Generate synthetic reference templates. Replace with recorded data."""
    t = np.linspace(0, 2 * math.pi, 50)

    return {
        "clockwise": _normalize_seq(np.sin(t)),
        "anti_clockwise": _normalize_seq(-np.sin(t)),
        # Right-wrist mount, X toward fingers:
        # +X accel (hand moves left) → positive hump → matches "left"
        # -X accel (hand moves right) → negative hump → matches "right"
        "left": _normalize_seq(np.concatenate([
            np.linspace(0, 2, 20),
            np.linspace(2, 0, 20),
            np.zeros(10),
        ])),
        "right": _normalize_seq(np.concatenate([
            np.linspace(0, -2, 20),
            np.linspace(-2, 0, 20),
            np.zeros(10),
        ])),
    }


# ── HMM (Hidden Markov Model) ────────────────────────────────────────────


class SimpleHMM:
    """Minimal Gaussian-emission HMM for gesture sequence modeling.

    States: hidden phases of the gesture.
    Observations: feature vectors (modeled as diagonal Gaussian per state).
    """

    def __init__(
        self,
        n_states: int,
        n_features: int,
        name: str = "hmm",
    ) -> None:
        self.n_states = n_states
        self.n_features = n_features
        self.name = name

        # Parameters (can be loaded from training)
        self.startprob_: np.ndarray = np.ones(n_states) / n_states
        self.transmat_: np.ndarray = np.ones((n_states, n_states)) / n_states
        self.means_: np.ndarray = np.zeros((n_states, n_features))
        self.covars_: np.ndarray = np.ones((n_states, n_features))

    def _emission_logprob(self, obs: np.ndarray) -> np.ndarray:
        """Compute log probability of observation under each state's Gaussian."""
        logprob = np.zeros(self.n_states, dtype=np.float64)
        for s in range(self.n_states):
            diff = obs - self.means_[s]
            # Diagonal Gaussian
            logprob[s] = -0.5 * np.sum(
                (diff ** 2) / (self.covars_[s] + 1e-10)
                + np.log(2 * math.pi * (self.covars_[s] + 1e-10))
            )
        return logprob

    def score(self, observations: np.ndarray) -> float:
        """Compute log-likelihood of observation sequence under this HMM.

        Uses the forward algorithm.
        observations: (T, n_features)
        """
        T = len(observations)
        if T == 0:
            return -float("inf")

        # Forward algorithm
        alpha = np.zeros((T, self.n_states), dtype=np.float64)

        # Initialization
        log_emission = self._emission_logprob(observations[0])
        alpha[0] = np.log(self.startprob_ + 1e-10) + log_emission

        # Induction
        for t in range(1, T):
            log_emission = self._emission_logprob(observations[t])
            for j in range(self.n_states):
                alpha[t, j] = log_emission[j] + self._logsumexp(
                    alpha[t - 1] + np.log(self.transmat_[:, j] + 1e-10)
                )

        return float(self._logsumexp(alpha[T - 1]))

    @staticmethod
    def _logsumexp(x: np.ndarray) -> float:
        """Numerically stable log-sum-exp."""
        max_x = np.max(x)
        if np.isinf(max_x) and max_x < 0:
            return -float("inf")
        return float(max_x + np.log(np.sum(np.exp(x - max_x))))


# ── Random Forest (minimal) ──────────────────────────────────────────────


class MinimalRandomForest:
    """Tiny random forest for gesture classification.

    For production use, replace with sklearn. This is a self-contained
    fallback that works without external ML dependencies.
    """

    def __init__(self, n_trees: int = 30, max_depth: int = 8) -> None:
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees: list[dict[str, Any]] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train a random forest. X: (N, F), y: (N,)."""
        self.trees = []
        n_samples = X.shape[0]
        self.classes_ = np.unique(y)

        for _ in range(self.n_trees):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot, y_boot = X[indices], y[indices]
            tree = self._build_tree(X_boot, y_boot, depth=0)
            self.trees.append(tree)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> dict[str, Any]:
        n_samples = len(X)
        unique_classes = np.unique(y)

        # Stopping criteria
        if depth >= self.max_depth or len(unique_classes) == 1 or n_samples < 5:
            # Leaf: majority class
            counts = {c: int(np.sum(y == c)) for c in unique_classes}
            return {"type": "leaf", "class": max(counts, key=lambda k: counts[k]), "counts": counts}

        # Random subset of features
        n_features = X.shape[1]
        feature_subset = np.random.choice(
            n_features, max(1, int(math.sqrt(n_features))), replace=False
        )

        # Best split
        best_gain = -float("inf")
        best_feat, best_thresh = -1, 0.0

        for feat in feature_subset:
            values = np.unique(X[:, feat])
            if len(values) < 2:
                continue
            thresholds = (values[:-1] + values[1:]) / 2
            for thresh in thresholds:
                left_mask = X[:, feat] <= thresh
                right_mask = ~left_mask
                if np.sum(left_mask) < 3 or np.sum(right_mask) < 3:
                    continue
                gain = self._gini_gain(y, left_mask, right_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh

        if best_feat < 0:
            # Can't split
            counts = {c: int(np.sum(y == c)) for c in unique_classes}
            return {"type": "leaf", "class": max(counts, key=lambda k: counts[k]), "counts": counts}

        # Split
        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask
        return {
            "type": "node",
            "feature": best_feat,
            "threshold": best_thresh,
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[right_mask], y[right_mask], depth + 1),
        }

    def _gini_gain(self, y: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
        def gini(subset: np.ndarray) -> float:
            _, counts = np.unique(subset, return_counts=True)
            probs = counts / len(subset)
            return 1.0 - float(np.sum(probs ** 2))

        parent_gini = gini(y)
        n = len(y)
        left_gini = gini(y[left])
        right_gini = gini(y[right])
        return parent_gini - (np.sum(left) / n) * left_gini - (np.sum(right) / n) * right_gini

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities. X: (N, F) → (N, n_classes)."""
        n_samples = X.shape[0] if X.ndim > 1 else 1
        if X.ndim == 1:
            X = X.reshape(1, -1)

        probs = np.zeros((n_samples, len(self.classes_)), dtype=np.float64)
        for i in range(n_samples):
            votes = np.zeros(len(self.classes_), dtype=np.float64)
            for tree in self.trees:
                cls = self._predict_tree(X[i], tree)
                if cls in self.classes_:
                    idx = np.where(self.classes_ == cls)[0][0]
                    votes[idx] += 1
            probs[i] = votes / len(self.trees)
        return probs

    def _predict_tree(self, x: np.ndarray, tree: dict) -> Any:
        if tree["type"] == "leaf":
            return tree["class"]
        if x[tree["feature"]] <= tree["threshold"]:
            return self._predict_tree(x, tree["left"])
        return self._predict_tree(x, tree["right"])

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"trees": self.trees, "classes_": self.classes_}, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.trees = data["trees"]
            self.classes_ = data["classes_"]


# ── Motion Gesture Detector ──────────────────────────────────────────────


class MotionGestureDetector:
    """Detects IMU-based motion gestures: CW, ACW, Left, Right,
    Bye-Bye, One-Arm Boxing, Two-Arm Boxing.
    """

    GESTURES: ClassVar[list[str]] = [
        "clockwise",
        "anti_clockwise",
        "left",
        "right",
        "bye_bye",
        "one_arm_boxing",
        "two_arm_boxing",
    ]

    def __init__(
        self,
        window_samples: int = 50,
        dtw_weight: float = 0.4,
        hmm_weight: float = 0.3,
        rf_weight: float = 0.3,
        rf_model_path: str | None = None,
    ) -> None:
        self._window = IMUWindow(window_samples)
        self._dtw_weight = dtw_weight
        self._hmm_weight = hmm_weight
        self._rf_weight = rf_weight

        # DTW templates
        self._templates = _default_templates()

        # HMMs (per gesture)
        self._hmms: dict[str, SimpleHMM] = {}

        # Random Forest
        self._rf = MinimalRandomForest(n_trees=30, max_depth=8)
        if rf_model_path and Path(rf_model_path).exists():
            self._rf.load(rf_model_path)

        # Observation accumulator for HMM
        self._obs_buffer: deque[np.ndarray] = deque(maxlen=50)

        # Simple feature names used by RF
        self._feature_names: list[str] = []

    def push(
        self,
        ax: float, ay: float, az: float,
        gx: float, gy: float, gz: float,
    ) -> None:
        self._window.push(ax, ay, az, gx, gy, gz)

    def detect(self) -> dict[str, float]:
        """Return belief masses for motion gestures using physics-informed rules.

        - CW/ACW: gyro_z mean sign + DTW on z-score-normalized sequences
        - Left/Right: accel_x transient peak detection
        - Bye-Bye: high ZCR on gyro_z (oscillation) + low |mean| (centered)
        - Boxing: accel magnitude peaks
        """
        result: dict[str, float] = {g: 0.0 for g in self.GESTURES}
        result["unknown"] = 1.0

        if not self._window.full:
            return result

        features = self._window.compute_features()
        if not features:
            return result

        ax_seq = np.array(list(self._window._ax), dtype=np.float64)
        ay_seq = np.array(list(self._window._ay), dtype=np.float64)
        az_seq = np.array(list(self._window._az), dtype=np.float64)
        gz_seq = np.array(list(self._window._gz), dtype=np.float64)
        amag_seq = np.array(list(self._window._amag), dtype=np.float64)

        gz_mean = float(np.mean(gz_seq))
        gz_std = float(np.std(gz_seq))
        gz_zcr = features.get("gz_zcr", 0.0)
        ax_mean = float(np.mean(ax_seq))
        ax_std = float(np.std(ax_seq))
        ax_range = float(np.ptp(ax_seq))

        # ── Bye-Bye: oscillating gyro_z (check FIRST — overrides circle) ──
        oscillation_ratio = gz_std / max(abs(gz_mean), 0.05)
        is_oscillation = gz_zcr > 0.02 and oscillation_ratio > 2.0 and gz_std > 1.0

        # ── CW / ACW: constant rotation (only if NOT oscillating) ─────
        circle_magnitude = abs(gz_mean)
        is_circle_like = (not is_oscillation) and circle_magnitude > 0.4 and gz_zcr < 0.12

        cw_score = 0.0
        acw_score = 0.0
        bye_score = 0.0
        left_score = 0.0
        right_score = 0.0

        if is_oscillation:
            bye_score = min(1.0, gz_zcr * 4.0 * min(1.0, gz_std / 5.0))

        elif is_circle_like:
            base = min(1.0, circle_magnitude / 1.5)
            if gz_mean > 0:
                cw_score = base
            else:
                acw_score = base
            # DTW corroboration
            gz_norm = _normalize_seq(gz_seq)
            cw_tmpl = _normalize_seq(self._templates.get("clockwise", np.zeros(1)))
            acw_tmpl = _normalize_seq(self._templates.get("anti_clockwise", np.zeros(1)))
            if _dtw_distance(gz_norm, cw_tmpl) < _dtw_distance(gz_norm, acw_tmpl):
                cw_score = min(1.0, cw_score * 1.4)
                acw_score *= 0.4
            else:
                acw_score = min(1.0, acw_score * 1.4)
                cw_score *= 0.4

        else:
            # ── Left / Right: accel transient + gyro_z direction ────
            # Both gestures produce similar accel_x profiles after DC block;
            # gyro_z sign reliably distinguishes them (wrist rotation direction).
            if len(ax_seq) >= 15:
                ax_sma = np.convolve(ax_seq, np.ones(10)/10, mode='same')
                ax_hp = ax_seq - ax_sma
            else:
                ax_hp = ax_seq
            ax_hp_range = float(np.ptp(ax_hp))
            ax_hp_std = float(np.std(ax_hp))
            ax_hp_max = float(np.max(np.abs(ax_hp)))
            if ax_hp_range > 0.15 and ax_hp_range > ax_hp_std * 1.8:
                base = min(1.0, ax_hp_max / 0.8)
                # Use gyro_z sign for direction — rightward wrist motion
                # produces +Z rotation, leftward produces -Z rotation
                if gz_mean > 0.05:
                    right_score = min(1.0, base * 1.3)
                    left_score = base * 0.15
                elif gz_mean < -0.05:
                    left_score = min(1.0, base * 1.3)
                    right_score = base * 0.15
                else:
                    # No clear gyro direction — use DTW as tiebreaker
                    ax_norm = _normalize_seq(ax_hp)
                    left_tmpl = _normalize_seq(self._templates.get("left", np.zeros(1)))
                    right_tmpl = _normalize_seq(self._templates.get("right", np.zeros(1)))
                    if _dtw_distance(ax_norm, left_tmpl) < _dtw_distance(ax_norm, right_tmpl):
                        left_score = min(1.0, base * 1.3)
                    else:
                        right_score = min(1.0, base * 1.3)
        # ── Boxing: accel magnitude peaks ─────────────────────────────
        amag_peaks = features.get("amag_peak_count", 0.0)
        amag_std = features.get("amag_std", 0.0)
        amag_range = features.get("amag_range", 0.0)

        boxing_score = 0.0
        if amag_peaks >= 1 and amag_std > 0.8:
            boxing_score = min(1.0, (amag_peaks / 4.0) * (amag_range / 3.0))

        # ── Assemble ──────────────────────────────────────────────────
        combined = {
            "clockwise": cw_score,
            "anti_clockwise": acw_score,
            "left": left_score,
            "right": right_score,
            "bye_bye": bye_score,
            "one_arm_boxing": boxing_score * 0.7,
            "two_arm_boxing": boxing_score * 0.5,
        }

        total = sum(combined.values())
        if total > 0:
            scale = 0.9 / total
            for g in self.GESTURES:
                result[g] = combined[g] * scale
            result["unknown"] = 1.0 - sum(result[g] for g in self.GESTURES)

        return result
