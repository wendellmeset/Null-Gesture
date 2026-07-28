"""Motion Gesture Detector: IMU-driven classification of dynamic hand motions.

Covers: Clockwise, Anti-Clockwise, Left, Right, Bye-Bye, One Arm Boxing, Two-Arm Boxing.

Uses three complementary methods:
1. DTW for template-matching gestures (CW, ACW, Left, Right)
2. HMM for sequential/periodic gestures (Bye-Bye, Boxing)
3. Random Forest as a catch-all statistical classifier
"""

from __future__ import annotations

import math
from collections import deque
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
    ) -> None:
        self._window = IMUWindow(window_samples)

        # ── Adaptive baseline via rolling percentile ──────────────────
        # 5th percentile over 500 samples (5s at 100Hz). A 2s gesture
        # occupies at most 40% of history, so resting samples dominate.
        self._gmag_history: deque[float] = deque(maxlen=500)

        # EMA state for sustained direction (survives window flips)
        self._gz_ema: float = 0.0
        self._gx_ema: float = 0.0

    def push(
        self,
        ax: float, ay: float, az: float,
        gx: float, gy: float, gz: float,
    ) -> None:
        self._window.push(ax, ay, az, gx, gy, gz)

        # Record for baseline (no adaptation — pure history)
        self._gmag_history.append(math.sqrt(gx * gx + gy * gy + gz * gz))


    def detect(self) -> dict[str, float]:
        """Return belief masses using adaptive, ratio-based, multi-axis features.

        All thresholds are expressed as ratios of the adaptive baseline,
        making detection robust across different users, speeds, and mounting.

        Circle (CW/ACW): dominant-axis rotation that is steady and sustained.
        Swipe (Left/Right): velocity pulse from integrated acceleration.
        """
        result: dict[str, float] = {g: 0.0 for g in self.GESTURES}
        result["unknown"] = 1.0

        if not self._window.full:
            return result

        features = self._window.compute_features()
        if not features:
            return result

        ax = np.array(list(self._window._ax), dtype=np.float64)
        ay = np.array(list(self._window._ay), dtype=np.float64)
        az = np.array(list(self._window._az), dtype=np.float64)
        gx = np.array(list(self._window._gx), dtype=np.float64)
        gy = np.array(list(self._window._gy), dtype=np.float64)
        gz = np.array(list(self._window._gz), dtype=np.float64)
        amag = np.array(list(self._window._amag), dtype=np.float64)


        # ── Gyro magnitude & axis analysis ────────────────────────────
        gmag = np.sqrt(gx**2 + gy**2 + gz**2)
        gmag_mean = float(np.mean(gmag))
        gmag_std = float(np.std(gmag))

        # ── Adaptive baseline via 5th percentile of long history ─────
        if len(self._gmag_history) >= 50:
            bl_gmag = max(float(np.percentile(list(self._gmag_history), 5)), 5.0)
        else:
            bl_gmag = 12.0
        # Accel noise floor: BMI270 resting noise ≈ 0.01-0.03 g RMS.
        # Fixed value avoids gravity contamination of adaptive tracking.
        bl_anoise = 0.02

        # ── Accel features for swipe detection ─────────────────────────
        # Demean ax for velocity integration (removes DC offset)
        ax_dm = ax - np.mean(ax)

        # Integrate → velocity (cumulative sum, no attenuation)
        vx = np.cumsum(ax_dm)
        vx_range = float(np.ptp(vx))

        # Swipe asymmetry from RAW ax (gravity on az, so ax is centered at 0)
        ax_raw_min = float(np.min(ax))
        ax_raw_max = float(np.max(ax))
        ax_raw_range = ax_raw_max - ax_raw_min
        ax_asymmetry = abs(ax_raw_max + ax_raw_min) / (ax_raw_range + 1e-6)
        # ── Dominant rotation axis via RMS (mean=0 for sine waves) ──
        gx_rms = float(np.sqrt(np.mean(gx**2)))
        gy_rms = float(np.sqrt(np.mean(gy**2)))
        gz_rms = float(np.sqrt(np.mean(gz**2)))
        gmag_rms = float(np.sqrt(np.mean(gmag**2)))

        g_rms = np.array([gx_rms, gy_rms, gz_rms])
        dominant_idx = int(np.argmax(g_rms))
        dominant_strength = float(g_rms[dominant_idx] / (gmag_rms + 1e-6))

        # Gyro steadiness: 1.0 = constant speed, 0.0 = wildly variable
        gyro_steadiness = 1.0 - min(gmag_std / (gmag_mean + 1e-6), 1.0)

        # ── Update EMAs for direction persistence across windows ──────
        dom_seq = [gx, gy, gz][dominant_idx]
        dom_integral = float(np.sum(dom_seq))
        self._gz_ema = self._gz_ema * 0.85 + float(np.sum(gz)) * 0.15
        self._gx_ema = self._gx_ema * 0.85 + float(np.sum(gx)) * 0.15

        # ═══════════════════════════════════════════════════════════════
        # SOFT SCORING: both gesture families score independently.
        # No mutual-exclusion gates — the evidence decides.
        # ═══════════════════════════════════════════════════════════════

        # ── Swipe evidence (accel-driven) ─────────────────────────────
        vx_snr = vx_range / (bl_anoise * 20.0 + 1e-6)
        swipe_mag = min(1.0, vx_snr / 6.0)          # accel strength
        swipe_shape = ax_asymmetry                    # unidirectional pulse
        swipe_evidence = swipe_mag * swipe_shape

        # ── Circle evidence (gyro-driven) ─────────────────────────────
        # How many multiples of baseline is the gyro?
        gyro_elevation = max(0.0, min(1.0, (gmag_mean / max(bl_gmag, 1.0) - 1.5) / 3.0))
        circle_evidence = gyro_elevation * dominant_strength * gyro_steadiness

        # ── Direction assignment ──────────────────────────────────────
        cw_score = 0.0
        acw_score = 0.0
        left_score = 0.0
        right_score = 0.0

        # Circle direction: sign of integrated dominant axis
        if circle_evidence > 0.05:
            base = min(1.0, circle_evidence * 1.2)
            # Standard wrist mount: CW → negative dominant integral
            if dom_integral < 0:
                cw_score = base
            else:
                acw_score = base
            # Tilted-wrist penalty
            if dominant_idx != 2:
                cw_score *= 0.7
                acw_score *= 0.7
        # Swipe direction: gyro_x EMA (persistent) + vx trend (per-window)
        if swipe_evidence > 0.05:
            base = min(1.0, swipe_evidence * 1.2)
            # Primary: gyro_x rotation direction (persistent across windows)
            gx_dir = 1.0 if self._gx_ema > 0 else -1.0
            # Secondary: net change in vx over the window
            vx_dir = 1.0 if vx[-1] > vx[0] else -1.0
            direction = gx_dir + vx_dir
            if direction > 0:
                left_score = base
            elif direction < 0:
                right_score = base

        # ═══════════════════════════════════════════════════════════════
        # BOXING DETECTION (unchanged logic, ratio-adapted thresholds)
        # ═══════════════════════════════════════════════════════════════
        amag_peaks = features.get("amag_peak_count", 0.0)
        amag_std = features.get("amag_std", 0.0)
        amag_range = features.get("amag_range", 0.0)

        boxing_score = 0.0
        # Boxing: repetitive acceleration spikes
        # Use ratio of amag_std to baseline noise instead of absolute 0.8
        if amag_peaks >= 1 and amag_std > bl_anoise * 40.0:
            boxing_score = min(1.0, (amag_peaks / 4.0) * (amag_range / max(bl_anoise * 60.0, 0.5)))

        # ── Assemble ──────────────────────────────────────────────────
        combined = {
            "clockwise": cw_score,
            "anti_clockwise": acw_score,
            "left": left_score,
            "right": right_score,
            "bye_bye": 0.0,  # IMU-only bye-bye disabled (needs UWB distance)
            "one_arm_boxing": boxing_score * 0.7,
            "two_arm_boxing": boxing_score * 0.5,
        }
        total = sum(combined.values())
        if total > 0.9:
            scale = 0.9 / total
            for g in self.GESTURES:
                result[g] = combined[g] * scale
        else:
            for g in self.GESTURES:
                result[g] = combined[g]
        result["unknown"] = 1.0 - sum(result[g] for g in self.GESTURES)

        return result
