"""Gesture ensemble: HMM forward filter + debouncing + output formatting.

Applies a Hidden Markov Model forward filter to smooth per-frame gesture
predictions over time, then debounces the output to prevent flickering.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np

from src.models.imu_model import NUM_CLASSES, GESTURE_CLASSES, INDEX_TO_GESTURE, GESTURE_TO_INDEX


# ── HMM transition model ─────────────────────────────────────────────────
# States: IDLE (index 0) + 15 gesture classes (indices 1-15)
NUM_STATES = NUM_CLASSES + 1  # IDLE + 15 gestures
IDLE_STATE = 0


class GestureEnsemble:
    """HMM forward filter with gesture debouncing.

    Smooths per-frame predictions and outputs final gesture detections
    with high confidence and hysteresis.

    Usage::

        ensemble = GestureEnsemble()
        for fused_probs in pipeline:
            result = ensemble.update(fused_probs)
            if result:
                print(f"Detected: {result['gesture']}")
    """

    def __init__(
        self,
        self_stay_prob: float = 0.92,
        to_idle_prob: float = 0.15,
        idle_stay_prob: float = 0.98,
        detection_threshold: float = 0.7,
        debounce_frames: int = 3,
        cooldown_ms: float = 600.0,
        boxing_cooldown_ms: float = 300.0,
    ):
        """
        Args:
            self_stay_prob: P(stay in same gesture).
            to_idle_prob: P(transition to IDLE after any gesture).
            idle_stay_prob: P(stay in IDLE).
            detection_threshold: Minimum HMM-filtered probability to report.
            debounce_frames: Required consecutive frames with same argmax.
            cooldown_ms: Minimum time between non-boxing gesture reports.
            boxing_cooldown_ms: Minimum time between boxing gesture reports.
        """
        self._detection_threshold = detection_threshold
        self._debounce_frames = debounce_frames
        self._cooldown_ms = cooldown_ms
        self._boxing_cooldown_ms = boxing_cooldown_ms

        # Build transition matrix
        self._transition = self._build_transition_matrix(
            self_stay_prob, to_idle_prob, idle_stay_prob
        )

        # State
        self._alpha: np.ndarray = np.ones(NUM_STATES, dtype=np.float64) / NUM_STATES
        self._debounce_counter: dict[int, int] = {}
        self._last_gesture_idx: int = IDLE_STATE
        self._last_detection_time: float = 0.0
        self._detection_history: deque[dict] = deque(maxlen=50)

        # Boxing gesture indices (for shorter cooldown)
        self._boxing_indices = {
            GESTURE_TO_INDEX.get("one_arm_boxing", -1),
            GESTURE_TO_INDEX.get("two_arm_boxing", -1),
        }

    def _build_transition_matrix(
        self,
        self_stay: float,
        to_idle: float,
        idle_stay: float,
    ) -> np.ndarray:
        """Build (NUM_STATES × NUM_STATES) transition probability matrix."""
        T = np.zeros((NUM_STATES, NUM_STATES), dtype=np.float64)

        # From IDLE
        T[IDLE_STATE, IDLE_STATE] = idle_stay
        other_from_idle = (1.0 - idle_stay) / NUM_CLASSES
        for i in range(1, NUM_STATES):
            T[IDLE_STATE, i] = other_from_idle

        # From each gesture
        for g in range(1, NUM_STATES):
            T[g, IDLE_STATE] = to_idle
            T[g, g] = self_stay
            other_prob = (1.0 - self_stay - to_idle) / (NUM_CLASSES - 1)
            for j in range(1, NUM_STATES):
                if j != g:
                    T[g, j] = other_prob

        # Normalize rows
        for i in range(NUM_STATES):
            row_sum = T[i].sum()
            if row_sum > 0:
                T[i] /= row_sum

        return T

    def update(self, fused_probs: np.ndarray) -> dict | None:
        """Process one frame of fused probabilities.

        Args:
            fused_probs: (15,) probability vector over gesture classes.

        Returns:
            Detection result dict or None if no gesture detected this frame.
        """
        # Extend to include IDLE state
        # P(IDLE) = 1 - max(P(any gesture))
        obs_probs = np.zeros(NUM_STATES, dtype=np.float64)
        obs_probs[1:] = np.clip(fused_probs, 0.0, 1.0)
        obs_probs[IDLE_STATE] = max(0.0, 1.0 - np.max(fused_probs))

        # Normalize
        total = obs_probs.sum()
        if total > 0:
            obs_probs /= total

        # HMM forward filter
        # α_t = P(x_t | y_{1:t})
        prediction = self._transition.T @ self._alpha
        self._alpha = prediction * obs_probs
        total = self._alpha.sum()
        if total > 0:
            self._alpha /= total
        else:
            self._alpha = np.ones(NUM_STATES, dtype=np.float64) / NUM_STATES

        # Current most likely state
        best_idx = int(np.argmax(self._alpha))
        best_prob = float(self._alpha[best_idx])

        # Debounce: track consecutive occurrences of same argmax
        if best_idx != IDLE_STATE and best_idx == self._last_gesture_idx:
            self._debounce_counter[best_idx] = self._debounce_counter.get(best_idx, 0) + 1
        else:
            self._debounce_counter = {best_idx: 1}
        self._last_gesture_idx = best_idx

        # Check for detection
        if best_idx != IDLE_STATE and best_prob >= self._detection_threshold:
            count = self._debounce_counter.get(best_idx, 0)
            if count >= self._debounce_frames:
                # Check cooldown
                now = time.time()
                cooldown = (
                    self._boxing_cooldown_ms
                    if best_idx in self._boxing_indices
                    else self._cooldown_ms
                ) / 1000.0

                if now - self._last_detection_time >= cooldown:
                    self._last_detection_time = now
                    self._debounce_counter[best_idx] = 0  # Reset counter

                    result = {
                        "gesture": INDEX_TO_GESTURE.get(best_idx - 1, "unknown"),
                        "confidence": best_prob,
                        "latency_ms": 0.0,  # Will be filled by pipeline
                        "timestamp": now,
                    }
                    self._detection_history.append(result)
                    return result

        return None

    def reset(self) -> None:
        """Reset HMM state and history."""
        self._alpha = np.ones(NUM_STATES, dtype=np.float64) / NUM_STATES
        self._debounce_counter.clear()
        self._last_gesture_idx = IDLE_STATE
        self._last_detection_time = 0.0
        self._detection_history.clear()

    @property
    def current_state(self) -> int:
        """Current most-likely state index (0 = IDLE, 1-15 = gesture)."""
        return int(np.argmax(self._alpha))

    @property
    def current_probability(self) -> float:
        """Probability of the current most-likely state."""
        return float(np.max(self._alpha))

    @property
    def state_probabilities(self) -> np.ndarray:
        """Full state probability vector (IDLE + 15 gestures)."""
        return self._alpha.copy()

    @property
    def history(self) -> list[dict]:
        """Recent detection history."""
        return list(self._detection_history)
