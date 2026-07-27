"""Temporal filtering and state machine for gesture detection.

Applies hysteresis, cooldown, minimum duration, and Kalman smoothing to
raw fused beliefs from the Dempster-Shafer engine.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any


class GestureState(Enum):
    IDLE = auto()
    CANDIDATE = auto()      # belief rising, not yet confirmed
    ACTIVE = auto()          # gesture confirmed, emitting events
    COOLDOWN = auto()        # gesture ended, waiting before next detection


@dataclass
class GestureEvent:
    """Emitted when a gesture is detected or ends."""
    gesture: str
    phase: str          # "start", "active", "end"
    confidence: float
    timestamp: float
    duration: float     # how long the gesture has been active (for "active"/"end")
    contributing_sensors: list[str] = field(default_factory=list)


class TemporalStateMachine:
    """Filters raw fused beliefs through temporal logic.

    - Hysteresis: onset > offset threshold
    - Minimum duration: gesture must persist ≥ N ms before emitting
    - Cooldown: gap between successive gestures
    - Kalman filter: smooths belief trajectory
    """

    def __init__(
        self,
        onset_threshold: float = 0.6,
        offset_threshold: float = 0.3,
        min_duration_ms: int = 200,
        cooldown_ms: int = 300,
        kalman_process_noise: float = 0.01,
        kalman_measurement_noise: float = 0.1,
    ) -> None:
        self._onset_threshold = onset_threshold
        self._offset_threshold = offset_threshold
        self._min_duration = min_duration_ms / 1000.0  # seconds
        self._cooldown = cooldown_ms / 1000.0

        # Kalman filter state per gesture (index → filter state)
        self._kalman_state: dict[int, tuple[float, float]] = {}  # idx → (belief, velocity)
        self._kalman_Q = kalman_process_noise
        self._kalman_R = kalman_measurement_noise

        # State machine state
        self._state: GestureState = GestureState.IDLE
        self._active_gesture: str | None = None
        self._active_start: float = 0.0
        self._active_confidence: float = 0.0
        self._last_event_time: float = 0.0

        # Candidate tracking
        self._candidate_gesture: str | None = None
        self._candidate_start: float = 0.0
        self._candidate_confidence_sum: float = 0.0
        self._candidate_count: int = 0

    def update(self, fused: dict[str, Any]) -> GestureEvent | None:
        """Process one fused detection result. Returns GestureEvent if state changed.

        Args:
            fused: dict from DempsterShaferFusion.fuse()

        Returns:
            GestureEvent or None (no state change).
        """
        now = time.time()
        _gesture = fused["gesture"]
        _raw_confidence = fused["confidence"]
        beliefs = fused["beliefs"]

        # Apply Kalman filter to smooth beliefs
        smoothed = self._kalman_smooth_all(beliefs, now)

        # Find best gesture from smoothed beliefs
        best_gesture = max(smoothed, key=lambda k: smoothed[k])
        best_confidence = smoothed[best_gesture]

        event: GestureEvent | None = None

        if self._state == GestureState.IDLE:
            event = self._handle_idle(now, best_gesture, best_confidence, fused)

        elif self._state == GestureState.CANDIDATE:
            event = self._handle_candidate(now, best_gesture, best_confidence, fused)

        elif self._state == GestureState.ACTIVE:
            event = self._handle_active(now, best_gesture, best_confidence, fused)

        elif self._state == GestureState.COOLDOWN:
            event = self._handle_cooldown(now, best_gesture, best_confidence, fused)

        return event

    def _handle_idle(
        self, now: float, gesture: str | None, confidence: float, fused: dict
    ) -> GestureEvent | None:
        if gesture is not None and confidence > self._onset_threshold:
            # Start candidate phase
            self._state = GestureState.CANDIDATE
            self._candidate_gesture = gesture
            self._candidate_start = now
            self._candidate_confidence_sum = confidence
            self._candidate_count = 1
        return None

    def _handle_candidate(
        self, now: float, gesture: str | None, confidence: float, fused: dict
    ) -> GestureEvent | None:
        # Same gesture continuing
        if gesture == self._candidate_gesture and confidence > self._offset_threshold:
            self._candidate_confidence_sum += confidence
            self._candidate_count += 1

            elapsed = now - self._candidate_start
            avg_confidence = self._candidate_confidence_sum / self._candidate_count

            if elapsed >= self._min_duration and avg_confidence > self._onset_threshold:
                # Confirm gesture
                self._state = GestureState.ACTIVE
                self._active_gesture = gesture
                self._active_start = self._candidate_start
                self._active_confidence = avg_confidence

                return GestureEvent(
                    gesture=gesture or "unknown",
                    phase="start",
                    confidence=avg_confidence,
                    timestamp=now,
                    duration=0.0,
                    contributing_sensors=fused.get("contributing_sensors", []),
                )
        else:
            # Lost the candidate
            self._state = GestureState.IDLE
            self._candidate_gesture = None

        return None

    def _handle_active(
        self, now: float, gesture: str | None, confidence: float, fused: dict
    ) -> GestureEvent | None:
        duration = now - self._active_start

        if gesture == self._active_gesture and confidence > self._offset_threshold:
            # Still active — emit "active" event for continuous tracking
            return GestureEvent(
                gesture=self._active_gesture or "unknown",
                phase="active",
                confidence=confidence,
                timestamp=now,
                duration=duration,
                contributing_sensors=fused.get("contributing_sensors", []),
            )
        else:
            # Gesture ended
            event = GestureEvent(
                gesture=self._active_gesture or "unknown",
                phase="end",
                confidence=self._active_confidence,
                timestamp=now,
                duration=duration,
                contributing_sensors=fused.get("contributing_sensors", []),
            )
            self._active_gesture = None
            self._last_event_time = now
            self._state = GestureState.COOLDOWN
            return event

    def _handle_cooldown(
        self, now: float, gesture: str | None, confidence: float, fused: dict
    ) -> GestureEvent | None:
        if now - self._last_event_time >= self._cooldown:
            self._state = GestureState.IDLE
            # Re-check: is there already a candidate?
            if gesture is not None and confidence > self._onset_threshold:
                self._state = GestureState.CANDIDATE
                self._candidate_gesture = gesture
                self._candidate_start = now
                self._candidate_confidence_sum = confidence
                self._candidate_count = 1
        return None

    # ── Kalman filtering ────────────────────────────────────────────────

    def _kalman_smooth_all(self, beliefs: dict[str, float], now: float) -> dict[str, float]:
        """Apply 1D Kalman filter to all gesture beliefs.

        Returns smoothed belief dict.
        """
        smoothed: dict[str, float] = {}
        dt = 0.01  # ~100 Hz

        for gesture, raw in beliefs.items():
            idx = hash(gesture)  # Use hash as key (collisions possible but rare)

            if idx not in self._kalman_state:
                self._kalman_state[idx] = (raw, 0.0)

            belief, velocity = self._kalman_state[idx]

            # Predict
            belief_pred = belief + velocity * dt
            velocity_pred = velocity
            P_pred = self._kalman_R + self._kalman_Q  # simplified 1D

            # Update
            K = P_pred / (P_pred + self._kalman_R)
            belief_new = belief_pred + K * (raw - belief_pred)
            velocity_new = velocity_pred + K * ((raw - belief_pred) / max(dt, 0.001))

            self._kalman_state[idx] = (belief_new, velocity_new)
            smoothed[gesture] = max(0.0, min(1.0, belief_new))

        return smoothed

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def state(self) -> GestureState:
        return self._state

    @property
    def active_gesture(self) -> str | None:
        return self._active_gesture

    def reset(self) -> None:
        self._state = GestureState.IDLE
        self._active_gesture = None
        self._candidate_gesture = None
        self._kalman_state.clear()
