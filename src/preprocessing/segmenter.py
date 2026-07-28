"""Energy-based gesture segmentation with adaptive thresholding.

Detects gesture boundaries (onset/offset) from combined sensor activity
metrics without any machine learning. Uses adaptive thresholding with
hysteresis for robust detection.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum, auto


class SegmenterState(Enum):
    IDLE = auto()
    ACTIVE = auto()
    COOLDOWN = auto()


@dataclass
class GestureWindow:
    """A detected gesture time window."""
    onset_time: float
    offset_time: float
    duration_sec: float
    peak_metric: float
    source: str  # 'imu', 'mmwave', or 'fused'
    onset_timestamp: float
    offset_timestamp: float


class GestureSegmenter:
    """Detects gesture boundaries using adaptive energy thresholding.

    Algorithm:
    1. Combine IMU and mmWave signals into a single activity metric
    2. Track baseline noise level via EMA during quiet periods
    3. Adaptive threshold = noise * multiplier
    4. Hysteresis: ON > high_thresh, OFF < low_thresh for confirm_time
    5. Extract gesture windows on ACTIVE→IDLE transitions

    Usage::

        seg = GestureSegmenter()
        # Call for each new sensor data point:
        for imu_sample, mmwave_result in zip(imu_stream, mmwave_stream):
            window = seg.update(imu_sample, mmwave_result)
            if window:
                print(f"Gesture detected: {window.duration_sec:.2f}s")
    """

    def __init__(
        self,
        imu_accel_weight: float = 0.4,
        imu_gyro_weight: float = 0.3,
        mmwave_range_weight: float = 0.3,
        noise_ema_alpha: float = 0.01,
        threshold_on_mult: float = 2.5,
        threshold_off_mult: float = 1.8,
        min_duration_ms: float = 200.0,
        max_duration_ms: float = 5000.0,
        quiet_confirm_ms: float = 300.0,
        cooldown_ms: float = 500.0,
    ):
        """
        Args:
            imu_accel_weight: Weight for IMU linear acceleration norm.
            imu_gyro_weight: Weight for IMU gyroscope norm.
            mmwave_range_weight: Weight for mmWave range change rate.
            noise_ema_alpha: EMA smoothing factor for baseline noise.
            threshold_on_mult: Multiplier above noise to trigger ACTIVATION.
            threshold_off_mult: Multiplier above noise to trigger DEACTIVATION.
            min_duration_ms: Minimum gesture duration (shorter = rejected).
            max_duration_ms: Maximum gesture duration (longer = split).
            quiet_confirm_ms: Time below off-threshold to confirm gesture end.
            cooldown_ms: Minimum time between gesture detections.
        """
        # Weights
        self._w_accel = imu_accel_weight
        self._w_gyro = imu_gyro_weight
        self._w_range = mmwave_range_weight

        # Threshold parameters
        self._noise_ema_alpha = noise_ema_alpha
        self._threshold_on_mult = threshold_on_mult
        self._threshold_off_mult = threshold_off_mult
        self._min_duration_ms = min_duration_ms
        self._max_duration_ms = max_duration_ms
        self._quiet_confirm_ms = quiet_confirm_ms
        self._cooldown_ms = cooldown_ms

        # State
        self._state = SegmenterState.IDLE
        self._baseline_noise = 0.01  # Initial estimate
        self._onset_time: float | None = None
        self._onset_metric: float = 0.0
        self._peak_metric: float = 0.0
        self._quiet_start: float | None = None
        self._last_gesture_time: float = 0.0
        self._metric_history: deque[tuple[float, float]] = deque(maxlen=100)
        self._gesture_source = "fused"

        # Previous mmWave centroid for range change computation
        self._prev_mmwave_centroid: tuple[float, float, float] | None = None

    def update(
        self,
        imu_sample: dict | None = None,
        mmwave_result: dict | None = None,
    ) -> GestureWindow | None:
        """Process one set of sensor data and check for gesture boundaries.

        Args:
            imu_sample: Processed IMU sample dict (from IMUProcessor.process()).
                        Must have 'linear_norm', 'gyro_norm', 't'.
            mmwave_result: Processed mmWave frame dict (from MMWaveProcessor.process()).
                           Must have 'centroid', 'frame_number'.

        Returns:
            GestureWindow if a complete gesture was just detected, None otherwise.
        """
        now = time.time()

        # Compute combined activity metric
        metric = self._compute_metric(imu_sample, mmwave_result)
        self._metric_history.append((now, metric))

        # Update baseline noise during quiet periods (IDLE or well below threshold)
        if self._state == SegmenterState.IDLE:
            self._baseline_noise = (
                (1.0 - self._noise_ema_alpha) * self._baseline_noise
                + self._noise_ema_alpha * metric
            )

        # Compute thresholds
        thresh_on = self._baseline_noise * self._threshold_on_mult
        thresh_off = self._baseline_noise * self._threshold_off_mult

        # Ensure minimum threshold
        thresh_on = max(thresh_on, 0.01)
        thresh_off = max(thresh_off, 0.005)

        # ── State Machine ────────────────────────────────────────────

        if self._state == SegmenterState.IDLE:
            # Check cooldown
            if now - self._last_gesture_time < self._cooldown_ms / 1000.0:
                return None

            if metric > thresh_on:
                self._state = SegmenterState.ACTIVE
                self._onset_time = now
                self._onset_metric = metric
                self._peak_metric = metric
                self._quiet_start = None
                self._determine_source(imu_sample, mmwave_result)

        elif self._state == SegmenterState.ACTIVE:
            # Track peak
            if metric > self._peak_metric:
                self._peak_metric = metric

            # Check for deactivation
            if metric < thresh_off:
                if self._quiet_start is None:
                    self._quiet_start = now
                quiet_duration = (now - self._quiet_start) * 1000.0  # ms

                if quiet_duration >= self._quiet_confirm_ms:
                    # Gesture complete — validate and emit
                    return self._finalize_gesture(now)
            else:
                self._quiet_start = None

            # Check for max duration (force-split)
            if self._onset_time is not None:
                duration_ms = (now - self._onset_time) * 1000.0
                if duration_ms >= self._max_duration_ms:
                    return self._finalize_gesture(now)

        return None

    def _compute_metric(
        self,
        imu_sample: dict | None,
        mmwave_result: dict | None,
    ) -> float:
        """Compute combined activity metric from available sensors."""
        parts = []
        weights = []

        if imu_sample is not None:
            la_norm = imu_sample.get("linear_norm", 0.0)
            g_norm = imu_sample.get("gyro_norm", 0.0)
            parts.append(la_norm * self._w_accel)
            parts.append(g_norm * self._w_gyro)
            weights.extend([self._w_accel, self._w_gyro])

        if mmwave_result is not None:
            centroid = mmwave_result.get("centroid")
            if centroid is not None:
                if self._prev_mmwave_centroid is not None:
                    # Range change rate
                    range_delta = abs(
                        float(np.linalg.norm(centroid))
                        - float(np.linalg.norm(self._prev_mmwave_centroid))
                    )
                    # Scale to be comparable to IMU norms
                    range_rate = min(range_delta * 10.0, 2.0)
                    parts.append(range_rate * self._w_range)
                    weights.append(self._w_range)
                self._prev_mmwave_centroid = (
                    float(centroid[0]), float(centroid[1]), float(centroid[2])
                )
            # Also consider velocity centroid
            vel = mmwave_result.get("velocity_centroid")
            if vel is not None:
                parts.append(abs(vel) * 0.5 * self._w_range)
                weights.append(self._w_range * 0.5)

        if not parts:
            return 0.0

        # Weighted average
        total_weight = sum(weights)
        if total_weight < 1e-9:
            return 0.0

        return sum(parts) / total_weight

    def _determine_source(
        self,
        imu_sample: dict | None,
        mmwave_result: dict | None,
    ) -> None:
        """Set the primary source of this gesture detection."""
        if imu_sample is not None and mmwave_result is not None:
            self._gesture_source = "fused"
        elif imu_sample is not None:
            self._gesture_source = "imu"
        elif mmwave_result is not None:
            self._gesture_source = "mmwave"
        else:
            self._gesture_source = "unknown"

    def _finalize_gesture(self, now: float) -> GestureWindow | None:
        """Create gesture window and reset state."""
        if self._onset_time is None:
            self._state = SegmenterState.IDLE
            self._quiet_start = None
            return None

        offset_time = now
        duration_ms = (offset_time - self._onset_time) * 1000.0

        # Validate duration
        if duration_ms < self._min_duration_ms:
            self._state = SegmenterState.IDLE
            self._onset_time = None
            self._quiet_start = None
            return None

        window = GestureWindow(
            onset_time=self._onset_time,
            offset_time=offset_time,
            duration_sec=duration_ms / 1000.0,
            peak_metric=self._peak_metric,
            source=self._gesture_source,
            onset_timestamp=self._onset_time,
            offset_timestamp=offset_time,
        )

        # Reset state
        self._state = SegmenterState.IDLE
        self._last_gesture_time = now
        self._onset_time = None
        self._peak_metric = 0.0
        self._quiet_start = None

        return window

    def reset(self) -> None:
        """Reset segmenter state (e.g., after sensor reconnect)."""
        self._state = SegmenterState.IDLE
        self._baseline_noise = 0.01
        self._onset_time = None
        self._peak_metric = 0.0
        self._quiet_start = None
        self._last_gesture_time = 0.0
        self._prev_mmwave_centroid = None
        self._metric_history.clear()

    @property
    def state(self) -> SegmenterState:
        return self._state

    @property
    def baseline_noise(self) -> float:
        return self._baseline_noise

    @property
    def is_active(self) -> bool:
        return self._state == SegmenterState.ACTIVE


# Need numpy for centroid norm
import numpy as np
