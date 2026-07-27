"""Proximity Gesture Detector: UWB + IMU-based Pull/Push detection.

Uses UWB distance dynamics as primary signal with IMU acceleration as
confirmatory evidence.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from src.features.imu_features import IMUWindow
from src.features.uwb_features import UWBWindow


class ProximityGestureDetector:
    """Detects Pull and Push via UWB distance change + IMU confirmation."""

    GESTURES: ClassVar[list[str]] = ["pull", "push"]

    def __init__(
        self,
        window_samples: int = 50,
        pull_velocity_threshold: float = -0.3,  # m/s (negative = toward)
        push_velocity_threshold: float = 0.3,   # m/s (positive = away)
    ) -> None:
        self._uwb_window = UWBWindow(window_samples)
        self._imu_window = IMUWindow(window_samples)
        self._pull_thresh = pull_velocity_threshold
        self._push_thresh = push_velocity_threshold

        # Adaptive threshold state
        self._velocity_noise_std: float = 0.05  # baseline noise level
        self._calibrated: bool = False
        self._calibration_samples: list[float] = []

    def push(
        self,
        uwb_distance: float | None,
        uwb_velocity: float | None,
        imu_ax: float | None = None,
        imu_ay: float | None = None,
        imu_az: float | None = None,
    ) -> None:
        self._uwb_window.push(uwb_distance, uwb_velocity)
        if imu_ax is not None and imu_ay is not None and imu_az is not None:
            self._imu_window.push(imu_ax, imu_ay, imu_az, 0.0, 0.0, 0.0)

        # Adaptive calibration (first 100 samples)
        if not self._calibrated and uwb_velocity is not None:
            self._calibration_samples.append(uwb_velocity)
            if len(self._calibration_samples) >= 100:
                self._velocity_noise_std = float(np.std(self._calibration_samples))
                self._pull_thresh = -max(0.15, 2.5 * self._velocity_noise_std)
                self._push_thresh = max(0.15, 2.5 * self._velocity_noise_std)
                self._calibrated = True

    def detect(self) -> dict[str, float]:
        """Return belief masses for Pull/Push."""
        result: dict[str, float] = {"pull": 0.0, "push": 0.0, "unknown": 1.0}

        if not self._uwb_window.full:
            return result

        uwb_feats = self._uwb_window.features()
        if not uwb_feats:
            return result

        vel_mean = uwb_feats.get("uwb_velocity_mean", 0.0)
        vel_consistency = uwb_feats.get("uwb_velocity_sign_consistency", 0.0)
        dist_slope = uwb_feats.get("uwb_distance_slope", 0.0)

        # ── Pull detection ──────────────────────────────────────────
        pull_score = 0.0
        if vel_mean < self._pull_thresh and vel_consistency > 0.6:
            # Sustained motion toward body
            velocity_ratio = abs(vel_mean) / abs(self._pull_thresh)
            pull_score = min(1.0, velocity_ratio * vel_consistency)

            # Distance decreasing (corroboration)
            if dist_slope < 0:
                pull_score = min(1.0, pull_score * 1.3)

        # ── Push detection ──────────────────────────────────────────
        push_score = 0.0
        if vel_mean > self._push_thresh and vel_consistency > 0.6:
            velocity_ratio = vel_mean / self._push_thresh
            push_score = min(1.0, velocity_ratio * vel_consistency)

            if dist_slope > 0:
                push_score = min(1.0, push_score * 1.3)

        # IMU confirmation (accel direction)
        if self._imu_window.full:
            imu_feats = self._imu_window.compute_features()
            az_mean = imu_feats.get("az_mean", 0.0)
            # In sensor frame, negative Z (removing gravity) →
            # acceleration toward user typically maps to +az in linear accel
            # This is sensor-frame dependent; calibrate per setup
            if pull_score > 0 and az_mean < -0.5:
                pull_score = min(1.0, pull_score * 1.2)
            if push_score > 0 and az_mean > 0.5:
                push_score = min(1.0, push_score * 1.2)

        # ── Assemble ────────────────────────────────────────────────
        result["pull"] = min(0.9, pull_score)
        result["push"] = min(0.9, push_score)
        total = result["pull"] + result["push"]
        result["unknown"] = max(0.1, 1.0 - total)

        return result
