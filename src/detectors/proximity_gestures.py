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
        """Return belief masses for Pull/Push using UWB distance change."""
        result: dict[str, float] = {"pull": 0.0, "push": 0.0, "bye_bye": 0.0, "unknown": 1.0}

        if not self._uwb_window.full:
            return result

        uwb_feats = self._uwb_window.features()
        if not uwb_feats:
            return result

        dist_range = uwb_feats.get("uwb_distance_range", 0.0)
        dist_slope = uwb_feats.get("uwb_distance_slope", 0.0)
        dist_mean = uwb_feats.get("uwb_distance_mean", 0.0)
        dist_std = uwb_feats.get("uwb_distance_std", 0.0)

        # ── Bye-Bye: stable distance ──
        bye_score = 0.0
        if dist_mean > 0.05 and dist_std > 0.001:
            cv = dist_std / dist_mean
            if cv < 0.15 and dist_std < 0.15:
                bye_score = min(1.0, (1.0 - cv / 0.15) * 1.5)

        # ── Pull / Push: distance range > 3cm + consistent slope ──
        pull_score = 0.0
        push_score = 0.0
        if dist_range > 0.03:  # 3cm minimum change
            base = min(1.0, dist_range / 0.15)  # scale to 1.0 at 15cm
            if dist_slope < -0.0005:  # distance decreasing → pull
                pull_score = base
                push_score = base * 0.1
            elif dist_slope > 0.0005:  # distance increasing → push
                push_score = base
                pull_score = base * 0.1

        # IMU confirmation
        if self._imu_window.full:
            imu_feats = self._imu_window.compute_features()
            az_mean = imu_feats.get("az_mean", 0.0)
            if pull_score > 0 and az_mean < -0.5:
                pull_score = min(1.0, pull_score * 1.2)
            if push_score > 0 and az_mean > 0.5:
                push_score = min(1.0, push_score * 1.2)

        # ── Assemble ────────────────────────────────────────────────
        result["pull"] = min(0.9, pull_score)
        result["push"] = min(0.9, push_score)
        result["bye_bye"] = min(0.9, bye_score)
        total = result["pull"] + result["push"] + result["bye_bye"]
        result["unknown"] = max(0.1, 1.0 - total)

        return result
