"""Hand Gesture Detector: mmWave + IMU + RFID for fine hand articulations.

Covers: Opening and Closing Fist, Palm Up and Down.

Fist Open/Close: mmWave point count modulation in the hand cluster.
Palm Up/Down: IMU gyro integration around the wrist axis.
"""

from __future__ import annotations

import math
from collections import deque
from typing import ClassVar

import numpy as np

from src.features.imu_features import IMUWindow
from src.features.mmwave_features import MMWaveWindow


class HandGestureDetector:
    """Detects hand articulation gestures: Fist Open/Close, Palm Up/Down."""

    GESTURES: ClassVar[list[str]] = ["fist_open_close", "palm_up_down"]

    def __init__(
        self,
        mmwave_window_frames: int = 10,
        imu_window_samples: int = 50,
    ) -> None:
        self._mmwave_window = MMWaveWindow(mmwave_window_frames)
        self._imu_window = IMUWindow(imu_window_samples)

        # ── Fist Open/Close state ───────────────────────────────────
        self._point_count_history: deque[float] = deque(maxlen=30)
        self._fist_state: str = "unknown"  # "open", "closed", "transitioning"

        # Thresholds
        self._point_count_low = 0.65   # ratio of mean: below = closed fist
        self._point_count_high = 1.35  # ratio of mean: above = open hand
        self._palm_gyro_threshold = math.radians(60)  # 60° integrated gyro

    def push(
        self,
        clusters: list,
        velocities: np.ndarray | None,
        imu_ax: float | None = None,
        imu_ay: float | None = None,
        imu_az: float | None = None,
        imu_gx: float | None = None,
        imu_gy: float | None = None,
        imu_gz: float | None = None,
    ) -> None:
        self._mmwave_window.push(clusters, velocities)

        # Track point count in largest cluster
        if clusters:
            largest = max(clusters, key=lambda c: c.point_count)
            self._point_count_history.append(float(largest.point_count))

        if imu_ax is not None:
            self._imu_window.push(
                imu_ax, imu_ay or 0.0, imu_az or 0.0,
                imu_gx or 0.0, imu_gy or 0.0, imu_gz or 0.0,
            )

    def detect(self) -> dict[str, float]:
        """Return belief masses for hand gestures."""
        result: dict[str, float] = {
            "fist_open_close": 0.0,
            "palm_up_down": 0.0,
            "unknown": 1.0,
        }

        # ── Fist Open/Close (mmWave point count modulation) ─────────
        fist_score = self._detect_fist_modulation()

        # ── Palm Up/Down (IMU gyro integration) ────────────────────
        palm_score = self._detect_palm_rotation()

        result["fist_open_close"] = min(0.9, fist_score)
        result["palm_up_down"] = min(0.9, palm_score)
        total = result["fist_open_close"] + result["palm_up_down"]
        result["unknown"] = max(0.1, 1.0 - total)

        return result

    def _detect_fist_modulation(self) -> float:
        """Detect oscillation in point count consistent with fist open/close."""
        if len(self._point_count_history) < 10:
            return 0.0

        counts = np.array(list(self._point_count_history), dtype=np.float64)
        mean_count = np.mean(counts)

        if mean_count < 2.0:  # too few points for reliable detection
            return 0.0

        # Bandpass filter: look for oscillations at 0.5-3 Hz
        # Use FFT to check for periodic modulation
        n = len(counts)
        if n >= 8:
            # Detrend
            detrended = counts - np.mean(counts)
            fft = np.abs(np.fft.rfft(detrended))
            freqs = np.fft.rfftfreq(n, d=0.02)  # ~50 Hz mmWave → 20ms between frames

            if len(fft) > 1:
                # Look for dominant frequency in 0.5-3 Hz range
                valid = (freqs >= 0.5) & (freqs <= 3.0)
                if np.any(valid):
                    energy_in_band = np.sum(fft[valid])
                    total_energy = np.sum(fft[1:])  # exclude DC
                    if total_energy > 0:
                        band_ratio = energy_in_band / total_energy
                        # Also check modulation depth
                        mod_depth = (np.max(counts) - np.min(counts)) / (mean_count + 1e-10)
                        if band_ratio > 0.3 and mod_depth > 0.2:
                            return min(1.0, band_ratio * 1.5 * mod_depth * 2.0)

        return 0.0

    def _detect_palm_rotation(self) -> float:
        """Detect palm rotation via IMU gyro integration around x-axis (wrist)."""
        if not self._imu_window.full:
            return 0.0

        imu_feats = self._imu_window.compute_features()
        if not imu_feats:
            return 0.0

        gx = np.array(list(self._imu_window._gx))
        if len(gx) < 5:
            return 0.0

        # Integrate gyro_x over the window (degrees)
        dt = 0.01  # 100 Hz
        integrated_angle = float(np.sum(gx) * dt)

        # Check if rotation magnitude exceeds threshold
        abs_angle = abs(integrated_angle)
        if abs_angle > self._palm_gyro_threshold:
            # Map to confidence
            return min(1.0, (abs_angle - self._palm_gyro_threshold) / math.radians(90))

        return 0.0
