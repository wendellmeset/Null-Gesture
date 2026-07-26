"""Physics-based IMU gesture detector — calibrated from real data.

Uses gyro as primary signal (clean, consistent across orientations).
Accel provides secondary confirmation.

Gestures: standing_still, push, pull, left, right, clockwise, anti_clockwise
"""

from __future__ import annotations

import time
from collections import deque
from typing import ClassVar

import numpy as np


class SimpleIMUDetector:
    """Multi-gesture IMU classifier calibrated on real hardware data."""

    GESTURES: ClassVar[list[str]] = [
        "standing_still", "push", "pull", "left", "right",
        "clockwise", "anti_clockwise",
    ]

    def __init__(
        self,
        gyro_threshold: float = 18.0,   # dps — gyro magnitude to count as motion
        still_gyro_max: float = 8.0,     # dps — below this on all axes = still
        still_accel_var: float = 0.008,  # g² — accel variance for still
        history_s: float = 0.6,          # hold prediction after motion
    ):
        self.gyro_threshold = gyro_threshold
        self.still_gyro_max = still_gyro_max
        self.still_accel_var = still_accel_var
        self.history_s = history_s

        self._last_label: str = "standing_still"
        self._last_confidence: float = 0.0
        self._last_motion_time: float = 0.0
        self._sample_count: int = 0

        # Hysteresis: need N consecutive frames of same label to switch
        self._vote_buf: deque[str] = deque(maxlen=10)
        self._vote_needed: int = 6

        # Pattern buffers
        self._gyro_dir_history: deque[tuple[float, float, float]] = deque(maxlen=15)

    def update(self, imu_window: np.ndarray) -> tuple[str, float]:
        """Returns (gesture_label, confidence 0-1)."""
        if imu_window.ndim != 2 or imu_window.shape[0] < 5:
            return "standing_still", 1.0

        now = time.time()

        # Recent window for motion detection (~200ms at 50Hz)
        recent = imu_window[-10:]
        accel = recent[:, :3]
        gyro = recent[:, 3:]

        mean_gyro = np.mean(gyro, axis=0)  # (gx, gy, gz)
        gyro_mag = float(np.linalg.norm(mean_gyro))

        np.mean(accel, axis=0)
        accel_var = float(np.mean(np.var(accel, axis=0)))

        # Store gyro direction
        self._gyro_dir_history.append((float(mean_gyro[0]), float(mean_gyro[1]), float(mean_gyro[2])))

        # ── Still detection ──────────────────────────────────────────
        all_axes_low = (
            abs(mean_gyro[0]) < self.still_gyro_max and
            abs(mean_gyro[1]) < self.still_gyro_max and
            abs(mean_gyro[2]) < self.still_gyro_max
        )
        is_still = all_axes_low and accel_var < self.still_accel_var

        # ── Motion classification ────────────────────────────────────
        raw_label = "standing_still"
        raw_conf = 0.0

        if gyro_mag > self.gyro_threshold:
            self._last_motion_time = now
            gx, gy, gz = mean_gyro

            # Determine dominant gyro axis
            abs_gx, abs_gy, abs_gz = abs(gx), abs(gy), abs(gz)
            dominant = np.argmax([abs_gx, abs_gy, abs_gz])

            # Check for circular motion (high gyro on multiple axes)
            secondary = sorted([abs_gx, abs_gy, abs_gz])[1]  # second largest
            is_circular = secondary > self.gyro_threshold * 0.8

            if is_circular:
                # Clockwise vs anti-clockwise: use gz sign
                # (when hand flat, clockwise = gz negative typically)
                if gz < -5:
                    raw_label = "clockwise"
                elif gz > 5:
                    raw_label = "anti_clockwise"
                else:
                    # Use the direction that gz was trending
                    recent_gz = [h[2] for h in self._gyro_dir_history]
                    avg_gz = sum(recent_gz) / len(recent_gz)
                    raw_label = "clockwise" if avg_gz < 0 else "anti_clockwise"
                raw_conf = min(1.0, gyro_mag / 120.0)

            elif dominant == 0:  # gx dominant — left/right roll
                if gx > 10:
                    raw_label = "left"
                elif gx < -10:
                    raw_label = "right"
                raw_conf = min(1.0, abs_gx / 50.0)

            elif dominant == 1:  # gy dominant — push/pull pitch
                if gy > 10:
                    raw_label = "push"
                elif gy < -10:
                    raw_label = "pull"
                raw_conf = min(1.0, abs_gy / 50.0)

            elif dominant == 2:  # gz dominant — yaw rotation
                if gz < -10:
                    raw_label = "clockwise"
                else:
                    raw_label = "anti_clockwise"
                raw_conf = min(1.0, abs_gz / 80.0)

        elif is_still and (now - self._last_motion_time) > self.history_s:
            raw_label = "standing_still"
            raw_conf = min(1.0, 1.0 - gyro_mag / self.still_gyro_max)

        # ── Hysteresis voting ────────────────────────────────────────
        self._vote_buf.append(raw_label)
        if len(self._vote_buf) >= 10:
            from collections import Counter
            counts = Counter(self._vote_buf)
            top_label, top_count = counts.most_common(1)[0]
            if top_count >= self._vote_needed:
                self._last_label = top_label
        else:
            self._last_label = raw_label

        self._last_confidence = raw_conf
        self._sample_count += 1
        return self._last_label, self._last_confidence
