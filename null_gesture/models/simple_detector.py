"""Physics-based IMU gesture detector — all detectable one-hand gestures.

Uses gyro for onset direction, accel for sustained posture changes.
No ML, no training — calibrated from real BMI270 data at 50Hz.

Detectable gestures:
  standing_still, push, pull, left, right,
  up, down, clockwise, anti_clockwise,
  bye_bye, palm_up, palm_down
"""

from __future__ import annotations

import time
from collections import Counter, deque
from typing import ClassVar

import numpy as np


class SimpleIMUDetector:
    """Multi-gesture IMU classifier. No model, no training."""

    GESTURES: ClassVar[list[str]] = [
        "standing_still", "push", "pull", "left", "right",
        "up", "down", "clockwise", "anti_clockwise",
        "bye_bye", "palm_up", "palm_down",
    ]

    def __init__(
        self,
        gyro_onset: float = 18.0,       # dps — gyro spike to trigger motion
        gyro_still: float = 8.0,         # dps — max gyro for stillness
        accel_still: float = 0.008,      # g² — max accel variance for stillness
        accel_vertical: float = 0.15,    # g — az deviation for up/down
        accel_roll: float = 0.25,        # g — az drop for palm_up detection
        history_s: float = 0.5,          # hold prediction after motion stops
        onset_window: float = 0.12,       # window for motion onset (~6 samples)
    ):
        self.gyro_onset = gyro_onset
        self.gyro_still = gyro_still
        self.accel_still = accel_still
        self.accel_vertical = accel_vertical
        self.accel_roll = accel_roll
        self.history_s = history_s
        self.onset_window = onset_window

        # Buffers
        self._gyro_buf: deque[tuple[float, float, float]] = deque(maxlen=30)
        self._accel_buf: deque[tuple[float, float, float]] = deque(maxlen=30)

        # State
        self._last_label: str = "standing_still"
        self._last_confidence: float = 0.0
        self._last_motion_time: float = 0.0
        self._sample_count: int = 0

        # Hysteresis — need N/10 votes to switch
        self._vote_buf: deque[str] = deque(maxlen=10)
        self._vote_needed: int = 6

        # For bye_bye detection
        self._gz_crossings: int = 0
        self._last_gz_sign: int = 0

    # ── Core update ────────────────────────────────────────────────────

    def update(self, imu_window: np.ndarray) -> tuple[str, float]:
        """Returns (gesture_label, confidence 0-1)."""
        if imu_window.ndim != 2 or imu_window.shape[0] < 5:
            return "standing_still", 1.0

        now = time.time()

        # Recent data from short window
        recent = imu_window[-6:]  # ~120ms
        accel = recent[:, :3]
        gyro = recent[:, 3:]

        mean_gyro = np.mean(gyro, axis=0)  # gx, gy, gz
        gyro_mag = float(np.linalg.norm(mean_gyro))

        mean_accel = np.mean(accel, axis=0)
        accel_var = float(np.mean(np.var(accel, axis=0)))

        # Store in buffers
        self._gyro_buf.append((float(mean_gyro[0]), float(mean_gyro[1]), float(mean_gyro[2])))
        self._accel_buf.append((float(mean_accel[0]), float(mean_accel[1]), float(mean_accel[2])))

        # Track zero-crossings for bye_bye
        gz_val = float(mean_gyro[2])
        current_gz_sign = 1 if gz_val > 5 else (-1 if gz_val < -5 else 0)
        if current_gz_sign != 0 and current_gz_sign != self._last_gz_sign:
            self._gz_crossings += 1
        self._last_gz_sign = current_gz_sign

        # ── Stillness ─────────────────────────────────────────────────
        is_still = (
            abs(mean_gyro[0]) < self.gyro_still
            and abs(mean_gyro[1]) < self.gyro_still
            and abs(mean_gyro[2]) < self.gyro_still
            and accel_var < self.accel_still
        )

        # ── Gesture classification ────────────────────────────────────
        raw_label = "standing_still"
        raw_conf = 0.0

        if gyro_mag > self.gyro_onset:
            # Motion detected by gyro
            self._last_motion_time = now
            gx, gy, gz = mean_gyro
            abs_g, abs_gy, abs_gz = abs(gx), abs(gy), abs(gz)
            dominant = np.argmax([abs_g, abs_gy, abs_gz])
            secondary = sorted([abs_g, abs_gy, abs_gz])[1]

            # Clockwise/anti-clockwise = multi-axis gyro
            is_circular = secondary > self.gyro_onset * 0.7

            # Wave/bye_bye = oscillating gz
            recent_gz = [h[2] for h in self._gyro_buf]
            if len(recent_gz) >= 8:
                crossings = sum(
                    1 for i in range(1, len(recent_gz))
                    if recent_gz[i] * recent_gz[i-1] < 0
                )
                if crossings >= 4:
                    raw_label = "bye_bye"
                    raw_conf = min(1.0, crossings / 8.0)
                    self._gz_crossings = 0

            if raw_label == "standing_still":
                if is_circular:
                    # Multi-axis = rotation
                    avg_gz = sum(h[2] for h in list(self._gyro_buf)[-6:]) / 6
                    raw_label = "clockwise" if avg_gz < -10 else "anti_clockwise"
                    raw_conf = min(1.0, gyro_mag / 120.0)

                elif dominant == 0:  # gx dominant — roll
                    raw_label = "left" if gx > 8 else "right"
                    raw_conf = min(1.0, abs_g / 50.0)

                elif dominant == 1:  # gy dominant — pitch
                    raw_label = "push" if gy > 8 else "pull"
                    raw_conf = min(1.0, abs_gy / 50.0)

                elif dominant == 2:  # gz dominant — yaw
                    raw_label = "clockwise" if gz < -10 else "anti_clockwise"
                    raw_conf = min(1.0, abs_gz / 80.0)

        elif self._check_vertical_motion(mean_accel):
            self._last_motion_time = now
            # Vertical motion — accel based
            float(mean_accel[2])
            recent_az = [h[2] for h in list(self._accel_buf)[-10:]]
            avg_az = sum(recent_az) / len(recent_az)

            if avg_az > 1.0 + self.accel_vertical * 0.5:
                raw_label = "up"
                raw_conf = min(1.0, (avg_az - 1.0) / 0.3)
            elif avg_az < 1.0 - self.accel_vertical:
                # Check if this is palm_up (sustained low az)
                if self._check_palm_rotation():
                    raw_label = "palm_up" if mean_gyro[0] > 5 else "palm_down"
                    raw_conf = min(1.0, abs(float(mean_gyro[0])) / 40.0)
                else:
                    raw_label = "down"
                    raw_conf = min(1.0, (1.0 - avg_az) / 0.3)

        elif is_still and (now - self._last_motion_time) > self.history_s:
            raw_label = "standing_still"
            raw_conf = min(1.0, 1.0 - gyro_mag / self.gyro_still)

        # ── Hysteresis voting ────────────────────────────────────────
        self._vote_buf.append(raw_label)
        if len(self._vote_buf) >= 10:
            counts = Counter(self._vote_buf)
            top_label, top_count = counts.most_common(1)[0]
            if top_count >= self._vote_needed:
                self._last_label = top_label
        else:
            self._last_label = raw_label

        self._last_confidence = raw_conf
        self._sample_count += 1
        return self._last_label, self._last_confidence

    # ── Helpers ───────────────────────────────────────────────────────

    def _check_vertical_motion(self, mean_accel: np.ndarray) -> bool:
        """Detect up/down motion via accel z deviation from 1g."""
        az = float(mean_accel[2])
        return abs(az - 1.0) > self.accel_vertical

    def _check_palm_rotation(self) -> bool:
        """Check if sustained low az is palm rotation vs just lowering hand."""
        if len(self._accel_buf) < 10:
            return False
        recent_az = [h[2] for h in list(self._accel_buf)[-10:]]
        recent_gx = [h[0] for h in list(self._gyro_buf)[-10:]]
        avg_az = sum(recent_az) / len(recent_az)
        avg_gx = sum(recent_gx) / len(recent_gx)
        # Palm rotation: az near 0 AND gx non-negligible
        return avg_az < 0.4 and abs(avg_gx) > 8
