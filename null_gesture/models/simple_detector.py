"""Physics-based IMU gesture detector — 7 core hand gestures."""

from __future__ import annotations

import time
from collections import Counter, deque
from typing import ClassVar

import numpy as np


class SimpleIMUDetector:
    """Multi-gesture IMU classifier calibrated on real BMI270 data."""

    GESTURES: ClassVar[list[str]] = [
        "standing_still", "push", "pull", "left", "right",
        "clockwise", "anti_clockwise",
    ]

    def __init__(
        self,
        gyro_onset: float = 12.0,
        gyro_still: float = 8.0,
        accel_still: float = 0.008,
        history_s: float = 0.5,
    ):
        self.gyro_onset = gyro_onset
        self.gyro_still = gyro_still
        self.accel_still = accel_still
        self.history_s = history_s

        self._gyro_buf: deque[tuple[float, float, float]] = deque(maxlen=30)
        self._onset_dir: tuple[float, float, float] | None = None
        self._last_label: str = "standing_still"
        self._last_confidence: float = 0.0
        self._last_motion_time: float = 0.0
        self._sample_count: int = 0
        self._vote_buf: deque[str] = deque(maxlen=10)
        self._vote_needed: int = 6

    def update(self, imu_window: np.ndarray) -> tuple[str, float]:
        if imu_window.ndim != 2 or imu_window.shape[0] < 5:
            return "standing_still", 1.0

        now = time.time()
        recent = imu_window[-6:]
        accel = recent[:, :3]
        gyro = recent[:, 3:]

        mean_gyro = np.mean(gyro, axis=0)
        gyro_mag = float(np.linalg.norm(mean_gyro))
        mean_accel = np.mean(accel, axis=0)
        accel_var = float(np.mean(np.var(accel, axis=0)))

        self._gyro_buf.append((float(mean_gyro[0]), float(mean_gyro[1]), float(mean_gyro[2])))

        # Stillness
        is_still = (
            abs(mean_gyro[0]) < self.gyro_still
            and abs(mean_gyro[1]) < self.gyro_still
            and abs(mean_gyro[2]) < self.gyro_still
            and accel_var < self.accel_still
        )

        # Capture onset when transitioning from still to moving
        if gyro_mag > self.gyro_onset and self._last_label == "standing_still":
            self._onset_dir = (float(mean_gyro[0]), float(mean_gyro[1]), float(mean_gyro[2]))

        # Classification
        raw_label = "standing_still"

        if gyro_mag > self.gyro_onset:
            self._last_motion_time = now

            # Use onset direction if available, otherwise current
            if self._onset_dir is not None:
                ogx, ogy, ogz = self._onset_dir
                self._onset_dir = None  # consume it
            else:
                ogx, ogy, ogz = float(mean_gyro[0]), float(mean_gyro[1]), float(mean_gyro[2])

            abs_ogx, abs_ogy, abs_ogz = abs(ogx), abs(ogy), abs(ogz)

            if abs_ogx >= abs_ogy and abs_ogx >= abs_ogz:
                raw_label = "left" if ogx > 0 else "right"
            elif abs_ogy >= abs_ogx and abs_ogy >= abs_ogz:
                raw_label = "push" if ogy > 0 else "pull"
            elif abs_ogz >= abs_ogx and abs_ogz >= abs_ogy:
                raw_label = "clockwise" if ogz < 0 else "anti_clockwise"

        elif is_still and (now - self._last_motion_time) > self.history_s:
            raw_label = "standing_still"

        # Hysteresis
        self._vote_buf.append(raw_label)
        if len(self._vote_buf) >= 10:
            counts = Counter(self._vote_buf)
            top_label, top_count = counts.most_common(1)[0]
            if top_count >= self._vote_needed:
                self._last_label = top_label
        else:
            self._last_label = raw_label

        self._sample_count += 1
        return self._last_label, self._last_confidence
