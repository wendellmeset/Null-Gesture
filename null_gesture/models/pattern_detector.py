"""Pattern-matched gesture detector — calibrated from real IMU data.

Key discriminators found in analysis:
  - Dominant gyro axis + sign → direction of motion
  - Oscillation count → repetitive vs single motions  
  - Gyro magnitude → high-energy (bye_bye, clapping) vs low-energy (push, palm)
  - Accel variance → clapping and boxing have high accel jitter
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np

GESTURES = [
    "standing_still",
    "push", "pull",
    "left", "right",
    "clockwise", "anti_clockwise",
    "bye_bye",
    "clapping", "one_arm_boxing",
    "t_arms", "raise_arms",
    "palm_up", "palm_down",
]


def classify_segment(gyr: np.ndarray, acc: np.ndarray) -> tuple[str, float]:
    """Classify a gesture segment from (T,3) gyro and (T,3) accel arrays."""
    
    gx, gy, gz = gyr.mean(axis=0)
    abs_g = np.abs([gx, gy, gz])
    dom_axis = int(np.argmax(abs_g))
    dom_sign = int(np.sign([gx, gy, gz][dom_axis]))
    
    gyr_mag = np.linalg.norm(gyr, axis=1)
    gyr_mean = float(gyr_mag.mean())
    gyr_peak = float(gyr_mag.max())
    
    # Oscillation count
    gyr_c = gyr - gyr.mean(axis=0)
    osc = int(np.sum(np.abs(np.diff(np.signbit(gyr_c), axis=0))))
    
    # Accel variance (jitter)
    acc_var = float(np.var(acc).mean())
    
    # Duration in seconds (assuming ~50Hz)
    T = gyr.shape[0]
    dur = T / 50.0

    # ── Too weak = subtle gesture ─────────────────────────────────
    if gyr_peak < 30 and gyr_mean < 12:
        if abs(gy) > abs(gx) and abs(gy) > abs(gz):
            return "palm_up" if gy > 0 else "palm_down", 0.82
        return "palm_up", 0.70

    # ── Very low energy, tiny motions ─────────────────────────────
    if gyr_mean < 18 and gyr_peak < 80:
        if abs(gy) > abs(gx):
            return "palm_up" if gy > 0 else "palm_down", 0.80
        return "push" if gy > 0 else "pull", 0.65

    # ── Oscillatory high-energy ───────────────────────────────────
    if osc > 55 and dur > 0.5:
        if acc_var > 0.55:  # high accel jitter = clapping or boxing
            if gyr_peak > 400:
                return "clapping", 0.88
            return "one_arm_boxing", 0.82
        if abs(gz) > abs(gx) and abs(gz) > abs(gy):
            return "bye_bye", 0.85
        return "one_arm_boxing", 0.75

    # ── Moderate oscillation → bye_bye ────────────────────────────
    if osc > 30 and gyr_mean > 80 and abs(gz) > abs(gx):
        return "bye_bye", 0.85

    # ── Dominant axis classification ──────────────────────────────
    if dom_axis == 0:  # gx dominant
        if gyr_mean > 25:
            if gx > 20:
                return "left", 0.88
            if gx < -20:
                return "right", 0.88
        # Weak gx + moderate gy → could be raise_arms
        if gy > 20:
            return "raise_arms", 0.78
        if gy < -20:
            return "palm_down", 0.75
        return "left" if gx > 0 else "right", 0.65

    if dom_axis == 1:  # gy dominant
        if gy > 25 and gyr_mean > 30:
            return "raise_arms", 0.85
        if gy > 10:
            return "palm_up", 0.78
        if gy < -10:
            return "palm_down", 0.78
        return "push" if gy > 0 else "pull", 0.65

    if dom_axis == 2:  # gz dominant
        if gyr_mean > 60:
            if gz > 0:
                return "anti_clockwise", 0.85
            return "clockwise", 0.85
        # Low gz → could be t_arms or small rotation
        if abs(gx) > 15:
            return "t_arms", 0.72
        return "clockwise" if gz < 0 else "anti_clockwise", 0.70

    return "standing_still", 0.50


# ── Real-time detector ──────────────────────────────────────────────

class GestureDetector:
    def __init__(self, onset_thresh: float = 18.0, offset_thresh: float = 8.0,
                 offset_frames: int = 8, display_hold: float = 1.5):
        self.onset_thresh = onset_thresh
        self.offset_thresh = offset_thresh
        self.offset_frames = offset_frames
        self.display_hold = display_hold

        self._state = "still"
        self._raw = deque(maxlen=400)
        self._onset_idx = 0
        self._still_count = 0
        self._classify_time = 0.0
        self._result = ("standing_still", 1.0)
        self._sample_count = 0

    def update(self, imu_window: np.ndarray) -> tuple[str, float]:
        self._sample_count += 1
        now = time.time()

        if imu_window.ndim != 2 or imu_window.shape[0] < 5:
            return self._result

        row = imu_window[-1].astype(np.float32)
        self._raw.append(row)
        gyro_mag = float(np.linalg.norm(row[3:]))

        if self._state == "still":
            if gyro_mag > self.onset_thresh:
                self._state = "collecting"
                self._onset_idx = max(0, len(self._raw) - 6)
                self._still_count = 0

        elif self._state == "collecting":
            if gyro_mag < self.offset_thresh:
                self._still_count += 1
                if self._still_count >= self.offset_frames:
                    self._classify()
                    self._state = "classified"
                    self._classify_time = now
            else:
                self._still_count = 0
            if len(self._raw) - self._onset_idx >= 250:
                self._classify()
                self._state = "classified"
                self._classify_time = now

        elif self._state == "classified":
            if now - self._classify_time > self.display_hold:
                self._state = "still"
                self._result = ("standing_still", 1.0)

        return self._result

    def _classify(self) -> None:
        start = self._onset_idx
        end = len(self._raw)
        segment = np.array([self._raw[i] for i in range(start, end)], dtype=np.float32)
        if len(segment) < 6:
            self._result = ("standing_still", 1.0)
            return
        self._result = classify_segment(segment[:, 3:], segment[:, :3])

    @property
    def gestures(self) -> list[str]:
        return GESTURES
