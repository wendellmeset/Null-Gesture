"""UWB + IMU fusion — acceleration direction + UWB distance constraint.

Instead of assuming IMU forward axis points to anchor (fragile), this uses:
  - IMU world-frame acceleration → direction of hand movement
  - UWB distance → absolute radial constraint from anchor
  - ZUPT → zero velocity when still
  - Continuous sphere projection → position converges to UWB distance

The two sensors correct each other: IMU provides direction (which UWB alone
can't), UWB provides absolute scale (which IMU double-integration can't).
"""

from __future__ import annotations

import logging
import time

import numpy as np

logger = logging.getLogger("null_gesture.sensors.fusion")


class UWBPositionTracker:
    """Fuses world-frame acceleration + UWB ranging for absolute position.

    Requires:
      - UWB anchor at fixed known position (default: origin)
      - UWB tag on the same hand as the IMU
      - Calibration: place hand at anchor, press calibrate (sets zero reference)
    """

    def __init__(
        self,
        anchor_pos: np.ndarray | None = None,
        still_thresh: float = 10.0,       # dps — below this = still
        vel_damp: float = 0.94,           # velocity damping (reduces overshoot)
        sphere_gain: float = 0.25,        # how aggressively to pull toward UWB sphere
        accel_gain: float = 0.15,         # how much acceleration drives position
        max_vel: float = 3.0,             # m/s — velocity clamp
    ):
        self.anchor_pos = (anchor_pos if anchor_pos is not None
                           else np.zeros(3, dtype=np.float64))

        self.still_thresh = still_thresh
        self.vel_damp = vel_damp
        self.sphere_gain = sphere_gain
        self.accel_gain = accel_gain
        self.max_vel = max_vel

        # State
        self.pos = np.zeros(3, dtype=np.float64)
        self.vel = np.zeros(3, dtype=np.float64)
        self.distance: float = 0.0       # latest UWB distance in meters
        self.last_update: float = 0.0
        self.is_still: bool = True

        # Calibration offset
        self._calib_distance: float = 0.0  # UWB distance when hand is at origin

    # ── Public API ─────────────────────────────────────────────────

    def calibrate(self, current_distance_m: float = 0.0) -> None:
        """Record current UWB distance as the 'zero position' reference.

        Call this while holding the hand at the desired origin point
        (e.g., resting on the desk where the anchor is).
        """
        self._calib_distance = current_distance_m
        self.pos.fill(0)
        self.vel.fill(0)
        logger.info("UWB+IMU calibrated — origin set at %.2fm", current_distance_m)

    def update(
        self,
        acc_world: np.ndarray,    # world-frame accel, gravity subtracted (m/s²)
        gyro_mag: float,          # gyro magnitude (dps)
        uwb_distance_cm: float,   # UWB distance in cm (0 if no measurement)
        dt: float,                # time step (seconds)
    ) -> np.ndarray:
        """One fusion step. Returns estimated (3,) position in meters."""
        self.is_still = gyro_mag < self.still_thresh

        # Update UWB distance (with calibration offset)
        if uwb_distance_cm > 0:
            raw_m = uwb_distance_cm / 100.0
            self.distance = max(0.0, raw_m - self._calib_distance)

        # ── Acceleration → velocity ───────────────────────────────
        acc_mag = np.linalg.norm(acc_world)

        if self.is_still:
            # ZUPT: kill velocity when still
            self.vel *= 0.0
        elif acc_mag > 0.05:
            # Integrate acceleration (with dead zone)
            acc_clean = acc_world.copy()
            if acc_mag < 0.15:
                acc_clean *= (acc_mag - 0.05) / 0.10  # smooth transition from dead zone
            self.vel += acc_clean * dt * self.accel_gain

        # Clamp velocity
        speed = np.linalg.norm(self.vel)
        if speed > self.max_vel:
            self.vel *= self.max_vel / speed

        # Damp velocity (prevents oscillation)
        self.vel *= self.vel_damp

        # ── Velocity → position ───────────────────────────────────
        self.pos += self.vel * dt

        # ── UWB sphere constraint ──────────────────────────────────
        # Pull position toward the sphere of radius `distance` around anchor
        if self.distance > 0.01:
            anchor_to_hand = self.pos - self.anchor_pos
            current_dist = np.linalg.norm(anchor_to_hand)

            if current_dist > 0.001:
                # Error: how far we are from the UWB-measured sphere
                radial_error = self.distance - current_dist
                # Direction from anchor to hand
                radial_dir = anchor_to_hand / current_dist
                # Nudge position along radial direction
                self.pos += radial_dir * radial_error * self.sphere_gain

                # Also nudge velocity to be tangential (hand moves on sphere surface)
                # Remove the radial component of velocity
                v_radial = np.dot(self.vel, radial_dir) * radial_dir
                self.vel -= v_radial * 0.3  # damp radial velocity

        # Gentle centering toward origin when still
        if self.is_still:
            self.pos *= 0.97

        self.last_update = time.time()
        return self.pos

    def reset(self) -> None:
        self.pos.fill(0)
        self.vel.fill(0)
        self.is_still = True
