"""Sensor preprocessing: calibration, filtering, and structured data extraction.

Performs:
- IMU: gravity removal via Madgwick AHRS, coordinate alignment
- mmWave: background subtraction, DBSCAN clustering, centroid extraction
- UWB: median filtering, velocity estimation
- RFID: tag identity mapping, presence tracking
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from src.sensor.multiplexer import SensorFrame

# ── Constants ────────────────────────────────────────────────────────────

GRAVITY = 9.80665  # m/s²

# RFID EPC → hand identity mapping (configured per setup)
_DEFAULT_TAG_MAP: dict[str, str] = {}


# ── Madgwick AHRS (orientation filter) ───────────────────────────────────


class MadgwickAHRS:
    """Madgwick orientation filter for 6-DOF IMU (accel + gyro).

    Estimates quaternion orientation relative to Earth frame.
    Beta = 0.1 is a good default for typical human motion.
    Sampling rate should match IMU (100 Hz).
    """

    def __init__(self, beta: float = 0.1, sample_freq: float = 100.0) -> None:
        self.beta = beta
        self.sample_freq = sample_freq
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # w, x, y, z

    def update(
        self, gx: float, gy: float, gz: float, ax: float, ay: float, az: float
    ) -> np.ndarray:
        """Update filter with gyroscope (rad/s) and accelerometer (g) readings.

        Returns updated quaternion [w, x, y, z].
        """
        dt = 1.0 / self.sample_freq
        q = self.q

        # Normalize accelerometer
        norm = math.sqrt(ax * ax + ay * ay + az * az)
        if norm == 0:
            return q
        ax_n, ay_n, az_n = ax / norm, ay / norm, az / norm

        # Gradient descent algorithm corrective step
        # Reference direction of Earth's magnetic field (approximation)
        # Using gravity vector as reference
        q1, q2, q3, q4 = q[0], q[1], q[2], q[3]

        # Estimated direction of gravity
        vx = 2.0 * (q2 * q4 - q1 * q3)
        vy = 2.0 * (q1 * q2 + q3 * q4)
        vz = q1 * q1 - q2 * q2 - q3 * q3 + q4 * q4

        # Error is cross product between estimated and measured direction of gravity
        ex = vy * az_n - vz * ay_n
        ey = vz * ax_n - vx * az_n
        ez = vx * ay_n - vy * ax_n

        # Apply feedback
        gx_c = gx + self.beta * ex
        gy_c = gy + self.beta * ey
        gz_c = gz + self.beta * ez

        # Integrate rate of change of quaternion
        q_dot = np.array(
            [
                -0.5 * q2 * gx_c - 0.5 * q3 * gy_c - 0.5 * q4 * gz_c,
                0.5 * q1 * gx_c + 0.5 * q3 * gz_c - 0.5 * q4 * gy_c,
                0.5 * q1 * gy_c - 0.5 * q2 * gz_c + 0.5 * q4 * gx_c,
                0.5 * q1 * gz_c + 0.5 * q2 * gy_c - 0.5 * q3 * gx_c,
            ],
            dtype=np.float64,
        )

        q_new = q + q_dot * dt
        norm = np.linalg.norm(q_new)
        if norm > 0:
            q_new /= norm

        self.q = q_new
        return q_new

    def gravity_vector(self) -> np.ndarray:
        """Return estimated gravity vector in sensor frame [x, y, z]."""
        q1, q2, q3, q4 = self.q
        return np.array(
            [
                2.0 * (q2 * q4 - q1 * q3),
                2.0 * (q1 * q2 + q3 * q4),
                q1 * q1 - q2 * q2 - q3 * q3 + q4 * q4,
            ]
        )

    def remove_gravity(self, ax: float, ay: float, az: float) -> tuple[float, float, float]:
        """Remove gravity component from accelerometer reading.

        Returns linear acceleration in sensor frame.
        """
        g_vec = self.gravity_vector()
        return ax - g_vec[0], ay - g_vec[1], az - g_vec[2]


# ── DBSCAN clustering ────────────────────────────────────────────────────


def _dbscan(
    points: np.ndarray,
    eps: float = 0.05,
    min_samples: int = 3,
) -> tuple[np.ndarray, int]:
    """Minimal DBSCAN clustering on 3D point cloud.

    Returns (labels, n_clusters). Label -1 = noise.
    """
    n = len(points)
    if n == 0:
        return np.array([], dtype=np.int32), 0

    labels = np.full(n, -1, dtype=np.int32)
    cluster_id = 0

    for i in range(n):
        if labels[i] != -1:
            continue

        # Find neighbors
        dists = np.linalg.norm(points - points[i], axis=1)
        neighbors = np.where(dists <= eps)[0]

        if len(neighbors) < min_samples:
            labels[i] = -1  # noise
            continue

        # Expand cluster
        labels[i] = cluster_id
        seed = list(neighbors)
        seed.remove(i)

        while seed:
            j = seed.pop()
            if labels[j] == -1:
                labels[j] = cluster_id
                dists_j = np.linalg.norm(points - points[j], axis=1)
                nbrs_j = np.where(dists_j <= eps)[0]
                if len(nbrs_j) >= min_samples:
                    seed.extend(nbrs_j)
            elif labels[j] != -1:
                continue

        cluster_id += 1

    return labels, cluster_id


# ── Cluster features dataclass ───────────────────────────────────────────


@dataclass
class ClusterInfo:
    centroid: np.ndarray  # (3,) in meters
    spread: np.ndarray  # (3,) standard deviation
    point_count: int
    points: np.ndarray  # (N,3) subset


# ── Preprocessed frame ───────────────────────────────────────────────────


@dataclass
class PreprocessedFrame:
    """Structured, calibrated data from all sensors, ready for feature extraction."""

    timestamp: float = field(default_factory=time.time)

    # IMU
    imu_linear_accel: tuple[float, float, float] | None = None  # gravity-removed, m/s²
    imu_gyro: tuple[float, float, float] | None = None  # rad/s
    imu_orientation: np.ndarray | None = None  # quaternion [w, x, y, z]

    # mmWave
    mmwave_clusters: list[ClusterInfo] | None = None  # clustered point cloud
    mmwave_raw_points: np.ndarray | None = None  # all points after BG removal
    mmwave_raw_velocities: np.ndarray | None = None  # all velocities

    # UWB
    uwb_distance: float | None = None  # filtered distance in meters
    uwb_velocity: float | None = None  # estimated velocity in m/s

    # RFID
    rfid_tags: dict[str, float] | None = None  # epc -> rssi
    rfid_hands: set[str] | None = None  # {"left", "right"} — which hands detected


class SensorPreprocessor:
    """Calibrates raw sensor frames into structured PreprocessedFrame.

    Handles:
    - IMU calibration (gravity estimation, orientation tracking)
    - mmWave background subtraction and clustering
    - UWB filtering and velocity
    - RFID tag identity mapping
    """

    def __init__(
        self,
        tag_map: dict[str, str] | None = None,
        calibration_frames: int = 50,
        uwb_filter_window: int = 5,
        dbscan_eps: float = 0.05,
        dbscan_min_samples: int = 3,
    ) -> None:
        self.tag_map = tag_map or _DEFAULT_TAG_MAP
        self.calibration_frames = calibration_frames
        self._uwb_window = deque(maxlen=uwb_filter_window)
        self._dbscan_eps = dbscan_eps
        self._dbscan_min_samples = dbscan_min_samples

        # Calibration state
        self._calibrated = False
        self._ahrs = MadgwickAHRS(beta=0.1, sample_freq=100.0)
        self._bg_points: np.ndarray | None = None
        self._bg_count = 0
        self._uwb_baseline: float | None = None
        self._uwb_samples: list[float] = []
        self._uwb_prev_dist: float | None = None
        self._uwb_prev_time: float | None = None

    # ── Public API ───────────────────────────────────────────────────────

    def process(self, frame: SensorFrame) -> PreprocessedFrame:
        """Process a raw SensorFrame into a PreprocessedFrame."""
        pf = PreprocessedFrame(timestamp=frame.timestamp)

        # IMU processing
        if frame.imu is not None:
            pf.imu_gyro = self._process_imu_gyro(frame.imu)
            pf.imu_linear_accel = self._process_imu_accel(frame.imu)
            pf.imu_orientation = self._ahrs.q.copy()

        # mmWave processing
        if frame.mmwave_points is not None and len(frame.mmwave_points) > 0:
            pf.mmwave_raw_points = frame.mmwave_points.copy()
            pf.mmwave_raw_velocities = (
                frame.mmwave_velocities.copy()
                if frame.mmwave_velocities is not None
                else None
            )
            pf.mmwave_clusters = self._process_mmwave(frame.mmwave_points)

        # UWB processing
        if frame.uwb is not None:
            pf.uwb_distance, pf.uwb_velocity = self._process_uwb(frame.uwb)

        # RFID processing
        if frame.rfid is not None:
            pf.rfid_tags, pf.rfid_hands = self._process_rfid(frame.rfid)

        # Calibration tracking
        if not self._calibrated:
            self._bg_count += 1
            if self._bg_count >= self.calibration_frames:
                self._finalize_calibration()

        return pf

    def is_calibrated(self) -> bool:
        return self._calibrated

    # ── IMU processing ───────────────────────────────────────────────────

    def _process_imu_gyro(self, imu: dict) -> tuple[float, float, float]:
        # IMU outputs gyro in degrees/sec; pipeline expects rad/sec
        D2R = 0.017453292519943295  # math.pi / 180
        gx = imu.get("gx", 0.0) * D2R
        gy = imu.get("gy", 0.0) * D2R
        gz = imu.get("gz", 0.0) * D2R
        return (gx, gy, gz)

    def _process_imu_accel(self, imu: dict) -> tuple[float, float, float]:
        ax = imu.get("ax", 0.0)
        ay = imu.get("ay", 0.0)
        az = imu.get("az", 0.0)

        # Update AHRS orientation (gyro must be rad/s)
        D2R = 0.017453292519943295
        gx = imu.get("gx", 0.0) * D2R
        gy = imu.get("gy", 0.0) * D2R
        gz = imu.get("gz", 0.0) * D2R
        self._ahrs.update(gx, gy, gz, ax, ay, az)

        # Remove gravity
        return self._ahrs.remove_gravity(ax, ay, az)

    # ── mmWave processing ────────────────────────────────────────────────

    def _process_mmwave(self, points: np.ndarray) -> list[ClusterInfo]:
        # Background subtraction (accumulate during calibration)
        if self._calibrated and self._bg_points is not None and len(self._bg_points) > 0:
            # Remove points near static background
            mask = np.ones(len(points), dtype=bool)
            for bg_pt in self._bg_points[:500]:  # limit for performance
                dists = np.linalg.norm(points - bg_pt, axis=1)
                mask &= dists > 0.03  # 3 cm threshold
            filtered = points[mask]
        else:
            filtered = points

        if len(filtered) < self._dbscan_min_samples:
            return []

        # Cluster
        labels, n_clusters = _dbscan(filtered, self._dbscan_eps, self._dbscan_min_samples)

        clusters: list[ClusterInfo] = []
        for cid in range(n_clusters):
            c_mask = labels == cid
            c_pts = filtered[c_mask]
            if len(c_pts) < self._dbscan_min_samples:
                continue
            centroid = np.mean(c_pts, axis=0)
            spread = np.std(c_pts, axis=0)
            clusters.append(
                ClusterInfo(
                    centroid=centroid,
                    spread=spread,
                    point_count=len(c_pts),
                    points=c_pts,
                )
            )

        return clusters

    # ── UWB processing ───────────────────────────────────────────────────

    def _process_uwb(self, uwb: dict) -> tuple[float | None, float | None]:
        dist = uwb.get("distance_m")
        # Reject startup spikes and invalid readings (> 100m is noise)
        if dist is None or dist > 100.0:
            return None, None

        # Accumulate for calibration
        if not self._calibrated and self._bg_count <= self.calibration_frames:
            self._uwb_samples.append(dist)

        # Median filter
        self._uwb_window.append(dist)
        filtered = float(np.median(list(self._uwb_window)))

        # Velocity estimation
        now = time.time()
        velocity = None
        if self._uwb_prev_dist is not None and self._uwb_prev_time is not None:
            dt = now - self._uwb_prev_time
            if dt > 0.001:
                velocity = (filtered - self._uwb_prev_dist) / dt

        self._uwb_prev_dist = filtered
        self._uwb_prev_time = now

        return filtered, velocity

    # ── RFID processing ──────────────────────────────────────────────────

    def _process_rfid(self, rfid: dict) -> tuple[dict[str, float], set[str]]:
        epc = rfid.get("epc", "UNKNOWN")
        rssi = float(rfid.get("rssi", -999.0))

        tags = {epc: rssi}
        hands: set[str] = set()

        for epc_val in tags:
            hand = self.tag_map.get(epc_val)
            if hand:
                hands.add(hand)

        return tags, hands

    # ── Calibration ──────────────────────────────────────────────────────

    def _finalize_calibration(self) -> None:
        """Finalize calibration after collecting baseline frames."""
        self._calibrated = True

        # UWB baseline: median of initial samples
        if self._uwb_samples:
            self._uwb_baseline = float(np.median(self._uwb_samples))

    def set_bg_points(self, points: np.ndarray) -> None:
        """Externally set the static background point cloud."""
        self._bg_points = points
