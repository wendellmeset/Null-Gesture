"""IMU signal processor: orientation estimation and signal enhancement.

Applies Madgwick AHRS filter for orientation tracking, removes gravity
from accelerometer readings, and computes derived signals (linear
acceleration, jerk, magnitude norms).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


class MadgwickAHRS:
    """Madgwick orientation filter for 6-DOF IMU (accel + gyro, no mag).

    Estimates orientation quaternion from gyroscope integration with
    accelerometer correction via gradient descent.

    Implements the algorithm from:
      Madgwick, S. "An efficient orientation filter for inertial and
      inertial/magnetic sensor arrays." 2010.

    Args:
        beta: Gradient descent gain. Higher = more accel trust, more noise.
              Default 0.041 is tuned for human hand motion at 100 Hz.
        sample_freq: Sensor sample rate in Hz.
    """

    def __init__(self, beta: float = 0.041, sample_freq: float = 100.0):
        self.beta = beta
        self.sample_freq = sample_freq
        self.dt = 1.0 / sample_freq
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def update(self, gx: float, gy: float, gz: float,
               ax: float, ay: float, az: float) -> np.ndarray:
        """Update filter with a new IMU sample.

        Args:
            gx, gy, gz: Gyroscope readings in degrees per second.
            ax, ay, az: Accelerometer readings in g.

        Returns:
            Quaternion [w, x, y, z] as numpy array.
        """
        # Convert gyro from deg/s to rad/s
        gx_rad = np.radians(gx)
        gy_rad = np.radians(gy)
        gz_rad = np.radians(gz)

        q = self.quaternion.copy()

        # Normalize accelerometer
        a_norm = np.linalg.norm([ax, ay, az])
        if a_norm < 1e-9:
            return q
        ax_n, ay_n, az_n = ax / a_norm, ay / a_norm, az / a_norm

        # Gradient descent corrective step
        # Objective function: f(q, a) and Jacobian J
        f_1 = 2.0 * (q[1] * q[3] - q[0] * q[2]) - ax_n
        f_2 = 2.0 * (q[0] * q[1] + q[2] * q[3]) - ay_n
        f_3 = 2.0 * (0.5 - q[1] * q[1] - q[2] * q[2]) - az_n

        J_11 = -2.0 * q[2]
        J_12 = 2.0 * q[3]
        J_13 = -2.0 * q[0]
        J_14 = 2.0 * q[1]
        J_21 = 2.0 * q[1]
        J_22 = 2.0 * q[0]
        J_23 = 2.0 * q[3]
        J_24 = 2.0 * q[2]
        J_31 = 0.0
        J_32 = -4.0 * q[1]
        J_33 = -4.0 * q[2]
        J_34 = 0.0

        # Gradient: ∇f = J^T * f
        grad = np.array([
            J_11 * f_1 + J_21 * f_2 + J_31 * f_3,
            J_12 * f_1 + J_22 * f_2 + J_32 * f_3,
            J_13 * f_1 + J_23 * f_2 + J_33 * f_3,
            J_14 * f_1 + J_24 * f_2 + J_34 * f_3,
        ])

        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-9:
            grad /= grad_norm

        # Gyroscope quaternion derivative
        q_dot = 0.5 * np.array([
            -q[1] * gx_rad - q[2] * gy_rad - q[3] * gz_rad,
             q[0] * gx_rad + q[2] * gz_rad - q[3] * gy_rad,
             q[0] * gy_rad - q[1] * gz_rad + q[3] * gx_rad,
             q[0] * gz_rad + q[1] * gy_rad - q[2] * gx_rad,
        ])

        # Fuse: gyro integration minus gradient correction
        q += (q_dot - self.beta * grad) * self.dt

        # Normalize
        q /= np.linalg.norm(q)
        self.quaternion = q
        return q.copy()

    def reset(self) -> None:
        """Reset orientation to identity."""
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    @property
    def euler(self) -> tuple[float, float, float]:
        """Return Euler angles (roll, pitch, yaw) in degrees."""
        r = Rotation.from_quat([
            self.quaternion[1], self.quaternion[2],
            self.quaternion[3], self.quaternion[0],
        ])
        roll, pitch, yaw = r.as_euler('xyz', degrees=True)
        return roll, pitch, yaw


class IMUProcessor:
    """Processes raw IMU samples into enhanced signals.

    Pipeline per sample:
    1. Calibration offset removal
    2. Madgwick AHRS → orientation quaternion + euler angles
    3. Gravity removal → linear acceleration
    4. Jerk computation
    5. Magnitude norms

    Usage::

        proc = IMUProcessor(sample_rate=100.0)
        for sample in imu_reader.stream():
            processed = proc.process(sample)
            # processed has: ax, ay, az, gx, gy, gz, qw, qx, qy, qz,
            #                lax, lay, laz, jx, jy, jz, |a|, |g|, |la|, |j|,
            #                roll, pitch, yaw, t
    """

    def __init__(
        self,
        sample_rate: float = 100.0,
        madgwick_beta: float = 0.041,
        accel_bias: tuple[float, float, float] = (0.0, 0.0, 0.0),
        gyro_bias: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ):
        """
        Args:
            sample_rate: IMU sample rate in Hz.
            madgwick_beta: AHRS gradient descent gain.
            accel_bias: (bx, by, bz) accelerometer bias in g.
            gyro_bias: (bx, by, bz) gyroscope bias in dps.
        """
        self.sample_rate = sample_rate
        self.dt = 1.0 / sample_rate
        self.ahrs = MadgwickAHRS(beta=madgwick_beta, sample_freq=sample_rate)
        self.accel_bias = np.array(accel_bias, dtype=np.float64)
        self.gyro_bias = np.array(gyro_bias, dtype=np.float64)

        # State for jerk computation
        self._prev_linear_accel: np.ndarray | None = None

        # Processing count
        self._processed_count = 0

    def process(self, sample: dict) -> dict | None:
        """Process a single IMU sample.

        Args:
            sample: Dict with {'ax','ay','az','gx','gy','gz','t'} (or without t).

        Returns:
            Enhanced sample dict with orientation, linear accel, jerk, norms.
            None if sample is missing required fields.
        """
        try:
            ax = sample.get("ax", 0.0) - self.accel_bias[0]
            ay = sample.get("ay", 0.0) - self.accel_bias[1]
            az = sample.get("az", 0.0) - self.accel_bias[2]
            gx = sample.get("gx", 0.0) - self.gyro_bias[0]
            gy = sample.get("gy", 0.0) - self.gyro_bias[1]
            gz = sample.get("gz", 0.0) - self.gyro_bias[2]
            t = sample.get("t", 0.0)
        except (TypeError, KeyError):
            return None

        # Orientation
        quat = self.ahrs.update(gx, gy, gz, ax, ay, az)
        qw, qx, qy, qz = quat
        roll, pitch, yaw = self.ahrs.euler

        # Gravity removal: rotate [0, 0, 1] by inverse orientation
        # The gravity vector in sensor frame
        gravity = np.array([
            2.0 * (qx * qz - qw * qy),
            2.0 * (qw * qx + qy * qz),
            qw * qw - qx * qx - qy * qy + qz * qz,
        ])
        accel = np.array([ax, ay, az])
        linear_accel = accel - gravity

        lax, lay, laz = linear_accel

        # Jerk
        if self._prev_linear_accel is not None:
            jerk = (linear_accel - self._prev_linear_accel) / self.dt
        else:
            jerk = np.zeros(3)
        self._prev_linear_accel = linear_accel.copy()
        jx, jy, jz = jerk

        # Magnitude norms
        accel_norm = float(np.linalg.norm(accel))
        gyro_norm = float(np.linalg.norm([gx, gy, gz]))
        linear_norm = float(np.linalg.norm(linear_accel))
        jerk_norm = float(np.linalg.norm(jerk))

        self._processed_count += 1

        return {
            "ax": float(ax), "ay": float(ay), "az": float(az),
            "gx": float(gx), "gy": float(gy), "gz": float(gz),
            "qw": float(qw), "qx": float(qx), "qy": float(qy), "qz": float(qz),
            "roll": float(roll), "pitch": float(pitch), "yaw": float(yaw),
            "lax": float(lax), "lay": float(lay), "laz": float(laz),
            "jx": float(jx), "jy": float(jy), "jz": float(jz),
            "accel_norm": accel_norm,
            "gyro_norm": gyro_norm,
            "linear_norm": linear_norm,
            "jerk_norm": jerk_norm,
            "t": t,
        }

    def calibrate_bias(self, samples: list[dict], num_samples: int = 100) -> None:
        """Estimate resting bias from stationary samples.

        Args:
            samples: List of IMU sample dicts from a stationary period.
            num_samples: Max samples to use.
        """
        if not samples:
            return
        use = samples[:num_samples]
        accel_means = np.mean([[s.get("ax", 0), s.get("ay", 0), s.get("az", 0)] for s in use], axis=0)
        gyro_means = np.mean([[s.get("gx", 0), s.get("gy", 0), s.get("gz", 0)] for s in use], axis=0)
        # Accel bias: difference from [0, 0, 1] (gravity)
        self.accel_bias = np.array([accel_means[0], accel_means[1], accel_means[2] - 1.0])
        self.gyro_bias = np.array(gyro_means)

    def reset(self) -> None:
        """Reset AHRS and jerk state."""
        self.ahrs.reset()
        self._prev_linear_accel = None

    @property
    def processed_count(self) -> int:
        return self._processed_count
