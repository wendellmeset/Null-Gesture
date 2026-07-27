"""6-DOF IMU tracking: Madgwick AHRS + ZUPT-bounded position integration.

Madgwick filter fuses gyro + accel for drift-free roll/pitch.
Yaw drifts (no magnetometer). ZUPT resets velocity/position when stationary.

Reference: Madgwick, S.O.H. "An efficient orientation filter for IMU/MARG
arrays." 2010.  http://x-io.co.uk/open-source-imu-and-ahrs-algorithms/
"""

from __future__ import annotations

import numpy as np


class MadgwickAHRS:
    """6-DOF Madgwick filter — gyro + accelerometer only.

    Estimates orientation quaternion [w, x, y, z] representing rotation
    from sensor frame to world frame (where gravity = [0, 0, -1]).
    """

    def __init__(self, beta: float = 0.05, sample_freq: float = 60.0):
        """
        Args:
            beta: filter gain (0.03-0.10 typical). Higher = faster convergence
                  but more noise-sensitive.
            sample_freq: sample rate in Hz for time normalization.
        """
        self.beta = beta
        self.sample_freq = sample_freq
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # w, x, y, z

    def update(self, gyr: np.ndarray, acc: np.ndarray, dt: float | None = None) -> np.ndarray:
        """Update filter with gyro (rad/s) and accel (g).

        Args:
            gyr: [gx, gy, gz] in rad/s
            acc: [ax, ay, az] in g
            dt: time step in seconds. If None, uses 1/sample_freq.

        Returns:
            Updated quaternion [w, x, y, z].
        """
        if dt is None:
            dt = 1.0 / self.sample_freq

        q = self.q.copy()
        gx, gy, gz = gyr
        ax, ay, az = acc

        # Normalise accelerometer
        acc_norm = np.sqrt(ax*ax + ay*ay + az*az)
        if acc_norm < 1e-6:
            # No accel reference — integrate gyro only
            q_dot = self._gyro_derivative(q, gx, gy, gz)
            q += q_dot * dt
            q /= np.linalg.norm(q)
            self.q = q
            return q

        ax /= acc_norm
        ay /= acc_norm
        az /= acc_norm

        # ── Gradient descent step (accel correction) ──────────────
        # Objective function f_g(q, [0,0,-1], acc) and its Jacobian
        # From Madgwick paper, equations 25-28

        q1, q2, q3, q4 = q  # q1=w, q2=x, q3=y, q4=z

        # f_g — error between estimated gravity direction and measured accel
        f1 = 2.0 * (q2*q4 - q1*q3) - ax
        f2 = 2.0 * (q1*q2 + q3*q4) - ay
        f3 = 2.0 * (0.5 - q2*q2 - q3*q3) - az

        # Jacobian of f_g w.r.t. q (4x3)
        J11 = -2.0 * q3;  J12 =  2.0 * q4;  J13 = -2.0 * q1;  J14 =  2.0 * q2
        J21 =  2.0 * q2;  J22 =  2.0 * q1;  J23 =  2.0 * q4;  J24 =  2.0 * q3
        J31 =  0.0;       J32 = -4.0 * q2;  J33 = -4.0 * q3;  J34 =  0.0

        # Gradient: ∇f = J^T · f
        grad_x = J11*f1 + J21*f2 + J31*f3
        grad_y = J12*f1 + J22*f2 + J32*f3
        grad_z = J13*f1 + J23*f2 + J33*f3
        grad_w = J14*f1 + J24*f2 + J34*f3

        # Normalise gradient
        grad_norm = np.sqrt(grad_w*grad_w + grad_x*grad_x + grad_y*grad_y + grad_z*grad_z)
        if grad_norm > 1e-9:
            grad_w /= grad_norm
            grad_x /= grad_norm
            grad_y /= grad_norm
            grad_z /= grad_norm

        # ── Gyro derivative ───────────────────────────────────────
        q_dot_w, q_dot_x, q_dot_y, q_dot_z = self._gyro_derivative(q, gx, gy, gz)

        # ── Fused derivative: gyro - beta * gradient ──────────────
        q += np.array([
            q_dot_w - self.beta * grad_w,
            q_dot_x - self.beta * grad_x,
            q_dot_y - self.beta * grad_y,
            q_dot_z - self.beta * grad_z,
        ]) * dt

        q /= np.linalg.norm(q)
        self.q = q
        return q

    @staticmethod
    def _gyro_derivative(q: np.ndarray, gx: float, gy: float, gz: float) -> tuple[float, float, float, float]:
        """Quaternion derivative from gyro rates only."""
        q1, q2, q3, q4 = q
        return (
            -0.5 * (q2*gx + q3*gy + q4*gz),
             0.5 * (q1*gx + q3*gz - q4*gy),
             0.5 * (q1*gy - q2*gz + q4*gx),
             0.5 * (q1*gz + q2*gy - q3*gx),
        )

    def reset(self) -> None:
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


# ── Quaternion helpers ──────────────────────────────────────────────

def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Quaternion [w,x,y,z] → 4×4 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z,   2*x*y-2*w*z,   2*x*z+2*w*y, 0],
        [  2*x*y+2*w*z, 1-2*x*x-2*z*z,   2*y*z-2*w*x, 0],
        [  2*x*z-2*w*y,   2*y*z+2*w*x, 1-2*x*x-2*y*y, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)

def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_mult(q: np.ndarray, r: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = r
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q."""
    qv = np.array([0.0, v[0], v[1], v[2]])
    return quat_mult(quat_mult(q, qv), quat_conj(q))[1:]


# ── Position tracker with ZUPT ───────────────────────────────────────

class ZUPTTracker:
    """Double-integrates acceleration for position, with Zero-Velocity Updates.

    When the IMU is detected as stationary (gyro below threshold for sufficient
    time), velocity is reset to zero and position drifts toward origin.
    Between ZUPT events, the tracker accumulates linear displacement from
    gravity-subtracted world-frame acceleration.
    """

    def __init__(
        self,
        still_thresh: float = 8.0,    # dps — gyro magnitude below this = still
        still_time: float = 0.3,      # seconds of stillness to trigger ZUPT
        vel_damp: float = 0.995,      # per-frame velocity damping during motion
        pos_damp: float = 0.998,      # per-frame position centering
        zupt_pos_damp: float = 0.90,  # position decay during ZUPT (return to origin)
    ):
        self.still_thresh = still_thresh
        self.still_time = still_time
        self.vel_damp = vel_damp
        self.pos_damp = pos_damp
        self.zupt_pos_damp = zupt_pos_damp

        self.pos = np.zeros(3, dtype=np.float64)
        self.vel = np.zeros(3, dtype=np.float64)
        self._still_accum: float = 0.0
        self._was_still: bool = True

    def update(
        self,
        acc_world: np.ndarray,   # world-frame accel, gravity already subtracted (m/s²)
        gyro_mag: float,         # gyro magnitude in dps
        dt: float,               # time step in seconds
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        """Integrate one frame.

        Returns (position, velocity, is_still).
        """
        is_still = gyro_mag < self.still_thresh

        if is_still:
            self._still_accum += dt
            if self._still_accum >= self.still_time:
                # ZUPT: zero velocity, pull position toward origin
                self.vel *= 0.0
                self.pos *= self.zupt_pos_damp
                self._was_still = True
                return self.pos, self.vel, True
        else:
            self._still_accum = 0.0
            self._was_still = False

        # ── Integrate ─────────────────────────────────────────────
        # Dead zone: ignore accelerations below noise floor
        acc_mag = np.linalg.norm(acc_world)
        if acc_mag < 0.03:
            acc_world_clean = np.zeros(3)
        elif acc_mag < 0.08:
            acc_world_clean = acc_world * (acc_mag - 0.03) / 0.05
        else:
            acc_world_clean = acc_world

        self.vel += acc_world_clean * dt
        self.vel *= self.vel_damp
        self.pos += self.vel * dt
        self.pos *= self.pos_damp

        return self.pos, self.vel, False

    def reset(self) -> None:
        self.pos.fill(0)
        self.vel.fill(0)
        self._still_accum = 0.0
        self._was_still = True
