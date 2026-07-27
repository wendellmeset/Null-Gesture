"""Feature extraction — MATLAB-style statistical features from IMU windows.

Based on MathWorks "Gesture Recognition Using Inertial Measurement Units":
  - Per-axis: mean, std, rms, peak-to-peak, skewness, kurtosis
  - Cross-axis: correlation, signal magnitude area
  - Frequency: dominant frequency, spectral energy
"""

from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal
from scipy import stats as scipy_stats


def extract_features(window: np.ndarray, fs: float = 50.0) -> np.ndarray:
    """Extract feature vector from a gesture window.

    Args:
        window: (T, 6) array [ax, ay, az, gx, gy, gz].
        fs: Sample rate in Hz.

    Returns:
        1D feature vector.
    """
    T, C = window.shape
    features: list[float] = []

    for c in range(C):
        channel = window[:, c]

        # Time-domain features
        features.append(float(np.mean(channel)))
        features.append(float(np.std(channel)))
        features.append(float(np.sqrt(np.mean(channel ** 2))))  # RMS
        features.append(float(np.max(channel) - np.min(channel)))  # peak-to-peak

        # Higher-order statistics
        if np.std(channel) > 1e-8:
            features.append(float(scipy_stats.skew(channel)))
            features.append(float(scipy_stats.kurtosis(channel)))
        else:
            features.append(0.0)
            features.append(0.0)

    # ── Signal magnitude area (SMA) ──────────────────────────────
    # SMA = sum(|ax| + |ay| + |az| + |gx| + |gy| + |gz|) / T
    sma = np.sum(np.abs(window)) / T
    features.append(float(sma))

    # ── Gyro magnitude mean, std ─────────────────────────────────
    gyr_mag = np.linalg.norm(window[:, 3:], axis=1)
    features.append(float(np.mean(gyr_mag)))
    features.append(float(np.std(gyr_mag)))

    # ── Accel magnitude mean, std ────────────────────────────────
    acc_mag = np.linalg.norm(window[:, :3], axis=1)
    features.append(float(np.mean(acc_mag)))
    features.append(float(np.std(acc_mag)))

    # ── Cross-axis correlation (gyro) ────────────────────────────
    gyr = window[:, 3:]
    gcorr = np.corrcoef(gyr.T)
    features.append(float(gcorr[0, 1]) if not np.isnan(gcorr[0, 1]) else 0.0)
    features.append(float(gcorr[0, 2]) if not np.isnan(gcorr[0, 2]) else 0.0)
    features.append(float(gcorr[1, 2]) if not np.isnan(gcorr[1, 2]) else 0.0)

    # ── Dominant frequency (gyro magnitude) ──────────────────────
    if T > 4:
        freqs, psd = scipy_signal.welch(gyr_mag, fs=fs, nperseg=min(T, 32))
        dom_freq = freqs[np.argmax(psd)]
        features.append(float(dom_freq))
        features.append(float(np.sum(psd)))  # total spectral energy
    else:
        features.append(0.0)
        features.append(0.0)

    # ── Zero-crossing rate (gyro) ────────────────────────────────
    gyr_centered = gyr - gyr.mean(axis=0)
    zc = np.sum(np.abs(np.diff(np.signbit(gyr_centered), axis=0))) / T
    features.append(float(zc))

    # ── Energy ratio: gyro / accel ───────────────────────────────
    gyr_energy = np.sum(gyr_mag ** 2)
    acc_energy = np.sum(acc_mag ** 2)
    features.append(float(gyr_energy / (acc_energy + 1e-8)))

    # ── Peak timing (where in the window the max gyro occurs) ────
    features.append(float(np.argmax(gyr_mag)) / max(T - 1, 1))

    return np.array(features, dtype=np.float64)


def extract_features_batch(windows: np.ndarray, fs: float = 50.0) -> np.ndarray:
    """Extract features from a batch of gesture windows.

    Args:
        windows: (N, T, 6) array of gesture windows.
        fs: Sample rate.

    Returns:
        (N, F) feature matrix.
    """
    return np.array([extract_features(w, fs) for w in windows], dtype=np.float64)


def get_feature_names() -> list[str]:
    """Return human-readable feature names for the feature vector."""
    names = []
    for ch in ["ax", "ay", "az", "gx", "gy", "gz"]:
        for stat in ["mean", "std", "rms", "ptp", "skew", "kurt"]:
            names.append(f"{ch}_{stat}")
    names += [
        "sma",
        "gyr_mag_mean", "gyr_mag_std",
        "acc_mag_mean", "acc_mag_std",
        "gyr_corr_xy", "gyr_corr_xz", "gyr_corr_yz",
        "dom_freq", "spectral_energy",
        "gyr_zcr",
        "gyr_acc_energy_ratio",
        "peak_timing",
    ]
    return names


# ══════════════════════════════════════════════════════════════════════
# mmWave-specific: geometric trajectory features
# ══════════════════════════════════════════════════════════════════════

def extract_mmwave_features(trajectory: np.ndarray, fs: float = 20.0) -> np.ndarray:
    """Extract geometric features from a mmWave 3D hand trajectory.

    Args:
        trajectory: (T, 3) array [x, y, z] in meters.
        fs: mmWave frame rate in Hz.

    Returns:
        1D feature vector (36 features).
    """
    T = trajectory.shape[0]
    x, y, z = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2]

    # Velocity (first difference)
    vel = np.diff(trajectory, axis=0) * fs  # m/s
    speed = np.linalg.norm(vel, axis=1)
    accel = np.diff(vel, axis=0) * fs
    accel_mag = np.linalg.norm(accel, axis=1)

    features: list[float] = []

    # ── Position statistics ──────────────────────────────────────
    for arr, name in [(x, "x"), (y, "y"), (z, "z")]:
        features.append(float(np.mean(arr)))
        features.append(float(np.std(arr)))
        features.append(float(np.max(arr) - np.min(arr)))  # range

    # ── Displacement (start → end) ───────────────────────────────
    displacement = trajectory[-1] - trajectory[0]
    features.append(float(np.linalg.norm(displacement)))
    features.append(float(displacement[0]))  # dx
    features.append(float(displacement[1]))  # dy
    features.append(float(displacement[2]))  # dz

    # ── Path length ──────────────────────────────────────────────
    path_len = float(np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)))
    features.append(path_len)

    # ── Tortuosity (path / displacement) ─────────────────────────
    disp_len = features[-4]  # displacement norm
    features.append(path_len / (disp_len + 1e-6))

    # ── Speed statistics ─────────────────────────────────────────
    features.append(float(np.mean(speed)))
    features.append(float(np.std(speed)))
    features.append(float(np.max(speed)))
    features.append(float(np.argmax(speed)) / max(len(speed) - 1, 1))  # peak timing

    # ── Acceleration statistics ──────────────────────────────────
    if len(accel_mag) > 0:
        features.append(float(np.mean(accel_mag)))
        features.append(float(np.std(accel_mag)))
        features.append(float(np.max(accel_mag)))
    else:
        features.extend([0.0, 0.0, 0.0])

    # ── Direction of dominant motion (PCA on trajectory) ─────────
    centered = trajectory - trajectory.mean(axis=0)
    if T > 2 and np.std(centered) > 1e-6:
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        main_dir = Vt[0]  # principal direction
        features.append(float(main_dir[0]))
        features.append(float(main_dir[1]))
        features.append(float(main_dir[2]))
        # Explained variance ratio of first PC
        vars_ = np.linalg.svd(centered, full_matrices=False)[1] ** 2
        features.append(float(vars_[0] / (vars_.sum() + 1e-8)))
    else:
        features.extend([0.0, 0.0, 0.0, 1.0])

    # ── Enclosed area (2D projection on principal plane) ─────────
    if T > 3:
        proj = centered[:, :2]  # project to first 2 dimensions of centered data
        try:
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            proj = centered @ Vt[:2].T  # project to principal plane
            # Polygon area via shoelace formula
            xp, yp = proj[:, 0], proj[:, 1]
            area = 0.5 * np.abs(np.dot(xp, np.roll(yp, 1)) - np.dot(yp, np.roll(xp, 1)))
            features.append(float(area))
        except np.linalg.LinAlgError:
            features.append(0.0)
    else:
        features.append(0.0)

    # ── Turning angle (total curvature) ──────────────────────────
    if T > 2:
        dirs = np.diff(trajectory, axis=0)
        dirs_norm = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8)
        cos_angles = np.sum(dirs_norm[:-1] * dirs_norm[1:], axis=1)
        cos_angles = np.clip(cos_angles, -1, 1)
        total_turn = float(np.sum(np.arccos(cos_angles)))
        features.append(total_turn)
        features.append(float(np.mean(np.abs(np.arccos(cos_angles)))) if len(cos_angles) > 0 else 0.0)
    else:
        features.extend([0.0, 0.0])

    # ── Zero-crossing rate (per axis) ────────────────────────────
    for arr in [x, y, z]:
        centered_arr = arr - arr.mean()
        if len(centered_arr) > 1:
            zcr = np.sum(np.abs(np.diff(np.signbit(centered_arr)))) / T
        else:
            zcr = 0.0
        features.append(float(zcr))

    # ── Duration ─────────────────────────────────────────────────
    features.append(float(T / fs))

    return np.array(features, dtype=np.float64)


def get_mmwave_feature_names() -> list[str]:
    return [
        "x_mean", "x_std", "x_range",
        "y_mean", "y_std", "y_range",
        "z_mean", "z_std", "z_range",
        "displacement", "dx", "dy", "dz",
        "path_length", "tortuosity",
        "speed_mean", "speed_std", "speed_max", "speed_peak_timing",
        "accel_mean", "accel_std", "accel_max",
        "main_dir_x", "main_dir_y", "main_dir_z", "pca_ratio",
        "enclosed_area",
        "total_turn", "mean_turn_angle",
        "zcr_x", "zcr_y", "zcr_z",
        "duration",
    ]
