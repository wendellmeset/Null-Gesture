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
