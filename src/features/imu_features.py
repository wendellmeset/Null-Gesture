"""IMU feature extraction: statistical, spectral, and temporal features.

Windowing is handled by the caller (detector). This module provides pure
functions that take a window of samples and return feature vectors.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np


class IMUWindow:
    """Accumulates IMU samples into a sliding window and computes features."""

    def __init__(self, window_samples: int = 50) -> None:
        """window_samples: number of samples in window (50 = 500ms at 100Hz)."""
        self._window_samples = window_samples
        self._ax: deque[float] = deque(maxlen=window_samples)
        self._ay: deque[float] = deque(maxlen=window_samples)
        self._az: deque[float] = deque(maxlen=window_samples)
        self._gx: deque[float] = deque(maxlen=window_samples)
        self._gy: deque[float] = deque(maxlen=window_samples)
        self._gz: deque[float] = deque(maxlen=window_samples)
        self._amag: deque[float] = deque(maxlen=window_samples)
        self._gmag: deque[float] = deque(maxlen=window_samples)

    def push(
        self,
        ax: float, ay: float, az: float,
        gx: float, gy: float, gz: float,
    ) -> None:
        self._ax.append(ax)
        self._ay.append(ay)
        self._az.append(az)
        self._gx.append(gx)
        self._gy.append(gy)
        self._gz.append(gz)
        self._amag.append(math.sqrt(ax * ax + ay * ay + az * az))
        self._gmag.append(math.sqrt(gx * gx + gy * gy + gz * gz))

    @property
    def full(self) -> bool:
        return len(self._ax) >= self._window_samples

    def __len__(self) -> int:
        return len(self._ax)

    def compute_features(self) -> dict[str, float]:
        """Compute feature vector for the current window.

        Returns a dict of feature_name -> float_value.
        """
        if len(self._ax) < 3:
            return {}

        ax = np.array(self._ax, dtype=np.float64)
        ay = np.array(self._ay, dtype=np.float64)
        az = np.array(self._az, dtype=np.float64)
        gx = np.array(self._gx, dtype=np.float64)
        gy = np.array(self._gy, dtype=np.float64)
        gz = np.array(self._gz, dtype=np.float64)
        amag = np.array(self._amag, dtype=np.float64)
        gmag = np.array(self._gmag, dtype=np.float64)

        features: dict[str, float] = {}

        # ── Statistical features ─────────────────────────────────────
        for name, signal in [
            ("ax", ax), ("ay", ay), ("az", az),
            ("gx", gx), ("gy", gy), ("gz", gz),
            ("amag", amag), ("gmag", gmag),
        ]:
            features[f"{name}_mean"] = float(np.mean(signal))
            features[f"{name}_std"] = float(np.std(signal))
            features[f"{name}_min"] = float(np.min(signal))
            features[f"{name}_max"] = float(np.max(signal))
            features[f"{name}_range"] = features[f"{name}_max"] - features[f"{name}_min"]

            # Zero-crossing rate
            zcr = np.sum(np.abs(np.diff(np.signbit(signal)))) / (len(signal) - 1)
            features[f"{name}_zcr"] = float(zcr)

        # ── Jerk (derivative of acceleration magnitude) ──────────────
        if len(amag) >= 2:
            jerk = np.diff(amag)
            features["jerk_mean"] = float(np.mean(np.abs(jerk)))
            features["jerk_std"] = float(np.std(jerk))
            features["jerk_max"] = float(np.max(np.abs(jerk)))
        else:
            features["jerk_mean"] = 0.0
            features["jerk_std"] = 0.0
            features["jerk_max"] = 0.0

        # ── Spectral features (FFT) ──────────────────────────────────
        for name, signal in [("gyro_z", gz), ("accel_x", ax), ("accel_y", ay)]:
            n = len(signal)
            if n >= 4:
                fft = np.abs(np.fft.rfft(signal))
                freqs = np.fft.rfftfreq(n, d=0.01)  # 100 Hz → dt=10ms
                if len(fft) > 1:
                    # Dominant frequency (skip DC)
                    dominant_idx = np.argmax(fft[1:]) + 1
                    features[f"{name}_dom_freq"] = float(freqs[dominant_idx])
                    features[f"{name}_dom_energy"] = float(fft[dominant_idx])

                    # Spectral centroid
                    features[f"{name}_spectral_centroid"] = float(
                        np.sum(freqs * fft) / (np.sum(fft) + 1e-10)
                    )

                    # Energy ratio: low (0-3 Hz) vs high (3-50 Hz)
                    low_mask = freqs <= 3.0
                    high_mask = freqs > 3.0
                    low_energy = np.sum(fft[low_mask])
                    high_energy = np.sum(fft[high_mask]) + 1e-10
                    features[f"{name}_energy_ratio_lo_hi"] = float(low_energy / high_energy)
                else:
                    features[f"{name}_dom_freq"] = 0.0
                    features[f"{name}_dom_energy"] = 0.0
                    features[f"{name}_spectral_centroid"] = 0.0
                    features[f"{name}_energy_ratio_lo_hi"] = 0.0
            else:
                features[f"{name}_dom_freq"] = 0.0
                features[f"{name}_dom_energy"] = 0.0
                features[f"{name}_spectral_centroid"] = 0.0
                features[f"{name}_energy_ratio_lo_hi"] = 0.0

        # ── Peak features ────────────────────────────────────────────
        # Count peaks in acceleration magnitude (for boxing detection)
        from scipy.signal import find_peaks

        if len(amag) >= 10:
            peaks, properties = find_peaks(amag, height=np.std(amag) * 1.5, distance=5)
            features["amag_peak_count"] = float(len(peaks))
            features["amag_peak_mean_height"] = (
                float(np.mean(properties["peak_heights"])) if len(peaks) > 0 else 0.0
            )
        else:
            features["amag_peak_count"] = 0.0
            features["amag_peak_mean_height"] = 0.0

        return features


def compute_imu_feature_vector(imu_features: dict[str, float], feature_names: list[str]) -> np.ndarray:
    """Extract ordered feature vector from feature dict. Missing features → 0.0."""
    return np.array([imu_features.get(name, 0.0) for name in feature_names], dtype=np.float64)
