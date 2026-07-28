"""IMU feature extraction.

Extracts ~200 physics-informed features from a gesture window of processed
IMU data. Features are organized into:
1. Statistical (per-axis moments, percentiles)
2. Orientation (quaternion, Euler statistics)
3. Spectral (FFT energy bands)
4. Cross-axis correlations
5. Jerk & impulse
6. Magnitude ratios
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats
from scipy.fft import fft, fftfreq


# Axes groups for feature computation
ACCEL_AXES = ["ax", "ay", "az"]
GYRO_AXES = ["gx", "gy", "gz"]
LINEAR_AXES = ["lax", "lay", "laz"]
JERK_AXES = ["jx", "jy", "jz"]
ALL_AXES = ACCEL_AXES + GYRO_AXES + LINEAR_AXES + JERK_AXES
IMPORTANT_SIGNALS = ALL_AXES + ["accel_norm", "gyro_norm", "linear_norm", "jerk_norm"]


class IMUFeatureExtractor:
    """Extracts comprehensive features from a window of processed IMU data.

    Usage::

        extractor = IMUFeatureExtractor(sample_rate=100.0)
        features = extractor.extract(processed_samples)
        # features is a 1D numpy array of ~200 features
        feature_names = extractor.feature_names()
    """

    def __init__(self, sample_rate: float = 100.0):
        """
        Args:
            sample_rate: IMU sample rate in Hz.
        """
        self.sample_rate = sample_rate
        self._feature_names: list[str] = []
        self._build_feature_names()

    def extract(self, samples: list[dict]) -> np.ndarray:
        """Extract feature vector from a list of processed IMU samples.

        Args:
            samples: List of dicts from IMUProcessor.process().
                     Must have at least 3 samples for statistical features.

        Returns:
            1D numpy array of feature values.
        """
        if len(samples) < 3:
            return np.zeros(len(self._feature_names), dtype=np.float32)

        n = len(samples)

        # Build numpy arrays for each signal
        signals = self._build_signal_arrays(samples)

        features: list[float] = []

        # ── 1. Statistical features per axis ─────────────────────────
        for axis in IMPORTANT_SIGNALS:
            arr = signals.get(axis)
            if arr is None or len(arr) < 3:
                features.extend([0.0] * 18)
                continue
            features.extend(self._statistical_features(arr))

        # ── 2. Orientation features ──────────────────────────────────
        features.extend(self._orientation_features(signals, n))

        # ── 3. Spectral features ─────────────────────────────────────
        features.extend(self._spectral_features(signals))

        # ── 4. Cross-axis correlations ───────────────────────────────
        features.extend(self._correlation_features(signals))

        # ── 5. Jerk & impulse features ───────────────────────────────
        features.extend(self._jerk_features(signals))

        # ── 6. Magnitude ratio features ──────────────────────────────
        features.extend(self._ratio_features(signals))

        return np.array(features, dtype=np.float32)

    def _build_signal_arrays(self, samples: list[dict]) -> dict[str, np.ndarray]:
        """Convert list of sample dicts to dict of numpy arrays."""
        n = len(samples)
        signals: dict[str, np.ndarray] = {}

        # Basic axes
        for key in ALL_AXES:
            signals[key] = np.array([s.get(key, 0.0) for s in samples], dtype=np.float64)

        # Norms
        for key in ["accel_norm", "gyro_norm", "linear_norm", "jerk_norm"]:
            signals[key] = np.array([s.get(key, 0.0) for s in samples], dtype=np.float64)

        # Orientation
        for key in ["qw", "qx", "qy", "qz", "roll", "pitch", "yaw"]:
            signals[key] = np.array([s.get(key, 0.0) for s in samples], dtype=np.float64)

        return signals

    @staticmethod
    def _statistical_features(arr: np.ndarray) -> list[float]:
        """Compute 18 statistical features for a 1D signal."""
        arr = np.asarray(arr, dtype=np.float64)
        n = len(arr)
        if n < 3:
            return [0.0] * 18

        try:
            return [
                float(np.mean(arr)),                            # mean
                float(np.std(arr, ddof=1)),                     # std
                float(np.min(arr)),                             # min
                float(np.max(arr)),                             # max
                float(np.max(arr) - np.min(arr)),               # range
                float(np.sqrt(np.mean(arr ** 2))),              # rms
                float(sp_stats.skew(arr)),                      # skewness
                float(sp_stats.kurtosis(arr, fisher=True)),     # kurtosis (excess)
                float(np.percentile(arr, 25)),                  # Q1
                float(np.percentile(arr, 50)),                  # median
                float(np.percentile(arr, 75)),                  # Q3
                float(np.percentile(arr, 75) - np.percentile(arr, 25)),  # IQR
                float(np.mean(np.abs(arr - np.mean(arr)))),     # MAD
                float(_zero_crossing_rate(arr)),                # ZCR
                float(_dominant_frequency(arr)),                # dominant freq
                float(_spectral_centroid(arr)),                 # spectral centroid
                float(_spectral_spread(arr)),                   # spectral spread
                float(_spectral_entropy(arr)),                  # spectral entropy
            ]
        except (ValueError, RuntimeWarning):
            return [0.0] * 18

    @staticmethod
    def _orientation_features(signals: dict, n: int) -> list[float]:
        """Compute ~24 orientation features."""
        feats = []
        for key in ["qw", "qx", "qy", "qz", "roll", "pitch", "yaw"]:
            arr = signals.get(key)
            if arr is None or len(arr) < 2:
                feats.extend([0.0, 0.0])
            else:
                feats.append(float(np.mean(arr)))
                feats.append(float(np.std(arr, ddof=1)))

        # Delta orientation (end - start)
        for key in ["roll", "pitch", "yaw"]:
            arr = signals.get(key)
            if arr is not None and len(arr) >= 2:
                feats.append(float(arr[-1] - arr[0]))
            else:
                feats.append(0.0)

        # Quaternion path length (geodesic distance)
        qw = signals.get("qw", np.array([]))
        qx = signals.get("qx", np.array([]))
        qy = signals.get("qy", np.array([]))
        qz = signals.get("qz", np.array([]))
        if len(qw) >= 2:
            path = 0.0
            for i in range(1, len(qw)):
                dot = abs(qw[i] * qw[i - 1] + qx[i] * qx[i - 1] +
                          qy[i] * qy[i - 1] + qz[i] * qz[i - 1])
                dot = min(dot, 1.0)
                path += np.arccos(dot)
            feats.append(float(path))
        else:
            feats.append(0.0)

        # Quaternion stability
        for key in ["qw", "qx", "qy", "qz"]:
            arr = signals.get(key)
            if arr is not None and len(arr) >= 2:
                feats.append(float(np.var(arr)))
            else:
                feats.append(0.0)

        return feats

    def _spectral_features(self, signals: dict) -> list[float]:
        """Compute spectral band energy features."""
        feats = []
        # Important signals for spectral analysis
        spectral_signals = [
            "ax", "ay", "az", "gx", "gy", "gz",
            "accel_norm", "gyro_norm", "linear_norm",
        ]

        for key in spectral_signals:
            arr = signals.get(key)
            if arr is None or len(arr) < 8:
                feats.extend([0.0] * 10)
                continue

            n = len(arr)
            yf = np.abs(fft(arr))
            freqs = fftfreq(n, 1.0 / self.sample_rate)
            positive_mask = freqs >= 0
            freqs_pos = freqs[positive_mask]
            yf_pos = yf[positive_mask]

            if len(yf_pos) < 4:
                feats.extend([0.0] * 10)
                continue

            total_energy = np.sum(yf_pos ** 2)
            if total_energy < 1e-12:
                feats.extend([0.0] * 10)
                continue

            # Energy in frequency bands
            bands = [(0, 1), (1, 3), (3, 8), (8, 20), (20, 50)]
            for low, high in bands:
                mask = (freqs_pos >= low) & (freqs_pos < high)
                band_energy = np.sum(yf_pos[mask] ** 2) / total_energy
                feats.append(float(band_energy))

            # Spectral entropy
            psd = yf_pos ** 2 / total_energy
            entropy = -np.sum(psd * np.log(psd + 1e-12))
            feats.append(float(entropy))

            # Number of spectral peaks
            peak_threshold = np.mean(yf_pos) + np.std(yf_pos)
            peak_count = np.sum(
                (yf_pos[1:-1] > yf_pos[:-2]) &
                (yf_pos[1:-1] > yf_pos[2:]) &
                (yf_pos[1:-1] > peak_threshold)
            )
            feats.append(float(peak_count))

            # Frequency and magnitude of highest peak
            if len(yf_pos) > 1:
                peak_idx = int(np.argmax(yf_pos[1:])) + 1  # Skip DC
                feats.append(float(freqs_pos[peak_idx]))
                feats.append(float(yf_pos[peak_idx] / (np.sqrt(total_energy) + 1e-12)))
            else:
                feats.extend([0.0, 0.0])

            # Ratio of peak to total
            feats.append(float(np.max(yf_pos[1:]) / (np.sqrt(total_energy) + 1e-12)))

        return feats

    @staticmethod
    def _correlation_features(signals: dict) -> list[float]:
        """Compute cross-axis correlation features."""
        feats = []
        pairs = [
            ("ax", "ay"), ("ax", "az"), ("ay", "az"),
            ("gx", "gy"), ("gx", "gz"), ("gy", "gz"),
            ("lax", "lay"), ("lax", "laz"), ("lay", "laz"),
            ("accel_norm", "gyro_norm"),
            ("accel_norm", "jerk_norm"),
        ]

        for k1, k2 in pairs:
            a1 = signals.get(k1)
            a2 = signals.get(k2)
            if a1 is not None and a2 is not None and len(a1) > 2 and len(a2) > 2:
                c = np.corrcoef(a1, a2)[0, 1]
                feats.append(float(0.0 if np.isnan(c) else c))
            else:
                feats.append(0.0)

        return feats

    @staticmethod
    def _jerk_features(signals: dict) -> list[float]:
        """Compute jerk and impulsiveness features."""
        feats = []
        jerk_norm = signals.get("jerk_norm")
        linear_norm = signals.get("linear_norm")

        if jerk_norm is not None and len(jerk_norm) > 2:
            feats.append(float(np.max(jerk_norm)))          # peak jerk
            feats.append(float(np.mean(jerk_norm)))         # mean jerk

            # Jerk peak count (above 2σ)
            threshold = np.mean(jerk_norm) + 2.0 * np.std(jerk_norm)
            peaks = (jerk_norm[1:-1] > jerk_norm[:-2]) & (jerk_norm[1:-1] > jerk_norm[2:])
            peak_count = np.sum(jerk_norm[1:-1][peaks] > threshold)
            feats.append(float(peak_count))

            # Peak spacing regularity
            if peak_count >= 2:
                peak_indices = np.where(jerk_norm[1:-1][peaks])[0]
                if len(peak_indices) >= 2:
                    spacing_std = float(np.std(np.diff(peak_indices)))
                    feats.append(spacing_std)
                else:
                    feats.append(0.0)
            else:
                feats.append(0.0)

            # Impulsiveness: max / mean ratio
            mean_j = np.mean(jerk_norm)
            feats.append(float(np.max(jerk_norm) / (mean_j + 1e-9)))

            # Timing features
            max_idx = int(np.argmax(jerk_norm))
            n = len(jerk_norm)
            feats.append(float(max_idx / max(n - 1, 1)))        # normalized time to peak
            feats.append(float((n - 1 - max_idx) / max(n - 1, 1)))  # time from peak to end
        else:
            feats.extend([0.0] * 7)

        return feats

    @staticmethod
    def _ratio_features(signals: dict) -> list[float]:
        """Compute magnitude ratio features."""
        feats = []

        # X / Y acceleration magnitude
        ax_arr = signals.get("ax")
        ay_arr = signals.get("ay")
        if ax_arr is not None and ay_arr is not None:
            ax_mag = np.sqrt(np.mean(ax_arr ** 2))
            ay_mag = np.sqrt(np.mean(ay_arr ** 2))
            feats.append(float(ax_mag / (ay_mag + 1e-9)))
        else:
            feats.append(0.0)

        # Z / horizontal ratio
        az_arr = signals.get("az")
        if ax_arr is not None and ay_arr is not None and az_arr is not None:
            z_mag = np.sqrt(np.mean(az_arr ** 2))
            h_mag = np.sqrt(np.mean(ax_arr ** 2) + np.mean(ay_arr ** 2))
            feats.append(float(z_mag / (h_mag + 1e-9)))
        else:
            feats.append(0.0)

        # Linear / gravitational ratio
        la_norm = signals.get("linear_norm")
        accel_norm = signals.get("accel_norm")
        if la_norm is not None and accel_norm is not None:
            feats.append(float(np.mean(la_norm) / (np.mean(accel_norm) + 1e-9)))
        else:
            feats.append(0.0)

        # Gyro / accel ratio
        gyro_norm = signals.get("gyro_norm")
        if gyro_norm is not None and accel_norm is not None:
            feats.append(float(np.mean(gyro_norm) / (np.mean(accel_norm) + 1e-9)))
        else:
            feats.append(0.0)

        # Dominant axis ratio
        if ax_arr is not None and ay_arr is not None and az_arr is not None:
            axes_energy = [
                np.mean(ax_arr ** 2),
                np.mean(ay_arr ** 2),
                np.mean(az_arr ** 2),
            ]
            feats.append(float(max(axes_energy) / (sum(axes_energy) + 1e-9)))
        else:
            feats.append(0.0)

        return feats

    def _build_feature_names(self) -> None:
        """Build list of human-readable feature names (called once in __init__)."""
        names: list[str] = []

        # Statistical
        stats_suffixes = [
            "mean", "std", "min", "max", "range", "rms",
            "skew", "kurtosis", "q25", "median", "q75", "iqr",
            "mad", "zcr", "dom_freq", "spec_centroid", "spec_spread", "spec_entropy",
        ]
        for axis in IMPORTANT_SIGNALS:
            for suffix in stats_suffixes:
                names.append(f"imu_{axis}_{suffix}")

        # Orientation
        for key in ["qw", "qx", "qy", "qz", "roll", "pitch", "yaw"]:
            names.append(f"imu_{key}_mean")
            names.append(f"imu_{key}_std")
        for key in ["roll", "pitch", "yaw"]:
            names.append(f"imu_delta_{key}")
        names.append("imu_quat_path_length")
        for key in ["qw", "qx", "qy", "qz"]:
            names.append(f"imu_{key}_variance")

        # Spectral
        spectral_signals = [
            "ax", "ay", "az", "gx", "gy", "gz",
            "accel_norm", "gyro_norm", "linear_norm",
        ]
        bands = ["0_1hz", "1_3hz", "3_8hz", "8_20hz", "20_50hz"]
        for sig in spectral_signals:
            for band in bands:
                names.append(f"imu_{sig}_energy_{band}")
            names.append(f"imu_{sig}_spec_entropy")
            names.append(f"imu_{sig}_peak_count")
            names.append(f"imu_{sig}_peak_freq")
            names.append(f"imu_{sig}_peak_mag")
            names.append(f"imu_{sig}_peak_ratio")

        # Correlations
        corr_pairs = [
            ("ax", "ay"), ("ax", "az"), ("ay", "az"),
            ("gx", "gy"), ("gx", "gz"), ("gy", "gz"),
            ("lax", "lay"), ("lax", "laz"), ("lay", "laz"),
            ("accel_norm", "gyro_norm"), ("accel_norm", "jerk_norm"),
        ]
        for k1, k2 in corr_pairs:
            names.append(f"imu_corr_{k1}_{k2}")

        # Jerk
        jerk_suffixes = [
            "max_jerk", "mean_jerk", "jerk_peak_count",
            "jerk_peak_spacing_std", "impulsiveness",
            "time_to_peak_jerk", "time_from_peak_jerk",
        ]
        for suffix in jerk_suffixes:
            names.append(f"imu_{suffix}")

        # Ratios
        ratio_names = [
            "xy_accel_ratio", "z_horizontal_ratio",
            "linear_grav_ratio", "gyro_accel_ratio",
            "dominant_axis_ratio",
        ]
        for name in ratio_names:
            names.append(f"imu_{name}")

        self._feature_names = names

    def feature_names(self) -> list[str]:
        """Return human-readable feature names in order."""
        return list(self._feature_names)

    @property
    def n_features(self) -> int:
        return len(self._feature_names)


# ── Spectral helpers ────────────────────────────────────────────────────

def _zero_crossing_rate(arr: np.ndarray) -> float:
    """Compute zero-crossing rate for a zero-centered signal."""
    arr = np.asarray(arr)
    if len(arr) < 2:
        return 0.0
    centered = arr - np.mean(arr)
    crossings = np.sum(np.abs(np.diff(np.sign(centered))) > 0)
    return crossings / (len(arr) - 1)


def _dominant_frequency(arr: np.ndarray, fs: float = 100.0) -> float:
    """Find the dominant frequency of a signal."""
    arr = np.asarray(arr)
    n = len(arr)
    if n < 4:
        return 0.0
    yf = np.abs(fft(arr))
    freqs = fftfreq(n, 1.0 / fs)
    pos_mask = freqs > 0
    if not pos_mask.any():
        return 0.0
    freqs_pos = freqs[pos_mask]
    yf_pos = yf[pos_mask]
    # Skip DC (index 0 in positive freqs might include DC)
    if len(yf_pos) > 1:
        peak_idx = int(np.argmax(yf_pos[1:])) + 1
        return float(freqs_pos[peak_idx])
    return 0.0


def _spectral_centroid(arr: np.ndarray) -> float:
    """Compute spectral centroid."""
    arr = np.asarray(arr)
    n = len(arr)
    if n < 4:
        return 0.0
    yf = np.abs(fft(arr))
    freqs = fftfreq(n, 1.0 / 1.0)[:n // 2]
    yf_half = yf[:n // 2]
    total = np.sum(yf_half)
    if total < 1e-12:
        return 0.0
    return float(np.sum(freqs * yf_half) / total)


def _spectral_spread(arr: np.ndarray) -> float:
    """Compute spectral spread (std around centroid)."""
    arr = np.asarray(arr)
    n = len(arr)
    if n < 4:
        return 0.0
    centroid = _spectral_centroid(arr)
    yf = np.abs(fft(arr))
    freqs = fftfreq(n, 1.0 / 1.0)[:n // 2]
    yf_half = yf[:n // 2]
    total = np.sum(yf_half)
    if total < 1e-12:
        return 0.0
    return float(np.sqrt(np.sum((freqs - centroid) ** 2 * yf_half) / total))


def _spectral_entropy(arr: np.ndarray) -> float:
    """Compute spectral entropy."""
    arr = np.asarray(arr)
    n = len(arr)
    if n < 4:
        return 0.0
    yf = np.abs(fft(arr))[:n // 2]
    total = np.sum(yf)
    if total < 1e-12:
        return 0.0
    psd = yf / total
    psd = psd[psd > 0]
    return float(-np.sum(psd * np.log(psd)))
