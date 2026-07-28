"""mmWave feature extraction.

Extracts ~150 physics-informed features from a gesture window of processed
mmWave radar frames. Features cover:
1. Centroid trajectory statistics
2. Trajectory shape descriptors
3. Point cloud geometry
4. Range-Doppler heatmap statistics
5. Micro-Doppler spectrogram features
6. Velocity profile features
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats
from scipy.fft import fft, fftfreq


class MMWaveFeatureExtractor:
    """Extracts comprehensive features from a window of mmWave radar frames.

    Usage::

        extractor = MMWaveFeatureExtractor(radar_frame_rate=20.0)
        features = extractor.extract(processed_frames)
        # features is a 1D numpy array of ~150 features
        feature_names = extractor.feature_names()
    """

    def __init__(self, radar_frame_rate: float = 20.0):
        """
        Args:
            radar_frame_rate: Approximate radar frame rate in Hz.
        """
        self.frame_rate = radar_frame_rate
        self._feature_names: list[str] = []
        self._build_feature_names()

    def extract(self, frames: list[dict]) -> np.ndarray:
        """Extract feature vector from a list of processed radar frames.

        Args:
            frames: List of dicts from MMWaveProcessor.process().
                    Each dict should have 'centroid', 'velocity_centroid',
                    'rd_heatmap', 'filtered_points', 'num_points', etc.

        Returns:
            1D numpy array of feature values.
        """
        if len(frames) < 2:
            return np.zeros(len(self._feature_names), dtype=np.float32)

        # Filter out None entries
        valid_frames = [f for f in frames if f is not None]
        if len(valid_frames) < 2:
            return np.zeros(len(self._feature_names), dtype=np.float32)

        features: list[float] = []

        # ── 1. Centroid trajectory features ─────────────────────────
        features.extend(self._trajectory_stats(valid_frames))

        # ── 2. Trajectory shape features ────────────────────────────
        features.extend(self._trajectory_shape(valid_frames))

        # ── 3. Point cloud geometry features ────────────────────────
        features.extend(self._point_cloud_geometry(valid_frames))

        # ── 4. Range-Doppler heatmap features ───────────────────────
        features.extend(self._rd_heatmap_features(valid_frames))

        # ── 5. Micro-Doppler features ───────────────────────────────
        features.extend(self._micro_doppler_features(valid_frames))

        # ── 6. Velocity profile features ────────────────────────────
        features.extend(self._velocity_profile_features(valid_frames))

        return np.array(features, dtype=np.float32)

    @staticmethod
    def _trajectory_stats(frames: list[dict]) -> list[float]:
        """Compute statistical features of the centroid trajectory.

        For each of {x, y, z, range, velocity}: mean, std, min, max, range,
        start, end, delta.
        """
        centroids = np.array([f["centroid"] for f in frames])  # (N, 3)
        velocities = np.array([f.get("velocity_centroid", 0.0) for f in frames])
        ranges = np.linalg.norm(centroids, axis=1)

        feats = []

        # Per-coordinate stats
        for i, label in enumerate(["x", "y", "z"]):
            arr = centroids[:, i]
            feats += [
                float(np.mean(arr)),
                float(np.std(arr, ddof=1)),
                float(np.min(arr)),
                float(np.max(arr)),
                float(np.max(arr) - np.min(arr)),
                float(arr[0]),
                float(arr[-1]),
                float(arr[-1] - arr[0]),
            ]

        # Range stats
        feats += [
            float(np.mean(ranges)),
            float(np.std(ranges, ddof=1)),
            float(np.min(ranges)),
            float(np.max(ranges)),
            float(np.max(ranges) - np.min(ranges)),
            float(ranges[0]),
            float(ranges[-1]),
            float(ranges[-1] - ranges[0]),
        ]

        # Velocity stats
        feats += [
            float(np.mean(velocities)),
            float(np.std(velocities, ddof=1)),
            float(np.min(velocities)),
            float(np.max(velocities)),
            float(np.max(velocities) - np.min(velocities)),
            float(velocities[0]),
            float(velocities[-1]),
            float(velocities[-1] - velocities[0]),
        ]

        return feats

    @staticmethod
    def _trajectory_shape(frames: list[dict]) -> list[float]:
        """Compute trajectory shape descriptors."""
        centroids = np.array([f["centroid"] for f in frames])

        feats = []

        # Total path length
        if len(centroids) >= 2:
            path_length = float(np.sum(np.linalg.norm(np.diff(centroids, axis=0), axis=1)))
            feats.append(path_length)

            # Straightness: direct distance / path length
            direct_dist = float(np.linalg.norm(centroids[-1] - centroids[0]))
            straightness = direct_dist / (path_length + 1e-9)
            feats.append(straightness)

            # Curvature (2nd derivative norm)
            if len(centroids) >= 3:
                velocity = np.diff(centroids, axis=0)
                accel = np.diff(velocity, axis=0)
                curvatures = np.linalg.norm(accel, axis=1)
                feats.append(float(np.max(curvatures)))
                feats.append(float(np.mean(curvatures)))
            else:
                feats.extend([0.0, 0.0])

            # Direction changes (velocity sign flips)
            if len(centroids) >= 3:
                velocity = np.diff(centroids, axis=0)
                sign_changes = 0
                for dim in range(3):
                    signs = np.sign(velocity[:, dim])
                    sign_changes += np.sum(np.abs(np.diff(signs)) > 0)
                feats.append(float(sign_changes))
            else:
                feats.append(0.0)

            # Circularity: fit to circle in the best-fit plane
            circularity, radius, angular_disp = _fit_circle(centroids)
            feats.append(circularity)
            feats.append(radius)
            feats.append(angular_disp)
        else:
            feats.extend([0.0] * 7)

        return feats

    @staticmethod
    def _point_cloud_geometry(frames: list[dict]) -> list[float]:
        """Compute point cloud geometry features."""
        feats = []

        # Per-frame metrics
        point_counts = np.array([f.get("num_points", 0) for f in frames], dtype=np.float64)
        volumes = []
        densities = []
        anisotropies = []
        planarities = []
        sphericities = []
        scatters = []

        for f in frames:
            pts = f.get("filtered_points")
            if pts is not None and pts.shape[0] >= 3:
                ranges = np.ptp(pts, axis=0)
                vol = np.prod(ranges + 1e-9)
                volumes.append(vol)
                densities.append(pts.shape[0] / (vol + 1e-9))
            else:
                volumes.append(0.0)
                densities.append(0.0)

            anisotropies.append(f.get("anisotropy", 0.0))
            planarities.append(f.get("planarity", 0.0))
            sphericities.append(f.get("sphericity", 0.0))

            ev = f.get("eigenvalues")
            if ev is not None and len(ev) == 3:
                scatters.append(float(ev[2]))  # Largest eigenvalue
            else:
                scatters.append(0.0)

        for arr, name in [
            (point_counts, "num_points"),
            (np.array(volumes), "volume"),
            (np.array(densities), "density"),
            (np.array(anisotropies), "anisotropy"),
            (np.array(planarities), "planarity"),
            (np.array(sphericities), "sphericity"),
            (np.array(scatters), "scatter"),
        ]:
            feats.append(float(np.mean(arr)))
            feats.append(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0)

        return feats

    @staticmethod
    def _rd_heatmap_features(frames: list[dict]) -> list[float]:
        """Compute statistical features from Range-Doppler heatmaps."""
        feats = []

        # Collect per-frame heatmap stats
        stats_per_frame = {
            "mean_range": [],
            "mean_velocity": [],
            "range_spread": [],
            "velocity_spread": [],
            "active_bins": [],
            "rv_correlation": [],
        }

        for f in frames:
            hm = f.get("rd_heatmap")
            if hm is None or hm.size == 0:
                for key in stats_per_frame:
                    stats_per_frame[key].append(0.0)
                continue

            # Range axis (0) and velocity axis (1)
            # Mean range (weighted by intensity)
            range_weights = np.sum(hm, axis=1)  # Sum over velocity dimension
            if range_weights.sum() > 0:
                range_bins = np.arange(hm.shape[0])
                mean_r = float(np.average(range_bins, weights=range_weights))
                range_spread = float(np.sqrt(
                    np.average((range_bins - mean_r) ** 2, weights=range_weights)
                ))
            else:
                mean_r = 0.0
                range_spread = 0.0

            # Mean velocity
            vel_weights = np.sum(hm, axis=0)
            if vel_weights.sum() > 0:
                vel_bins = np.arange(hm.shape[1])
                mean_v = float(np.average(vel_bins, weights=vel_weights))
                vel_spread = float(np.sqrt(
                    np.average((vel_bins - mean_v) ** 2, weights=vel_weights)
                ))
            else:
                mean_v = 0.0
                vel_spread = 0.0

            # Active bins
            active = float(np.sum(hm > 0.1))

            # Range-velocity correlation
            if hm.sum() > 0:
                # Compute weighted correlation
                r_idx, v_idx = np.meshgrid(
                    np.arange(hm.shape[0]),
                    np.arange(hm.shape[1]),
                    indexing='ij',
                )
                w = hm / hm.sum()
                r_mean = np.sum(r_idx * w)
                v_mean = np.sum(v_idx * w)
                r_var = np.sum((r_idx - r_mean) ** 2 * w)
                v_var = np.sum((v_idx - v_mean) ** 2 * w)
                if r_var > 1e-9 and v_var > 1e-9:
                    cov = np.sum((r_idx - r_mean) * (v_idx - v_mean) * w)
                    rv_corr = cov / np.sqrt(r_var * v_var)
                else:
                    rv_corr = 0.0
            else:
                rv_corr = 0.0

            stats_per_frame["mean_range"].append(mean_r)
            stats_per_frame["mean_velocity"].append(mean_v)
            stats_per_frame["range_spread"].append(range_spread)
            stats_per_frame["velocity_spread"].append(vel_spread)
            stats_per_frame["active_bins"].append(active)
            stats_per_frame["rv_correlation"].append(rv_corr)

        # Aggregate: mean and std of each per-frame stat
        for key in stats_per_frame:
            arr = np.array(stats_per_frame[key])
            feats.append(float(np.mean(arr)))
            feats.append(float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0)

        return feats

    @staticmethod
    def _micro_doppler_features(frames: list[dict]) -> list[float]:
        """Compute micro-Doppler spectrogram features.

        Uses velocity centroid series as proxy for micro-Doppler signal.
        """
        feats = []
        velocities = np.array([f.get("velocity_centroid", 0.0) for f in frames])

        if len(velocities) < 8:
            return [0.0] * 10

        # Compute STFT manually
        n = len(velocities)
        yf = np.abs(fft(velocities * np.hanning(n)))
        freqs = fftfreq(n, 1.0 / 20.0)  # Assuming ~20 Hz radar frame rate
        pos_mask = freqs >= 0
        freqs_pos = freqs[pos_mask]
        yf_pos = yf[pos_mask]

        if len(yf_pos) < 4:
            return [0.0] * 10

        total_energy = np.sum(yf_pos ** 2) + 1e-12
        psd = yf_pos ** 2 / total_energy

        # Mean spectral power
        feats.append(float(np.mean(psd)))

        # Peak spectral frequency
        peak_idx = int(np.argmax(yf_pos[1:])) + 1
        feats.append(float(freqs_pos[peak_idx]))

        # Spectral bandwidth
        centroid = np.sum(freqs_pos * psd)
        bandwidth = np.sqrt(np.sum((freqs_pos - centroid) ** 2 * psd))
        feats.append(float(bandwidth))

        # Spectral centroid over time (just the centroid)
        feats.append(float(centroid))

        # Energy in frequency bands
        bands_hz = [(0, 2), (2, 5), (5, 10), (10, 50)]
        for low, high in bands_hz:
            mask = (freqs_pos >= low) & (freqs_pos < high)
            band_energy = np.sum(yf_pos[mask] ** 2) / total_energy
            feats.append(float(band_energy))

        # Ratio high-freq to low-freq
        low_energy = np.sum(yf_pos[(freqs_pos >= 0) & (freqs_pos < 5)] ** 2)
        high_energy = np.sum(yf_pos[(freqs_pos >= 5) & (freqs_pos < 50)] ** 2)
        feats.append(float(high_energy / (low_energy + 1e-9)))

        # Spectral entropy
        psd_pos = psd[psd > 0]
        entropy = -np.sum(psd_pos * np.log(psd_pos)) if len(psd_pos) > 0 else 0.0
        feats.append(float(entropy))

        # Temporal variance of spectral centroid (using half-window splits)
        if len(velocities) >= 16:
            mid = len(velocities) // 2
            v1 = velocities[:mid]
            v2 = velocities[mid:]
            yf1 = np.abs(fft(v1 * np.hanning(len(v1))))
            yf2 = np.abs(fft(v2 * np.hanning(len(v2))))
            psd1 = yf1[:len(yf1)//2] ** 2 / (np.sum(yf1[:len(yf1)//2] ** 2) + 1e-12)
            psd2 = yf2[:len(yf2)//2] ** 2 / (np.sum(yf2[:len(yf2)//2] ** 2) + 1e-12)
            f1 = fftfreq(len(v1), 1.0/20.0)[:len(yf1)//2]
            f2 = fftfreq(len(v2), 1.0/20.0)[:len(yf2)//2]
            c1 = np.sum(f1 * psd1)
            c2 = np.sum(f2 * psd2)
            feats.append(float(abs(c2 - c1)))
        else:
            feats.append(0.0)

        return feats

    @staticmethod
    def _velocity_profile_features(frames: list[dict]) -> list[float]:
        """Compute velocity profile features."""
        feats = []
        velocities = np.array([f.get("velocity_centroid", 0.0) for f in frames])

        if len(velocities) < 3:
            return [0.0] * 6

        # Zero-crossing rate
        zcr = float(np.sum(np.abs(np.diff(np.sign(velocities))) > 0)) / (len(velocities) - 1)
        feats.append(zcr)

        # Mean absolute velocity
        feats.append(float(np.mean(np.abs(velocities))))

        # Max velocity
        feats.append(float(np.max(np.abs(velocities))))

        # Velocity at midpoint
        mid = len(velocities) // 2
        feats.append(float(velocities[mid]))

        # Velocity asymmetry
        pos_max = float(np.max(velocities))
        neg_max = float(abs(np.min(velocities)))
        feats.append(pos_max / (neg_max + 1e-9))

        # Velocity kurtosis
        try:
            feats.append(float(sp_stats.kurtosis(velocities, fisher=True)))
        except (ValueError, RuntimeWarning):
            feats.append(0.0)

        return feats

    def _build_feature_names(self) -> None:
        """Build list of feature names."""
        names: list[str] = []

        # Trajectory stats
        for dim in ["x", "y", "z"]:
            for stat in ["mean", "std", "min", "max", "range", "start", "end", "delta"]:
                names.append(f"mmw_traj_{dim}_{stat}")
        for stat in ["mean", "std", "min", "max", "range", "start", "end", "delta"]:
            names.append(f"mmw_traj_range_{stat}")
        for stat in ["mean", "std", "min", "max", "range", "start", "end", "delta"]:
            names.append(f"mmw_traj_vel_{stat}")

        # Trajectory shape
        names += [
            "mmw_path_length", "mmw_straightness",
            "mmw_max_curvature", "mmw_mean_curvature",
            "mmw_direction_changes",
            "mmw_circularity", "mmw_circle_radius", "mmw_angular_disp",
        ]

        # Point cloud geometry
        for metric in ["num_points", "volume", "density", "anisotropy",
                       "planarity", "sphericity", "scatter"]:
            names.append(f"mmw_geom_{metric}_mean")
            names.append(f"mmw_geom_{metric}_std")

        # R-D heatmap
        for stat in ["mean_range", "mean_velocity", "range_spread",
                     "velocity_spread", "active_bins", "rv_correlation"]:
            names.append(f"mmw_rd_{stat}_mean")
            names.append(f"mmw_rd_{stat}_std")

        # Micro-Doppler
        names += [
            "mmw_ud_mean_power", "mmw_ud_peak_freq",
            "mmw_ud_bandwidth", "mmw_ud_centroid",
            "mmw_ud_energy_0_2hz", "mmw_ud_energy_2_5hz",
            "mmw_ud_energy_5_10hz", "mmw_ud_energy_10_50hz",
            "mmw_ud_high_low_ratio", "mmw_ud_entropy",
            "mmw_ud_centroid_variance",
        ]

        # Velocity profile
        names += [
            "mmw_vel_zcr", "mmw_vel_mean_abs",
            "mmw_vel_max", "mmw_vel_midpoint",
            "mmw_vel_asymmetry", "mmw_vel_kurtosis",
        ]

        self._feature_names = names

    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    @property
    def n_features(self) -> int:
        return len(self._feature_names)


# ── Helpers ──────────────────────────────────────────────────────────────

def _fit_circle(centroids: np.ndarray) -> tuple[float, float, float]:
    """Fit a circle to 2D projected centroids and return quality metrics.

    Projects 3D trajectory onto its best-fit plane, then fits a circle.

    Returns:
        (circularity_score, radius, angular_displacement_degrees)
        circularity_score: 1.0 = perfect circle, 0.0 = not circular.
    """
    n = len(centroids)
    if n < 4:
        return 0.0, 0.0, 0.0

    # Project to 2D using PCA
    centered = centroids - np.mean(centroids, axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Use the two largest eigenvectors for projection
    proj_2d = centered @ eigenvectors[:, 1:]  # (N, 2)

    # Fit circle using algebraic method (Taubin or simple)
    x, y = proj_2d[:, 0], proj_2d[:, 1]

    # Solve: (x^2 + y^2) = a*x + b*y + c  via least squares
    A = np.column_stack([x, y, np.ones(n)])
    b = x ** 2 + y ** 2
    try:
        params, residuals, rank, singular = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0, 0.0, 0.0

    a, b_coeff, c = params
    center_x = a / 2.0
    center_y = b_coeff / 2.0
    radius = np.sqrt(c + center_x ** 2 + center_y ** 2)

    if radius < 1e-6:
        return 0.0, 0.0, 0.0

    # Goodness of fit: residual RMS / radius
    distances = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    fit_error = np.std(distances - radius) / (radius + 1e-9)
    circularity = float(np.clip(1.0 - fit_error * 5.0, 0.0, 1.0))

    # Angular displacement
    angles = np.arctan2(y - center_y, x - center_x)
    angle_diff = np.diff(np.unwrap(angles))
    total_angular = float(np.sum(np.abs(angle_diff)))
    angular_deg = float(np.degrees(total_angular))

    return circularity, float(radius), angular_deg
