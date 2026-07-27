"""mmWave feature extraction: spatial, cluster, and micro-Doppler features."""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from src.sensor.preprocessor import ClusterInfo


class MMWaveWindow:
    """Accumulates mmWave frames and computes spatial + micro-Doppler features."""

    def __init__(self, window_frames: int = 10) -> None:
        """window_frames: number of mmWave frames in window (10 ≈ 200ms at 50Hz)."""
        self._window_frames = window_frames
        self._centroids: deque[np.ndarray] = deque(maxlen=window_frames)
        self._spreads: deque[np.ndarray] = deque(maxlen=window_frames)
        self._point_counts: deque[int] = deque(maxlen=window_frames)
        self._cluster_counts: deque[int] = deque(maxlen=window_frames)
        self._velocities: deque[np.ndarray] = deque(maxlen=window_frames)

    def push(self, clusters: list[ClusterInfo], velocities: np.ndarray | None) -> None:
        """Push a frame's cluster data.

        If multiple clusters, uses the largest (by point count).
        """
        if clusters:
            largest = max(clusters, key=lambda c: c.point_count)
            self._centroids.append(largest.centroid)
            self._spreads.append(largest.spread)
            self._point_counts.append(largest.point_count)
            self._cluster_counts.append(len(clusters))
        else:
            self._centroids.append(np.zeros(3))
            self._spreads.append(np.zeros(3))
            self._point_counts.append(0)
            self._cluster_counts.append(0)

        if velocities is not None and len(velocities) > 0:
            self._velocities.append(velocities)
        else:
            self._velocities.append(np.zeros(0))

    @property
    def full(self) -> bool:
        return len(self._centroids) >= self._window_frames

    def spatial_features(self) -> dict[str, float]:
        """Compute spatial features: centroid position, velocity, spread."""
        feats: dict[str, float] = {}

        if len(self._centroids) < 2:
            return feats

        centroids = np.array(list(self._centroids))
        spreads = np.array(list(self._spreads))

        # Centroid features (on largest cluster)
        for i, axis in enumerate(["x", "y", "z"]):
            feats[f"centroid_{axis}_mean"] = float(np.mean(centroids[:, i]))
            feats[f"centroid_{axis}_std"] = float(np.std(centroids[:, i]))

        # Centroid velocity (frame-to-frame)
        if len(centroids) >= 2:
            centroid_vel = np.diff(centroids, axis=0)
            feats["centroid_vel_mag_mean"] = float(np.mean(np.linalg.norm(centroid_vel, axis=1)))
            feats["centroid_vel_y_mean"] = float(np.mean(centroid_vel[:, 1]))  # vertical
            feats["centroid_vel_x_mean"] = float(np.mean(centroid_vel[:, 0]))  # horizontal

        # Spread features
        for i, axis in enumerate(["x", "y", "z"]):
            feats[f"spread_{axis}_mean"] = float(np.mean(spreads[:, i]))
            feats[f"spread_{axis}_max"] = float(np.max(spreads[:, i]))

        # Point count features
        counts = np.array(list(self._point_counts), dtype=np.float64)
        feats["point_count_mean"] = float(np.mean(counts))
        feats["point_count_std"] = float(np.std(counts))
        feats["point_count_range"] = float(np.max(counts) - np.min(counts))

        # Cluster count features
        clust_counts = np.array(list(self._cluster_counts), dtype=np.float64)
        feats["cluster_count_mean"] = float(np.mean(clust_counts))
        feats["cluster_count_variance"] = float(np.var(clust_counts))

        # Cluster merge detection (for clapping: two blobs → one)
        if len(clust_counts) >= 2:
            merges = np.sum(np.diff(clust_counts) < 0)
            feats["cluster_merge_events"] = float(merges)

        return feats

    def micro_doppler_features(self) -> dict[str, float]:
        """Compute micro-Doppler features from velocity channel.

        Key for Soli detection: small, periodic velocity oscillations
        in a tight spatial region.
        """
        feats: dict[str, float] = {}

        if len(self._velocities) < 2:
            return feats

        vel_list = [v for v in self._velocities if len(v) > 0]
        if not vel_list:
            return feats
        all_vels = np.concatenate(vel_list)

        if len(all_vels) < 5:
            return feats

        # Velocity statistics
        feats["doppler_vel_std"] = float(np.std(all_vels))
        feats["doppler_vel_mean"] = float(np.mean(np.abs(all_vels)))
        feats["doppler_vel_max"] = float(np.max(np.abs(all_vels)))

        # Velocity time series smoothing
        vel_timeseries: list[float] = []
        for v in self._velocities:
            if len(v) > 0:
                vel_timeseries.append(float(np.std(v)))
            else:
                vel_timeseries.append(0.0)

        if len(vel_timeseries) >= 4:
            vel_ts = np.array(vel_timeseries, dtype=np.float64)
            fft = np.abs(np.fft.rfft(vel_ts))
            freqs = np.fft.rfftfreq(len(vel_ts), d=0.02)  # ~50 Hz frame rate

            if len(fft) > 1:
                dominant_idx = np.argmax(fft[1:]) + 1
                feats["doppler_dom_freq"] = float(freqs[dominant_idx])
                feats["doppler_dom_energy"] = float(fft[dominant_idx])

                # Spectral flatness (Wiener entropy)
                # Low flatness = tonal (Soli), high flatness = noise
                if np.sum(fft) > 0:
                    geometric_mean = math.exp(np.mean(np.log(fft + 1e-10)))
                    arithmetic_mean = np.mean(fft)
                    feats["doppler_spectral_flatness"] = float(
                        geometric_mean / (arithmetic_mean + 1e-10)
                    )

        # Point count stability (Soli has stable point count)
        counts = np.array(list(self._point_counts), dtype=np.float64)
        if np.mean(counts) > 0:
            feats["point_count_stability"] = float(1.0 - np.std(counts) / (np.mean(counts) + 1e-10))

        # Spatial spread (Soli has tight cluster)
        spreads = np.array(list(self._spreads)) if len(self._spreads) > 0 else np.zeros((0, 3))
        if len(spreads) > 0:
            feats["doppler_spatial_spread"] = float(np.mean(np.linalg.norm(spreads, axis=1)))

        return feats
