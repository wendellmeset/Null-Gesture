"""mmWave radar signal processor.

Transforms raw point cloud frames into structured representations:
- Filtered point clouds with centroid/covariance
- Range-Doppler heatmaps
- Micro-Doppler spectrograms via STFT
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy.signal import spectrogram


class MMWaveProcessor:
    """Processes mmWave point cloud frames into structured representations.

    Maintains a rolling window of recent frames and computes per-frame
    statistics plus accumulated representations.

    Usage::

        proc = MMWaveProcessor(range_bins=32, velocity_bins=32)
        for points, velocities in radar.stream():
            result = proc.process(points, velocities)
            # result has: centroid, covariance, rd_heatmap, filtered_points, etc.
    """

    def __init__(
        self,
        range_bins: int = 32,
        velocity_bins: int = 32,
        max_range: float = 1.5,
        max_velocity: float = 2.0,
        min_range: float = 0.05,
        stft_window: int = 20,
        stft_overlap: int = 10,
        outlier_std_threshold: float = 3.0,
    ):
        """
        Args:
            range_bins: Range-Doppler heatmap range dimension.
            velocity_bins: Range-Doppler heatmap velocity dimension.
            max_range: Maximum range in meters for heatmap.
            max_velocity: Maximum absolute velocity in m/s for heatmap.
            min_range: Minimum range in meters (filter closer points).
            stft_window: Number of frames for micro-Doppler STFT.
            stft_overlap: STFT overlap in frames.
            outlier_std_threshold: Mahalanobis distance threshold for outlier removal.
        """
        self.range_bins = range_bins
        self.velocity_bins = velocity_bins
        self.max_range = max_range
        self.max_velocity = max_velocity
        self.min_range = min_range
        self.stft_window = stft_window
        self.stft_overlap = stft_overlap
        self.outlier_std_threshold = outlier_std_threshold

        # Bin edges
        self._range_edges = np.linspace(0, max_range, range_bins + 1)
        self._velocity_edges = np.linspace(-max_velocity, max_velocity, velocity_bins + 1)

        # Rolling history
        self._centroid_history: deque[np.ndarray] = deque(maxlen=stft_window)
        self._velocity_history: deque[float] = deque(maxlen=stft_window)
        self._point_count_history: deque[int] = deque(maxlen=stft_window)
        self._frame_count = 0

    def process(
        self, points: np.ndarray, velocities: np.ndarray
    ) -> dict | None:
        """Process one radar frame.

        Args:
            points: (N, 3) array of [x, y, z] positions in meters.
            velocities: (N,) array of Doppler velocities in m/s.

        Returns:
            Dict with processed frame data, or None if no valid points.
        """
        if points.size == 0:
            self._frame_count += 1
            return None

        # Step 1: Filter points
        filtered_pts, filtered_vels = self._filter_points(points, velocities)

        if filtered_pts.size == 0:
            self._frame_count += 1
            return None

        # Step 2: Centroid and covariance
        centroid = np.mean(filtered_pts, axis=0)
        cov = np.cov(filtered_pts.T) if filtered_pts.shape[0] > 1 else np.zeros((3, 3))
        velocity_centroid = float(np.mean(filtered_vels))

        # Weighted centroid (by absolute velocity — emphasizes moving parts)
        abs_vels = np.abs(filtered_vels)
        vel_sum = abs_vels.sum()
        if vel_sum > 1e-9:
            weighted_centroid = np.average(filtered_pts, axis=0, weights=abs_vels)
        else:
            weighted_centroid = centroid

        # Step 3: Range-Doppler heatmap
        rd_heatmap = self._build_rd_heatmap(filtered_pts, filtered_vels)

        # Step 4: Point cloud geometry
        eigenvalues, eigenvectors = np.linalg.eigh(cov) if filtered_pts.shape[0] > 2 else (
            np.zeros(3), np.eye(3)
        )
        eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative
        eigen_sum = eigenvalues.sum()
        if eigen_sum > 1e-9:
            anisotropy = (eigenvalues[2] - eigenvalues[0]) / eigen_sum
            planarity = (eigenvalues[1] - eigenvalues[0]) / eigen_sum
            sphericity = eigenvalues[0] / eigen_sum
        else:
            anisotropy = planarity = sphericity = 0.0

        # Update history
        self._centroid_history.append(centroid.copy())
        self._velocity_history.append(velocity_centroid)
        self._point_count_history.append(filtered_pts.shape[0])
        self._frame_count += 1

        result = {
            "filtered_points": filtered_pts,
            "filtered_velocities": filtered_vels,
            "centroid": centroid,
            "weighted_centroid": weighted_centroid,
            "velocity_centroid": velocity_centroid,
            "covariance": cov,
            "eigenvalues": eigenvalues,
            "anisotropy": float(anisotropy),
            "planarity": float(planarity),
            "sphericity": float(sphericity),
            "rd_heatmap": rd_heatmap,
            "num_points": filtered_pts.shape[0],
            "frame_number": self._frame_count,
        }

        return result

    def _filter_points(
        self, points: np.ndarray, velocities: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply range gate and statistical outlier rejection."""
        if points.size == 0:
            return points, velocities

        # Range gate
        ranges = np.linalg.norm(points, axis=1)
        mask = (ranges >= self.min_range) & (ranges <= self.max_range)

        if not mask.any():
            return np.zeros((0, 3)), np.zeros((0,))

        pts = points[mask]
        vels = velocities[mask]

        # Statistical outlier removal (Mahalanobis distance)
        if pts.shape[0] > 3:
            mean = np.mean(pts, axis=0)
            cov = np.cov(pts.T)
            if np.linalg.det(cov) > 1e-12:
                inv_cov = np.linalg.inv(cov)
                diffs = pts - mean
                mahal = np.sqrt(np.sum(diffs @ inv_cov * diffs, axis=1))
                inlier_mask = mahal < self.outlier_std_threshold
                pts = pts[inlier_mask]
                vels = vels[inlier_mask]

        return pts, vels

    def _build_rd_heatmap(
        self, points: np.ndarray, velocities: np.ndarray
    ) -> np.ndarray:
        """Build a 2D range-Doppler heatmap from point data."""
        heatmap = np.zeros((self.range_bins, self.velocity_bins), dtype=np.float32)
        ranges = np.linalg.norm(points, axis=1)

        for r, v in zip(ranges, velocities):
            r_bin = int(np.clip(r / self.max_range * (self.range_bins - 1), 0, self.range_bins - 1))
            v_bin = int(np.clip(
                (v + self.max_velocity) / (2 * self.max_velocity) * (self.velocity_bins - 1),
                0, self.velocity_bins - 1,
            ))
            heatmap[r_bin, v_bin] += 1.0

        # Log normalization for dynamic range compression
        heatmap = np.log1p(heatmap)

        # Normalize
        max_val = heatmap.max()
        if max_val > 0:
            heatmap /= max_val

        return heatmap

    def get_micro_doppler_spectrogram(self) -> np.ndarray | None:
        """Compute micro-Doppler spectrogram from velocity history.

        Returns:
            2D array (freq_bins x time_frames) or None if insufficient data.
        """
        if len(self._velocity_history) < self.stft_window:
            return None

        velocities = np.array(list(self._velocity_history))

        # Use scipy's spectrogram for STFT
        # With nperseg=self.stft_window and noverlap=self.stft_overlap,
        # we get multiple time frames for longer histories
        f, t, Sxx = spectrogram(
            velocities,
            fs=20.0,  # Approximate radar frame rate
            nperseg=min(self.stft_window, len(velocities)),
            noverlap=min(self.stft_overlap, self.stft_window // 2),
            scaling='density',
        )

        return Sxx

    def get_centroid_trajectory(self, window: int | None = None) -> np.ndarray:
        """Get the centroid trajectory as (N, 3) array.

        Args:
            window: Number of recent frames to include. None = all.

        Returns:
            (N, 3) array of centroid positions. Most recent last.
        """
        hist = list(self._centroid_history)
        if window is not None:
            hist = hist[-window:]
        if not hist:
            return np.zeros((0, 3))
        return np.array(hist)

    def get_velocity_series(self, window: int | None = None) -> np.ndarray:
        """Get the velocity centroid time series as (N,) array."""
        hist = list(self._velocity_history)
        if window is not None:
            hist = hist[-window:]
        return np.array(hist, dtype=np.float64)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def latest_centroid(self) -> np.ndarray | None:
        if self._centroid_history:
            return self._centroid_history[-1].copy()
        return None

    @property
    def latest_velocity(self) -> float | None:
        if self._velocity_history:
            return self._velocity_history[-1]
        return None
