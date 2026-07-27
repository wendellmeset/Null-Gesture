"""Posture Gesture Detector: mmWave-driven classification of body configurations.

Covers: T-Arms, Raise Arms, Clapping.

Uses spatial features from mmWave point cloud clustering. RFID provides
hand identity confirmation.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from src.features.mmwave_features import MMWaveWindow
from src.features.uwb_features import RFIDTracker


class PostureGestureDetector:
    """Detects postural gestures: T-Arms, Raise Arms, Clapping."""

    GESTURES: ClassVar[list[str]] = ["t_arms", "raise_arms", "clapping"]

    def __init__(
        self,
        window_frames: int = 10,
        rfid_tracker: RFIDTracker | None = None,
    ) -> None:
        self._window = MMWaveWindow(window_frames)
        self._rfid = rfid_tracker or RFIDTracker()

        # Decision thresholds (tunable)
        self._t_arms_spread_x_threshold = 0.25  # meters
        self._raise_arms_vel_y_threshold = 0.15  # m/s upward
        self._clapping_cluster_variance_threshold = 0.6

    def push(
        self,
        clusters: list,
        velocities: np.ndarray | None,
        rfid_hands: set[str] | None,
    ) -> None:
        self._window.push(clusters, velocities)

    def detect(self) -> dict[str, float]:
        """Return belief masses for posture gestures."""
        result: dict[str, float] = {g: 0.0 for g in self.GESTURES}
        result["unknown"] = 1.0

        if not self._window.full:
            return result

        spatial = self._window.spatial_features()
        if not spatial:
            return result

        # ── T-Arms: wide horizontal spread ──────────────────────────
        spread_x = spatial.get("spread_x_mean", 0.0)
        _spread_y = spatial.get("spread_y_mean", 0.0)
        centroid_y = spatial.get("centroid_y_mean", 0.0)

        # T-Arms: arms out horizontally → large x-spread, moderate y-spread
        t_arms_score = 0.0
        if spread_x > self._t_arms_spread_x_threshold:
            # Map spread to confidence (sigmoid-like)
            t_arms_score = min(1.0, (spread_x - self._t_arms_spread_x_threshold) / 0.3)
            # T-Arms should have centroid at shoulder height, not raised
            if centroid_y > 0.3:  # hands above shoulders → could be Raise Arms
                t_arms_score *= 0.5

        # ── Raise Arms: centroid moving upward ──────────────────────
        centroid_vel_y = spatial.get("centroid_vel_y_mean", 0.0)

        raise_arms_score = 0.0
        if centroid_vel_y > self._raise_arms_vel_y_threshold:
            raise_arms_score = min(1.0, (centroid_vel_y - self._raise_arms_vel_y_threshold) / 0.3)

        # High centroid position also increases confidence
        if centroid_y > 0.3 and raise_arms_score < 0.5:
            raise_arms_score = max(raise_arms_score, min(1.0, (centroid_y - 0.3) / 0.3))

        # ── Clapping: cluster merging ───────────────────────────────
        cluster_var = spatial.get("cluster_count_variance", 0.0)
        cluster_mean = spatial.get("cluster_count_mean", 0.0)
        merge_events = spatial.get("cluster_merge_events", 0.0)

        clapping_score = 0.0
        if cluster_var > self._clapping_cluster_variance_threshold and merge_events > 0:
            clapping_score = min(1.0, merge_events / 3.0)
        # Even without explicit merge: two clusters that vary in count
        elif cluster_var > 0.3 and 1.0 <= cluster_mean <= 2.5:
            clapping_score = min(0.5, cluster_var * 0.8)

        # ── Assemble belief masses ──────────────────────────────────
        scores = {
            "t_arms": t_arms_score,
            "raise_arms": raise_arms_score,
            "clapping": clapping_score,
        }

        total = sum(scores.values())
        if total > 0:
            # Scale to 0.85 max, remainder → unknown
            scale = 0.85 / total
            for g in self.GESTURES:
                result[g] = scores[g] * scale
            result["unknown"] = 1.0 - sum(result[g] for g in self.GESTURES)
        else:
            result["unknown"] = 1.0

        return result
