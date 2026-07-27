"""UWB and RFID feature extraction: distance dynamics and tag identity."""

from __future__ import annotations

from collections import deque

import numpy as np


class UWBWindow:
    """Accumulates UWB distance/velocity samples and computes proximity features."""

    def __init__(self, window_samples: int = 50) -> None:
        self._window_samples = window_samples
        self._distances: deque[float] = deque(maxlen=window_samples)
        self._velocities: deque[float] = deque(maxlen=window_samples)

    def push(self, distance: float | None, velocity: float | None) -> None:
        if distance is not None:
            self._distances.append(distance)
        if velocity is not None:
            self._velocities.append(velocity)

    @property
    def full(self) -> bool:
        return len(self._distances) >= max(3, self._window_samples // 2)

    def features(self) -> dict[str, float]:
        """Compute proximity features for Pull/Push detection."""
        feats: dict[str, float] = {}

        if len(self._distances) < 2:
            return feats

        dists = np.array(list(self._distances), dtype=np.float64)

        feats["uwb_distance_mean"] = float(np.mean(dists))
        feats["uwb_distance_std"] = float(np.std(dists))
        feats["uwb_distance_min"] = float(np.min(dists))
        feats["uwb_distance_max"] = float(np.max(dists))
        feats["uwb_distance_range"] = feats["uwb_distance_max"] - feats["uwb_distance_min"]

        # Distance trend (linear regression slope)
        x = np.arange(len(dists), dtype=np.float64)
        slope = np.polyfit(x, dists, 1)[0]
        feats["uwb_distance_slope"] = float(slope)

        if len(self._velocities) >= 2:
            vels = np.array(list(self._velocities), dtype=np.float64)
            feats["uwb_velocity_mean"] = float(np.mean(vels))
            feats["uwb_velocity_std"] = float(np.std(vels))
            # Positive = moving away (push), negative = moving toward (pull)
            feats["uwb_velocity_sign_consistency"] = float(
                np.abs(np.sum(np.signbit(vels)) - len(vels) / 2) / (len(vels) / 2)
            )

        return feats


class RFIDTracker:
    """Tracks RFID tag presence and RSSI over time for hand identity."""

    def __init__(self, tag_map: dict[str, str] | None = None) -> None:
        self._tag_map = tag_map or {}
        self._last_seen: dict[str, float] = {}  # epc -> timestamp
        self._latest_rssi: dict[str, float] = {}  # epc -> rssi

    def update(self, tags: dict[str, float] | None, timestamp: float) -> None:
        """Update tracker with latest tag readings."""
        if tags is None:
            return
        for epc, rssi in tags.items():
            self._last_seen[epc] = timestamp
            self._latest_rssi[epc] = rssi

    def active_tags(self, max_age: float = 5.0) -> dict[str, float]:
        """Return tags seen within max_age seconds. {epc: rssi}."""
        now = max(self._last_seen.values()) if self._last_seen else 0.0
        return {
            epc: self._latest_rssi[epc]
            for epc, t in self._last_seen.items()
            if now - t < max_age
        }

    def active_hands(self, max_age: float = 2.0) -> set[str]:
        """Return set of hand identities currently detected. e.g. {"left", "right"}."""
        now = max(self._last_seen.values()) if self._last_seen else 0.0
        hands: set[str] = set()
        for epc, t in self._last_seen.items():
            if now - t < max_age:
                hand = self._tag_map.get(epc)
                if hand:
                    hands.add(hand)
        return hands

    def rssi_to_hand(self, hand: str) -> float | None:
        """Get RSSI for a specific hand identity. Returns None if not seen."""
        for epc, h in self._tag_map.items():
            if h == hand and epc in self._latest_rssi:
                return self._latest_rssi[epc]
        return None
