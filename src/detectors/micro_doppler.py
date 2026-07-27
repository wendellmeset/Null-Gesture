"""Micro-Doppler Gesture Detector: mmWave velocity-based Soli detection.

Detects the Google Soli gesture — rubbing thumb and index finger together —
via small, periodic velocity oscillations in the mmWave Doppler channel.

Physics: at 60 GHz, λ ≈ 5 mm. Finger movements at 0.01-0.2 m/s produce
Doppler shifts at 4-80 Hz. The IWRL6432 velocity resolution captures this.
"""

from __future__ import annotations

import math

import numpy as np

from src.features.mmwave_features import MMWaveWindow


class MicroDopplerDetector:
    """Binary detector for Soli (thumb/index finger rub) via micro-Doppler."""

    GESTURES = ["soli"]

    def __init__(
        self,
        window_frames: int = 10,  # 10 frames ≈ 200 ms at 50 Hz
    ) -> None:
        self._window = MMWaveWindow(window_frames)

        # Tuned thresholds
        self._vel_std_low = 0.005   # m/s — must have some velocity variation
        self._vel_std_high = 0.15   # m/s — but not too much (hand wave)
        self._spread_max = 0.15     # m — Soli has tight spatial cluster
        self._flatness_max = 0.5    # low flatness = tonal (Soli-like)
        self._dom_freq_low = 4.0    # Hz
        self._dom_freq_high = 60.0  # Hz
        self._point_stability_min = 0.6  # Soli has stable point count

    def push(
        self,
        clusters: list,
        velocities: np.ndarray | None,
    ) -> None:
        self._window.push(clusters, velocities)

    def detect(self) -> dict[str, float]:
        """Return belief masses: {"soli": ..., "unknown": ...}."""
        result: dict[str, float] = {"soli": 0.0, "unknown": 1.0}

        if not self._window.full:
            return result

        doppler = self._window.micro_doppler_features()
        spatial = self._window.spatial_features()

        if not doppler:
            return result

        # ── Extract discriminative features ─────────────────────────
        vel_std = doppler.get("doppler_vel_std", 999.0)
        dom_freq = doppler.get("doppler_dom_freq", 0.0)
        flatness = doppler.get("doppler_spectral_flatness", 1.0)
        spread = doppler.get("doppler_spatial_spread", 999.0)
        stability = doppler.get("point_count_stability", 0.0)

        # ── Evidence scoring ────────────────────────────────────────
        evidence: list[float] = []

        # 1. Velocity standard deviation in Soli range
        if self._vel_std_low <= vel_std <= self._vel_std_high:
            # Within range — high evidence
            optimal = (self._vel_std_low + self._vel_std_high) / 2
            evidence.append(1.0 - min(1.0, abs(vel_std - optimal) / optimal))
        elif vel_std < self._vel_std_low:
            evidence.append(0.1)  # too quiet
        else:
            evidence.append(0.1)  # too loud (hand wave)

        # 2. Dominant frequency in 4-60 Hz
        if self._dom_freq_low <= dom_freq <= self._dom_freq_high:
            f_score = 1.0
            # Penalize very low frequencies (body motion)
            if dom_freq < 8.0:
                f_score = dom_freq / 8.0
            evidence.append(f_score)
        else:
            evidence.append(0.0)

        # 3. Low spectral flatness → tonal, structured signal
        if flatness < self._flatness_max:
            evidence.append(1.0 - flatness / self._flatness_max)
        else:
            evidence.append(0.1)

        # 4. Tight spatial spread
        if spread < self._spread_max:
            evidence.append(1.0 - spread / self._spread_max)
        else:
            evidence.append(0.0)

        # 5. Stable point count
        if stability > self._point_stability_min:
            evidence.append(min(1.0, (stability - self._point_stability_min) / 0.3))
        else:
            evidence.append(0.2)

        # ── Combine evidence ────────────────────────────────────────
        # Weighted geometric mean of evidence
        weights = [0.3, 0.2, 0.2, 0.15, 0.15]  # vel_std matters most
        log_score = sum(
            w * math.log(e + 1e-10) for w, e in zip(weights, evidence)
        )
        soli_score = math.exp(log_score)

        result["soli"] = min(0.9, soli_score)
        result["unknown"] = 1.0 - result["soli"]

        return result
