"""Dempster-Shafer evidence theory fusion engine.

Combines belief mass functions from multiple gesture detectors using
Yager's modified combination rule (conflict → Θ, not normalization).

Frame of discernment Θ = {G0..G14} — all 15 gestures as singletons.
Only singleton and Θ masses are assigned (computationally efficient,
theoretically sufficient for this domain).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# All 15 gesture labels in canonical order
ALL_GESTURES: list[str] = [
    "pull",
    "push",
    "clockwise",
    "anti_clockwise",
    "left",
    "right",
    "bye_bye",
    "one_arm_boxing",
    "clapping",
    "two_arm_boxing",
    "t_arms",
    "raise_arms",
    "soli",
    "fist_open_close",
    "palm_up_down",
]

GESTURE_TO_IDX: dict[str, int] = {g: i for i, g in enumerate(ALL_GESTURES)}
IDX_TO_GESTURE: dict[int, str] = {i: g for i, g in enumerate(ALL_GESTURES)}


def _make_mass(labels: list[str], beliefs: dict[str, float]) -> np.ndarray:
    """Convert detector output dict into a mass vector over ALL_GESTURES.

    Mass vector is shape (16,): indices 0-14 = singleton masses, index 15 = m(Θ).

    The input beliefs dict should contain gesture names and "unknown".
    "unknown" maps to Θ.
    Unknown/unmentioned gestures get mass 0.
    """
    mass = np.zeros(16, dtype=np.float64)

    for g, b in beliefs.items():
        if b <= 0:
            continue
        if g == "unknown":
            mass[15] = b  # Θ
        elif g in GESTURE_TO_IDX:
            mass[GESTURE_TO_IDX[g]] = b

    # Ensure mass sums to ≤ 1 (clip if needed)
    total = mass.sum()
    if total > 1.0:
        mass /= total
    elif total < 1.0:
        # Redistribute remainder to Θ
        mass[15] += 1.0 - total

    return mass


def yager_combine(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    """Combine two mass functions using Yager's rule.

    Yager's rule: conflict K is added to m(Θ) instead of normalizing.
    This prevents conflict from being swept under the rug.

    For singleton-only + Θ mass functions:
      m12({ω}) = m1({ω})·m2({ω}) + m1({ω})·m2(Θ) + m1(Θ)·m2({ω})
      m12(Θ)   = m1(Θ)·m2(Θ) + K
      where K = Σ_{i≠j} m1({ωᵢ})·m2({ωⱼ})

    Args:
        m1, m2: shape (16,) mass vectors. Last element = m(Θ).

    Returns:
        Combined mass vector, shape (16,).
    """
    result = np.zeros(16, dtype=np.float64)

    # Singleton masses: ωᵢ = m1[i]·m2[i] + m1[i]·m2[Θ] + m1[Θ]·m2[i]
    for i in range(15):
        result[i] = (
            m1[i] * m2[i]
            + m1[i] * m2[15]
            + m1[15] * m2[i]
        )

    # Conflict K: all cross terms between different singletons
    # K = Σ_{i≠j} m1[i] * m2[j]
    total_m1_singletons = m1[:15].sum()
    total_m2_singletons = m2[:15].sum()
    agreement = np.dot(m1[:15], m2[:15])
    K = total_m1_singletons * total_m2_singletons - agreement

    # Θ = m1[Θ]·m2[Θ] + K
    result[15] = m1[15] * m2[15] + K

    # Normalize (should sum to 1 already, but floating-point)
    total = result.sum()
    if total > 1e-10:
        result /= total
    else:
        result[15] = 1.0  # total ignorance

    return result


class DempsterShaferFusion:
    """Combines evidence from multiple gesture detectors via DS theory.

    Usage::

        fusion = DempsterShaferFusion()
        fusion.add_evidence("motion", motion_detector.detect())
        fusion.add_evidence("posture", posture_detector.detect())
        result = fusion.fuse()
        print(result["gesture"], result["confidence"])
    """

    def __init__(
        self,
        belief_threshold: float = 0.6,
        conflict_threshold: float = 0.3,
    ) -> None:
        self._belief_threshold = belief_threshold
        self._conflict_threshold = conflict_threshold
        self._evidence: list[tuple[str, dict[str, float]]] = []

    def add_evidence(self, source: str, beliefs: dict[str, float]) -> None:
        """Add a belief mass dict from a detector.

        Args:
            source: detector name (for debugging)
            beliefs: dict with gesture names + 'unknown' → mass values
        """
        self._evidence.append((source, beliefs))

    def fuse(self) -> dict[str, Any]:
        """Combine all evidence and produce a fused decision.

        Returns:
            {
                "gesture": str | None,     # detected gesture or None
                "confidence": float,        # Bel({gesture})
                "conflict": float,          # K (0 = perfect agreement)
                "ignorance": float,         # m(Θ) after fusion
                "beliefs": dict[str, float], # Bel for all gestures
                "contributing_sensors": [str],
            }
        """
        if not self._evidence:
            return self._empty_result()

        # Convert all evidence to mass vectors
        masses: list[np.ndarray] = []
        for _source, beliefs in self._evidence:
            mass = _make_mass(ALL_GESTURES, beliefs)
            masses.append(mass)

        # Sequential combination
        fused = masses[0].copy()
        conflict_total = 0.0

        for i in range(1, len(masses)):
            # Compute conflict between current fused and next mass
            total_singletons_fused = fused[:15].sum()
            total_singletons_next = masses[i][:15].sum()
            agreement = np.dot(fused[:15], masses[i][:15])
            K = total_singletons_fused * total_singletons_next - agreement
            conflict_total += K

            fused = yager_combine(fused, masses[i])

        # Extract beliefs (for singletons, Bel = m)
        beliefs: dict[str, float] = {}
        for i in range(15):
            beliefs[IDX_TO_GESTURE[i]] = float(fused[i])

        ignorance = float(fused[15])

        # Find best gesture
        best_idx = int(np.argmax(fused[:15]))
        best_gesture = IDX_TO_GESTURE[best_idx]
        best_belief = float(fused[best_idx])

        # Decision logic
        if (
            best_belief >= self._belief_threshold
            and best_belief > ignorance
            and conflict_total < self._conflict_threshold
        ):
            detected = best_gesture
        else:
            detected = None

        # Collect contributing sensors
        contributing = [src for src, bel in self._evidence if bel.get("unknown", 1.0) < 0.9]

        self._evidence.clear()

        return {
            "gesture": detected,
            "confidence": best_belief,
            "conflict": float(conflict_total),
            "ignorance": ignorance,
            "beliefs": beliefs,
            "contributing_sensors": contributing,
        }

    def reset(self) -> None:
        self._evidence.clear()

    @staticmethod
    def _empty_result() -> dict[str, Any]:
        return {
            "gesture": None,
            "confidence": 0.0,
            "conflict": 0.0,
            "ignorance": 1.0,
            "beliefs": {g: 0.0 for g in ALL_GESTURES},
            "contributing_sensors": [],
        }
