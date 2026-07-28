"""Motion primitive detection system.

Decomposes complex gestures into atomic motion primitives that are
physically meaningful and person-invariant. This is the key enabler for
few-shot calibration — primitives are detected by rule-based thresholds
on physics-informed features, not learned from data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class MotionPrimitive(Enum):
    """Atomic motion primitives that compose into gestures."""

    TRANSLATE_X_PLUS = "TRANSLATE_X_PLUS"      # Rightward motion
    TRANSLATE_X_MINUS = "TRANSLATE_X_MINUS"    # Leftward motion
    TRANSLATE_Y_PLUS = "TRANSLATE_Y_PLUS"      # Forward / push
    TRANSLATE_Y_MINUS = "TRANSLATE_Y_MINUS"    # Backward / pull
    TRANSLATE_Z_PLUS = "TRANSLATE_Z_PLUS"      # Upward motion
    TRANSLATE_Z_MINUS = "TRANSLATE_Z_MINUS"    # Downward motion
    ROTATE_CW = "ROTATE_CW"                    # Clockwise rotation
    ROTATE_CCW = "ROTATE_CCW"                  # Anti-clockwise rotation
    OSCILLATE_FAST = "OSCILLATE_FAST"          # 3-10 Hz oscillation
    OSCILLATE_SLOW = "OSCILLATE_SLOW"          # 0.5-3 Hz oscillation
    IMPULSE = "IMPULSE"                        # Sudden energy burst
    HOLD = "HOLD"                              # Static pose


@dataclass
class PrimitiveDetection:
    """Result of a primitive detection."""
    primitive: MotionPrimitive
    confidence: float
    evidence: dict[str, float] = field(default_factory=dict)


class PrimitiveDetector:
    """Detects motion primitives from extracted feature vectors.

    Detection is rule-based: each primitive has a set of feature
    conditions (feature > threshold, feature < threshold) derived
    from physics, not learned from data. This makes detection
    person-invariant by design.

    The thresholds can be scaled during calibration to adapt to
    different arm lengths, movement speeds, etc.

    Usage::

        detector = PrimitiveDetector(gestures_config)
        detections = detector.detect(imu_features, mmwave_features)
        for d in detections:
            print(f"{d.primitive.value}: {d.confidence:.2f}")
    """

    def __init__(
        self,
        gestures_config: dict | None = None,
        scaling_factors: dict[str, float] | None = None,
    ):
        """
        Args:
            gestures_config: Parsed gestures YAML config (from config/gestures.yaml).
                             If None, uses built-in defaults.
            scaling_factors: Person-specific scaling factors for thresholds.
                             Keys like 'arm_speed', 'arm_length', 'rotation_speed'.
                             Updated during calibration.
        """
        self._gestures_config = gestures_config or {}
        self._scaling_factors = scaling_factors or {
            "arm_speed": 1.0,
            "arm_length": 1.0,
            "rotation_speed": 1.0,
            "vibration_sensitivity": 1.0,
            "jerk_sensitivity": 1.0,
        }

        # Primitive definitions from config
        self._primitive_defs = self._gestures_config.get("primitives", {})

        # Feature index maps (set by set_feature_names)
        self._imu_feature_index: dict[str, int] = {}
        self._mmwave_feature_index: dict[str, int] = {}

    def set_feature_names(
        self,
        imu_names: list[str],
        mmwave_names: list[str],
    ) -> None:
        """Register feature name → index mappings.

        Must be called before detect() so the detector knows which
        feature indices correspond to which feature names.
        """
        self._imu_feature_index = {name: i for i, name in enumerate(imu_names)}
        self._mmwave_feature_index = {name: i for i, name in enumerate(mmwave_names)}

    def detect(
        self,
        imu_features: np.ndarray | None = None,
        mmwave_features: np.ndarray | None = None,
    ) -> list[PrimitiveDetection]:
        """Detect active primitives from feature vectors.

        Args:
            imu_features: IMU feature vector (1D) or None.
            mmwave_features: mmWave feature vector (1D) or None.

        Returns:
            List of detected primitives with confidence scores.
        """
        detections: list[PrimitiveDetection] = []

        for primitive in MotionPrimitive:
            result = self._check_primitive(
                primitive, imu_features, mmwave_features
            )
            if result is not None and result.confidence > 0.0:
                detections.append(result)

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def _check_primitive(
        self,
        primitive: MotionPrimitive,
        imu_features: np.ndarray | None,
        mmwave_features: np.ndarray | None,
    ) -> PrimitiveDetection | None:
        """Check if a primitive is active based on feature conditions."""
        key = primitive.value
        defn = self._primitive_defs.get(key, {})
        if not defn:
            return None

        imu_conditions = defn.get("imu_features", [])
        mmwave_conditions = defn.get("mmwave_features", [])

        evidence: dict[str, float] = {}
        total_weight = 0.0
        satisfied_weight = 0.0

        # Check IMU conditions
        for cond in imu_conditions:
            feat_name = cond.get("feature")
            direction = cond.get("direction")  # 'gt' or 'lt'
            threshold_rel = cond.get("threshold_rel", 0.0)

            feat_val = self._get_imu_feature(feat_name, imu_features)
            if feat_val is None:
                continue

            # Apply scaling factor
            scale = self._get_scale_for_feature(feat_name)
            threshold = threshold_rel * scale

            # Check condition
            if direction == "gt":
                satisfied = feat_val > threshold
                # Confidence proportional to how far past threshold
                conf_contrib = max(0.0, min(1.0, (feat_val - threshold) / (abs(threshold) + 1e-9)))
            elif direction == "lt":
                satisfied = feat_val < threshold
                conf_contrib = max(0.0, min(1.0, (threshold - feat_val) / (abs(threshold) + 1e-9)))
            else:
                satisfied = False
                conf_contrib = 0.0

            evidence[f"imu.{feat_name}"] = feat_val
            total_weight += 1.0
            if satisfied:
                satisfied_weight += conf_contrib

        # Check mmWave conditions
        for cond in mmwave_conditions:
            feat_name = cond.get("feature")
            direction = cond.get("direction")
            threshold_rel = cond.get("threshold_rel", 0.0)

            feat_val = self._get_mmwave_feature(feat_name, mmwave_features)
            if feat_val is None:
                continue

            scale = self._get_scale_for_feature(feat_name)
            threshold = threshold_rel * scale

            if direction == "gt":
                satisfied = feat_val > threshold
                conf_contrib = max(0.0, min(1.0, (feat_val - threshold) / (abs(threshold) + 1e-9)))
            elif direction == "lt":
                satisfied = feat_val < threshold
                conf_contrib = max(0.0, min(1.0, (threshold - feat_val) / (abs(threshold) + 1e-9)))
            else:
                satisfied = False
                conf_contrib = 0.0

            evidence[f"mmwave.{feat_name}"] = feat_val
            total_weight += 1.0
            if satisfied:
                satisfied_weight += conf_contrib

        if total_weight == 0:
            return None

        confidence = satisfied_weight / total_weight
        return PrimitiveDetection(
            primitive=primitive,
            confidence=confidence,
            evidence=evidence,
        )

    def _get_imu_feature(
        self, name: str, features: np.ndarray | None
    ) -> float | None:
        """Get a named IMU feature value from the feature vector."""
        if features is None:
            return None
        idx = self._imu_feature_index.get(name)
        if idx is None:
            return None
        if idx >= len(features):
            return None
        return float(features[idx])

    def _get_mmwave_feature(
        self, name: str, features: np.ndarray | None
    ) -> float | None:
        """Get a named mmWave feature value from the feature vector."""
        if features is None:
            return None
        idx = self._mmwave_feature_index.get(name)
        if idx is None:
            return None
        if idx >= len(features):
            return None
        return float(features[idx])

    def _get_scale_for_feature(self, feat_name: str) -> float:
        """Map feature name to appropriate scaling factor."""
        if any(k in feat_name for k in ["ax_", "ay_", "az_", "accel", "linear"]):
            return self._scaling_factors["arm_speed"]
        elif any(k in feat_name for k in ["gz_", "gyro", "yaw", "roll", "pitch"]):
            return self._scaling_factors["rotation_speed"]
        elif any(k in feat_name for k in ["jerk", "impuls"]):
            return self._scaling_factors["jerk_sensitivity"]
        elif "mmw" in feat_name and any(k in feat_name for k in ["range", "path", "displacement", "x_", "y_", "z_"]):
            return self._scaling_factors["arm_length"]
        elif "spectral" in feat_name or "energy" in feat_name or "oscillation" in feat_name:
            return self._scaling_factors["vibration_sensitivity"]
        return 1.0

    def update_scaling(self, scaling_factors: dict[str, float]) -> None:
        """Update person-specific scaling factors (from calibration)."""
        self._scaling_factors.update(scaling_factors)

    def get_dominant_primitive(
        self,
        imu_features: np.ndarray | None = None,
        mmwave_features: np.ndarray | None = None,
        min_confidence: float = 0.4,
    ) -> MotionPrimitive | None:
        """Get the single most confident primitive, or None."""
        detections = self.detect(imu_features, mmwave_features)
        if detections and detections[0].confidence >= min_confidence:
            return detections[0].primitive
        return None


# ── Gesture-to-Primitive Mapping ─────────────────────────────────────────

# Maps each gesture to its defining primitive(s)
# Used by the ensemble classifier for primitive-based classification

GESTURE_PRIMITIVE_MAP: dict[str, list[MotionPrimitive]] = {
    "pull": [MotionPrimitive.TRANSLATE_Y_MINUS],
    "push": [MotionPrimitive.TRANSLATE_Y_PLUS],
    "clockwise": [MotionPrimitive.ROTATE_CW],
    "anti_clockwise": [MotionPrimitive.ROTATE_CCW],
    "left": [MotionPrimitive.TRANSLATE_X_MINUS],
    "right": [MotionPrimitive.TRANSLATE_X_PLUS],
    "bye_bye": [MotionPrimitive.OSCILLATE_FAST, MotionPrimitive.TRANSLATE_Z_PLUS],
    "one_arm_boxing": [MotionPrimitive.IMPULSE, MotionPrimitive.TRANSLATE_Y_PLUS],
    "clapping": [MotionPrimitive.IMPULSE],
    "two_arm_boxing": [MotionPrimitive.IMPULSE, MotionPrimitive.IMPULSE],
    "t_arms": [MotionPrimitive.HOLD],
    "raise_arms": [MotionPrimitive.TRANSLATE_Z_PLUS, MotionPrimitive.HOLD],
    "soli": [MotionPrimitive.OSCILLATE_FAST],
    "opening_closing_fist": [MotionPrimitive.OSCILLATE_SLOW],
    "palm_up_down": [MotionPrimitive.ROTATE_CW, MotionPrimitive.ROTATE_CCW],
}
