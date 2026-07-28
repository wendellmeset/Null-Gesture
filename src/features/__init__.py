"""Feature extraction layer: physics-informed features from sensor streams."""

from src.features.imu_features import IMUFeatureExtractor
from src.features.mmwave_features import MMWaveFeatureExtractor
from src.features.primitives import PrimitiveDetector, MotionPrimitive

__all__ = [
    "IMUFeatureExtractor",
    "MMWaveFeatureExtractor",
    "PrimitiveDetector",
    "MotionPrimitive",
]
