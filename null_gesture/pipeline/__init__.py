"""Gesture recognition pipeline — MATLAB-style IMU/mmWave processing.

Modules:
  acquisition   — Record labeled gesture data (IMU or mmWave)
  preprocessing — Filter, segment, normalise
  features      — Statistical (IMU) and geometric (mmWave) feature extraction
  classifier    — PCA + k-NN classification
  detector      — Real-time IMU and mmWave detectors
"""

from null_gesture.pipeline.acquisition import record_gestures, record_mmwave_gestures
from null_gesture.pipeline.preprocessing import (
    GestureSegmenter, lowpass_filter, resample_window, normalize,
)
from null_gesture.pipeline.features import (
    extract_features, extract_features_batch,
    extract_mmwave_features,
    get_feature_names, get_mmwave_feature_names,
)
from null_gesture.pipeline.classifier import GestureClassifier
from null_gesture.pipeline.detector import GestureDetector, MMWaveDetector

__all__ = [
    "record_gestures", "record_mmwave_gestures",
    "GestureSegmenter", "lowpass_filter", "resample_window", "normalize",
    "extract_features", "extract_features_batch", "extract_mmwave_features",
    "get_feature_names", "get_mmwave_feature_names",
    "GestureClassifier",
    "GestureDetector", "MMWaveDetector",
]
