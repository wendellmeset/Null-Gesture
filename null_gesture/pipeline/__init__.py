"""Gesture recognition pipeline — MATLAB-style IMU processing.

Modules:
  acquisition   — Record labeled gesture data
  preprocessing — Filter, segment, normalise
  features      — Statistical feature extraction
  classifier    — PCA + k-NN classification
  detector      — Real-time detector combining all of the above
"""

from null_gesture.pipeline.acquisition import record_gestures
from null_gesture.pipeline.preprocessing import (
    GestureSegmenter, lowpass_filter, resample_window, normalize,
)
from null_gesture.pipeline.features import extract_features, extract_features_batch
from null_gesture.pipeline.classifier import GestureClassifier
from null_gesture.pipeline.detector import GestureDetector

__all__ = [
    "record_gestures",
    "GestureSegmenter", "lowpass_filter", "resample_window", "normalize",
    "extract_features", "extract_features_batch",
    "GestureClassifier",
    "GestureDetector",
]
