"""Real-time gesture detector — segment → extract → classify.

Wires together GestureSegmenter + feature extraction + GestureClassifier.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from null_gesture.pipeline.preprocessing import GestureSegmenter, resample_window
from null_gesture.pipeline.features import extract_features
from null_gesture.pipeline.classifier import GestureClassifier

logger = logging.getLogger("null_gesture.pipeline.detector")

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "gesture_model.npz"


class GestureDetector:
    """Real-time IMU gesture detector.

    Usage:
        detector = GestureDetector()
        detector.load_model("model.npz")       # or train first
        detector.fit_from_recordings("data/raw/")

        # In a loop:
        label, conf = detector.feed(sample)     # sample = (6,) array
    """

    def __init__(
        self,
        onset_thresh: float = 20.0,
        offset_thresh: float = 10.0,
        offset_frames: int = 8,
        display_hold: float = 1.5,
    ):
        self._segmenter = GestureSegmenter(
            onset_thresh=onset_thresh,
            offset_thresh=offset_thresh,
            offset_frames=offset_frames,
        )
        self._classifier = GestureClassifier()
        self.display_hold = display_hold

        self._result: tuple[str, float] = ("standing_still", 1.0)
        self._classify_time: float = 0.0
        self._state: str = "still"  # still | classified

    # ── Model ─────────────────────────────────────────────────────

    def load_model(self, path: str | Path = DEFAULT_MODEL) -> bool:
        """Load a trained classifier model."""
        path = Path(path)
        if not path.exists():
            logger.warning("No model at %s", path)
            return False
        self._classifier.load(path)
        return True

    def fit_from_recordings(self, data_dir: str | Path = "data/raw") -> None:
        """Train classifier from recorded .npy gesture files.

        Each .npy file should be named like: gesture_001.npy
        and contain (T, 6) IMU data.
        """
        data_dir = Path(data_dir)
        feature_list, label_list = [], []

        for fp in sorted(data_dir.glob("*.npy")):
            gesture = fp.stem.rsplit("_", 1)[0]
            window = np.load(fp).astype(np.float32)
            if window.ndim == 2 and window.shape[1] >= 6 and window.shape[0] > 10:
                # Use only first 6 channels (ax,ay,az,gx,gy,gz)
                w = window[:, :6]
                # Resample to fixed length
                w = resample_window(w, target_len=100)
                feats = extract_features(w)
                feature_list.append(feats)
                label_list.append(gesture)

        if not feature_list:
            raise RuntimeError(f"No valid recordings found in {data_dir}")

        features = np.array(feature_list, dtype=np.float64)
        labels = np.array(label_list)
        self._classifier.fit(features, labels)
        logger.info("Trained on %d samples, %d gestures", len(labels), len(set(labels)))

    # ── Real-time feed ────────────────────────────────────────────

    def feed(self, sample: np.ndarray) -> tuple[str, float]:
        """Feed one IMU sample (6,). Returns (gesture_label, confidence)."""
        now = time.time()

        # If we're in "classified" hold state, check if hold expired
        if self._state == "classified":
            if now - self._classify_time > self.display_hold:
                self._state = "still"
                self._result = ("standing_still", 1.0)
            return self._result

        # Feed segmenter
        segment = self._segmenter.feed(sample)

        if segment is not None and self._classifier.fitted:
            # Gesture completed — classify
            w = resample_window(segment, target_len=100)
            feats = extract_features(w)
            label, conf = self._classifier.predict(feats)
            self._result = (label, conf)
            self._state = "classified"
            self._classify_time = now
            logger.debug("Detected: %s (%.2f)", label, conf)

        return self._result

    # ── Properties ────────────────────────────────────────────────

    @property
    def state(self) -> str:
        if self._state == "classified":
            return "classified"
        return self._segmenter.state

    @property
    def onset_thresh(self) -> float:
        return self._segmenter.onset_thresh

    @onset_thresh.setter
    def onset_thresh(self, v: float) -> None:
        self._segmenter.onset_thresh = v

    @property
    def offset_thresh(self) -> float:
        return self._segmenter.offset_thresh

    @offset_thresh.setter
    def offset_thresh(self, v: float) -> None:
        self._segmenter.offset_thresh = v

    @property
    def gestures(self) -> list[str]:
        return self._classifier.gestures
