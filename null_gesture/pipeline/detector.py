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


# ══════════════════════════════════════════════════════════════════════
# mmWave detector
# ══════════════════════════════════════════════════════════════════════

class MMWaveDetector:
    """Real-time mmWave gesture detector.

    mmWave gives absolute 3D hand position — no onset/offset needed.
    Instead: buffer positions, detect gesture by speed threshold,
    extract geometric features, classify.
    """

    def __init__(
        self,
        speed_onset: float = 0.15,     # m/s — speed above this = gesturing
        speed_offset: float = 0.05,    # m/s — speed below this = still
        offset_frames: int = 8,
        display_hold: float = 1.5,
    ):
        self.speed_onset = speed_onset
        self.speed_offset = speed_offset
        self.offset_frames = offset_frames
        self.display_hold = display_hold

        self._classifier = GestureClassifier()

        self._buffer: list[np.ndarray] = []   # (3,) position samples
        self._state = "still"
        self._still_count = 0
        self._result = ("standing_still", 1.0)
        self._classify_time = 0.0

    def load_model(self, path: str | Path = DEFAULT_MODEL) -> bool:
        path = Path(path)
        if not path.exists():
            return False
        self._classifier.load(path)
        return True

    def fit_from_recordings(self, data_dir: str | Path = "data/raw") -> None:
        from null_gesture.pipeline.features import extract_mmwave_features

        data_dir = Path(data_dir)
        feature_list, label_list = [], []

        for fp in sorted(data_dir.glob("*.npy")):
            gesture = fp.stem.rsplit("_", 1)[0]
            traj = np.load(fp).astype(np.float32)
            if traj.ndim == 2 and traj.shape[1] == 3 and traj.shape[0] > 10:
                feats = extract_mmwave_features(traj)
                feature_list.append(feats)
                label_list.append(gesture)

        if not feature_list:
            raise RuntimeError(f"No valid recordings in {data_dir}")

        features = np.array(feature_list, dtype=np.float64)
        labels = np.array(label_list)
        self._classifier.fit(features, labels)

    def feed(self, position: np.ndarray | None) -> tuple[str, float]:
        """Feed one hand position (3,) [x, y, z] or None if no detection."""
        now = time.time()

        if self._state == "classified":
            if now - self._classify_time > self.display_hold:
                self._state = "still"
                self._result = ("standing_still", 1.0)
            return self._result

        if position is None:
            return self._result

        self._buffer.append(position)

        # Compute speed from recent positions
        if len(self._buffer) >= 3:
            recent = np.array(self._buffer[-3:])
            speed = float(np.linalg.norm(recent[-1] - recent[0])) / 0.1  # rough m/s
        else:
            speed = 0.0

        if self._state == "still":
            if speed > self.speed_onset:
                self._state = "collecting"
                self._still_count = 0

        elif self._state == "collecting":
            if speed < self.speed_offset:
                self._still_count += 1
                if self._still_count >= self.offset_frames:
                    self._classify()
                    self._state = "classified"
                    self._classify_time = now
            else:
                self._still_count = 0

            if len(self._buffer) > 200:  # safety cap
                self._classify()
                self._state = "classified"
                self._classify_time = now

        return self._result

    def _classify(self) -> None:
        if len(self._buffer) < 8 or not self._classifier.fitted:
            self._result = ("standing_still", 1.0)
            self._buffer.clear()
            return

        from null_gesture.pipeline.features import extract_mmwave_features

        traj = np.array(self._buffer, dtype=np.float32)
        self._buffer.clear()

        feats = extract_mmwave_features(traj)
        label, conf = self._classifier.predict(feats)
        self._result = (label, conf)

    @property
    def state(self) -> str:
        if self._state == "classified":
            return "classified"
        return self._state

    @property
    def gestures(self) -> list[str]:
        return self._classifier.gestures
