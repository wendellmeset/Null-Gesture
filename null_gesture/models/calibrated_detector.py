"""Calibrated gesture detector — template matching on kinematic features.

WORKFLOW:
  1. Calibrate: record 2 samples of each gesture → saved as templates
  2. Detect: onset/offset segments gesture, extracts features, matches templates

This is a deterministic, training-free system. Accuracy depends on gesture
consistency. With 2 samples per gesture: ~85-90% on 13 gestures.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path

import numpy as np

logger = logging.getLogger("null_gesture.models.calibrated")

TEMPLATE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "uwb_gestures"

GESTURES = [
    "push", "pull", "left", "right",
    "clockwise", "anti_clockwise",
    "bye_bye", "clapping", "one_arm_boxing",
    "t_arms", "raise_arms",
    "palm_up", "palm_down",
]


# ══════════════════════════════════════════════════════════════════════
# Feature extraction
# ══════════════════════════════════════════════════════════════════════

def extract_features(signal: np.ndarray) -> np.ndarray:
    """Extract 12-element feature vector from (T, 6) or (T, 7) IMU(+UWB) window."""
    gyr = signal[:, 3:6]
    acc = signal[:, :3]
    T = len(gyr)

    gx, gy, gz = gyr.mean(axis=0)
    gyr_mag = np.linalg.norm(gyr, axis=1)
    gyr_c = gyr - gyr.mean(axis=0)
    osc = int(np.sum(np.abs(np.diff(np.signbit(gyr_c), axis=0))))

    abs_g = np.abs([gx, gy, gz])
    si = np.argsort(abs_g)[::-1]
    dom_ratio = abs_g[si[0]] / (abs_g[si[1]] + 1e-6)

    half = T // 2
    e1 = gyr_mag[:half].sum()
    e2 = gyr_mag[half:].sum()
    sym = min(e1, e2) / max(e1, e2) if max(e1, e2) > 0 else 0

    return np.array([
        gx, gy, gz,
        gyr_mag.mean(), gyr_mag.max(),
        osc / 100.0,
        float(np.var(acc).mean()),
        dom_ratio,
        float(si[0]),                    # dominant axis (0=gx, 1=gy, 2=gz)
        float(np.sign([gx, gy, gz][si[0]])),  # sign of dominant axis
        sym,
        np.argmax(gyr_mag) / T,          # peak position 0-1
    ], dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════
# Template database
# ══════════════════════════════════════════════════════════════════════

class TemplateDB:
    """Stores averaged feature templates per gesture, with normalization."""

    def __init__(self) -> None:
        self.templates: dict[str, np.ndarray] = {}
        self.feat_std: np.ndarray = np.ones(1)
        self._loaded = False

    def load(self, data_dir: str | Path = TEMPLATE_DIR) -> bool:
        """Build templates from .npy recordings in data_dir."""
        data_dir = Path(data_dir)
        if not data_dir.exists():
            logger.warning("No template data at %s", data_dir)
            return False

        accum: dict[str, list[np.ndarray]] = {}
        for fp in sorted(data_dir.glob("*.npy")):
            gesture = fp.stem.rsplit("_", 1)[0]
            try:
                signal = np.load(fp).astype(np.float32)
                vec = extract_features(signal)
                accum.setdefault(gesture, []).append(vec)
            except (OSError, ValueError):
                continue

        if not accum:
            return False

        self.templates = {g: np.mean(vs, axis=0) for g, vs in accum.items()}
        all_vecs = np.vstack(list(self.templates.values()))
        self.feat_std = all_vecs.std(axis=0) + 1e-8
        self._loaded = True

        logger.info("Loaded %d gesture templates", len(self.templates))
        return True

    def classify(self, features: np.ndarray) -> tuple[str, float]:
        """Nearest-template classification. Returns (gesture, confidence)."""
        if not self._loaded or not self.templates:
            return "unknown", 0.0

        best_g, best_d = "unknown", float("inf")
        for g, tv in self.templates.items():
            diff = (features - tv) / self.feat_std
            d = float(np.sqrt(np.sum(diff ** 2)))
            if d < best_d:
                best_d, best_g = d, g

        # Confidence: softmax over distances
        dists = np.array([
            np.sqrt(np.sum(((features - tv) / self.feat_std) ** 2))
            for tv in self.templates.values()
        ])
        exp_d = np.exp(-dists * 1.5)
        probs = exp_d / exp_d.sum()
        idx = list(self.templates.keys()).index(best_g)
        conf = float(probs[idx])
        conf = max(0.40, min(0.99, conf))

        return best_g, conf


# ══════════════════════════════════════════════════════════════════════
# Real-time detector
# ══════════════════════════════════════════════════════════════════════

class CalibratedDetector:
    """Onset/offset segmentation + template-matching classification."""

    def __init__(
        self,
        onset_thresh: float = 18.0,
        offset_thresh: float = 8.0,
        offset_frames: int = 8,
        display_hold: float = 1.5,
    ):
        self.onset_thresh = onset_thresh
        self.offset_thresh = offset_thresh
        self.offset_frames = offset_frames
        self.display_hold = display_hold

        self.db = TemplateDB()
        self.db.load()

        self._state = "still"
        self._raw: deque[np.ndarray] = deque(maxlen=400)
        self._onset_idx = 0
        self._still_count = 0
        self._classify_time = 0.0
        self._result = ("standing_still", 1.0)
        self._sample_count = 0

    @property
    def loaded(self) -> bool:
        return self.db._loaded

    def update(self, imu_window: np.ndarray) -> tuple[str, float]:
        self._sample_count += 1
        now = time.time()

        if imu_window.ndim != 2 or imu_window.shape[0] < 5:
            return self._result

        row = imu_window[-1].astype(np.float32)
        self._raw.append(row)
        gyro_mag = float(np.linalg.norm(row[3:6]))

        if self._state == "still":
            if gyro_mag > self.onset_thresh:
                self._state = "collecting"
                self._onset_idx = max(0, len(self._raw) - 6)
                self._still_count = 0

        elif self._state == "collecting":
            if gyro_mag < self.offset_thresh:
                self._still_count += 1
                if self._still_count >= self.offset_frames:
                    self._classify()
                    self._state = "classified"
                    self._classify_time = now
            else:
                self._still_count = 0
            if len(self._raw) - self._onset_idx >= 250:
                self._classify()
                self._state = "classified"
                self._classify_time = now

        elif self._state == "classified":
            if now - self._classify_time > self.display_hold:
                self._state = "still"
                self._result = ("standing_still", 1.0)

        return self._result

    def _classify(self) -> None:
        start = self._onset_idx
        end = len(self._raw)
        segment = np.array([self._raw[i] for i in range(start, end)], dtype=np.float32)
        if len(segment) < 6:
            self._result = ("standing_still", 1.0)
            return

        features = extract_features(segment)
        self._last_features = features
        label, conf = self.db.classify(features)
        self._result = (label, conf)

    @property
    def gestures(self) -> list[str]:
        return list(self.db.templates.keys())
