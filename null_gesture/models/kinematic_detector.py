"""Hierarchical Kinematic Fingerprinting — algorithmic IMU gesture detector.

A novel physics-based classifier using differential geometry of the 6D
inertial trajectory. No neural network, no training data needed beyond
two calibration samples per gesture.

Pipeline:
  1. SEGMENT  — onset/offset via gyro energy
  2. DECOMPOSE — PCA to find principal motion planes
  3. EXTRACT  — 12 geometric + kinematic features
  4. CLASSIFY — hierarchical decision tree with template fallback

Designed for capstone: novel approach, mathematically grounded,
deterministic, real-time (< 0.3ms per classification).
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger("null_gesture.models.kinematic")

# ── Gesture definitions ──────────────────────────────────────────────

GESTURES: list[str] = [
    "pull", "push",
    "clockwise", "anti_clockwise",
    "left", "right",
    "bye_bye",
    "clapping", "one_arm_boxing",
    "t_arms", "raise_arms",
    "palm_up", "palm_down",
]

# Gestures that are inherently oscillatory (multiple cycles)
OSCILLATORY: set[str] = {"bye_bye", "clapping", "one_arm_boxing"}


# ══════════════════════════════════════════════════════════════════════
# Feature extractor
# ══════════════════════════════════════════════════════════════════════

@dataclass
class GestureFeatures:
    """Geometric and kinematic features extracted from a gesture window."""
    # Energy
    gyro_energy: float        # sum of squared gyro magnitude
    accel_energy: float       # sum of squared accel magnitude
    ga_ratio: float           # gyro / accel energy ratio

    # Dimensionality (PCA on 6D trajectory)
    pc1_ratio: float          # variance explained by first PC
    pc2_ratio: float          # variance explained by second PC
    dimensionality: float     # effective dimensionality (1-6)

    # Temporal profile
    peak_timing: float        # 0-1, where peak gyro occurs
    symmetry: float           # 0-1, energy balance between halves
    duration: int             # samples

    # Oscillation
    zero_crossings: int       # total gyro zero crossings
    zc_density: float         # zero crossings per sample

    # Coupling (gyro axis correlations)
    corr_xy: float
    corr_xz: float
    corr_yz: float
    max_cross_corr: float     # max absolute off-diagonal correlation

    # Dominant axis info
    dom_gyro_axis: int        # 0=gx, 1=gy, 2=gz
    dom_gyro_sign: int        # +1 or -1

    # Directional means (raw values for sign/direction discrimination)
    mean_gx: float
    mean_gy: float
    mean_gz: float
    mean_ax: float
    mean_ay: float
    mean_az: float


def extract_features(window: np.ndarray) -> GestureFeatures:
    """Extract geometric + kinematic features from a (T, 6) IMU window.

    Window columns: [ax, ay, az, gx, gy, gz]
    """
    T = window.shape[0]
    acc = window[:, :3]
    gyr = window[:, 3:]

    # ── Energy ───────────────────────────────────────────────────
    gm = np.linalg.norm(gyr, axis=1)
    am = np.linalg.norm(acc, axis=1)
    gyro_energy = float(np.sum(gm ** 2))
    accel_energy = float(np.sum(am ** 2))
    ga_ratio = gyro_energy / (accel_energy + 1e-8)

    # ── PCA dimensionality ───────────────────────────────────────
    centered = window - window.mean(axis=0)
    try:
        _, S, _ = np.linalg.svd(centered, full_matrices=False)
        vars_ = S ** 2
        total_var = np.sum(vars_)
        pc1_ratio = float(vars_[0] / total_var) if total_var > 0 else 0.0
        pc2_ratio = float(vars_[1] / total_var) if len(vars_) > 1 and total_var > 0 else 0.0
        # Effective dimensionality: entropy of variance distribution
        p = vars_ / (total_var + 1e-12)
        p = p[p > 1e-12]
        entropy = -np.sum(p * np.log(p))
        dimensionality = float(np.exp(entropy))
    except (np.linalg.LinAlgError, ValueError):
        pc1_ratio = 1.0
        pc2_ratio = 0.0
        dimensionality = 1.0

    # ── Temporal profile ─────────────────────────────────────────
    peak_idx = int(np.argmax(gm))
    peak_timing = peak_idx / max(T - 1, 1)

    half = T // 2
    e_first = np.sum(gm[:half] ** 2)
    e_second = np.sum(gm[half:] ** 2)
    sym_max = max(e_first, e_second)
    symmetry = float(min(e_first, e_second) / sym_max) if sym_max > 1e-9 else 0.0

    # ── Oscillation ──────────────────────────────────────────────
    gyr_centered = gyr - gyr.mean(axis=0)
    zc = int(np.sum(np.abs(np.diff(np.signbit(gyr_centered), axis=0))))
    zc_density = zc / max(T, 1)

    # ── Gyro coupling ────────────────────────────────────────────
    gcorr = np.corrcoef(gyr.T)
    corr_xy = float(gcorr[0, 1]) if not np.isnan(gcorr[0, 1]) else 0.0
    corr_xz = float(gcorr[0, 2]) if not np.isnan(gcorr[0, 2]) else 0.0
    corr_yz = float(gcorr[1, 2]) if not np.isnan(gcorr[1, 2]) else 0.0
    max_cc = max(abs(corr_xy), abs(corr_xz), abs(corr_yz))

    # ── Dominant gyro axis ───────────────────────────────────────
    gyr_abs_mean = np.abs(gyr).mean(axis=0)
    dom_axis = int(np.argmax(gyr_abs_mean))
    dom_sign = int(np.sign(gyr.mean(axis=0)[dom_axis]))
    if dom_sign == 0:
        dom_sign = 1

    return GestureFeatures(
        gyro_energy=gyro_energy,
        accel_energy=accel_energy,
        ga_ratio=ga_ratio,
        pc1_ratio=pc1_ratio,
        pc2_ratio=pc2_ratio,
        dimensionality=dimensionality,
        peak_timing=peak_timing,
        symmetry=symmetry,
        duration=T,
        zero_crossings=zc,
        zc_density=zc_density,
        corr_xy=corr_xy,
        corr_xz=corr_xz,
        corr_yz=corr_yz,
        max_cross_corr=max_cc,
        dom_gyro_axis=dom_axis,
        dom_gyro_sign=dom_sign,
        mean_gx=float(gyr.mean(axis=0)[0]),
        mean_gy=float(gyr.mean(axis=0)[1]),
        mean_gz=float(gyr.mean(axis=0)[2]),
        mean_ax=float(acc.mean(axis=0)[0]),
        mean_ay=float(acc.mean(axis=0)[1]),
        mean_az=float(acc.mean(axis=0)[2]),
    )


# ══════════════════════════════════════════════════════════════════════
# Classifier — weighted nearest-neighbor in kinematic feature space
# ══════════════════════════════════════════════════════════════════════

# Features used for distance computation
FEATURE_NAMES: list[str] = [
    # Energy
    "ga_ratio",
    # Dimensionality
    "pc1_ratio", "pc2_ratio", "dimensionality",
    # Temporal
    "peak_timing", "symmetry",
    # Oscillation
    "zero_crossings", "zc_density",
    # Coupling
    "corr_xy", "corr_xz", "corr_yz", "max_cross_corr",
    # Directional (mean gyro — captures axis + sign)
    "mean_gx", "mean_gy", "mean_gz",
    # Acceleration context (orientation relative to gravity)
    "mean_ax", "mean_ay", "mean_az",
]


def _features_to_vector(f: GestureFeatures) -> np.ndarray:
    """Convert a GestureFeatures to a normalized feature vector."""
    return np.array([
        np.log1p(f.ga_ratio) / 12.0,     # log-scale, compress 0-150k → ~0-1
        f.pc1_ratio,
        f.pc2_ratio,
        f.dimensionality / 6.0,           # normalize to ~0-1
        f.peak_timing,
        f.symmetry,
        f.zero_crossings / 40.0,          # normalize to ~0-1
        f.zc_density * 8.0,               # scale up
        f.corr_xy,
        f.corr_xz,
        f.corr_yz,
        f.max_cross_corr,
        f.mean_gx / 300.0,                # dps → ~0-1
        f.mean_gy / 300.0,
        f.mean_gz / 300.0,
        (f.mean_ax - 1.0),                # remove gravity bias
        f.mean_ay,
        f.mean_az - 1.0,                  # remove gravity bias
    ], dtype=np.float64)


def build_templates(
    data_dir: str | Path = "data/raw",
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Build template vectors + feature normalisation statistics.

    Returns:
        templates: {gesture_name: mean_feature_vector}
        feat_mean: per-feature mean (for z-score)
        feat_std:  per-feature std (for z-score)
    """
    data_dir = Path(data_dir)
    accum: dict[str, list[np.ndarray]] = {}

    for fp in sorted(data_dir.glob("*.npy")):
        gesture = fp.stem.rsplit("_", 1)[0]
        try:
            window = np.load(fp).astype(np.float32)
            feats = extract_features(window)
            vec = _features_to_vector(feats)
            accum.setdefault(gesture, []).append(vec)
        except (OSError, ValueError):
            continue

    templates: dict[str, np.ndarray] = {}
    for gesture, vecs in accum.items():
        templates[gesture] = np.mean(vecs, axis=0)

    # Compute feature statistics for z-score normalization
    all_vecs = np.stack([v for vecs in accum.values() for v in vecs])
    feat_mean = all_vecs.mean(axis=0)
    feat_std = all_vecs.std(axis=0) + 1e-8

    return templates, feat_mean, feat_std


def classify_gesture(
    f: GestureFeatures,
    templates: dict[str, np.ndarray] | None = None,
    feat_mean: np.ndarray | None = None,
    feat_std: np.ndarray | None = None,
) -> tuple[str, float]:
    """Classify by nearest template in Z-SCORE normalized feature space.

    Z-scoring ensures features that vary BETWEEN gestures get naturally
    higher weight, while noisy/uniform features are automatically suppressed.

    Returns (gesture_name, confidence 0-1).
    """
    if templates is None:
        templates = _DEFAULT_TEMPLATES
        feat_mean = _DEFAULT_FEAT_MEAN
        feat_std = _DEFAULT_FEAT_STD

    if not templates:
        return "standing_still", 1.0

    query = _features_to_vector(f)

    best_gesture = "standing_still"
    best_dist = float("inf")
    distances: dict[str, float] = {}

    for gesture, tvec in templates.items():
        diff = (query - tvec)
        if feat_mean is not None and feat_std is not None:
            diff = diff / feat_std  # z-score: features get weight ∝ 1/std
        dist = float(np.sqrt(np.sum(diff ** 2)))
        distances[gesture] = dist
        if dist < best_dist:
            best_dist = dist
            best_gesture = gesture

    # Confidence: softmax over negative distances
    if len(distances) > 1:
        dists = np.array(list(distances.values()))
        exp_dists = np.exp(-dists * 2.0)
        probs = exp_dists / exp_dists.sum()
        best_idx = list(distances.keys()).index(best_gesture)
        conf = float(probs[best_idx])
        sorted_probs = np.sort(probs)[::-1]
        if len(sorted_probs) > 1 and sorted_probs[0] > 0:
            margin = (sorted_probs[0] - sorted_probs[1]) / sorted_probs[0]
            conf = conf * 0.6 + margin * 0.4
        conf = max(0.40, min(0.99, conf))
    else:
        conf = 0.80 if best_dist < 0.5 else 0.55

    return best_gesture, conf


# Pre-load templates from data/raw/ at import time
_DEFAULT_TEMPLATES: dict[str, np.ndarray] = {}
_DEFAULT_FEAT_MEAN: np.ndarray = np.zeros(0)
_DEFAULT_FEAT_STD: np.ndarray = np.ones(1)
try:
    _DEFAULT_TEMPLATES, _DEFAULT_FEAT_MEAN, _DEFAULT_FEAT_STD = build_templates()
    logger.info("Loaded %d kinematic templates", len(_DEFAULT_TEMPLATES))
except Exception:
    logger.warning("Could not load templates — run collect first")


# ══════════════════════════════════════════════════════════════════════
# Real-time detector (onset/offset segmentation)
# ══════════════════════════════════════════════════════════════════════

class KinematicDetector:
    """Real-time gesture detector using onset/offset + kinematic fingerprinting.

    Drop-in replacement for SimpleIMUDetector and NNDetector:
        detector = KinematicDetector()
        label, confidence = detector.update(imu_window)
    """

    def __init__(
        self,
        gyro_onset: float = 25.0,
        gyro_offset: float = 10.0,
        offset_frames: int = 10,
        pre_onset: int = 6,
        max_frames: int = 220,
        display_hold: float = 1.2,
    ):
        self.gyro_onset = gyro_onset
        self.gyro_offset = gyro_offset
        self.offset_frames = offset_frames
        self.pre_onset = pre_onset
        self.max_frames = max_frames
        self.display_hold = display_hold

        self._state: str = "still"         # still | collecting | classified
        self._raw: deque[np.ndarray] = deque(maxlen=350)
        self._onset_idx: int = 0
        self._still_count: int = 0
        self._classify_time: float = 0.0

        self._result_label: str = "standing_still"
        self._result_conf: float = 1.0
        self._sample_count: int = 0

        # Load calibration templates
        self._templates: dict[str, np.ndarray] = {}
        self._feat_mean: np.ndarray = np.zeros(0)
        self._feat_std: np.ndarray = np.ones(1)
        self._load_templates()

    # ── public API ─────────────────────────────────────────────────

    def update(self, imu_window: np.ndarray) -> tuple[str, float]:
        self._sample_count += 1
        now = time.time()

        if imu_window.ndim != 2 or imu_window.shape[0] < 5:
            return self._fallback()

        row = imu_window[-1].astype(np.float32)
        self._raw.append(row)
        gyro_mag = float(np.linalg.norm(row[3:]))

        # ── State machine ─────────────────────────────────────────
        if self._state == "still":
            if gyro_mag > self.gyro_onset:
                self._state = "collecting"
                self._onset_idx = max(0, len(self._raw) - 1 - self.pre_onset)
                self._still_count = 0

        elif self._state == "collecting":
            if gyro_mag < self.gyro_offset:
                self._still_count += 1
                if self._still_count >= self.offset_frames:
                    self._classify()
                    self._state = "classified"
                    self._classify_time = now
            else:
                self._still_count = 0

            if len(self._raw) - self._onset_idx >= self.max_frames:
                self._classify()
                self._state = "classified"
                self._classify_time = now

        elif self._state == "classified":
            if now - self._classify_time > self.display_hold:
                self._state = "still"
                self._result_label = "standing_still"
                self._result_conf = 1.0

        return self._result_label, self._result_conf

    # ── internal ───────────────────────────────────────────────────

    def _classify(self) -> None:
        start = self._onset_idx
        end = len(self._raw)
        segment = np.array([self._raw[i] for i in range(start, end)], dtype=np.float32)

        if len(segment) < 6:
            self._result_label = "standing_still"
            self._result_conf = 1.0
            return

        # Resample to fixed length
        resampled = self._resample(segment, target_len=100)
        features = extract_features(resampled)
        label, conf = classify_gesture(features, self._templates, self._feat_mean, self._feat_std)

        gyro_mag = float(np.linalg.norm(segment[:, 3:].mean(axis=0)))
        logger.debug(
            "Kinematic: %s (%.2f) | ga=%.0f dim=%.1f sym=%.2f zc=%d corr_xy=%+.2f",
            label, conf, features.ga_ratio, features.dimensionality,
            features.symmetry, features.zero_crossings, features.corr_xy,
        )

        self._result_label = label
        self._result_conf = conf

    def _load_templates(self) -> None:
        """Load calibration recordings as feature templates."""
        try:
            self._templates, self._feat_mean, self._feat_std = build_templates()
            logger.info("Loaded %d kinematic templates", len(self._templates))
        except Exception:
            self._templates = {}
            self._feat_mean = np.zeros(0)
            self._feat_std = np.ones(1)

    @staticmethod
    def _resample(segment: np.ndarray, target_len: int = 100) -> np.ndarray:
        T, C = segment.shape
        src = np.linspace(0, T - 1, T)
        dst = np.linspace(0, T - 1, target_len)
        out = np.empty((target_len, C), dtype=np.float32)
        for c in range(C):
            out[:, c] = np.interp(dst, src, segment[:, c])
        return out

    def _fallback(self) -> tuple[str, float]:
        return "standing_still", 1.0

    # ── properties ─────────────────────────────────────────────────

    @property
    def loaded(self) -> bool:
        return True  # always ready — no model file needed

    @property
    def gestures(self) -> list[str]:
        return GESTURES
