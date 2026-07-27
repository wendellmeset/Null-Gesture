"""Neural-network IMU gesture detector with onset/offset segmentation.

Loads a trained GestureCNN and uses onset/offset detection so the model
always sees a COMPLETE gesture — never a partial sliding window.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger("null_gesture.models.nn_detector")

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "saved" / "gesture_cnn.pt"


class NNDetector:
    """Neural gesture detector — onset/offset segmentation + EMA smoothing.

    State machine:
      still → collecting → classified → still → ...

    1. Waits for gyro spike (onset).
    2. Buffers every sample during the motion.
    3. Detects end-of-gesture (gyro drops below threshold for N frames).
    4. Resamples the buffered segment to exactly 100 timesteps.
    5. Runs the CNN on the complete gesture.
    6. Holds the result briefly, then returns to still.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        gyro_onset: float = 18.0,
        gyro_offset: float = 8.0,
        offset_frames: int = 8,
        pre_onset: int = 8,
        max_gesture_frames: int = 200,
        display_hold: float = 1.2,
        ema_alpha: float = 0.45,
        confidence_floor: float = 0.50,
        device: str = "cpu",
    ):
        """
        Args:
            model_path: path to .pt checkpoint.
            gyro_onset: gyro magnitude (dps) to trigger gesture start.
            gyro_offset: gyro magnitude below which gesture is considered done.
            offset_frames: consecutive still frames needed to end a gesture.
            pre_onset: frames before onset to include in the gesture window.
            max_gesture_frames: safety cap — force-classify if gesture exceeds this.
            display_hold: seconds to hold the classification result before resetting.
            ema_alpha: smoothing for class probabilities across gestures.
            confidence_floor: minimum confidence to report a gesture.
            device: torch device.
        """
        self.gyro_onset = gyro_onset
        self.gyro_offset = gyro_offset
        self.offset_frames = offset_frames
        self.pre_onset = pre_onset
        self.max_gesture_frames = max_gesture_frames
        self.display_hold = display_hold
        self.ema_alpha = ema_alpha
        self.confidence_floor = confidence_floor
        self.device = device

        # ── Load model ────────────────────────────────────────────
        model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self._model: torch.nn.Module | None = None
        self._gestures: list[str] = []
        self._ch_mean: np.ndarray | None = None
        self._ch_std: np.ndarray | None = None
        self._loaded = False

        if model_path.exists():
            self._load(model_path)
        else:
            logger.warning("No model at %s — falling back", model_path)

        # ── State machine ─────────────────────────────────────────
        self._state: str = "still"           # still | collecting | classified
        self._raw: deque[np.ndarray] = deque(maxlen=300)
        self._onset_idx: int = 0
        self._still_count: int = 0
        self._classify_time: float = 0.0

        # ── Output ────────────────────────────────────────────────
        self._result_label: str = "standing_still"
        self._result_conf: float = 1.0
        self._ema_probs: np.ndarray | None = None
        self._sample_count: int = 0

    # ── public API ─────────────────────────────────────────────────

    def update(self, imu_window: np.ndarray) -> tuple[str, float]:
        """Feed the latest IMU window. Returns (label, confidence).

        The window is (100, 6) — a sliding buffer. We extract the latest row
        for onset/offset detection but classify on the full segmented gesture.
        """
        self._sample_count += 1
        now = time.time()

        if imu_window.ndim != 2 or imu_window.shape[0] < 5:
            return self._fallback()

        # Take the latest row as the current IMU reading
        row = imu_window[-1].astype(np.float32)  # (6,)
        self._raw.append(row)
        gyro_mag = float(np.linalg.norm(row[3:]))

        # ── State: STILL — wait for onset ─────────────────────────
        if self._state == "still":
            if gyro_mag > self.gyro_onset:
                self._state = "collecting"
                # Mark onset with a few pre-onset frames for context
                self._onset_idx = max(0, len(self._raw) - 1 - self.pre_onset)
                self._still_count = 0

        # ── State: COLLECTING — buffer until offset ───────────────
        elif self._state == "collecting":
            # Check for offset (sustained stillness)
            if gyro_mag < self.gyro_offset:
                self._still_count += 1
                if self._still_count >= self.offset_frames:
                    self._classify()
                    self._state = "classified"
                    self._classify_time = now
            else:
                self._still_count = 0

            # Safety: max duration exceeded
            duration = len(self._raw) - self._onset_idx
            if duration >= self.max_gesture_frames:
                self._classify()
                self._state = "classified"
                self._classify_time = now

        # ── State: CLASSIFIED — hold result briefly ───────────────
        elif self._state == "classified":
            if now - self._classify_time > self.display_hold:
                self._state = "still"
                self._result_label = "standing_still"
                self._result_conf = 1.0

        return self._result_label, self._result_conf

    # ── internal ───────────────────────────────────────────────────

    def _classify(self) -> None:
        """Extract gesture segment, resample, and run the CNN."""
        if not self._loaded:
            return

        # Extract raw samples from onset to now
        start = self._onset_idx
        end = len(self._raw)
        segment = np.array([self._raw[i] for i in range(start, end)], dtype=np.float32)

        if len(segment) < 8:
            # Too short — ignore as noise spike
            self._result_label = "standing_still"
            self._result_conf = 1.0
            return

        # Resample to exactly 100 timesteps via linear interpolation
        resampled = self._resample(segment, target_len=100)

        # Normalise
        if self._ch_mean is not None and self._ch_std is not None:
            resampled = (resampled - self._ch_mean) / (self._ch_std + 1e-8)

        # (C, T) for Conv1D
        x = torch.from_numpy(resampled).unsqueeze(0).permute(0, 2, 1).to(self.device)

        with torch.no_grad():
            logits = self._model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy().squeeze()

        # EMA smoothing across successive gestures
        if self._ema_probs is None:
            self._ema_probs = probs
        else:
            self._ema_probs = (
                self.ema_alpha * probs + (1 - self.ema_alpha) * self._ema_probs
            )

        best_idx = int(np.argmax(self._ema_probs))
        best_conf = float(self._ema_probs[best_idx])

        gyro_mag = float(np.linalg.norm(segment[:, 3:].mean(axis=0)))
        logger.debug(
            "Classified: %s (%.2f)  |  segment=%d frames  |  gyro=%.0f dps",
            self._gestures[best_idx], best_conf, len(segment), gyro_mag,
        )

        if best_conf >= self.confidence_floor:
            self._result_label = self._gestures[best_idx]
            self._result_conf = best_conf
        else:
            self._result_label = "standing_still"
            self._result_conf = 1.0 - best_conf

    @staticmethod
    def _resample(segment: np.ndarray, target_len: int = 100) -> np.ndarray:
        """Resample a (T, C) segment to target_len via linear interpolation."""
        T, C = segment.shape
        src = np.linspace(0, T - 1, T)
        dst = np.linspace(0, T - 1, target_len)
        out = np.empty((target_len, C), dtype=np.float32)
        for c in range(C):
            out[:, c] = np.interp(dst, src, segment[:, c])
        return out

    def _load(self, path: Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        from null_gesture.models.nn_model import GestureCNN

        self._gestures = checkpoint["gestures"]
        self._ch_mean = checkpoint["ch_mean"]
        self._ch_std = checkpoint["ch_std"]

        self._model = GestureCNN(n_classes=checkpoint["n_classes"])
        self._model.load_state_dict(checkpoint["state_dict"])
        self._model.to(self.device)
        self._model.eval()
        self._loaded = True

        logger.info(
            "Loaded GestureCNN: %d classes (%s), val_acc=%.2f%%",
            checkpoint["n_classes"],
            ", ".join(self._gestures),
            checkpoint.get("val_acc", 0) * 100,
        )

    def _fallback(self) -> tuple[str, float]:
        return "standing_still", 1.0

    # ── properties ─────────────────────────────────────────────────

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def gestures(self) -> list[str]:
        return list(self._gestures)
