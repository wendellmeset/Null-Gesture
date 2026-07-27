"""Preprocessing — filter, segment, and normalize IMU data.

Pipeline:
  raw → low-pass filter → onset/offset segmentation → resample → normalize
"""

from __future__ import annotations

from collections import deque

import numpy as np
from scipy import signal as scipy_signal


# ══════════════════════════════════════════════════════════════════════
# Filtering
# ══════════════════════════════════════════════════════════════════════

def lowpass_filter(data: np.ndarray, cutoff: float = 10.0, fs: float = 50.0,
                   order: int = 4) -> np.ndarray:
    """Apply Butterworth low-pass filter to IMU data.

    Args:
        data: (T, C) array.
        cutoff: Cutoff frequency in Hz.
        fs: Sample rate in Hz.
        order: Filter order.

    Returns:
        Filtered array, same shape.
    """
    nyquist = fs / 2
    normal_cutoff = cutoff / nyquist
    if normal_cutoff >= 1.0:
        return data
    b, a = scipy_signal.butter(order, normal_cutoff, btype="low")
    return scipy_signal.filtfilt(b, a, data, axis=0)


# ══════════════════════════════════════════════════════════════════════
# Segmentation (onset/offset)
# ══════════════════════════════════════════════════════════════════════

class GestureSegmenter:
    """Detects gesture boundaries from gyro magnitude.

    State machine:
      still → collecting (gyro > onset_thresh)
      collecting → still (gyro < offset_thresh for N frames)

    The segmented window is the raw samples between onset and offset.
    """

    def __init__(
        self,
        onset_thresh: float = 20.0,     # dps
        offset_thresh: float = 10.0,    # dps
        offset_frames: int = 8,         # consecutive still frames to end
        pre_onset: int = 6,             # frames before onset to include
        max_frames: int = 250,          # safety cap
    ):
        self.onset_thresh = onset_thresh
        self.offset_thresh = offset_thresh
        self.offset_frames = offset_frames
        self.pre_onset = pre_onset
        self.max_frames = max_frames

        self._state = "still"
        self._buffer: deque[np.ndarray] = deque(maxlen=400)
        self._onset_idx = 0
        self._still_count = 0

    def feed(self, sample: np.ndarray) -> np.ndarray | None:
        """Feed one IMU sample (6,). Returns segmented window (T,6) or None.

        When a gesture completes, returns the window and resets.
        """
        self._buffer.append(sample)
        gyro_mag = float(np.linalg.norm(sample[3:]))

        if self._state == "still":
            if gyro_mag > self.onset_thresh:
                self._state = "collecting"
                self._onset_idx = max(0, len(self._buffer) - 1 - self.pre_onset)
                self._still_count = 0

        elif self._state == "collecting":
            if gyro_mag < self.offset_thresh:
                self._still_count += 1
                if self._still_count >= self.offset_frames:
                    segment = self._extract()
                    self._state = "still"
                    return segment
            else:
                self._still_count = 0

            if len(self._buffer) - self._onset_idx >= self.max_frames:
                segment = self._extract()
                self._state = "still"
                return segment

        return None

    def _extract(self) -> np.ndarray | None:
        """Pull the gesture segment from the buffer."""
        start = self._onset_idx
        end = len(self._buffer)
        if end - start < 5:
            return None
        return np.array([self._buffer[i] for i in range(start, end)], dtype=np.float32)

    def reset(self) -> None:
        self._state = "still"
        self._buffer.clear()
        self._still_count = 0

    @property
    def state(self) -> str:
        return self._state

    @property
    def buffer_size(self) -> int:
        return max(0, len(self._buffer) - self._onset_idx)


# ══════════════════════════════════════════════════════════════════════
# Resampling & normalisation
# ══════════════════════════════════════════════════════════════════════

def resample_window(data: np.ndarray, target_len: int = 100) -> np.ndarray:
    """Resample (T, C) to fixed target_len via linear interpolation."""
    T, C = data.shape
    if T == target_len:
        return data
    src = np.linspace(0, T - 1, T)
    dst = np.linspace(0, T - 1, target_len)
    out = np.empty((target_len, C), dtype=np.float32)
    for c in range(C):
        out[:, c] = np.interp(dst, src, data[:, c])
    return out


def normalize(data: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score normalise each channel independently.

    Args:
        data: (T, C) or (N, T, C) array.

    Returns:
        Normalised array, same shape.
    """
    if data.ndim == 2:
        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True) + eps
        return (data - mean) / std
    elif data.ndim == 3:
        mean = data.mean(axis=(0, 1), keepdims=True)
        std = data.std(axis=(0, 1), keepdims=True) + eps
        return (data - mean) / std
    return data
