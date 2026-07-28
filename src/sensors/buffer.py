"""Thread-safe ring buffers for time-series sensor data."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

import numpy as np


class SensorRingBuffer:
    """Thread-safe ring buffer with timestamp-based access.

    Stores samples as dicts (IMU) or tuples (mmWave) with associated
    timestamps. Supports retrieval by time window and count-limited queries.

    Thread safety: uses a reentrant lock for all mutations.

    Usage::

        buf = SensorRingBuffer(maxlen=500)
        buf.append({"ax": 0.1, "ay": 0.2, ...}, timestamp=time.time())

        # Get all samples from the last 1.5 seconds:
        window = buf.get_window(duration_sec=1.5)
    """

    def __init__(self, maxlen: int = 500, name: str = "sensor"):
        """
        Args:
            maxlen: Maximum number of samples stored.
            name: Human-readable label for debug logging.
        """
        self._buffer: deque[tuple[float, Any]] = deque(maxlen=maxlen)
        self._lock = threading.RLock()
        self._name = name
        self._total_appended = 0
        self._total_dropped = 0

    # ── Mutation ─────────────────────────────────────────────────────────

    def append(self, sample: Any, timestamp: float | None = None) -> None:
        """Add a sample to the buffer.

        Args:
            sample: Sensor sample (dict for IMU, tuple for mmWave, etc.).
            timestamp: Unix timestamp. Auto-generated if None.
        """
        if timestamp is None:
            timestamp = time.time()
        with self._lock:
            if len(self._buffer) == self._buffer.maxlen:
                self._total_dropped += 1
            self._buffer.append((timestamp, sample))
            self._total_appended += 1

    def clear(self) -> None:
        """Empty the buffer."""
        with self._lock:
            self._buffer.clear()

    # ── Queries ──────────────────────────────────────────────────────────

    def get_window(
        self, duration_sec: float | None = None, max_samples: int | None = None
    ) -> list[Any]:
        """Get samples from a recent time window.

        Args:
            duration_sec: Look back this many seconds from now.
            max_samples: Limit to at most this many samples.

        Returns:
            List of samples (without timestamps), most recent last.
        """
        with self._lock:
            if not self._buffer:
                return []

            now = time.time()
            if duration_sec is not None:
                cutoff = now - duration_sec
                # Walk from oldest to newest
                result = [
                    sample
                    for ts, sample in self._buffer
                    if ts >= cutoff
                ]
                if max_samples is not None and len(result) > max_samples:
                    result = result[-max_samples:]
                return result
            else:
                items = list(self._buffer)
                if max_samples is not None:
                    items = items[-max_samples:]
                return [sample for _, sample in items]

    def get_window_with_timestamps(
        self, duration_sec: float | None = None
    ) -> list[tuple[float, Any]]:
        """Like get_window, but returns (timestamp, sample) pairs."""
        with self._lock:
            if not self._buffer:
                return []
            if duration_sec is not None:
                cutoff = time.time() - duration_sec
                return [(ts, s) for ts, s in self._buffer if ts >= cutoff]
            return list(self._buffer)

    def latest(self) -> Any | None:
        """Return the most recent sample, or None if empty."""
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer[-1][1]

    def latest_timestamp(self) -> float | None:
        """Return timestamp of most recent sample, or None."""
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer[-1][0]

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._name

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._buffer)

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return len(self._buffer) == 0

    @property
    def total_appended(self) -> int:
        return self._total_appended

    def __len__(self) -> int:
        return self.count

    def __repr__(self) -> str:
        return (
            f"SensorRingBuffer(name={self._name!r}, count={self.count}, "
            f"maxlen={self._buffer.maxlen})"
        )


class IMUBuffer(SensorRingBuffer):
    """Specialized ring buffer for IMU data with numpy export."""

    CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]

    def to_numpy(self, duration_sec: float | None = None) -> dict[str, np.ndarray]:
        """Export buffered IMU data as numpy arrays.

        Args:
            duration_sec: Look-back window. None = all buffered.

        Returns:
            Dict with keys 'ax','ay','az','gx','gy','gz','t', each a 1D array.
        """
        window = self.get_window_with_timestamps(duration_sec)
        if not window:
            return {k: np.array([], dtype=np.float32) for k in self.CHANNELS + ["t"]}

        n = len(window)
        arrays = {k: np.zeros(n, dtype=np.float32) for k in self.CHANNELS}
        timestamps = np.zeros(n, dtype=np.float64)

        for i, (ts, sample) in enumerate(window):
            timestamps[i] = ts
            for ch in self.CHANNELS:
                arrays[ch][i] = sample.get(ch, 0.0)

        arrays["t"] = timestamps
        return arrays


class MMWaveBuffer(SensorRingBuffer):
    """Specialized ring buffer for mmWave point cloud data with numpy export."""

    def to_numpy(self, duration_sec: float | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
        """Export buffered mmWave frames.

        Args:
            duration_sec: Look-back window.

        Returns:
            List of (points, velocities) tuples, where points is (N,3) and
            velocities is (N,). Most recent frame last.
        """
        window = self.get_window_with_timestamps(duration_sec)
        return [(points.copy(), vels.copy()) for _, (points, vels) in window]
