"""IMU sensor: TCP client for ESP32 + BMI270 data stream."""

from __future__ import annotations

import json
import logging
import socket
import struct
import time
from collections import deque
from typing import Callable

import numpy as np

from null_gesture.config import IMUConfig, imu_config as default_imu_config

logger = logging.getLogger("null_gesture.sensors.imu")

# IMU channel names in order
IMU_CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz")


class IMUClient:
    """Connects to the ESP32 TCP data server and buffers IMU samples."""

    def __init__(self, config: IMUConfig | None = None) -> None:
        self.config = config or default_imu_config
        self._socket: socket.socket | None = None
        self._rbuf: bytearray = bytearray()
        self._buffer: deque[tuple[float, np.ndarray]] = deque()
        self._connected = False
        self._sample_count = 0

    def connect(self) -> bool:
        """Connect to the ESP32 TCP server. Returns True on success."""
        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(5.0)
            self._socket.connect((self.config.tcp_host, self.config.tcp_port))
            self._socket.settimeout(1.0)
            self._connected = True
            logger.info(
                "IMU connected to %s:%d", self.config.tcp_host, self.config.tcp_port
            )
            return True
        except (OSError, ConnectionRefusedError, socket.timeout) as exc:
            logger.error("IMU connection failed: %s", exc)
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Close the TCP connection."""
        self._connected = False
        if self._socket:
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._socket.close()
            self._socket = None
        logger.info("IMU disconnected (%d samples received)", self._sample_count)

    def read_sample(self) -> dict | None:
        """Read one JSON line from the socket. Returns parsed dict or None."""
        if not self._socket or not self._connected:
            return None
        try:
            # Check if we already have a complete line in the buffer
            while b"\n" not in self._rbuf:
                chunk = self._socket.recv(4096)
                if not chunk:
                    self._connected = False
                    logger.warning("IMU server closed connection")
                    return None
                self._rbuf.extend(chunk)

            # Extract first complete line
            idx = self._rbuf.index(b"\n")
            line = bytes(self._rbuf[:idx])
            del self._rbuf[: idx + 1]

            if not line.strip():
                return None
            data = json.loads(line.decode("utf-8"))
            return data
        except (socket.timeout, json.JSONDecodeError, UnicodeDecodeError):
            return None
        except OSError as exc:
            logger.error("IMU read error: %s", exc)
            self._connected = False
            return None

    def ingest(self, max_samples: int = 50) -> int:
        """Drain pending samples into the internal buffer. Returns count added."""
        added = 0
        for _ in range(max_samples):
            data = self.read_sample()
            if data is None:
                break
            if data.get("type") != "sample":
                continue
            ts = data.get("timestamp", time.monotonic())
            arr = np.array(
                [data.get(ch, 0.0) for ch in IMU_CHANNELS], dtype=np.float32
            )
            self._buffer.append((ts, arr))
            self._sample_count += 1
            added += 1
        # Prune old samples outside window
        self._prune()
        return added

    def _prune(self) -> None:
        cutoff = time.monotonic() - self.config.window_seconds * 2
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()

    def get_window(self) -> np.ndarray:
        """Return a fixed-length (timesteps, channels) window of the most recent data."""
        window_s = self.config.window_seconds
        timesteps = self.config.timesteps
        now = time.monotonic()
        # Collect samples within the window
        window_data = [
            arr for ts, arr in self._buffer if now - ts <= window_s
        ]
        if len(window_data) < 3:
            return np.zeros((timesteps, self.config.channels), dtype=np.float32)
        arr = np.array(window_data[-timesteps:], dtype=np.float32)
        # Zero-pad or resample to exact timesteps
        if arr.shape[0] < timesteps:
            padded = np.zeros((timesteps, self.config.channels), dtype=np.float32)
            padded[-arr.shape[0] :] = arr
            return padded
        if arr.shape[0] > timesteps:
            indices = np.linspace(0, arr.shape[0] - 1, timesteps).astype(int)
            return arr[indices]
        return arr

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def sample_count(self) -> int:
        return self._sample_count
