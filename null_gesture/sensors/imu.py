"""IMU sensor — ESP32 + BMI270 via serial or TCP.

Reads 6-axis data: [ax, ay, az (g), gx, gy, gz (dps)].
"""

from __future__ import annotations

import json
import logging
import re
import socket
import time
from collections import deque

import numpy as np

logger = logging.getLogger("null_gesture.sensors.imu")

IMU_CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz")

_SAMPLE_RE = re.compile(
    r"accel\[g\]\s+x=\s*([-+]?\d+\.?\d*)\s+y=\s*([-+]?\d+\.?\d*)\s+z=\s*([-+]?\d+\.?\d*)"
    r"\s*\|\s*gyro\[dps\]\s+x=\s*([-+]?\d+\.?\d*)\s+y=\s*([-+]?\d+\.?\d*)\s+z=\s*([-+]?\d+\.?\d*)"
)


class IMUSensor:
    """Reads BMI270 IMU data via TCP or serial."""

    def __init__(self) -> None:
        self._socket: socket.socket | None = None
        self._serial = None
        self._mode: str = "none"
        self._rbuf: bytearray = bytearray()
        self._buffer: deque[tuple[float, np.ndarray]] = deque()
        self._connected = False

    # ── Connect ──────────────────────────────────────────────────

    def connect_tcp(self, host: str = "127.0.0.1", port: int = 9999) -> bool:
        try:
            self._socket = socket.create_connection((host, port), timeout=3)
            self._socket.settimeout(0.5)
            self._mode = "tcp"
            self._connected = True
            self._rbuf.clear()
            return True
        except OSError:
            return False

    def connect_serial(self, port: str = "/dev/ttyACM0", baud: int = 115200) -> bool:
        try:
            import serial as _ser
        except ImportError:
            logger.error("pyserial not installed")
            return False
        try:
            self._serial = _ser.Serial(port, baud, timeout=0.3)
            self._serial.dtr = False
            self._serial.rts = False
            self._serial.reset_input_buffer()
            self._mode = "serial"
            self._connected = True
            self._rbuf.clear()
            return True
        except OSError:
            return False

    # ── Read ─────────────────────────────────────────────────────

    def read_sample(self) -> np.ndarray | None:
        """Read one 6-channel sample. Returns (6,) array or None."""
        if self._mode == "tcp":
            return self._read_tcp()
        if self._mode == "serial":
            return self._read_serial()
        return None

    def _read_tcp(self) -> np.ndarray | None:
        if not self._socket or not self._connected:
            return None
        try:
            while b"\n" not in self._rbuf:
                chunk = self._socket.recv(4096)
                if not chunk:
                    self._connected = False
                    return None
                self._rbuf.extend(chunk)
            idx = self._rbuf.index(b"\n")
            line = bytes(self._rbuf[:idx])
            del self._rbuf[: idx + 1]
            if not line.strip():
                return None
            data = json.loads(line.decode("utf-8"))
            if data.get("type") != "sample":
                return None
            return np.array([data.get(ch, 0.0) for ch in IMU_CHANNELS], dtype=np.float32)
        except (TimeoutError, json.JSONDecodeError, UnicodeDecodeError, OSError):
            return None

    def _read_serial(self) -> np.ndarray | None:
        if not self._serial or not self._connected:
            return None
        try:
            n = self._serial.in_waiting
            if n > 0:
                self._rbuf.extend(self._serial.read(n))
            if b"\n" not in self._rbuf:
                return None
            idx = self._rbuf.index(b"\n")
            line = bytes(self._rbuf[:idx])
            del self._rbuf[: idx + 1]
            if not line.strip():
                return None
            text = line.decode("utf-8", errors="replace")
            m = _SAMPLE_RE.search(text)
            if m is None:
                return None
            return np.array([float(m.group(i)) for i in range(1, 7)], dtype=np.float32)
        except (OSError, UnicodeDecodeError):
            return None

    # ── Buffered reading ─────────────────────────────────────────

    def ingest(self, max_samples: int = 200) -> int:
        """Pull available samples into internal buffer."""
        added = 0
        for _ in range(max_samples):
            sample = self.read_sample()
            if sample is None:
                if not self._connected:
                    break
                continue
            self._buffer.append((time.monotonic(), sample))
            added += 1
        self._prune()
        return added

    def _prune(self) -> None:
        cutoff = time.monotonic() - 6.0
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()

    def get_window(self, window_s: float = 2.0, timesteps: int = 100) -> np.ndarray:
        """Return (timesteps, 6) array of recent data."""
        now = time.monotonic()
        data = [arr for ts, arr in self._buffer if now - ts <= window_s]
        if len(data) < 3:
            return np.zeros((timesteps, 6), dtype=np.float32)
        arr = np.array(data[-timesteps:], dtype=np.float32)
        if arr.shape[0] < timesteps:
            padded = np.zeros((timesteps, 6), dtype=np.float32)
            padded[-arr.shape[0]:] = arr
            return padded
        return arr

    def disconnect(self) -> None:
        self._connected = False
        if self._socket:
            try:
                self._socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._socket.close()
            self._socket = None
        if self._serial:
            self._serial.close()
            self._serial = None
        self._mode = "none"

    @property
    def connected(self) -> bool:
        return self._connected
