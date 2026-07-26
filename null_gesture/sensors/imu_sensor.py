"""IMU sensor: reads ESP32 + BMI270 via TCP or direct serial."""

from __future__ import annotations

import json
import logging
import re
import socket
import time
from collections import deque

import numpy as np

from null_gesture.config import IMUConfig
from null_gesture.config import imu_config as default_imu_config

logger = logging.getLogger("null_gesture.sensors.imu")

IMU_CHANNELS = ("ax", "ay", "az", "gx", "gy", "gz")

_SAMPLE_RE = re.compile(
    r"accel\[g\]\s+x=\s*([-+]?\d+\.?\d*)\s+y=\s*([-+]?\d+\.?\d*)\s+z=\s*([-+]?\d+\.?\d*)"
    r"\s*\|\s*gyro\[dps\]\s+x=\s*([-+]?\d+\.?\d*)\s+y=\s*([-+]?\d+\.?\d*)\s+z=\s*([-+]?\d+\.?\d*)"
)


class IMUClient:
    """Reads BMI270 IMU data via TCP or direct serial."""

    def __init__(self, config: IMUConfig | None = None) -> None:
        self.config = config or default_imu_config
        self._socket: socket.socket | None = None
        self._serial: object | None = None  # serial.Serial (lazy import)
        self._rbuf: bytearray = bytearray()
        self._buffer: deque[tuple[float, np.ndarray]] = deque()
        self._connected = False
        self._sample_count = 0

    def connect_tcp(self, host: str = "127.0.0.1", port: int = 9999) -> bool:
        try:
            self._socket = socket.create_connection((host, port), timeout=3)
            self._socket.settimeout(0.5)
            self._mode = "tcp"
            self._connected = True
            self._rbuf.clear()
            logger.info("IMU TCP connected %s:%d", host, port)
            return True
        except OSError as exc:
            logger.error("IMU TCP failed: %s", exc)
            return False

    def connect_serial(self, port: str = "/dev/ttyACM0", baud: int = 115200) -> bool:
        try:
            import serial  # type: ignore[import-untyped]
        except ImportError:
            logger.error("pyserial not installed")
            return False
        try:
            self._serial = serial.Serial(port, baud, timeout=0.3)
            self._serial.dtr = False
            self._serial.rts = False
            self._serial.reset_input_buffer()
            self._mode = "serial"
            self._connected = True
            self._rbuf.clear()
            logger.info("IMU serial connected %s @ %d", port, baud)
            return True
        except OSError as exc:
            logger.error("IMU serial failed: %s", exc)
            return False

    def connect(self, host: str = "127.0.0.1", port: int = 9999) -> bool:
        return self.connect_tcp(host, port)

    def read_sample(self) -> dict | None:
        if self._mode == "tcp":
            return self._read_tcp()
        if self._mode == "serial":
            return self._read_serial()
        return None

    def _read_tcp(self) -> dict | None:
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
            data: dict = json.loads(line.decode("utf-8"))
            return data
        except (TimeoutError, json.JSONDecodeError, UnicodeDecodeError):
            return None
        except OSError:
            self._connected = False
            return None

    def _read_serial(self) -> dict | None:
        if not self._serial or not self._connected:
            return None
        try:
            while b"\n" not in self._rbuf:
                n = self._serial.in_waiting or 1
                chunk = self._serial.read(n)
                if not chunk:
                    return None
                self._rbuf.extend(chunk)
            idx = self._rbuf.index(b"\n")
            line = bytes(self._rbuf[:idx])
            del self._rbuf[: idx + 1]
            if not line.strip():
                return None
            text = line.decode("utf-8", errors="replace")
            m = _SAMPLE_RE.search(text)
            if m is None:
                return None
            return {
                "type": "sample",
                "timestamp": time.monotonic(),
                "ax": float(m.group(1)), "ay": float(m.group(2)), "az": float(m.group(3)),
                "gx": float(m.group(4)), "gy": float(m.group(5)), "gz": float(m.group(6)),
            }
        except (OSError, UnicodeDecodeError):
            return None

    def ingest(self, max_samples: int = 200) -> int:
        added = 0
        for _ in range(max_samples):
            data = self.read_sample()
            if data is None:
                if not self._connected:
                    break
                continue
            if data.get("type") != "sample":
                continue
            ts = data.get("timestamp", time.monotonic())
            arr = np.array([data.get(ch, 0.0) for ch in IMU_CHANNELS], dtype=np.float32)
            self._buffer.append((ts, arr))
            self._sample_count += 1
            added += 1
        self._prune()
        return added

    def _prune(self) -> None:
        cutoff = time.monotonic() - self.config.window_seconds * 2
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()

    def get_window(self) -> np.ndarray:
        timesteps = self.config.timesteps
        now = time.monotonic()
        ws = self.config.window_seconds
        data = [arr for ts, arr in self._buffer if now - ts <= ws]
        if len(data) < 3:
            return np.zeros((timesteps, self.config.channels), dtype=np.float32)
        arr = np.array(data[-timesteps:], dtype=np.float32)
        if arr.shape[0] < timesteps:
            padded = np.zeros((timesteps, self.config.channels), dtype=np.float32)
            padded[-arr.shape[0] :] = arr
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
        logger.info("IMU disconnected (%d samples)", self._sample_count)

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def sample_count(self) -> int:
        return self._sample_count
