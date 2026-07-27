"""IMU Reader: ESP32 + BMI270 via direct serial.

Reads accelerometer and gyroscope data from an ESP32 streaming
BMI270 samples over USB serial.

Usage::

    from Readers import IMUReader

    imu = IMUReader("/dev/ttyUSB0")
    imu.connect()
    sample = imu.read()
    # {'ax': 0.01, 'ay': 0.02, 'az': 1.01, 'gx': 0.1, 'gy': -0.2, 'gz': 0.0, 't': 1234.5}
    for sample in imu.stream():
        print(f"accel: ({sample['ax']:.3f}, {sample['ay']:.3f}, {sample['az']:.3f})")
    imu.disconnect()
"""

from __future__ import annotations

import re
import time
from typing import Iterator

# Regex for lines like: accel[g] x= 0.123 y=-0.456 z= 1.001 | gyro[dps] x= 1.2 y=-0.3 z= 0.0
_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_SAMPLE_RE = re.compile(
    rf"accel\[g\]\s+x=\s*({_FLOAT})\s+y=\s*({_FLOAT})\s+z=\s*({_FLOAT})"
    rf"\s+\|\s+gyro\[dps\]\s+x=\s*({_FLOAT})\s+y=\s*({_FLOAT})\s+z=\s*({_FLOAT})"
)

# Fallback: JSON lines like {"ax": 0.01, "ay": ..., "type": "sample"}
_JSON_SAMPLE_RE = re.compile(r'"type"\s*:\s*"sample"')

CHANNEL_KEYS = ("ax", "ay", "az", "gx", "gy", "gz")


class IMUReader:
    """Minimal ESP32 + BMI270 IMU reader over direct serial."""

    def __init__(self, port: str = "/dev/ttyUSB0", baud: int = 115200):
        """
        Args:
            port: Serial port of the ESP32.
            baud: Baud rate.
        """
        self._port = port
        self._baud = baud
        self._serial = None
        self._connected = False
        self._last_sample: dict | None = None
        self._sample_count = 0

    # ── Connection ───────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Open serial port. Returns True on success."""
        try:
            import serial
        except ImportError:
            raise ImportError("pyserial is required: pip install pyserial")

        try:
            self._serial = serial.Serial(self._port, self._baud, timeout=0.1)
            self._serial.reset_input_buffer()
            self._connected = True
            return True
        except OSError as e:
            raise ConnectionError(f"Failed to open {self._port}: {e}")

    def disconnect(self) -> None:
        """Close serial port."""
        self._connected = False
        if self._serial:
            self._serial.close()
            self._serial = None

    # ── Parsing ──────────────────────────────────────────────────────────

    @staticmethod
    def parse_line(line: str) -> dict | None:
        """Parse a text line into an IMU sample dict, or None."""
        m = _SAMPLE_RE.search(line)
        if m:
            return {
                "ax": float(m.group(1)),
                "ay": float(m.group(2)),
                "az": float(m.group(3)),
                "gx": float(m.group(4)),
                "gy": float(m.group(5)),
                "gz": float(m.group(6)),
                "t": time.time(),
            }
        return None

    # ── Reading ──────────────────────────────────────────────────────────

    def read(self) -> dict | None:
        """Read one sample. Returns dict with ax,ay,az,gx,gy,gz,t or None."""
        if not self._connected or self._serial is None:
            return None
        try:
            while True:
                raw = self._serial.readline()
                if not raw:
                    return None
                line = raw.decode("utf-8", errors="replace").strip()
                sample = self.parse_line(line)
                if sample is not None:
                    self._last_sample = sample
                    self._sample_count += 1
                    return sample
        except OSError:
            self._connected = False
            return None

    def stream(self) -> Iterator[dict]:
        """Generator yielding IMU samples continuously.

        Yields dicts with keys: ``ax, ay, az, gx, gy, gz, t``.
        """
        if not self._connected or self._serial is None:
            return
        try:
            while self._connected:
                raw = self._serial.readline()
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace").strip()
                sample = self.parse_line(line)
                if sample is not None:
                    self._last_sample = sample
                    self._sample_count += 1
                    yield sample
        except OSError:
            self._connected = False

    def __iter__(self) -> Iterator[dict]:
        return self.stream()

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def last_sample(self) -> dict | None:
        return self._last_sample

    @property
    def sample_count(self) -> int:
        return self._sample_count
