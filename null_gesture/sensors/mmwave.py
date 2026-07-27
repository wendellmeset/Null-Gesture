"""mmWave radar sensor — TI WRL6432 60GHz point cloud reader.

Parses TLV-formatted UART output from TI mmWave sensors running
the People Tracking or Gesture Recognition demo firmware.

Each frame: magic word → header → TLV items → detected points [x, y, z, vel].
"""

from __future__ import annotations

import logging
import struct
from collections import deque

import numpy as np

logger = logging.getLogger("null_gesture.sensors.mmwave")

# TLV magic word
MAGIC = bytes([0x02, 0x01, 0x04, 0x03, 0x06, 0x05, 0x08, 0x07])

# TLV types
TLV_DETECTED_POINTS = 1
TLV_RANGE_PROFILE = 2
TLV_TARGET_LIST = 7
TLV_POINT_CLOUD_SIDEBAND = 8


class MMWaveSensor:
    """Reads 3D point cloud from TI mmWave radar over UART."""

    def __init__(self) -> None:
        self._serial = None
        self._connected = False
        self._rbuf: bytearray = bytearray()
        self._buffer: deque[np.ndarray] = deque(maxlen=200)  # (N, 3) point clouds

        # Latest frame data
        self._points: np.ndarray = np.zeros((0, 3), dtype=np.float32)  # (N, 3) [x, y, z]
        self._velocities: np.ndarray = np.zeros((0,), dtype=np.float32)
        self._frame_number: int = 0
        self._num_frames: int = 0

    # ── Connect ──────────────────────────────────────────────────

    def connect(self, port: str = "/dev/ttyACM0", baud: int = 921600,
                config_file: str | None = None) -> bool:
        """Connect to mmWave radar over serial.

        The WRL6432 typically enumerates as two serial ports:
          - Application/User UART (data)
          - CLI/Configuration UART
        Connect to the DATA port (usually the higher-numbered one).
        """
        try:
            import serial as _ser
        except ImportError:
            logger.error("pyserial not installed")
            return False
        try:
            self._serial = _ser.Serial(port, baud, timeout=0.1)
            self._serial.reset_input_buffer()
            self._rbuf.clear()
            self._connected = True

            # Send config file if provided
            if config_file:
                self._send_config(config_file)

            logger.info("mmWave connected on %s @ %d", port, baud)
            return True
        except OSError as e:
            logger.error("mmWave connect failed: %s", e)
            return False

    # ── Read frames ──────────────────────────────────────────────

    def read_frame(self) -> bool:
        """Read one complete TLV frame. Returns True if a new frame was parsed."""
        if not self._connected or self._serial is None:
            return False

        # Read available bytes
        try:
            n = self._serial.in_waiting
            if n > 0:
                self._rbuf.extend(self._serial.read(n))
        except OSError:
            self._connected = False
            return False

        # Find magic word
        idx = self._rbuf.find(MAGIC)
        if idx == -1:
            # No frame start found — discard all but last 7 bytes (partial magic)
            if len(self._rbuf) > 7:
                self._rbuf = self._rbuf[-7:]
            return False

        # Discard bytes before magic
        if idx > 0:
            del self._rbuf[:idx]

        # Need at least: magic(8) + header(32) = 40 bytes
        if len(self._rbuf) < 40:
            return False

        # Parse header
        header = self._rbuf[8:40]
        try:
            _version, total_len, _platform, frame_num, _time_cpu = struct.unpack(
                "<IIIII", header[:20]
            )
        except struct.error:
            del self._rbuf[:8]  # skip bad magic
            return False

        # Need full frame
        total_len = total_len & 0xFFFF  # lower 16 bits
        frame_size = 40 + total_len
        if frame_size < 48:
            del self._rbuf[:8]
            return False

        if len(self._rbuf) < frame_size:
            return False  # incomplete frame — wait for more data

        # Extract TLV items
        tlv_data = self._rbuf[40:frame_size]
        del self._rbuf[:frame_size]

        self._parse_tlvs(tlv_data)
        self._frame_number = frame_num
        self._num_frames += 1

        # Store point cloud in buffer
        if len(self._points) > 0:
            self._buffer.append(self._points.copy())

        return True

    def _parse_tlvs(self, data: bytearray) -> None:
        """Extract detected points from TLV items."""
        offset = 0
        points_list = []
        vels_list = []

        while offset + 8 <= len(data):
            tlv_type, tlv_len = struct.unpack("<II", data[offset:offset + 8])
            offset += 8

            if tlv_len == 0 or offset + tlv_len > len(data):
                break

            if tlv_type in (TLV_DETECTED_POINTS, TLV_POINT_CLOUD_SIDEBAND):
                # Each point: 4 floats = 16 bytes (x, y, z, velocity)
                num_points = tlv_len // 16
                for i in range(num_points):
                    p_start = offset + i * 16
                    if p_start + 16 <= len(data):
                        x, y, z, vel = struct.unpack("<ffff", data[p_start:p_start + 16])
                        points_list.append([x, y, z])
                        vels_list.append(vel)

            elif tlv_type == TLV_TARGET_LIST:
                # Target list: each target = 68 bytes, position at offset 0
                num_targets = tlv_len // 68
                for i in range(num_targets):
                    t_start = offset + i * 68
                    if t_start + 16 <= len(data):
                        x, y, z, vel = struct.unpack("<f?f?f?f?", data[t_start:t_start + 16])  # try unpack
                        # Fallback: just first 4 floats
                        try:
                            x, y, z, vel = struct.unpack("<ffff", data[t_start:t_start + 16])
                            points_list.append([x, y, z])
                            vels_list.append(vel)
                        except struct.error:
                            pass

            offset += tlv_len

        self._points = np.array(points_list, dtype=np.float32) if points_list else np.zeros((0, 3), dtype=np.float32)
        self._velocities = np.array(vels_list, dtype=np.float32) if vels_list else np.zeros((0,), dtype=np.float32)

    # ── Get dominant point (hand) ─────────────────────────────────

    def get_dominant_point(self) -> np.ndarray | None:
        """Return the most likely hand position: [x, y, z] in meters.

        Heuristic: pick the strongest non-static point (highest SNR proxy
        via velocity, or closest point if only static objects).
        """
        if len(self._points) == 0:
            return None

        # Prefer moving points (velocity > threshold)
        moving = np.abs(self._velocities) > 0.2
        if moving.any():
            candidates = self._points[moving]
            vels = np.abs(self._velocities[moving])
            # Pick point with median velocity (most "hand-like")
            idx = np.argmin(np.abs(vels - np.median(vels)))
            return candidates[idx].astype(np.float32)

        # No moving points — return closest (could be hand at rest)
        dists = np.linalg.norm(self._points, axis=1)
        return self._points[np.argmin(dists)].astype(np.float32)

    def get_point_cloud(self) -> np.ndarray:
        """Return all detected points: (N, 3) [x, y, z]."""
        return self._points.copy()

    # ── Ingest ────────────────────────────────────────────────────

    def ingest(self) -> int:
        """Read all available frames. Returns number of new frames."""
        count = 0
        for _ in range(20):
            if self.read_frame():
                count += 1
            else:
                break
        return count

    def get_window(self, window_s: float = 2.0, timesteps: int = 100) -> np.ndarray:
        """Build a (timesteps, 3) window of dominant point positions.

        This mimics the IMU get_window() API for pipeline compatibility.
        Returns zeros if insufficient data.
        """
        if len(self._buffer) < 3:
            return np.zeros((timesteps, 3), dtype=np.float32)

        # Take recent frames up to window_s worth
        # mmWave typically runs at 10-20 fps, so timesteps=100 ≈ 5-10s
        pts = []
        for cloud in self._buffer:
            if len(cloud) > 0:
                # Use closest point as proxy for hand
                dists = np.linalg.norm(cloud, axis=1)
                pts.append(cloud[np.argmin(dists)])
            elif pts:
                pts.append(pts[-1])  # hold last position

        if len(pts) < 3:
            return np.zeros((timesteps, 3), dtype=np.float32)

        arr = np.array(pts[-timesteps:], dtype=np.float32)
        if arr.shape[0] < timesteps:
            padded = np.zeros((timesteps, 3), dtype=np.float32)
            padded[-arr.shape[0]:] = arr
            return padded
        return arr

    def disconnect(self) -> None:
        self._connected = False
        if self._serial:
            self._serial.close()
            self._serial = None

    # ── Configuration ────────────────────────────────────────────

    def _send_config(self, path: str) -> None:
        """Send a .cfg file to the radar over the data port.

        The IWRL6432 typically needs configuration before it starts
        outputting point cloud data. The config file contains CLI commands
        like 'channelCfg', 'profileCfg', 'frameCfg', etc.
        """
        import time as _time
        logger.info("Sending config: %s", path)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(("%", "#")):
                    continue
                cmd = line + "\r\n"
                if self._serial is not None:
                    self._serial.write(cmd.encode())
                    _time.sleep(0.05)
                    # Read echo
                    self._serial.read(self._serial.in_waiting)
        logger.info("Config sent")

    def dump_raw(self, count: int = 200) -> bytes:
        """Read and return raw bytes for debugging."""
        if not self._connected or self._serial is None:
            return b""
        _time = __import__("time")
        data = bytearray()
        t0 = _time.time()
        while len(data) < count and _time.time() - t0 < 3:
            n = self._serial.in_waiting
            if n:
                data.extend(self._serial.read(n))
            _time.sleep(0.01)
        return bytes(data)

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def frame_number(self) -> int:
        return self._frame_number
