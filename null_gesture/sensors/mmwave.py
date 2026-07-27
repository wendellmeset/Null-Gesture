"""mmWave radar sensor — TI IWRL6432 60GHz point cloud reader.

Based on wshanmu/mmwave_lab. Sends .cfg configuration, then reads
TLV-framed point cloud data (float + fixed-point types).
"""

from __future__ import annotations

import logging
import struct
import time
from collections import deque
from pathlib import Path

import numpy as np

logger = logging.getLogger("null_gesture.sensors.mmwave")

MAGIC_WORD = b"\x02\x01\x04\x03\x06\x05\x08\x07"

# TLV types
POINT_CLOUD_FLOAT = 1
POINT_CLOUD_FIXED = {301, 1020}

CLI_FAILURE = ("error", "not recognized", "invalid", "failed")
CLI_OK = ("done", "mmwdemo:", "skipped")


class MMWaveSensor:
    """Reads 3D point cloud from TI IWRL6432 radar.

    Usage:
        radar = MMWaveSensor()
        radar.connect("/dev/ttyUSB0", config_file="iwrl6432.cfg")
        radar.ingest()         # read available frames
        pts = radar.points     # (N, 3) current point cloud
    """

    def __init__(self) -> None:
        self._serial = None
        self._connected = False
        self._rbuf: bytearray = bytearray()
        self._buffer: deque[np.ndarray] = deque(maxlen=200)

        self._points = np.zeros((0, 3), dtype=np.float32)
        self._velocities = np.zeros((0,), dtype=np.float32)
        self._frame_number = 0
        self._num_frames = 0

    # ── Connect & configure ───────────────────────────────────────

    def connect(
        self,
        port: str = "/dev/ttyACM0",
        baud: int = 115200,
        config_file: str | None = None,
    ) -> bool:
        """Connect to radar and optionally send configuration."""
        try:
            import serial as _ser
        except ImportError:
            logger.error("pyserial not installed")
            return False

        try:
            self._serial = _ser.Serial(port, baud, timeout=0.2)
            self._serial.reset_input_buffer()
            time.sleep(0.5)
            self._connected = True
            logger.info("mmWave connected on %s @ %d", port, baud)

            if config_file:
                self._send_config(config_file)

            return True
        except OSError as e:
            logger.error("mmWave connect failed: %s", e)
            return False

    def _send_config(self, path: str) -> None:
        """Send .cfg file and start sensor."""
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {path}")

        commands = self._load_config(config_path)

        start_cmd = None
        for cmd in commands:
            if cmd.startswith("sensorStart"):
                start_cmd = cmd
                continue

            name = cmd.split()[0]
            if name == "baudRate":
                continue  # skip baud rate changes

            logger.debug("> %s", cmd)
            self._write_cli(cmd)
            reply = self._read_text(quiet_time=0.15)

            reply_lower = reply.lower()
            if any(p in reply_lower for p in CLI_FAILURE):
                if name in {"cfarScndPassCfg", "compressionCfg"}:
                    logger.debug("Skipping unsupported: %s", name)
                    continue
                raise RuntimeError(f"CLI error for {cmd}: {reply.strip()}")

            if not any(p in reply_lower for p in CLI_OK) and reply.strip():
                raise RuntimeError(
                    f"Unexpected CLI response for {cmd!r}: {reply.strip()}"
                )

        if start_cmd is None:
            raise ValueError("No sensorStart in config")

        if self._serial is not None:
            self._serial.reset_input_buffer()
        logger.debug("> %s", start_cmd)
        self._write_cli(start_cmd)
        logger.info("Radar configured — streaming started")

    @staticmethod
    def _load_config(path: Path) -> list[str]:
        cmds = []
        for line in path.read_text("utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line and not line.startswith(("%", "#")):
                cmds.append(line)
        if not any(c.startswith("sensorStart") for c in cmds):
            raise ValueError("Config has no sensorStart command")
        return cmds

    def _write_cli(self, cmd: str) -> None:
        assert self._serial is not None
        self._serial.write((cmd + "\n").encode("ascii"))
        self._serial.flush()

    def _read_text(self, quiet_time: float = 0.15, max_time: float = 2.0) -> str:
        assert self._serial is not None
        start = time.monotonic()
        last_rx = start
        chunks = []
        while time.monotonic() - start < max_time:
            n = self._serial.in_waiting
            if n:
                chunks.append(self._serial.read(n))
                last_rx = time.monotonic()
            elif time.monotonic() - last_rx >= quiet_time:
                break
            time.sleep(0.01)
        return b"".join(chunks).decode("ascii", errors="ignore")

    # ── Frame reading ─────────────────────────────────────────────

    def read_frame(self, timeout_s: float = 2.0) -> bool:
        """Read one TLV frame. Returns True on success."""
        if not self._connected or self._serial is None:
            return False

        try:
            # Wait for magic word
            window = bytearray()
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                b = self._serial.read(1)
                if not b:
                    continue
                window.extend(b)
                if len(window) > len(MAGIC_WORD):
                    del window[0]
                if bytes(window) == MAGIC_WORD:
                    break
            else:
                return False  # timeout — no frame available

            # Read header (40 bytes total, 32 after magic)
            header = MAGIC_WORD + self._read_exact(32, timeout_s)
            _, total_len, _, frame_num, _, _, num_tlvs, _ = struct.unpack_from(
                "<8I", header, 8
            )
            total_len = total_len & 0xFFFFFFFF

            if not (40 <= total_len <= 2_000_000) or num_tlvs > 64:
                return False

            # Read rest of frame
            packet = header + self._read_exact(total_len - 40, timeout_s)
            self._parse_tlvs(packet, num_tlvs)
            self._frame_number = frame_num
            self._num_frames += 1

            if len(self._points) > 0:
                self._buffer.append(self._points.copy())

            return True

        except (TimeoutError, struct.error, ValueError):
            return False

    def _read_exact(self, count: int, timeout_s: float) -> bytes:
        assert self._serial is not None
        out = bytearray()
        deadline = time.monotonic() + timeout_s
        while len(out) < count:
            if time.monotonic() > deadline:
                raise TimeoutError(f"Read {len(out)}/{count} bytes")
            chunk = self._serial.read(count - len(out))
            if chunk:
                out.extend(chunk)
        return bytes(out)

    def _parse_tlvs(self, packet: bytes, num_tlvs: int) -> None:
        """Try both 40 and 52 byte header offsets."""
        for header_size in (40, 52):
            offset = header_size
            points_list = []
            vels_list = []

            for _ in range(num_tlvs):
                if offset + 8 > len(packet):
                    break
                tlv_type, tlv_len = struct.unpack_from("<II", packet, offset)
                offset += 8

                if tlv_type == 0 or tlv_len > len(packet) - offset:
                    break

                payload = packet[offset:offset + tlv_len]
                offset += tlv_len

                if tlv_type == POINT_CLOUD_FLOAT:
                    pts, vels = self._decode_float(payload)
                    points_list.extend(pts)
                    vels_list.extend(vels)
                elif tlv_type in POINT_CLOUD_FIXED:
                    pts, vels = self._decode_fixed(payload)
                    points_list.extend(pts)
                    vels_list.extend(vels)

            if points_list:
                self._points = np.array(points_list, dtype=np.float32)
                self._velocities = np.array(vels_list, dtype=np.float32)
                return

        self._points = np.zeros((0, 3), dtype=np.float32)
        self._velocities = np.zeros((0,), dtype=np.float32)

    @staticmethod
    def _decode_float(payload: bytes) -> tuple[list, list]:
        count = len(payload) // 16
        if count == 0:
            return [], []
        vals = np.frombuffer(payload[:count * 16], dtype="<f4").reshape(count, 4)
        return vals[:, :3].tolist(), vals[:, 3].tolist()

    @staticmethod
    def _decode_fixed(payload: bytes) -> tuple[list, list]:
        if len(payload) < 20:
            return [], []
        xyz_unit, doppler_unit, _, _ = struct.unpack_from("<ffff", payload, 0)
        num_major, _ = struct.unpack_from("<HH", payload, 16)

        pts, vels = [], []
        offset = 20
        for _ in range(num_major):
            if offset + 10 > len(payload):
                break
            x, y, z, doppler, _, _ = struct.unpack_from("<hhhhBB", payload, offset)
            pts.append([x * xyz_unit, y * xyz_unit, z * xyz_unit])
            vels.append(doppler * doppler_unit)
            offset += 10
        return pts, vels

    # ── Public access ─────────────────────────────────────────────

    def ingest(self) -> int:
        """Read all available frames. Returns number of new frames."""
        count = 0
        for _ in range(20):
            if self.read_frame(timeout_s=0.3):
                count += 1
            else:
                break
        return count

    def get_dominant_point(self) -> np.ndarray | None:
        """Return most likely hand position: (3,) [x, y, z] in meters."""
        if len(self._points) == 0:
            return None
        moving = np.abs(self._velocities) > 0.2
        if moving.any():
            candidates = self._points[moving]
            vels = np.abs(self._velocities[moving])
            idx = np.argmin(np.abs(vels - np.median(vels)))
            return candidates[idx].astype(np.float32)
        dists = np.linalg.norm(self._points, axis=1)
        return self._points[np.argmin(dists)].astype(np.float32)

    @property
    def points(self) -> np.ndarray:
        """Current point cloud: (N, 3)."""
        return self._points.copy()

    def get_window(self, window_s: float = 2.0, timesteps: int = 100) -> np.ndarray:
        """Build (timesteps, 3) window for pipeline compat."""
        if len(self._buffer) < 3:
            return np.zeros((timesteps, 3), dtype=np.float32)
        pts = []
        for cloud in self._buffer:
            if len(cloud) > 0:
                dists = np.linalg.norm(cloud, axis=1)
                pts.append(cloud[np.argmin(dists)])
            elif pts:
                pts.append(pts[-1])
        if len(pts) < 3:
            return np.zeros((timesteps, 3), dtype=np.float32)
        arr = np.array(pts[-timesteps:], dtype=np.float32)
        if arr.shape[0] < timesteps:
            padded = np.zeros((timesteps, 3), dtype=np.float32)
            padded[-arr.shape[0]:] = arr
            return padded
        return arr

    def disconnect(self) -> None:
        if self._serial and self._connected:
            try:
                self._write_cli("sensorStop 0")
                time.sleep(0.1)
            except OSError:
                logger.debug("sensorStop ignored during disconnect")
        self._connected = False
        if self._serial:
            self._serial.close()
            self._serial = None

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def frame_number(self) -> int:
        return self._frame_number
