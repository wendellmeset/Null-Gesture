"""mmWave Reader: TI IWRL6432 60GHz radar point-cloud reader.

Sends a .cfg configuration file to the radar, then reads TLV-framed
point cloud data from the UART stream.

Usage::

    from Readers import MMWaveReader

    radar = MMWaveReader()
    radar.connect("/dev/ttyACM0", config="mmwave_hand_50cm.cfg")
    for points, velocities in radar.stream():
        # points: (N, 3) array of [x, y, z] in meters
        # velocities: (N,) array of doppler velocities
        print(f"Frame: {len(points)} points")
    radar.disconnect()
"""

from __future__ import annotations

import struct
import time
from collections import deque
from pathlib import Path
from typing import Iterator

import numpy as np

MAGIC_WORD = b"\x02\x01\x04\x03\x06\x05\x08\x07"

TLV_POINT_CLOUD_FLOAT = 1
TLV_POINT_CLOUD_FIXED = {301, 1020}

CLI_FAILURE = ("error", "not recognized", "invalid", "failed")
CLI_OK = ("done", "mmwdemo:", "skipped")


class MMWaveReader:
    """Minimal TI IWRL6432 mmWave radar point-cloud reader."""

    def __init__(self, port: str = "/dev/ttyACM0", baud: int = 115200):
        self._port = port
        self._baud = baud
        self._serial = None
        self._connected = False
        self._buffer: deque[np.ndarray] = deque(maxlen=200)
        self._points = np.zeros((0, 3), dtype=np.float32)
        self._velocities = np.zeros((0,), dtype=np.float32)
        self._frame_number = 0
        self._num_frames = 0

    # ── Connection ───────────────────────────────────────────────────────

    def connect(self, config: str | None = None) -> bool:
        """Connect to radar and optionally send a .cfg file.

        Args:
            config: Path to .cfg configuration file. If None, radar must
                    already be streaming.

        Returns True on success.
        """
        try:
            import serial
        except ImportError:
            raise ImportError("pyserial is required: pip install pyserial")

        try:
            self._serial = serial.Serial(self._port, self._baud, timeout=0.2)
            self._serial.reset_input_buffer()
            time.sleep(0.5)
            self._connected = True

            if config:
                self._send_config(config)

            return True
        except OSError as e:
            raise ConnectionError(f"Failed to open {self._port}: {e}")

    def disconnect(self) -> None:
        """Stop sensor and close serial."""
        if self._serial and self._connected:
            try:
                self._serial.write(b"sensorStop 0\n")
                self._serial.flush()
                time.sleep(0.1)
            except OSError:
                pass
        self._connected = False
        if self._serial:
            self._serial.close()
            self._serial = None

    # ── Config ───────────────────────────────────────────────────────────

    def _send_config(self, path: str) -> None:
        cfg = Path(path)
        if not cfg.exists():
            raise FileNotFoundError(f"Config not found: {path}")

        commands = [l.strip() for l in cfg.read_text("utf-8", errors="ignore").splitlines()
                    if l.strip() and not l.strip().startswith(("%", "#"))]

        start_cmd = None
        for cmd in commands:
            if cmd.startswith("sensorStart"):
                start_cmd = cmd
                continue
            if cmd.startswith("baudRate"):
                continue

            self._serial.write((cmd + "\n").encode("ascii"))
            self._serial.flush()
            reply = self._read_reply()

            # Non-ASCII reply = radar already streaming
            if reply and not all(32 <= ord(c) < 127 or c in '\n\r\t' for c in reply):
                self._serial.reset_input_buffer()
                return

            rl = reply.lower()
            if any(p in rl for p in CLI_FAILURE):
                name = cmd.split()[0]
                skippable = {
                    "cfarScndPassCfg", "compressionCfg", "antGeometryCfg",
                    "sigProcChainCfg", "aoaFovCfg", "rangeSelCfg",
                    "clutterRemoval", "compRangeBiasAndRxChanPhase",
                    "adcDataSource", "adcLogging", "factoryCalibCfg",
                }
                if name in skippable:
                    continue
                raise RuntimeError(f"CLI error: {cmd!r} → {reply.strip()}")

        if start_cmd is None:
            raise ValueError("Config has no sensorStart command")

        self._serial.reset_input_buffer()
        self._serial.write((start_cmd + "\n").encode("ascii"))
        self._serial.flush()

    def _read_reply(self, quiet_time: float = 0.15, max_time: float = 2.0) -> str:
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

    # ── Frame reading ────────────────────────────────────────────────────

    def read(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Read one TLV frame.

        Returns ``(points, velocities)`` where points is (N, 3) and
        velocities is (N,). Returns None on timeout or error.
        """
        if not self._connected or self._serial is None:
            return None

        try:
            # Sync to magic word
            window = bytearray()
            deadline = time.monotonic() + 2.0
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
                return None

            # Read header
            header = MAGIC_WORD + self._read_exact(32)
            _, total_len, _, frame_num, _, _, num_tlvs, _ = struct.unpack_from("<8I", header, 8)
            total_len &= 0xFFFFFFFF

            if not (40 <= total_len <= 2_000_000) or num_tlvs > 64:
                return None

            packet = header + self._read_exact(total_len - 40)
            self._parse_tlvs(packet, num_tlvs)
            self._frame_number = frame_num
            self._num_frames += 1

            if len(self._points) > 0:
                self._buffer.append(self._points.copy())

            return self._points.copy(), self._velocities.copy()
        except (TimeoutError, struct.error, ValueError):
            return None

    def _read_exact(self, count: int, timeout_s: float = 2.0) -> bytes:
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
        for header_size in (40, 52):
            offset = header_size
            pts, vels = [], []

            for _ in range(num_tlvs):
                if offset + 8 > len(packet):
                    break
                tlv_type, tlv_len = struct.unpack_from("<II", packet, offset)
                offset += 8

                if tlv_type == 0 or tlv_len > len(packet) - offset:
                    break
                payload = packet[offset:offset + tlv_len]
                offset += tlv_len

                if tlv_type == TLV_POINT_CLOUD_FLOAT:
                    p, v = self._decode_float(payload)
                    pts.extend(p)
                    vels.extend(v)
                elif tlv_type in TLV_POINT_CLOUD_FIXED:
                    p, v = self._decode_fixed(payload)
                    pts.extend(p)
                    vels.extend(v)

            if pts:
                self._points = np.array(pts, dtype=np.float32)
                self._velocities = np.array(vels, dtype=np.float32)
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

    # ── Streaming ────────────────────────────────────────────────────────

    def stream(self, max_frames: int = 0) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generator yielding (points, velocities) per frame.

        Args:
            max_frames: Stop after this many frames. 0 = infinite.
        """
        count = 0
        while self._connected:
            result = self.read()
            if result is not None:
                yield result
                count += 1
                if max_frames and count >= max_frames:
                    return

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        return self.stream()

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def points(self) -> np.ndarray:
        """Most recent point cloud: (N, 3)."""
        return self._points.copy()

    @property
    def velocities(self) -> np.ndarray:
        """Most recent velocities: (N,)."""
        return self._velocities.copy()

    @property
    def frame_number(self) -> int:
        return self._frame_number

    @property
    def num_frames(self) -> int:
        return self._num_frames
