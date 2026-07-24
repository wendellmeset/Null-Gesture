"""RFID sensor: M7E Hecto UHF reader wrapper via python-mercuryapi."""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any, cast

import numpy as np

from null_gesture.config import RFIDConfig, rfid_config as default_rfid_config

logger = logging.getLogger("null_gesture.sensors.rfid")

# Lazy import — mercuryapi may not be installed
_mercury: Any = None
_MercuryReader: Any = None


def _ensure_mercury() -> None:
    global _mercury, _MercuryReader
    if _mercury is not None:
        return
    try:
        import mercury as _m

        _mercury = _m
        _MercuryReader = cast(Any, _m).Reader
        logger.debug("mercuryapi loaded")
    except ImportError:
        raise ImportError(
            "python-mercuryapi not installed. Run: bash install_mercury.sh"
        ) from None


class RFIDReader:
    """Wraps the M7E Hecto RFID reader for continuous tag monitoring."""

    def __init__(self, config: RFIDConfig | None = None) -> None:
        self.config = config or default_rfid_config
        self._reader: Any = None
        self._buffer: deque[tuple[float, float, float]] = deque()  # (ts, rssi, phase)
        self._tag_count = 0
        self._connected = False
        self._target_epcs: set[str] = set()

    def connect(self, port: str, target_epcs: list[str] | None = None) -> bool:
        """Connect to reader on the given serial port."""
        try:
            _ensure_mercury()
        except ImportError as exc:
            logger.error("RFID connect failed: %s", exc)
            return False
        try:
            self._reader = _MercuryReader(f"tmr://{port}", baudrate=115200)
            self._connected = True
            self._reader.set_region(self.config.region)
            self._reader.set_read_plan([1], "GEN2")
            if hasattr(self._reader, "set_read_power"):
                self._reader.set_read_power(self.config.read_power)
            model = self._reader.get_model()
            version = self._reader.get_software_version()
            logger.info("RFID reader connected: %s (FW %s)", model, version)
            if target_epcs:
                self._target_epcs = {epc.upper() for epc in target_epcs}
            return True
        except Exception as exc:
            logger.error("RFID connection failed: %s", exc)
            self._connected = False
            return False

    def disconnect(self) -> None:
        if self._reader:
            try:
                self._reader.stop_reading()
            except Exception:
                pass
            self._reader = None
        self._connected = False
        logger.info("RFID reader disconnected (%d tags)", self._tag_count)

    def start_streaming(self, on_time_ms: int | None = None) -> None:
        """Begin asynchronous tag reading with callback."""
        if not self._reader:
            raise RuntimeError("Not connected")
        ot = on_time_ms if on_time_ms is not None else self.config.on_time_ms
        off = self.config.off_time_ms
        self._reader.start_reading(
            callback=self._on_tag, on_time=ot, off_time=off
        )

    def _on_tag(self, tag: Any) -> None:
        """Internal callback for each tag read."""
        epc = tag.epc.hex().upper() if hasattr(tag.epc, "hex") else str(tag.epc)
        # Filter by target EPCs if specified
        if self._target_epcs and epc not in self._target_epcs:
            return
        rssi = float(tag.rssi)
        phase = float(getattr(tag, "phase", 0.0) or 0.0)
        ts = time.time()
        self._buffer.append((ts, rssi, phase))
        self._tag_count += 1
        self._prune()

    def _prune(self) -> None:
        cutoff = time.time() - self.config.window_seconds * 2
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()

    def get_window(self) -> np.ndarray:
        """Return (timesteps, 2) window: [rssi, phase]."""
        timesteps = self.config.timesteps
        now = time.time()
        window_s = self.config.window_seconds
        window_data = [
            (rssi, phase)
            for ts, rssi, phase in self._buffer
            if now - ts <= window_s
        ]
        if len(window_data) < 2:
            return np.zeros((timesteps, 2), dtype=np.float32)
        arr = np.array(window_data[-timesteps:], dtype=np.float32)
        if arr.shape[0] < timesteps:
            padded = np.zeros((timesteps, 2), dtype=np.float32)
            padded[-arr.shape[0] :] = arr
            return padded
        if arr.shape[0] > timesteps:
            indices = np.linspace(0, arr.shape[0] - 1, timesteps).astype(int)
            return arr[indices]
        return arr

    @staticmethod
    def find_port() -> str | None:
        """Auto-detect the M7E reader serial port by probing candidates."""
        _ensure_mercury()
        import glob
        import sys

        candidates: list[str] = []
        # Try known USB-serial devices
        try:
            import serial.tools.list_ports
        except ImportError:
            serial = None
        else:
            serial = sys.modules.get("serial")

        if serial is not None:
            known_vid_pid = {
                (0x10C4, 0xEA60),  # CP210x
                (0x1A86, 0x7523),  # CH340
                (0x0403, 0x6015),  # FT231X
                (0x0403, 0x6001),  # FT232R
            }
            for pi in serial.tools.list_ports.comports():
                if pi.device and (pi.vid, pi.pid) in known_vid_pid:
                    candidates.append(pi.device)

        if not candidates:
            for pat in ["/dev/ttyUSB*", "/dev/ttyACM*", "/dev/cu.usbserial*"]:
                candidates.extend(sorted(glob.glob(pat)))

        for port in candidates:
            try:
                reader = _MercuryReader(f"tmr://{port}", baudrate=115200)
                model = reader.get_model()
                reader.stop_reading()
                logger.info("RFID reader found on %s: %s", port, model)
                return port
            except Exception:
                continue

        return None

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def tag_count(self) -> int:
        return self._tag_count
