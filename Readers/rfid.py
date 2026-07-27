"""RFID Reader: M7E Hecto UHF RFID via python-mercuryapi.

Requires the Mercury API C library from Novanta/Jadak.

Install:
  python Readers/mercury/install.py               # cross-platform
  bash Readers/mercury/install.sh                 # Linux / macOS

Usage::

    from Readers import RFIDReader

    rfid = RFIDReader()
    rfid.connect()                     # auto-detect port
    for tag in rfid.stream():          # generator yielding dicts
        print(tag["epc"], tag["rssi"])
    rfid.disconnect()
"""

from __future__ import annotations

import glob
import time
from pathlib import Path
from typing import Any, Iterator

try:
    import serial.tools.list_ports
    _has_serial_tools = True
except ImportError:
    _has_serial_tools = False

KNOWN_VID_PID = {
    (0x10C4, 0xEA60),  # Silicon Labs CP210x
    (0x1A86, 0x7523),  # CH340
    (0x0403, 0x6015),  # FTDI FT231X
    (0x0403, 0x6001),  # FTDI FT232R
}


class RFIDReader:
    """Minimal M7E Hecto UHF RFID reader interface."""

    def __init__(self, port: str | None = None, baud: int = 115200):
        """
        Args:
            port: Serial port (e.g. /dev/ttyUSB0). Auto-detects if None.
            baud: Baud rate.
        """
        self._port = port
        self._baud = baud
        self._reader: Any = None
        self._connected = False
        self._last_tag: dict[str, Any] | None = None
        self._tag_queue: list[dict[str, Any]] = []

    # ── Connection ───────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Connect to the reader. Auto-detects port if not specified."""
        try:
            import mercury  # noqa: F811
        except ImportError:
            raise ImportError(
                "python-mercuryapi not installed.\n"
                "Run: python Readers/mercury/install.py"
            )

        port = self._port or _find_m7e_port(self._baud)
        if port is None:
            return False

        try:
            self._reader = mercury.Reader(f"tmr://{port}", baudrate=self._baud)
        except TypeError as exc:
            if "Streaming" in str(exc):
                _stop_streaming(port)
                self._reader = mercury.Reader(f"tmr://{port}", baudrate=self._baud)
            else:
                raise

        self._reader.set_region("NA")
        self._reader.set_read_plan([1], "GEN2")
        self._connected = True
        return True

    def disconnect(self) -> None:
        """Stop reading and close connection."""
        if self._reader:
            try:
                self._reader.stop_reading()
            except Exception:
                pass
            self._reader = None
        self._connected = False

    # ── Reading ──────────────────────────────────────────────────────────

    def read(self) -> dict[str, Any] | None:
        """Read a single tag inventory. Returns dict or None.

        Return keys: ``epc`` (hex str), ``rssi`` (float), ``timestamp`` (float).
        """
        if not self._connected or self._reader is None:
            return None
        try:
            tags = self._reader.read(timeout=500)
        except Exception:
            return None
        if not tags:
            return None
        tag = tags[0]
        result = {
            "epc": tag.epc.hex().upper(),
            "rssi": float(tag.rssi),
            "timestamp": time.time(),
        }
        self._last_tag = result
        return result

    def stream(self, on_time_ms: int = 500, off_time_ms: int = 0) -> Iterator[dict[str, Any]]:
        """Generator that yields tags continuously via async callback.

        Args:
            on_time_ms: How long the radio is on per cycle.
            off_time_ms: How long the radio is off per cycle.
        """
        if not self._connected or self._reader is None:
            return

        def _on_tag(tag: Any) -> None:
            self._tag_queue.append({
                "epc": tag.epc.hex().upper(),
                "rssi": float(tag.rssi),
                "timestamp": time.time(),
            })

        self._reader.start_reading(callback=_on_tag, on_time=on_time_ms, off_time=off_time_ms)
        try:
            while self._connected:
                while self._tag_queue:
                    yield self._tag_queue.pop(0)
                time.sleep(0.01)
        finally:
            try:
                self._reader.stop_reading()
            except Exception:
                pass

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return self.stream()

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def last_tag(self) -> dict[str, Any] | None:
        return self._last_tag


# ── Helpers ───────────────────────────────────────────────────────────────

def _stop_streaming(port: str) -> None:
    """Send raw stop command to break streaming mode."""
    try:
        import serial as _serial
        ser = _serial.Serial(port, 115200, timeout=1)
        ser.dtr = False
        ser.rts = False
        time.sleep(0.5)
        ser.reset_input_buffer()
        ser.write(b'\xff\x03\x2f\x00\x00\x02\x5e\x86')
        ser.flush()
        time.sleep(2)
        ser.close()
    except Exception:
        pass


def _find_m7e_port(baud: int = 115200) -> str | None:
    """Probe serial ports for an M7E reader."""
    import mercury

    candidates: list[str] = []

    if _has_serial_tools:
        for info in serial.tools.list_ports.comports():
            if info.device and (info.vid, info.pid) in KNOWN_VID_PID:
                candidates.append(info.device)

    if not candidates:
        for pat in ["/dev/ttyUSB*", "/dev/ttyACM*", "/dev/cu.usbserial*"]:
            candidates.extend(sorted(glob.glob(pat)))

    for port in candidates:
        try:
            reader = mercury.Reader(f"tmr://{port}", baudrate=baud)
            reader.get_model()
            reader.stop_reading()
            return port
        except TypeError as exc:
            if "Streaming" in str(exc):
                _stop_streaming(port)
                try:
                    reader = mercury.Reader(f"tmr://{port}", baudrate=baud)
                    reader.get_model()
                    reader.stop_reading()
                    return port
                except Exception:
                    continue
        except Exception:
            continue
    return None
