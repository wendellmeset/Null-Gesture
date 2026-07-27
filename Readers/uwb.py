"""UWB Reader: DWM3001CDK FiRa TWR ranging via direct serial.

Requires two DWM3001CDK boards connected over USB (Responder + Initiator).
Uses the native AT-style firmware interface — no Qorvo UCI library needed.

Usage::

    from Readers import UWBReader

    uwb = UWBReader()
    uwb.connect(port="/dev/ttyACM0")     # single board mode
    dist = uwb.read()                    # {'distance_m': 1.23, 'addr': '...'}
    uwb.disconnect()

Dual-board mode (Initiator + Responder)::

    uwb = UWBReader()
    uwb.connect(
        initiator="/dev/ttyACM1",
        responder="/dev/ttyACM0",
    )
    for sample in uwb:
        print(f"{sample['distance_m']:.2f} m")
"""

from __future__ import annotations

import json
import re
import threading
import time
from typing import Iterator

_JSON_RE = re.compile(r"\[\s*\{.*?\}\s*\]", re.DOTALL)
_BAUD = 115200

# Default native firmware commands
_CMD_RESPONDER = "RESPF 9 2400 200 25 2 42 01:02:03:04:05:06:07:08 2 0 0 1"
_CMD_INITIATOR = "INITF 9 2400 200 25 2 42 01:02:03:04:05:06:07:08 1 0 0 1 2"


class UWBReader:
    """Minimal DWM3001CDK UWB ranging reader.

    Supports single-board (passive listen) and dual-board (active ranging) modes.
    """

    def __init__(
        self,
        port: str | None = None,
        initiator_port: str | None = None,
        responder_port: str | None = None,
    ):
        """
        Args:
            port: Single port for passive listening (reads whatever the board emits).
            initiator_port: Initiator board port (sends INITF).
            responder_port: Responder board port (sends RESPF).
                If both are provided, dual-board active ranging is used.
        """
        self._port = port
        self._initiator_port = initiator_port
        self._responder_port = responder_port
        self._ser = None
        self._ser_initiator = None
        self._ser_responder = None
        self._connected = False
        self._last_sample: dict | None = None
        self._running = False

    # ── Connection ───────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Open serial connection(s) and start ranging."""
        try:
            import serial
        except ImportError:
            raise ImportError("pyserial is required: pip install pyserial")

        # Dual-board mode
        if self._initiator_port and self._responder_port:
            return self._connect_dual(serial)

        # Single-board mode
        if self._port:
            return self._connect_single(serial)

        raise ValueError("Provide port= or initiator_port + responder_port.")

    def _connect_single(self, serial) -> bool:
        self._ser = serial.Serial(self._port, _BAUD, timeout=0.2)
        time.sleep(0.5)
        self._ser.reset_input_buffer()
        self._connected = True
        return True

    def _connect_dual(self, serial) -> bool:
        self._ser_responder = serial.Serial(self._responder_port, _BAUD, timeout=0.2)
        self._ser_initiator = serial.Serial(self._initiator_port, _BAUD, timeout=0.2)

        # Configure responder
        time.sleep(1.0)
        self._ser_responder.reset_input_buffer()
        self._ser_responder.write(b"\r\n")
        time.sleep(0.3)
        self._ser_responder.write(b"quit\r\n")
        time.sleep(0.3)
        self._ser_responder.reset_input_buffer()
        self._ser_responder.write(f"{_CMD_RESPONDER}\r\n".encode())
        time.sleep(0.5)

        # Configure initiator
        self._ser_initiator.reset_input_buffer()
        self._ser_initiator.write(b"\r\n")
        time.sleep(0.3)
        self._ser_initiator.write(b"quit\r\n")
        time.sleep(0.3)
        self._ser_initiator.reset_input_buffer()
        self._ser_initiator.write(f"{_CMD_INITIATOR}\r\n".encode())
        time.sleep(0.5)

        self._connected = True
        return True

    def disconnect(self) -> None:
        """Close all serial connections."""
        self._connected = False
        for s in [self._ser, self._ser_initiator, self._ser_responder]:
            if s:
                try:
                    s.close()
                except Exception:
                    pass
        self._ser = self._ser_initiator = self._ser_responder = None

    # ── Reading ──────────────────────────────────────────────────────────

    def read(self) -> dict | None:
        """Read one distance sample.

        Returns ``{'distance_m': float, 'addr': str}`` or None.
        """
        if not self._connected:
            return None

        # Dual-board: read from initiator
        ser = self._ser_initiator or self._ser
        if ser is None:
            return None

        try:
            raw = b""
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                chunk = ser.read(ser.in_waiting or 1)
                if not chunk:
                    continue
                raw += chunk
                m = _JSON_RE.search(raw.decode("utf-8", errors="ignore"))
                if m:
                    data = json.loads(m.group(0))
                    for node in data:
                        dist_cm = node.get("D_cm")
                        if dist_cm is not None:
                            sample = {
                                "distance_m": dist_cm / 100.0,
                                "addr": node.get("Addr", "unknown"),
                            }
                            self._last_sample = sample
                            return sample
        except Exception:
            pass
        return None

    def stream(self) -> Iterator[dict]:
        """Generator yielding distance samples continuously."""
        if not self._connected:
            return
        self._running = True
        buf = ""
        ser = self._ser_initiator or self._ser
        if ser is None:
            return

        try:
            while self._running and self._connected:
                chunk = ser.read(ser.in_waiting or 1).decode("utf-8", errors="ignore")
                if not chunk:
                    time.sleep(0.01)
                    continue
                buf += chunk
                # Keep buffer bounded
                if len(buf) > 4096:
                    buf = buf[-2048:]

                m = _JSON_RE.search(buf)
                if m:
                    try:
                        data = json.loads(m.group(0))
                        for node in data:
                            dist_cm = node.get("D_cm")
                            if dist_cm is not None:
                                sample = {
                                    "distance_m": dist_cm / 100.0,
                                    "addr": node.get("Addr", "unknown"),
                                }
                                self._last_sample = sample
                                yield sample
                    except json.JSONDecodeError:
                        pass
                    buf = buf[m.end():]
        finally:
            self._running = False

    def __iter__(self) -> Iterator[dict]:
        return self.stream()

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def last_sample(self) -> dict | None:
        return self._last_sample
