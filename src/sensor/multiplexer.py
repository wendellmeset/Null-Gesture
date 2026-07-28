"""Thread-safe sensor multiplexer with timestamp-aligned circular buffers.

All four sensors run in independent threads, pushing into lock-free ring buffers.
The multiplexer provides a synchronous `read_frame()` that collects the latest
sample from each active sensor.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from Readers import IMUReader, MMWaveReader, RFIDReader, UWBReader


@dataclass
class SensorFrame:
    """A single timestamp-aligned snapshot across all active sensors."""

    timestamp: float = field(default_factory=time.time)

    # IMU: dict with ax, ay, az, gx, gy, gz, t
    imu: dict[str, float] | None = None

    # mmWave: (points (N,3), velocities (N,)) numpy arrays
    mmwave_points: np.ndarray | None = None
    mmwave_velocities: np.ndarray | None = None

    # UWB: dict with distance_m, addr
    uwb: dict[str, Any] | None = None

    # RFID: dict with epc, rssi, timestamp
    rfid: dict[str, Any] | None = None

    @property
    def active_sensors(self) -> list[str]:
        """Return names of sensors that contributed data in this frame."""
        active: list[str] = []
        if self.imu is not None:
            active.append("imu")
        if self.mmwave_points is not None:
            active.append("mmwave")
        if self.uwb is not None:
            active.append("uwb")
        if self.rfid is not None:
            active.append("rfid")
        return active


class RingBuffer:
    """Thread-safe, bounded ring buffer with overwrite-on-full behavior."""

    def __init__(self, capacity: int = 200) -> None:
        self._buf: deque[Any] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def push(self, item: Any) -> None:
        with self._lock:
            self._buf.append(item)

    def latest(self) -> Any:
        with self._lock:
            if self._buf:
                return self._buf[-1]
            return None

    def drain(self) -> list[Any]:
        with self._lock:
            items = list(self._buf)
            self._buf.clear()
            return items

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)


class SensorMultiplexer:
    """Manages threaded sensor readers and provides synchronized frame access.

    Usage::

        mux = SensorMultiplexer(
            imu_port="/dev/ttyUSB0",
            mmwave_port="/dev/ttyACM0",
            mmwave_config="Readers/configs/mmwave_hand_50cm.cfg",
            uwb_initiator="/dev/ttyACM1",
            uwb_responder="/dev/ttyACM2",
        )
        mux.start()
        for frame in mux:
            print(frame.active_sensors)
        mux.stop()
    """

    def __init__(
        self,
        # IMU
        imu_port: str | None = None,
        # mmWave
        mmwave_port: str | None = None,
        mmwave_config: str | None = None,
        # UWB
        uwb_initiator: str | None = None,
        uwb_responder: str | None = None,
        # RFID
        rfid_port: str | None = None,
        # Buffer
        buffer_capacity: int = 200,
    ) -> None:
        self._imu_port = imu_port
        self._mmwave_port = mmwave_port
        self._mmwave_config = mmwave_config
        self._uwb_initiator = uwb_initiator
        self._uwb_responder = uwb_responder
        self._rfid_port = rfid_port

        self._buffers: dict[str, RingBuffer] = {}
        self._readers: dict[str, Any] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._stop_event = threading.Event()
        self._running = False

        self._buffer_capacity = buffer_capacity

    # ── Auto-detection helpers ──────────────────────────────────────────

    @staticmethod
    def _list_usb_ports() -> list[tuple[str, int, int, str]]:
        """Return [(device, vid, pid, description), ...] for USB serial ports."""
        try:
            import serial.tools.list_ports
        except ImportError:
            return []
        results: list[tuple[str, int, int, str]] = []
        for info in serial.tools.list_ports.comports():
            if info.device and info.vid is not None:
                results.append((info.device, info.vid, info.pid, info.description or ""))
        return results

    @staticmethod
    def _probe_imu_port(candidate: str) -> bool:
        """Test if a port produces valid IMU data (fast — stops on first valid sample)."""
        try:
            import time as _time
            from Readers.imu import IMUReader as _IMU
            reader = _IMU(candidate)
            if not reader.connect():
                return False
            # Read up to 1 second — but stop on first valid sample
            deadline = _time.time() + 1.0
            while _time.time() < deadline:
                sample = reader.read()
                if sample and all(k in sample for k in ("ax", "ay", "az")):
                    reader.disconnect()
                    return True
                _time.sleep(0.02)
            reader.disconnect()
        except Exception:
            pass
        return False

    @staticmethod
    def _probe_mmwave_port(candidate: str) -> bool:
        """Test if a port has mmWave radar by attempting a lightweight connection."""
        try:
            import serial, time as _time
            # Open without DTR/RTS toggle to avoid resetting the radar
            ser = serial.Serial(candidate, 115200, timeout=0.5)
            ser.dtr = False
            ser.rts = False
            _time.sleep(0.5)
            # Send a newline to wake CLI if in config mode
            ser.write(b"\n")
            _time.sleep(0.3)
            # Read whatever is available
            ser.timeout = 0.3
            data = b""
            for _ in range(5):
                chunk = ser.read(512)
                if chunk:
                    data += chunk
                else:
                    break
            ser.close()
            if not data:
                return False
            MAGIC = b"\x02\x01\x04\x03\x06\x05\x08\x07"
            if MAGIC in data:
                return True
            text = data.decode("utf-8", errors="ignore").lower()
            if any(kw in text for kw in ("tlv", "frame time", "mmwdemo", "mmwave", "iwrl", "xwr", "pointcloud")):
                return True
            # If we received substantial binary data, it's likely a radar
            return len(data) > 200
        except Exception:
            return False

    @staticmethod
    def _find_uwb_ports() -> list[str]:
        """Find UWB DWM3001CDK boards by Nordic nRF52 VID:PID."""
        ports = SensorMultiplexer._list_usb_ports()
        return sorted(
            dev for dev, vid, pid, _desc in ports
            if vid == 0x1915 and pid in (0x520F, 0x521F)
        )


    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> dict[str, bool]:
        """Start all configured sensor reader threads.

        Ports explicitly provided → use those.
        Ports left as None → auto-detect from available USB devices.

        Returns a dict of sensor_name -> connected status.
        """
        status: dict[str, bool] = {}
        self._running = False

        # ── UWB (auto-detect nRF52 boards) ──────────────────────────
        initiator = self._uwb_initiator
        responder = self._uwb_responder
        if not initiator or not responder:
            uwb_ports = self._find_uwb_ports()
            if len(uwb_ports) >= 2:
                if not responder:
                    responder = uwb_ports[0]
                if not initiator:
                    initiator = uwb_ports[1]

        if initiator and responder:
            try:
                uwb = UWBReader()
                if uwb.connect(initiator=initiator, responder=responder):
                    self._readers["uwb"] = uwb
                    self._buffers["uwb"] = RingBuffer(self._buffer_capacity)
                    self._threads["uwb"] = threading.Thread(
                        target=self._uwb_worker, name="uwb-reader", daemon=True
                    )
                    status["uwb"] = True
                else:
                    status["uwb"] = False
            except Exception:
                status["uwb"] = False

        # ── IMU (auto-detect by probing all non-UWB ports) ──────────
        imu_port = self._imu_port
        if not imu_port:
            claimed = {initiator, responder}
            all_ports = self._list_usb_ports()
            # Prefer /dev/ttyACM* for IMU (ESP32 typically on ACM)
            candidates = sorted(
                [dev for dev, _v, _p, _d in all_ports if dev not in claimed],
                key=lambda d: (0 if 'ACM' in d else 1, d)
            )
            for c in candidates:
                if self._probe_imu_port(c):
                    imu_port = c
                    break

        if imu_port:
            try:
                imu = IMUReader(imu_port)
                if imu.connect():
                    self._readers["imu"] = imu
                    self._buffers["imu"] = RingBuffer(self._buffer_capacity)
                    self._threads["imu"] = threading.Thread(
                        target=self._imu_worker, name="imu-reader", daemon=True
                    )
                    status["imu"] = True
                else:
                    status["imu"] = False
            except Exception:
                status["imu"] = False

        # ── mmWave (auto-detect by probing all non-claimed ports) ────
        mmwave_port = self._mmwave_port
        if not mmwave_port:
            claimed = {initiator, responder, imu_port}
            all_ports = self._list_usb_ports()
            # Prefer /dev/ttyUSB* for mmWave (radar typically on USB-UART)
            candidates = sorted(
                [dev for dev, _v, _p, _d in all_ports if dev not in claimed],
                key=lambda d: (0 if 'USB' in d else 1, d)
            )
            for c in candidates:
                if self._probe_mmwave_port(c):
                    mmwave_port = c
                    break

        if mmwave_port:
            try:
                radar = MMWaveReader(mmwave_port)
                radar.connect(config=self._mmwave_config)
                self._readers["mmwave"] = radar
                self._buffers["mmwave"] = RingBuffer(self._buffer_capacity)
                self._threads["mmwave"] = threading.Thread(
                    target=self._mmwave_worker, name="mmwave-reader", daemon=True
                )
                status["mmwave"] = True
            except Exception:
                status["mmwave"] = False

        # ── RFID ────────────────────────────────────────────────────
        if self._rfid_port:
            try:
                rfid = RFIDReader(port=self._rfid_port)
                if rfid.connect():
                    self._readers["rfid"] = rfid
                    self._buffers["rfid"] = RingBuffer(self._buffer_capacity)
                    self._threads["rfid"] = threading.Thread(
                        target=self._rfid_worker, name="rfid-reader", daemon=True
                    )
                    status["rfid"] = True
                else:
                    status["rfid"] = False
            except Exception:
                status["rfid"] = False
        elif self._rfid_port is None:
            try:
                rfid = RFIDReader()
                if rfid.connect():
                    self._readers["rfid"] = rfid
                    self._buffers["rfid"] = RingBuffer(self._buffer_capacity)
                    self._threads["rfid"] = threading.Thread(
                        target=self._rfid_worker, name="rfid-reader", daemon=True
                    )
                    status["rfid"] = True
                else:
                    status["rfid"] = False
            except Exception:
                status["rfid"] = False

        self._running = True

        # Start threads
        for thread in self._threads.values():
            thread.start()

        return status

    def stop(self) -> None:
        """Signal all reader threads to stop and disconnect sensors."""
        self._stop_event.set()
        self._running = False

        for thread in self._threads.values():
            thread.join(timeout=2.0)

        for reader in self._readers.values():
            try:
                reader.disconnect()
            except OSError:
                pass

        self._readers.clear()
        self._threads.clear()

    # ── Frame access ─────────────────────────────────────────────────────

    def read_frame(self) -> SensorFrame:
        """Return the latest synchronized snapshot from all active sensors."""
        frame = SensorFrame()

        if "imu" in self._buffers:
            frame.imu = self._buffers["imu"].latest()

        if "mmwave" in self._buffers:
            latest = self._buffers["mmwave"].latest()
            if latest is not None:
                frame.mmwave_points = latest[0]
                frame.mmwave_velocities = latest[1]

        if "uwb" in self._buffers:
            frame.uwb = self._buffers["uwb"].latest()

        if "rfid" in self._buffers:
            frame.rfid = self._buffers["rfid"].latest()

        return frame

    def __iter__(self):
        """Generator yielding SensorFrame at ~100 Hz (IMU-driven)."""
        if not self._running:
            self.start()

        while self._running and not self._stop_event.is_set():
            yield self.read_frame()
            time.sleep(0.01)  # ~100 Hz

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def running(self) -> bool:
        return self._running

    @property
    def active_sensors(self) -> list[str]:
        return list(self._buffers.keys())

    # ── Workers ──────────────────────────────────────────────────────────

    def _imu_worker(self) -> None:
        reader = self._readers["imu"]
        buf = self._buffers["imu"]
        try:
            for sample in reader:
                if self._stop_event.is_set():
                    break
                buf.push(sample)
        except Exception:
            pass

    def _mmwave_worker(self) -> None:
        reader = self._readers["mmwave"]
        buf = self._buffers["mmwave"]
        try:
            for points, velocities in reader:
                if self._stop_event.is_set():
                    break
                buf.push((points, velocities))
        except Exception:
            pass

    def _uwb_worker(self) -> None:
        reader = self._readers["uwb"]
        buf = self._buffers["uwb"]
        try:
            for sample in reader:
                if self._stop_event.is_set():
                    break
                buf.push(sample)
        except Exception:
            pass

    def _rfid_worker(self) -> None:
        reader = self._readers["rfid"]
        buf = self._buffers["rfid"]
        try:
            for tag in reader:
                if self._stop_event.is_set():
                    break
                buf.push(tag)
        except Exception:
            pass
