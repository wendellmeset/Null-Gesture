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

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> dict[str, bool]:
        """Start all configured sensor reader threads.

        Returns a dict of sensor_name -> connected status.
        """
        status: dict[str, bool] = {}

        # IMU
        if self._imu_port:
            try:
                imu = IMUReader(self._imu_port)
                if imu.connect():
                    self._readers["imu"] = imu
                    self._buffers["imu"] = RingBuffer(self._buffer_capacity)
                    self._threads["imu"] = threading.Thread(
                        target=self._imu_worker, name="imu-reader", daemon=True
                    )
                    status["imu"] = True
                else:
                    status["imu"] = False
            except OSError:
                status["imu"] = False

        # mmWave
        if self._mmwave_port:
            try:
                radar = MMWaveReader(self._mmwave_port)
                radar.connect(config=self._mmwave_config)
                self._readers["mmwave"] = radar
                self._buffers["mmwave"] = RingBuffer(self._buffer_capacity)
                self._threads["mmwave"] = threading.Thread(
                    target=self._mmwave_worker, name="mmwave-reader", daemon=True
                )
                status["mmwave"] = True
            except OSError:
                status["mmwave"] = False

        # UWB (dual-board mode)
        if self._uwb_initiator and self._uwb_responder:
            try:
                uwb = UWBReader(
                    initiator_port=self._uwb_initiator,
                    responder_port=self._uwb_responder,
                )
                if uwb.connect():
                    self._readers["uwb"] = uwb
                    self._buffers["uwb"] = RingBuffer(self._buffer_capacity)
                    self._threads["uwb"] = threading.Thread(
                        target=self._uwb_worker, name="uwb-reader", daemon=True
                    )
                    status["uwb"] = True
                else:
                    status["uwb"] = False
            except OSError:
                status["uwb"] = False

        # RFID
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
            except OSError:
                status["rfid"] = False
        elif self._rfid_port is None:
            # Try auto-detect
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
            except OSError:
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
        for sample in reader:
            if self._stop_event.is_set():
                break
            buf.push(sample)

    def _mmwave_worker(self) -> None:
        reader = self._readers["mmwave"]
        buf = self._buffers["mmwave"]
        for points, velocities in reader:
            if self._stop_event.is_set():
                break
            buf.push((points, velocities))

    def _uwb_worker(self) -> None:
        reader = self._readers["uwb"]
        buf = self._buffers["uwb"]
        for sample in reader:
            if self._stop_event.is_set():
                break
            buf.push(sample)

    def _rfid_worker(self) -> None:
        reader = self._readers["rfid"]
        buf = self._buffers["rfid"]
        for tag in reader:
            if self._stop_event.is_set():
                break
            buf.push(tag)
