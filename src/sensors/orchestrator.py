"""Multi-threaded sensor orchestrator.

Manages concurrent data acquisition from IMU and mmWave sensors,
feeding thread-safe ring buffers for consumption by the main pipeline.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np

from Readers import IMUReader, MMWaveReader
from src.sensors.buffer import IMUBuffer, MMWaveBuffer

_log = logging.getLogger(__name__)


class SensorOrchestrator:
    """Manages concurrent sensor streams with thread-safe buffering.

    Starts background daemon threads for IMU and mmWave data acquisition.
    The main pipeline thread reads from the buffers at its own pace.

    Usage::

        orch = SensorOrchestrator(
            imu_port="/dev/ttyUSB0",
            mmwave_port="/dev/ttyACM0",
            mmwave_config="Readers/configs/mmwave_hand_50cm.cfg",
        )
        orch.start()
        # ... main pipeline loop reads from orch.imu_buffer and orch.mmwave_buffer
        orch.stop()
    """

    def __init__(
        self,
        imu_port: str = "/dev/ttyUSB0",
        imu_baud: int = 115200,
        mmwave_port: str = "/dev/ttyACM0",
        mmwave_baud: int = 115200,
        mmwave_config: str | None = "Readers/configs/mmwave_hand_50cm.cfg",
        imu_buffer_sec: float = 2.0,
        mmwave_buffer_sec: float = 2.0,
        imu_rate_hz: float = 100.0,
        mmwave_rate_hz: float = 20.0,
    ):
        """
        Args:
            imu_port: Serial port for the ESP32 IMU.
            imu_baud: Baud rate for IMU serial.
            mmwave_port: Serial port for mmWave radar.
            mmwave_baud: Baud rate for mmWave serial.
            mmwave_config: Path to radar .cfg file (None = skip config).
            imu_buffer_sec: Ring buffer duration in seconds.
            mmwave_buffer_sec: Ring buffer duration in seconds.
            imu_rate_hz: Expected IMU sample rate (for buffer sizing).
            mmwave_rate_hz: Expected mmWave frame rate (for buffer sizing).
        """
        # Readers
        self._imu_reader = IMUReader(port=imu_port, baud=imu_baud)
        self._mmwave_reader = MMWaveReader(port=mmwave_port, baud=mmwave_baud)

        # Config
        self._mmwave_config = mmwave_config

        # Buffers
        self.imu_buffer = IMUBuffer(
            maxlen=int(imu_buffer_sec * imu_rate_hz * 1.5),
            name="imu",
        )
        self.mmwave_buffer = MMWaveBuffer(
            maxlen=int(mmwave_buffer_sec * mmwave_rate_hz * 1.5),
            name="mmwave",
        )

        # Threading
        self._imu_thread: threading.Thread | None = None
        self._mmwave_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._running = False

        # Stats
        self._imu_sample_count = 0
        self._mmwave_frame_count = 0
        self._start_time: float | None = None

    # ── Lifecycle ────────────────────────────────────────────────────────

    def start(self) -> bool:
        """Connect sensors and start background acquisition threads.

        Returns:
            True if both sensors connected successfully.
        """
        _log.info("Starting sensor orchestrator...")

        # Connect IMU
        try:
            _log.info("Connecting IMU on %s...", self._imu_reader._port)
            self._imu_reader.connect()
            _log.info("IMU connected.")
        except Exception as e:
            _log.error("IMU connection failed: %s", e)
            return False

        # Connect mmWave
        try:
            _log.info("Connecting mmWave on %s...", self._mmwave_reader._port)
            self._mmwave_reader.connect(config=self._mmwave_config)
            _log.info("mmWave connected.")
        except Exception as e:
            _log.error("mmWave connection failed: %s", e)
            try:
                self._imu_reader.disconnect()
            except Exception:
                pass
            return False

        # Start threads
        self._stop_event.clear()
        self._running = True
        self._start_time = time.time()

        self._imu_thread = threading.Thread(
            target=self._imu_loop, name="imu-stream", daemon=True
        )
        self._mmwave_thread = threading.Thread(
            target=self._mmwave_loop, name="mmwave-stream", daemon=True
        )
        self._imu_thread.start()
        self._mmwave_thread.start()

        _log.info("Sensor orchestrator running.")
        return True

    def stop(self) -> None:
        """Stop acquisition threads and disconnect sensors."""
        _log.info("Stopping sensor orchestrator...")
        self._running = False
        self._stop_event.set()

        # Wait for threads
        for thread in (self._imu_thread, self._mmwave_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)

        # Disconnect
        try:
            self._imu_reader.disconnect()
        except Exception as e:
            _log.warning("IMU disconnect error: %s", e)

        try:
            self._mmwave_reader.disconnect()
        except Exception as e:
            _log.warning("mmWave disconnect error: %s", e)

        _log.info(
            "Sensor orchestrator stopped. IMU: %d samples, mmWave: %d frames",
            self._imu_sample_count,
            self._mmwave_frame_count,
        )

    # ── Sensor Loops ─────────────────────────────────────────────────────

    def _imu_loop(self) -> None:
        """Background loop: read IMU samples → buffer."""
        try:
            for sample in self._imu_reader.stream():
                if self._stop_event.is_set():
                    break
                t = sample.get("t", time.time())
                self.imu_buffer.append(sample, timestamp=t)
                self._imu_sample_count += 1
        except Exception as e:
            _log.error("IMU stream error: %s", e)
        _log.debug("IMU loop exiting.")

    def _mmwave_loop(self) -> None:
        """Background loop: read mmWave frames → buffer."""
        try:
            for points, velocities in self._mmwave_reader.stream():
                if self._stop_event.is_set():
                    break
                # Store copies to avoid mutation
                self.mmwave_buffer.append(
                    (points.copy(), velocities.copy()),
                    timestamp=time.time(),
                )
                self._mmwave_frame_count += 1
        except Exception as e:
            _log.error("mmWave stream error: %s", e)
        _log.debug("mmWave loop exiting.")

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def running(self) -> bool:
        return self._running

    @property
    def elapsed_sec(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    @property
    def stats(self) -> dict[str, Any]:
        return {
            "running": self._running,
            "elapsed_sec": self.elapsed_sec,
            "imu_samples": self._imu_sample_count,
            "imu_buffer_count": self.imu_buffer.count,
            "mmwave_frames": self._mmwave_frame_count,
            "mmwave_buffer_count": self.mmwave_buffer.count,
        }
