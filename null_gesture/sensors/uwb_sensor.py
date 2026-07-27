"""UWB sensor: FiRa two-way ranging via DWM3001CDK boards.

Wraps the external uwb-qorvo-tools library to measure inter-hand distance.
Requires uwb-qorvo-tools/ cloned into the project root (see README).
"""

from __future__ import annotations

import logging
import os
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from null_gesture.config import UWBConfig, uwb_config as default_uwb_config

logger = logging.getLogger("null_gesture.sensors.uwb")

UWB_TOOLS_ROOT = Path(__file__).resolve().parent.parent.parent / "uwb-qorvo-tools"

UWB_CHANNELS = ("distance_cm",)


def _uwb_tools_available() -> bool:
    """Check if uwb-qorvo-tools is importable."""
    uci_path = UWB_TOOLS_ROOT / "lib" / "uwb-uci"
    return uci_path.is_dir()


class UWBSensor:
    """DWM3001CDK UWB sensor — measures inter-hand distance via FiRa TWR.

    Uses a background thread to continuously range and buffer distance
    samples. Provides the same ingest/get_window interface as IMUClient.
    """

    def __init__(self, config: UWBConfig | None = None) -> None:
        self.config = config or default_uwb_config
        self._connected = False
        self._sample_count = 0
        self._buffer: deque[tuple[float, np.ndarray]] = deque()
        self._ranging_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._latest_distance: float = 0.0
        self._lock = threading.Lock()

    def connect(self) -> bool:
        """Initialize both DWM3001CDK boards and start ranging.

        Returns True on success.
        """
        if not _uwb_tools_available():
            logger.error(
                "uwb-qorvo-tools not found at %s. "
                "Clone from https://github.com/wshanmu/UWB_lab.git "
                "and copy uwb-qorvo-tools/ into the project root.",
                UWB_TOOLS_ROOT,
            )
            return False

        try:
            # Add uwb-qorvo-tools to Python path for imports
            lib_paths = [
                str(UWB_TOOLS_ROOT / "lib" / "uwb-uci"),
                str(UWB_TOOLS_ROOT / "lib" / "uqt-utils"),
                str(UWB_TOOLS_ROOT),
            ]
            for p in reversed(lib_paths):
                if p not in sys.path:
                    sys.path.insert(0, p)
            os.environ["UWB_TOOLS"] = str(UWB_TOOLS_ROOT)

            self._start_ranging_thread()
            self._connected = True
            logger.info(
                "UWB connected: controller=%s, controlee=%s",
                self.config.controller_port,
                self.config.controlee_port,
            )
            return True
        except Exception as exc:
            logger.error("UWB connection failed: %s", exc)
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Stop ranging and clean up."""
        self._stop_event.set()
        if self._ranging_thread and self._ranging_thread.is_alive():
            self._ranging_thread.join(timeout=3.0)
        self._connected = False
        logger.info("UWB disconnected (%d samples received)", self._sample_count)

    def reconnect(self) -> bool:
        """Attempt to reconnect."""
        self.disconnect()
        self._stop_event.clear()
        self._buffer.clear()
        time.sleep(0.5)
        return self.connect()

    def _start_ranging_thread(self) -> None:
        """Launch background thread that polls ranging data."""
        self._stop_event.clear()
        self._ranging_thread = threading.Thread(
            target=self._ranging_loop, daemon=True
        )
        self._ranging_thread.start()

    def _ranging_loop(self) -> None:
        """Background loop: read ranging results and buffer them.

        This tries to import the UCI library and open sessions on both
        boards. If the hardware isn't present, it logs errors and stops.
        """
        try:
            from uwb_uci import UCI  # type: ignore[import-untyped]
        except ImportError:
            logger.error(
                "Cannot import uwb_uci. Ensure uwb-qorvo-tools/lib/uwb-uci "
                "is on PYTHONPATH."
            )
            self._connected = False
            return

        ctrl_session = None
        ctrllee_session = None
        try:
            ctrl_session = UCI(self.config.controller_port)
            ctrllee_session = UCI(self.config.controlee_port)

            ctrl_session.reset()
            ctrllee_session.reset()
            time.sleep(0.5)

            ctrl_session.set_mode("CONTROLLER")
            ctrllee_session.set_mode("CONTROLEE")

            ctrl_session.start_session()
            ctrllee_session.start_session()
            time.sleep(0.5)

            interval_s = self.config.ranging_interval_ms / 1000.0

            while not self._stop_event.is_set():
                try:
                    result = ctrl_session.get_ranging_result()
                    if result is not None:
                        dist = float(result.distance_cm)
                        with self._lock:
                            self._latest_distance = dist
                        self._buffer.append((time.monotonic(), np.array([dist], dtype=np.float32)))
                        self._sample_count += 1
                except Exception:
                    pass
                time.sleep(interval_s)

        except Exception as exc:
            logger.error("UWB ranging error: %s", exc)
            self._connected = False
        finally:
            try:
                if ctrl_session:
                    ctrl_session.stop_session()
                if ctrllee_session:
                    ctrllee_session.stop_session()
            except Exception:
                pass

    def read_sample(self) -> dict | None:
        """Return latest distance reading as a dict, or None."""
        if not self._connected:
            return None
        with self._lock:
            return {
                "type": "sample",
                "timestamp": time.monotonic(),
                "distance_cm": self._latest_distance,
            }

    def ingest(self, max_samples: int = 50) -> int:
        """Drain pending samples into the internal buffer. Returns count added."""
        added = 0
        with self._lock:
            while self._buffer and added < max_samples:
                ts, arr = self._buffer.popleft()
                added += 1
        self._prune()
        return added

    def _prune(self) -> None:
        cutoff = time.monotonic() - self.config.window_seconds * 2
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()

    def get_window(self) -> np.ndarray:
        """Return a fixed-length (timesteps, 1) window of distance data."""
        window_s = self.config.window_seconds
        timesteps = self.config.timesteps
        now = time.monotonic()
        window_data = [arr for ts, arr in self._buffer if now - ts <= window_s]
        if len(window_data) < 3:
            return np.zeros((timesteps, self.config.channels), dtype=np.float32)
        arr = np.array(window_data[-timesteps:], dtype=np.float32)
        if arr.shape[0] < timesteps:
            padded = np.zeros((timesteps, self.config.channels), dtype=np.float32)
            padded[-arr.shape[0]:] = arr
            return padded
        if arr.shape[0] > timesteps:
            indices = np.linspace(0, arr.shape[0] - 1, timesteps).astype(int)
            return arr[indices]
        return arr

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def sample_count(self) -> int:
        return self._sample_count
