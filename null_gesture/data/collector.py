"""Multi-modal data collector for labeled gesture samples.

Orchestrates simultaneous recording from IMU, RFID, and UWB sensors
with a guided labeling interface for training data creation.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from null_gesture.config import (
    DATA_DIR,
    GESTURES,
    IMUConfig,
    RFIDConfig,
    UWBConfig,
    imu_config,
    rfid_config,
    uwb_config,
)
from null_gesture.sensors.imu_sensor import IMUClient
from null_gesture.sensors.rfid_sensor import RFIDReader
from null_gesture.sensors.uwb_sensor import UWBRanger

logger = logging.getLogger("null_gesture.data.collector")


class DataCollector:
    """Guided data collection for multi-modal gesture samples.

    Usage:
        collector = DataCollector("my_dataset")
        collector.connect(imu_host="127.0.0.1", rfid_port="/dev/ttyUSB0",
                          uwb_controller="/dev/ttyACM0", uwb_controlee="/dev/ttyACM1")
        collector.collect_all(seconds_per_gesture=5.0)
    """

    def __init__(
        self,
        dataset_name: str,
        imu_cfg: IMUConfig | None = None,
        rfid_cfg: RFIDConfig | None = None,
        uwb_cfg: UWBConfig | None = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.dataset_dir = DATA_DIR / dataset_name
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        self.imu = IMUClient(imu_cfg)
        self.rfid = RFIDReader(rfid_cfg)
        self.uwb = UWBRanger(uwb_cfg)

        self._imu_cfg = imu_cfg or imu_config
        self._rfid_cfg = rfid_cfg or rfid_config
        self._uwb_cfg = uwb_cfg or uwb_config

        # Collected data
        self.imu_windows: list[np.ndarray] = []
        self.rfid_windows: list[np.ndarray] = []
        self.uwb_windows: list[np.ndarray] = []
        self.labels: list[int] = []

    def connect(
        self,
        *,
        imu_host: str | None = None,
        rfid_port: str | None = None,
        uwb_controller: str | None = None,
        uwb_controlee: str | None = None,
        rfid_epcs: list[str] | None = None,
        uwb_duration: int = 3600,
    ) -> bool:
        """Connect to all available sensors. Returns True if at least one connected."""
        connected_any = False

        if imu_host is not None:
            self._imu_cfg.tcp_host = imu_host
            if self.imu.connect():
                connected_any = True

        if rfid_port is not None:
            if self.rfid.connect(rfid_port, target_epcs=rfid_epcs):
                self.rfid.start_streaming()
                connected_any = True

        if uwb_controller is not None and uwb_controlee is not None:
            if self.uwb.start(uwb_controller, uwb_controlee, duration=uwb_duration):
                connected_any = True

        return connected_any

    def disconnect(self) -> None:
        self.imu.disconnect()
        self.rfid.disconnect()
        self.uwb.stop()

    def collect_all(
        self,
        seconds_per_gesture: float = 5.0,
        samples_per_gesture: int = 25,
        gestures: list[str] | None = None,
    ) -> None:
        """Interactive guided collection for all gestures.

        For each gesture, prompts the user to perform it, then captures
        windows from every connected sensor.
        """
        gesture_list = gestures or GESTURES
        print(f"\n{'=' * 60}")
        print(f"  DATA COLLECTION: {self.dataset_name}")
        print(f"  {len(gesture_list)} gestures × {samples_per_gesture} samples")
        print(f"  {seconds_per_gesture}s per gesture")
        print(f"{'=' * 60}\n")

        for label_idx, gesture in enumerate(gesture_list):
            self._collect_gesture(
                gesture, label_idx, seconds_per_gesture, samples_per_gesture
            )

        self._save()
        print(f"\n✅ Saved {len(self.labels)} samples to {self.dataset_dir}")

    def _collect_gesture(
        self,
        gesture: str,
        label_idx: int,
        duration: float,
        samples: int,
    ) -> None:
        """Collect samples for a single gesture."""
        print(f"\n{'─' * 50}")
        print(f"  GESTURE: {gesture}  ({label_idx + 1}/{len(GESTURES)})")
        print(f"  Prepare to perform this gesture for {duration:.0f}s.")
        print(f"  Collecting {samples} sample windows...")

        for i in range(3, 0, -1):
            sys.stdout.write(f"\r  Starting in {i}... ")
            sys.stdout.flush()
            time.sleep(1)

        print(f"\r  Recording '{gesture}' for {duration:.0f}s...  ")
        start = time.time()
        sample_interval = duration / max(samples, 1)
        next_sample = start + sample_interval
        collected = 0

        while time.time() - start < duration and collected < samples:
            now = time.time()
            if now >= next_sample:
                # Ingest new data from sensors
                self.imu.ingest(max_samples=50)

                imu_win = self.imu.get_window()
                rfid_win = self.rfid.get_window()
                uwb_win = self.uwb.get_window()

                self.imu_windows.append(imu_win)
                self.rfid_windows.append(rfid_win)
                self.uwb_windows.append(uwb_win)
                self.labels.append(label_idx)

                collected += 1
                next_sample += sample_interval

                sys.stdout.write(
                    f"\r  {gesture}: {collected}/{samples} samples  "
                )
                sys.stdout.flush()

            time.sleep(0.05)

        print(f"\r  ✅ {gesture}: {collected} samples collected     ")

    def _save(self) -> None:
        """Save all collected data to disk."""
        if not self.labels:
            logger.warning("No data collected — nothing saved")
            return

        imu_arr = np.array(self.imu_windows, dtype=np.float32)
        rfid_arr = np.array(self.rfid_windows, dtype=np.float32)
        uwb_arr = np.array(self.uwb_windows, dtype=np.float32)
        labels_arr = np.array(self.labels, dtype=np.int32)

        np.savez_compressed(
            self.dataset_dir / "data.npz",
            imu=imu_arr,
            rfid=rfid_arr,
            uwb=uwb_arr,
            labels=labels_arr,
        )

        metadata = {
            "dataset_name": self.dataset_name,
            "gestures": GESTURES,
            "num_samples": len(self.labels),
            "imu_shape": list(imu_arr.shape),
            "rfid_shape": list(rfid_arr.shape),
            "uwb_shape": list(uwb_arr.shape),
            "imu_config": {
                "sample_rate_hz": self._imu_cfg.sample_rate_hz,
                "window_seconds": self._imu_cfg.window_seconds,
            },
            "rfid_config": {
                "sample_rate_hz": self._rfid_cfg.sample_rate_hz,
                "window_seconds": self._rfid_cfg.window_seconds,
            },
            "uwb_config": {
                "sample_rate_hz": self._uwb_cfg.sample_rate_hz,
                "window_seconds": self._uwb_cfg.window_seconds,
            },
        }
        with open(self.dataset_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Label distribution
        for i, g in enumerate(GESTURES):
            count = int(np.sum(labels_arr == i))
            print(f"    {g}: {count} samples")
