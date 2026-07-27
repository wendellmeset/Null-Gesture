#!/usr/bin/env python3
"""Collect labeled training data for gesture classifiers.

Records synchronized sensor data while the user performs prompted gestures.
Outputs a structured dataset for training the detectors.

Usage:
    python scripts/collect_training_data.py --output data/session_01
    python scripts/collect_training_data.py --gestures clockwise,bye_bye --reps 5

Workflow:
    1. Start the script — it connects to all available sensors
    2. It prompts you: "Perform: clockwise (3... 2... 1... GO!)"
    3. You perform the gesture for ~3 seconds
    4. Data is saved to the output directory
    5. Repeat for each gesture × reps
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

# Add parent to path for src imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sensor.multiplexer import SensorFrame, SensorMultiplexer

_log = logging.getLogger(__name__)

ALL_GESTURES = [
    "pull", "push", "clockwise", "anti_clockwise",
    "left", "right", "bye_bye", "one_arm_boxing",
    "clapping", "two_arm_boxing", "t_arms", "raise_arms",
    "soli", "fist_open_close", "palm_up_down",
]


class DataCollector:
    """Records labeled sensor data for gesture training."""

    def __init__(
        self,
        output_dir: str,
        imu_port: str | None = None,
        mmwave_port: str | None = None,
        mmwave_config: str | None = None,
        uwb_initiator: str | None = None,
        uwb_responder: str | None = None,
        rfid_port: str | None = None,
        record_duration: float = 3.0,
        pre_buffer: float = 0.5,
        post_buffer: float = 0.5,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._mux = SensorMultiplexer(
            imu_port=imu_port,
            mmwave_port=mmwave_port,
            mmwave_config=mmwave_config,
            uwb_initiator=uwb_initiator,
            uwb_responder=uwb_responder,
            rfid_port=rfid_port,
        )

        self._record_duration = record_duration
        self._pre_buffer = pre_buffer
        self._post_buffer = post_buffer

        self._session_data: dict[str, list[dict]] = defaultdict(list)
        self._running = False

    def start(self) -> None:
        status = self._mux.start()
        self._running = True
        print("Sensor status:", status)

    def stop(self) -> None:
        self._running = False
        self._mux.stop()

    def record_gesture(self, gesture: str, label: str) -> dict[str, Any]:
        """Record one gesture instance.

        Args:
            gesture: display name for prompt
            label: gesture label to save

        Returns:
            dict with recording stats
        """
        # Pre-buffer (capture baseline)
        pre_frames: list[SensorFrame] = []
        t_start = time.time()
        while time.time() - t_start < self._pre_buffer:
            frame = self._mux.read_frame()
            if frame.imu is not None or frame.mmwave_points is not None:
                pre_frames.append(frame)
            time.sleep(0.005)

        # Countdown
        for i in range(3, 0, -1):
            print(f"\r  {i}...", end="", flush=True)
            time.sleep(0.7)
        print("\r  GO!              ", flush=True)

        # Record
        record_frames: list[SensorFrame] = []
        t_record = time.time()
        while time.time() - t_record < self._record_duration:
            frame = self._mux.read_frame()
            record_frames.append(frame)
            # Progress bar
            elapsed = time.time() - t_record
            bar_len = int(20 * elapsed / self._record_duration)
            print(f"\r  [{'#' * bar_len}{'-' * (20 - bar_len)}]", end="", flush=True)
            time.sleep(0.005)

        # Post-buffer (capture return to rest)
        post_frames: list[SensorFrame] = []
        t_post = time.time()
        while time.time() - t_post < self._post_buffer:
            frame = self._mux.read_frame()
            post_frames.append(frame)
            time.sleep(0.005)

        print("\r  Done.                ")

        # Serialize frames
        session_entry = {
            "label": label,
            "timestamp": time.time(),
            "pre_frames": self._serialize_frames(pre_frames),
            "record_frames": self._serialize_frames(record_frames),
            "post_frames": self._serialize_frames(post_frames),
        }

        self._session_data[label].append(session_entry)

        return {
            "label": label,
            "pre_frames": len(pre_frames),
            "record_frames": len(record_frames),
            "post_frames": len(post_frames),
        }

    @staticmethod
    def _serialize_frames(frames: list[SensorFrame]) -> list[dict]:
        """Convert SensorFrame objects to JSON-serializable dicts."""
        result: list[dict] = []
        for f in frames:
            entry: dict[str, Any] = {"timestamp": f.timestamp}

            if f.imu is not None:
                entry["imu"] = {
                    "ax": f.imu.get("ax"),
                    "ay": f.imu.get("ay"),
                    "az": f.imu.get("az"),
                    "gx": f.imu.get("gx"),
                    "gy": f.imu.get("gy"),
                    "gz": f.imu.get("gz"),
                }

            if f.mmwave_points is not None and len(f.mmwave_points) > 0:
                entry["mmwave_points"] = f.mmwave_points.tolist()
                if f.mmwave_velocities is not None and len(f.mmwave_velocities) > 0:
                    entry["mmwave_velocities"] = f.mmwave_velocities.tolist()

            if f.uwb is not None:
                entry["uwb"] = f.uwb

            if f.rfid is not None:
                entry["rfid"] = f.rfid

            result.append(entry)
        return result

    def save(self, session_name: str | None = None) -> str:
        """Save all collected data to output directory."""
        if session_name is None:
            session_name = f"session_{int(time.time())}"

        session_dir = self._output_dir / session_name
        session_dir.mkdir(parents=True, exist_ok=True)

        # Save per-gesture files
        for gesture, entries in self._session_data.items():
            gesture_file = session_dir / f"{gesture}.json"
            with open(gesture_file, "w") as f:
                json.dump(entries, f, indent=2)

        # Save summary
        summary = {
            "session": session_name,
            "timestamp": time.time(),
            "gestures_recorded": list(self._session_data.keys()),
            "samples_per_gesture": {
                g: len(e) for g, e in self._session_data.items()
            },
        }
        with open(session_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nData saved to: {session_dir}")
        print(f"  Gestures: {list(self._session_data.keys())}")
        print(f"  Total recordings: {sum(len(e) for e in self._session_data.values())}")

        return str(session_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect labeled gesture training data")

    parser.add_argument("--output", "-o", default="data/training",
                        help="Output directory for training data")
    parser.add_argument("--gestures", help="Comma-separated gesture names to record")
    parser.add_argument("--reps", type=int, default=5,
                        help="Number of repetitions per gesture (default: 5)")
    parser.add_argument("--duration", type=float, default=3.0,
                        help="Recording duration per gesture in seconds (default: 3.0)")

    # Sensor ports
    parser.add_argument("--imu", help="IMU port")
    parser.add_argument("--mmwave", help="mmWave port")
    parser.add_argument("--mmwave-config", default="Readers/configs/mmwave_hand_50cm.cfg")
    parser.add_argument("--uwb-initiator", help="UWB initiator port")
    parser.add_argument("--uwb-responder", help="UWB responder port")
    parser.add_argument("--rfid", help="RFID port")

    args = parser.parse_args()

    # Parse gestures
    if args.gestures:
        gestures = [g.strip() for g in args.gestures.split(",")]
        for g in gestures:
            if g not in ALL_GESTURES:
                print(f"Unknown gesture: {g}")
                print(f"Available: {', '.join(ALL_GESTURES)}")
                sys.exit(1)
    else:
        gestures = ALL_GESTURES

    print("=" * 60)
    print("  Null-Gesture — Training Data Collection")
    print("=" * 60)
    print(f"\nGestures to record: {len(gestures)} gestures × {args.reps} reps")
    print(f"Duration per recording: {args.duration}s")
    print(f"Output: {args.output}")
    print()

    collector = DataCollector(
        output_dir=args.output,
        imu_port=args.imu,
        mmwave_port=args.mmwave,
        mmwave_config=args.mmwave_config,
        uwb_initiator=args.uwb_initiator,
        uwb_responder=args.uwb_responder,
        rfid_port=args.rfid,
        record_duration=args.duration,
    )

    # Signal handler
    def _signal_handler(sig, frame):
        print("\n\nSaving collected data before exit...")
        collector.save()
        collector.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)

    collector.start()

    try:
        for gesture in gestures:
            print(f"\n{'─' * 40}")
            print(f"Gesture: {gesture.upper()} ({args.reps} reps)")
            print(f"{'─' * 40}")

            for rep in range(args.reps):
                print(f"\nRep {rep + 1}/{args.reps}: {gesture}")
                input("Press ENTER when ready (or 's' to skip)... ")

                stats = collector.record_gesture(gesture, gesture)
                print(f"  Recorded: {stats['record_frames']} frames")

                # Short rest between reps
                if rep < args.reps - 1:
                    time.sleep(1.0)

        # Save
        session_dir = collector.save()

        print("\nTraining data collection complete!")
        print(f"Next: python scripts/train_models.py --data {session_dir}")

    finally:
        collector.stop()


if __name__ == "__main__":
    main()
