#!/usr/bin/env python3
"""Null-Gesture: IMU-based gesture detection on ESP32+BMI270.

Usage:
    # Test IMU connection
    python -m null_gesture debug imu --host 127.0.0.1

    # Real-time gesture detection GUI
    python -m null_gesture live --imu-host 127.0.0.1
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

logger = logging.getLogger("null_gesture")


def cmd_debug_imu(args: argparse.Namespace) -> int:
    from null_gesture.sensors.imu_sensor import IMUClient, IMU_CHANNELS
    imu = IMUClient()
    print(f"Connecting to IMU at {args.host}:{args.port}...")
    if not imu.connect():
        print("❌ Failed. Is esp32_reader.py running on port {args.port}?")
        return 1
    print(f"✅ Connected. Streaming 6-axis IMU data. Press Ctrl+C to stop.\n")
    try:
        while True:
            imu.ingest(max_samples=20)
            window = imu.get_window()
            if window.sum() != 0:
                latest = window[-1]
                vals = "  ".join(f"{IMU_CHANNELS[i]}: {latest[i]:+7.3f}" for i in range(6))
                sys.stdout.write(f"\r  {vals}  (samples: {imu.sample_count})  ")
                sys.stdout.flush()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        imu.disconnect()
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    from null_gesture.gui.live_train import LiveDetectWindow
    app = __import__("PyQt6.QtWidgets", fromlist=["QApplication"]).QApplication(sys.argv)
    win = LiveDetectWindow(imu_host=args.imu_host)
    win.show()
    return app.exec()


def cmd_collect(args: argparse.Namespace) -> int:
    from null_gesture.config import GESTURES
    from null_gesture.data.collector import MultiModalCollector

    gestures = args.gestures.split(",") if args.gestures else None
    sensors = args.sensors.split(",") if args.sensors else ["imu", "uwb"]

    collector = MultiModalCollector(
        dataset_name=args.dataset,
        gestures=gestures,
        samples_per_gesture=args.samples,
        duration=args.duration,
        imu_host=args.imu_host,
        uwb_controller=args.uwb_controller,
        uwb_controlee=args.uwb_controlee,
        sensor_filter=sensors,
    )

    total = collector.run()
    if total > 0:
        collector.save()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Null-Gesture: multi-modal gesture detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="Commands")

    # debug imu
    dp = sub.add_parser("debug", help="Test IMU connection")
    dp.add_argument("--host", default="127.0.0.1", help="ESP32 TCP host")
    dp.add_argument("--port", default="9999", help="ESP32 TCP port")

    # live
    lp = sub.add_parser("live", help="Real-time gesture detection (GUI)")
    lp.add_argument("--imu-host", default="127.0.0.1", help="ESP32 TCP host")

    # collect
    cp = sub.add_parser("collect", help="Record IMU+UWB gesture data")
    cp.add_argument("--dataset", required=True, help="Dataset name (saved under data/<name>/)")
    cp.add_argument("--gestures", default=None, help="Comma-separated gesture names (default: all 15)")
    cp.add_argument("--samples", type=int, default=25, help="Samples per gesture (default: 25)")
    cp.add_argument("--duration", type=float, default=3.0, help="Seconds per sample (default: 3.0)")
    cp.add_argument("--sensors", default=None, help="Comma-separated sensor list: imu,uwb (default: both)")
    cp.add_argument("--imu-host", default="127.0.0.1", help="ESP32 TCP host for IMU")
    cp.add_argument("--uwb-controller", default=None, help="UWB controller serial port")
    cp.add_argument("--uwb-controlee", default=None, help="UWB controlee serial port")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "debug":
        return cmd_debug_imu(args)
    elif args.command == "live":
        return cmd_live(args)
    elif args.command == "collect":
        return cmd_collect(args)
    else:
        print("Commands: debug, live, collect")
        print("  debug imu       — test IMU connection and stream raw data")
        print("  live            — real-time gesture detection GUI")
        print("  collect         — record IMU+UWB gesture data for training")
        return 1


if __name__ == "__main__":
    from null_gesture.utils.logging import setup_logging
    setup_logging()
    sys.exit(main())
