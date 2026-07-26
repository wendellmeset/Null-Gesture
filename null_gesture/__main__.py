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
    from null_gesture.sensors.imu_sensor import IMU_CHANNELS, IMUClient
    imu = IMUClient()
    print(f"Connecting to IMU at {args.host}:{args.port}...")
    if not imu.connect():
        print("❌ Failed. Is esp32_reader.py running on port {args.port}?")
        return 1
    print("✅ Connected. Streaming 6-axis IMU data. Press Ctrl+C to stop.\n")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Null-Gesture: IMU-based gesture detection.",
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

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "debug":
        return cmd_debug_imu(args)
    elif args.command == "live":
        return cmd_live(args)
    else:
        print("Commands: debug, live")
        print("  debug imu   — test IMU connection and stream raw data")
        print("  live        — real-time gesture detection GUI")
        return 1


if __name__ == "__main__":
    from null_gesture.utils.logging import setup_logging
    setup_logging()
    sys.exit(main())
