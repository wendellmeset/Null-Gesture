#!/usr/bin/env python3
"""Null-Gesture — IMU-based gesture recognition.

Commands:
  record   Record labeled gesture samples
  train    Train classifier on recorded data
  live     Real-time gesture detection GUI
"""

from __future__ import annotations

import argparse
import sys


def cmd_record(args: argparse.Namespace) -> int:
    from null_gesture.sensors.imu import IMUSensor
    from null_gesture.pipeline.acquisition import record_gestures

    imu = IMUSensor()
    ok = False
    if args.serial:
        ok = imu.connect_serial(args.serial)
    if not ok:
        ok = imu.connect_tcp(args.host, args.port)
    if not ok:
        print("❌ IMU connection failed")
        return 1

    gestures = args.gestures.split(",") if args.gestures else [
        "push", "pull", "left", "right",
        "clockwise", "anti_clockwise",
        "bye_bye", "clapping", "one_arm_boxing",
        "t_arms", "raise_arms",
        "palm_up", "palm_down",
    ]

    record_gestures(imu, gestures, args.out, args.samples, args.duration)
    imu.disconnect()
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from pathlib import Path
    from null_gesture.pipeline.detector import GestureDetector

    data_dir = Path(args.data)
    if not data_dir.exists() or not list(data_dir.glob("*.npy")):
        print(f"❌ No recordings found in {data_dir}")
        print("   Run: python -m null_gesture record")
        return 1

    detector = GestureDetector()
    detector.fit_from_recordings(data_dir)
    detector._classifier.save(args.output)
    print(f"✅ Model saved to {args.output}")
    print(f"   Gestures: {detector.gestures}")
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    from PyQt6.QtWidgets import QApplication
    from null_gesture.gui.live import LiveWindow

    app = QApplication(sys.argv)
    win = LiveWindow(serial=args.serial, host=args.host, port=args.port)
    win.show()
    return app.exec()


def main() -> int:
    parser = argparse.ArgumentParser(description="Null-Gesture — IMU gesture recognition")
    sub = parser.add_subparsers(dest="command")

    # record
    rp = sub.add_parser("record", help="Record gesture samples")
    rp.add_argument("--serial")
    rp.add_argument("--host", default="127.0.0.1")
    rp.add_argument("--port", type=int, default=9999)
    rp.add_argument("--out", default="data/raw")
    rp.add_argument("--samples", type=int, default=2)
    rp.add_argument("--duration", type=float, default=2.0)
    rp.add_argument("--gestures")

    # train
    tp = sub.add_parser("train", help="Train classifier on recorded data")
    tp.add_argument("--data", default="data/raw")
    tp.add_argument("--output", default="null_gesture/models/gesture_model.npz")

    # live
    lp = sub.add_parser("live", help="Real-time gesture detection")
    lp.add_argument("--serial")
    lp.add_argument("--host", default="127.0.0.1")
    lp.add_argument("--port", type=int, default=9999)

    args = parser.parse_args()

    if args.command == "record":
        return cmd_record(args)
    elif args.command == "train":
        return cmd_train(args)
    elif args.command == "live":
        return cmd_live(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
