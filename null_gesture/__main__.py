#!/usr/bin/env python3
"""Null-Gesture — IMU/mmWave gesture recognition.

Commands:
  record    Record labeled gesture samples (IMU or mmWave)
  train     Train classifier on recorded data
  live      Real-time IMU gesture detection GUI
  mmlive    Real-time mmWave gesture detection GUI
"""

from __future__ import annotations

import argparse
import sys


def cmd_record(args: argparse.Namespace) -> int:
    out = args.out or "data/raw"
    gestures = args.gestures.split(",") if args.gestures else [
        "push", "pull", "left", "right",
        "clockwise", "anti_clockwise",
        "bye_bye", "clapping", "one_arm_boxing",
        "t_arms", "raise_arms", "palm_up", "palm_down",
    ]

    if args.mmwave:
        from null_gesture.sensors.mmwave import MMWaveSensor
        from null_gesture.pipeline.acquisition import record_mmwave_gestures
        radar = MMWaveSensor()
        if not radar.connect(args.mmwave):
            print(f"❌ mmWave connection failed on {args.mmwave}")
            return 1
        record_mmwave_gestures(radar, gestures, out, args.samples, args.duration)
        radar.disconnect()
        return 0

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
    record_gestures(imu, gestures, out, args.samples, args.duration)
    imu.disconnect()
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from pathlib import Path

    data_dir = Path(args.data)
    if not data_dir.exists() or not list(data_dir.glob("*.npy")):
        print(f"❌ No recordings found in {data_dir}")
        print("   Run: python -m null_gesture record")
        return 1

    # Auto-detect sensor type from data shape
    sample = np.load(next(data_dir.glob("*.npy")))
    is_mmwave = sample.shape[1] == 3  # (T, 3) = mmWave position data

    if is_mmwave:
        from null_gesture.pipeline.detector import MMWaveDetector
        detector = MMWaveDetector()
        print(f"Detected mmWave data ({sample.shape[1]} channels)")
    else:
        from null_gesture.pipeline.detector import GestureDetector
        detector = GestureDetector()
        print(f"Detected IMU data ({sample.shape[1]} channels)")

    detector.fit_from_recordings(data_dir)
    out = args.output or "null_gesture/models/gesture_model.npz"
    detector._classifier.save(out)
    print(f"✅ Model saved to {out}")
    print(f"   Gestures: {detector.gestures}")
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    from PyQt6.QtWidgets import QApplication
    from null_gesture.gui.live import LiveWindow

    app = QApplication(sys.argv)
    win = LiveWindow(serial=args.serial, host=args.host, port=args.port)
    win.show()
    return app.exec()


def cmd_mmlive(args: argparse.Namespace) -> int:
    from PyQt6.QtWidgets import QApplication
    from null_gesture.gui.mmwave_live import MMWaveLiveWindow

    app = QApplication(sys.argv)
    win = MMWaveLiveWindow(port=args.port)
    win.show()
    return app.exec()


def main() -> int:
    parser = argparse.ArgumentParser(description="Null-Gesture — IMU/mmWave gesture recognition")
    sub = parser.add_subparsers(dest="command")

    # record
    rp = sub.add_parser("record", help="Record gesture samples")
    rp.add_argument("--serial", help="IMU serial port")
    rp.add_argument("--mmwave", help="mmWave serial port (use instead of --serial)")
    rp.add_argument("--host", default="127.0.0.1")
    rp.add_argument("--port", type=int, default=9999)
    rp.add_argument("--out", default="data/raw")
    rp.add_argument("--samples", type=int, default=2)
    rp.add_argument("--duration", type=float, default=2.0)
    rp.add_argument("--gestures")

    # train
    tp = sub.add_parser("train", help="Train classifier on recorded data")
    tp.add_argument("--data", default="data/raw")
    tp.add_argument("--output")

    # live (IMU)
    lp = sub.add_parser("live", help="Real-time IMU gesture detection")
    lp.add_argument("--serial")
    lp.add_argument("--host", default="127.0.0.1")
    lp.add_argument("--port", type=int, default=9999)

    # mmlive (mmWave)
    mp = sub.add_parser("mmlive", help="Real-time mmWave gesture detection")
    mp.add_argument("--port", default="/dev/ttyACM0")

    args = parser.parse_args()

    if args.command == "record":
        return cmd_record(args)
    elif args.command == "train":
        return cmd_train(args)
    elif args.command == "live":
        return cmd_live(args)
    elif args.command == "mmlive":
        return cmd_mmlive(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    import numpy as np  # needed for train auto-detect
    sys.exit(main())
