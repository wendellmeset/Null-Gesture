#!/usr/bin/env python3
"""Null-Gesture: IMU-based gesture detection on ESP32+BMI270.

Usage:
    # Test IMU connection
    python -m null_gesture debug imu --host 127.0.0.1

    # Real-time gesture detection GUI
    python -m null_gesture live --imu-host 127.0.0.1

    # Collect gesture recordings for NN training
    python -m null_gesture collect --host 127.0.0.1

    # Train neural network on collected data
    python -m null_gesture train
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

logger = logging.getLogger("null_gesture")


def cmd_debug_imu(args: argparse.Namespace) -> int:
    from null_gesture.sensors.imu_sensor import IMUClient
    imu = IMUClient()
    ok = False
    if args.serial:
        print(f"Connecting to {args.serial}...")
        ok = imu.connect_serial(args.serial)
    if not ok:
        print(f"Connecting via TCP {args.host}:{args.port}...")
        ok = imu.connect_tcp(args.host, args.port)
    if not ok:
        print("❌ Failed to connect")
        return 1
    print("✅ Connected. Press Ctrl+C to stop.\n")
    try:
        while True:
            imu.ingest(max_samples=200)
            window = imu.get_window()
            if window.sum() != 0:
                latest = window[-1]
                a = latest[:3]; g = latest[3:]
                sys.stdout.write(f"\r  a={a[0]:+.3f} {a[1]:+.3f} {a[2]:+.3f}  g={g[0]:+.1f} {g[1]:+.1f} {g[2]:+.1f}  ({imu.sample_count})  ")
                sys.stdout.flush()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        imu.disconnect()
    return 0


def cmd_live(args: argparse.Namespace) -> int:
    from null_gesture.gui.live_train import LiveDetectWindow
    app = __import__("PyQt6.QtWidgets", fromlist=["QApplication"]).QApplication(sys.argv)
    win = LiveDetectWindow(imu_host=args.imu_host)
    if args.serial:
        win.serial_port = args.serial
    win.show()
    return app.exec()


def calibrate(args: argparse.Namespace) -> int:
    """Stream raw IMU for calibration: user does each motion while I watch."""
    from null_gesture.sensors.imu_sensor import IMUClient
    imu = IMUClient()
    ok = False
    if args.serial:
        ok = imu.connect_serial(args.serial)
    if not ok:
        ok = imu.connect_tcp(args.host, args.port)
    if not ok:
        print("Connect failed"); return 1

    gestures_str = args.gestures or "push,pull,left,right,up,down,clockwise,anti_clockwise,bye_bye,palm_up"
    gestures = gestures_str.split(",")
    for g in gestures:
        input(f"\nPress ENTER then do: {g.upper()}")
        print(f"RECORDING {g}...", flush=True)
        time.sleep(0.3)
        imu._rbuf.clear()
        imu._buffer.clear()
        start = time.time()
        while time.time() - start < 2.0:
            imu.ingest(200)
            w = imu.get_window()
            if w.sum() != 0:
                a = w[-1, :3]; gr = w[-1, 3:]
                sys.stdout.write(f"\r  a={a[0]:+.2f} {a[1]:+.2f} {a[2]:+.2f}  g={gr[0]:+.0f} {gr[1]:+.0f} {gr[2]:+.0f}  ")
                sys.stdout.flush()
            time.sleep(0.005)
        print(f"\n  Done. {imu.sample_count} samples")

    imu.disconnect()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Null-Gesture: IMU-based gesture detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="Commands")

    # debug imu
    dp = sub.add_parser("debug", help="Test IMU connection")
    dp.add_argument("--host", default="127.0.0.1")
    dp.add_argument("--port", default="9999")
    dp.add_argument("--serial", help="Serial port (e.g. /dev/ttyACM0)")

    # live
    lp = sub.add_parser("live", help="Real-time gesture detection (GUI)")
    lp.add_argument("--imu-host", default="127.0.0.1")
    lp.add_argument("--serial", help="Serial port (e.g. /dev/ttyACM0)")

    # calibrate
    cp = sub.add_parser("calibrate", help="Record gestures for calibration")
    cp.add_argument("--serial", default="/dev/ttyACM0", help="Serial port")
    cp.add_argument("--host", default="127.0.0.1")
    cp.add_argument("--port", default="9999")
    cp.add_argument("--gestures", help="Comma-separated gestures to record")

    # collect
    clp = sub.add_parser("collect", help="Collect gesture recordings for NN training")
    clp.add_argument("--host", default="127.0.0.1")
    clp.add_argument("--port", type=int, default=9999)
    clp.add_argument("--serial", help="Serial port (overrides TCP)")
    clp.add_argument("--out", default="data/raw")
    clp.add_argument("--samples", type=int, default=2)
    clp.add_argument("--duration", type=float, default=2.0)
    clp.add_argument("--gestures", help="Comma-separated gestures to record")

    # train
    trp = sub.add_parser("train", help="Train GestureCNN on collected data")
    trp.add_argument("--data", default="data/raw")
    trp.add_argument("--epochs", type=int, default=150)
    trp.add_argument("--batch", type=int, default=64)
    trp.add_argument("--lr", type=float, default=0.001)
    trp.add_argument("--augment", type=int, default=200)
    trp.add_argument("--output", default="null_gesture/models/saved/gesture_cnn.pt")

    # prompter
    prp = sub.add_parser("prompter", help="Gesture prompt display — cycles through gestures with countdown")
    prp.add_argument("--serial", help="Serial port (e.g. /dev/ttyACM0)")
    prp.add_argument("--host", default="127.0.0.1")
    prp.add_argument("--port", type=int, default=9999)

    # imu3d
    i3p = sub.add_parser("imu3d", help="3D IMU cube visualiser — debug gesture motion")
    i3p.add_argument("--serial", help="Serial port (e.g. /dev/ttyACM0)")
    i3p.add_argument("--host", default="127.0.0.1")
    i3p.add_argument("--port", type=int, default=9999)
    i3p.add_argument("--uwb-controller", help="UWB controller serial port")
    i3p.add_argument("--uwb-controlee", help="UWB controlee serial port")

    # uwbdraw
    uwp = sub.add_parser("uwbdraw", help="2D IMU+UWB drawing canvas")
    uwp.add_argument("--serial", help="IMU serial port")
    uwp.add_argument("--host", default="127.0.0.1")
    uwp.add_argument("--port", type=int, default=9999)
    uwp.add_argument("--uwb-controller", help="UWB controller port")
    uwp.add_argument("--uwb-controlee", help="UWB controlee port")

    # detect
    dtp = sub.add_parser("detect", help="IMU+UWB real-time gesture detection")
    dtp.add_argument("--serial", help="IMU port")
    dtp.add_argument("--host", default="127.0.0.1")
    dtp.add_argument("--port", type=int, default=9999)
    dtp.add_argument("--uwb-controller")
    dtp.add_argument("--uwb-controlee")

    # record
    rp = sub.add_parser("record", help="Record IMU+UWB gesture samples")
    rp.add_argument("--serial", default="/dev/ttyACM0")
    rp.add_argument("--host", default="127.0.0.1")
    rp.add_argument("--port", type=int, default=9999)
    rp.add_argument("--uwb-controller", default="/dev/ttyACM1")
    rp.add_argument("--uwb-controlee", default="/dev/ttyACM2")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "debug":
        return cmd_debug_imu(args)
    elif args.command == "live":
        return cmd_live(args)
    elif args.command == "calibrate":
        return calibrate(args)
    elif args.command == "collect":
        from scripts.collect_data import main as collect_main
        return collect_main(sys.argv[2:])
    elif args.command == "train":
        from scripts.train_model import main as train_main
        return train_main(sys.argv[2:])
    elif args.command == "prompter":
        from scripts.gesture_prompter import main as prompter_main
        return prompter_main(sys.argv[2:])
    elif args.command == "imu3d":
        from scripts.imu_3d import main as imu3d_main
        return imu3d_main(sys.argv[2:])
    elif args.command == "uwbdraw":
        from scripts.uwb_draw import main as uwbdraw_main
        return uwbdraw_main(sys.argv[2:])
    elif args.command == "detect":
        from scripts.gesture_detect import main as detect_main
        return detect_main(sys.argv[2:])
    elif args.command == "record":
        from scripts.record_gestures import main as record_main
        return record_main()
    else:
        print("Commands: debug, live, calibrate, collect, train, prompter, imu3d, uwbdraw, detect")
        return 1


if __name__ == "__main__":
    from null_gesture.utils.logging import setup_logging
    setup_logging()
    sys.exit(main())
