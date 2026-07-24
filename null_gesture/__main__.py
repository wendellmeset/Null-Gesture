#!/usr/bin/env python3
"""Null-Gesture: Multi-modal gesture detection without a camera.

Usage:
    # Debug — test individual sensors in isolation (no model needed)
    python -m null_gesture debug imu --host 127.0.0.1
    python -m null_gesture debug rfid --port /dev/ttyUSB0
    python -m null_gesture debug uwb --controller /dev/ttyACM0 --controlee /dev/ttyACM1

    # Collect training data from one or more sensors
    python -m null_gesture collect --dataset my_data --imu-host 127.0.0.1 \\
        --rfid-port /dev/ttyUSB0 --uwb-controller /dev/ttyACM0 --uwb-controlee /dev/ttyACM1
    python -m null_gesture collect --dataset imu_only --imu-host 127.0.0.1 --sensors imu

    # Train a model from collected data
    python -m null_gesture train --data data/my_data/data.npz --model-name my_model

    # Run real-time prediction with GUI (all sensors)
    python -m null_gesture predict --model models/my_model/best_model.pt \\
        --imu-host 127.0.0.1 --rfid-port /dev/ttyUSB0 \\
        --uwb-controller /dev/ttyACM0 --uwb-controlee /dev/ttyACM1

    # Single-modality prediction (zero-pads missing sensors)
    python -m null_gesture predict --model models/my_model/best_model.pt \\
        --imu-host 127.0.0.1 --modalities imu
    python -m null_gesture predict --model models/my_model/best_model.pt \\
        --rfid-port /dev/ttyUSB0 --modalities rfid --no-gui
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("null_gesture")


# ═══════════════════════════════════════════════════════════════════════════════
#  Debug — test individual sensors
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_debug(args: argparse.Namespace) -> int:
    """Test a single sensor in isolation and display raw readings."""

    if args.sensor == "imu":
        return _debug_imu(args)
    elif args.sensor == "rfid":
        return _debug_rfid(args)
    elif args.sensor == "uwb":
        return _debug_uwb(args)
    else:
        print(f"Unknown sensor: {args.sensor}")
        return 1


def _debug_imu(args: argparse.Namespace) -> int:
    """Connect to IMU and display raw accelerometer/gyroscope readings."""
    from null_gesture.sensors.imu_sensor import IMUClient, IMU_CHANNELS

    imu = IMUClient()
    print(f"Connecting to IMU at {args.host}:{args.port}...")
    if not imu.connect():
        print("❌ Failed to connect. Is esp32_reader.py running?")
        return 1

    print(f"✅ Connected. Streaming 6-axis IMU data. Press Ctrl+C to stop.\n")
    try:
        while True:
            imu.ingest(max_samples=10)
            window = imu.get_window()
            if window.sum() != 0:
                latest = window[-1]
                vals = "  ".join(
                    f"{IMU_CHANNELS[i]}: {latest[i]:+7.3f}" for i in range(6)
                )
                sys.stdout.write(f"\r  {vals}  (samples: {imu.sample_count})  ")
                sys.stdout.flush()
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        imu.disconnect()
    return 0


def _debug_rfid(args: argparse.Namespace) -> int:
    """Connect to RFID reader and display tag reads in real time."""
    try:
        from null_gesture.sensors.rfid_sensor import RFIDReader
    except ImportError as exc:
        print(f"❌ Cannot import RFID module: {exc}")
        print("   Run: bash install_mercury.sh")
        return 1

    rfid = RFIDReader()
    port = args.port
    if not port:
        try:
            port = RFIDReader.find_port()
        except ImportError as exc:
            print(f"❌ {exc}")
            return 1
    if not port:
        print("❌ No RFID reader found. Use --port to specify manually.")
        return 1

    print(f"Connecting to RFID reader on {port}...")
    if not rfid.connect(port):
        print("❌ Failed to connect.")
        return 1

    rfid.start_streaming()
    print(f"✅ Connected. Streaming tag reads. Press Ctrl+C to stop.\n")
    try:
        while True:
            time.sleep(0.2)
            window = rfid.get_window()
            if window.shape[0] > 0 and window.sum() != 0:
                latest_rssi = window[-1, 0]
                latest_phase = window[-1, 1]
                sys.stdout.write(
                    f"\r  RSSI: {latest_rssi:+6.1f} dBm  "
                    f"Phase: {latest_phase:+7.3f} rad  "
                    f"(total: {rfid.tag_count} tags)  "
                )
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        rfid.disconnect()
    return 0


def _debug_uwb(args: argparse.Namespace) -> int:
    """Run UWB ranging and display inter-hand distance in real time."""
    from null_gesture.sensors.uwb_sensor import UWBRanger

    uwb = UWBRanger()
    print(f"Starting UWB ranging session...")
    print(f"  Controller: {args.controller}")
    print(f"  Controlee:  {args.controlee}")

    samples_received = []
    def on_sample(s):
        samples_received.append(s)

    if not uwb.start(args.controller, args.controlee, duration=0, on_sample=on_sample):
        print("❌ Failed to start UWB session. Check ports and uwb-qorvo-tools/.")
        return 1

    print(f"✅ UWB session active. Streaming distances. Press Ctrl+C to stop.\n")
    try:
        while True:
            time.sleep(0.15)
            window = uwb.get_window()
            n = len(samples_received)
            if window.shape[0] > 0 and window.sum() != 0:
                latest_cm = window[-1, 0]
                sys.stdout.write(
                    f"\r  Distance: {latest_cm:+7.1f} cm  "
                    f"(samples: {n})  "
                )
                sys.stdout.flush()
    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        uwb.stop()
    return 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Collect
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_collect(args: argparse.Namespace) -> int:
    """Collect multi-modal training data with guided labeling."""
    from null_gesture.data.collector import DataCollector

    collector = DataCollector(args.dataset)

    # Determine which sensors to use (supports "all", "imu", "rfid", "uwb", "imu+uwb", etc.)
    sensors = (args.sensors or "all").lower()
    use_imu = "imu" in sensors
    use_rfid = "rfid" in sensors
    use_uwb = "uwb" in sensors

    imu_host = args.imu_host if use_imu else None
    rfid_port = args.rfid_port if use_rfid else None
    uwb_ctrl = args.uwb_controller if use_uwb else None
    uwb_ctee = args.uwb_controlee if use_uwb else None

    if not collector.connect(
        imu_host=imu_host,
        rfid_port=rfid_port,
        uwb_controller=uwb_ctrl,
        uwb_controlee=uwb_ctee,
        rfid_epcs=args.rfid_epcs.split(",") if args.rfid_epcs else None,
    ):
        print("❌ No sensors connected. Check your ports and try again.")
        return 1

    try:
        collector.collect_all(
            seconds_per_gesture=args.duration,
            samples_per_gesture=args.samples,
            gestures=args.gestures.split(",") if args.gestures else None,
        )
    except KeyboardInterrupt:
        print("\n⚠ Collection interrupted.")
    finally:
        collector.disconnect()

    return 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Train
# ═══════════════════════════════════════════════════════════════════════════════

def cmd_train(args: argparse.Namespace) -> int:
    """Train the multi-modal gesture classifier."""
    import torch

    from null_gesture.config import ModelConfig
    from null_gesture.models.trainer import train_from_npz

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return 1

    config = ModelConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )

    print(f"\nTraining model '{args.model_name}'...")
    print(f"  Data:     {data_path}")
    print(f"  Epochs:   {config.epochs}")
    print(f"  Batch:    {config.batch_size}")
    print(f"  LR:       {config.learning_rate}")

    model, preprocessor, history = train_from_npz(
        data_path, model_name=args.model_name, config=config,
    )

    best_acc = max(history["val_acc"])
    print(f"\n✅ Training complete! Best validation accuracy: {best_acc:.2%}")
    print(f"   Model saved to: models/{args.model_name}/")

    return 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Predict
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_modalities(args: argparse.Namespace) -> str:
    """Determine which modalities to use based on connected sensors."""
    if args.modalities:
        return args.modalities.lower()
    # Auto-detect from connected sensors
    parts = []
    if args.imu_host:
        parts.append("imu")
    if args.rfid_port:
        parts.append("rfid")
    if args.uwb_controller and args.uwb_controlee:
        parts.append("uwb")
    return "+".join(parts) if parts else "all"


def cmd_predict(args: argparse.Namespace) -> int:
    """Run real-time gesture prediction with optional GUI."""
    import torch

    from null_gesture.config import GESTURES, NUM_GESTURES
    from null_gesture.data.preprocessor import Preprocessor
    from null_gesture.models.fusion import GestureFusionModel
    from null_gesture.models.trainer import Trainer
    from null_gesture.sensors.imu_sensor import IMUClient
    from null_gesture.sensors.rfid_sensor import RFIDReader
    from null_gesture.sensors.uwb_sensor import UWBRanger

    modalities = _resolve_modalities(args)
    use_imu = "imu" in modalities
    use_rfid = "rfid" in modalities
    use_uwb = "uwb" in modalities

    # ── Load model ───────────────────────────────────────────────────
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return 1

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"Loading model from {model_path} on {device}...")
    model, ckpt = Trainer.load(model_path, device=device)
    val_acc = ckpt.get("val_acc", ckpt.get("best_val_acc", 0.0))
    print(f"  Model loaded (val_acc: {val_acc:.2%})")
    print(f"  Active modalities: {modalities}")

    # ── Load preprocessor ────────────────────────────────────────────
    pp_path = model_path.parent / "preprocessor.json"
    if pp_path.exists():
        preprocessor = Preprocessor.load(pp_path)
        print(f"  Preprocessor loaded from {pp_path}")
    else:
        preprocessor = None
        print("  ⚠ No preprocessor found — using raw data (may reduce accuracy)")

    # ── Connect sensors ──────────────────────────────────────────────
    imu = IMUClient()
    rfid = RFIDReader()
    uwb = UWBRanger()

    if use_imu and args.imu_host:
        imu.connect()
    if use_rfid and args.rfid_port:
        rfid.connect(args.rfid_port)
        rfid.start_streaming()
    if use_uwb and args.uwb_controller and args.uwb_controlee:
        uwb.start(args.uwb_controller, args.uwb_controlee)

    # ── GUI mode ─────────────────────────────────────────────────────
    if not args.no_gui:
        return _run_gui(model, preprocessor, imu, rfid, uwb, device, modalities)

    # ── Terminal mode ────────────────────────────────────────────────
    return _run_terminal(model, preprocessor, imu, rfid, uwb, device, modalities)


def _run_gui(
    model: "GestureFusionModel",
    preprocessor: "Preprocessor | None",
    imu: "IMUClient",
    rfid: "RFIDReader",
    uwb: "UWBRanger",
    device: str,
    modalities: str,
) -> int:
    """Run prediction with the PyQt6 GUI."""
    import torch

    from null_gesture.gui.main_window import GestureGUI

    use_imu = "imu" in modalities
    use_rfid = "rfid" in modalities
    use_uwb = "uwb" in modalities

    def prediction_callback():
        imu.ingest(max_samples=10)
        imu_win = imu.get_window()
        rfid_win = rfid.get_window()
        uwb_win = uwb.get_window()

        # Check for all-zero windows (but only for active sensors)
        active_sum = 0.0
        if use_imu:
            active_sum += imu_win.sum()
        if use_rfid:
            active_sum += rfid_win.sum()
        if use_uwb:
            active_sum += uwb_win.sum()
        if active_sum == 0:
            return imu_win, rfid_win, uwb_win, None

        # Preprocess
        if preprocessor is not None:
            imu_n = (imu_win - preprocessor._imu_mean) / preprocessor._imu_std
            rfid_n = (rfid_win - preprocessor._rfid_mean) / preprocessor._rfid_std
            uwb_n = (uwb_win - preprocessor._uwb_mean) / preprocessor._uwb_std
        else:
            imu_n, rfid_n, uwb_n = imu_win, rfid_win, uwb_win

        # Inference
        imu_t = torch.from_numpy(imu_n).float().unsqueeze(0).to(device)
        rfid_t = torch.from_numpy(rfid_n).float().unsqueeze(0).to(device)
        uwb_t = torch.from_numpy(uwb_n).float().unsqueeze(0).to(device)

        with torch.no_grad():
            _, probs = model.predict(imu_t, rfid_t, uwb_t, modalities=modalities)

        return imu_win, rfid_win, uwb_win, probs[0].cpu().numpy()

    app = __import__("PyQt6.QtWidgets", fromlist=["QApplication"]).QApplication(sys.argv)
    gui = GestureGUI(prediction_callback=prediction_callback)
    gui.update_status(f"Modalities: {modalities}")
    gui.show()
    return app.exec()


def _run_terminal(
    model: "GestureFusionModel",
    preprocessor: "Preprocessor | None",
    imu: "IMUClient",
    rfid: "RFIDReader",
    uwb: "UWBRanger",
    device: str,
    modalities: str,
) -> int:
    """Run prediction in terminal mode."""
    import torch
    from null_gesture.config import GESTURES

    use_imu = "imu" in modalities
    use_rfid = "rfid" in modalities
    use_uwb = "uwb" in modalities

    print(f"\nReal-time prediction [{modalities}]. Press Ctrl+C to stop.\n")
    try:
        while True:
            imu.ingest(max_samples=10)
            imu_win = imu.get_window()
            rfid_win = rfid.get_window()
            uwb_win = uwb.get_window()

            active_sum = 0.0
            if use_imu:
                active_sum += imu_win.sum()
            if use_rfid:
                active_sum += rfid_win.sum()
            if use_uwb:
                active_sum += uwb_win.sum()
            if active_sum == 0:
                time.sleep(0.1)
                continue

            if preprocessor is not None:
                imu_n = (imu_win - preprocessor._imu_mean) / preprocessor._imu_std
                rfid_n = (rfid_win - preprocessor._rfid_mean) / preprocessor._rfid_std
                uwb_n = (uwb_win - preprocessor._uwb_mean) / preprocessor._uwb_std
            else:
                imu_n, rfid_n, uwb_n = imu_win, rfid_win, uwb_win

            imu_t = torch.from_numpy(imu_n).float().unsqueeze(0).to(device)
            rfid_t = torch.from_numpy(rfid_n).float().unsqueeze(0).to(device)
            uwb_t = torch.from_numpy(uwb_n).float().unsqueeze(0).to(device)

            with torch.no_grad():
                _, probs = model.predict(imu_t, rfid_t, uwb_t, modalities=modalities)

            probs_np = probs[0].cpu().numpy()
            top3 = np.argsort(probs_np)[::-1][:3]

            sys.stdout.write("\033[J")
            for rank, idx in enumerate(top3):
                conf = probs_np[idx]
                gesture = GESTURES[idx].replace("_", " ").title()
                bar = "█" * int(conf * 30)
                marker = "→" if rank == 0 else " "
                sys.stdout.write(f"  {marker} {gesture:20s} {bar} {conf:.0%}\n")
            sys.stdout.write("\033[F" * min(3, len(top3)))
            sys.stdout.flush()
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        imu.disconnect()
        rfid.disconnect()
        uwb.stop()

    return 0


def cmd_live(args: argparse.Namespace) -> int:
    """Interactive real-time training with live feedback GUI."""
    import sys as _sys

    from null_gesture.gui.live_train import LiveTrainWindow

    gestures = [g.strip() for g in args.gestures.split(",") if g.strip()]
    if len(gestures) < 2:
        print("Need at least 2 gestures (e.g. --gestures pull,push)")
        return 1

    app = __import__("PyQt6.QtWidgets", fromlist=["QApplication"]).QApplication(_sys.argv)
    win = LiveTrainWindow(gestures, imu_host=args.imu_host, window_seconds=args.window, reps=args.reps)
    win.show()
    return app.exec()


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Null-Gesture: Multi-modal gesture detection without a camera.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # ── debug ────────────────────────────────────────────────────────
    dp = sub.add_parser("debug", help="Test a single sensor in isolation (no model)")
    dp.add_argument(
        "sensor", choices=["imu", "rfid", "uwb"],
        help="Sensor to test",
    )
    dp.add_argument("--host", default="127.0.0.1", help="[imu] ESP32 TCP host")
    dp.add_argument("--port", default="9999", help="[imu] TCP port / [rfid] serial port")
    dp.add_argument("--controller", help="[uwb] Controller serial port")
    dp.add_argument("--controlee", help="[uwb] Controlee serial port")

    # ── collect ──────────────────────────────────────────────────────
    cp = sub.add_parser("collect", help="Collect training data from sensors")
    cp.add_argument("--dataset", required=True, help="Dataset name for output directory")
    cp.add_argument("--sensors", default="all",
                    help="Which sensors to record: all, imu, rfid, uwb, imu+uwb, etc. (default: all)")
    cp.add_argument("--imu-host", default="127.0.0.1", help="ESP32 TCP host")
    cp.add_argument("--rfid-port", help="M7E serial port (auto-detect if omitted)")
    cp.add_argument("--rfid-epcs", help="Comma-separated target EPCs to filter")
    cp.add_argument("--uwb-controller", help="UWB controller serial port")
    cp.add_argument("--uwb-controlee", help="UWB controlee serial port")
    cp.add_argument("--duration", type=float, default=5.0, help="Seconds per gesture")
    cp.add_argument("--samples", type=int, default=25, help="Samples per gesture")
    cp.add_argument("--gestures", help="Comma-separated gestures (default: all 15)")

    # ── train ────────────────────────────────────────────────────────
    tp = sub.add_parser("train", help="Train the gesture classifier")
    tp.add_argument("--data", required=True, help="Path to data.npz from collection")
    tp.add_argument("--model-name", default="gesture_model", help="Output model name")
    tp.add_argument("--epochs", type=int, default=150, help="Training epochs")
    tp.add_argument("--batch-size", type=int, default=32, help="Batch size")
    tp.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    # ── predict ──────────────────────────────────────────────────────
    pp = sub.add_parser("predict", help="Run real-time gesture prediction")
    pp.add_argument("--model", required=True, help="Path to model checkpoint (.pt)")
    pp.add_argument("--imu-host", default="127.0.0.1", help="ESP32 TCP host")
    pp.add_argument("--rfid-port", help="M7E serial port")
    pp.add_argument("--uwb-controller", help="UWB controller port")
    pp.add_argument("--uwb-controlee", help="UWB controlee port")
    pp.add_argument(
        "--modalities", default="",
        help="Which modalities to use: all, imu, rfid, uwb, imu+rfid, etc. "
             "(default: auto-detect from connected sensors)"
    )
    pp.add_argument("--no-gui", action="store_true", help="Run in terminal mode")
    pp.add_argument("--cpu", action="store_true", help="Force CPU inference")
    # ── live ─────────────────────────────────────────────────────────
    lp = sub.add_parser("live", help="Interactive real-time training with live feedback")
    lp.add_argument("--imu-host", default="127.0.0.1", help="ESP32 TCP host")
    lp.add_argument("--gestures", default="pull,push",
                    help="Comma-separated gestures to train (default: pull,push)")
    lp.add_argument("--window", type=float, default=2.0,
                    help="Window length in seconds (default: 2.0)")
    lp.add_argument("--reps", type=int, default=10,
                    help="Repetitions per gesture (default: 10)")

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.command == "debug":
        return cmd_debug(args)
    elif args.command == "collect":
        return cmd_collect(args)
    elif args.command == "train":
        return cmd_train(args)
    elif args.command == "predict":
        return cmd_predict(args)
    elif args.command == "live":
        return cmd_live(args)
    else:
        print("Please specify a command: debug, collect, train, or predict")
        print("Run with --help for details.")
        return 1


if __name__ == "__main__":
    from null_gesture.utils.logging import setup_logging
    setup_logging()
    sys.exit(main())
