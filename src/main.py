#!/usr/bin/env python3
"""Null-Gesture: Real-Time In-Motion Gesture Detection System.

Physics-informed multi-modal gesture recognition using mmWave radar + IMU.
Designed for few-shot calibration (one example per gesture) and real-time
CPU-only inference.

Usage:
    python src/main.py                  # Run detection (requires calibration or models)
    python src/main.py calibrate        # Run few-shot calibration (~30 seconds)
    python src/main.py visualize        # Run detection with live matplotlib plots
    python src/main.py --help           # Show all options
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `from src.…` imports work
# regardless of the current working directory or how the script is invoked.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.pipeline import GesturePipeline
from src.utils.config import load_config, load_gestures_config

# ── Logging ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
_log = logging.getLogger("nullgesture")


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Null-Gesture: Real-Time Gesture Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py                    Run detection
  python src/main.py calibrate          Run calibration (~30s)
  python src/main.py visualize          Run with live plots
  python src/main.py --config my_config.yaml
        """,
    )

    parser.add_argument(
        "mode",
        nargs="?",
        default="run",
        choices=["run", "calibrate", "visualize"],
        help="Operation mode (default: run)",
    )
    parser.add_argument(
        "--config",
        default=str(_PROJECT_ROOT / "config" / "pipeline.yaml"),
        help="Path to pipeline config YAML (default: config/pipeline.yaml)",
    )
    parser.add_argument(
        "--gestures",
        default=str(_PROJECT_ROOT / "config" / "gestures.yaml"),
        help="Path to gestures config YAML (default: config/gestures.yaml)",
    )
    parser.add_argument(
        "--imu-port",
        default=None,
        help="Override IMU serial port (default: from config)",
    )
    parser.add_argument(
        "--mmwave-port",
        default=None,
        help="Override mmWave serial port (default: from config)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress printed gesture output",
    )

    args = parser.parse_args()

    # ── Load configuration ──────────────────────────────────────────
    print("\n  Null-Gesture v1.0.0")
    print("  Physics-Informed Multi-Modal Gesture Recognition\n")

    try:
        config = load_config(args.config)
        gestures_config = load_gestures_config(args.gestures)
    except FileNotFoundError as e:
        print(f"  Error: {e}")
        sys.exit(1)

    # Override ports from CLI
    if args.imu_port:
        config["sensors"]["imu"]["port"] = args.imu_port
    if args.mmwave_port:
        config["sensors"]["mmwave"]["port"] = args.mmwave_port
    if args.quiet:
        config["output"]["print_gestures"] = False

    # ── Build pipeline ──────────────────────────────────────────────
    pipeline = GesturePipeline(config, gestures_config)

    use_visualization = args.mode == "visualize"
    if not pipeline.initialize(with_visualization=use_visualization):
        print("  Error: Pipeline initialization failed.")
        sys.exit(1)

    # ── Start sensors ───────────────────────────────────────────────
    if not pipeline.start():
        print("  Error: Failed to connect to sensors.")
        print("  Check that:")
        print("    1. ESP32 IMU is connected via USB")
        print("    2. IWRL6432 mmWave radar is connected via USB")
        print("    3. Ports match config/pipeline.yaml")
        print(f"       IMU port: {config['sensors']['imu']['port']}")
        print(f"       mmWave port: {config['sensors']['mmwave']['port']}")
        sys.exit(1)

    # ── Handle Ctrl+C gracefully ────────────────────────────────────
    def _signal_handler(sig, frame):
        print("\n\n  Shutting down...")
        pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)

    # ── Run mode ────────────────────────────────────────────────────
    try:
        if args.mode == "calibrate":
            # Run calibration
            print("  Starting calibration routine...")
            success = pipeline.calibrate(countdown_sec=2.0, record_duration=2.0)
            if success:
                print("  Calibration successful! Run without 'calibrate' to start detection.")
            else:
                print("  Calibration failed.")
            pipeline.stop()

        else:
            # Run detection
            if not pipeline.calibrated:
                print("  ⚠ No calibration profile found.")
                print("  Detection will use default thresholds.")
                print("  For best accuracy, run: python src/main.py calibrate\n")

            print("  Listening for gestures...\n")
            print(f"  {'Gesture':>20s}  {'Conf':>6s}  {'Latency':>8s}  {'Source':>8s}")
            print(f"  {'─' * 20}  {'─' * 6}  {'─' * 8}  {'─' * 8}")

            for detection in pipeline.run():
                # Detection is already printed by the pipeline
                # Additional processing could go here (e.g., OSC output, logging)
                pass

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()

    print("\n  Null-Gesture stopped.\n")


if __name__ == "__main__":
    main()
