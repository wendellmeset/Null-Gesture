#!/usr/bin/env python3
"""Null-Gesture: In-Motion Real-Time Gesture Detection System.

Usage:
    python -m src.main                              # Auto-detect all sensors
    python -m src.main --imu /dev/ttyUSB0           # Specify IMU port
    python -m src.main --mmwave /dev/ttyACM0        # Specify mmWave port
    python -m src.main --list-ports                 # List available serial ports
    python -m src.main --json                       # Output JSON events to stdout
    python -m src.main --verbose                    # Verbose logging
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any, ClassVar

from src.fusion.temporal import GestureEvent
from src.pipeline import GesturePipeline

_log = logging.getLogger(__name__)


def _list_serial_ports() -> None:
    """List available serial ports for sensor connection."""
    try:
        import serial.tools.list_ports
    except ImportError:
        print("pyserial not installed. Run: pip install pyserial")
        return

    ports = serial.tools.list_ports.comports()
    if not ports:
        print("No serial ports found.")
        return

    print(f"{'Port':<25} {'Description':<40} {'HWID'}")
    print("-" * 90)
    for port in ports:
        print(f"{port.device:<25} {port.description:<40} {port.hwid}")


def _load_config(config_path: str | None) -> dict[str, Any]:
    """Load configuration from YAML or JSON file."""
    if config_path is None:
        return {}

    config_file = Path(config_path)
    if not config_file.exists():
        _log.warning("Config file not found: %s", config_path)
        return {}

    if config_file.suffix in (".yaml", ".yml"):
        try:
            import yaml
            with open(config_file) as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            _log.error("PyYAML required for YAML configs. Install: pip install pyyaml")
            return {}

    if config_file.suffix == ".json":
        with open(config_file) as f:
            return json.load(f)

    _log.warning("Unknown config format: %s", config_file.suffix)
    return {}


class GestureDisplay:
    """Formatted terminal output for gesture events."""

    # Color codes for terminal
    COLORS: ClassVar[dict[str, str]] = {
        "start": "\033[92m",   # green
        "active": "\033[94m",  # blue
        "end": "\033[93m",     # yellow
        "reset": "\033[0m",
        "bold": "\033[1m",
    }

    def __init__(self, json_output: bool = False) -> None:
        self._json_output = json_output
        self._active_gesture: str | None = None
        self._last_line_len = 0

    def show(self, event: GestureEvent) -> None:
        if self._json_output:
            print(json.dumps({
                "gesture": event.gesture,
                "phase": event.phase,
                "confidence": round(event.confidence, 4),
                "timestamp": event.timestamp,
                "duration": round(event.duration, 3),
                "contributing_sensors": event.contributing_sensors,
            }))
            sys.stdout.flush()
            return

        color = self.COLORS.get(event.phase, "")
        reset = self.COLORS["reset"]
        bold = self.COLORS["bold"]

        if event.phase == "start":
            print(f"\n{bold}{color}>>> GESTURE: {event.gesture.upper()}{reset} "
                  f"(conf={event.confidence:.3f})")
            self._active_gesture = event.gesture

        elif event.phase == "active":
            # In-place update
            bar_len = min(20, int(event.confidence * 20))
            bar = "█" * bar_len + "░" * (20 - bar_len)
            line = f"\r  {event.gesture:20s} [{bar}] {event.confidence:.3f}  dur={event.duration:.2f}s"
            print(line, end="", flush=True)

        elif event.phase == "end":
            print(f"\n{color}<<< END: {event.gesture.upper()}{reset} "
                  f"(dur={event.duration:.2f}s, max_conf={event.confidence:.3f})")
            self._active_gesture = None

        sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Null-Gesture: Real-Time Gesture Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Sensor ports
    parser.add_argument("--imu", help="IMU serial port (e.g., /dev/ttyUSB0)")
    parser.add_argument("--mmwave", help="mmWave radar serial port (e.g., /dev/ttyACM0)")
    parser.add_argument("--mmwave-config", default="Readers/configs/mmwave_hand_50cm.cfg",
                        help="mmWave config file path")
    parser.add_argument("--uwb-initiator", help="UWB initiator port")
    parser.add_argument("--uwb-responder", help="UWB responder port")
    parser.add_argument("--rfid", help="RFID reader port (auto-detect if omitted)")

    # Configuration
    parser.add_argument("--config", "-c", help="Path to config file (YAML or JSON)")
    parser.add_argument("--tag-map", help="JSON file mapping RFID EPCs to hand labels")

    # Tuning
    parser.add_argument("--belief-threshold", type=float, default=0.6,
                        help="Minimum belief for gesture detection (default: 0.6)")
    parser.add_argument("--calibration-frames", type=int, default=50,
                        help="Frames for auto-calibration (default: 50)")

    # Output
    parser.add_argument("--json", action="store_true", help="Output events as JSON lines")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    # Utilities
    parser.add_argument("--list-ports", action="store_true", help="List available serial ports and exit")

    args = parser.parse_args()

    # Utility: list ports
    if args.list_ports:
        _list_serial_ports()
        return

    # Logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load config
    config = _load_config(args.config)
    sensor_cfg = config.get("sensors", {})

    # Tag map
    tag_map: dict[str, str] = {}
    if args.tag_map:
        with open(args.tag_map) as f:
            tag_map = json.load(f)
    elif "tag_map" in config:
        tag_map = config["tag_map"]

    # ── Build pipeline ──────────────────────────────────────────────
    pipeline = GesturePipeline(
        imu_port=args.imu or sensor_cfg.get("imu"),
        mmwave_port=args.mmwave or sensor_cfg.get("mmwave"),
        mmwave_config=args.mmwave_config or sensor_cfg.get("mmwave_config"),
        uwb_initiator=args.uwb_initiator or sensor_cfg.get("uwb_initiator"),
        uwb_responder=args.uwb_responder or sensor_cfg.get("uwb_responder"),
        rfid_port=args.rfid or sensor_cfg.get("rfid"),
        tag_map=tag_map,
        belief_threshold=args.belief_threshold,
        calibration_frames=args.calibration_frames,
        verbose=args.verbose,
    )

    # ── Signal handling ─────────────────────────────────────────────
    running = True

    def _signal_handler(sig, frame):
        nonlocal running
        print("\nShutting down...")
        running = False
        pipeline.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # ── Start ───────────────────────────────────────────────────────
    print("=" * 60)
    print("  Null-Gesture — Real-Time Gesture Detection")
    print("=" * 60)

    status = pipeline.start()

    print("\nSensor status:")
    for sensor, ok in status.items():
        icon = "✓" if ok else "✗"
        print(f"  {icon} {sensor}")

    if not any(status.values()):
        print("\nERROR: No sensors connected. Check ports and try --list-ports.")
        sys.exit(1)

    print(f"\nCalibrating ({args.calibration_frames} frames)...")
    while not pipeline.calibrated:
        time.sleep(0.01)
    print("Calibration complete.\n")
    print("Detecting gestures... Press Ctrl+C to stop.\n")

    # ── Run ─────────────────────────────────────────────────────────
    display = GestureDisplay(json_output=args.json)

    try:
        for event in pipeline.events():
            if not running:
                break
            display.show(event)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        print("\nPipeline stopped.")


if __name__ == "__main__":
    main()
