#!/usr/bin/env python3
"""Quick UWB ranging test — prints raw distance readings.

Usage:
    python scripts/test_uwb.py
    python scripts/test_uwb.py --initiator /dev/ttyACM1 --responder /dev/ttyACM2
    python scripts/test_uwb.py --port /dev/ttyACM1    # single-board mode
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Readers import UWBReader


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw UWB ranging test")
    parser.add_argument("--initiator", "-i", help="Initiator port")
    parser.add_argument("--responder", "-r", help="Responder port")
    parser.add_argument("--port", "-p", help="Single-board port")
    parser.add_argument("--count", "-n", type=int, default=20,
                        help="Number of readings to show (default: 20)")
    args = parser.parse_args()

    # Build reader
    if args.port:
        uwb = UWBReader(port=args.port)
    elif args.initiator and args.responder:
        uwb = UWBReader(initiator_port=args.initiator, responder_port=args.responder)
    else:
        # Auto-detect UWB ports by nRF52 VID/PID
        from src.sensor.multiplexer import SensorMultiplexer
        ports = SensorMultiplexer._find_uwb_ports()
        if len(ports) >= 2:
            print(f"Auto-detected UWB ports: {ports[0]} (responder), {ports[1]} (initiator)")
            uwb = UWBReader(initiator_port=ports[1], responder_port=ports[0])
        else:
            print("No UWB ports found. Specify --initiator and --responder.")
            sys.exit(1)

    print(f"Connecting...")
    try:
        ok = uwb.connect()
    except Exception as e:
        print(f"Connect failed: {e}")
        sys.exit(1)

    if not ok:
        print("Connect returned False")
        sys.exit(1)

    print(f"Connected. Reading {args.count} samples...\n")
    print(f"{'#':>4s}  {'Distance (m)':>12s}  {'Address':>20s}")
    print("-" * 45)

    count = 0
    try:
        for sample in uwb:
            count += 1
            dist = sample.get("distance_m", "?")
            addr = sample.get("addr", "?")
            print(f"{count:4d}  {dist:12.4f}  {addr:>20s}")
            if count >= args.count:
                break
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        print(f"\nError during streaming: {e}")
    finally:
        uwb.disconnect()
        print(f"\nDone. Received {count} samples.")


if __name__ == "__main__":
    main()
