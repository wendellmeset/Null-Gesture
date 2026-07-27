#!/usr/bin/env python3
"""Interactive gesture data collector.

Records 2-second IMU windows for each gesture and saves as .npy.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect IMU gesture recordings")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--serial", help="Serial port (overrides TCP)")
    parser.add_argument("--out", default="data/raw", help="Output directory")
    parser.add_argument("--samples", type=int, default=2, help="Recordings per gesture")
    parser.add_argument("--duration", type=float, default=2.0, help="Recording window in seconds")
    parser.add_argument(
        "--gestures",
        default="clapping,one_arm_boxing,t_arms,raise_arms",
        help="Comma-separated gesture names (underscore naming)",
    )
    args = parser.parse_args(argv)

    PROJ = Path(__file__).resolve().parent.parent
    out_dir = (PROJ / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import so argparse --help is fast
    from null_gesture.sensors.imu_sensor import IMUClient

    imu = IMUClient()
    ok = False
    if args.serial:
        print(f"Connecting via serial: {args.serial} ...")
        ok = imu.connect_serial(args.serial)
    if not ok:
        print(f"Connecting via TCP {args.host}:{args.port} ...")
        ok = imu.connect_tcp(args.host, args.port)

    if not ok:
        print("❌ Failed to connect to IMU.")
        return 1

    print("✅ Connected.\n")

    gestures = [g.strip() for g in args.gestures.split(",")]
    n_samples = args.samples
    window_samples = int(args.duration * 50)  # 50 Hz

    total = len(gestures) * n_samples
    count = 0

    for gesture in gestures:
        for sample_num in range(1, n_samples + 1):
            count += 1
            print(f"[{count}/{total}]  {gesture.upper()}  —  sample {sample_num}/{n_samples}")
            input("    Press ENTER when ready, then perform the gesture immediately...")

            # Flush stale buffers
            imu._rbuf.clear()
            imu._buffer.clear()
            time.sleep(0.05)

            # Collect raw samples for args.duration seconds
            raw: list[np.ndarray] = []
            t0 = time.monotonic()
            while time.monotonic() - t0 < args.duration:
                imu.ingest(max_samples=10)
                time.sleep(0.002)

            window = imu.get_window()
            if window is None or window.sum() == 0:
                print("    ⚠️  No data received — retry this sample.\n")
                count -= 1
                continue

            # Trim to the expected length
            if window.shape[0] > window_samples:
                window = window[-window_samples:]
            elif window.shape[0] < window_samples:
                pad = np.zeros((window_samples - window.shape[0], 6), dtype=np.float32)
                window = np.concatenate([pad, window], axis=0)

            gyro_mag = float(np.linalg.norm(window[:, 3:], axis=1).mean())
            fname = out_dir / f"{gesture}_{sample_num:03d}.npy"
            np.save(fname, window)
            print(f"    ✅ Saved  {fname.name}  (gyro mean: {gyro_mag:.1f} dps)\n")

    imu.disconnect()
    print(f"Done — {count} recordings saved to {out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
