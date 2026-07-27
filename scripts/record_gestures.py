#!/usr/bin/env python3
"""Record IMU + UWB gesture samples for pattern analysis.

Usage:
    source venv/bin/activate
    python scripts/record_gestures.py --serial /dev/ttyACM0 --uwb-controller /dev/ttyACM1 --uwb-controlee /dev/ttyACM2
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

GESTURES = [
    "pull", "push",
    "clockwise", "anti_clockwise",
    "left", "right",
    "bye_bye",
    "clapping", "one_arm_boxing",
    "t_arms", "raise_arms",
    "palm_up", "palm_down",
]

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "uwb_gestures"
SAMPLES_PER = 2
DURATION = 3.0  # seconds


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--serial", default="/dev/ttyACM0")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=9999)
    p.add_argument("--uwb-controller", default="/dev/ttyACM1")
    p.add_argument("--uwb-controlee", default="/dev/ttyACM2")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Connect IMU ───────────────────────────────────────────────
    from null_gesture.sensors.imu_sensor import IMUClient
    imu = IMUClient()
    ok = imu.connect_serial(args.serial) if args.serial else False
    if not ok:
        ok = imu.connect_tcp(args.host, args.port)
    if not ok:
        print("IMU connect failed")
        return 1
    print("IMU connected ✓")

    # ── Connect UWB ───────────────────────────────────────────────
    from null_gesture.sensors.uwb_sensor import UWBRanger
    uwb = UWBRanger()
    uwb_dist = 0.0

    def on_sample(s):
        nonlocal uwb_dist
        if s.get("status") == "Ok":
            uwb_dist = float(s["distance_cm"])

    uwb_ok = uwb.start(args.uwb_controller, args.uwb_controlee, on_sample=on_sample)
    if not uwb_ok:
        print("UWB connect failed")
        imu.disconnect()
        return 1
    print("UWB connected ✓")
    time.sleep(1)

    # ── Record ────────────────────────────────────────────────────
    total = len(GESTURES) * SAMPLES_PER
    count = 0

    for gesture in GESTURES:
        for sample_num in range(1, SAMPLES_PER + 1):
            count += 1
            print(f"\n[{count}/{total}]  {gesture.upper()}  sample {sample_num}/{SAMPLES_PER}")
            input("    Press ENTER, then do the gesture...")

            # Flush
            imu._rbuf.clear()
            imu._buffer.clear()
            time.sleep(0.1)

            # Record DURATION seconds of IMU + UWB
            imu_rows = []
            uwb_rows = []
            t0 = time.monotonic()
            while time.monotonic() - t0 < DURATION:
                imu.ingest(max_samples=5)
                w = imu.get_window()
                if w.sum() != 0:
                    imu_rows.append(w[-1].copy())  # [ax,ay,az,gx,gy,gz]
                uwb_rows.append(uwb_dist)
                time.sleep(0.005)

            if len(imu_rows) < 10:
                print("    ⚠ Not enough IMU data — retry")
                count -= 1
                continue

            # Combine: (timesteps, 7) where cols = [ax, ay, az, gx, gy, gz, uwb_cm]
            imu_arr = np.array(imu_rows, dtype=np.float32)
            uwb_arr = np.array(uwb_rows[:len(imu_rows)], dtype=np.float32).reshape(-1, 1)
            combined = np.concatenate([imu_arr, uwb_arr], axis=1)

            fname = OUT_DIR / f"{gesture}_{sample_num:03d}.npy"
            np.save(fname, combined)

            gyro_peak = float(np.linalg.norm(imu_arr[:, 3:], axis=1).max())
            uwb_range = f"{uwb_arr.min():.0f}-{uwb_arr.max():.0f}cm"
            print(f"    ✅ {fname.name}  gyro_peak={gyro_peak:.0f}dps  uwb={uwb_range}")

    uwb.stop()
    imu.disconnect()
    print(f"\nDone — {count} recordings in {OUT_DIR}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
