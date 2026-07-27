"""Data acquisition — record labeled gesture samples from IMU or mmWave."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from null_gesture.sensors.imu import IMUSensor
from null_gesture.sensors.mmwave import MMWaveSensor


def record_gestures(
    imu: IMUSensor,
    gestures: list[str],
    output_dir: str | Path = "data/raw",
    samples_per: int = 2,
    duration: float = 2.0,
    prompt: bool = True,
) -> int:
    """Record labeled gesture samples from an IMU sensor.

    Args:
        imu: Connected IMUSensor instance.
        gestures: List of gesture names to record.
        output_dir: Directory to save .npy files.
        samples_per: Number of recordings per gesture.
        duration: Recording window in seconds.
        prompt: If True, waits for ENTER between gestures.

    Returns:
        Number of recordings saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_rate = 50  # Hz
    window_samples = int(duration * sample_rate)
    total = len(gestures) * samples_per
    count = 0

    for gesture in gestures:
        for sample_num in range(1, samples_per + 1):
            count += 1
            if prompt:
                input(f"\n[{count}/{total}]  {gesture.upper()}  sample {sample_num}/{samples_per}"
                      f"\n    Press ENTER, then perform the gesture...")

            # Flush
            imu._rbuf.clear()
            imu._buffer.clear()
            time.sleep(0.05)

            # Record
            rows: list[np.ndarray] = []
            t0 = time.monotonic()
            while time.monotonic() - t0 < duration:
                imu.ingest(max_samples=5)
                w = imu.get_window()
                if w.sum() != 0:
                    rows.append(w[-1].copy())
                time.sleep(0.005)

            if len(rows) < 10:
                print("    ⚠  Not enough data — retry")
                count -= 1
                continue

            data = np.array(rows, dtype=np.float32)
            if data.shape[0] > window_samples:
                data = data[-window_samples:]
            elif data.shape[0] < window_samples:
                pad = np.zeros((window_samples - data.shape[0], 6), dtype=np.float32)
                data = np.concatenate([pad, data], axis=0)

            fname = output_dir / f"{gesture}_{sample_num:03d}.npy"
            np.save(fname, data)

            gyro_peak = float(np.linalg.norm(data[:, 3:], axis=1).max())
            print(f"    ✅ {fname.name}  (gyro peak: {gyro_peak:.0f} dps)")

    print(f"\nDone — {count} recordings saved to {output_dir}/")
    return count


def record_mmwave_gestures(
    radar: MMWaveSensor,
    gestures: list[str],
    output_dir: str | Path = "data/raw",
    samples_per: int = 2,
    duration: float = 2.0,
    prompt: bool = True,
) -> int:
    """Record labeled gesture samples from mmWave radar.

    mmWave outputs 3D point cloud at ~15-20 fps.
    We capture the dominant (hand) point position over time.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = len(gestures) * samples_per
    count = 0

    for gesture in gestures:
        for sample_num in range(1, samples_per + 1):
            count += 1
            if prompt:
                input(f"\n[{count}/{total}]  {gesture.upper()}  sample {sample_num}/{samples_per}"
                      f"\n    Press ENTER, then perform the gesture...")

            # Flush radar buffer
            radar._rbuf.clear()
            radar._buffer.clear()
            time.sleep(0.1)

            # Record for `duration` seconds
            rows: list[np.ndarray] = []
            t0 = time.monotonic()
            while time.monotonic() - t0 < duration:
                radar.ingest()
                pt = radar.get_dominant_point()
                if pt is not None:
                    rows.append(pt)  # (3,) [x, y, z]
                time.sleep(0.01)

            if len(rows) < 5:
                print("    ⚠  Not enough points — retry")
                count -= 1
                continue

            data = np.array(rows, dtype=np.float32)  # (T, 3)
            fname = output_dir / f"{gesture}_{sample_num:03d}.npy"
            np.save(fname, data)

            disp = np.linalg.norm(data[-1] - data[0]) if len(data) > 1 else 0
            print(f"    ✅ {fname.name}  ({len(data)} pts, displacement: {disp:.2f}m)")

    print(f"\nDone — {count} recordings saved to {output_dir}/")
    return count
