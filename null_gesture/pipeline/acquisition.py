"""Data acquisition — record labeled IMU gesture samples.

Usage:
    from null_gesture.pipeline.acquisition import record_gestures
    record_gestures(imu, gestures, output_dir, samples_per=2, duration=2.0)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from null_gesture.sensors.imu import IMUSensor


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
