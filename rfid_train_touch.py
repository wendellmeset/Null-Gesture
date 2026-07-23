#!/usr/bin/env -S /home/mattin/Documents/Projects/Null-Gesture/venv/bin/python
"""
RFID Touch Training — collect data and train a neural network
=============================================================

Trains a model to classify tag states: CLEAR + 4 touch positions.

Usage:
    # Collect training data (follow the prompts)
    python rfid_train_touch.py --collect --output touch_data.npz

    # Train model from collected data
    python rfid_train_touch.py --train --input touch_data.npz --model touch_model.keras

    # Collect + Train in one go
    python rfid_train_touch.py --collect --train --model touch_model.keras
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, cast

import numpy as np

try:
    import mercury
except ImportError:
    print("python-mercuryapi not installed. Run: bash install_mercury.sh")
    sys.exit(1)
_MercuryReader = cast(Any, mercury).Reader

try:
    from keras import layers, models
except ImportError:
    print("TensorFlow/Keras not installed. Run: pip install tensorflow")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════
#  Feature extraction
# ═══════════════════════════════════════════════════════════════════════════

WINDOW_SECONDS = 2.0
FEATURE_COUNT = 9


def extract_features(rssi_history: deque, rate: float) -> np.ndarray:
    """Extract feature vector from a window of RSSI readings.

    Features:
        0: mean RSSI
        1: std RSSI
        2: min RSSI
        3: max RSSI
        4: RSSI range
        5: RSSI slope (linear fit over time)
        6: read rate (reads/sec)
        7: recent trend (last 5 vs previous 5 mean RSSI)
        8: time since last read (clamped)
    """
    if len(rssi_history) < 2:
        return np.zeros(FEATURE_COUNT, dtype=np.float32)

    times = np.array([t for t, _ in rssi_history])
    rssis = np.array([r for _, r in rssi_history])
    now = time.time()

    feat = np.zeros(FEATURE_COUNT, dtype=np.float32)
    feat[0] = float(np.mean(rssis))
    feat[1] = float(np.std(rssis)) if len(rssis) > 1 else 0.0
    feat[2] = float(np.min(rssis))
    feat[3] = float(np.max(rssis))
    feat[4] = feat[3] - feat[2]
    feat[6] = rate

    # Slope via linear regression
    if len(rssis) >= 3:
        dt = times - times[0]
        if dt[-1] > 0.01:
            slope = np.polyfit(dt, rssis, 1)[0]
            feat[5] = float(np.clip(slope * 10, -5, 5))

    # Trend: last 5 vs previous 5
    if len(rssis) >= 10:
        feat[7] = float(np.mean(rssis[-5:]) - np.mean(rssis[-10:-5]))
    elif len(rssis) >= 4:
        half = len(rssis) // 2
        feat[7] = float(np.mean(rssis[-half:]) - np.mean(rssis[:-half]))

    # Time since last read
    feat[8] = min((now - times[-1]), 5.0)

    return feat


# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _stop_streaming(port: str) -> bool:
    try:
        import serial as _serial
        _ser = _serial.Serial(port, 115200, timeout=1)
        _ser.dtr = False; _ser.rts = False
        time.sleep(0.3)
        _ser.reset_input_buffer()
        _ser.write(b'\xff\x03\x2f\x00\x00\x02\x5e\x86')
        _ser.flush(); time.sleep(1.5)
        while _ser.in_waiting:
            _ser.read(_ser.in_waiting); time.sleep(0.1)
        _ser.close(); time.sleep(0.3)
        return True
    except OSError:
        return False


def _connect(port: str):
    try:
        return _MercuryReader(f"tmr://{port}", baudrate=115200)
    except TypeError as exc:
        if "Streaming" not in str(exc):
            raise
        print("streaming detected, stopping...")
        _stop_streaming(port)
        return _MercuryReader(f"tmr://{port}", baudrate=115200)


# ═══════════════════════════════════════════════════════════════════════════
#  Data Collection
# ═══════════════════════════════════════════════════════════════════════════

LABELS = ["CLEAR", "TOUCHED"]
LABEL_COUNT = len(LABELS)
SECONDS_PER_CLASS = 10.0
SAMPLES_PER_SECOND = 5


def collect_data(port: str | None, output_path: str) -> None:
    """Interactive data collection with live feedback."""
    import rfid_data_reader
    if port is None:
        port = rfid_data_reader.find_m7e_port()
        if port is None:
            sys.exit(1)

    print(f"\nConnecting to reader on {port}...")
    reader = _connect(port)
    reader.set_region("NA")
    reader.set_read_plan([1], "GEN2")
    print(f"✅ {reader.get_model()}")

    all_features: list[np.ndarray] = []
    all_labels: list[int] = []
    rssi_buf: deque = deque()
    tag_count = 0

    def on_tag(tag):
        nonlocal tag_count
        tag_count += 1
        rssi_buf.append((time.time(), float(tag.rssi)))
        # Keep only last WINDOW_SECONDS
        cutoff = time.time() - WINDOW_SECONDS
        while rssi_buf and rssi_buf[0][0] < cutoff:
            rssi_buf.popleft()

    reader.start_reading(callback=on_tag, on_time=1000, off_time=0)
    time.sleep(1)

    print("\n" + "=" * 60)
    print("  TRAINING DATA COLLECTION")
    print("=" * 60)
    print(f"  Will collect {SECONDS_PER_CLASS}s per state, {LABEL_COUNT} states")
    print(f"  Total: ~{SECONDS_PER_CLASS * LABEL_COUNT}s")
    print()

    try:
        for label_idx, label_name in enumerate(LABELS):
            print(f"\n{'─' * 60}")
            if label_idx == 0:
                desc = "Open hand — fingers NOT touching thumb"
            else:
                desc = "Make a fist — thumb touching fingers"
            print(f"  NEXT: {label_name} — {desc}")
            print(f"  Hold this position for {SECONDS_PER_CLASS} seconds.")
            print("  Starting in...")

            for i in range(3, 0, -1):
                sys.stdout.write(f"\r  {i}... ")
                sys.stdout.flush()
                time.sleep(1)

            print(f"\r  Recording {label_name} for {SECONDS_PER_CLASS}s...")
            rssi_buf.clear()
            time.sleep(0.3)  # settle

            start = time.time()
            sample_interval = 1.0 / SAMPLES_PER_SECOND
            next_sample = start + sample_interval
            collected = 0

            while time.time() - start < SECONDS_PER_CLASS:
                now = time.time()
                if now >= next_sample:
                    rate = tag_count / (now - start + 0.001)
                    feat = extract_features(rssi_buf, rate)
                    all_features.append(feat)
                    all_labels.append(label_idx)
                    collected += 1
                    next_sample += sample_interval

                # Show live feedback
                if rssi_buf:
                    latest = rssi_buf[-1][1]
                    sys.stdout.write(f"\r  RSSI: {latest:6.1f} dBm  samples: {collected}  tags: {tag_count}  ")
                    sys.stdout.flush()
                time.sleep(0.05)

            print(f"\r  ✅ {label_name}: {collected} samples collected  ")

    except KeyboardInterrupt:
        print("\n\nCollection interrupted.")
    finally:
        reader.stop_reading()

    if not all_features:
        print("No data collected.")
        return

    X = np.array(all_features, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    np.savez(output_path, features=X, labels=y, label_names=LABELS)

    # Save label mapping as JSON too
    label_path = Path(output_path).with_suffix(".json")
    with open(label_path, "w") as f:
        json.dump({"labels": LABELS}, f)

    print(f"\n{'=' * 60}")
    print(f"  Saved {len(X)} samples to: {output_path}")
    print(f"  Label mapping:   {label_path}")
    counts = [np.sum(y == i) for i in range(LABEL_COUNT)]
    for name, c in zip(LABELS, counts):
        print(f"    {name}: {c} samples")
    print(f"{'=' * 60}")


# ═══════════════════════════════════════════════════════════════════════════
#  Model Definition & Training
# ═══════════════════════════════════════════════════════════════════════════

EPOCHS = 40
BATCH_SIZE = 16


def build_model(input_dim: int, num_classes: int) -> models.Sequential:
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.15),
        layers.Dense(8, activation='relu'),
        layers.Dense(num_classes, activation='softmax'),
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def train_model(data_path: str, model_path: str) -> None:
    print(f"\nLoading data from: {data_path}")
    data = np.load(data_path)
    X = data['features']
    y = data['labels']

    print(f"  Features shape: {X.shape}")
    print(f"  Labels shape:   {y.shape}")
    print(f"  Classes:        {len(np.unique(y))}")

    # Normalize features
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-8
    X_norm = (X - mean) / std

    # Split
    split = int(len(X) * 0.8)
    X_train, X_val = X_norm[:split], X_norm[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"\nTraining set:   {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples")

    model = build_model(X.shape[1], len(np.unique(y)))
    model.summary()

    print(f"\nTraining for {EPOCHS} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,  # pyright: ignore[reportArgumentType]
    )

    val_acc = max(history.history['val_accuracy'])
    print(f"\nBest validation accuracy: {val_acc:.2%}")

    # Save model
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    # Save normalization params
    norm_path = Path(model_path).with_suffix(".norm.json")
    with open(norm_path, "w") as f:
        json.dump({"mean": mean.tolist(), "std": std.tolist(), "labels": LABELS}, f)
    print(f"Normalization params saved to: {norm_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RFID touch classifier")
    p.add_argument("--collect", action="store_true", help="Collect training data")
    p.add_argument("--train", action="store_true", help="Train model from collected data")
    p.add_argument("--input", default="touch_data.npz", help="Input data file (default: touch_data.npz)")
    p.add_argument("--model", default="touch_model.keras", help="Output model file (default: touch_model.keras)")
    p.add_argument("--port", "-p", help="Reader serial port")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.collect:
        collect_data(port=args.port, output_path=args.input)

    if args.train:
        if not Path(args.input).exists():
            print(f"Data file not found: {args.input}")
            print("Run with --collect first, or check the path.")
            sys.exit(1)
        train_model(args.input, args.model)

    if not args.collect and not args.train:
        print("Nothing to do. Use --collect, --train, or both.")
        print("Example: python rfid_train_touch.py --collect --train")


if __name__ == "__main__":
    main()
