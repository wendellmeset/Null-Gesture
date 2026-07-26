# Null-Gesture

**IMU-based gesture detection** on ESP32+BMI270. No camera, no training.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pyserial PyQt6 pyqtgraph
```

## Run

```bash
# Terminal 1 — start IMU bridge
python esp32_reader.py

# Terminal 2 — stream raw 6-axis data
python -m null_gesture debug imu

# Terminal 2 — real-time gesture detection GUI
python -m null_gesture live
```

## Gestures

Standing Still, Push, Pull, Left, Right, Clockwise, Anti-Clockwise

Detected via gyro-dominant analysis. No ML, no training.
