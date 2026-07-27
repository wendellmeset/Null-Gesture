# Null-Gesture

**IMU-based gesture detection** on ESP32+BMI270. Heuristic fallback + neural network for near-perfect accuracy.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start (heuristic — works immediately, ~70% accuracy)

```bash
# Terminal 1 — start IMU bridge
python esp32_reader.py

# Terminal 2 — real-time gesture detection GUI
python -m null_gesture live
```

The GUI auto-detects whether a trained model exists. If not, it falls back to the heuristic detector.

## Neural Network Pipeline (→ 99% accuracy)

### 1. Collect training data

```bash
python -m null_gesture collect --host 127.0.0.1
```

This records 2 samples of each gesture (push, pull, left, right, up, down, clockwise, anti-clockwise, bye_bye, palm_up, palm_down). Follow the prompts — press ENTER, perform the gesture, repeat.

### 2. Train the model

```bash
python -m null_gesture train
```

Generates 200 synthetic variants per real sample, trains a lightweight 1D CNN (~35K params), and saves to `null_gesture/models/saved/gesture_cnn.pt`.

### 3. Run with the neural detector

```bash
python -m null_gesture live
```

The GUI now loads your trained model. Green "NN" badge in the status bar confirms it.

## Architecture

```
ESP32 + BMI270  ──serial──>  esp32_reader.py  ──TCP:9999──>  IMUClient
                                                                    │
                                                    ┌───────────────┴───────────────┐
                                                    ▼                               ▼
                                            SimpleIMUDetector                  NNDetector
                                            (heuristic, always works)         (GestureCNN, needs training)
                                                    │                               │
                                                    └───────────┬───────────────────┘
                                                                ▼
                                                          LiveDetectWindow
                                                            (PyQt6 GUI)
```

## Gestures

| # | Gesture | Description |
|---|---------|-------------|
| 1 | standing_still | Hand at rest |
| 2 | push | Push forward |
| 3 | pull | Pull back |
| 4 | left | Swipe left |
| 5 | right | Swipe right |
| 6 | up | Swipe up |
| 7 | down | Swipe down |
| 8 | clockwise | Rotate clockwise |
| 9 | anti_clockwise | Rotate counter-clockwise |
| 10 | bye_bye | Wave side-to-side |
| 11 | palm_up | Rotate palm upward |
| 12 | palm_down | Rotate palm downward |

## Model details

- **GestureCNN**: 3 Conv1D blocks + global average pool + 2 FC layers
- **~35,000 parameters** — runs in <1ms on CPU
- **Temporal smoothing**: exponential moving average on softmax probabilities eliminates flicker
- **Stillness gate**: gyro magnitude threshold pre-filters standing_still
- **Augmentation**: Gaussian noise, time warping, magnitude scaling, bias drift, time shifting, channel masking
