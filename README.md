# Null-Gesture

IMU-based gesture recognition — MATLAB-style pipeline (PCA + k-NN).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # or: venv/bin/activate.fish
pip install -r requirements.txt
```

## Usage

```bash
# 1. Record gestures (IMU required)
PYTHONPATH=. python -m null_gesture record --serial /dev/ttyACM0

# 2. Train classifier
PYTHONPATH=. python -m null_gesture train

# 3. Live detection
PYTHONPATH=. python -m null_gesture live --serial /dev/ttyACM0
```

### Quick test (4 gestures)

```bash
PYTHONPATH=. python -m null_gesture record --serial /dev/ttyACM0 --gestures push,pull,left,right --samples 3
PYTHONPATH=. python -m null_gesture train
PYTHONPATH=. python -m null_gesture live --serial /dev/ttyACM0
```

## Architecture

```
record  →  data/raw/*.npy     (acquisition.py)
train   →  model.npz           (features.py → classifier.py)
live    →  GUI display         (detector.py → live.py)
              │
         IMUSensor (imu.py)
```

## Pipeline

1. **Acquisition** — record labeled IMU windows (50 Hz, 6-axis)
2. **Preprocessing** — Butterworth low-pass filter, onset/offset segmentation, resample to 100 samples
3. **Features** — 49 statistical features (mean, std, RMS, skew, kurtosis, SMA, correlations, spectral)
4. **Classifier** — PCA dimensionality reduction + k-NN (k=3)
5. **Detector** — real-time onset/offset segment → extract features → classify

## Gestures

push, pull, left, right, clockwise, anti_clockwise, bye_bye, clapping, one_arm_boxing, t_arms, raise_arms, palm_up, palm_down
