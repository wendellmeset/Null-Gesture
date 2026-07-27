# Null-Gesture

IMU or mmWave gesture recognition — PCA + k-NN pipeline.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## mmWave (recommended — absolute 3D hand tracking, no drift)

Requires TI WRL6432FSPEVM or compatible 60GHz radar running People Tracking / Gesture Recognition demo firmware.

```bash
# Record (replace /dev/ttyACM1 with mmWave DATA port)
PYTHONPATH=. python -m null_gesture record --mmwave /dev/ttyACM1 --samples 5

# Train (auto-detects mmWave vs IMU data)
PYTHONPATH=. python -m null_gesture train

# Live detection
PYTHONPATH=. python -m null_gesture mmlive --port /dev/ttyACM1
```

## IMU (BMI270 via ESP32, 6-axis)

```bash
# Record
PYTHONPATH=. python -m null_gesture record --serial /dev/ttyACM0 --samples 5

# Train
PYTHONPATH=. python -m null_gesture train

# Live
PYTHONPATH=. python -m null_gesture live --serial /dev/ttyACM0
```

## Architecture

```
null_gesture/
├── __main__.py           CLI: record, train, live, mmlive
├── config.py
├── sensors/
│   ├── imu.py            BMI270 (serial/TCP)
│   └── mmwave.py         TI mmWave TLV parser
├── pipeline/
│   ├── acquisition.py    Record labeled samples
│   ├── preprocessing.py  Filter, segment, normalize
│   ├── features.py       Statistical (IMU) + geometric (mmWave) features
│   ├── classifier.py     PCA + k-NN
│   └── detector.py       Real-time detectors
├── gui/
│   ├── live.py           IMU display
│   └── mmwave_live.py    mmWave display
└── models/               Saved .npz models
```

## Pipeline

1. **Acquisition** — record labeled gesture windows
2. **Features** — IMU: 49 statistical features (mean, std, RMS, skew, kurtosis, SMA, correlations, spectral). mmWave: 36 geometric features (displacement, path length, curvature, speed, enclosed area, turning angle)
3. **Classifier** — PCA dimensionality reduction + k-NN
4. **Detector** — real-time classification with display hold

## Gestures

push, pull, left, right, clockwise, anti_clockwise, bye_bye, clapping, one_arm_boxing, t_arms, raise_arms, palm_up, palm_down
