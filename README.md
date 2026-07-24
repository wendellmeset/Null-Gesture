# Null-Gesture v2.0

**Multi-modal gesture detection without a camera.**  
IMU (ESP32+BMI270) + RFID (M7E Hecto) + UWB (DWM3001CDK) fused through a deep neural network into real-time gesture classification across 15 gestures.

```
                  ┌──────────────┐
   Left Hand      │ ESP32+BMI270 │──accel/gyro──┐
   (back)         └──────────────┘              │
                                                │  ┌──────────────┐    ┌──────────┐
                  ┌──────────────┐              ├──►│  IMU Encoder │───►│          │
   Thumb          │  RFID Tag    │──RSSI/phase──┤  │  (Conv+Trans) │    │  Fusion  │
   (tag)          └──────────────┘              │  └──────────────┘    │   MLP    │──► 15 gestures
                                                │                      │          │
   Between        ┌──────────────┐              │  ┌──────────────┐    │          │
   Hands          │ DWM3001CDK×2 │──distance────┤  │  UWB Encoder │───►│          │
   (UWB)          └──────────────┘              │  │  (Conv1D)    │    └──────────┘
                                                │  └──────────────┘
                  ┌──────────────┐              │
   RFID Reader    │  M7E Hecto   │──reads───────┘
   (table)        └──────────────┘
```

## Gestures (15 classes)

| # | Gesture | Description |
|---|---------|-------------|
| 0 | pull | Pull motion toward body |
| 1 | push | Push motion away from body |
| 2 | clockwise | Circular clockwise motion |
| 3 | anti_clockwise | Circular counter-clockwise motion |
| 4 | left | Motion to the left |
| 5 | right | Motion to the right |
| 6 | bye_bye | Waving goodbye |
| 7 | one_arm_boxing | Single-arm boxing punch |
| 8 | clapping | Clapping hands together |
| 9 | two_arm_boxing | Two-arm alternating boxing |
| 10 | t_arms | Arms forming a T |
| 11 | raise_arms | Raising both arms |
| 12 | soli | Google Soli-like micro-gestures |
| 13 | open_close_fist | Opening and closing fist |
| 14 | palm_up_down | Palm rotation up and down |

## Architecture

```
null_gesture/
├── config.py              # All tunables in one place
├── __main__.py             # CLI: debug | collect | train | predict
├── sensors/
│   ├── imu_sensor.py       # TCP client for ESP32 (6-axis: ax,ay,az,gx,gy,gz)
│   ├── rfid_sensor.py      # M7E Hecto wrapper (RSSI + phase, auto-port-detect)
│   └── uwb_sensor.py       # DWM3001CDK FiRa TWR wrapper (refactored py3.11)
├── data/
│   ├── collector.py        # Guided multi-modal data collection
│   ├── preprocessor.py     # Per-channel z-score normalization
│   └── dataset.py          # PyTorch Dataset with augmentation
├── models/
│   ├── imu_encoder.py      # Conv1D residual blocks + Transformer (128d output)
│   ├── rfid_encoder.py     # Conv1D blocks (64d output)
│   ├── uwb_encoder.py      # Conv1D blocks (64d output)
│   ├── fusion.py           # Late fusion → MLP head → 15-class softmax
│   └── trainer.py          # AMP training, early stopping, cosine annealing
├── gui/
│   └── main_window.py      # PyQt6 dark-themed real-time prediction GUI
└── utils/
    └── logging.py
```

**Model**: 678K parameters. Mixed-precision (AMP) training. AdamW + cosine annealing.

## Quick Start

### 1. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. RFID setup (optional)

The M7E Hecto reader requires a C library. Install via the bundled script:

```bash
bash install_mercury.sh /path/to/mercuryapi-BILBO-1.37.x.xx.zip
# Or use the pre-built wheel:
pip install wheels/python_mercuryapi-*.whl
```

### 3. UWB setup (optional)

Clone the UWB tools library into the project root:

```bash
git clone https://github.com/wshanmu/UWB_lab.git /tmp/UWB_lab
cp -r /tmp/UWB_lab/uwb-qorvo-tools .
pip install pyserial colorama toml
```

### 4. Start the IMU data stream

Upload the ESP32 firmware that streams IMU data over serial, then:

```bash
python esp32_reader.py  # Starts TCP server on port 9999
```

## Usage

### Debug — test individual sensors

```bash
# Test IMU connection (displays raw accel/gyro values)
python -m null_gesture debug imu --host 127.0.0.1

# Test RFID reader (displays RSSI and phase per tag)
python -m null_gesture debug rfid --port /dev/ttyUSB0

# Test UWB ranging (displays inter-hand distance in cm)
python -m null_gesture debug uwb --controller /dev/ttyACM0 --controlee /dev/ttyACM1
```

### Collect training data

```bash
# All sensors (15 gestures, 25 samples each, 5s per gesture)
python -m null_gesture collect --dataset my_data \
    --imu-host 127.0.0.1 --rfid-port /dev/ttyUSB0 \
    --uwb-controller /dev/ttyACM0 --uwb-controlee /dev/ttyACM1

# Single sensor only
python -m null_gesture collect --dataset imu_only --sensors imu --imu-host 127.0.0.1
python -m null_gesture collect --dataset rfid_only --sensors rfid --rfid-port /dev/ttyUSB0
python -m null_gesture collect --dataset uwb_only --sensors uwb \
    --uwb-controller /dev/ttyACM0 --uwb-controlee /dev/ttyACM1

# Specific gestures only
python -m null_gesture collect --dataset quick_test --gestures pull,push,clapping \
    --samples 50 --duration 3.0
```

### Train

```bash
python -m null_gesture train --data data/my_data/data.npz --model-name my_model

# With custom hyperparams
python -m null_gesture train --data data/my_data/data.npz \
    --model-name my_model --epochs 300 --batch-size 64 --lr 5e-4
```

The model and preprocessor are saved to `models/my_model/`.

### Predict

```bash
# GUI mode (all sensors)
python -m null_gesture predict --model models/my_model/best_model.pt \
    --imu-host 127.0.0.1 --rfid-port /dev/ttyUSB0 \
    --uwb-controller /dev/ttyACM0 --uwb-controlee /dev/ttyACM1

# Terminal mode (no GUI)
python -m null_gesture predict --model models/my_model/best_model.pt \
    --imu-host 127.0.0.1 --no-gui

# IMU-only prediction (RFID and UWB zero-padded)
python -m null_gesture predict --model models/my_model/best_model.pt \
    --imu-host 127.0.0.1 --modalities imu

# RFID-only prediction
python -m null_gesture predict --model models/my_model/best_model.pt \
    --rfid-port /dev/ttyUSB0 --modalities rfid --no-gui

# Force CPU
python -m null_gesture predict --model models/my_model/best_model.pt \
    --imu-host 127.0.0.1 --cpu --no-gui
```

## Hardware Setup

### ESP32 + BMI270 (IMU)
- Placed on the back of the left hand
- Streams 6-axis data (3 accel + 3 gyro) at 50 Hz over serial
- `esp32_reader.py` bridges serial → TCP on port 9999

### M7E Hecto (RFID)
- RFID tag around the thumb
- Detects thumb-to-fingers contact via RSSI drop and phase shift
- UART switch must be in **USB** position (not SER)

### DWM3001CDK (UWB)
- One board on each wrist/forearm
- Measures inter-hand distance via FiRa two-way ranging at 50 Hz
- Controller (initiator) + Controlee (responder) setup

## Requirements

| Component | Dependency |
|-----------|-----------|
| IMU only | `numpy`, `pyserial` |
| RFID only | `numpy`, `pyserial`, `python-mercuryapi` (C lib) |
| UWB only | `numpy`, `pyserial`, `colorama`, `toml`, `uwb-qorvo-tools/` |
| Models | `torch>=2.0` |
| GUI | `PyQt6`, `pyqtgraph` |
| Training | `scikit-learn`, `joblib` |

## Legacy Files

The original v1 scripts are preserved as reference:

- `rfid_data_reader.py` — original M7E data reader
- `rfid_touch_detector.py` — original touch detection (RSSI heuristics + NN)
- `rfid_train_touch.py` — original training pipeline
- `esp32_reader.py` — original ESP32 serial→TCP bridge (still used by the new IMU sensor)
- `IMU_Network.py` — original Keras IMU model (Conv1D, 15 classes)
- `RFID_Network.py` — original Keras RFID model (Conv1D, binary)

## License

See [LICENSE](LICENSE).
