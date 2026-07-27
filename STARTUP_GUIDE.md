# Null-Gesture Startup Guide
# IMPORTANT
Prerequisites
- Python 3.11+
- ESP32 Core 2 with BMI270 IMU (flashed with streaming firmware)
- 2x Qorvo DWM3001CDK UWB boards (optional)
- Linux/macOS (tested on Arch)
Step 1: Environment Setup
cd Null-Gesture
python3 -m venv venv
source venv/bin/activate        # Linux/macOS
pip install -r requirements.txt
Verify:
python -c "import torch; print(torch.__version__)"
python -c "from null_gesture import __version__; print(__version__)"
Step 2: IMU Data Bridge (Required)
The ESP32 streams IMU data over serial. esp32_reader.py bridges it to TCP on port 9999.
1. Edit esp32_reader.py — set serial_port to your ESP32's port (line 249):
serial_port='/dev/ttyACM2',  # or /dev/ttyUSB0, check with: ls /dev/ttyACM*
2. Run it in a separate terminal:
python esp32_reader.py
# Output: Data streaming server listening on 0.0.0.0:9999
3. Test the IMU connection:
python -m null_gesture debug imu --host 127.0.0.1
# Should show changing ax/ay/az/gx/gy/gz values
Step 3: UWB Setup (Optional)
1. Clone the UWB tools:
git clone https://github.com/wshanmu/UWB_lab.git /tmp/UWB_lab
cp -r /tmp/UWB_lab/uwb-qorvo-tools .
pip install pyserial colorama toml
2. Connect both DWM3001CDK boards via USB. Find their ports:
ls /dev/ttyACM*
# Typically /dev/ttyACM0 and /dev/ttyACM1
3. Test each board:
export PYTHONPATH="$(pwd)/uwb-qorvo-tools/lib/uwb-uci:$(pwd)/uwb-qorvo-tools/lib/uqt-utils:$(pwd)/uwb-qorvo-tools:$PYTHONPATH"
python uwb-qorvo-tools/scripts/device/get_device_info/get_device_info.py -p /dev/ttyACM0
Step 4: Record Training Data
# All 15 gestures, 25 samples each (~20 min)
python -m null_gesture collect --dataset my_data \
    --imu-host 127.0.0.1 \
    --uwb-controller /dev/ttyACM0 \
    --uwb-controlee /dev/ttyACM1

# Quick test: 3 gestures, 10 samples each
python -m null_gesture collect --dataset quick_test \
    --gestures pull,push,clapping \
    --samples 10 --duration 3.0

# IMU only (no UWB hardware)
python -m null_gesture collect --dataset imu_only --sensors imu
The collector will:
1. Connect to sensors
2. For each gesture, countdown 3s → record 3s → rest 1s → repeat
3. Save to data/<name>/data.npz + metadata.json
Step 5: Live Detection GUI
python -m null_gesture live --imu-host 127.0.0.1
Opens a PyQt6 window showing real-time IMU waveforms and gyro-based gesture detection (no model needed — rule-based).
What Each Command Does
Command	Purpose	Requires
debug imu	Stream raw IMU values	ESP32 + esp32_reader.py
live	Real-time gesture GUI	ESP32 + esp32_reader.py
collect	Record labeled training data	ESP32 + optionally UWB boards
File Layout After Recording
data/
  my_data/
    data.npz          # arrays: imu (N,100,6), uwb (N,100,1), labels (N,)
    metadata.json     # gesture list, sensor config, sample counts
Training & Prediction (Not Yet Implemented)
The training pipeline (train command) and prediction pipeline (predict command) are not yet built. When ready, they'll use:
- data/preprocessor.py — normalize with ZScoreNormalizer
- data/dataset.py — load with GestureDataset, augment, split train/val
- models/ — encoder + fusion architecture (678K params target)


# DONE
Step-by-step guide from zero to real-time gesture prediction.

## Prerequisites

- Linux (tested on Arch) or macOS with Python 3.11+
- ESP32 Core 2 board with BMI270 IMU
- SparkFun M7E Hecto RFID reader
- 2× Qorvo DWM3001CDK UWB boards
- UHF RFID tag (for thumb)

---

## Phase 1: Environment Setup

```bash
cd Null-Gesture

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### Verify installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__} — CUDA: {torch.cuda.is_available()}')"
python -c "import numpy; print(f'NumPy {numpy.__version__}')"
python -c "from null_gesture import __version__; print(f'Null-Gesture v{__version__}')"
```

Expected output:
```
PyTorch 2.x.x — CUDA: True/False
NumPy 1.x.x
Null-Gesture v2.0.0
```

---

## Phase 2: RFID Setup (skip if IMU-only)

The M7E Hecto reader needs ThingMagic's Mercury API C library.

### Option A — Pre-built wheel (Linux x86_64, Python 3.11)

```bash
pip install wheels/python_mercuryapi-0.5.4-cp311-cp311-linux_x86_64.whl
python -c "import mercury; print(mercury.Reader)"  # Should print <class 'mercury.Reader'>
```

### Option B — Build from source

1. Download the Mercury API from https://novanta.com/precision-medicine/product/thingmagic-mercury-api/
   (free registration required — click "ThingMagic Mercury API BILBO")
2. Run the install script:

```bash
bash install_mercury.sh /path/to/mercuryapi-BILBO-1.37.3.29.zip
```

### Test the RFID reader

```bash
# Auto-detect port + test scan
python rfid_data_reader.py --test

# If auto-detect fails, specify the port:
python rfid_data_reader.py --test --port /dev/ttyUSB0
```

You should see:
```
✅ Reader connected!
   Model: M7E Hecto
   Firmware: ...
```

---

## Phase 3: UWB Setup (skip if IMU-only or RFID-only)

### Install the Qorvo tools

```bash
# Clone the UWB lab repo (we only need the uwb-qorvo-tools/ subdirectory)
git clone https://github.com/wshanmu/UWB_lab.git /tmp/UWB_lab
cp -r /tmp/UWB_lab/uwb-qorvo-tools .
pip install pyserial colorama toml
```

### Verify board detection

Connect both DWM3001CDK boards via USB. Find their serial ports:

```bash
ls /dev/ttyACM* /dev/ttyUSB*
# Typical: /dev/ttyACM0 and /dev/ttyACM1
```

Test each board:

```bash
export PYTHONPATH="$(pwd)/uwb-qorvo-tools/lib/uwb-uci:$(pwd)/uwb-qorvo-tools/lib/uqt-utils:$(pwd)/uwb-qorvo-tools:$PYTHONPATH"
export UWB_TOOLS="$(pwd)/uwb-qorvo-tools"

python uwb-qorvo-tools/scripts/device/get_device_info/get_device_info.py -p /dev/ttyACM0
python uwb-qorvo-tools/scripts/device/get_device_info/get_device_info.py -p /dev/ttyACM1
```

Both should report `status: Ok`.

---

## Phase 4: IMU Setup (ESP32)

### Flash the ESP32

Upload firmware that streams IMU data over serial at 115200 baud.
The expected format (from `esp32_reader.py`):

```
accel[g]  x= 0.123 y=-0.456 z= 1.001 | gyro[dps] x= 1.2 y=-3.4 z= 5.6
```

### Start the data bridge

```bash
# Edit esp32_reader.py first — set serial_port to your ESP32's port
python esp32_reader.py
```

You should see:
```
Data streaming server listening on 0.0.0.0:9999
Streaming system started. Press Ctrl+C to stop.
```

### Test the IMU connection

```bash
python -m null_gesture debug imu --host 127.0.0.1
```

Expected output:
```
Connecting to IMU at 127.0.0.1:9999...
✅ Connected. Streaming 6-axis IMU data. Press Ctrl+C to stop.

  ax: +0.012  ay: -0.003  az: +1.001  gx: +0.100  gy: -0.200  gz: +0.050  (samples: 147)
```

**Move the ESP32 board — the numbers should change.** If they stay at 0, check the serial connection.

---

## Phase 5: Verify All Sensors

```bash
# Terminal 1: IMU bridge
python esp32_reader.py

# Terminal 2: Test each sensor individually
python -m null_gesture debug imu    # Should show changing values
python -m null_gesture debug rfid   # Should show RSSI readings from tag
python -m null_gesture debug uwb --controller /dev/ttyACM0 --controlee /dev/ttyACM1
                                    # Should show distance in cm between hands
```

**All three must produce real data before proceeding to collection.**

---

## Phase 6: Collect Training Data

For a first test, collect a small dataset with just 2 gestures:

```bash
python -m null_gesture collect --dataset quick_test \
    --gestures pull,push \
    --samples 20 --duration 3.0 \
    --imu-host 127.0.0.1 --rfid-port /dev/ttyUSB0 \
    --uwb-controller /dev/ttyACM0 --uwb-controlee /dev/ttyACM1
```

For a production dataset (all 15 gestures, 25 samples each):

```bash
python -m null_gesture collect --dataset full_dataset \
    --imu-host 127.0.0.1 --rfid-port /dev/ttyUSB0 \
    --uwb-controller /dev/ttyACM0 --uwb-controlee /dev/ttyACM1
```

The collector will guide you through each gesture with a countdown and live feedback.

### Single-sensor datasets

If you're missing hardware, collect from individual sensors:

```bash
python -m null_gesture collect --dataset imu_data --sensors imu --imu-host 127.0.0.1
python -m null_gesture collect --dataset rfid_data --sensors rfid --rfid-port /dev/ttyUSB0
python -m null_gesture collect --dataset uwb_data --sensors uwb \
    --uwb-controller /dev/ttyACM0 --uwb-controlee /dev/ttyACM1
```

---

## Phase 7: Train the Model

```bash
python -m null_gesture train --data data/quick_test/data.npz --model-name quick_model
```

Expected output:
```
Training on cuda | 150 epochs | batch_size=32
Train samples: 32 | Val samples: 8
Epoch   1/150 | train loss: 2.4372 acc: 12.50% | val loss: 2.1234 acc: 25.00% | lr: 9.98e-04 | 0.3s
...
Epoch  50/150 | train loss: 0.0123 acc: 100.00% | val loss: 0.0891 acc: 95.83% | lr: 3.45e-04 | 0.2s
  → New best model (95.83%)
...
✅ Training complete! Best validation accuracy: 98.44%
   Model saved to: models/quick_model/
```

### What "good" accuracy means

- **60-80%**: Working, needs more data
- **80-95%**: Good, production-ready for most use cases
- **95%+**: Excellent, near-perfect classification

Single-modality models will have lower accuracy than multi-modal ones. Train the full 15-gesture model with 100+ samples per gesture for best results.

---

## Phase 8: Real-Time Prediction

### GUI mode (recommended)

```bash
python -m null_gesture predict --model models/quick_model/best_model.pt \
    --imu-host 127.0.0.1 --rfid-port /dev/ttyUSB0 \
    --uwb-controller /dev/ttyACM0 --uwb-controlee /dev/ttyACM1
```

A dark-themed window opens showing:
- **Top**: Live IMU, RFID, and UWB waveform plots
- **Middle**: Current prediction with confidence percentage
- **Bottom**: Bar chart of probabilities for all 15 gestures

### Terminal mode

```bash
python -m null_gesture predict --model models/quick_model/best_model.pt \
    --imu-host 127.0.0.1 --no-gui
```

### Single-modality prediction

The fusion model supports running with only one sensor active (others zero-padded):

```bash
# IMU only
python -m null_gesture predict --model models/quick_model/best_model.pt \
    --imu-host 127.0.0.1 --modalities imu

# RFID only
python -m null_gesture predict --model models/quick_model/best_model.pt \
    --rfid-port /dev/ttyUSB0 --modalities rfid --no-gui

# UWB only
python -m null_gesture predict --model models/quick_model/best_model.pt \
    --uwb-controller /dev/ttyACM0 --uwb-controlee /dev/ttyACM1 --modalities uwb
```

---

## Troubleshooting

### IMU: "Connection refused"
- Is `esp32_reader.py` running?
- Check the ESP32 serial port in `esp32_reader.py`
- Verify with `python -m null_gesture debug imu`

### RFID: "python-mercuryapi not installed"
- Run `bash install_mercury.sh` or install the wheel
- Verify: `python -c "import mercury; print(mercury.Reader)"`

### RFID: "No reader found"
- Check the UART switch is in **USB** position (not SER)
- Try a powered USB hub (M7E draws ~700mA)
- Unplug and replug USB-C

### UWB: "UWB tools not found"
- The `uwb-qorvo-tools/` directory must be in the project root
- Run: `cp -r /tmp/UWB_lab/uwb-qorvo-tools .`

### UWB: Board not responding
- Check no other terminal is using the serial port
- Unplug and replug both boards
- Run `python -m null_gesture debug uwb --controller ... --controlee ...`

### Training: "CUDA out of memory"
- Reduce batch size: `--batch-size 8`
- Use CPU only: training will be slower but works

### GUI: "No module named PyQt6"
- `pip install PyQt6 pyqtgraph`
- On headless servers, use `--no-gui` for terminal mode

---

## Directory Layout After Setup

```
Null-Gesture/
├── null_gesture/          # Package (all source code)
├── uwb-qorvo-tools/       # UWB library (cloned from UWB_lab)
├── data/                  # Collected datasets
│   └── my_data/
│       ├── data.npz
│       └── metadata.json
├── models/                # Trained models
│   └── my_model/
│       ├── best_model.pt
│       ├── final_model.pt
│       ├── preprocessor.json
│       └── training_history.json
├── logs/                  # Runtime logs
├── venv/                  # Virtual environment
├── esp32_reader.py        # IMU serial→TCP bridge
├── requirements.txt
└── README.md
```
