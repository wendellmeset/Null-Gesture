# Null-Gesture: In-Motion Real-Time Gesture Detection System

A multi-modal, physics-informed gesture detection system that identifies 15 hand
and body gestures in real time using IMU, mmWave radar, UWB ranging, and RFID
identification — fused via **Dempster-Shafer evidence theory**.

**Designed to win.** Not a black-box neural net. Specialized detectors, classical
DSP, and rigorous uncertainty handling.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
pip install -r Readers/requirements.txt
```

For RFID support, install the Mercury API C library:
```bash
python Readers/mercury/install.py
```

### 2. Connect Hardware

| Device               | Where It Goes          | USB Port Example    |
|----------------------|------------------------|---------------------|
| ESP32 + BMI270 (IMU) | Wristband, right wrist | `/dev/ttyUSB0`     |
| IWRL6432 (mmWave)    | Tabletop, facing user  | `/dev/ttyACM0`     |
| DWM3001CDK #1 (UWB)  | Wristband with IMU     | `/dev/ttyACM1`     |
| DWM3001CDK #2 (UWB)  | Tabletop, next to radar| `/dev/ttyACM2`     |
| M7E Hecto (RFID)     | Tabletop               | Auto-detect        |
| RFID Tags ×2          | Left/right wristbands  | —                   |

**Minimal setup** (for demo speed): IMU + mmWave + one UWB board. 30 seconds.

### 3. Run

```bash
# Auto-detect sensors
python -m src.main

# Specify ports explicitly
python -m src.main \
    --imu /dev/ttyUSB0 \
    --mmwave /dev/ttyACM0 \
    --uwb-initiator /dev/ttyACM1 \
    --uwb-responder /dev/ttyACM2

# JSON output for integration
python -m src.main --json

# With config file
python -m src.main --config configs/default.yaml

# List available ports
python -m src.main --list-ports
```

The system auto-calibrates in ~0.5 seconds. No training needed — start gesturing
immediately.

---

## Detected Gestures (15 total)

| # | Gesture              | Primary Sensor     | Method                  |
|---|----------------------|--------------------|-------------------------|
| 1 | Pull                 | UWB + IMU          | Distance derivative     |
| 2 | Push                 | UWB + IMU          | Distance derivative     |
| 3 | Clockwise            | IMU (gyro_z)       | DTW template match      |
| 4 | Anti-Clockwise       | IMU (gyro_z)       | DTW template match      |
| 5 | Left                 | IMU (accel_x)      | DTW + threshold         |
| 6 | Right                | IMU (accel_x)      | DTW + threshold         |
| 7 | Bye-Bye              | IMU (gyro_z)       | HMM sequential model    |
| 8 | One Arm Boxing       | IMU (accel mag)    | Peak detection + HMM    |
| 9 | Clapping             | mmWave (clusters)  | Cluster merge detection |
|10 | Two-Arm Boxing       | IMU + RFID         | Alternating peaks + HMM |
|11 | T-Arms               | mmWave (spread)    | Spatial variance        |
|12 | Raise Arms           | mmWave (centroid)  | Centroid velocity       |
|13 | Soli                 | mmWave (velocity)  | Micro-Doppler spectrum  |
|14 | Open/Close Fist      | mmWave (pt count)  | Point count modulation  |
|15 | Palm Up/Down         | IMU (gyro_x)       | Gyro integration        |

## Architecture

```
Sensors → Multiplexer → Preprocessor → Feature Extractors → 5 Detectors
                                                              │
                                                    ┌─────────┴─────────┐
                                                    │ Dempster-Shafer   │
                                                    │ Fusion Engine     │
                                                    └────────┬──────────┘
                                                             │
                                                    ┌────────┴──────────┐
                                                    │ Temporal State    │
                                                    │ Machine           │
                                                    └────────┬──────────┘
                                                             │
                                                       Gesture Event
```

### Five Specialized Detectors (not one monolithic NN)

1. **Motion Detector** — IMU-driven: HMM + DTW + Random Forest for 7 gestures
2. **Posture Detector** — mmWave + RFID: spatial features for T-Arms, Raise Arms, Clapping
3. **Micro-Doppler Detector** — mmWave velocity: spectral analysis for Soli
4. **Proximity Detector** — UWB + IMU: distance dynamics for Pull/Push
5. **Hand Gesture Detector** — mmWave + IMU: point count + gyro for fist/palm

### Dempster-Shafer Fusion (not softmax averaging)

- Models **ignorance** explicitly — a detector that didn't fire contributes
  `m(Θ)=1` (vacuous belief), not noise.
- Handles **conflict** — if classifiers disagree, conflict goes to Θ rather
  than being normalized away (Yager's rule).
- **Sensor dropout** is graceful — missing sensors contribute vacuous belief.

Full architecture details: see `Docs/` directory.

---

## Why This Beats Other Approaches

| Dimension           | Typical (One NN)          | Null-Gesture                   |
|---------------------|---------------------------|--------------------------------|
| Training needed     | Thousands per gesture     | Zero (auto-calibrates)         |
| Setup time          | ~5+ minutes               | ~30 seconds                    |
| Sensor requirement  | All 4 required            | Works with 1+ sensors          |
| Explainability      | Black box                 | Per-detector inspectable       |
| False positives     | High (NN hallucination)   | Low (DS conflict rejection)    |
| Latency             | GPU/server needed         | CPU, <10ms per frame           |

See `Docs/05_competitive_advantages.md` for the full analysis.

---

## Training Custom Models

If you want to improve accuracy with your own data:

```bash
# 1. Collect training data
python scripts/collect_training_data.py \
    --output data/training/session_01 \
    --gestures clockwise,bye_bye,one_arm_boxing \
    --reps 10

# 2. Train models
python scripts/train_models.py \
    --data data/training/session_01 \
    --output src/models/pretrained

# 3. Run with custom models
python -m src.main --config configs/default.yaml
```

---

## Project Structure

```
Null-Gesture/
├── Docs/                          # Architecture and design documentation
│   ├── 01_architecture_overview.md
│   ├── 02_sensor_strategy.md
│   ├── 03_gesture_detection_design.md
│   ├── 04_fusion_engine.md
│   └── 05_competitive_advantages.md
├── Readers/                       # Hardware device drivers
│   ├── imu.py, mmwave.py, uwb.py, rfid.py
│   ├── docs.md
│   └── configs/
├── src/                           # Gesture detection pipeline
│   ├── main.py                    # CLI entry point
│   ├── pipeline.py                # End-to-end pipeline orchestrator
│   ├── sensor/                    # Multiplexer & preprocessor
│   ├── features/                  # Feature extraction per sensor
│   ├── detectors/                 # Five specialized gesture detectors
│   ├── fusion/                    # Dempster-Shafer + temporal filtering
│   └── models/                    # Pretrained model parameters
├── scripts/                       # Training & data collection
├── configs/                       # Configuration files
└── README.md
```

---

## Demoing Live

For the best live demo:

1. **Setup (30s)**: Place tabletop box (mmWave + UWB responder + RFID reader).
   Strap wristband (ESP32 IMU + UWB initiator) on right wrist.
2. **Start**: `python -m src.main -v`
3. **Calibration**: Wait 0.5 seconds for "Calibration complete."
4. **Demo gestures**:
   - **Wave (Bye-Bye)** — wave hand side to side
   - **Punch (One Arm Boxing)** — jab forward
   - **Circle** — draw circles in the air
   - **Soli** — rub thumb and index finger near radar
   - **Clap** — bring hands together
   - **T-Arms** — extend arms out to sides
5. **Show the terminal**: see gesture names, confidence bars, and contributing
   sensors printed in real time.

---

## License

MIT — see LICENSE file.
