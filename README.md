# Null-Gesture

**Physics-Informed Multi-Modal Real-Time Gesture Detection System**

A capstone project that detects 15 hand/arm gestures in real time using a
TI IWRL6432 60GHz mmWave radar and an ESP32 with BMI270 IMU. Designed to
beat naive "feed-everything-into-a-neural-net" approaches through
physics-informed feature engineering, motion primitive decomposition, and
**30-second few-shot calibration**.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
# If using RFID (optional):
pip install -r Readers/requirements.txt
python Readers/mercury/install.py
```

### 2. Connect Hardware

- **ESP32 + BMI270 IMU** → USB port (e.g., `/dev/ttyUSB0`)
- **TI IWRL6432 mmWave Radar** → USB port (e.g., `/dev/ttyACM0`)

Edit `config/pipeline.yaml` if your ports differ from the defaults.

### 3. Calibrate (~30 seconds)

```bash
python src/main.py calibrate
```

Follow the on-screen prompts. You'll perform each of the 15 gestures once.
The system records one example per gesture and computes per-person scaling
factors — no neural network training needed.

### 4. Run Detection

```bash
# Text output mode
python src/main.py

# Live visualization mode
python src/main.py visualize
```

### 5. Gestures

| # | Gesture | Description |
|---|---------|-------------|
| 1 | Pull | Pull hand toward body |
| 2 | Push | Push hand away from body |
| 3 | Clockwise | Rotate hand clockwise |
| 4 | Anti-Clockwise | Rotate hand anti-clockwise |
| 5 | Left | Move hand to the left |
| 6 | Right | Move hand to the right |
| 7 | Bye-Bye | Wave hand side to side |
| 8 | One Arm Boxing | Throw a punch with one arm |
| 9 | Clapping | Clap hands together |
| 10 | Two-Arm Boxing | Alternate punches with both arms |
| 11 | T-Arms | Extend both arms out to sides |
| 12 | Raise Arms | Raise both arms above head |
| 13 | Soli | Rub thumb against index finger |
| 14 | Opening & Closing Fist | Open and close hand |
| 15 | Palm Up & Down | Rotate palm up, then down |

## How It Beats Other Teams

### 1. Not a Black Box

Most groups will dump raw sensor data into a CNN/LSTM. Our pipeline is a
transparent multi-stage system where every stage is inspectable:

```
Raw Sensors → Preprocessing → Feature Extraction → Classification → Fusion → Output
                    ↑                    ↑                ↑              ↑
              Madgwick AHRS      200+ physics        Random          Bayesian
              Range-Doppler      features per       Forest +        weighted
              Micro-Doppler      sensor             CNN hybrid      geometric mean
                                                                   + HMM smoothing
```

### 2. Physics-Informed Feature Engineering

Instead of learning features from data, we encode domain knowledge:
- **IMU:** orientation quaternions, linear acceleration, jerk, spectral bands
- **mmWave:** range-Doppler heatmaps, trajectory curvature, micro-Doppler spectrograms
- **Motion primitives:** 11 atomic motions that compose into all 15 gestures

This means features are **person-invariant by design** — they work on day one
with a new user, new room, new lighting.

### 3. 30-Second Calibration (Not 30-Minute Training)

Our few-shot calibration protocol records **one example per gesture** and
computes per-person scaling factors. Decision boundaries are shifted, not
relearned. Compare this to the alternative: collect 100+ examples per gesture
and train for hours.

### 4. Adaptive Multi-Sensor Fusion

When the mmWave radar is occluded, the IMU takes over. When the user's wrist
is still, the radar provides the spatial context. Each sensor is weighted by
its **known reliability per gesture class** and **real-time confidence score**.
The system degrades gracefully, not catastrophically.

### 5. Motion Primitive Composition

The 11 motion primitives (TRANSLATE_X, ROTATE_CW, OSCILLATE_FAST, etc.) are
shared across all 15 gestures. Improving detection of one primitive improves
all gestures that use it. This compositional generalization is something
end-to-end neural networks fundamentally cannot do — they must learn each
gesture as a separate pattern.

## Architecture

```
IMU (100 Hz)  ──→ IMUProcessor ──→ IMUFeatureExtractor ──→ IMUClassifier ──┐
                    (AHRS)            (~200 features)        (Random Forest)  │
                                                                              ├──→ BayesianFusion ──→ GestureEnsemble ──→ Output
mmWave (20fps) ──→ MMWaveProcessor ──→ MMWaveFeatureExtractor ──→ MMWaveClassifier ──┘   (weighted        (HMM filter
                   (R-D, μD)           (~150 features)           (CNN + RF)              geometric mean)   + debouncing)
                       │
                GestureSegmenter ──→ PrimitiveDetector ──→ bias
                (energy-based)       (rule-based, 11    (adjusts fusion
                 no ML)               physics rules)     probabilities)
```

## Project Structure

```
Null-Gesture/
├── Docs/                        # Architecture & design documentation
│   ├── 01-architecture-overview.md
│   ├── 02-sensor-strategy.md
│   ├── 03-signal-processing-pipeline.md
│   ├── 04-feature-engineering.md
│   ├── 05-classification-and-fusion.md
│   └── 06-implementation-plan.md
├── config/
│   ├── pipeline.yaml            # Runtime configuration
│   └── gestures.yaml            # Gesture-to-primitive mappings
├── src/
│   ├── sensors/                 # Sensor orchestration & buffering
│   ├── preprocessing/           # Signal processing (AHRS, R-D, segmentation)
│   ├── features/                # Feature extraction & primitives
│   ├── models/                  # Classifiers & few-shot calibration
│   ├── fusion/                  # Bayesian fusion & HMM smoothing
│   ├── utils/                   # Config loading, visualization
│   ├── pipeline.py              # Core pipeline orchestrator
│   └── main.py                  # CLI entry point
├── Readers/                     # Low-level sensor readers (provided)
├── requirements.txt
└── README.md
```

## Configuration

### `config/pipeline.yaml`

Tune sensor ports, segmentation thresholds, fusion parameters, HMM transition
probabilities, and output settings.

### `config/gestures.yaml`

Defines each gesture as a composition of motion primitives with their
detection rules. Adjustable per-gesture.

## Requirements

- Python 3.10+
- NumPy, SciPy, scikit-learn
- PyTorch (CPU) — for mmWave CNN head
- PyYAML, pyserial
- [Readers library](Readers/) — custom sensor drivers

## Platform Support

- **Linux**: Fully supported (primary development platform)
- **macOS**: Supported (port names differ: `/dev/cu.usbmodem*`)
- **Windows**: Supported (port names differ: `COM3`, etc.)

## Performance Targets

| Metric | Target |
|--------|--------|
| Detection latency | < 100 ms |
| False positive rate | < 5% |
| True positive rate | > 90% |
| CPU usage | < 30% (1 core) |
| Memory | < 200 MB |
| Calibration time | < 45 seconds |
