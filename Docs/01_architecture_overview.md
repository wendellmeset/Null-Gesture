# Null-Gesture: Architecture Overview

## System Philosophy

Most teams will feed raw sensor data into a deep neural network and call it done. That
approach is **fragile**: it needs massive labeled training data per user, it's a black
box you can't debug, and it wastes compute on 15-way classification when a gesture has
already been narrowed to 2-3 candidates by physics.

Null-Gesture takes the opposite approach: **physics-informed multi-stage pipeline**
where each stage reduces uncertainty before passing to the next. We don't guess — we
eliminate impossibilities, then classify the remainder with purpose-built lightweight
models.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     SENSOR LAYER                                 │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │   IMU    │  │  mmWave  │  │   UWB    │  │   RFID   │        │
│  │100 Hz    │  │60 GHz    │  │TWR range │  │UHF tags  │        │
│  │BMI270    │  │IWRL6432  │  │DWM3001   │  │M7E Hecto │        │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
│       │              │              │              │             │
└───────┼──────────────┼──────────────┼──────────────┼────────────┘
        │              │              │              │
        ▼              ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  STREAM MULTIPLEXER                              │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Threaded readers → timestamp-aligned circular buffers    │   │
│  │  Lock-free ring buffers (200 samples per sensor)          │   │
│  └──────────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────────┘
                            │  Synchronized frames
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     PREPROCESSOR                                 │
│  ├─ IMU: gravity removal (Madgwick/Mahony), calibration         │
│  ├─ mmWave: static BG subtraction, DBSCAN clustering            │
│  ├─ UWB: median filter, velocity estimation                     │
│  └─ RFID: tag presence filter, left/right hand disambiguation   │
└───────────────────────────┬─────────────────────────────────────┘
                            │  Clean, structured features
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                 FEATURE ENGINEERING                              │
│                                                                  │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐       │
│  │IMU Feats  │ │mmWave F.  │ │UWB Feats   │ │RFID Feats │       │
│  │├ stat.    │ │├ centroid │ │├ distance  │ │├ tag EPC  │       │
│  │├ spectral │ │├ spread   │ │├ velocity  │ │├ RSSI     │       │
│  │├ ZCR      │ │├ pt cnt   │ │├ accel     │ │├ presence │       │
│  │├ jerk     │ │├ velocity │ │└───────────┘ │└───────────┘       │
│  │└──────────┘ │└──────────┘                                   │
│  └───────────┘                                                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │  Feature vectors per window
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              GESTURE DETECTOR POOL  (parallel)                   │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │Motion        │  │Posture       │  │Micro-Doppler │          │
│  │Classifier    │  │Classifier    │  │Classifier    │          │
│  │(IMU-driven)  │  │(mmWave-drv.) │  │(Soli detect) │          │
│  │              │  │              │  │              │          │
│  │ Bye-Bye      │  │ T-Arms       │  │ Soli         │          │
│  │ 1-Arm Boxing │  │ Raise Arms   │  │              │          │
│  │ 2-Arm Boxing │  │ Clapping     │  └──────────────┘          │
│  │ CW / ACW     │  └──────────────┘                             │
│  │ Left / Right │  ┌──────────────┐  ┌──────────────┐          │
│  └──────────────┘  │Proximity     │  │Hand Gesture  │          │
│                    │Analyzer      │  │Classifier    │          │
│                    │(UWB-driven)  │  │(mmWave+IMU)  │          │
│                    │              │  │              │          │
│                    │ Pull         │  │ Fist Open/   │          │
│                    │ Push         │  │ Close        │          │
│                    └──────────────┘  │ Palm Up/Down │          │
│                                      └──────────────┘          │
│                                                                  │
│  Models: HMM + DTW + Random Forest + SVM (not one giant NN)     │
└───────────────────────────┬─────────────────────────────────────┘
                            │  Per-classifier confidence scores
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              FUSION ENGINE (Dempster-Shafer)                     │
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │ Evidence combination from 5 classifiers → mass fn    │       │
│  │ Conflict resolution via Yager's rule                  │       │
│  │ Frame of discernment: {G1..G15, Θ}                    │       │
│  │ Reject if belief < threshold OR conflict > limit     │       │
│  └──────────────────────────────────────────────────────┘       │
└───────────────────────────┬─────────────────────────────────────┘
                            │  Fused belief mass per gesture
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              TEMPORAL STATE MACHINE                              │
│                                                                  │
│  ├─ Hysteresis: onset threshold > offset threshold               │
│  ├─ Gesture boundary detection: min inter-gesture gap            │
│  ├─ Kalman filter on belief trajectory                           │
│  └─ Output: (gesture_id, confidence, timestamp)                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
                    Gesture Event Stream
```

---

## Data Flow Detail

### 1. Sensor Layer
Four independent threaded readers. Each pushes into its own ring buffer at native
rate. The multiplexer time-aligns buffers when the pipeline requests a frame.

### 2. Frame Assembly
Every ~10 ms (100 Hz, driven by IMU), the multiplexer collects the latest sample
from each sensor buffer. mmWave and UWB are slower (~20-50 Hz) — their last value
is held between updates.

### 3. Preprocessing
- **IMU**: Madgwick AHRS filter for orientation estimation, gravity vector removal,
  coordinate frame alignment (sensor frame → world frame).
- **mmWave**: Background subtraction (first N frames = static scene), DBSCAN
  clustering (ε=0.05 m, min_samples=3), centroid + spread per cluster.
- **UWB**: 5-sample median filter for outlier rejection, finite-difference velocity.
- **RFID**: Tag EPC → hand identity (pre-registered EPC → "left"/"right" mapping).

### 4. Feature Windows
Sliding windows of configurable length (default: 500 ms for motion, 200 ms for
posture, 100 ms for micro-Doppler). Features computed per window with stride =
framerate (real-time, no batch processing).

### 5. Detection
Five specialized detectors run in parallel. Each only activates when its
prerequisite sensors are available:

| Detector           | Active Sensors     | Gestures Covered                        |
|--------------------|--------------------|----------------------------------------|
| Motion Classifier  | IMU                | CW, ACW, Left, Right, Bye-Bye, Boxing |
| Posture Classifier | mmWave + RFID      | T-Arms, Raise Arms, Clapping           |
| Micro-Doppler      | mmWave (velocity)  | Soli                                   |
| Proximity Analyzer | UWB + IMU          | Pull, Push                             |
| Hand Gesture       | mmWave + IMU + RFID| Fist Open/Close, Palm Up/Down         |

Each detector outputs a **belief mass** vector over its gesture subset plus
an "unknown" mass.

### 6. Fusion
Dempster-Shafer combination rule merges the five mass functions. The frame of
discernment Θ = {G1..G15}. Classifiers that didn't fire for a frame contribute
vacuous belief (m(Θ)=1). Yager's rule handles conflict when classifiers disagree.

### 7. Temporal Filtering
- **Hysteresis**: onset at belief > 0.6, offset at belief < 0.3
- **Minimum duration**: gesture must persist ≥ 200 ms
- **Cooldown**: 300 ms gap before new gesture accepted
- **Kalman smoother**: reduces jitter in confidence trajectory

---

## Key Design Decisions

### Why not one neural network?
- A 15-class NN needs **all sensors online** and massive training data.
- It can't explain *why* it chose a gesture — debugging failures is impossible.
- It conflates gestures that are physically impossible to co-occur (e.g., Soli
  and Clapping use fundamentally different sensor signatures).

### Why Dempster-Shafer instead of softmax averaging?
- DS theory explicitly models **ignorance** (m(Θ) > 0) — "I don't know."
- It handles **conflict** — if IMU says "Bye-Bye" and mmWave says "Raise Arms,"
  a softmax average might pick "T-Arms" (nonsense). DS detects high conflict and
  rejects the frame.
- It composes naturally: add a new sensor → add its mass function, no retraining.

### Why HMM for sequential gestures?
Bye-Bye, boxing, and circular gestures have temporal *structure*. A static
classifier on a single window sees one snapshot. An HMM models the transition
probabilities between hidden states (phases of the gesture).

### Why RFID for hand identity?
mmWave point clouds can't tell left hand from right hand without spatial
heuristics that break when people cross their arms or turn sideways. A
lightweight RFID tag on each wrist gives absolute identity with zero ambiguity.

### Minimum sensor principle
Each gesture is detected by its *most discriminative* sensor(s), not all of them.
This means:
- Gestures work even if a sensor disconnects (graceful degradation).
- Less data → faster pipeline → lower latency.
- Fewer false positives from irrelevant sensor noise.
