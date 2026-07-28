# 01 — Architecture Overview

## Philosophy: Physics-Informed, Not Data-Hungry

The dominant approach in gesture recognition is **end-to-end deep learning**: collect
thousands of labeled samples, feed raw sensor streams into a CNN/LSTM/Transformer,
and hope the network learns useful representations. This works—but it is fragile,
data-hungry, and opaque.

**Null-Gesture takes a different path.** We embed domain knowledge about human
motion directly into the feature extraction stage, then use lightweight statistical
models that can be calibrated with a **single example per gesture** (~30 seconds
total). The result is a system that is:

| Property            | Naïve End-to-End DNN      | Null-Gesture                        |
|---------------------|---------------------------|-------------------------------------|
| Training data       | 100s–1000s per gesture    | 1–3 examples per gesture           |
| Calibration time    | 30+ minutes               | ~30 seconds                        |
| Person portability  | Retrain or fine-tune      | Immediate after short calibration  |
| Interpretability    | Black box                 | Every feature has physical meaning |
| Robustness to drift | Poor                      | Physics anchors are drift-resilient |
| Model size          | 10s–100s MB               | < 5 MB                             |
| Inference speed     | GPU often needed          | CPU real-time at 100 Hz            |

## System Architecture

```
┌───────────────────────────────────────────────────────────┐
│                    PHYSICAL LAYER                          │
│  ┌──────────────────────┐  ┌───────────────────────────┐  │
│  │  ESP32 + BMI270 IMU  │  │  TI IWRL6432 60GHz Radar  │  │
│  │  (Wrist-worn, 100Hz) │  │  (Desktop-facing, ~20fps) │  │
│  └──────────┬───────────┘  └───────────┬───────────────┘  │
└─────────────┼──────────────────────────┼──────────────────┘
              │                          │
              ▼                          ▼
┌───────────────────────────────────────────────────────────┐
│                   ACQUISITION LAYER                        │
│  ┌──────────────────────┐  ┌───────────────────────────┐  │
│  │  IMUReader            │  │  MMWaveReader              │  │
│  │  Readers library      │  │  Readers library           │  │
│  └──────────┬───────────┘  └───────────┬───────────────┘  │
└─────────────┼──────────────────────────┼──────────────────┘
              │                          │
              ▼                          ▼
┌───────────────────────────────────────────────────────────┐
│                 PREPROCESSING LAYER                        │
│  ┌──────────────────────┐  ┌───────────────────────────┐  │
│  │  IMU Processor        │  │  mmWave Processor          │  │
│  │  • Madgwick AHRS      │  │  • Range-Doppler heatmap   │  │
│  │  • Gravity removal    │  │  • Point cloud filtering   │  │
│  │  • Jerk computation   │  │  • Trajectory extraction   │  │
│  │  • Ring buffer (2s)   │  │  • Micro-Doppler spect.    │  │
│  └──────────┬───────────┘  └───────────┬───────────────┘  │
│             │                          │                   │
│             └──────────┬───────────────┘                   │
│                        ▼                                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Gesture Segmenter (Energy-based + Adaptive Threshold)│  │
│  │  Detects: gesture start / end / ongoing              │  │
│  │  No ML — purely signal-processing                    │  │
│  └──────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────┘
                         │
                         ▼
┌───────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING                       │
│  ┌──────────────────────┐  ┌───────────────────────────┐  │
│  │  IMU Features (~200)  │  │  mmWave Features (~150)    │  │
│  │  • Spectral (FFT)     │  │  • Range-Doppler moments  │  │
│  │  • Temporal stats     │  │  • Trajectory curvature   │  │
│  │  • Orientation quat.  │  │  • Velocity histograms    │  │
│  │  • Motion primitives  │  │  • Point cloud geometry   │  │
│  └──────────┬───────────┘  └───────────┬───────────────┘  │
└─────────────┼──────────────────────────┼──────────────────┘
              │                          │
              ▼                          ▼
┌───────────────────────────────────────────────────────────┐
│                 CLASSIFICATION LAYER                       │
│  ┌──────────────────────┐  ┌───────────────────────────┐  │
│  │  IMU Classifier       │  │  mmWave Classifier         │  │
│  │  Random Forest        │  │  CNN (R-D) + RF (stats)   │  │
│  │  → confidence vector  │  │  → confidence vector       │  │
│  └──────────┬───────────┘  └───────────┬───────────────┘  │
└─────────────┼──────────────────────────┼──────────────────┘
              │                          │
              ▼                          ▼
┌───────────────────────────────────────────────────────────┐
│                    FUSION LAYER                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Bayesian Multi-Modal Fusion                         │  │
│  │  • Sensor confidence weighting                       │  │
│  │  • Temporal smoothing (HMM forward filter)           │  │
│  │  • Gesture grammar constraints                       │  │
│  │  • Output: gesture label + confidence + latency      │  │
│  └──────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────┘
                         │
                         ▼
                   ┌──────────┐
                   │  OUTPUT  │
                   │ Gesture  │
                   │ Label    │
                   └──────────┘
```

## Key Architectural Decisions

### 1. Two-Sensor Strategy (IMU + mmWave)

We use the **IMU** and **mmWave radar** as complementary sensors. The RFID reader
is held in reserve for optional environment-calibration tags, but is not required
for the core pipeline—minimizing setup complexity.

- **mmWave radar** provides non-contact spatial tracking: range, angle, velocity.
  It is the primary sensor for spatial gestures (Pull, Push, Left, Right, Raise Arms).
- **IMU** provides high-rate (100 Hz) acceleration and angular velocity at the wrist.
  It is the primary sensor for fine rotational motion (Clockwise, Palm Up/Down).

Each sensor has blind spots. The fusion layer is designed to exploit their
complementary strengths.

### 2. Motion Primitive Decomposition

Instead of treating 15 gestures as 15 opaque classes, we decompose every gesture
into a sequence of **motion primitives** — physically meaningful atomic motions:

| Primitive     | Physical Meaning              | Detected By           |
|---------------|-------------------------------|-----------------------|
| TRANSLATE_X+  | Movement to the right         | mmWave angle + IMU ax |
| TRANSLATE_X-  | Movement to the left          | mmWave angle + IMU ax |
| TRANSLATE_Y+  | Movement forward (push)       | mmWave range + IMU ay |
| TRANSLATE_Y-  | Movement backward (pull)      | mmWave range + IMU ay |
| TRANSLATE_Z+  | Movement upward               | mmWave elevation      |
| TRANSLATE_Z-  | Movement downward             | mmWave elevation      |
| ROTATE_CW     | Clockwise rotation            | IMU gz                |
| ROTATE_CCW    | Anti-clockwise rotation       | IMU gz                |
| OSCILLATE     | Periodic back-and-forth       | Spectral peak in both |
| IMPULSE       | Sudden burst of energy        | Jerk spike            |
| HOLD          | Static pose maintained        | Low variance in both  |

A gesture is then defined as a **sequence of primitives** over a time window. For
example:
- **Pull** = TRANSLATE_Y- with decreasing range
- **Clockwise** = ROTATE_CW with period ~0.5–2s
- **Bye-Bye** = OSCILLATE in Z-axis at 2–5 Hz
- **Two-Arm Boxing** = alternating IMPULSE left + IMPULSE right at ~1–2 Hz

This decomposition is what enables **few-shot calibration**: instead of learning
15 complex decision boundaries, the system learns person-specific scaling factors
for ~11 primitives, then composes them via a rule engine.

### 3. Three-Stage Classification Cascade

**Stage 1 — Activity Detection (no ML)**
Energy-based segmentation using combined signal magnitude. Detects *when* a
gesture starts and ends. Zero training required.

**Stage 2 — Per-Sensor Classification (lightweight ML)**
Each sensor independently produces a confidence vector over the 15 gesture classes.
Models are trained offline on a diverse dataset and shipped with the system.

**Stage 3 — Bayesian Fusion + Temporal Smoothing**
The per-sensor confidence vectors are fused using a Bayesian model that accounts
for each sensor's known reliability per gesture class. A Hidden Markov Model
forward filter smooths predictions over time.

### 4. Few-Shot Calibration (the Moat)

This is the key differentiator. During a 30-second calibration routine:
1. User is prompted to perform each gesture once
2. System extracts feature vectors for each gesture
3. Per-person scaling factors are computed (e.g., arm length from range data,
   rotation speed from IMU)
4. Decision boundaries are shifted, not relearned

Other teams require users to record dozens of repetitions per gesture and retrain
neural networks. We reduce this to one example per gesture by using physics-informed
features that are **already person-invariant by design**.

## Pipeline Lifecycle

```
┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  SETUP  │───▶│CALIBRATE │───▶│  STREAM  │───▶│ SHUTDOWN │
│ Connect │    │ ~30 sec  │    │Real-time │    │ Cleanup  │
│ sensors │    │ 1 rep    │    │ detection│    │          │
└─────────┘    └──────────┘    └──────────┘    └──────────┘
```

## What Makes This Beat Other Teams

1. **Not a black box.** Every stage is inspectable. If the system misclassifies a
   gesture, you can trace exactly which feature caused it and adjust.

2. **Physics-informed features generalize.** A "Push" is defined by forward
   acceleration and increasing range—physics, not dataset statistics. This means
   the system works on day one with a new person, new room, new lighting.

3. **30-second calibration vs. 30-minute training.** The few-shot calibration
   is the headline feature. It's not just faster—it fundamentally changes the
   user experience from "lab experiment" to "practical system."

4. **Adaptive sensor fusion.** When the mmWave is occluded, IMU takes over.
   When the user's wrist is still, the mmWave provides spatial context. The
   system degrades gracefully, not catastrophically.

5. **Motion primitive reuse.** The 11 primitives are shared across all 15
   gestures. Improving a primitive improves all gestures that use it. This is
   compositional generalization—something end-to-end networks cannot do.

## Directory Structure

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
│   ├── preprocessing/           # Signal processing per sensor
│   ├── features/                # Feature extraction & primitives
│   ├── models/                  # Classifiers & calibration
│   ├── fusion/                  # Bayesian fusion & temporal smoothing
│   └── utils/                   # Config, visualization, helpers
├── README.md
└── requirements.txt
```
