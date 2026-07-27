# Gesture Detection Design

## Detector Architecture

Each detector is a self-contained module that:
1. Receives only its relevant sensor features
2. Outputs a **belief mass vector** over its gesture subset plus Θ (ignorance)
3. Runs in its own thread with configurable window/strike rate

---

## 1. Motion Gesture Detector (IMU)

### Input
- 500 ms sliding window (50 samples at 100 Hz), stride = 1 sample
- Features: accel_{x,y,z}, gyro_{x,y,z}, accel_magnitude, gyro_magnitude

### Feature Extraction
```
Statistical (per window, per channel):
  mean, variance, skewness, kurtosis, min, max, range

Spectral (FFT over window, per channel):
  dominant_freq, spectral_centroid, spectral_bandwidth,
  energy_ratio(low 0-3Hz / high 3-50Hz)

Temporal:
  zero_crossing_rate (all 6 channels)
  jerk (derivative of acceleration magnitude)
  mean cross-rate (rate of sign changes)
```

### Classification Architecture

#### HMM (Hidden Markov Model) — for sequential gestures
Used for: Bye-Bye, One Arm Boxing, Two-Arm Boxing

- **States**: 3-5 hidden states representing gesture phases
  - Bye-Bye: {left_swing, right_swing, transition}
  - Boxing: {prep, punch, retract, rest}
- **Observations**: Gaussian emission probabilities over feature vectors
- **Training**: Baum-Welch (EM) per gesture class
- **Inference**: Forward algorithm → log-likelihood per HMM → normalized to belief mass

Rationale: These gestures are defined by their *temporal structure* (sequence of
phases), not a single snapshot. A static classifier on one window sees only one
phase. HMM models the full trajectory.

#### DTW (Dynamic Time Warping) — for template gestures
Used for: Clockwise, Anti-Clockwise, Left, Right

- **Templates**: 5-10 exemplar gyro_z trajectories (CW/ACW) and accel_x
  trajectories (Left/Right), recorded at calibration or from a reference dataset
- **Matching**: Warp input window against each template, compute normalized
  DTW distance
- **Belief**: `belief = exp(-λ * min_distance)` normalized across templates

Rationale: These gestures have a canonical shape that may vary in speed. DTW
aligns temporally before comparing.

#### Random Forest — for catch-all classification
Fallback classifier on statistical + spectral features. Trained on labeled
examples. 30 trees, max depth 8 (small, fast).

### Output
```python
{
    "clockwise": 0.0,
    "anti_clockwise": 0.0,
    "left": 0.0,
    "right": 0.0,
    "bye_bye": 0.0,
    "one_arm_boxing": 0.0,
    "two_arm_boxing": 0.0,
    "unknown": 0.0,  # Θ — belief assigned to "any of the above"
}
```

---

## 2. Posture Gesture Detector (mmWave + RFID)

### Input
- 200 ms window (~10 mmWave frames at ~50 Hz)
- Features derived from point cloud clustering

### mmWave Preprocessing
1. **Background subtraction**: first 20 frames = static scene. Median of static
   points removed from subsequent frames.
2. **DBSCAN clustering**: ε = 0.05 m (hand-scale), min_samples = 3
3. **Cluster features**:
   - centroid (x, y, z) per cluster
   - spread (σ_x, σ_y, σ_z) per cluster
   - point_count per cluster
   - number_of_clusters in frame

### Feature Vector (per window)
```
centroid_y_mean, centroid_y_velocity  → for Raise Arms
spread_x, spread_y                    → for T-Arms
cluster_count_mean, cluster_count_variance → for Clapping
cluster_merge_events                  → binary: did 2 clusters become 1?
```

### Classification: Random Forest
Small RF (20 trees, max_depth=6). 3-class: {T-Arms, Raise Arms, Clapping}.

### RFID Integration
RFID tag presence confirms *which* hands are in view. T-Arms requires both tags
detected. Clapping requires both tags approaching. This eliminates false positives
from single-hand gestures that coincidentally look like T-Arms.

### Output
```python
{
    "t_arms": 0.0,
    "raise_arms": 0.0,
    "clapping": 0.0,
    "unknown": 0.0,
}
```

---

## 3. Micro-Doppler Detector (mmWave velocity channel)

### Physics
At 60 GHz, λ ≈ 5 mm. Small finger movements produce Doppler shifts:
```
f_d = 2 * v / λ
v = 0.05 m/s (finger rubbing) → f_d = 20 Hz
v = 1 m/s (hand wave)         → f_d = 400 Hz
```
Soli produces velocities in the 0.01-0.2 m/s range → Doppler shifts at 4-80 Hz,
which our radar's velocity resolution can capture.

### Input
- 100 ms window of velocity values from all detected points
- Focus on points within 0.2-0.8 m range (hand region)

### Feature Extraction
```
velocity_std_dev        → micro-Doppler bandwidth (key discriminator)
velocity_spectrum       → FFT of point velocity time series
peak_frequency          → dominant micro-Doppler frequency
spectral_flatness       → distinguishes Soli (tonal-ish) from noise (flat)
point_count_stability   → Soli has stable point count; hand waves don't
spatial_spread          → Soli has very tight spatial cluster (<10 cm)
```

### Classification: SVM with RBF kernel
Binary classifier: Soli vs. not-Soli. Trained on labeled micro-Doppler windows.

The key discriminator is that Soli produces **stable, small-amplitude velocity
oscillations in a tight spatial region**. Other gestures (waving, boxing) produce
large velocity variance over a wide spatial region.

### Output
```python
{
    "soli": 0.0,       # belief in Soli
    "unknown": 0.0,    # Θ
}
```

---

## 4. Proximity Gesture Detector (UWB + IMU)

### Input
- UWB distance time series (50-sample window, ~25 Hz after filtering)
- IMU acceleration magnitude (for confirmation)

### UWB Preprocessing
1. **Median filter**: 5-sample sliding window to reject outliers
2. **Velocity**: finite difference of filtered distance
3. **Acceleration**: finite difference of velocity

### Detection Logic
```
if mean(distance_velocity) < -THRESHOLD_PULL  # e.g., -0.3 m/s
  AND imu_accel_z > 0 (hand accelerating toward body in device frame):
    → Pull

if mean(distance_velocity) > +THRESHOLD_PUSH  # e.g., +0.3 m/s
  AND imu_accel_z < 0 (hand accelerating away):
    → Push
```

Thresholds are adaptive: computed as 3σ above the baseline noise level during
an initial calibration period of 2 seconds.

### Output
```python
{
    "pull": 0.0,
    "push": 0.0,
    "unknown": 0.0,
}
```
Belief mass is proportional to `|velocity| / threshold` (saturated at 1.0).

---

## 5. Hand Gesture Detector (mmWave + IMU + RFID)

### 5a. Opening and Closing Fist (mmWave)

**Physics**: An open hand presents more reflective surface area to mmWave than a
closed fist. Point count modulates as the hand opens and closes.

```
Feature: point_count in the "hand cluster" (cluster closest to RFID-tagged wrist)
Detection: If point_count oscillates at 0.5-3 Hz with amplitude > 30% of mean:
  → Fist Open/Close
```

Implementation: peak detection on bandpass-filtered (0.5-3 Hz) point count signal.

### 5b. Palm Up and Down (IMU)

**Physics**: Rotating the palm ≈ rotating the wrist ≈ integrated gyro_x.

```
Feature: integrated gyro_x over a 300 ms window
Detection:
  gyro_x_integral > +90°  → Palm Up
  gyro_x_integral < -90°  → Palm Down
```

RFID hand identity tells us which hand is rotating. Orientation relative to
gravity (from Madgwick filter) confirms the palm orientation.

### Output
```python
{
    "fist_open_close": 0.0,
    "palm_up_down": 0.0,
    "unknown": 0.0,
}
```

---

## Summary: Detector-Gesture Matrix

| Gesture             | Motion | Posture | μDoppler | Proximity | Hand |
|---------------------|--------|---------|----------|-----------|------|
| Pull                | ○      |         |          | ●         |      |
| Push                | ○      |         |          | ●         |      |
| Clockwise           | ●      |         |          |           |      |
| Anti-Clockwise      | ●      |         |          |           |      |
| Left                | ●      | ○       |          |           |      |
| Right               | ●      | ○       |          |           |      |
| Bye-Bye             | ●      |         |          |           |      |
| One Arm Boxing      | ●      |         |          |           |      |
| Clapping            | ○      | ●       |          |           |      |
| Two-Arm Boxing      | ●      |         |          |           |      |
| T-Arms              | ○      | ●       |          |           |      |
| Raise Arms          | ○      | ●       |          |           |      |
| Soli                |        |         | ●        |           |      |
| Open/Close Fist     |        | ○       |          |           | ●    |
| Palm Up/Down        | ○      |         |          |           | ●    |

● = primary detector, ○ = secondary/confirmation only
