# 03 — Signal Processing Pipeline

## Overview

The signal processing pipeline transforms raw sensor streams into clean,
segmented gesture windows ready for feature extraction. It has three
components:

1. **Per-sensor processors** — Clean and transform raw data
2. **Activity detector** — Detect when a gesture is occurring
3. **Gesture segmenter** — Extract the exact gesture window

## 1. IMU Processor

```
Raw IMU samples {ax, ay, az, gx, gy, gz, t}
                 │
                 ▼
┌─────────────────────────────────────┐
│  Step 1: Calibration offset removal │
│  (subtract resting bias)            │
└─────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Step 2: Madgwick AHRS Filter       │
│  Gyroscope + Accelerometer →        │
│  Orientation Quaternion (qw,qx,qy,qz)│
│  Gradient-descent, gain β=0.041     │
└─────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Step 3: Gravity Removal            │
│  Rotate gravity vector [0,0,1] by   │
│  quaternion → subtract from accel   │
│  → Linear acceleration [lax,lay,laz]│
└─────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Step 4: Jerk Computation           │
│  d(linear_accel)/dt                 │
│  → [jx, jy, jz] (m/s³)             │
└─────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Step 5: Norm Magnitudes            │
│  |a| = sqrt(ax²+ay²+az²)           │
│  |g| = sqrt(gx²+gy²+gz²)           │
│  |la| = sqrt(lax²+lay²+laz²)       │
│  |j| = sqrt(jx²+jy²+jz²)           │
└─────────────────────────────────────┘
                 │
                 ▼
          Processed IMU stream (100 Hz)
          Stored in 2-second ring buffer
```

### Madgwick AHRS Filter Details

The Madgwick filter is chosen over a Kalman filter because:
- It is computationally cheaper (no matrix inversion)
- It handles the non-linear quaternion kinematics natively
- It works well at 100 Hz with only accelerometer + gyroscope (no magnetometer)
- The gradient-descent gain β=0.041 is tuned for human hand motion dynamics

The filter outputs a unit quaternion `[qw, qx, qy, qz]` representing the
IMU's orientation relative to the Earth frame (gravity points down).

## 2. mmWave Processor

```
Raw mmWave frame {points (N,3), velocities (N,)}
                 │
                 ▼
┌─────────────────────────────────────┐
│  Step 1: Point Filtering            │
│  • Range gate: 0.05m < r < 1.5m     │
│  • SNR threshold: > 15 dB (if avail)│
│  • Statistical outlier removal      │
│    (Mahalanobis distance > 3σ)      │
└─────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Step 2: Centroid & Covariance      │
│  centroid = mean(points, axis=0)    │
│  cov = covariance matrix (3×3)      │
│  weighted centroid (by velocity)   │
└─────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Step 3: Range-Doppler Binning      │
│  Discretize range × velocity into   │
│  a 32×32 heatmap (0-1.5m × ±2 m/s) │
│  Accumulate points into bins        │
└─────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│  Step 4: Micro-Doppler Spectrogram  │
│  STFT on centroid velocity time     │
│  series (32 FFT bins, 50% overlap)  │
│  20-frame sliding window            │
└─────────────────────────────────────┘
                 │
                 ▼
          Processed mmWave stream (~20 fps)
          Stored in 2-second frame buffer
```

### Range-Doppler Heatmap

The R-D heatmap is a 2D histogram of detected points in (range, velocity) space.
It is the single most informative representation of radar data for gesture
recognition because:

- **Horizontal axis (range):** Tells us where the hand is
- **Vertical axis (velocity/Doppler):** Tells us how fast and in which direction
- **Spread:** The width of the cluster tells us about the hand size/pose
- **Multiple clusters:** Indicate multiple moving objects (two hands, etc.)

### Micro-Doppler Spectrogram

The micro-Doppler spectrogram is computed via Short-Time Fourier Transform (STFT)
on the centroid's velocity component over time:

```
For each frame i:
    velocity_signal = centroid_velocity[i-20 : i]  (1 second of data)
    spectrum = |FFT(velocity_signal)|
    → 32 frequency bins × 1 magnitude per bin
```

This reveals periodic motions:
- **2-5 Hz peak** → Bye-Bye, waving, shaking
- **10-50 Hz peak** → Soli (finger rubbing), tremor
- **Broadband transient** → Clapping, punching

## 3. Activity Detection & Gesture Segmentation

This is the most critical signal-processing component. It must detect **when**
a gesture starts and ends, without using ML, to pass the correct window to the
classifiers.

### Algorithm: Adaptive Energy Threshold with Hysteresis

```
For each new sample from either sensor:
    1. Compute combined activity metric:
       M(t) = α·|la(t)| + β·|g(t)| + γ·R(t)
       where:
         |la(t)| = linear acceleration norm (IMU)
         |g(t)|  = gyroscope norm (IMU)
         R(t)    = range change rate (mmWave)
         α,β,γ   = mixing weights (default 0.4, 0.3, 0.3)

    2. Track baseline noise level:
       noise[t] = (1-ε)·noise[t-1] + ε·M(t)  when M(t) < threshold
       (exponential moving average of quiet periods)

    3. Compute adaptive threshold:
       θ_on  = noise[t] * K_on   (K_on  = 2.5)
       θ_off = noise[t] * K_off  (K_off = 1.8)

    4. State machine with hysteresis:
       IDLE ──M(t) > θ_on──▶ ACTIVE
       ACTIVE ──M(t) < θ_off for 300ms──▶ IDLE

    5. On ACTIVE→IDLE transition:
       Extract window: [T_onset - 100ms, T_offset + 100ms]
       → Send to feature extractors
```

### Why This Approach

- **No training data needed** — purely signal-processing
- **Adaptive threshold** — adjusts to different noise floors per person/environment
- **Hysteresis** — prevents chatter at gesture boundaries
- **Multi-sensor metric** — uses whichever sensor is providing signal
- **Fixed computation budget** — O(1) per sample

### Gesture Boundary Refinement

After the initial segmentation, the boundaries are refined:

1. **Leading edge:** Walk backward from T_onset to find the last local minimum
   of M(t) before onset
2. **Trailing edge:** Walk forward from T_offset to find the first local minimum
   of M(t) after offset
3. **Minimum duration check:** If window < 200ms, discard (too short to be a gesture)
4. **Maximum duration check:** If window > 5s, split at local minima (compound gesture)

### Merging Overlapping Detections

When both sensors detect activity independently, their time windows are merged:

```
merged_start = min(IMU_start, mmWave_start) - 100ms
merged_end   = max(IMU_end, mmWave_end) + 100ms
```

This ensures we capture the full gesture even when one sensor detects it slightly
before the other.
