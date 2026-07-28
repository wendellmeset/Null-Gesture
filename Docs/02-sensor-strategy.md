# 02 — Sensor Strategy

## Sensor Complementarity Matrix

Each of the 15 gestures has a different signature across the two sensors. This
matrix drives our fusion weighting.

| Gesture                  | mmWave Primary Signal              | IMU Primary Signal              | Best Sensor |
|--------------------------|-------------------------------------|----------------------------------|-------------|
| Pull                     | Range decreasing monotonically      | -ay acceleration (toward body)   | mmWave      |
| Push                     | Range increasing monotonically      | +ay acceleration (away)         | mmWave      |
| Clockwise                | Angular rotation in x-y plane       | -gz angular velocity            | IMU         |
| Anti-Clockwise           | Angular rotation in x-y plane       | +gz angular velocity            | IMU         |
| Left                     | -x angle shift, lateral velocity    | -ax acceleration                | mmWave      |
| Right                    | +x angle shift, lateral velocity    | +ax acceleration                | mmWave      |
| Bye-Bye                  | Oscillating elevation, periodic μD  | Oscillating gx, az (2-5 Hz)    | Both        |
| One Arm Boxing           | Impulsive range change, repeated    | Impulsive |ay| spikes (1-3 Hz)  | Both        |
| Clapping                 | Range burst + micro-Doppler spike   | Impulsive ax spike              | mmWave      |
| Two-Arm Boxing           | Alternating range L/R pattern       | (if both wrists instrumented)   | mmWave      |
| T-Arms                   | Static wide angle spread            | Static orientation hold         | mmWave      |
| Raise Arms               | +z elevation shift                  | +az acceleration + hold         | Both        |
| Soli                     | Micro-Doppler vibration (10-50 Hz)  | High-frequency vibration in g   | mmWave      |
| Opening & Closing Fist   | Subtle micro-Doppler modulation     | Subtle accel variance change    | IMU*        |
| Palm Up & Down           | Range+angle of palm orientation     | Gyro roll/pitch rotation        | IMU         |

*Fist gesture is the hardest for both sensors. We use IMU with a specialized
variance-ratio feature, but this gesture will have the lowest confidence.

## Sensor 1: TI IWRL6432 60GHz mmWave Radar

### Hardware Capabilities
- **Frequency:** 60 GHz (millimeter wave)
- **Range resolution:** ~4 cm
- **Velocity resolution:** ~0.1 m/s (Doppler)
- **Field of view:** ~120° azimuth, ~80° elevation (configurable)
- **Frame rate:** ~15-25 fps (configurable via .cfg)
- **Output:** Point cloud (x, y, z, Doppler velocity) per frame

### What Makes mmWave Exceptional for Gesture Recognition

1. **Privacy-preserving.** No camera. Cannot identify individuals. Works in
   darkness, through smoke, etc.

2. **Range-Doppler coupling.** Every detected point has both position AND
   velocity. This is unique to radar—cameras and IMUs don't provide this
   joint measurement. A point at 40cm moving toward the sensor at 0.5 m/s
   is very different from a point at 40cm moving laterally at 0.5 m/s.

3. **Micro-Doppler.** Small periodic motions (finger rubbing for Soli, hand
   tremor, fist opening/closing) create distinctive sidebands in the Doppler
   spectrum. This is the radar superpower.

4. **Multi-person capable.** The point cloud naturally separates multiple
   people in the field of view.

### mmWave Configuration

We use the `mmwave_hand_50cm.cfg` preset (shipped in `Readers/configs/`),
which is optimized for hand tracking at 50 cm range. Key parameters:
- Maximum unambiguous range: ~0.5 m
- Range resolution: ~4 cm
- Maximum unambiguous velocity: ~2 m/s
- Velocity resolution: ~0.1 m/s

### mmWave Data Flow

```
Radar frames (~20 fps)
    │
    ▼
Point cloud: (N, 3) positions + (N,) velocities
    │
    ▼
Frame buffer (last 2 seconds = ~40 frames)
    │
    ▼
Per-frame processing:
  ├── Point filtering (SNR, range gating, outlier removal)
  ├── Centroid & covariance computation
  ├── Range-Doppler heatmap accumulation
  └── Micro-Doppler spectrogram (STFT on centroid velocity)
```

## Sensor 2: ESP32 + BMI270 IMU

### Hardware Capabilities
- **Accelerometer:** ±2/4/8/16g, 16-bit
- **Gyroscope:** ±125/250/500/1000/2000 dps, 16-bit
- **Sample rate:** 100 Hz
- **Output:** 6-axis data (ax, ay, az in g; gx, gy, gz in dps)

### IMU Data Flow

```
IMU samples (100 Hz)
    │
    ▼
Ring buffer (last 2 seconds = 200 samples)
    │
    ▼
Per-sample processing:
  ├── Madgwick AHRS filter → orientation quaternion
  ├── Gravity removal → linear acceleration
  ├── Jerk computation (derivative of acceleration)
  └── Magnitude norms (|accel|, |gyro|, |jerk|)
    │
    ▼
Per-window processing (when gesture detected):
  ├── Spectral features (FFT of accel & gyro)
  ├── Statistical features (mean, var, skew, kurtosis, etc.)
  ├── Orientation features (quaternion statistics)
  └── Motion primitive extraction
```

### Placement

The IMU is worn on the **dominant wrist** via a simple elastic band. This:
- Captures hand orientation changes (gyroscope)
- Captures arm acceleration (accelerometer)
- Is unobtrusive and quick to put on

## Sensor 3: RFID (Optional / Enhancement)

The M7E Hecto RFID reader is available but **not required** for the core pipeline.
It can be used optionally for:

1. **Environment calibration tags** — Place 2-3 RFID tags at known positions in
   the environment. The mmWave radar can be auto-calibrated to these positions.

2. **Two-wrist disambiguation** — A small RFID tag on each wrist lets the system
   distinguish left from right arm in Two-Arm Boxing and similar gestures.

3. **Person identification** — A tag in the user's pocket could auto-load their
   calibration profile.

Since RFID increases setup complexity (tags, reader placement, potential read
failures), it is implemented as an **optional plugin** and is not part of the
core detection pipeline. The mmWave radar can handle two-arm disambiguation
spatially in most cases.

## Sensor Synchronization

The IMU streams at 100 Hz; the mmWave at ~20 fps. These are asynchronous streams.

Our approach: **timestamp-based alignment with interpolation.**

1. Every sample/frame is timestamped on arrival (wall clock, `time.time()`).
2. When a gesture window is detected (by the Segmenter), the system pulls all
   data in a [T_start, T_end] window from both sensors.
3. IMU samples are downsampled to match mmWave frame times using nearest-neighbor
   temporal alignment.
4. Features are extracted independently from each sensor's window, then fused at
   the classification stage.

This avoids the complexity of online Kalman-filter-style sensor fusion and is
more robust to variable frame rates.
