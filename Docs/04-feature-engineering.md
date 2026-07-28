# 04 — Feature Engineering

## Philosophy

Feature engineering is where Null-Gesture earns its advantage. Rather than
feeding raw sensor data into a neural network and hoping it discovers useful
representations, we **explicitly encode domain knowledge about human hand
motion** into physically meaningful features.

This has three benefits:
1. **Person invariance** — Features like "ratio of Z to X acceleration" are
   independent of arm length or strength
2. **Few-shot compatibility** — Physics-based features cluster naturally, so
   one example per gesture is enough for calibration
3. **Interpretability** — Every feature has a name and a physical meaning,
   making the system debuggable

## Motion Primitive System

All 15 gestures are decomposed into combinations of 11 motion primitives:

```yaml
# From config/gestures.yaml
primitives:
  TRANSLATE_X_PLUS:    "Movement to the right"
  TRANSLATE_X_MINUS:   "Movement to the left"
  TRANSLATE_Y_PLUS:    "Movement forward / away from body"
  TRANSLATE_Y_MINUS:   "Movement backward / toward body"
  TRANSLATE_Z_PLUS:    "Movement upward"
  TRANSLATE_Z_MINUS:   "Movement downward"
  ROTATE_CW:           "Clockwise rotation"
  ROTATE_CCW:          "Anti-clockwise rotation"
  OSCILLATE_FAST:      "Periodic motion at 3-10 Hz"
  OSCILLATE_SLOW:      "Periodic motion at 0.5-3 Hz"
  IMPULSE:             "Sudden burst of energy"
  HOLD:                "Static pose with low variance"
```

### Gesture-to-Primitive Mappings

| Gesture                  | Primitive Sequence                                    |
|--------------------------|-------------------------------------------------------|
| Pull                     | TRANSLATE_Y_MINUS (dominant)                          |
| Push                     | TRANSLATE_Y_PLUS (dominant)                           |
| Clockwise                | ROTATE_CW (dominant)                                  |
| Anti-Clockwise           | ROTATE_CCW (dominant)                                 |
| Left                     | TRANSLATE_X_MINUS (dominant)                          |
| Right                    | TRANSLATE_X_PLUS (dominant)                           |
| Bye-Bye                  | OSCILLATE_FAST + TRANSLATE_Z (minor)                  |
| One Arm Boxing           | IMPULSE + TRANSLATE_Y (alternating signs)             |
| Clapping                 | IMPULSE (high magnitude) + brief HOLD before/after    |
| Two-Arm Boxing           | IMPULSE (alternating left/right arms)                 |
| T-Arms                   | HOLD (arms extended laterally)                        |
| Raise Arms               | TRANSLATE_Z_PLUS → HOLD at top                        |
| Soli                     | OSCILLATE_FAST (10-50 Hz micro-motion)                |
| Opening & Closing Fist   | OSCILLATE_SLOW (hand expansion/contraction)           |
| Palm Up & Down           | ROTATE_CW/CCW (roll axis rotation)                    |

## IMU Features (~200 features per window)

Features are computed over the segmented gesture window (typically 0.5–2 seconds
= 50–200 samples at 100 Hz).

### 1. Statistical Features (per axis: ax, ay, az, gx, gy, gz, lax, lay, laz, jx, jy, jz)

```
mean, std, min, max, range, rms,
skewness, kurtosis,
25th percentile, 50th percentile, 75th percentile,
interquartile range,
mean absolute deviation,
zero-crossing rate,
dominant frequency (from FFT),
spectral centroid,
spectral spread,
spectral entropy,
peak-to-peak amplitude
```

**~18 features × 12 axes = ~216 statistical features**

Key discriminators:
- **Skewness of ax:** Positive → acceleration to the right. Negative → left.
- **Gyro gz mean:** Positive → CCW rotation. Negative → CW rotation.
- **Az kurtosis:** High → sharp upward movement (Raise Arms).

### 2. Orientation Features (from quaternion)

```
mean(qw), mean(qx), mean(qy), mean(qz),
std(qw), std(qx), std(qy), std(qz),
mean(pitch), mean(roll), mean(yaw),
range(pitch), range(roll), range(yaw),
pitch at start, pitch at end, delta_pitch,
roll at start, roll at end, delta_roll,
yaw at start, yaw at end, delta_yaw,
path length in quaternion space (geodesic distance),
quaternion stability (variance of q over window)
```

**~24 orientation features**

Key discriminators:
- **Delta roll:** Large positive → Palm Up. Large negative → Palm Down.
- **Delta pitch:** Large positive → Raise Arms. Large negative → Lower Arms.
- **Path length in quaternion space:** High → complex rotation (Clockwise).

### 3. Cross-Axis Correlation Features

```
corr(ax, ay), corr(ax, az), corr(ay, az),
corr(gx, gy), corr(gx, gz), corr(gy, gz),
corr(lax, lay), corr(lax, laz), corr(lay, laz),
corr(|a|, |g|), corr(|a|, |j|),
mutual information between accel axes (binned)
```

**~12 correlation features**

Key discriminators:
- **corr(ax, az) high:** Diagonal movement (distinguishes Left from diagonal).
- **corr(|a|, |g|) high:** Coordinated rotation+translation (Clockwise).

### 4. Spectral Features (FFT-based)

For each of {ax, ay, az, gx, gy, gz, |a|, |g|, |la|}:

```
spectral energy in [0-1] Hz  (DC / static)
spectral energy in [1-3] Hz  (slow motion)
spectral energy in [3-8] Hz  (medium motion)
spectral energy in [8-20] Hz (fast motion)
spectral energy in [20-50] Hz (vibration / Soli)
spectral entropy
number of spectral peaks above threshold
frequency of highest peak
magnitude of highest peak
ratio of peak magnitude to total energy
```

**~10 features × 9 signals = ~90 spectral features**

Key discriminators:
- **Energy in [3-8] Hz for gyro:** Bye-Bye has strong peak here.
- **Energy in [20-50] Hz:** Soli detection.
- **Number of peaks:** Boxing has multiple harmonic peaks.

### 5. Jerk & Impulse Features

```
max(|jerk|) — peak jerk magnitude
mean(|jerk|) — average jerkiness
jerk peak count — number of jerk spikes above 2σ
jerk peak spacing std — regularity of impulses
ratio of max jerk to mean jerk — impulsiveness
time from window start to peak jerk
time from peak jerk to window end
```

**~7 jerk features**

Key discriminators:
- **High impulsiveness:** Clapping, Boxing (impulse gestures).
- **Low impulsiveness:** Raise Arms, T-Arms (smooth gestures).

### 6. Magnitude Ratio Features

```
ratio of X to Y acceleration magnitude
ratio of Z to horizontal (X+Y) acceleration magnitude
ratio of linear to gravitational acceleration
ratio of gyro to accel magnitude
ratio of dominant axis energy to total energy
```

**~5 ratio features**

Key discriminators:
- **Z/horizontal ratio high:** Raise Arms.
- **X/Y ratio:** Left vs Right vs Forward.

## mmWave Features (~150 features per window)

### 1. Centroid Trajectory Features

From the sequence of point cloud centroids (x, y, z, v) over the gesture window:

```
mean(x), std(x), range(x), start(x), end(x), delta(x)
mean(y), std(y), range(y), start(y), end(y), delta(y)
mean(z), std(z), range(z), start(z), end(z), delta(z)
mean(v), std(v), range(v), start(v), end(v), delta(v)
```

**~28 trajectory statistics**

Key discriminators:
- **delta(y) positive:** Push (hand moving away).
- **delta(y) negative:** Pull (hand moving toward).
- **delta(x) positive:** Right. **delta(x) negative:** Left.
- **delta(z) positive:** Raise Arms.

### 2. Trajectory Shape Features

```
total path length (sum of Euclidean distances between frames)
straightness = |end_pos - start_pos| / path_length
max curvature (2nd derivative of trajectory)
mean curvature
number of direction changes (sign flips in velocity)
circularity (fit to circle, residual error)
radius of best-fit circle
angular displacement (total accumulated angle)
```

**~8 shape features**

Key discriminators:
- **Straightness ~1:** Pull, Push, Left, Right (linear gestures).
- **Straightness < 0.5:** Clockwise, Anti-Clockwise (curved trajectories).
- **Low circularity error:** Clockwise/Anti-Clockwise (circular gestures).
- **High direction changes:** Boxing (back-and-forth).

### 3. Point Cloud Geometry Features

```
number of points (mean, std, range)
point cloud volume (product of XYZ ranges)
point cloud density (N / volume)
covariance eigenvalues (λ1, λ2, λ3)
anisotropy = (λ1-λ2)/(λ1+λ2+λ3)
planarity = (λ2-λ3)/(λ1+λ2+λ3)
sphericity = λ3/λ1
scatter = λ1 (primary axis spread)
```

**~12 point cloud features**

Key discriminators:
- **High planarity:** T-Arms (arms spread in a plane).
- **High scatter during impulse:** Boxing punch extends the arm.

### 4. Range-Doppler Heatmap Features (~20 frames × per-frame stats)

```
Per-frame:
  mean range of non-zero bins
  mean velocity of non-zero bins
  range spread (std of range distribution)
  velocity spread (std of velocity distribution)
  number of active bins
  range-velocity correlation

Aggregated over window:
  mean and std of each per-frame stat
```

**~6 per-frame × 2 aggregated = ~12 features per window**

### 5. Micro-Doppler Features

From the micro-Doppler spectrogram (STFT of centroid velocity):

```
mean spectral power
peak spectral frequency
spectral bandwidth
spectral centroid over time
number of distinct spectral tracks
entropy of spectral distribution
energy in [0-2] Hz / [2-5] Hz / [5-10] Hz / [10-50] Hz
ratio of high-frequency to low-frequency energy
temporal variance of spectral centroid
```

**~10 micro-Doppler features**

Key discriminators:
- **Strong [10-50] Hz energy:** Soli (finger rubbing).
- **Strong [2-5] Hz energy:** Bye-Bye, slow waving.
- **High spectral entropy:** Clapping (broadband impulsive).

### 6. Velocity Profile Features

From the sequence of centroid velocities:

```
velocity sign changes per second (zero-crossing rate)
mean absolute velocity
max velocity
velocity at gesture midpoint
velocity asymmetry (max positive / |max negative|)
velocity profile flatness (kurtosis)
```

**~6 velocity features**

Key discriminators:
- **Velocity asymmetry ~1:** Symmetric gesture (Bye-Bye).
- **Velocity asymmetry >>1 or <<1:** Directional gesture (Pull, Push, Left, Right).

## Feature Selection & Dimensionality Reduction

With ~350+ features, we apply:

1. **Mutual information ranking** — Features are ranked by MI with gesture labels
   (computed offline during model training)
2. **Top-100 retained** — Only the 100 most informative features are used for
   classification
3. **Per-sensor selection** — IMU and mmWave feature sets are selected independently

This keeps the classifiers fast while retaining all critical discriminants.
