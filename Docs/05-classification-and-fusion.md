# 05 — Classification and Fusion

## Overview

The classification and fusion system operates in three stages:

1. **Per-sensor classification** — Each sensor independently produces a
   probability distribution over the 15 gesture classes
2. **Bayesian fusion** — The two distributions are combined, weighted by
   per-sensor per-class reliability
3. **Temporal smoothing** — A Hidden Markov Model forward filter smooths
   predictions across consecutive gesture windows

## Stage 1: Per-Sensor Classification

### IMU Classifier: Random Forest

**Why Random Forest?**
- Excellent on tabular, engineered features
- Provides calibrated probability outputs (via class frequencies in leaf nodes)
- Resistant to overfitting with small calibration data
- Fast inference (< 1ms for 100 features)
- Interpretable (feature importance rankings)
- No GPU needed

**Architecture:**
```
Input: 100-element feature vector (top MI-ranked IMU features)
    │
    ▼
Random Forest (100 trees, max_depth=12, min_samples_leaf=5)
    │
    ▼
Output: 15-element probability vector P_IMU(gesture | features)
        + confidence score c_IMU ∈ [0, 1]
```

**Confidence computation:**
```
c_IMU = 1 - entropy(P_IMU) / log(15)     (normalized entropy)
       * (tree_agreement / n_trees)       (proportion of trees voting majority)
       * activity_quality                 (SNR of the gesture window)
```

A low-confidence prediction (entropic distribution, disagreeing trees) will
be down-weighted in the fusion stage.

### mmWave Classifier: Two-Headed Hybrid

The mmWave data is richer and more complex than the IMU data, so we use a
hybrid architecture:

```
Input: Range-Doppler heatmap sequence (20 frames × 32 × 32)
    │
    ▼
┌─────────────────────────────┐
│  Lightweight 2D CNN         │
│  Conv2d(1→8, 3×3) + ReLU    │
│  MaxPool(2×2)               │
│  Conv2d(8→16, 3×3) + ReLU   │
│  MaxPool(2×2)               │
│  Global Average Pooling     │
│  → 16-dim embedding vector  │
└─────────────────────────────┘
    │
    ▼
    Concatenate with 100-dim statistical feature vector
    │
    ▼
┌─────────────────────────────┐
│  Random Forest (80 trees)   │
│  → 15-element probability   │
│    vector P_mmW(gesture)    │
└─────────────────────────────┘
```

**Why two-headed?**
The CNN head extracts spatial-temporal patterns from the raw R-D heatmap
(which captures complex motion signatures), while the RF head processes
engineered physics features. Concatenating both gives the best of both
worlds: learned representations AND physics-informed features.

The CNN is small enough to run on CPU in real time (< 5ms inference).

**Confidence computation:** Same normalized-entropy approach as IMU, plus:
```
c_mmW *= point_cloud_quality     (number of tracked points / expected)
       * range_validity          (is target in optimal 0.1-0.8m range?)
```

## Stage 2: Bayesian Multi-Modal Fusion

### Sensor Reliability Matrix

Each sensor has different reliability per gesture class, encoded in a
**prior reliability matrix R[15 × 2]**:

| Gesture                  | IMU Reliability | mmWave Reliability |
|--------------------------|-----------------|---------------------|
| Pull                     | 0.7             | 0.9                 |
| Push                     | 0.7             | 0.9                 |
| Clockwise                | 0.95            | 0.6                 |
| Anti-Clockwise           | 0.95            | 0.6                 |
| Left                     | 0.6             | 0.9                 |
| Right                    | 0.6             | 0.9                 |
| Bye-Bye                  | 0.85            | 0.85                |
| One Arm Boxing           | 0.8             | 0.8                 |
| Clapping                 | 0.5             | 0.9                 |
| Two-Arm Boxing           | 0.3             | 0.85                |
| T-Arms                   | 0.5             | 0.8                 |
| Raise Arms               | 0.8             | 0.85                |
| Soli                     | 0.2             | 0.85                |
| Opening & Closing Fist   | 0.55            | 0.4                 |
| Palm Up & Down           | 0.9             | 0.3                 |

These values are derived from the physics of each gesture:
- **IMU dominates** rotational gestures (Clockwise, Palm Up/Down) because
  the gyroscope directly measures rotation
- **mmWave dominates** spatial gestures (Pull, Push, Left, Right, Clapping)
  because range and angle directly measure spatial motion
- **Both contribute** to complex gestures (Bye-Bye, Boxing, Raise Arms)

### Fusion Equation

The fused probability for gesture g is:

```
P_fused(g) = P_IMU(g)^w_IMU(g) * P_mmW(g)^w_mmW(g) / Z

where:
  w_IMU(g)  = R[g, 0] * c_IMU               (reliability × confidence)
  w_mmW(g)  = R[g, 1] * c_mmW
  Z         = normalization constant
```

This is a **weighted geometric mean** of the two probability distributions.
The geometric mean (rather than arithmetic) penalizes disagreement — if one
sensor says "definitely Pull" and the other says "definitely NOT Pull," the
fused probability for Pull is low. This conservative approach reduces false
positives.

### Handling Sensor Dropout

If one sensor is disconnected or producing no data:
- Set that sensor's confidence `c = 0`
- Those weights become 0, effectively ignoring that sensor
- The fused output = the remaining sensor's output (with reliability weighting)

This graceful degradation is critical for a demo — if the IMU battery dies
or the mmWave is blocked, the system continues with reduced but non-zero
accuracy.

## Stage 3: Temporal Smoothing (HMM Forward Filter)

Raw frame-by-frame classification is noisy. We model the gesture sequence as
a Hidden Markov Model and apply a forward filter.

### State Space

```
States: {IDLE, Pull, Push, CW, CCW, Left, Right, ByeBye,
         Boxing_1, Clap, Boxing_2, TArms, RaiseArms,
         Soli, Fist, PalmUpDown}
+ optionally: {AFTER_Pull, AFTER_Push, ...} for post-gesture hold states
```

### Transition Model

```
P(stay in same gesture)        = 0.92     (high self-transition)
P(transition to any other)     = 0.08/14  (uniform distribution)
P(transition to IDLE)          = 0.15     (after a gesture, likely go idle)
P(stay in IDLE)                = 0.98     (mostly idle)
```

These are **hardcoded Bayesian priors** — not trained, but designed from
domain knowledge. They encode the heuristic that gestures persist for
multiple frames and transitions between gestures are rare.

### Forward Algorithm

For each new frame t with fused probability vector P_fused:

```
α_t(g) = P_fused(g) * Σ_{prev} α_{t-1}(prev) * P(g | prev)

where:
  α_0 = uniform distribution
  P(g | prev) = transition probability
```

The output at time t is `argmax_g α_t(g)`, and the system only reports a
gesture when `α_t(g) > 0.7` (high confidence threshold).

### Gesture Debouncing

To prevent flickering (rapid alternation between two gestures):

1. A gesture must be the argmax for **3 consecutive frames** to be reported
2. Once reported, a **600ms cooldown** prevents re-reporting the same gesture
3. Boxing gestures (One Arm, Two Arm) can be reported **per-punch** (no cooldown
   for repeated gestures in this class)

## Output Format

When a gesture is detected, the system emits:

```python
{
    "gesture": "Pull",
    "confidence": 0.87,
    "latency_ms": 45,           # time from gesture end to detection
    "sensor_contributions": {
        "imu": 0.42,
        "mmwave": 0.58
    },
    "features": {
        "primitive": "TRANSLATE_Y_MINUS",
        "displacement_m": 0.23,
        "peak_velocity_ms": 0.8,
        "duration_ms": 680
    },
    "timestamp": 1722123456.789
}
```

This rich output makes the system transparent and debuggable — every detection
can be traced back to its physical evidence.
