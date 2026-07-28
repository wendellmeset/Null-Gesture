# 06 — Implementation Plan

## Technology Stack

| Component            | Choice                  | Rationale                               |
|----------------------|-------------------------|-----------------------------------------|
| Language             | Python 3.10+            | Rich ecosystem, cross-platform          |
| Numerical computing  | NumPy, SciPy            | Signal processing, statistics           |
| Machine learning     | scikit-learn            | Random Forest, calibration tools        |
| Deep learning (CNN)  | PyTorch (CPU mode)      | Lightweight CNN for R-D heatmaps        |
| Serial communication | pyserial                 | Sensor data acquisition                |
| Configuration        | PyYAML                   | Pipeline and gesture config            |
| Visualization (dev)  | Matplotlib               | Debug plots and live feedback          |
| Project management   | pip + requirements.txt   | Simple dependency management           |

## Module Breakdown

### `src/sensors/`
- **`buffer.py`** — Ring buffer for time-series sensor data with timestamp indexing
- **`orchestrator.py`** — Multi-threaded sensor streaming with synchronized readout

### `src/preprocessing/`
- **`imu_processor.py`** — Madgwick AHRS, gravity removal, jerk, norms
- **`mmwave_processor.py`** — Point filtering, R-D heatmap, micro-Doppler STFT
- **`segmenter.py`** — Energy-based activity detection with hysteresis

### `src/features/`
- **`imu_features.py`** — ~200 statistical/spectral/orientation features from IMU
- **`mmwave_features.py`** — ~150 trajectory/R-D/μD features from mmWave
- **`primitives.py`** — Motion primitive extraction from feature vectors

### `src/models/`
- **`imu_model.py`** — Random Forest classifier with confidence scoring
- **`mmwave_model.py`** — Hybrid CNN+RF classifier with confidence scoring
- **`calibration.py`** — Few-shot calibration routine (one example per gesture)

### `src/fusion/`
- **`bayesian.py`** — Weighted geometric mean fusion with reliability matrix
- **`ensemble.py`** — HMM forward filter + gesture debouncing + output formatting

### `src/utils/`
- **`config.py`** — YAML config loading with defaults
- **`visualization.py`** — Optional live plot of sensor streams and detections

## Data Flow (End-to-End)

```
main.py
  │
  ├─ Load config (config/pipeline.yaml + config/gestures.yaml)
  ├─ Load or initialize models
  │
  ├─ [Optional] Calibration mode (--calibrate flag)
  │   └─ calibration.py: prompt user → record 1 rep per gesture → save params
  │
  └─ Detection mode (default)
      │
      ├─ Start Sensor Orchestrator (background threads for IMU + mmWave)
      │   ├─ IMUReader.stream() → IMUProcessor → ring buffer
      │   └─ MMWaveReader.stream() → MMWaveProcessor → frame buffer
      │
      ├─ Main loop (runs at ~20 Hz, triggered by mmWave frames):
      │   │
      │   ├─ 1. Read latest data from both buffers
      │   │
      │   ├─ 2. Segmenter.update(imu_data, mmwave_data)
      │   │      → if gesture window detected:
      │   │
      │   ├─ 3. Extract IMU features from window
      │   ├─ 4. Extract mmWave features from window
      │   │
      │   ├─ 5. IMU model → P_IMU, c_IMU
      │   ├─ 6. mmWave model → P_mmW, c_mmW
      │   │
      │   ├─ 7. Bayesian fusion → P_fused
      │   ├─ 8. HMM forward filter → α_t
      │   ├─ 9. Gesture debouncer → final label
      │   │
      │   └─ 10. If gesture detected, emit output + optional visualization
      │
      └─ On SIGINT/Ctrl-C: graceful shutdown
```

## Threading Model

```
┌─────────────────┐
│   Main Thread   │  Gesture detection loop (sequential)
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐ ┌───────┐
│ IMU   │ │ mmWave│  Background daemon threads
│ Thread│ │ Thread│  (producer-consumer with thread-safe buffers)
└───────┘ └───────┘
```

- IMU thread: reads at 100 Hz, processes, appends to ring buffer
- mmWave thread: reads at ~20 fps, processes, appends to frame buffer
- Main thread: reads latest data from both buffers, runs detection pipeline

All buffers are thread-safe (using `threading.Lock` or `collections.deque`
with `append`/`popleft` atomicity).

## Configuration

### `config/pipeline.yaml`

```yaml
sensors:
  imu:
    port: "/dev/ttyUSB0"
    baud: 115200
    buffer_size_sec: 2.0
  mmwave:
    port: "/dev/ttyACM0"
    baud: 115200
    config: "Readers/configs/mmwave_hand_50cm.cfg"
    buffer_size_sec: 2.0

segmenter:
  metric_weights: {imu_accel: 0.4, imu_gyro: 0.3, mmwave_range: 0.3}
  noise_ema_alpha: 0.01
  threshold_on_multiplier: 2.5
  threshold_off_multiplier: 1.8
  min_gesture_duration_ms: 200
  max_gesture_duration_ms: 5000
  post_gesture_cooldown_ms: 300

fusion:
  transition_self_stay: 0.92
  transition_to_idle: 0.15
  idlestay: 0.98
  detection_threshold: 0.7
  debounce_frames: 3
  cooldown_ms: 600
  boxing_cooldown_ms: 300

output:
  print_gestures: true
  log_file: null
  visualization: false
  osc_output: null
```

### `config/gestures.yaml`

```yaml
gestures:
  pull:
    primitives: [TRANSLATE_Y_MINUS]
    priority_sensor: mmwave
    cooldown_group: linear
  push:
    primitives: [TRANSLATE_Y_PLUS]
    priority_sensor: mmwave
    cooldown_group: linear
  # ... (all 15 gestures)

primitives:
  TRANSLATE_X_PLUS:
    description: "Movement to the right"
    imu_features: [{feature: ax_mean, threshold: 0.1, direction: gt}]
    mmwave_features: [{feature: delta_x, threshold: 0.05, direction: gt}]
  # ... (all 11 primitives)
```

## Calibration Protocol

The calibration routine (`--calibrate` flag or `main.py calibrate`) takes
approximately 30 seconds:

```
$ python src/main.py calibrate

========================================
  Null-Gesture Calibration
========================================
  Please stand ~50cm in front of the radar
  Wear the IMU on your dominant wrist
========================================

[1/15] PULL — Pull your hand toward your body
        Ready... [3] [2] [1] GO!
        Recording... ✓

[2/15] PUSH — Push your hand away from your body
        Ready... [3] [2] [1] GO!
        Recording... ✓

...

[15/15] PALM UP & DOWN — Rotate your palm up, then down
        Ready... [3] [2] [1] GO!
        Recording... ✓

========================================
  Calibration complete!
  Profile saved to: profiles/default.npz
========================================
```

The calibration records one example per gesture. From each example, it extracts:
- Feature vector (for computing class prototypes)
- Range scale (arm length normalization)
- Velocity scale (movement speed normalization)
- Orientation range (ROM normalization)

These are used to normalize features for this specific person. Models trained
on a diverse dataset use these normalized features for classification.

## Testing Strategy

1. **Unit tests** — Each module has isolated tests with synthetic data
2. **Integration tests** — Pipeline end-to-end with recorded sensor data
3. **Live testing** — Run `python src/main.py visualize` for real-time
   feedback with matplotlib visualization

## Performance Targets

| Metric              | Target        |
|---------------------|---------------|
| Detection latency   | < 100ms       |
| False positive rate | < 5%          |
| True positive rate  | > 90%         |
| CPU usage           | < 30% (1 core)|
| Memory usage        | < 200 MB      |
| Calibration time    | < 45 seconds  |
