# AGENTS.md

## What this repo is

Multi-modal gesture detection (IMU + RFID + UWB → 15 gesture classes). Python package `null_gesture/` invoked via `python -m null_gesture`.

## Reality vs. README

The README describes a full 3-sensor system with 15 gestures. The actual codebase now has:

- **IMU sensor** (`sensors/imu_sensor.py`) — TCP client reading 6-axis data from `esp32_reader.py`
- **UWB sensor** (`sensors/uwb_sensor.py`) — wraps `uwb-qorvo-tools` for FiRa TWR distance
- **Data collector** (`data/collector.py`) — synchronized IMU+UWB recording with guided prompts
- **Preprocessor** (`data/preprocessor.py`) — per-channel z-score normalization
- **Dataset** (`data/dataset.py`) — loads .npz with optional augmentation, train/val split
- **CLI** (`__main__.py`) — `debug`, `live`, `collect` subcommands
- **Config** (`config.py`) — all 15 gestures, IMU/UWB/collector configs

Still missing: RFID sensor, all models (imu_encoder, rfid_encoder, uwb_encoder, fusion, trainer), training pipeline, prediction pipeline.

## Quick commands

```bash
# Debug IMU (requires esp32_reader.py running)
python -m null_gesture debug imu --host 127.0.0.1

# Live detection GUI
python -m null_gesture live --imu-host 127.0.0.1

# Record IMU+UWB data (all 15 gestures, 25 samples each)
python -m null_gesture collect --dataset my_data --imu-host 127.0.0.1 \
    --uwb-controller /dev/ttyACM0 --uwb-controlee /dev/ttyACM1

# Record specific gestures, fewer samples
python -m null_gesture collect --dataset quick_test \
    --gestures pull,push,clapping --samples 10 --duration 3.0

# Record IMU only (no UWB)
python -m null_gesture collect --dataset imu_only --sensors imu
```

## Recording system

- `collector.py` drains sensor buffers before each sample to avoid stale data
- Each sample is a synchronized window: IMU (timesteps, 6) + UWB (timesteps, 1)
- Saved to `data/<name>/data.npz` + `metadata.json`
- `preprocessor.py`: `ZScoreNormalizer` — fit on train, save/load JSON
- `dataset.py`: `GestureDataset` — supports jitter/scale augmentation, `split()` for train/val

## Config constants

- `config.py` line 8: `DATA_DIR = PROJECT_ROOT / "data"` — where datasets go
- `config.py` line 9: `MODELS_DIR = PROJECT_ROOT / "models"` — where trained models go
- `config.py` line 10-24: full 15-gesture list + `GESTURE_TO_IDX` mapping
- IMU: 50 Hz, 2s window, 100 timesteps, 6 channels (ax,ay,az,gx,gy,gz)
- UWB: 50 Hz, 2s window, 100 timesteps, 1 channel (distance_cm)

## Setup

- Python 3.11+, venv-based (`venv/` is gitignored)
- `pip install -r requirements.txt` — numpy, pyserial, PyQt6, pyqtgraph, torch
- RFID requires `python-mercuryapi` C library: `bash install_mercury.sh` or `pip install wheels/python_mercuryapi-*.whl`
- UWB requires `uwb-qorvo-tools/` cloned from external repo (gitignored)

## Key quirks

- `esp32_reader.py` is a standalone script (not part of the package) that bridges ESP32 serial → TCP on port 9999. Must be running before any IMU work.
- `config.py` creates `logs/` directory on import (side effect).
- `uwb_sensor.py` adds `uwb-qorvo-tools/lib/*` to `sys.path` at connect time — import order matters.
- No tests, no linter, no type checker, no CI configured.
- `data/`, `models/`, `logs/`, `uwb-qorvo-tools/` are all gitignored.

## Hardware dependency

This project requires physical sensors. IMU debug mode is testable with just a TCP connection to `esp32_reader.py`. UWB requires both DWM3001CDK boards connected via USB.
