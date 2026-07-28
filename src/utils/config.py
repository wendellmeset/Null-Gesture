"""Configuration loading and management."""

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path = "config/pipeline.yaml") -> dict[str, Any]:
    """Load the pipeline configuration YAML file.

    Args:
        config_path: Path to pipeline.yaml.

    Returns:
        Config dict with defaults filled in.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    # Apply defaults for any missing sections
    defaults = _default_config()
    config = _deep_merge(defaults, config)

    return config


def load_gestures_config(gestures_path: str | Path = "config/gestures.yaml") -> dict[str, Any]:
    """Load the gestures configuration YAML file.

    Args:
        gestures_path: Path to gestures.yaml.

    Returns:
        Gestures config dict.
    """
    path = Path(gestures_path)
    if not path.exists():
        raise FileNotFoundError(f"Gestures config file not found: {path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)


def _default_config() -> dict[str, Any]:
    """Return default pipeline configuration."""
    return {
        "sensors": {
            "imu": {
                "port": "/dev/ttyUSB0",
                "baud": 115200,
                "buffer_size_sec": 2.0,
            },
            "mmwave": {
                "port": "/dev/ttyACM0",
                "baud": 115200,
                "config": "Readers/configs/mmwave_hand_50cm.cfg",
                "buffer_size_sec": 2.0,
            },
            "rfid": {
                "enabled": False,
                "port": None,
                "baud": 115200,
            },
        },
        "segmenter": {
            "metric_weights": {
                "imu_accel": 0.4,
                "imu_gyro": 0.3,
                "mmwave_range": 0.3,
            },
            "noise_ema_alpha": 0.01,
            "threshold_on_multiplier": 2.5,
            "threshold_off_multiplier": 1.8,
            "min_gesture_duration_ms": 200,
            "max_gesture_duration_ms": 5000,
            "quiet_confirm_ms": 300,
        },
        "fusion": {
            "transition_self_stay": 0.92,
            "transition_to_idle": 0.15,
            "idlestay": 0.98,
            "detection_threshold": 0.7,
            "debounce_frames": 3,
            "cooldown_ms": 600,
            "boxing_cooldown_ms": 300,
        },
        "output": {
            "print_gestures": True,
            "log_file": None,
            "visualization": False,
            "osc_output": None,
        },
        "models": {
            "imu_model_path": "models/imu_rf.pkl",
            "mmwave_model_path": "models/mmwave_hybrid.pkl",
            "calibration_profile": "profiles/default.npz",
            "selected_features": 100,
        },
    }


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
