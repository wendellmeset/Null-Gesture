"""Central configuration — IMU + UWB multi-modal."""
from __future__ import annotations
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final = Path(__file__).resolve().parent.parent
LOGS_DIR: Final = PROJECT_ROOT / "logs"
DATA_DIR: Final = PROJECT_ROOT / "data"
MODELS_DIR: Final = PROJECT_ROOT / "models"
LOGS_DIR.mkdir(exist_ok=True)

GESTURES: Final[list[str]] = [
    "pull", "push", "clockwise", "anti_clockwise",
    "left", "right", "bye_bye", "one_arm_boxing",
    "clapping", "two_arm_boxing", "t_arms", "raise_arms",
    "soli", "open_close_fist", "palm_up_down",
]
GESTURE_TO_IDX: Final[dict[str, int]] = {g: i for i, g in enumerate(GESTURES)}
NUM_GESTURES: Final = len(GESTURES)

from dataclasses import dataclass, field


@dataclass
class IMUConfig:
    tcp_host: str = "127.0.0.1"
    tcp_port: int = 9999
    sample_rate_hz: int = 50
    window_seconds: float = 2.0
    channels: int = 6
    timesteps: int = 0

    def __post_init__(self):
        self.timesteps = int(self.sample_rate_hz * self.window_seconds)


@dataclass
class UWBConfig:
    controller_port: str = "/dev/ttyACM0"
    controlee_port: str = "/dev/ttyACM1"
    sample_rate_hz: int = 50
    window_seconds: float = 2.0
    channels: int = 1
    timesteps: int = 0
    ranging_interval_ms: int = 20

    def __post_init__(self):
        self.timesteps = int(self.sample_rate_hz * self.window_seconds)


@dataclass
class CollectorConfig:
    samples_per_gesture: int = 25
    duration_per_sample: float = 3.0
    countdown_seconds: float = 3.0
    rest_between_samples: float = 1.0


imu_config = IMUConfig()
uwb_config = UWBConfig()
collector_config = CollectorConfig()
