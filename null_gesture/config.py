"""Central configuration — IMU-only."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent

GESTURES: Final[list[str]] = [
    "standing_still", "push", "pull", "left", "right",
    "up", "down", "clockwise", "anti_clockwise",
    "bye_bye", "palm_up", "palm_down",
]
NUM_GESTURES: Final = len(GESTURES)


@dataclass
class IMUConfig:
    tcp_host: str = "127.0.0.1"
    tcp_port: int = 9999
    sample_rate_hz: int = 50
    window_seconds: float = 2.0
    channels: int = 6
    timesteps: int = 100

    def __post_init__(self) -> None:
        self.timesteps = int(self.sample_rate_hz * self.window_seconds)


imu_config = IMUConfig()
