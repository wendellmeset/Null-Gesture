"""Central configuration — IMU-only."""
from __future__ import annotations
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final = Path(__file__).resolve().parent.parent
LOGS_DIR: Final = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

GESTURES: Final[list[str]] = [
    "standing_still", "push", "pull", "left", "right",
    "clockwise", "anti_clockwise",
]
NUM_GESTURES: Final = len(GESTURES)

from dataclasses import dataclass, field

@dataclass
class IMUConfig:
    tcp_host: str = "127.0.0.1"
    tcp_port: int = 9999
    sample_rate_hz: int = 50
    window_seconds: float = 2.0
    channels: int = 6
    timesteps: int = sample_rate_hz * int(window_seconds)

    def __post_init__(self):
        self.timesteps = int(self.sample_rate_hz * self.window_seconds)

imu_config = IMUConfig()
