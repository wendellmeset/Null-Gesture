"""Central configuration for Null-Gesture. All tunables live here."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# ── Project root ────────────────────────────────────────────────────────────
PROJECT_ROOT: Final = Path(__file__).resolve().parent.parent
MODELS_DIR: Final = PROJECT_ROOT / "models"
DATA_DIR: Final = PROJECT_ROOT / "data"
LOGS_DIR: Final = PROJECT_ROOT / "logs"

MODELS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ── Gesture labels (order matters for classifier output) ────────────────────
GESTURES: Final[list[str]] = [
    "pull",
    "push",
    "clockwise",
    "anti_clockwise",
    "left",
    "right",
    "bye_bye",
    "one_arm_boxing",
    "clapping",
    "two_arm_boxing",
    "t_arms",
    "raise_arms",
    "soli",
    "open_close_fist",
    "palm_up_down",
]
NUM_GESTURES: Final = len(GESTURES)

# ── IMU ─────────────────────────────────────────────────────────────────────
@dataclass
class IMUConfig:
    """ESP32 + BMI270 IMU configuration."""
    tcp_host: str = "127.0.0.1"
    tcp_port: int = 9999
    sample_rate_hz: int = 50
    window_seconds: float = 2.0
    channels: int = 6  # ax, ay, az, gx, gy, gz
    timesteps: int = sample_rate_hz * int(window_seconds)  # 100

    def __post_init__(self) -> None:
        self.timesteps = int(self.sample_rate_hz * self.window_seconds)


# ── RFID ────────────────────────────────────────────────────────────────────
@dataclass
class RFIDConfig:
    """M7E Hecto UHF RFID reader configuration."""
    region: str = "NA"
    sample_rate_hz: int = 30
    window_seconds: float = 2.0
    channels: int = 2  # RSSI, phase
    timesteps: int = sample_rate_hz * int(window_seconds)  # 60
    read_power: int = 2700  # centi-dBm (27 dBm)
    on_time_ms: int = 500
    off_time_ms: int = 0

    def __post_init__(self) -> None:
        self.timesteps = int(self.sample_rate_hz * self.window_seconds)


# ── UWB ─────────────────────────────────────────────────────────────────────
@dataclass
class UWBConfig:
    """Qorvo DWM3001CDK UWB configuration."""
    sample_rate_hz: int = 50
    window_seconds: float = 2.0
    channels: int = 1  # distance_cm
    timesteps: int = sample_rate_hz * int(window_seconds)  # 100
    preamble_code: int = 10
    uwb_channel: int = 9  # UWB channel 5 or 9
    slot_span: int = 2400
    slots_per_rr: int = 6
    group_id: int = 0

    def __post_init__(self) -> None:
        self.timesteps = int(self.sample_rate_hz * self.window_seconds)


# ── Model ───────────────────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    """Neural network architecture hyperparameters."""
    # IMU encoder
    imu_conv_filters: list[int] = field(default_factory=lambda: [32, 64, 128])
    imu_kernel_size: int = 3
    imu_transformer_heads: int = 4
    imu_transformer_layers: int = 2
    imu_embed_dim: int = 128

    # RFID encoder
    rfid_conv_filters: list[int] = field(default_factory=lambda: [16, 32, 64])
    rfid_kernel_size: int = 3

    # UWB encoder
    uwb_conv_filters: list[int] = field(default_factory=lambda: [16, 32, 64])
    uwb_kernel_size: int = 3

    # Fusion
    fusion_hidden: list[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.3

    # Training
    batch_size: int = 32
    epochs: int = 150
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_split: float = 0.2
    early_stopping_patience: int = 25

    # Mixed precision
    use_amp: bool = True


# ── Default instances ───────────────────────────────────────────────────────
imu_config = IMUConfig()
rfid_config = RFIDConfig()
uwb_config = UWBConfig()
model_config = ModelConfig()
