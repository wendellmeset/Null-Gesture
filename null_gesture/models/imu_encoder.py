"""IMU encoder: Conv1D + Transformer for 6-channel accelerometer/gyroscope data.

Input:  (batch, timesteps, 6)  — accel xyz + gyro xyz
Output: (batch, embed_dim)     — feature vector
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from null_gesture.config import IMUConfig, imu_config


class ResidualConvBlock(nn.Module):
    """Conv1D → BatchNorm → ReLU → Conv1D → BatchNorm with residual connection."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding="same")
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding="same")
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.pool = nn.MaxPool1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + residual)
        return self.pool(out)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer input."""

    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class IMUEncoder(nn.Module):
    """Hierarchical Conv1D encoder → Transformer → global pooling → embedding."""

    def __init__(
        self,
        config: IMUConfig | None = None,
        conv_filters: list[int] | None = None,
        kernel_size: int = 3,
        transformer_heads: int = 4,
        transformer_layers: int = 2,
        embed_dim: int = 128,
    ) -> None:
        super().__init__()
        cfg = config or imu_config

        filters = conv_filters or [32, 64, 128]
        in_ch = cfg.channels

        # Conv1D stack with residual blocks
        self.conv_blocks = nn.ModuleList()
        ch = in_ch
        for f in filters:
            self.conv_blocks.append(ResidualConvBlock(ch, f, kernel_size))
            ch = f

        self.post_conv_dim = ch
        # Project to transformer dim
        self.input_proj = nn.Linear(ch, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=transformer_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        self.output_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → (B, embed_dim)"""
        # Conv expects (B, C, T)
        x = x.permute(0, 2, 1)  # (B, C, T)
        for block in self.conv_blocks:
            x = block(x)  # (B, C', T//2^k)
        x = x.permute(0, 2, 1)  # (B, T', C')
        x = self.input_proj(x)  # (B, T', embed_dim)
        x = self.pos_encoding(x)
        x = self.transformer(x)  # (B, T', embed_dim)
        # Global average + max pooling
        avg = x.mean(dim=1)
        max_val = x.max(dim=1).values
        return avg + max_val  # (B, embed_dim)
