"""UWB encoder: Conv1D for 1-channel inter-hand distance data.

Input:  (batch, timesteps, 1)  — distance_cm
Output: (batch, 64)            — feature vector
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from null_gesture.config import UWBConfig, uwb_config


class UWBEncoder(nn.Module):
    """Stacked Conv1D blocks for UWB ranging distance processing."""

    def __init__(
        self,
        config: UWBConfig | None = None,
        conv_filters: list[int] | None = None,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        cfg = config or uwb_config

        filters = conv_filters or [16, 32, 64]
        in_ch = cfg.channels

        self.blocks = nn.ModuleList()
        ch = in_ch
        for f in filters:
            self.blocks.append(self._make_block(ch, f, kernel_size))
            ch = f

        self.output_dim = filters[-1]

    @staticmethod
    def _make_block(in_ch: int, out_ch: int, kernel_size: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding="same"),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding="same"),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, 1) → (B, output_dim)"""
        x = x.permute(0, 2, 1)  # (B, C, T)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=-1)
        return x
