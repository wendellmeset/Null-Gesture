"""1D CNN for IMU gesture classification — ~35K params, <1ms CPU inference."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GestureCNN(nn.Module):
    """Lightweight 1D CNN for gesture classification from IMU windows.

    Input:  (batch, 6, timesteps)  — 6 channels (ax,ay,az,gx,gy,gz)
    Output: (batch, n_classes)     — raw logits
    """

    def __init__(self, n_classes: int = 11, n_channels: int = 6, dropout: float = 0.4):
        super().__init__()

        # Block 1 — wide kernel for broad temporal patterns
        self.conv1 = nn.Conv1d(n_channels, 24, kernel_size=9, padding=4, bias=False)
        self.bn1 = nn.BatchNorm1d(24)

        # Block 2
        self.conv2 = nn.Conv1d(24, 48, kernel_size=5, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(48)

        # Block 3 — narrow kernel for fine features
        self.conv3 = nn.Conv1d(48, 72, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(72)

        # Global average pool → fixed-size representation
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Classifier head
        self.fc1 = nn.Linear(72, 36)
        self.fc2 = nn.Linear(36, n_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T) → logits: (B, n_classes)"""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)  # (B, 128)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # logits
        return x

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
