"""PyTorch Dataset classes for multi-modal gesture data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class GestureDataset(Dataset):
    """Multi-modal gesture dataset for training.

    Each sample is a tuple (imu, rfid, uwb, label) where each modality
    is a tensor of shape (timesteps, channels).
    """

    def __init__(
        self,
        imu: np.ndarray,
        rfid: np.ndarray,
        uwb: np.ndarray,
        labels: np.ndarray,
        *,
        augment: bool = False,
    ) -> None:
        self.imu = torch.from_numpy(imu).float()
        self.rfid = torch.from_numpy(rfid).float()
        self.uwb = torch.from_numpy(uwb).float()
        self.labels = torch.from_numpy(labels).long()
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        imu = self.imu[idx]
        rfid = self.rfid[idx]
        uwb = self.uwb[idx]
        label = self.labels[idx]

        if self.augment:
            imu = self._augment_imu(imu)
            rfid = self._augment_rfid(rfid)
            uwb = self._augment_uwb(uwb)

        return {"imu": imu, "rfid": rfid, "uwb": uwb, "label": label}

    @staticmethod
    def _augment_imu(x: torch.Tensor) -> torch.Tensor:
        # Small Gaussian noise + random time shift
        noise = torch.randn_like(x) * 0.02
        shift = torch.randint(-5, 6, (1,)).item()
        if shift != 0:
            x = torch.roll(x, shift, dims=0)
        return x + noise

    @staticmethod
    def _augment_rfid(x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * 0.03
        shift = torch.randint(-3, 4, (1,)).item()
        if shift != 0:
            x = torch.roll(x, shift, dims=0)
        return x + noise

    @staticmethod
    def _augment_uwb(x: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(x) * 0.5
        shift = torch.randint(-5, 6, (1,)).item()
        if shift != 0:
            x = torch.roll(x, shift, dims=0)
        return x + noise

    @classmethod
    def from_npz(
        cls, path: Path | str, *, augment: bool = False
    ) -> tuple[GestureDataset, GestureDataset]:
        """Load a dataset from .npz and return (train, val) splits."""
        data = np.load(path)
        imu = data["imu"]
        rfid = data["rfid"]
        uwb = data["uwb"]
        labels = data["labels"]

        # Shuffle and split
        n = len(labels)
        indices = np.random.default_rng(42).permutation(n)
        split = int(n * 0.8)

        train_idx = indices[:split]
        val_idx = indices[split:]

        train_ds = cls(
            imu[train_idx], rfid[train_idx], uwb[train_idx], labels[train_idx],
            augment=augment,
        )
        val_ds = cls(
            imu[val_idx], rfid[val_idx], uwb[val_idx], labels[val_idx],
            augment=False,
        )
        return train_ds, val_ds
