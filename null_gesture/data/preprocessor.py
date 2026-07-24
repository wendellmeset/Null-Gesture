"""Preprocessing: feature extraction, normalization, and data alignment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class Preprocessor:
    """Normalizes multi-modal gesture data and provides feature extraction.

    Each modality gets independent per-channel mean/std normalization.
    """

    def __init__(self) -> None:
        self._fitted = False
        self._imu_mean: np.ndarray | None = None
        self._imu_std: np.ndarray | None = None
        self._rfid_mean: np.ndarray | None = None
        self._rfid_std: np.ndarray | None = None
        self._uwb_mean: np.ndarray | None = None
        self._uwb_std: np.ndarray | None = None

    def fit(
        self,
        imu: np.ndarray,   # (N, T, 6)
        rfid: np.ndarray,  # (N, T, 2)
        uwb: np.ndarray,   # (N, T, 1)
    ) -> Preprocessor:
        """Compute per-channel mean and std for each modality."""
        # IMU: normalize per channel across all samples and timesteps
        imu_flat = imu.reshape(-1, imu.shape[-1])
        self._imu_mean = imu_flat.mean(axis=0)
        self._imu_std = imu_flat.std(axis=0) + 1e-8

        rfid_flat = rfid.reshape(-1, rfid.shape[-1])
        self._rfid_mean = rfid_flat.mean(axis=0)
        self._rfid_std = rfid_flat.std(axis=0) + 1e-8

        uwb_flat = uwb.reshape(-1, uwb.shape[-1])
        self._uwb_mean = uwb_flat.mean(axis=0)
        self._uwb_std = uwb_flat.std(axis=0) + 1e-8

        self._fitted = True
        return self

    def transform(
        self,
        imu: np.ndarray,
        rfid: np.ndarray,
        uwb: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply z-score normalization to all modalities."""
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")

        imu_norm = (imu - self._imu_mean) / self._imu_std
        rfid_norm = (rfid - self._rfid_mean) / self._rfid_std
        uwb_norm = (uwb - self._uwb_mean) / self._uwb_std
        return imu_norm.astype(np.float32), rfid_norm.astype(np.float32), uwb_norm.astype(np.float32)

    def fit_transform(
        self,
        imu: np.ndarray,
        rfid: np.ndarray,
        uwb: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit and transform in one call."""
        self.fit(imu, rfid, uwb)
        return self.transform(imu, rfid, uwb)

    def save(self, path: Path | str) -> None:
        """Save normalization parameters to JSON."""
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted preprocessor.")
        params = {
            "imu_mean": self._imu_mean.tolist(),
            "imu_std": self._imu_std.tolist(),
            "rfid_mean": self._rfid_mean.tolist(),
            "rfid_std": self._rfid_std.tolist(),
            "uwb_mean": self._uwb_mean.tolist(),
            "uwb_std": self._uwb_std.tolist(),
        }
        with open(path, "w") as f:
            json.dump(params, f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> Preprocessor:
        """Load normalization parameters from JSON."""
        with open(path) as f:
            params = json.load(f)
        pp = cls()
        pp._imu_mean = np.array(params["imu_mean"], dtype=np.float32)
        pp._imu_std = np.array(params["imu_std"], dtype=np.float32)
        pp._rfid_mean = np.array(params["rfid_mean"], dtype=np.float32)
        pp._rfid_std = np.array(params["rfid_std"], dtype=np.float32)
        pp._uwb_mean = np.array(params["uwb_mean"], dtype=np.float32)
        pp._uwb_std = np.array(params["uwb_std"], dtype=np.float32)
        pp._fitted = True
        return pp

    @property
    def fitted(self) -> bool:
        return self._fitted
