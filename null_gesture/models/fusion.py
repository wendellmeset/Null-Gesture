"""Multi-modal fusion model: combines IMU + RFID + UWB encoders into a classifier.

Architecture: Late fusion — each modality has its own encoder, outputs are
concatenated and passed through a shared MLP classifier head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from null_gesture.config import ModelConfig, NUM_GESTURES, model_config as default_model_config
from null_gesture.models.imu_encoder import IMUEncoder
from null_gesture.models.rfid_encoder import RFIDEncoder
from null_gesture.models.uwb_encoder import UWBEncoder


class GestureFusionModel(nn.Module):
    """Late-fusion multi-modal gesture classifier.

    Each modality is encoded independently, features are concatenated,
    then classified through a shared MLP head.

    Supports single-modality inference via `modalities` parameter:
    - "all" (default): use IMU + RFID + UWB
    - "imu": IMU only (UWB and RFID zero-padded)
    - "rfid": RFID only
    - "uwb": UWB only
    - Any combination like "imu+uwb", "rfid+uwb", "imu+rfid"
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        cfg = config or default_model_config

        self.imu_encoder = IMUEncoder(
            conv_filters=cfg.imu_conv_filters,
            kernel_size=cfg.imu_kernel_size,
            transformer_heads=cfg.imu_transformer_heads,
            transformer_layers=cfg.imu_transformer_layers,
            embed_dim=cfg.imu_embed_dim,
        )
        self.rfid_encoder = RFIDEncoder(
            conv_filters=cfg.rfid_conv_filters,
            kernel_size=cfg.rfid_kernel_size,
        )
        self.uwb_encoder = UWBEncoder(
            conv_filters=cfg.uwb_conv_filters,
            kernel_size=cfg.uwb_kernel_size,
        )

        self._imu_dim = self.imu_encoder.output_dim
        self._rfid_dim = self.rfid_encoder.output_dim
        self._uwb_dim = self.uwb_encoder.output_dim

        # MLP classifier head — always expects full fusion_dim
        fusion_dim = self._imu_dim + self._rfid_dim + self._uwb_dim
        layers: list[nn.Module] = []
        in_dim = fusion_dim
        for hidden in cfg.fusion_hidden:
            layers.extend([
                nn.Linear(in_dim, hidden),
                nn.BatchNorm1d(hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.dropout),
            ])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, NUM_GESTURES))

        self.classifier = nn.Sequential(*layers)
        self._config = cfg

    def _parse_modalities(self, modalities: str) -> tuple[bool, bool, bool]:
        """Parse a modalities string into (use_imu, use_rfid, use_uwb)."""
        m = modalities.lower().strip()
        if m == "all":
            return True, True, True
        parts = set(m.replace(",", "+").split("+"))
        return ("imu" in parts, "rfid" in parts, "uwb" in parts)

    def forward(
        self,
        imu: torch.Tensor,
        rfid: torch.Tensor,
        uwb: torch.Tensor,
        *,
        modalities: str = "all",
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            imu:  (B, T_imu, 6)
            rfid: (B, T_rfid, 2)
            uwb:  (B, T_uwb, 1)
            modalities: "all", "imu", "rfid", "uwb", "imu+rfid", etc.

        Returns:
            (B, NUM_GESTURES) logits
        """
        use_imu, use_rfid, use_uwb = self._parse_modalities(modalities)
        B = imu.size(0)
        dev = imu.device

        if use_imu:
            imu_feat = self.imu_encoder(imu)
        else:
            imu_feat = torch.zeros(B, self._imu_dim, device=dev)

        if use_rfid:
            rfid_feat = self.rfid_encoder(rfid)
        else:
            rfid_feat = torch.zeros(B, self._rfid_dim, device=dev)

        if use_uwb:
            uwb_feat = self.uwb_encoder(uwb)
        else:
            uwb_feat = torch.zeros(B, self._uwb_dim, device=dev)

        fused = torch.cat([imu_feat, rfid_feat, uwb_feat], dim=1)
        return self.classifier(fused)

    def predict(
        self,
        imu: torch.Tensor,
        rfid: torch.Tensor,
        uwb: torch.Tensor,
        *,
        modalities: str = "all",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (class indices, probabilities)."""
        logits = self.forward(imu, rfid, uwb, modalities=modalities)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds, probs

    def encode(
        self,
        imu: torch.Tensor,
        rfid: torch.Tensor,
        uwb: torch.Tensor,
        *,
        modalities: str = "all",
    ) -> torch.Tensor:
        """Extract the fused feature vector before classification."""
        use_imu, use_rfid, use_uwb = self._parse_modalities(modalities)
        B = imu.size(0)
        dev = imu.device

        imu_feat = self.imu_encoder(imu) if use_imu else torch.zeros(B, self._imu_dim, device=dev)
        rfid_feat = self.rfid_encoder(rfid) if use_rfid else torch.zeros(B, self._rfid_dim, device=dev)
        uwb_feat = self.uwb_encoder(uwb) if use_uwb else torch.zeros(B, self._uwb_dim, device=dev)
        return torch.cat([imu_feat, rfid_feat, uwb_feat], dim=1)

    @property
    def config(self) -> ModelConfig:
        return self._config
