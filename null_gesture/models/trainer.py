"""Training pipeline for the multi-modal gesture classifier.

Features:
- Mixed precision (AMP) training
- Early stopping with patience
- Model checkpointing (best validation accuracy)
- Comprehensive metrics logging
- Cosine annealing LR schedule
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from null_gesture.config import (
    GESTURES,
    MODELS_DIR,
    ModelConfig,
    model_config as default_model_config,
)
from null_gesture.data.dataset import GestureDataset
from null_gesture.data.preprocessor import Preprocessor
from null_gesture.models.fusion import GestureFusionModel

logger = logging.getLogger("null_gesture.models.trainer")


class EarlyStopping:
    """Stops training when validation loss stops improving."""

    def __init__(self, patience: int = 25, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.counter = 0

    def __call__(self, val_loss: float, epoch: int) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class Trainer:
    """Handles the full training loop for GestureFusionModel."""

    def __init__(
        self,
        model: GestureFusionModel | None = None,
        config: ModelConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.cfg = config or default_model_config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = (model or GestureFusionModel(self.cfg)).to(self.device)
        self.scaler = torch.amp.GradScaler("cuda") if self.cfg.use_amp and self.device == "cuda" else None

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.cfg.epochs
        )

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }
        self.best_val_acc = 0.0
        self.best_model_path: Path | None = None

    def train(
        self,
        train_dataset: GestureDataset,
        val_dataset: GestureDataset,
        *,
        model_name: str = "gesture_model",
    ) -> dict:
        """Run the full training loop.

        Returns the training history dict.
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.cfg.batch_size * 2,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        early_stop = EarlyStopping(patience=self.cfg.early_stopping_patience)
        total_start = time.time()

        model_dir = MODELS_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Training on %s | %d epochs | batch_size=%d",
                     self.device, self.cfg.epochs, self.cfg.batch_size)
        logger.info("Train samples: %d | Val samples: %d",
                     len(train_dataset), len(val_dataset))

        for epoch in range(self.cfg.epochs):
            epoch_start = time.time()

            # ── Training phase ──────────────────────────────────────────
            train_loss, train_acc = self._run_epoch(train_loader, training=True)

            # ── Validation phase ────────────────────────────────────────
            val_loss, val_acc = self._run_epoch(val_loader, training=False)

            # ── Update LR ───────────────────────────────────────────────
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # ── Logging ─────────────────────────────────────────────────
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            epoch_s = time.time() - epoch_start
            logger.info(
                "Epoch %3d/%d | train loss: %.4f acc: %.2f%% | "
                "val loss: %.4f acc: %.2f%% | lr: %.2e | %.1fs",
                epoch + 1, self.cfg.epochs,
                train_loss, train_acc * 100,
                val_loss, val_acc * 100,
                current_lr, epoch_s,
            )

            # ── Checkpoint best model ───────────────────────────────────
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_path = model_dir / "best_model.pt"
                self._save_checkpoint(self.best_model_path, epoch, val_acc)
                logger.info("  → New best model (%.2f%%)", val_acc * 100)

            # ── Early stopping ──────────────────────────────────────────
            if early_stop(val_loss, epoch):
                logger.info(
                    "Early stopping at epoch %d (best val_loss: %.4f at epoch %d)",
                    epoch + 1, early_stop.best_loss, early_stop.best_epoch + 1,
                )
                break

        total_s = time.time() - total_start
        logger.info("Training complete in %.1f minutes | Best val_acc: %.2f%%",
                     total_s / 60, self.best_val_acc * 100)

        # Save final model and history
        self._save_checkpoint(model_dir / "final_model.pt", epoch, val_acc)
        with open(model_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

        return self.history

    def _run_epoch(
        self, loader: DataLoader, *, training: bool
    ) -> tuple[float, float]:
        """Run one epoch. Returns (avg_loss, accuracy)."""
        if training:
            self.model.train()
        else:
            self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        amp_ctx = (
            torch.amp.autocast("cuda") if self.scaler is not None
            else torch.no_grad() if not training
            else None
        )

        for batch in loader:
            imu = batch["imu"].to(self.device, non_blocking=True)
            rfid = batch["rfid"].to(self.device, non_blocking=True)
            uwb = batch["uwb"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True)

            if training:
                self.optimizer.zero_grad()

            with amp_ctx if amp_ctx is not None else contextlib_null():
                logits = self.model(imu, rfid, uwb)
                loss = self.criterion(logits, labels)

            if training:
                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

            total_loss += loss.item() * imu.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += imu.size(0)

        return total_loss / total, correct / total

    def _save_checkpoint(self, path: Path, epoch: int, val_acc: float) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_acc": val_acc,
                "best_val_acc": self.best_val_acc,
                "config": self.cfg,
            },
            path,
        )

    @classmethod
    def load(
        cls,
        checkpoint_path: Path | str,
        device: str | None = None,
    ) -> tuple[GestureFusionModel, dict]:
        """Load a trained model from checkpoint. Returns (model, checkpoint_dict)."""
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = ckpt.get("config", default_model_config)
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = GestureFusionModel(config).to(dev)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, ckpt


class contextlib_null:
    """Minimal null context for when AMP isn't available."""
    def __enter__(self): return None
    def __exit__(self, *args): return None


def train_from_npz(
    data_path: Path | str,
    *,
    model_name: str = "gesture_model",
    config: ModelConfig | None = None,
) -> tuple[GestureFusionModel, Preprocessor, dict]:
    """End-to-end: load data, preprocess, train, save preprocessor, return model.

    This is the main entry point for training from a .npz dataset.
    """
    cfg = config or default_model_config

    # Load and split data
    train_ds, val_ds = GestureDataset.from_npz(data_path, augment=True)

    # Fit preprocessor on training data
    preprocessor = Preprocessor()
    imu_np = train_ds.imu.numpy()
    rfid_np = train_ds.rfid.numpy()
    uwb_np = train_ds.uwb.numpy()
    preprocessor.fit(imu_np, rfid_np, uwb_np)

    # Transform both splits
    imu_train, rfid_train, uwb_train = preprocessor.transform(
        imu_np, rfid_np, uwb_np,
    )
    imu_val, rfid_val, uwb_val = preprocessor.transform(
        val_ds.imu.numpy(), val_ds.rfid.numpy(), val_ds.uwb.numpy(),
    )

    # Create normalized datasets
    from null_gesture.data.dataset import GestureDataset as GD
    train_ds_norm = GD(imu_train, rfid_train, uwb_train, train_ds.labels.numpy())
    val_ds_norm = GD(imu_val, rfid_val, uwb_val, val_ds.labels.numpy())

    # Train
    trainer = Trainer(config=cfg)
    history = trainer.train(train_ds_norm, val_ds_norm, model_name=model_name)

    # Save preprocessor
    pp_path = MODELS_DIR / model_name / "preprocessor.json"
    preprocessor.save(pp_path)

    return trainer.model, preprocessor, history
