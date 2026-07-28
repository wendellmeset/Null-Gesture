"""mmWave gesture classifier.

Uses a hybrid architecture: a lightweight 2D CNN processes Range-Doppler
heatmaps while a Random Forest processes engineered features. The outputs
are concatenated for final classification.

If PyTorch is unavailable or no trained model exists, falls back to a
pure Random Forest on the engineered feature vector.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from src.models.imu_model import (
    GESTURE_CLASSES,
    NUM_CLASSES,
    INDEX_TO_GESTURE,
    GESTURE_TO_INDEX,
)


class MMWaveClassifier:
    """Hybrid CNN + RF classifier for mmWave radar data.

    CNN head: processes Range-Doppler heatmap sequences (N_frames × 32 × 32)
    RF head: processes engineered feature vectors (~150 features)
    Concatenated: CNN embedding + RF features → final Random Forest

    If models aren't available, falls back to feature-based prediction
    with uniform priors.

    Usage::

        clf = MMWaveClassifier()
        clf.load("models/mmwave_hybrid.pkl")
        probs, confidence = clf.predict(features, rd_heatmaps)
    """

    def __init__(
        self,
        selected_feature_indices: list[int] | None = None,
        feature_names: list[str] | None = None,
        use_cnn: bool = True,
    ):
        """
        Args:
            selected_feature_indices: Top-MI feature indices for RF head.
            feature_names: Human-readable feature names.
            use_cnn: Whether to use CNN head (requires PyTorch).
        """
        self._model = None  # sklearn RandomForestClassifier (full model or RF head)
        self._cnn = None     # PyTorch CNN model (optional)
        self._selected_indices = selected_feature_indices
        self._feature_names = feature_names or []
        self._use_cnn = use_cnn
        self._is_loaded = False
        self._cnn_embedding_dim = 16

    def load(self, path: str) -> bool:
        """Load pre-trained model(s) from disk.

        Args:
            path: Base path for model files.
                  Looks for {path} (full RF) and {path}_cnn.pt (CNN weights).

        Returns:
            True if at least the RF model loaded.
        """
        model_path = Path(path)

        # Try to load the full hybrid model
        if model_path.exists():
            try:
                with open(model_path, "rb") as f:
                    self._model = pickle.load(f)
                self._is_loaded = True
            except (pickle.PickleError, OSError, EOFError):
                self._model = None

        # Try to load CNN weights
        cnn_path = model_path.with_suffix(".pt")
        if self._use_cnn and cnn_path.exists():
            try:
                self._cnn = self._build_cnn()
                import torch
                self._cnn.load_state_dict(torch.load(cnn_path, map_location="cpu"))
                self._cnn.eval()
            except Exception:
                self._cnn = None

        self._is_loaded = True
        return self._model is not None

    def predict(
        self,
        features: np.ndarray,
        rd_heatmaps: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """Classify a gesture window.

        Args:
            features: 1D numpy array of mmWave engineered features.
            rd_heatmaps: (N_frames, 32, 32) array of range-Doppler heatmaps,
                         or None if unavailable.

        Returns:
            (probabilities, confidence)
        """
        # Apply feature selection
        if self._selected_indices is not None and len(self._selected_indices) > 0:
            selected = np.array([
                features[i] for i in self._selected_indices if i < len(features)
            ], dtype=np.float32)
        else:
            selected = np.asarray(features, dtype=np.float32)

        if len(selected) == 0:
            return self._uniform_probs(), 0.0

        # CNN embedding (if available)
        cnn_embedding = None
        if self._cnn is not None and rd_heatmaps is not None and rd_heatmaps.size > 0:
            cnn_embedding = self._extract_cnn_embedding(rd_heatmaps)

        # Combine features
        if cnn_embedding is not None:
            combined = np.concatenate([selected, cnn_embedding])
        else:
            combined = selected

        combined = combined.reshape(1, -1)

        if self._model is not None:
            try:
                probs = self._model.predict_proba(combined)[0]

                full_probs = np.zeros(NUM_CLASSES, dtype=np.float64)
                classes = getattr(self._model, "classes_", [])
                for i, cls in enumerate(classes):
                    if i < NUM_CLASSES:
                        full_probs[i] = probs[i]

                total = full_probs.sum()
                if total > 0:
                    full_probs /= total
                else:
                    full_probs = self._uniform_probs()

                confidence = self._compute_confidence(full_probs, combined)
                return full_probs.astype(np.float64), confidence
            except Exception:
                pass

        return self._uniform_probs(), 0.3

    def _extract_cnn_embedding(self, rd_heatmaps: np.ndarray) -> np.ndarray | None:
        """Extract CNN embedding from a sequence of R-D heatmaps."""
        if self._cnn is None:
            return None

        try:
            import torch

            # Stack heatmaps as channels: (N_frames, H, W) → (1, N_frames, H, W)
            if rd_heatmaps.ndim == 3:
                # Already (N, H, W)
                tensor = torch.from_numpy(rd_heatmaps).float().unsqueeze(0)  # (1, N, H, W)
            elif rd_heatmaps.ndim == 2:
                tensor = torch.from_numpy(rd_heatmaps).float().unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            else:
                return None

            with torch.no_grad():
                embedding = self._cnn(tensor)
                return embedding.numpy().flatten()
        except Exception:
            return None

    def _build_cnn(self):
        """Build lightweight CNN for R-D heatmap processing."""
        try:
            import torch
            import torch.nn as nn

            class RDCNN(nn.Module):
                def __init__(self, embedding_dim=16):
                    super().__init__()
                    self.conv = nn.Sequential(
                        nn.Conv2d(20, 8, kernel_size=3, padding=1),  # 20 time frames
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(8, 16, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        nn.Conv2d(16, 32, kernel_size=3, padding=1),
                        nn.ReLU(),
                        nn.AdaptiveAvgPool2d(1),
                    )
                    self.fc = nn.Linear(32, embedding_dim)

                def forward(self, x):
                    # x: (batch, frames, H, W)
                    # Transpose to (batch, H, W, frames) then to (batch, frames, H, W) for Conv2d
                    b, f, h, w = x.shape
                    x = x.view(b, f, h, w)  # Keep as (batch, channels=frames, H, W)
                    x = self.conv(x)
                    x = x.view(b, -1)
                    x = self.fc(x)
                    return x

            return RDCNN(embedding_dim=self._cnn_embedding_dim)
        except ImportError:
            return None

    def _compute_confidence(self, probs: np.ndarray, features: np.ndarray) -> float:
        """Compute confidence from entropy and model agreement."""
        eps = 1e-12
        entropy = -np.sum(probs * np.log(probs + eps))
        max_entropy = np.log(NUM_CLASSES)
        entropy_conf = 1.0 - entropy / max_entropy

        if hasattr(self._model, "estimators_"):
            tree_preds = np.array([
                tree.predict(features)[0] for tree in self._model.estimators_
            ])
            most_common = np.bincount(tree_preds.astype(int)).max()
            tree_agreement = most_common / len(self._model.estimators_)
        else:
            tree_agreement = 0.5

        confidence = float(0.6 * entropy_conf + 0.4 * tree_agreement)
        return np.clip(confidence, 0.0, 1.0)

    @staticmethod
    def _uniform_probs() -> np.ndarray:
        return np.ones(NUM_CLASSES, dtype=np.float64) / NUM_CLASSES

    @staticmethod
    def top_gesture(probs: np.ndarray) -> str:
        if len(probs) != NUM_CLASSES:
            return "unknown"
        idx = int(np.argmax(probs))
        return INDEX_TO_GESTURE.get(idx, "unknown")

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
