"""Gesture classifier — PCA + k-NN, following MATLAB gesture recognition workflow.

Pipeline:
  1. Load labeled feature vectors from recorded gestures
  2. Apply PCA for dimensionality reduction
  3. Train k-NN classifier on reduced features
  4. Save model (PCA transform + k-NN) for deployment
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("null_gesture.pipeline.classifier")


class GestureClassifier:
    """k-NN classifier with PCA dimensionality reduction.

    Usage:
        clf = GestureClassifier()
        clf.fit(features, labels)       # features: (N, F), labels: (N,) str
        label, conf = clf.predict(query) # query: (F,)
        clf.save("model.npz")
        clf.load("model.npz")
    """

    def __init__(self, n_components: int = 15, k: int = 3):
        """
        Args:
            n_components: Number of PCA components to retain.
            k: Number of nearest neighbors for k-NN.
        """
        self.n_components = n_components
        self.k = k

        # Fitted state
        self._labels: np.ndarray | None = None          # (N,) str
        self._label_set: list[str] = []
        self._features_pca: np.ndarray | None = None     # (N, n_components)
        self._pca_mean: np.ndarray | None = None         # (F,)
        self._pca_components: np.ndarray | None = None   # (n_components, F)

    # ── Fit ───────────────────────────────────────────────────────

    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """Train the classifier.

        Args:
            features: (N, F) feature matrix.
            labels: (N,) array of string gesture names.
        """
        N, F = features.shape

        # Store labels
        self._labels = np.array(labels)
        self._label_set = sorted(set(labels))

        # Standardise
        self._pca_mean = features.mean(axis=0)
        self._pca_std = features.std(axis=0) + 1e-8
        X = (features - self._pca_mean) / self._pca_std

        # PCA via SVD
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        n_comp = min(self.n_components, F, N)
        self._pca_components = Vt[:n_comp]  # (n_comp, F)
        self._features_pca = X @ self._pca_components.T  # (N, n_comp)

        logger.info(
            "Classifier fitted: %d samples, %d features → %d PCA components, k=%d",
            N, F, n_comp, self.k,
        )

    # ── Predict ───────────────────────────────────────────────────

    def predict(self, query: np.ndarray) -> tuple[str, float]:
        """Classify a feature vector.

        Args:
            query: (F,) feature vector.

        Returns:
            (gesture_label, confidence_0_to_1).
        """
        if self._features_pca is None or self._pca_components is None or self._labels is None:
            return "unknown", 0.0

        # Standardise + project
        x = (query - self._pca_mean) / self._pca_std
        x_pca = x @ self._pca_components.T  # (n_comp,)

        # Distances to all training points
        diff = self._features_pca - x_pca  # (N, n_comp)
        dists = np.sqrt(np.sum(diff ** 2, axis=1))  # (N,)

        # k nearest neighbors
        k = min(self.k, len(dists))
        top_k = np.argpartition(dists, k)[:k]
        top_labels = self._labels[top_k]
        top_dists = dists[top_k]

        # Majority vote (distance-weighted)
        votes: dict[str, float] = {}
        for lbl, d in zip(top_labels, top_dists):
            weight = 1.0 / (d + 1e-6)
            votes[lbl] = votes.get(lbl, 0.0) + weight

        best_label = max(votes, key=votes.get)  # type: ignore[arg-type]
        total_weight = sum(votes.values())
        conf = votes[best_label] / total_weight if total_weight > 0 else 0.0

        return str(best_label), float(conf)

    # ── Persistence ───────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save fitted model to .npz file."""
        if self._features_pca is None:
            raise RuntimeError("Classifier not fitted — call fit() first")
        assert self._labels is not None
        assert self._pca_mean is not None
        assert self._pca_std is not None
        assert self._pca_components is not None
        np.savez(
            path,
            labels=self._labels,
            label_set=np.array(self._label_set),
            pca_mean=self._pca_mean,
            pca_std=self._pca_std,
            pca_components=self._pca_components,
            features_pca=self._features_pca,
            k=self.k,
            n_components=self.n_components,
        )
        logger.info("Classifier saved to %s", path)

    def load(self, path: str | Path) -> None:
        """Load fitted model from .npz file."""
        data = np.load(path, allow_pickle=True)
        self._labels = data["labels"]
        self._label_set = list(data["label_set"])
        self._pca_mean = data["pca_mean"]
        self._pca_std = data["pca_std"]
        self._pca_components = data["pca_components"]
        self._features_pca = data["features_pca"]
        self.k = int(data["k"])
        self.n_components = int(data["n_components"])
        assert self._labels is not None
        assert self._pca_components is not None
        logger.info(
            "Classifier loaded: %d samples, %d features → %d PCA components",
            len(self._labels), self._pca_components.shape[1], self.n_components,
        )

    @property
    def fitted(self) -> bool:
        return self._features_pca is not None

    @property
    def gestures(self) -> list[str]:
        return list(self._label_set)
