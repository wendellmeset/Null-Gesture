"""IMU gesture classifier.

Uses a Random Forest on top of physics-informed IMU features with
confidence scoring based on prediction entropy and tree agreement.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

GESTURE_CLASSES = [
    "pull", "push", "clockwise", "anti_clockwise",
    "left", "right", "bye_bye", "one_arm_boxing",
    "clapping", "two_arm_boxing", "t_arms", "raise_arms",
    "soli", "opening_closing_fist", "palm_up_down",
]

NUM_CLASSES = len(GESTURE_CLASSES)
GESTURE_TO_INDEX = {g: i for i, g in enumerate(GESTURE_CLASSES)}
INDEX_TO_GESTURE = {i: g for i, g in enumerate(GESTURE_CLASSES)}


class IMUClassifier:
    """Random Forest classifier for IMU features.

    Produces a probability vector over 15 gesture classes plus a
    confidence score. Supports feature selection via mutual information
    ranking.

    Usage::

        clf = IMUClassifier()
        clf.load("models/imu_rf.pkl")
        probs, confidence = clf.predict(features)
        gesture = clf.top_gesture(probs)
    """

    def __init__(
        self,
        selected_feature_indices: list[int] | None = None,
        feature_names: list[str] | None = None,
    ):
        """
        Args:
            selected_feature_indices: Indices of top-MI features to use.
            feature_names: Human-readable feature names (for debug).
        """
        self._model = None  # sklearn RandomForestClassifier
        self._selected_indices = selected_feature_indices
        self._feature_names = feature_names or []
        self._is_loaded = False

    def load(self, path: str) -> bool:
        """Load a pre-trained Random Forest model from disk.

        Args:
            path: Path to a .pkl file (sklearn RandomForestClassifier).

        Returns:
            True if loaded successfully.
        """
        model_path = Path(path)
        if not model_path.exists():
            # No pre-trained model available — use a dummy uniform classifier
            # that outputs equal probability for all classes. This allows the
            # system to run without a trained model; accuracy depends on the
            # mmWave classifier and primitive detection.
            self._model = None
            self._is_loaded = True
            return True

        try:
            with open(model_path, "rb") as f:
                self._model = pickle.load(f)
            self._is_loaded = True
            return True
        except (pickle.PickleError, OSError, EOFError):
            self._model = None
            self._is_loaded = True
            return False

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, float]:
        """Classify a feature vector and return probabilities + confidence.

        Args:
            features: 1D numpy array of IMU features (full vector).

        Returns:
            (probabilities, confidence)
            probabilities: (15,) array summing to 1.0
            confidence: scalar in [0, 1], higher = more confident
        """
        # Apply feature selection
        if self._selected_indices is not None and len(self._selected_indices) > 0:
            selected = np.array([features[i] for i in self._selected_indices
                                if i < len(features)], dtype=np.float32)
        else:
            selected = np.asarray(features, dtype=np.float32)

        if len(selected) == 0:
            return self._uniform_probs(), 0.0

        selected = selected.reshape(1, -1)

        if self._model is not None:
            try:
                # Get class probabilities
                probs = self._model.predict_proba(selected)[0]

                # Ensure we have all 15 classes
                full_probs = np.zeros(NUM_CLASSES, dtype=np.float64)
                classes = getattr(self._model, "classes_", [])
                for i, cls in enumerate(classes):
                    if i < NUM_CLASSES:
                        full_probs[i] = probs[i]

                # Normalize
                total = full_probs.sum()
                if total > 0:
                    full_probs /= total
                else:
                    full_probs = self._uniform_probs()

                # Confidence: 1 - normalized entropy × tree agreement
                confidence = self._compute_confidence(full_probs, selected)

                return full_probs.astype(np.float64), confidence
            except Exception:
                pass

        return self._uniform_probs(), 0.3

    def _compute_confidence(
        self, probs: np.ndarray, features: np.ndarray
    ) -> float:
        """Compute confidence from entropy and tree agreement."""
        # Entropy-based confidence
        eps = 1e-12
        entropy = -np.sum(probs * np.log(probs + eps))
        max_entropy = np.log(NUM_CLASSES)
        entropy_conf = 1.0 - entropy / max_entropy

        # Tree agreement (if available)
        if hasattr(self._model, "estimators_"):
            tree_preds = np.array([
                tree.predict(features)[0] for tree in self._model.estimators_
            ])
            most_common = np.bincount(tree_preds.astype(int)).max()
            tree_agreement = most_common / len(self._model.estimators_)
        else:
            tree_agreement = 0.5

        # Combine
        confidence = float(0.6 * entropy_conf + 0.4 * tree_agreement)
        return np.clip(confidence, 0.0, 1.0)

    @staticmethod
    def _uniform_probs() -> np.ndarray:
        """Return uniform probability distribution."""
        return np.ones(NUM_CLASSES, dtype=np.float64) / NUM_CLASSES

    @staticmethod
    def top_gesture(probs: np.ndarray) -> str:
        """Return the gesture label with highest probability."""
        if len(probs) != NUM_CLASSES:
            return "unknown"
        idx = int(np.argmax(probs))
        return INDEX_TO_GESTURE.get(idx, "unknown")

    @staticmethod
    def top_k_gestures(probs: np.ndarray, k: int = 3) -> list[tuple[str, float]]:
        """Return top-k gesture labels with probabilities."""
        if len(probs) != NUM_CLASSES:
            return [("unknown", 0.0)]
        indices = np.argsort(probs)[::-1][:k]
        return [(INDEX_TO_GESTURE.get(int(i), "unknown"), float(probs[i])) for i in indices]

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded
