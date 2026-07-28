"""Fusion layer: Bayesian multi-modal fusion and temporal smoothing."""

from src.fusion.bayesian import BayesianFusion
from src.fusion.ensemble import GestureEnsemble

__all__ = ["BayesianFusion", "GestureEnsemble"]
