"""Module 2 detector classes.

Public API
----------
GradientBoostingDetector  — sklearn GBC, XGBoost-equivalent surrogate (Track A)
RandomForestDetector      — sklearn RF (Track A)
DecisionTreeDetector      — sklearn DT (Track A)
DAEDetector               — Keras denoising autoencoder (Track B novelty)
XGBoostDetector           — DEPRECATED alias for GradientBoostingDetector
"""

from .DAE import DAEDetector
from .DecisionTree import DecisionTreeDetector
from .GradientBoosting import GradientBoostingDetector
from .RandomForest import RandomForestDetector
from .XGBoost import XGBoostDetector  # deprecation alias — kept for compat

__all__ = [
    "GradientBoostingDetector",
    "RandomForestDetector",
    "DecisionTreeDetector",
    "DAEDetector",
    "XGBoostDetector",
]
