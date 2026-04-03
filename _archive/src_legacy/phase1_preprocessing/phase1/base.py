"""Base transformer interface for Phase 1 preprocessing.

All DataFrame-level transformers implement ``BaseTransformer``, satisfying
the Liskov Substitution Principle — any transformer can be swapped in
the pipeline without changing the orchestrator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd


class BaseTransformer(ABC):
    """Abstract base for all preprocessing transformers.

    Subclasses must implement ``transform()``.  The default ``fit()``
    is a no-op (stateless transformers).  Subclasses that carry state
    (e.g. ``RobustScalerTransformer``) override ``fit()`` to learn
    parameters from the training partition only.
    """

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the transformation to a DataFrame.

        Args:
            df: Input DataFrame.

        Returns:
            Transformed DataFrame.
        """

    def fit(self, df: pd.DataFrame) -> BaseTransformer:
        """Learn parameters from training data (no-op by default).

        Args:
            df: Training DataFrame.

        Returns:
            self
        """
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step.

        Args:
            df: Training DataFrame.

        Returns:
            Transformed DataFrame.
        """
        return self.fit(df).transform(df)

    def get_report(self) -> Dict[str, Any]:
        """Return step metadata for the pipeline report.

        Returns:
            Dict with step-specific statistics.
        """
        return {}
