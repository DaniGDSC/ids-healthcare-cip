"""Unit tests for BaseVisualizer (abstract base class)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from src.phase5_explanation_engine.phase5.base import BaseVisualizer


class TestBaseVisualizer:
    def test_is_abstract(self) -> None:
        with pytest.raises(TypeError):
            BaseVisualizer()  # type: ignore[abstract]

    def test_plot_is_abstract(self) -> None:
        assert hasattr(BaseVisualizer, "plot")

    def test_get_config_is_abstract(self) -> None:
        assert hasattr(BaseVisualizer, "get_config")

    def test_concrete_subclass(self) -> None:
        class DummyVisualizer(BaseVisualizer):
            def plot(self, data: Dict[str, Any], output_path: Path) -> None:
                pass

            def get_config(self) -> Dict[str, Any]:
                return {"type": "dummy"}

        viz = DummyVisualizer()
        assert isinstance(viz, BaseVisualizer)
        assert viz.get_config() == {"type": "dummy"}
