"""Unit tests for RiskLevel enum."""

from __future__ import annotations

from src.phase4_risk_engine.phase4.risk_level import RiskLevel


class TestRiskLevel:
    """Validate RiskLevel enum values and behavior."""

    def test_enum_values(self) -> None:
        assert RiskLevel.NORMAL.value == "NORMAL"
        assert RiskLevel.LOW.value == "LOW"
        assert RiskLevel.MEDIUM.value == "MEDIUM"
        assert RiskLevel.HIGH.value == "HIGH"
        assert RiskLevel.CRITICAL.value == "CRITICAL"

    def test_all_five_levels(self) -> None:
        assert len(RiskLevel) == 5

    def test_string_comparison(self) -> None:
        assert RiskLevel.NORMAL == "NORMAL"
        assert RiskLevel.CRITICAL == "CRITICAL"

    def test_membership(self) -> None:
        assert "NORMAL" in [lvl.value for lvl in RiskLevel]
        assert "CRITICAL" in [lvl.value for lvl in RiskLevel]
