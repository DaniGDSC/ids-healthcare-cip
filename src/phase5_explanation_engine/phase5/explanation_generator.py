"""ExplanationGenerator — generate human-readable explanations from templates."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from .config import ExplanationTemplates

logger = logging.getLogger(__name__)


class ExplanationGenerator:
    """Generate human-readable explanations from config-driven templates.

    Args:
        templates: ExplanationTemplates instance from config.
    """

    def __init__(self, templates: ExplanationTemplates) -> None:
        self._templates = templates

    def generate(
        self,
        risk_level: str,
        sample_idx: int,
        top3: List[Dict[str, Any]],
    ) -> str:
        """Generate explanation string from template.

        Args:
            risk_level: One of LOW/MEDIUM/HIGH/CRITICAL.
            sample_idx: Sample index for identification.
            top3: Top 3 contributing features with SHAP values.

        Returns:
            Human-readable explanation string.
        """
        template = getattr(self._templates, risk_level, self._templates.LOW)

        if risk_level == "CRITICAL" and len(top3) >= 3:
            return template.format(
                idx=sample_idx,
                time=sample_idx,
                f1=top3[0]["feature"],
                v1=top3[0]["shap_value"],
                p1=top3[0]["contribution_pct"],
                f2=top3[1]["feature"],
                v2=top3[1]["shap_value"],
                p2=top3[1]["contribution_pct"],
                f3=top3[2]["feature"],
                v3=top3[2]["shap_value"],
                p3=top3[2]["contribution_pct"],
            )
        elif risk_level == "HIGH" and len(top3) >= 1:
            return template.format(
                idx=sample_idx,
                f1=top3[0]["feature"],
                p1=top3[0]["contribution_pct"],
            )
        elif risk_level == "MEDIUM" and len(top3) >= 1:
            return template.format(
                idx=sample_idx,
                f1=top3[0]["feature"],
            )
        else:
            return template.format(idx=sample_idx)

    def get_config(self) -> Dict[str, Any]:
        """Return generator configuration."""
        return {
            "template_levels": list(self._templates.model_dump().keys()),
        }
