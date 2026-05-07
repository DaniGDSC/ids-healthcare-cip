"""Legacy adapter — thin wrapper over ``src.context_enrichment``.

The canonical context-enrichment + alert-scoring API lives at
``src/context_enrichment.py`` (per ARCHITECTURE.md Step [8]). This file
is kept as a backward-compatibility shim for any caller still importing
``module6_evaluation._src_adapter.scored_from_eval_alert``; it delegates
to ``src.context_enrichment.score_alert_from_dict`` which fails loudly
when the required ``patchable`` field is absent (no silent default).
"""
from __future__ import annotations

from typing import Any

from src.context_enrichment import score_alert_from_dict
from src.data_models import ScoredAlert


def scored_from_eval_alert(alert_data: dict[str, Any]) -> ScoredAlert:
    """Run a pipeline evaluation-alert dict through the prototype scorer.

    Deprecated: prefer ``src.context_enrichment.score_alert_from_dict``
    in new code. This shim is retained because external integrations
    or notebooks may still import it; it now delegates to the canonical
    location and inherits the strict ``patchable`` contract.
    """
    return score_alert_from_dict(alert_data)
