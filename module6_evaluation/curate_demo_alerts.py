"""ARCHITECTURE.md M6 — Demo-pool alert curation.

Reads ``results/reports/demo_scores.npz`` (produced by
``module3_risk_scoring.module3_demo_scores``) and the corresponding
``data/processed/demo_phase1.parquet``, performs stratified sampling
across (risk_tier × attack_category × fusion_class), and writes
``results/reports/evaluation_alerts.json`` (~20 alerts) for the
Streamlit dashboard + Phase-2 user study.

Per the doc's separation-of-concerns invariant: this module sources
**only** from ``demo_phase1.parquet``. Test-split rows can never enter
the dashboard — the 4-way Strategy 1 split is byte-disjoint.

The implementation lives in ``module6_evaluation.module6_evaluation``
(historic location); this module is the canonical entry point per
ARCHITECTURE.md M6 description and re-exports the curation function so
new code targets the doc-named path.
"""
from __future__ import annotations

from module6_evaluation.module6_evaluation import (  # noqa: F401
    curate_evaluation_alerts as curate_demo_alerts,
)

__all__ = ["curate_demo_alerts"]


def main() -> int:
    """CLI: regenerate ``evaluation_alerts.json`` from demo_scores.npz."""
    import json
    import logging
    from pathlib import Path

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    project_root = Path(__file__).resolve().parents[1]
    out_path = project_root / "results" / "reports" / "evaluation_alerts.json"
    alerts = curate_demo_alerts()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(alerts, f, indent=2, default=str)
    print(f"Saved {len(alerts)} alerts → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
