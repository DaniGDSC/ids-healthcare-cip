"""Merge supporting analyses into rq1_metrics.json
(RQ1_pipeline.md §6.4 / Stage 5E).

Inputs (any subset may be missing — script is idempotent):

  results/rq1_weight_sensitivity.json   — new 5B output (if produced)
  results/rq1_sensitivity_analysis.json — legacy 5B evidence
                                          (analysis/compute_rq1.py)
  results/rq1_cascade_ablation.json     — 5C output

Updates ``results/rq1_metrics.json`` in place, replacing the
``weight_sensitivity`` and ``track_b_ablation.cascade`` placeholder
blocks emitted by ``compute_rq1_metrics.py``.

When both ``rq1_weight_sensitivity.json`` (new) and
``rq1_sensitivity_analysis.json`` (legacy) exist, the new one wins; the
legacy block is preserved under
``weight_sensitivity._legacy_evidence`` for traceability.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RQ1 = REPO_ROOT / "results/rq1_metrics.json"
WS_NEW = REPO_ROOT / "results/rq1_weight_sensitivity.json"
WS_LEGACY = REPO_ROOT / "results/rq1_sensitivity_analysis.json"
CA = REPO_ROOT / "results/rq1_cascade_ablation.json"


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    if not RQ1.exists():
        raise SystemExit(
            f"{RQ1} not found — run "
            "`python -m module6_evaluation.compute_rq1_metrics` first."
        )

    metrics = json.loads(RQ1.read_text(encoding="utf-8"))
    now = datetime.now(timezone.utc).isoformat()
    merged_any = False

    # ── Weight sensitivity (5B) ────────────────────────────────────────
    ws_new = _load(WS_NEW)
    ws_legacy = _load(WS_LEGACY)
    if ws_new is not None:
        metrics["weight_sensitivity"] = ws_new
        metrics["weight_sensitivity"]["_merged_at"] = now
        metrics["weight_sensitivity"]["_source"] = WS_NEW.name
        if ws_legacy is not None:
            metrics["weight_sensitivity"]["_legacy_evidence"] = ws_legacy
        print(f"Merged weight_sensitivity from {WS_NEW.name}")
        merged_any = True
    elif ws_legacy is not None:
        metrics["weight_sensitivity"] = {
            "_status": (
                "v1 evidence from legacy analysis/compute_rq1.py — "
                "perturbation protocol pending finalisation per "
                "RQ1_pipeline.md §6.1"
            ),
            "_source": WS_LEGACY.name,
            "_merged_at": now,
            **ws_legacy,
        }
        print(f"Merged weight_sensitivity (v1 legacy) from {WS_LEGACY.name}")
        merged_any = True

    # ── Cascade ablation (5C) ──────────────────────────────────────────
    ca = _load(CA)
    if ca is not None:
        metrics.setdefault("track_b_ablation", {})["cascade"] = {
            **ca,
            "_merged_at": now,
            "_source": CA.name,
        }
        print(f"Merged cascade ablation from {CA.name}")
        merged_any = True

    RQ1.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    if merged_any:
        print(f"Updated {RQ1.relative_to(REPO_ROOT)}")
    else:
        print(
            "Nothing to merge — both weight-sensitivity and cascade "
            "inputs are missing.  rq1_metrics.json left untouched."
        )


if __name__ == "__main__":
    main()
