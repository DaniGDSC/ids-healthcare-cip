"""RQ1 sibling metrics for MedSec-25 (RQ1_pipeline.md §6.3 / Stage 5D).

**Scope (constrained):** MedSec-25's frozen test parquet does not carry
``device_class`` annotations (verified during repo audit:
``data/processed/medsec25/test.parquet`` has 71 network-flow columns
plus ``Attack Category`` but no device taxonomy).  The full RQ1
headline schema therefore cannot be reproduced here.  Per
RQ1_pipeline.md §6.3 the fallback is to emit **only** the Track B
per-class AUC block, drawing on the leave-one-class-out evidence
already persisted at ``results/reports/dae_ablation_loo_medsec25.yaml``.

All other RQ1 sections are emitted with a ``_status`` flag explaining
why they're absent on this dataset, so downstream readers can render a
"partial" badge without re-implementing the gap analysis.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MEDSEC_YAML = REPO_ROOT / "results/reports/dae_ablation_loo_medsec25.yaml"
OUT_PATH = REPO_ROOT / "results/rq1_metrics_medsec.json"


def _classify_auc(auc_value: float | None) -> str:
    if auc_value is None:
        return "insufficient_data"
    if auc_value >= 0.90:
        return "good_to_excellent"
    if auc_value >= 0.75:
        return "acceptable"
    if auc_value >= 0.60:
        return "weak"
    return "fails — benign-mimicking"


def _track_b_per_class_from_yaml(doc: dict) -> dict:
    """Extract per-class DAE-raw AUC from the LOO ablation yaml.

    Schema:
      results: [{holdout_class, config_results: [{config, auc_benign_vs_novel}, ...]}]
    """
    out: dict[str, dict] = {}
    for row in doc.get("results", []) or []:
        cls = row.get("holdout_class")
        if cls is None:
            continue
        for cfg in row.get("config_results", []) or []:
            if cfg.get("config") != "DAE-raw":
                continue
            auc_val = cfg.get("auc_benign_vs_novel")
            n_pos = cfg.get("n_test_novel")
            n_neg = cfg.get("n_test_benign")
            out[str(cls)] = {
                "auc": float(auc_val) if auc_val is not None else None,
                "n_positive": int(n_pos) if n_pos is not None else None,
                "n_negative": int(n_neg) if n_neg is not None else None,
                "verdict": _classify_auc(
                    float(auc_val) if auc_val is not None else None
                ),
            }
    return out


def main() -> None:
    if not MEDSEC_YAML.exists():
        out = {
            "_meta": {
                "schema_version": "1.0",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "generated_by": "module6_evaluation/compute_rq1_metrics_medsec.py",
                "dataset": "MedSec-25",
                "split": "loo_ablation_proxy",
                "_status": "pending",
                "reason": f"{MEDSEC_YAML.name} not found",
            },
        }
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"Wrote {OUT_PATH.relative_to(REPO_ROOT)} (pending)")
        return

    doc = yaml.safe_load(MEDSEC_YAML.read_text(encoding="utf-8"))
    track_b_per_class = _track_b_per_class_from_yaml(doc)

    out = {
        "_meta": {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": "module6_evaluation/compute_rq1_metrics_medsec.py",
            "dataset": "MedSec-25",
            "split": "loo_ablation_proxy",
            "source": str(MEDSEC_YAML.relative_to(REPO_ROOT)),
            "scope_note": (
                "MedSec-25 test parquet lacks the device_class taxonomy "
                "required for FNR_critical, surfacing, and correlation "
                "diagnostics.  Per RQ1_pipeline.md §6.3 the sibling JSON "
                "is therefore scoped to Track B per-class AUC, sourced "
                "from Module 2's leave-one-class-out ablation yaml."
            ),
        },
        "track_b_per_class": track_b_per_class,
        "headline": {
            "_status": "scoped_out",
            "reason": (
                "Headline FNR_critical / sensitivity / specificity "
                "require device-class-derived severity, which MedSec-25 "
                "does not carry in this prototype."
            ),
        },
        "track_a_ablation": {
            "_status": "scoped_out",
            "reason": "Track A models not retrained on MedSec-25.",
        },
        "fusion_classes": {
            "_status": "scoped_out",
            "reason": "Cascade fusion classes require full pipeline run.",
        },
        "risk_tier_distribution": {
            "_status": "scoped_out",
            "reason": "Composite R not computed on MedSec-25.",
        },
        "surfacing_summary": {
            "_status": "scoped_out",
            "reason": (
                "Surfacing invariants require device_criticality + "
                "patchable, which MedSec-25 lacks."
            ),
        },
        "correlation_diagnostics": {
            "_status": "scoped_out",
            "reason": (
                "D_crit / D_clinical_tier are not defined for MedSec-25 "
                "rows."
            ),
        },
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH.relative_to(REPO_ROOT)}")
    n = len(track_b_per_class)
    print(f"Track B per-class AUC: {n} classes")
    for cls, info in track_b_per_class.items():
        auc_str = (
            f"{info['auc']:.4f}" if info["auc"] is not None else "n/a"
        )
        print(f"  {cls:<24s}  AUC={auc_str}  ({info['verdict']})")


if __name__ == "__main__":
    main()
