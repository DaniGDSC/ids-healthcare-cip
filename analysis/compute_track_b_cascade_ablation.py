"""Track B cascade-ablation summary (RQ1_pipeline.md §6.2 / Stage 5C).

The cascade-vs-raw comparison was run as part of Module 2's
leave-one-class-out (LOO) ablation; the per-fold AUCs are persisted as
YAML at:

  results/reports/dae_ablation_loo.yaml          (EHMS-2020)
  results/reports/dae_ablation_loo_medsec25.yaml (MedSec-25)

This script aggregates those yamls into a single JSON consumed by the
RQ1 merge step (``analysis/merge_rq1_metrics.py``).  It does NOT retrain
any DAE — per the open-question resolution in RQ1_pipeline.md §10 (#3),
the cascade verdict ("rejected") is already grounded in the LOO
evidence and the MedSec generalisation regression.

Output: ``results/rq1_cascade_ablation.json``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
EHMS_YAML = REPO_ROOT / "results/reports/dae_ablation_loo.yaml"
MEDSEC_YAML = REPO_ROOT / "results/reports/dae_ablation_loo_medsec25.yaml"
OUT_PATH = REPO_ROOT / "results/rq1_cascade_ablation.json"


def _aggregate(yaml_doc: dict) -> dict:
    """Average per-fold AUC for each named config across holdout classes.

    Returns ``{config_name: {"auc": mean_auc, "n_folds": k}}`` plus the
    raw per-fold table for traceability.
    """
    rows = yaml_doc.get("results", []) or []
    by_config: dict[str, list[float]] = {}
    per_fold: list[dict] = []
    for row in rows:
        cls = row.get("holdout_class")
        for cfg in row.get("config_results", []) or []:
            name = cfg.get("config")
            auc_val = cfg.get("auc_benign_vs_novel")
            if name is None or auc_val is None:
                continue
            by_config.setdefault(name, []).append(float(auc_val))
            per_fold.append({
                "holdout_class": cls,
                "config": name,
                "auc": float(auc_val),
                "n_test_novel": cfg.get("n_test_novel"),
            })

    summary = {
        name: {
            "auc_mean": round(mean(vals), 4),
            "auc_min": round(min(vals), 4),
            "auc_max": round(max(vals), 4),
            "n_folds": len(vals),
        }
        for name, vals in by_config.items()
    }
    return {"summary": summary, "per_fold": per_fold}


def _delta(summary: dict, baseline: str, candidate: str) -> float | None:
    if baseline not in summary or candidate not in summary:
        return None
    return round(
        summary[candidate]["auc_mean"] - summary[baseline]["auc_mean"], 4
    )


def _verdict(delta_ehms: float | None, delta_medsec: float | None) -> str:
    if delta_ehms is None and delta_medsec is None:
        return (
            "indeterminate — neither EHMS nor MedSec ablation results were "
            "available."
        )
    parts = []
    if delta_ehms is not None:
        sign = "+" if delta_ehms >= 0 else ""
        parts.append(f"EHMS Δ={sign}{delta_ehms:.4f}")
    if delta_medsec is not None:
        sign = "+" if delta_medsec >= 0 else ""
        parts.append(f"MedSec Δ={sign}{delta_medsec:.4f}")

    # Heuristic verdict logic (matches the recorded design decision in
    # module3_risk_scoring/module3_risk_scores.py: cascade rejected when
    # MedSec generalisation regresses).
    if delta_medsec is not None and delta_medsec <= -0.05:
        return (
            f"Cascade rejected — generalisation regression on MedSec-25 "
            f"({', '.join(parts)}). DAE on raw 25-dim input retained as "
            "production configuration (Phase B)."
        )
    if delta_ehms is not None and delta_ehms <= 0.005 and (
        delta_medsec is None or delta_medsec <= 0.0
    ):
        return (
            f"Cascade not adopted — EHMS gain is marginal "
            f"({', '.join(parts)}) and does not justify the added "
            "cross-dataset risk."
        )
    return f"Cascade evaluation: {', '.join(parts)}."


def main() -> None:
    out = {
        "_meta": {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": "analysis/compute_track_b_cascade_ablation.py",
            "source_yamls": {
                "ehms_2020": str(EHMS_YAML.relative_to(REPO_ROOT))
                if EHMS_YAML.exists() else None,
                "medsec_25": str(MEDSEC_YAML.relative_to(REPO_ROOT))
                if MEDSEC_YAML.exists() else None,
            },
            "note": (
                "Cascade vs raw AUCs aggregated from Module 2's LOO "
                "ablation YAMLs.  No retraining performed."
            ),
        },
        "results": {},
    }

    delta_ehms: float | None = None
    delta_medsec: float | None = None

    if EHMS_YAML.exists():
        ehms = _aggregate(yaml.safe_load(EHMS_YAML.read_text(encoding="utf-8")))
        delta_ehms = _delta(ehms["summary"], "DAE-raw", "DAE-cascade")
        out["results"]["ehms_2020"] = {
            "configs": ehms["summary"],
            "delta_cascade_minus_raw": delta_ehms,
            "per_fold": ehms["per_fold"],
        }
    else:
        out["results"]["ehms_2020"] = {
            "_status": "pending",
            "reason": f"{EHMS_YAML.name} not found",
        }

    if MEDSEC_YAML.exists():
        medsec = _aggregate(yaml.safe_load(MEDSEC_YAML.read_text(encoding="utf-8")))
        delta_medsec = _delta(medsec["summary"], "DAE-raw", "DAE-cascade")
        out["results"]["medsec_25"] = {
            "configs": medsec["summary"],
            "delta_cascade_minus_raw": delta_medsec,
            "per_fold": medsec["per_fold"],
        }
    else:
        out["results"]["medsec_25"] = {
            "_status": "pending",
            "reason": f"{MEDSEC_YAML.name} not found",
        }

    out["verdict"] = _verdict(delta_ehms, delta_medsec)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH.relative_to(REPO_ROOT)}")
    print(out["verdict"])


if __name__ == "__main__":
    main()
