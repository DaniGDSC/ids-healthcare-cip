"""RQ2.a — MVE word-budget audit (RQ2_Compliance.md Phase 2).

Audits ``results/reports/mve_outputs.jsonl`` (produced by
``module5_responses.module5_mve_batch``) against the per-layer and
total word budgets defined by the ``MVEOutput`` contract.

Binding budgets (per ``src/data_models.py:232`` + ``tests/test_coverage_mve.py``):

  * Layer 1 (WHY)                ≤ 60 words   (baseline_behavior +
                                               deviation_description +
                                               confidence_indicator +
                                               role_authorization_check)
  * Layer 2 (CLINICAL SEVERITY)  ≤ 50 words   (affected_system +
                                               patient_care_impact +
                                               phi_exposure +
                                               severity_label +
                                               severity_rationale)
  * Layer 3 (RECOMMENDED ACTION) ≤ 60 words   (immediate_action +
                                               clinical_constraint +
                                               escalation_path +
                                               timeframe)
  * TOTAL                        ≤ 150 words

Drift note (RQ2_Compliance.md §5.1 vs reality): the spec proposes
40/50/60/30 = 180 total split as 4 buckets including a separate
``layer3_do_not``.  The codebase contract is 60/50/60 = 150 total with
the DO-NOT clause folded into Layer 3 as ``clinical_constraint``.  This
audit honours the codebase contract (which is what ``MVEOutput``
actually enforces) — see ``_meta.config`` in the output JSON for the
binding numbers.

Word-count method: ``len(text.split())`` (whitespace tokens).
Hard fail on any per-layer or total violation.

Output: ``results/rq2_word_budget_audit.json``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
MVE_OUTPUTS = REPO_ROOT / "results/reports/mve_outputs.jsonl"
OUT = REPO_ROOT / "results/rq2_word_budget_audit.json"

# Mirrors src/data_models.py:MVEOutput._L1/_L2/_L3 lists and the
# layer-budget docstrings.  Kept here (not imported) so this audit
# stays single-file readable; the test verifies budgets against the
# MVEOutput contract directly.
LAYER_FIELDS = {
    "layer_1": ["baseline_behavior", "deviation_description",
                "confidence_indicator", "role_authorization_check"],
    "layer_2": ["affected_system", "patient_care_impact", "phi_exposure",
                "severity_label", "severity_rationale"],
    "layer_3": ["immediate_action", "clinical_constraint",
                "escalation_path", "timeframe"],
}
LAYER_BUDGETS = {"layer_1": 60, "layer_2": 50, "layer_3": 60}
TOTAL_BUDGET = 150


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _word_count(text: Any) -> int:
    if not isinstance(text, str):
        return 0
    return len(text.split())


def _layer_word_count(record: dict, layer_key: str) -> int:
    layer = record.get(layer_key) or {}
    if not isinstance(layer, dict):
        return 0
    return sum(_word_count(layer.get(f, "")) for f in LAYER_FIELDS[layer_key])


def _load_mve_outputs() -> list[dict]:
    if not MVE_OUTPUTS.exists():
        raise SystemExit(
            f"{MVE_OUTPUTS} missing — run "
            "`python -m module5_responses.module5_mve_batch` first."
        )
    records: list[dict] = []
    with MVE_OUTPUTS.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main() -> None:
    records = _load_mve_outputs()
    n = len(records)

    per_layer_stats: dict[str, dict] = {
        layer: {"max": 0, "mean": 0.0, "n_over": 0, "budget": budget}
        for layer, budget in LAYER_BUDGETS.items()
    }
    total_stats: dict[str, Any] = {
        "max": 0, "mean": 0.0, "n_over": 0, "budget": TOTAL_BUDGET,
    }

    violations: list[dict] = []
    all_totals: list[int] = []
    by_mode_totals: dict[str, list[int]] = {}

    for rec in records:
        layer_counts = {
            layer: _layer_word_count(rec, layer)
            for layer in LAYER_BUDGETS
        }
        # ``total_word_count`` is embedded by MVEOutput.to_dict() in the
        # JSONL; prefer it (it's the canonical contract value) but fall
        # back to the sum if absent.
        total = (
            int(rec.get("total_word_count"))
            if isinstance(rec.get("total_word_count"), (int, float))
            else sum(layer_counts.values())
        )
        all_totals.append(total)

        mode = str(rec.get("mode_used", "unknown"))
        by_mode_totals.setdefault(mode, []).append(total)

        rec_violations: list[dict] = []
        for layer, count in layer_counts.items():
            stats = per_layer_stats[layer]
            stats["max"] = max(stats["max"], count)
            if count > stats["budget"]:
                stats["n_over"] += 1
                rec_violations.append({
                    "layer": layer,
                    "count": count,
                    "budget": stats["budget"],
                    "over_by": count - stats["budget"],
                })

        total_stats["max"] = max(total_stats["max"], total)
        if total > TOTAL_BUDGET:
            total_stats["n_over"] += 1
            rec_violations.append({
                "layer": "TOTAL",
                "count": total,
                "budget": TOTAL_BUDGET,
                "over_by": total - TOTAL_BUDGET,
            })

        if rec_violations:
            violations.append({
                "row_id": rec.get("row_id"),
                "alert_id": rec.get("alert_id"),
                "mode_used": mode,
                "per_layer_counts": layer_counts,
                "total": total,
                "violations": rec_violations,
            })

    # Means.
    if n > 0:
        for layer in LAYER_BUDGETS:
            per_layer_stats[layer]["mean"] = round(
                sum(_layer_word_count(r, layer) for r in records) / n, 2
            )
        total_stats["mean"] = round(sum(all_totals) / n, 2)

    by_mode_summary = {
        mode: {
            "n": len(totals),
            "total_max": max(totals) if totals else 0,
            "total_mean": round(sum(totals) / len(totals), 2)
            if totals else 0.0,
            "total_over": sum(1 for t in totals if t > TOTAL_BUDGET),
        }
        for mode, totals in by_mode_totals.items()
    }

    out: dict[str, Any] = {
        "_meta": {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": "analysis/audit_word_budgets.py",
            "inputs": {
                "mve_outputs": str(MVE_OUTPUTS.relative_to(REPO_ROOT)),
                "mve_outputs_sha256": _sha256(MVE_OUTPUTS),
                "n_records": n,
            },
            "config": {
                "binding_contract": "src/data_models.py:MVEOutput",
                "word_count_method": "len(text.split())",
                "layer_budgets": LAYER_BUDGETS,
                "total_budget": TOTAL_BUDGET,
                "layer_fields": LAYER_FIELDS,
                "pass_criterion": (
                    "Hard fail on any per-layer or total violation "
                    "(RQ2_Compliance.md §2 locked decision)"
                ),
                "drift_note": (
                    "Spec proposes 40/50/60/30=180; codebase contract "
                    "is 60/50/60=150.  Audit honours the codebase "
                    "contract — see RQ2_Compliance.md plan §D1 for "
                    "the drift resolution."
                ),
            },
        },
        "headline": {
            "n_records": n,
            "n_records_with_violations": len(violations),
            "audit_pass": len(violations) == 0,
            "violation_rate": (len(violations) / n) if n else 0.0,
        },
        "per_layer_stats": per_layer_stats,
        "total_stats": total_stats,
        "by_mode": by_mode_summary,
        "violations": violations[:50],
        "violations_total_count": len(violations),
        "violations_truncated_at": 50,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(REPO_ROOT)}")
    print(
        f"Audit: {'PASS' if out['headline']['audit_pass'] else 'FAIL'} "
        f"({len(violations)}/{n} records over budget)"
    )
    print(
        f"  Layer 1 max/mean: {per_layer_stats['layer_1']['max']}/"
        f"{per_layer_stats['layer_1']['mean']} (budget {LAYER_BUDGETS['layer_1']})"
    )
    print(
        f"  Layer 2 max/mean: {per_layer_stats['layer_2']['max']}/"
        f"{per_layer_stats['layer_2']['mean']} (budget {LAYER_BUDGETS['layer_2']})"
    )
    print(
        f"  Layer 3 max/mean: {per_layer_stats['layer_3']['max']}/"
        f"{per_layer_stats['layer_3']['mean']} (budget {LAYER_BUDGETS['layer_3']})"
    )
    print(
        f"  TOTAL   max/mean: {total_stats['max']}/{total_stats['mean']} "
        f"(budget {TOTAL_BUDGET})"
    )


if __name__ == "__main__":
    main()
