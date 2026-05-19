"""Canonical RQ2 aggregator — pulls every Track 1-5 sub-file into one JSON.

Rename context (Phase 1 of RQ2_Doc.md, executed 2026-05-19 as a soft
rename): the original detection-metrics file was archived at
``results/_pre_m6_drift_fix_20260508_090749/compute_detection_metrics.py``
on 2026-05-08; its old output ``results/rq2_metrics.json`` (detection
content) was moved to ``results/detection_metrics.json``. No live
callers grep'd to the original symbol, so this file takes over the
``compute_rq2_metrics`` name cleanly for the MVE aggregation use case.

Inputs (any subset present; missing → ``_status: pending``):
  Track 1 — faithfulness
    results/rq2_shap_stability.json
    results/rq2_mve_shap_alignment.json
  Track 2 — MITRE grounding
    results/rq2_mitre_audit.json
    results/rq2_mitre_grounding.json
  Track 3 — compliance (Layer-2/3 word budgets + evidence manifest)
    results/rq2_word_budget_audit.json
    results/rq2_compliance_audit.json
  Track 4 — user study (Path C — LLM-persona simulation)
    survey/study_data_audit.json
    survey/rq2c_exclusions.json
    analysis/outputs/rq2c_per_role.json
    survey/qualitative_themes.yaml
  Track 5 — failure-mode catalog
    results/rq2_failure_mode_catalog.json

Output:
  results/rq2_metrics.json

Runtime: sub-second. No model inference.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = REPO_ROOT / "results" / "rq2_metrics.json"

# Numeric targets (mirroring RQ2_Doc.md §5.2). Targets where the underlying
# sample is too small to be meaningful are reported with ``pass: None`` so
# the CI gate tolerates them (pending != failed).
TARGETS_NUMERIC = {
    "shap_stability_mean":      0.90,
    "shap_stability_pass_rate": 0.80,
    "alignment_all_three":      0.80,
    "alignment_at_least_two":   0.95,
    "mitre_grounding_rate":     0.90,
}
# Below these n thresholds, mark numeric targets as "insufficient data"
# rather than fail. The current pilot test split is ~16-18 alerts; real
# evaluation runs are expected to be larger.
MIN_N_FAITHFULNESS = 100   # stability / alignment per-class N


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _try_load_json(rel_path: str) -> Optional[dict]:
    p = REPO_ROOT / rel_path
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _make_block(status: str, subfile_paths: list[str], **contents: Any) -> dict:
    out: dict = {
        "_status": status,
        "_merged_at": _now_iso() if status != "pending" else None,
        "_subfile_paths": subfile_paths,
    }
    out.update(contents)
    return out


# ─── Sub-block loaders ─────────────────────────────────────────

def _load_faithfulness() -> dict:
    stability = _try_load_json("results/rq2_shap_stability.json")
    alignment = _try_load_json("results/rq2_mve_shap_alignment.json")
    paths = [
        "results/rq2_shap_stability.json",
        "results/rq2_mve_shap_alignment.json",
    ]
    if stability and alignment:
        status = "complete"
    elif stability or alignment:
        status = "partial"
    else:
        status = "pending"
    return _make_block(status, paths,
                       shap_stability=stability,
                       mve_shap_alignment=alignment)


def _load_mitre_grounding() -> dict:
    audit = _try_load_json("results/rq2_mitre_audit.json")
    grounding = _try_load_json("results/rq2_mitre_grounding.json")
    paths = [
        "results/rq2_mitre_audit.json",
        "results/rq2_mitre_grounding.json",
    ]
    if audit and grounding:
        status = "complete"
    elif audit or grounding:
        status = "partial"
    else:
        status = "pending"
    return _make_block(status, paths,
                       config_audit=audit, layer1_grounding=grounding)


def _load_compliance() -> dict:
    word_budget = _try_load_json("results/rq2_word_budget_audit.json")
    manifest = _try_load_json("results/rq2_compliance_audit.json")
    paths = [
        "results/rq2_word_budget_audit.json",
        "results/rq2_compliance_audit.json",
    ]
    note = (
        "PHI flow control and cross-role consistency are pytest-only gates; "
        "see tests/test_phi_not_in_llm_prompt.py and "
        "tests/test_step13_cross_role_consistency.py."
    )
    if word_budget and manifest:
        status = "complete"
    elif word_budget or manifest:
        status = "partial"
    else:
        status = "pending"
    return _make_block(status, paths,
                       word_budget_audit=word_budget,
                       compliance_manifest_audit=manifest,
                       _note=note)


def _load_user_study() -> dict:
    """Path C — LLM-persona simulation; not human study."""
    audit = _try_load_json("survey/study_data_audit.json")
    exclusions = _try_load_json("survey/rq2c_exclusions.json")
    per_role = _try_load_json("analysis/outputs/rq2c_per_role.json")
    themes_path = REPO_ROOT / "survey" / "qualitative_themes.yaml"
    paths = [
        "survey/study_data_audit.json",
        "survey/rq2c_exclusions.json",
        "analysis/outputs/rq2c_per_role.json",
        "survey/qualitative_themes.yaml",
    ]
    quant_complete = bool(audit and per_role)
    themes_present = themes_path.exists()
    if quant_complete and themes_present:
        status = "complete"
    elif quant_complete or themes_present:
        status = "partial"
    else:
        status = "pending"
    return _make_block(
        status, paths,
        data_source="LLM-persona simulation (gpt-4o-mini); not human study",
        data_audit=audit,
        exclusions=exclusions,
        per_role_analysis=per_role,
        qualitative_themes_path=(
            str(themes_path.relative_to(REPO_ROOT)) if themes_present else None),
    )


def _load_failure_catalog() -> dict:
    catalog = _try_load_json("results/rq2_failure_mode_catalog.json")
    paths = ["results/rq2_failure_mode_catalog.json"]
    if catalog is None:
        return _make_block("pending", paths,
                           summary=None, disclosure=None,
                           catalog_path=None, catalog_md_path=None)
    md_path = REPO_ROOT / "results" / "rq2_failure_mode_catalog.md"
    cat_meta = catalog.get("_meta") or {}
    status = ("partial" if cat_meta.get("sources_missing") else "complete")
    return _make_block(
        status, paths,
        summary=catalog.get("summary"),
        disclosure=catalog.get("_disclosure"),
        catalog_path="results/rq2_failure_mode_catalog.json",
        catalog_md_path=(
            "results/rq2_failure_mode_catalog.md" if md_path.exists() else None),
    )


# ─── Target extraction (schema-adapted to real files) ─────────

def _t(value: Optional[float], target: float, n: Optional[int],
       min_n: int) -> dict:
    """Build a numeric target dict, tolerating small-n by setting pass=None."""
    if value is None:
        return {"value": None, "target": target, "pass": None,
                "_note": "metric_missing_in_source"}
    if n is not None and n < min_n:
        return {"value": value, "target": target, "pass": None, "n": n,
                "_note": f"insufficient_data (n={n} < {min_n})"}
    return {"value": value, "target": target, "pass": value >= target,
            "n": n}


def _b(value: Optional[bool], target: bool = True) -> dict:
    if value is None:
        return {"value": None, "target": target, "pass": None,
                "_note": "metric_missing_in_source"}
    return {"value": bool(value), "target": target, "pass": bool(value) == target}


def _extract_targets(faithfulness: dict, mitre: dict, compliance: dict,
                     failure: dict) -> dict:
    out: dict = {}

    # SHAP stability — rq2_shap_stability.json::results.aggregate
    stab = (faithfulness.get("shap_stability") or {}).get("results", {}) \
        .get("aggregate") or {}
    n_stab = stab.get("n_alerts")
    out["shap_stability_mean"] = _t(
        stab.get("mean_stability"),
        TARGETS_NUMERIC["shap_stability_mean"], n_stab, MIN_N_FAITHFULNESS)
    out["shap_stability_pass_rate"] = _t(
        stab.get("pct_stable"),
        TARGETS_NUMERIC["shap_stability_pass_rate"], n_stab, MIN_N_FAITHFULNESS)

    # MVE-SHAP alignment — rq2_mve_shap_alignment.json::results.aggregate_with_caveats
    align = (faithfulness.get("mve_shap_alignment") or {}).get("results", {})
    align_agg = align.get("aggregate_with_caveats") or {}
    n_align = align_agg.get("n_alerts_total")
    # Real file reports only overall_all_3; compute at_least_two from
    # by_fusion_class as n-weighted mean. If unavailable, leave null.
    overall_all_3 = align_agg.get("overall_all_3")
    by_class = align.get("by_fusion_class") or {}
    if by_class:
        total = sum(c.get("n_alerts", 0) for c in by_class.values())
        if total:
            two_plus = sum(c.get("n_alerts", 0) * c.get("two_plus_present", 0)
                           for c in by_class.values()) / total
        else:
            two_plus = None
    else:
        two_plus = None
    out["alignment_all_three"] = _t(
        overall_all_3, TARGETS_NUMERIC["alignment_all_three"],
        n_align, MIN_N_FAITHFULNESS)
    out["alignment_at_least_two"] = _t(
        two_plus, TARGETS_NUMERIC["alignment_at_least_two"],
        n_align, MIN_N_FAITHFULNESS)

    # MITRE audit — boolean
    mitre_audit_h = (mitre.get("config_audit") or {}).get("headline") or {}
    out["mitre_audit"] = _b(mitre_audit_h.get("audit_pass"))

    # MITRE grounding — numeric, but pass/target already encoded in source
    ground_h = (mitre.get("layer1_grounding") or {}).get("headline") or {}
    if ground_h:
        out["mitre_grounding_rate"] = {
            "value": ground_h.get("grounded_pct"),
            "target": ground_h.get("target",
                                   TARGETS_NUMERIC["mitre_grounding_rate"]),
            "pass": bool(ground_h.get("pass")),
            "n": ground_h.get("n_evaluated"),
        }
    else:
        out["mitre_grounding_rate"] = _t(
            None, TARGETS_NUMERIC["mitre_grounding_rate"], None, 0)

    # Word budget — boolean
    wb_h = (compliance.get("word_budget_audit") or {}).get("headline") or {}
    out["word_budget_audit"] = _b(wb_h.get("audit_pass"))

    # Compliance manifest — boolean (top-level in that file)
    cm = compliance.get("compliance_manifest_audit") or {}
    out["compliance_manifest_evidence"] = _b(
        cm.get("all_required_evidence_present"))

    # Failure catalog framing — string
    disclosure = failure.get("disclosure") or {}
    framing = disclosure.get("framing")
    out["failure_catalog_disclosure"] = {
        "value": framing,
        "target": "observation_not_improvement",
        "pass": (framing == "observation_not_improvement"
                 if framing is not None else None),
    }
    return out


# ─── Headline ──────────────────────────────────────────────────

def _build_headline(blocks: dict[str, dict]) -> dict:
    statuses = {
        "rq2_a_compliance":       blocks["compliance"]["_status"],
        "rq2_b_faithfulness":     blocks["faithfulness"]["_status"],
        "rq2_c_user_study":       blocks["user_study"]["_status"],
        "rq2_e_mitre_grounding":  blocks["mitre_grounding"]["_status"],
        "rq2_d_failure_catalog":  blocks["failure_catalog"]["_status"],
    }
    values = list(statuses.values())
    if all(v == "complete" for v in values):
        overall = "complete"
    elif all(v == "pending" for v in values):
        overall = "pending"
    else:
        missing = [k for k, v in statuses.items() if v != "complete"]
        overall = ("complete" if not missing
                   else f"partial — incomplete: {', '.join(missing)}")
    return {
        "_description": "Highest-level pass/fail per sub-RQ. Read this first.",
        **statuses,
        "_overall_status": overall,
    }


# ─── Main ──────────────────────────────────────────────────────

def main() -> None:
    blocks = {
        "faithfulness":    _load_faithfulness(),
        "mitre_grounding": _load_mitre_grounding(),
        "compliance":      _load_compliance(),
        "user_study":      _load_user_study(),
        "failure_catalog": _load_failure_catalog(),
    }
    tracks_present = [k for k, v in blocks.items()
                      if v["_status"] in ("complete", "partial")]
    tracks_pending = [k for k, v in blocks.items() if v["_status"] == "pending"]

    out = {
        "_meta": {
            "schema_version": "1.0",
            "generated_at": _now_iso(),
            "generated_by": "module6_evaluation/compute_rq2_metrics.py",
            "research_question": (
                "RQ2 — Can MVE provide role-tailored security explanations "
                "enabling non-specialist hospital stakeholders to make "
                "informed threat triage decisions?"
            ),
            "active_subquestions": ["RQ2.a", "RQ2.b", "RQ2.c", "RQ2.e"],
            "rescoped_subquestions": ["RQ2.d (moved to thesis §7.2.3)"],
            "blocks_present": tracks_present,
            "blocks_pending": tracks_pending,
            "rename_note": (
                "Old compute_rq2_metrics.py archived at "
                "results/_pre_m6_drift_fix_20260508_090749/"
                "compute_detection_metrics.py; old detection-metrics JSON "
                "preserved at results/detection_metrics.json."
            ),
        },
        "headline": _build_headline(blocks),
        **blocks,
        "targets": _extract_targets(
            blocks["faithfulness"], blocks["mitre_grounding"],
            blocks["compliance"], blocks["failure_catalog"]),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"Overall: {out['headline']['_overall_status']}")
    for k, v in out["headline"].items():
        if k.startswith("rq2_"):
            print(f"  {k}: {v}")
    targets = out["targets"]
    n_pass = sum(1 for t in targets.values() if t.get("pass") is True)
    n_fail = sum(1 for t in targets.values() if t.get("pass") is False)
    n_skip = sum(1 for t in targets.values() if t.get("pass") is None)
    print(f"\nTargets: {n_pass} pass, {n_fail} fail, {n_skip} insufficient/missing")
    for k, t in targets.items():
        if t.get("pass") is True:
            mark = "PASS"
        elif t.get("pass") is False:
            mark = "FAIL"
        else:
            mark = "SKIP"
        suffix = f"  [{t.get('_note')}]" if t.get("_note") else ""
        print(f"  {mark:4s} {k}: value={t.get('value')} "
              f"target={t.get('target')}{suffix}")


if __name__ == "__main__":
    main()
