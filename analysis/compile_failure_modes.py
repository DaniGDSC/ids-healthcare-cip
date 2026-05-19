"""Aggregate RQ2 failure-shaped observations from Tracks 1-4 into a catalog.

Inputs (any subset present; missing inputs degrade gracefully):
  - results/rq2_mve_shap_alignment.json    (Track 1 — alignment)
  - results/rq2_shap_stability.json        (Track 1 — stability)
  - results/rq2_mitre_grounding.json       (Track 2 — MITRE grounding)
  - results/rq2_word_budget_audit.json     (Track 3 — word budgets)
  - survey/qualitative_themes.yaml         (Track 4 — qualitative)
  + configs/rq2_failure_categories.yaml    (manifest, required)

Output:
  results/rq2_failure_mode_catalog.json

Framing: RQ2.d was rescoped to future work; the catalog records observations
only. Each entry carries `_status: observed_not_fixed` and the top-level
`_disclosure` block asserts `iteration_performed: false`.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "configs" / "rq2_failure_categories.yaml"
OUT_PATH = REPO_ROOT / "results" / "rq2_failure_mode_catalog.json"

EVIDENCE_TRUNCATION = 10
LOW_ALIGNMENT_THRESHOLD = 0.5      # all_3_present < 0.5 → TECHNICAL_LAYER_1
LOW_STABILITY_THRESHOLD = 0.90     # per spec — < 0.90 → OTHER known-limitation


def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise SystemExit(
            f"Manifest missing: {MANIFEST_PATH.relative_to(REPO_ROOT)}. "
            "Create per RQ2_failure.md Phase 1."
        )
    return yaml.safe_load(MANIFEST_PATH.read_text())


def _try_load_json(rel_path: str):
    p = REPO_ROOT / rel_path
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _try_load_yaml(rel_path: str):
    p = REPO_ROOT / rel_path
    if not p.exists():
        return None
    try:
        return yaml.safe_load(p.read_text())
    except yaml.YAMLError:
        return None


# ─── Per-source extractors (adapted to real schemas) ────────────

def _extract_alignment_failures(data) -> list[dict]:
    """Track 1 — alignment underperformance per fusion class → TECHNICAL_LAYER_1.

    The current rq2_mve_shap_alignment.json does not carry per-alert
    failure_examples; it summarises by fusion class. We emit one
    observation per class where `all_3_present` falls below threshold —
    that is the operational signal that Layer 1 missed SHAP features for
    that class (proxy for "too technical / missing vocabulary").
    """
    if not data:
        return []
    by_class = (data.get("results", {}).get("by_fusion_class") or {})
    out: list[dict] = []
    for cls, stats in by_class.items():
        n = stats.get("n_alerts", 0)
        all3 = stats.get("all_3_present")
        if n == 0 or all3 is None or all3 >= LOW_ALIGNMENT_THRESHOLD:
            continue
        out.append({
            "source": "results/rq2_mve_shap_alignment.json",
            "source_entry": f"results.by_fusion_class.{cls}",
            "fusion_class": cls,
            "n_alerts": n,
            "summary": (
                f"fusion_class={cls} (n={n}): only "
                f"{all3:.0%} of alerts surface all 3 top SHAP features in "
                f"Layer 1 (threshold {LOW_ALIGNMENT_THRESHOLD:.0%}). Proxy "
                "for Layer-1 jargon/vocabulary gap."
            ),
            "all_3_present": all3,
            "two_plus_present": stats.get("two_plus_present"),
            "any_present": stats.get("any_present"),
            "_category_assignment": "TECHNICAL_LAYER_1",
            "_subtype": "alignment_underperformance",
        })
    return out


_NOVEL_FUSION_CLASSES = ("NOVEL_ANOMALY", "STRONG_NOVEL_ANOMALY")


def _extract_stability_limitations(data) -> list[dict]:
    """Track 1 — NOVEL_ANOMALY-family low stability → OTHER (known limitation).

    Per spec, this category captures the documented XGBoost-SHAP/DAE
    faithfulness gap on novel-anomaly alerts. We deliberately do NOT emit
    observations for BENIGN or other classes: low stability there is not
    the same theoretical limitation.
    """
    if not data:
        return []
    by_class = (data.get("results", {}).get("aggregate", {})
                .get("by_fusion_class") or {})
    out: list[dict] = []
    for cls in _NOVEL_FUSION_CLASSES:
        stats = by_class.get(cls)
        if not stats:
            continue
        n = stats.get("n", 0)
        mean = stats.get("mean")
        if n == 0 or mean is None or mean >= LOW_STABILITY_THRESHOLD:
            continue
        out.append({
            "source": "results/rq2_shap_stability.json",
            "source_entry": f"results.aggregate.by_fusion_class.{cls}",
            "fusion_class": cls,
            "n_alerts": n,
            "summary": (
                f"fusion_class={cls} alerts (n={n}) show mean stability "
                f"{mean:.2f}, below the {LOW_STABILITY_THRESHOLD:.2f} "
                "threshold. Known limitation: XGBoost SHAP not faithful "
                "for DAE-driven alerts."
            ),
            "mean_stability": mean,
            "pct_stable": stats.get("pct_stable"),
            "_category_assignment": "OTHER",
            "_subtype": "known_limitation_novel_anomaly",
        })
    return out


def _extract_mitre_failures(data) -> list[dict]:
    """Track 2 — grounding failure examples → MITRE_NOT_REFERENCED."""
    if not data:
        return []
    out: list[dict] = []
    for i, ex in enumerate(data.get("failure_examples", []) or []):
        out.append({
            "source": "results/rq2_mitre_grounding.json",
            "source_entry": f"failure_examples[{i}]",
            "row_id": ex.get("row_id") or ex.get("alert_id"),
            "mode": ex.get("mode"),
            "category": ex.get("category") or ex.get("attack_category"),
            "summary": (
                f"Expected MITRE terms {ex.get('expected_terms', [])} not "
                f"found in Layer 1 for "
                f"category={ex.get('category') or ex.get('attack_category')}."
            ),
            "excerpt": (ex.get("layer1_excerpt") or ex.get("layer_1") or "")[:200],
            "_category_assignment": "MITRE_NOT_REFERENCED",
            "_subtype": "grounding_failure",
        })
    return out


def _extract_word_budget_violations(data) -> list[dict]:
    """Track 3 — word budget violations → OTHER (word_budget subtype)."""
    if not data:
        return []
    out: list[dict] = []
    for i, v in enumerate(data.get("violations", []) or []):
        out.append({
            "source": "results/rq2_word_budget_audit.json",
            "source_entry": f"violations[{i}]",
            "row_id": v.get("row_id") or v.get("alert_id"),
            "mode": v.get("mode"),
            "summary": (
                f"Word budget violation: "
                f"{v.get('violations') or v.get('layers_over') or 'unspecified'}. "
                f"Total: {v.get('total') or v.get('total_words')} words."
            ),
            "_category_assignment": "OTHER",
            "_subtype": "word_budget",
        })
    return out


_THEME_KEYWORD_MAP: list[tuple[list[str], str]] = [
    (["jargon", "technical", "vocabulary", "feature name", "terminology"],
     "TECHNICAL_LAYER_1"),
    (["mitre", "t1", "technique", "att&ck", "attck"],
     "MITRE_NOT_REFERENCED"),
    (["do not", "do_not", "constraint", "safety", "isolation", "ventilator"],
     "DO_NOT_IGNORED"),
    (["role", "view mismatch", "irrelevant", "wrong audience", "audience"],
     "ROLE_VIEW_MISMATCH"),
]


def _extract_qualitative_themes(themes_doc) -> list[dict]:
    """Track 4 — confusion_patterns → mapped to categories via keyword scan.

    Anything unmapped lands in OTHER. Empty themes (template) yields [].
    """
    if not themes_doc:
        return []
    out: list[dict] = []
    themes_per_role = themes_doc.get("themes_per_role") or {}
    for role, blocks in themes_per_role.items():
        for theme in (blocks or {}).get("confusion_patterns") or []:
            theme_text = (theme.get("theme") or "").lower().strip()
            if not theme_text:
                continue
            category = "OTHER"
            for keywords, cat in _THEME_KEYWORD_MAP:
                if any(k in theme_text for k in keywords):
                    category = cat
                    break
            out.append({
                "source": "survey/qualitative_themes.yaml",
                "source_entry": f"themes_per_role.{role}.confusion_patterns",
                "role": role,
                "summary": (
                    f"[{role}] {theme.get('theme', '<unnamed>')} "
                    f"(frequency: {theme.get('frequency', 0)})"
                ),
                "excerpt": (theme.get("example_quote") or "")[:200],
                "_category_assignment": category,
                "_subtype": "qualitative_theme",
            })
    return out


# ─── Assembly ──────────────────────────────────────────────────

def main() -> None:
    manifest = _load_manifest()
    categories = manifest.get("categories", []) or []
    category_by_id = {c["id"]: c for c in categories}

    sources = {
        "results/rq2_mve_shap_alignment.json": _try_load_json,
        "results/rq2_shap_stability.json": _try_load_json,
        "results/rq2_mitre_grounding.json": _try_load_json,
        "results/rq2_word_budget_audit.json": _try_load_json,
        "survey/qualitative_themes.yaml": _try_load_yaml,
    }
    loaded = {rel: loader(rel) for rel, loader in sources.items()}

    sources_used = [rel for rel, data in loaded.items() if data is not None]
    sources_missing = [rel for rel, data in loaded.items() if data is None]

    all_evidence: list[dict] = []
    all_evidence += _extract_alignment_failures(
        loaded["results/rq2_mve_shap_alignment.json"])
    all_evidence += _extract_stability_limitations(
        loaded["results/rq2_shap_stability.json"])
    all_evidence += _extract_mitre_failures(
        loaded["results/rq2_mitre_grounding.json"])
    all_evidence += _extract_word_budget_violations(
        loaded["results/rq2_word_budget_audit.json"])
    all_evidence += _extract_qualitative_themes(
        loaded["survey/qualitative_themes.yaml"])

    catalog: dict = {}
    for cat_id in [c["id"] for c in categories]:
        cat_meta = category_by_id[cat_id]
        cat_evidence = [e for e in all_evidence
                        if e.get("_category_assignment") == cat_id]
        catalog[cat_id] = {
            "name": cat_meta["name"],
            "description": cat_meta["description"],
            "impact_metric": cat_meta.get("impact_metric"),
            "recommended_iteration": cat_meta.get("recommended_iteration"),
            "n_observations": len(cat_evidence),
            "_status": ("observed_not_fixed" if cat_evidence
                        else "no_observations_collected"),
            "evidence": cat_evidence[:EVIDENCE_TRUNCATION],
            "evidence_truncated_at": EVIDENCE_TRUNCATION,
            "evidence_total_count": len(cat_evidence),
        }

    by_cat = {cid: catalog[cid]["n_observations"] for cid in catalog}
    total = sum(by_cat.values())
    other_size = by_cat.get("OTHER", 0)
    other_pct = round(other_size / total, 4) if total else 0.0
    if total == 0:
        other_assessment = (
            "No observations collected — catalog awaiting upstream content "
            "or qualitative coding."
        )
    elif other_pct < 0.20:
        other_assessment = "Within acceptable range (<20%). Taxonomy stable."
    elif other_pct < 0.40:
        other_assessment = (
            "Elevated (20-40%) — taxonomy stable for now but worth watching."
        )
    else:
        other_assessment = (
            "Above 40% of total — taxonomy likely needs revision (consider "
            "adding a new fixed category in next iteration)."
        )

    disclosure = {
        "framing": "observation_not_improvement",
        "iteration_performed": False,
        "evaluation_rounds": 1,
        "intended_use": (
            "Failure observations are reported as evidence that the MVE "
            "evaluation framework can systematically surface problems. "
            "Per thesis Section 7.2.3, addressing these is future work."
        ),
        "taxonomy_predates_data": bool(manifest.get("taxonomy_predates_data")),
        "taxonomy_locked_on": manifest.get("taxonomy_locked_on"),
        "taxonomy_source": manifest.get("taxonomy_source"),
    }

    warning = (f"Catalog is INCOMPLETE — {len(sources_missing)} source(s) "
               f"unavailable: {sources_missing}") if sources_missing else None
    # Flag empty-but-present qualitative themes — distinct from "missing".
    # A "real" theme has a non-empty `theme` string; template placeholders
    # (theme: "") don't count.
    themes = loaded.get("survey/qualitative_themes.yaml")
    themes_empty = bool(themes) and not any(
        (t.get("theme") or "").strip()
        for b in (themes.get("themes_per_role") or {}).values()
        for t in (b or {}).get("confusion_patterns") or []
    )
    if themes_empty:
        notice = ("survey/qualitative_themes.yaml present but contains no "
                  "coded confusion_patterns yet; qualitative observations "
                  "will be empty until manual coding completes.")
        warning = f"{warning} | {notice}" if warning else notice

    out = {
        "_meta": {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": "analysis/compile_failure_modes.py",
            "manifest_path": str(MANIFEST_PATH.relative_to(REPO_ROOT)),
            "taxonomy_locked_on": manifest.get("taxonomy_locked_on"),
            "rescope_note": manifest.get("rescope_note"),
            "sources_used": sources_used,
            "sources_missing": sources_missing,
            "_warning": warning,
        },
        "_disclosure": disclosure,
        "summary": {
            "total_observations": total,
            "by_category": by_cat,
            "other_bucket_diagnostic": {
                "size": other_size,
                "size_pct": other_pct,
                "_assessment": other_assessment,
            },
        },
        "catalog": catalog,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))

    print(f"Wrote {OUT_PATH.relative_to(REPO_ROOT)}")
    print(f"Total observations: {total}")
    for cid, n in by_cat.items():
        print(f"  {cid:25s} {n}")
    if sources_missing:
        print(f"WARNING: {len(sources_missing)} source(s) missing")
    if themes_empty:
        print("NOTICE: qualitative_themes.yaml has no coded confusion_patterns yet")


if __name__ == "__main__":
    main()
