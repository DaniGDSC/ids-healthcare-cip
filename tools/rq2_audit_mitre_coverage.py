#!/usr/bin/env python3
"""RQ2.e — Audit MITRE ATT&CK coverage + Layer 1 MITRE-reference rate.

Two outputs:
  1. `results/rq2_mitre_coverage.json` — per-attack-category mapping
     summary + coverage statistics
  2. Embedded in the same JSON: Layer 1 MITRE-reference rate computed
     across MVE narratives (sample_explanations.json + alert_responses.json
     explanation.clinician_summary).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG = PROJECT_ROOT / "config" / "attack_to_mitre_mapping.yaml"
REPORTS = PROJECT_ROOT / "results" / "reports"
OUT = PROJECT_ROOT / "results" / "rq2_mitre_coverage.json"


# Regex matches MITRE technique IDs like T1234, T1234.001, T0830 (ICS)
_MITRE_ID_RE = re.compile(r"\bT\d{3,4}(?:\.\d{3})?\b")


def audit_config_coverage(cfg: dict) -> dict:
    """Compute coverage statistics from the YAML mapping."""
    categories = cfg.get("attack_categories", {})
    n_total = len(categories)
    n_mapped = 0
    n_with_subs = 0
    n_with_ics = 0
    orphans = []
    per_category = []

    for name, info in categories.items():
        if info.get("excluded_from_coverage_audit"):
            n_total -= 1  # don't count "normal" in coverage denominator
            continue
        primary = info.get("primary_technique", {}) or {}
        primary_id = primary.get("id")
        has_mapping = bool(primary_id) and primary_id != "NONE"
        subs = info.get("sub_techniques", []) or []
        ics = info.get("ics_techniques", []) or []

        if has_mapping:
            n_mapped += 1
        else:
            orphans.append(name)
        if subs:
            n_with_subs += 1
        if ics:
            n_with_ics += 1

        per_category.append({
            "category": name,
            "in_current_corpus": info.get("in_current_corpus", True),
            "primary_technique_id": primary_id,
            "primary_technique_name": primary.get("name"),
            "primary_confidence": primary.get("confidence"),
            "n_sub_techniques": len(subs),
            "n_ics_techniques": len(ics),
            "n_related_techniques": len(info.get("related_techniques", []) or []),
        })

    return {
        "n_total_categories": n_total,
        "n_mapped": n_mapped,
        "pct_mapped": round(n_mapped / n_total * 100, 2) if n_total else 0.0,
        "n_with_sub_techniques": n_with_subs,
        "n_with_ics_techniques": n_with_ics,
        "orphans": orphans,
        "target_pct_mapped": 100.0,
        "target_met_no_orphans": len(orphans) == 0,
        "framework_version_pinned": bool(cfg.get("mitre_framework_version")),
        "framework_version": cfg.get("mitre_framework_version"),
        "per_category": per_category,
    }


def compute_mitre_reference_rate(cfg: dict) -> dict:
    """Layer 1 MITRE-reference rate over MVE narratives.

    Sources:
      • results/reports/sample_explanations.json  — 20 explanations
      • results/reports/example_explanations.json — 5 explanations (analyst views)

    A narrative is counted as "referencing MITRE" if it contains a
    technique ID matching the regex T\\d{3,4}(\\.\\d+)?  OR  the technique
    NAME (case-insensitive) declared in the mapping for that alert's
    attack category.
    """
    # Build lookup: attack_category → set of expected names + ids
    expected = {}
    for cat, info in cfg.get("attack_categories", {}).items():
        names = []
        ids = []
        primary = info.get("primary_technique", {}) or {}
        if primary.get("id") and primary.get("id") != "NONE":
            ids.append(primary["id"])
            names.append(primary.get("name", "").lower())
        for sub in (info.get("sub_techniques", []) or []):
            ids.append(sub["id"])
            names.append(sub.get("name", "").lower())
        for ics in (info.get("ics_techniques", []) or []):
            ids.append(ics["id"])
            names.append(ics.get("name", "").lower())
        expected[cat] = {"ids": set(ids), "names": {n for n in names if n}}

    def _references_mitre(narrative: str, category: str) -> tuple[bool, str]:
        """Return (matches, reason)."""
        if not narrative:
            return False, "empty narrative"
        # Strategy 1: any MITRE-style ID present
        m = _MITRE_ID_RE.search(narrative)
        if m:
            return True, f"matched_id: {m.group(0)}"
        # Strategy 2: expected technique name (per the alert's category)
        exp = expected.get(category, {"ids": set(), "names": set()})
        lower = narrative.lower()
        for name in exp["names"]:
            if name and name in lower:
                return True, f"matched_name: {name}"
        return False, "no MITRE reference"

    # Load explanations
    narratives = []
    for path, src_label in [
        (REPORTS / "sample_explanations.json", "sample_explanations.json"),
        (REPORTS / "example_explanations.json", "example_explanations.json"),
    ]:
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        for s in data:
            if "clinician_summary" in s:
                narratives.append({
                    "source": src_label,
                    "sample_index": s.get("sample_index"),
                    "attack_category": s.get("attack_category", "unknown"),
                    "narrative": s["clinician_summary"],
                })
            elif "views" in s:
                # example_explanations format: views.clinician.content is text
                v = s.get("views", {}).get("clinician") or {}
                if isinstance(v, dict) and isinstance(v.get("content"), str):
                    narratives.append({
                        "source": src_label,
                        "sample_index": s.get("sample_index"),
                        "attack_category": s.get("attack_category", "unknown"),
                        "narrative": v["content"],
                    })

    # Evaluate references
    per_narrative = []
    n_referencing = 0
    per_category_hits = {}
    for n in narratives:
        cat = n["attack_category"]
        matches, reason = _references_mitre(n["narrative"], cat)
        per_narrative.append({
            "source": n["source"],
            "sample_index": n["sample_index"],
            "category": cat,
            "references_mitre": matches,
            "match_reason": reason,
        })
        if matches:
            n_referencing += 1
        bucket = per_category_hits.setdefault(cat, {"total": 0, "hits": 0})
        bucket["total"] += 1
        if matches:
            bucket["hits"] += 1

    for cat, b in per_category_hits.items():
        b["hit_rate_pct"] = round(b["hits"] / b["total"] * 100, 2) if b["total"] else 0.0

    n = len(narratives)
    return {
        "n_narratives_evaluated": n,
        "n_referencing_mitre": n_referencing,
        "pct_referencing_mitre": round(n_referencing / n * 100, 2) if n else 0.0,
        "target_pct": 90.0,
        "target_met": bool(n_referencing / n >= 0.90) if n else False,
        "per_category": per_category_hits,
        "per_narrative": per_narrative,
    }


def compute_fresh_mve_reference_rate(cfg: dict, n_samples: int = 50) -> dict:
    """Generate fresh MVE narratives via src.mve_generator and measure
    MITRE-reference rate on the freshly produced Layer 1 text.

    The cached `sample_explanations.json` / `example_explanations.json`
    files were produced before the MITRE-injection fix (RQ2.e G3), so
    auditing them shows the old behavior. This function regenerates
    narratives at runtime to validate the fix.
    """
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.mve_generator import generate_mve

    # Pull a sample of test-split alerts spread across attack categories
    resp_path = REPORTS / "alert_responses.json"
    if not resp_path.exists():
        return {"error": f"{resp_path} missing — cannot regenerate"}
    with open(resp_path) as f:
        data = json.load(f)
    records = data.get("records", data) if isinstance(data, dict) else data

    # Per-sample top SHAP features (G3 v2 fix — was a synthetic constant
    # for every sample, misleading because Layer 1 generation actually
    # cites the features). When analyst_report is unavailable, fall back
    # to the synthetic constant.
    shap_by_sidx: dict[int, list[str]] = {}
    analyst_path = REPORTS / "analyst_report.json"
    if analyst_path.exists():
        with open(analyst_path) as f:
            analyst = json.load(f)
        for a in analyst:
            xgb = (a.get("models") or {}).get("xgboost") or {}
            top = [f["feature"] for f in xgb.get("top_features") or []]
            if top:
                shap_by_sidx[a["sample_index"]] = top
    _FALLBACK_SHAP = ["Flgs", "DIntPkt", "Dur"]

    # Stratified sample: ensure each non-normal category gets ≥10 entries
    by_cat = {}
    for r in records:
        cat = r.get("attack_category", "unknown")
        by_cat.setdefault(cat, []).append(r)

    chosen = []
    target_per_category = max(10, n_samples // max(1, len(by_cat)))
    for cat, recs in by_cat.items():
        take = min(len(recs), target_per_category)
        chosen.extend(recs[:take])
    chosen = chosen[:n_samples]

    # Build lookup for MITRE expected names (same logic as cached path)
    expected = {}
    for cat, info in cfg.get("attack_categories", {}).items():
        names = []
        ids = []
        primary = info.get("primary_technique", {}) or {}
        if primary.get("id") and primary.get("id") != "NONE":
            ids.append(primary["id"])
            names.append(primary.get("name", "").lower())
        expected[cat] = {"ids": set(ids), "names": {n for n in names if n}}

    n_total = 0
    n_ref = 0
    n_total_attack_class = 0   # excludes benign baseline per spec
    n_ref_attack_class = 0
    n_gen_failures = 0
    gen_failures_log: list[dict] = []
    per_category_hits = {}
    samples_audit = []

    # Categories that don't get a MITRE reference by design — these
    # are excluded from the denominator when computing the spec's
    # ≥90% reference-rate target.
    excluded_cats = set()
    for cat_name, cat_info in (cfg.get("attack_categories") or {}).items():
        if cat_info.get("excluded_from_coverage_audit"):
            excluded_cats.add(cat_name)

    for r in chosen:
        cat = r.get("attack_category", "unknown")
        sidx = r.get("sample_index")
        raw_alert = {
            "alert_id": f"SAMPLE-{sidx:04d}",
            "severity": r.get("risk_level", "MEDIUM"),
            "alert_type": "anomalous_outbound_connection",
            "attack_category": cat,
        }
        device_context = {
            "device_type": "patient_monitor",
            "clinical_function": "vitals_monitoring",
            "location": "clinical area",
            "criticality": r.get("risk_level", "MEDIUM"),
            "patchable": True,
        }
        baseline = {
            "normal_destinations": ["internal hosts"],
            "normal_protocols": ["HTTPS"],
            "normal_hours": "business hours",
            "baseline_days": 90,
        }
        top_features = shap_by_sidx.get(sidx, _FALLBACK_SHAP)
        try:
            mve = generate_mve(
                raw_alert=raw_alert, device_context=device_context,
                baseline=baseline, user_context=None,
                shap_context={
                    "top_features": top_features,
                    "top_category": "network_protocol",
                    "shap_direction": "elevated",
                },
                force_rule_based=True,
            )
            text = " ".join(mve.layer_1.get(k, "") for k in
                            ("baseline_behavior", "deviation_description",
                             "confidence_indicator"))
        except Exception as e:
            # G3 v2 fix — was: text = "" (counted failed gen as "no MITRE
            # reference" → silently biased the rate down). Now: skip the
            # sample, log, and surface the count in the report so reviewers
            # can spot generator regressions.
            n_gen_failures += 1
            gen_failures_log.append({
                "sample_index": sidx,
                "category": cat,
                "error": f"{type(e).__name__}: {e}",
            })
            print(f"WARN: generate_mve failed for sample {sidx} ({cat}): "
                  f"{type(e).__name__}: {e}")
            continue

        n_total += 1
        m = _MITRE_ID_RE.search(text)
        matches = bool(m)
        if not matches:
            exp = expected.get(cat, {"names": set()})
            for name in exp["names"]:
                if name and name in text.lower():
                    matches = True
                    break
        if matches:
            n_ref += 1
        # Attack-class denominator (excludes benign baseline)
        if cat not in excluded_cats:
            n_total_attack_class += 1
            if matches:
                n_ref_attack_class += 1
        bucket = per_category_hits.setdefault(cat, {"total": 0, "hits": 0})
        bucket["total"] += 1
        if matches:
            bucket["hits"] += 1
        samples_audit.append({
            "sample_index": r.get("sample_index"),
            "category": cat,
            "references_mitre": matches,
            "layer1_excerpt": text[:160],
        })

    for cat, b in per_category_hits.items():
        b["hit_rate_pct"] = round(b["hits"] / b["total"] * 100, 2)

    # Two denominators reported: all-categories (includes benign noise)
    # vs attack-class-only (the spec's "when applicable" filter).
    pct_all = (n_ref / n_total * 100) if n_total else 0.0
    pct_attack = (n_ref_attack_class / n_total_attack_class * 100) if n_total_attack_class else 0.0
    return {
        "n_narratives_evaluated": n_total,
        "n_referencing_mitre": n_ref,
        "pct_referencing_mitre_all": round(pct_all, 2),
        "n_attack_class_evaluated": n_total_attack_class,
        "n_attack_class_referencing": n_ref_attack_class,
        "pct_referencing_mitre_attack_class": round(pct_attack, 2),
        "target_pct": 90.0,
        "target_met_attack_class": bool(pct_attack >= 90.0),
        "target_met_all": bool(pct_all >= 90.0),
        "n_gen_failures": n_gen_failures,
        "gen_failures": gen_failures_log[:20],
        "per_category": per_category_hits,
        "method": (
            "Regenerated at runtime via src.mve_generator (post-G3 fix). "
            "attack_class denominator excludes 'normal' baseline samples "
            "since benign traffic has no MITRE mapping by design (see "
            f"config/attack_to_mitre_mapping.yaml — excluded: {sorted(excluded_cats)}). "
            "Per-sample top-3 SHAP features pulled from analyst_report.json; "
            "samples missing from that file fall back to a synthetic constant."
        ),
        "samples_audit": samples_audit[:10],
    }


def main():
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)

    config_audit = audit_config_coverage(cfg)
    reference_rate = compute_mitre_reference_rate(cfg)
    reference_rate_fresh = compute_fresh_mve_reference_rate(cfg)

    report = {
        "_meta": {
            "description": "MITRE ATT&CK coverage + Layer 1 MITRE-reference rate",
            "config_source": str(CONFIG.relative_to(PROJECT_ROOT)),
            "framework_version": cfg.get("mitre_framework_version"),
            "mapping_yaml_version": cfg.get("version"),
        },
        "config_coverage": config_audit,
        "layer1_mitre_reference_rate_cached": reference_rate,
        "layer1_mitre_reference_rate_fresh": reference_rate_fresh,
        "overall_status": "PASS" if (
            config_audit["target_met_no_orphans"]
            and config_audit["framework_version_pinned"]
            and reference_rate_fresh.get("target_met_attack_class")
        ) else "PARTIAL",
    }

    with open(OUT, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"=== MITRE Config Coverage ===")
    print(f"  total categories (excluding 'normal'): {config_audit['n_total_categories']}")
    print(f"  mapped: {config_audit['n_mapped']} ({config_audit['pct_mapped']}%)")
    print(f"  with sub-techniques: {config_audit['n_with_sub_techniques']}")
    print(f"  with ICS techniques: {config_audit['n_with_ics_techniques']}")
    print(f"  orphans: {config_audit['orphans'] or '(none)'}")
    print(f"  framework version pinned: {config_audit['framework_version_pinned']} ({config_audit['framework_version']})")

    print()
    print(f"=== Layer 1 MITRE Reference Rate — CACHED narratives ===")
    print(f"  (sample_explanations.json — pre-G3-fix; expected 0%)")
    print(f"  narratives: {reference_rate['n_narratives_evaluated']}  "
          f"refs: {reference_rate['n_referencing_mitre']} "
          f"({reference_rate['pct_referencing_mitre']}%)")

    print()
    print(f"=== Layer 1 MITRE Reference Rate — FRESH narratives (post-G3 fix) ===")
    fr = reference_rate_fresh
    if "error" in fr:
        print(f"  ERROR: {fr['error']}")
    else:
        print(f"  all categories:    {fr['n_referencing_mitre']}/{fr['n_narratives_evaluated']} "
              f"({fr['pct_referencing_mitre_all']}%)")
        print(f"  attack-class only: {fr['n_attack_class_referencing']}/{fr['n_attack_class_evaluated']} "
              f"({fr['pct_referencing_mitre_attack_class']}%)  ← spec metric")
        print(f"  target: {fr['target_pct']}% — met (attack-class): {fr['target_met_attack_class']}")
        if fr.get("n_gen_failures"):
            print(f"  ⚠ generator failures: {fr['n_gen_failures']} sample(s) skipped — "
                  "see gen_failures in JSON output")
        print(f"  per-category hit rates:")
        for cat, b in fr["per_category"].items():
            print(f"    {cat:20s} {b['hits']}/{b['total']} ({b['hit_rate_pct']}%)")

    print()
    print(f"OVERALL: {report['overall_status']}")
    print(f"  → {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
