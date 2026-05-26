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


def main():
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)

    config_audit = audit_config_coverage(cfg)
    reference_rate = compute_mitre_reference_rate(cfg)

    report = {
        "_meta": {
            "description": "MITRE ATT&CK coverage + Layer 1 MITRE-reference rate",
            "config_source": str(CONFIG.relative_to(PROJECT_ROOT)),
            "framework_version": cfg.get("mitre_framework_version"),
            "mapping_yaml_version": cfg.get("version"),
        },
        "config_coverage": config_audit,
        "layer1_mitre_reference_rate": reference_rate,
        "overall_status": "PASS" if (
            config_audit["target_met_no_orphans"]
            and config_audit["framework_version_pinned"]
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
    print(f"=== Layer 1 MITRE Reference Rate ===")
    print(f"  narratives evaluated: {reference_rate['n_narratives_evaluated']}")
    print(f"  referencing MITRE:    {reference_rate['n_referencing_mitre']} ({reference_rate['pct_referencing_mitre']}%)")
    print(f"  target: {reference_rate['target_pct']}% — met: {reference_rate['target_met']}")
    print(f"  per-category hit rates:")
    for cat, b in reference_rate["per_category"].items():
        print(f"    {cat:20s} {b['hits']}/{b['total']} ({b['hit_rate_pct']}%)")

    print()
    print(f"OVERALL: {report['overall_status']}")
    print(f"  → {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
