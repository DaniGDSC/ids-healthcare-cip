"""Phase-0 instrumentation metrics for the faithfulness/actionability upgrade.

Three baseline metrics, all derived from artifacts that Module 4/5 already
write to ``results/reports/``:

  - ``narrative_faithfulness``  — P(clinician narrative top-1 category ==
        SHAP top-1 category). Computed by joining ``analyst_report.json``
        (SHAP top features) with ``clinician_summaries.json`` (rendered
        narrative) on ``sample_index`` and reverse-mapping the narrative
        phrase back to its feature category.

  - ``action_specificity``      — % of clinician + MVE actions that contain
        at least one concrete identifier (device id, port, IP, contact
        extension, named role). Generic templates score 0; parametrised
        actions score 1.

  - ``counterfactual_coverage`` — % of alert records that carry a
        ``counterfactual`` payload. Phase 0 expects 0; the metric exists
        so later phases have a measurable baseline to lift.

The metrics are deliberately read-only over on-disk artifacts so they can
run as a CI gate without re-executing Module 4. See ``tools/phase0_baseline.py``
for the driver that writes ``results/phase0_baseline.json``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .feature_groups import _FEATURE_GROUPS, _feature_to_narrative


# ── Narrative reverse-mapping ────────────────────────────────────────


def _build_narrative_to_category() -> dict[str, str]:
    """Invert ``_FEATURE_GROUPS`` to map narrative phrase → category.

    Multiple features map to the same narrative (by design — that's the
    stability-defense collapse), so the inversion is phrase→category,
    not phrase→feature. Ambiguity between categories is resolved by
    first occurrence (the dict is iteration-ordered in Python 3.7+, and
    ``_FEATURE_GROUPS`` is grouped by category in the source file).
    """
    narrative_to_cat: dict[str, str] = {}
    for _feat, (phrase, category) in _FEATURE_GROUPS.items():
        narrative_to_cat.setdefault(phrase.strip().lower(), category)
    return narrative_to_cat


_NARRATIVE_TO_CATEGORY = _build_narrative_to_category()


def narrative_category_from_summary(summary: str) -> str | None:
    """Reverse-map a clinician summary string to a feature category.

    The summary is produced by ``format_clinician_template`` which
    embeds the narrative phrase verbatim. We search for the longest
    matching phrase to avoid prefix collisions (e.g. "abnormal blood
    pressure" before "abnormal").

    Returns the category slug (e.g. ``"biometric"``) or ``None`` when
    no known phrase appears — that happens for unknown-feature fallback
    or for LOW-severity boilerplate.
    """
    if not summary:
        return None
    text = summary.lower()
    best: tuple[int, str | None] = (0, None)
    for phrase, category in _NARRATIVE_TO_CATEGORY.items():
        if phrase in text and len(phrase) > best[0]:
            best = (len(phrase), category)
    return best[1]


# ── Metric 1: narrative_faithfulness ─────────────────────────────────


def compute_narrative_faithfulness(
    analyst_report: list[dict],
    clinician_summaries: list[dict],
    *,
    model_name: str = "xgboost",
) -> dict:
    """Match rate between SHAP top-1 category and narrative top-1 category.

    Both artefacts are keyed by ``sample_index``; clinician summaries
    are XGBoost-flagged only (see ``build_clinician_summaries``) so we
    inner-join against the XGBoost top features in the analyst report.

    Returns a dict with ``n``, ``n_matched``, ``rate``, two unknown
    counters (for the LOW boilerplate and unknown-feature cases), and
    up to 10 example mismatches for debugging.
    """
    analyst_by_idx = {r["sample_index"]: r for r in analyst_report}
    n = 0
    n_matched = 0
    n_unknown_narrative = 0
    n_unknown_shap_cat = 0
    mismatches: list[dict] = []

    for entry in clinician_summaries:
        idx = entry["sample_index"]
        analyst_entry = analyst_by_idx.get(idx)
        if analyst_entry is None:
            continue
        model_block = analyst_entry.get("models", {}).get(model_name, {})
        top = model_block.get("top_features")
        if not top:
            continue
        shap_feat = top[0]["feature"]
        _, shap_cat = _feature_to_narrative(shap_feat)
        if shap_cat == "unknown":
            n_unknown_shap_cat += 1
            continue

        narr_cat = narrative_category_from_summary(entry.get("summary", ""))
        if narr_cat is None:
            n_unknown_narrative += 1
            continue

        n += 1
        if narr_cat == shap_cat:
            n_matched += 1
        elif len(mismatches) < 10:
            mismatches.append({
                "sample_index": idx,
                "shap_top_feature": shap_feat,
                "shap_category": shap_cat,
                "narrative_category": narr_cat,
            })

    rate = (n_matched / n) if n else 0.0
    return {
        "metric": "narrative_faithfulness",
        "description": (
            f"P(clinician narrative top-1 category == SHAP top-1 category) "
            f"for model={model_name}"
        ),
        "n": n,
        "n_matched": n_matched,
        "rate": round(rate, 4),
        "n_unknown_narrative": n_unknown_narrative,
        "n_unknown_shap_category": n_unknown_shap_cat,
        "sample_mismatches": mismatches,
    }


# ── Metric 2: action_specificity ─────────────────────────────────────


_SPECIFICITY_PATTERNS: dict[str, re.Pattern[str]] = {
    "ipv4":          re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"),
    "mac_address":   re.compile(r"\b(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}\b"),
    "port_number":   re.compile(r"\b(?:port|tcp|udp)[\s/:]*\d{2,5}\b", re.IGNORECASE),
    "bed_or_room":   re.compile(r"\b(?:bed|room|ward)[-\s]?\d+\b", re.IGNORECASE),
    "device_id":     re.compile(r"\b(?:device|monitor|pump|sensor)[-_\s]?id[-_\s:=]?\S+", re.IGNORECASE),
    "extension":     re.compile(r"\bext(?:ension)?[\s.:]*\d{3,5}\b", re.IGNORECASE),
    "named_person":  re.compile(r"\bDr\.?\s+[A-Z][a-z]+\b"),
    "mitre_id":      re.compile(r"\bT\d{4}(?:\.\d{3})?\b"),
    "alert_id":      re.compile(r"\bALERT-\d{3,}\b"),
}


def is_specific(text: str) -> tuple[bool, list[str]]:
    """Return ``(is_specific, matched_signals)``.

    A string is "specific" iff at least one signal pattern matches. The
    signal list is returned so the baseline can report *which* kind of
    specificity is present in the current corpus (useful for choosing
    where to invest Phase 1 effort).
    """
    if not text:
        return False, []
    hits = [name for name, pat in _SPECIFICITY_PATTERNS.items() if pat.search(text)]
    return bool(hits), hits


def compute_action_specificity(
    alert_responses: list[dict],
    clinician_summaries: list[dict],
) -> dict:
    """% of action strings (MVE Layer 3 + clinician summary) that contain
    at least one concrete identifier.

    Considered action sources per record:
      - ``response.action_descriptions`` (joined with " | ")
      - ``explanation.mve.layer_3.immediate_action``
      - ``explanation.mve.layer_3.escalation_path``
      - the clinician summary string (joined separately so the score is
        attributed to the clinician view, not the action engine)
    """
    sources_seen = {"layer3_action": 0, "layer3_escalation": 0,
                    "action_descriptions": 0, "clinician_summary": 0}
    sources_specific = {k: 0 for k in sources_seen}
    signal_counter: dict[str, int] = {k: 0 for k in _SPECIFICITY_PATTERNS}

    clinician_by_idx = {c["sample_index"]: c.get("summary", "") for c in clinician_summaries}

    for rec in alert_responses:
        idx = rec.get("sample_index")
        mve = (rec.get("explanation") or {}).get("mve") or {}
        l3 = mve.get("layer_3") or {}
        for src_key, text in (
            ("layer3_action", l3.get("immediate_action", "")),
            ("layer3_escalation", l3.get("escalation_path", "")),
        ):
            if text:
                sources_seen[src_key] += 1
                ok, hits = is_specific(text)
                if ok:
                    sources_specific[src_key] += 1
                    for h in hits:
                        signal_counter[h] += 1

        descs = (rec.get("response") or {}).get("action_descriptions") or []
        if descs:
            sources_seen["action_descriptions"] += 1
            joined = " | ".join(descs)
            ok, hits = is_specific(joined)
            if ok:
                sources_specific["action_descriptions"] += 1
                for h in hits:
                    signal_counter[h] += 1

        cs = clinician_by_idx.get(idx, "")
        if cs:
            sources_seen["clinician_summary"] += 1
            ok, hits = is_specific(cs)
            if ok:
                sources_specific["clinician_summary"] += 1
                for h in hits:
                    signal_counter[h] += 1

    rates = {
        k: round(sources_specific[k] / sources_seen[k], 4) if sources_seen[k] else 0.0
        for k in sources_seen
    }
    total_seen = sum(sources_seen.values())
    total_specific = sum(sources_specific.values())
    overall = round(total_specific / total_seen, 4) if total_seen else 0.0

    return {
        "metric": "action_specificity",
        "description": (
            "% of action strings (MVE Layer 3 + clinician summary + "
            "response.action_descriptions) containing at least one concrete "
            "identifier (IP/port/device/contact/MITRE/alert id)."
        ),
        "overall_rate": overall,
        "per_source_rate": rates,
        "per_source_seen": sources_seen,
        "per_source_specific": sources_specific,
        "signal_breakdown": signal_counter,
    }


# ── Metric 3: counterfactual_coverage ────────────────────────────────


def _has_counterfactual(record: dict) -> bool:
    """Return True if the record carries a non-empty counterfactual payload.

    Phase 0 expects this to be uniformly False — the metric only exists
    so later phases (Phase 2) have a baseline to lift. We accept the
    payload at any of three plausible locations so the Phase 2 schema
    decision is not pre-empted:
      - ``record["counterfactual"]``  (top-level)
      - ``record["explanation"]["counterfactual"]``
      - ``record["explanation"]["mve"]["counterfactual"]``
    """
    if record.get("counterfactual"):
        return True
    expl = record.get("explanation") or {}
    if expl.get("counterfactual"):
        return True
    mve = expl.get("mve") or {}
    if mve.get("counterfactual"):
        return True
    return False


_ACTIONABLE_TIERS = {"CRITICAL", "HIGH", "MEDIUM"}


def _is_feasible_cf(record: dict) -> bool:
    """True iff the record has a counterfactual marked ``feasible=True``."""
    for cf in (
        record.get("counterfactual"),
        (record.get("explanation") or {}).get("counterfactual"),
        ((record.get("explanation") or {}).get("mve") or {}).get("counterfactual"),
    ):
        if cf and isinstance(cf, dict) and cf.get("feasible"):
            return True
    return False


def compute_counterfactual_coverage(alert_responses: list[dict]) -> dict:
    """Counterfactual coverage, split by severity tier.

    Phase 2 acceptance is keyed on the **actionable** tier (records
    where a clinician/admin would actually take an action — CRITICAL,
    HIGH, MEDIUM). LOW alerts mostly come from non-XGBoost detectors
    (DAE-only or risk-elevated), where the current XGBoost counterfactual
    engine can't produce a result. The overall rate is kept for
    backwards-compat with the Phase-0 baseline file.

    Returns the same shape as before plus:
      - ``actionable_rate``       — % of actionable records (MEDIUM/HIGH/
        CRITICAL) with any counterfactual payload
      - ``actionable_feasible_rate`` — same denominator, but only counts
        counterfactuals marked ``feasible=True``
      - ``by_severity`` — per-tier ``{seen, with_cf, feasible}`` for
        debugging / dashboard rendering
    """
    n = len(alert_responses)
    n_cf = sum(1 for r in alert_responses if _has_counterfactual(r))
    rate = (n_cf / n) if n else 0.0

    by_severity: dict[str, dict[str, int]] = {
        tier: {"seen": 0, "with_cf": 0, "feasible": 0}
        for tier in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
    }
    for r in alert_responses:
        sev = r.get("risk_level") or "LOW"
        if sev not in by_severity:
            continue
        by_severity[sev]["seen"] += 1
        if _has_counterfactual(r):
            by_severity[sev]["with_cf"] += 1
        if _is_feasible_cf(r):
            by_severity[sev]["feasible"] += 1

    actionable_seen     = sum(by_severity[t]["seen"]     for t in _ACTIONABLE_TIERS)
    actionable_with_cf  = sum(by_severity[t]["with_cf"]  for t in _ACTIONABLE_TIERS)
    actionable_feasible = sum(by_severity[t]["feasible"] for t in _ACTIONABLE_TIERS)
    actionable_rate          = (actionable_with_cf  / actionable_seen) if actionable_seen else 0.0
    actionable_feasible_rate = (actionable_feasible / actionable_seen) if actionable_seen else 0.0

    return {
        "metric": "counterfactual_coverage",
        "description": (
            "% of alert records with a counterfactual payload. The "
            "``actionable_rate`` (denominator = CRITICAL+HIGH+MEDIUM) is "
            "the Phase 2 acceptance signal; the overall ``rate`` is kept "
            "for backwards-compat with Phase 0 baselines."
        ),
        "n": n,
        "n_with_counterfactual": n_cf,
        "rate": round(rate, 4),
        "actionable_rate":          round(actionable_rate, 4),
        "actionable_feasible_rate": round(actionable_feasible_rate, 4),
        "by_severity": by_severity,
    }


# ── Top-level driver ─────────────────────────────────────────────────


def _load_json(path: Path) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "records" in data:
        return data["records"]
    return data


def collect_baseline(reports_dir: Path) -> dict:
    """Compute all Phase-0 metrics from a reports directory.

    Required files (raises ``FileNotFoundError`` if missing):
      - ``analyst_report.json``
      - ``clinician_summaries.json``
      - ``alert_responses.json``
    """
    analyst = _load_json(reports_dir / "analyst_report.json")
    clinician = _load_json(reports_dir / "clinician_summaries.json")
    responses = _load_json(reports_dir / "alert_responses.json")

    return {
        "_meta": {
            "reports_dir": str(reports_dir),
            "n_analyst_alerts": len(analyst),
            "n_clinician_summaries": len(clinician),
            "n_alert_responses": len(responses),
        },
        "narrative_faithfulness":  compute_narrative_faithfulness(analyst, clinician),
        "action_specificity":      compute_action_specificity(responses, clinician),
        "counterfactual_coverage": compute_counterfactual_coverage(responses),
    }


__all__ = [
    "compute_narrative_faithfulness",
    "compute_action_specificity",
    "compute_counterfactual_coverage",
    "collect_baseline",
    "narrative_category_from_summary",
    "is_specific",
]
