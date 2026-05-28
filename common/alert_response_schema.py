"""Pydantic 2 schema for Module 5's ``alert_responses{,_demo}.json``.

This is the single source of truth for the shape of the file Module 5
writes and Module 6 (Dashboard) consumes. Both ends validate against
this schema so a divergence in either direction fails loud at
build-time or load-time instead of surfacing as a KeyError 200 lines
deep inside a Streamlit render.

Envelope format (introduced alongside this schema):

    {
        "_provenance": {...},   # see Provenance below
        "records": [AlertRecord, ...]
    }

The loader in ``module6_app.load_responses_for`` accepts the legacy
bare-list shape too so an old artifact on disk still works while
Module 5 re-runs are pending.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RiskComponents(BaseModel):
    """Six normalised risk components from Module 3 (all in [0, 1])."""

    model_config = ConfigDict(extra="forbid")

    C_detect: float = Field(ge=0.0, le=1.0)
    C_track_a: float = Field(ge=0.0, le=1.0)
    C_track_b: float = Field(ge=0.0, le=1.0)
    D_crit: float = Field(ge=0.0, le=1.0)
    S_data: float = Field(ge=0.0, le=1.0)
    D_clinical_tier: float = Field(ge=0.0, le=1.0)


class EscalationChain(BaseModel):
    """Three-tier escalation routing — populated from ESCALATION_ROUTING.

    Any tier can be ``None`` when the attack category has no routing
    entry (the DEFAULT_ROUTING uses None for all three tiers).
    """

    model_config = ConfigDict(extra="forbid")

    primary: str | None
    secondary: str | None
    tertiary: str | None


class ActionMetadata(BaseModel):
    """Per-action operational properties surfaced to stakeholder views.

    Sourced from ``module5_responses.config.ACTION_CATALOGUE`` so the
    response engine, audit trail, and dashboard see the same values.
    Added in Phase 1.3 of the faithfulness/actionability upgrade —
    clinician + admin views render reversibility / disruption badges
    from this list instead of re-walking the catalogue.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    cost: float = Field(ge=0.0, le=1.0)
    reversible: bool
    requires_approval: bool
    expected_disruption: str = ""


class RoutingWarning(BaseModel):
    """Per-alert routing-mismatch signal (Phase 3.2).

    Populated when the alert's SHAP top-category implies a different
    primary audience than the one the response engine assigned (e.g.
    biometric SHAP routed to IT Security). ``mismatch=False`` for
    aligned alerts; ``mismatch=True`` carries the suggested alternate
    role and a one-sentence reason a non-ML user can follow.
    """

    model_config = ConfigDict(extra="forbid")

    mismatch: bool
    current_primary: str = ""
    suggested_role: str = ""
    reason: str = ""


class Response(BaseModel):
    """Output of ``select_adaptive_response`` — the policy-engine verdict."""

    model_config = ConfigDict(extra="forbid")

    actions: list[str]
    action_descriptions: list[str]
    actions_metadata: list[ActionMetadata] = Field(default_factory=list)
    escalation_chain: EscalationChain
    escalation_rationale: str
    max_response_min: int = Field(ge=0)
    priority: int = Field(ge=1, le=5)
    rationale: str
    device_tier: str
    device_constraint_applied: bool
    # Phase 2.4 — counterfactual-derived "try first" remediation. Set when
    # the alert has a feasible counterfactual; the value is the
    # ``remediation_hint`` from ``module4_explanations.counterfactual``,
    # surfaced as a less-disruptive option the operator should attempt
    # before the containment actions in ``actions``. Empty string when
    # no counterfactual is available.
    try_first_action: str = ""
    # Phase 3.1 — conditional action playbook. Each step is one node in
    # a decision tree the operator can follow without ML knowledge.
    # Shape mirrors ``module5_responses.playbooks.Playbook.to_dict``.
    # Optional so pre-Phase-3 artefacts still validate.
    playbook: dict | None = None
    # Phase 3.2 — routing-mismatch warning. Always present (default
    # mismatch=False); the dashboard only surfaces it when
    # ``mismatch=True``.
    routing_warning: RoutingWarning | None = None
    # Phase 4.1 — auto_execute may be demoted to False when the
    # explanation is UNSTABLE (Phase 4 stability gate). Default True
    # so legacy artefacts without the field continue to behave
    # exactly as before.
    auto_execute: bool = True


class MVEPayload(BaseModel):
    """Three-layer Minimum Viable Explanation from ``src.mve_generator``.

    Stored on each alert record so the dashboard can render Layer 1
    (``Why anomalous``) without reusing Module 4's clinician_summary —
    those two strings track different concepts (composite tier vs.
    detection consensus) and previously disagreed in wording. Added
    2026-05-25 alongside the Layer 1 wording fix.

    Optional on AlertRecord so legacy artefacts without MVE still validate.
    """

    model_config = ConfigDict(extra="forbid")

    layer_1: dict[str, str]
    """Keys: baseline_behavior, deviation_description, confidence_indicator."""

    layer_2: dict[str, str]
    """Keys: affected_system, patient_care_impact, phi_exposure,
    severity_label, severity_rationale."""

    layer_3: dict[str, str]
    """Keys: immediate_action, clinical_constraint, escalation_path, timeframe."""

    why_anomalous: str
    """Concatenated layer_1 fields — direct lookup target for the dashboard's
    render_mve_layers, which expects a string at this key."""

    alert_involves_clinical_system: bool = True
    total_word_count: int = Field(ge=0)
    provider: Literal["openai", "anthropic", "rule_based"] = "rule_based"
    """Which generator path produced this MVE. Surfaces in the provenance
    summary so a reviewer can spot LLM-vs-rule-based mixes."""


class Explanation(BaseModel):
    """Pointer to Module 4 explanation artefacts."""

    model_config = ConfigDict(extra="forbid")

    clinician_summary: str
    analyst_available: bool
    mve: MVEPayload | None = None
    # Phase 2 — minimal-sparsity feature perturbation that would have
    # flipped the model from "attack" to "benign". Shape mirrors
    # ``module4_explanations.counterfactual.CounterfactualResult.to_dict``.
    # Optional so pre-Phase-2 artefacts still validate.
    counterfactual: dict | None = None
    # Phase 4.1 — explanation-stability score (bootstrap perturbation
    # of top-K SHAP). Shape mirrors
    # ``module4_explanations.stability.StabilityResult.to_dict``.
    # An UNSTABLE band downstream demotes ``auto_execute`` and adds an
    # escalate_clinical step.
    stability: dict | None = None


class AlertRecord(BaseModel):
    """One alert: join of Module 2/3/4 outputs at a per-split row offset.

    ``sample_index`` is the per-split offset (0..N-1), not a global
    row_id. Upper bound is enforced by the drift check in
    ``module5_responses._assert_no_score_drift``, not by the schema —
    the schema cannot know the split's N without context, and the
    cross-check against ``risk_scores.npz`` is the canonical proof of
    test-split provenance anyway.
    """

    model_config = ConfigDict(extra="forbid")

    sample_index: int = Field(ge=0)
    ground_truth: Literal["attack", "benign"]
    attack_category: str
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_level: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW", "NORMAL"]
    risk_components: RiskComponents
    response: Response
    explanation: Explanation


# ── Provenance (P0-1) ────────────────────────────────────────────────────


class InputFile(BaseModel):
    """Per-input-file fingerprint at the moment Module 5 ran."""

    model_config = ConfigDict(extra="forbid")

    path: str
    mtime_iso: str
    sha256: str
    size_bytes: int


class Provenance(BaseModel):
    """Self-describing metadata embedded in every Module 5 output.

    Lets the Dashboard surface a stale-data warning when an upstream
    artefact (risk_scores.npz, analyst_report.json, ...) has been
    regenerated since this file was built.
    """

    model_config = ConfigDict(extra="forbid")

    split: Literal["test", "demo"]
    generated_at: str
    module5_git_rev: str | None
    n_input_samples: int = Field(ge=0)
    n_alerts_emitted: int = Field(ge=0)
    n_normal_excluded: int = Field(ge=0)
    filter_applied: Literal["non_normal", "none"]
    inputs: dict[str, InputFile | None]


class AlertResponsesEnvelope(BaseModel):
    """Top-level wrapper Module 5 writes from 2026-05-25 onwards.

    The ``_provenance`` key is aliased (leading underscore is a Python
    convention for "private", but the on-disk JSON uses the underscored
    form so a quick ``jq '._provenance'`` works).
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    provenance: Provenance = Field(alias="_provenance")
    records: list[AlertRecord]
