"""Legacy import paths — guard back-compat for the 9 external consumers."""
from __future__ import annotations


def test_pipeline_module_legacy_symbols():
    from module5_responses.module5_pipeline import (
        AuditLogger,
        FeedbackLoop,
        PolicyEngine,
        clinical_safety_check,
        ActionExecutor,
        NotificationService,
        OUTPUT_DIR,
        RESPONSE_POLICY,
        _HAVE_CRYPTOGRAPHY,
        _canonical_json,
        _load_signing_key,
    )
    # Sanity: each symbol is callable or non-None.
    for sym in (
        AuditLogger, FeedbackLoop, PolicyEngine, clinical_safety_check,
        ActionExecutor, NotificationService, _canonical_json, _load_signing_key,
    ):
        assert sym is not None
    assert OUTPUT_DIR.name == "reports"
    assert "action_catalogue" in RESPONSE_POLICY
    assert isinstance(_HAVE_CRYPTOGRAPHY, bool)


def test_signing_module_public_api():
    from module5_responses.signing import (
        HAVE_CRYPTOGRAPHY,
        canonical_json,
        load_signing_key,
    )
    assert isinstance(HAVE_CRYPTOGRAPHY, bool)
    assert callable(canonical_json)
    assert callable(load_signing_key)


def test_responses_module_legacy_symbols():
    from module5_responses.module5_responses import (
        MITIGATION_ACTIONS,
        DEVICE_TIERS,
        BASE_PROTOCOL,
        ESCALATION_ROUTING,
        select_adaptive_response,
        build_audit_record,
        compute_effectiveness,
        compute_response_stats,
        build_all_records,
        _assert_no_score_drift,
        _build_provenance,
        _paths,
    )
    assert "log_event" in MITIGATION_ACTIONS
    assert "vital_monitoring" in DEVICE_TIERS
    assert "CRITICAL" in BASE_PROTOCOL
    assert "Spoofing" in ESCALATION_ROUTING
    for fn in (
        select_adaptive_response, build_audit_record, compute_effectiveness,
        compute_response_stats, build_all_records, _assert_no_score_drift,
        _build_provenance, _paths,
    ):
        assert callable(fn)


def test_top_level_init_public_api():
    import module5_responses as m5
    expected_subset = {
        "AuditLogger", "PolicyEngine", "FeedbackLoop", "ActionExecutor",
        "NotificationService", "select_adaptive_response", "build_audit_record",
        "compute_effectiveness", "build_all_records",
        "ACTION_CATALOGUE", "DEVICE_TIERS", "TIER_POLICIES", "ATTACK_ROUTING",
        "HAVE_CRYPTOGRAPHY", "canonical_json", "load_signing_key",
    }
    assert expected_subset.issubset(set(m5.__all__))
