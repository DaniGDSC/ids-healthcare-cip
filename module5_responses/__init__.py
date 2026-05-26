"""Module 5 — Closed-Loop Response Engine (RQ3/RO3).

Public API surface — re-exports from the decomposed sub-modules. Both CLI
entry points (``module5_pipeline.py`` for worked-examples + audit
management, ``module5_responses.py`` for the real-split batch) carry
their own back-compat re-exports too.
"""
from __future__ import annotations

from .adaptive import build_audit_record, select_adaptive_response
from .audit.logger import (
    ARCHIVE_DIR,
    AuditLogger,
    DEFAULT_RETENTION_DAYS,
    OUTPUT_DIR,
)
from .audit.signing import (
    DEFAULT_PRIVATE_KEY_PATH,
    DEFAULT_PUBLIC_KEY_PATH,
    SIGNATURE_ALG,
)
from .config import (
    ACTION_CATALOGUE,
    ACUITY_OVERRIDES,
    ATTACK_ROUTING,
    DEFAULT_DEVICE_TIER,
    DEFAULT_ROUTING,
    DEVICE_TIERS,
    MVE_LLM_FAIL_STREAK_MAX,
    RESPONSE_POLICY_VERSION,
    TIER_POLICIES,
)
from .effectiveness import compute_effectiveness, compute_response_stats
from .executor import ActionExecutor, NotificationService
from .feedback import FeedbackLoop
from .loaders import (
    _paths,
    load_attack_categories,
    load_explanations,
    load_risk_scores,
)
from .pipeline import (
    _assert_no_score_drift,
    _build_provenance,
    build_all_records,
    run_one_split,
)
from .policy import (
    PolicyEngine,
    clinical_safety_check,
    export_response_policy,
)
from .signing import HAVE_CRYPTOGRAPHY, canonical_json, load_signing_key
from .worked_examples import run_worked_examples

__all__ = [
    # config
    "ACTION_CATALOGUE", "DEVICE_TIERS", "TIER_POLICIES", "ATTACK_ROUTING",
    "ACUITY_OVERRIDES", "DEFAULT_DEVICE_TIER", "DEFAULT_ROUTING",
    "MVE_LLM_FAIL_STREAK_MAX", "RESPONSE_POLICY_VERSION",
    # policy + executor + feedback
    "PolicyEngine", "clinical_safety_check", "export_response_policy",
    "ActionExecutor", "NotificationService", "FeedbackLoop",
    # audit primitives
    "AuditLogger", "ARCHIVE_DIR", "OUTPUT_DIR", "DEFAULT_RETENTION_DAYS",
    "DEFAULT_PRIVATE_KEY_PATH", "DEFAULT_PUBLIC_KEY_PATH", "SIGNATURE_ALG",
    "HAVE_CRYPTOGRAPHY", "canonical_json", "load_signing_key",
    # batch pipeline
    "load_risk_scores", "load_explanations", "load_attack_categories", "_paths",
    "select_adaptive_response", "build_audit_record",
    "compute_effectiveness", "compute_response_stats",
    "build_all_records", "_build_provenance", "_assert_no_score_drift",
    "run_one_split",
    # worked examples
    "run_worked_examples",
]
