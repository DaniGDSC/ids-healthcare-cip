#!/usr/bin/env python3
"""Module 5 — Response Pipeline Integration (Tasks 5.1–5.8).

Wraps the closed-loop response engine (generate_responses.py) into
the class-based structure required by the thesis spec:
  5.1  Export standalone response_policy.json
  5.2  PolicyEngine class
  5.3  Clinical safety override with confirmation request
  5.4  Simulated ActionExecutor with audit trail
  5.5  NotificationService per stakeholder
  5.6  Immutable audit logger (JSONL)
  5.7  End-to-end worked examples (CRITICAL/HIGH/LOW)
  5.8  Feedback loop stub

Usage:
    python run_response_pipeline.py
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Ensure the project root is importable when this script is invoked
# directly (e.g. via run_all_modules.py); the absolute imports below
# (``from common.phi import ...``) need it.
_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd


# ── Per-stakeholder views (closes GAP-A2) ────────────────────────────────

def render_views_for_alert(
    mve,
    alert_type: str = "T1",
    *,
    shared_anchor=None,
) -> dict:
    """Return all three role-scoped MVEOutput views for one alert.

    Wraps src.mve_generator.derive_role_view so Module 6 (dashboard) can
    fetch every authorised view in one call. INVARIANT 6 (cross-role
    consistency on layer_2), INVARIANT 7 (DO NOT preserved), and
    INVARIANT 9 (shared anchor — alert_id / risk_tier / device_id /
    one_line_summary / timestamp identical across role views) are
    enforced here.

    Args:
        mve: Default MVEOutput from src.mve_generator.generate_mve.
        alert_type: T1..T5 for ATT&CK grounding (passed through today).
        shared_anchor: Optional :class:`src.data_models.SharedAnchor`
            instance. When provided, its serialised dict is attached
            verbatim under each role view's ``"shared_anchor"`` key —
            byte-identical across all three roles. The anchor block is
            built once at view-render time so phone-based incident
            handling can rely on every operator seeing the same header.

    Returns:
        Mapping of role → ``{"view": MVEOutput, "shared_anchor": dict}``
        when ``shared_anchor`` is provided; otherwise ``{role: MVEOutput}``
        (back-compat for existing callers).
    """
    from src.mve_generator import derive_role_view
    from src.data_models import OperatorRole

    views = {
        role.value: derive_role_view(mve, role.value, alert_type=alert_type)
        for role in OperatorRole
    }

    if shared_anchor is None:
        return views

    # INVARIANT 9: every role view carries the SAME serialised anchor
    # dict. We compute the dict once, then attach the same reference to
    # each view so a regression that mutates one anchor would mutate
    # all three (caught by ``test_step13_cross_role_consistency.py``).
    anchor_dict = (
        shared_anchor.to_dict()
        if hasattr(shared_anchor, "to_dict")
        else dict(shared_anchor)
    )
    return {
        role: {"view": view, "shared_anchor": anchor_dict}
        for role, view in views.items()
    }

# `cryptography` is used for ECDSA P-256 signing of audit records.
# Imported lazily so that the rest of the module is still importable in
# environments where the package is not installed (e.g. unit tests that
# only exercise the policy engine). The AuditLogger constructor will
# fail loudly if it can't load the library when signing is required.
try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    _HAVE_CRYPTOGRAPHY = True
except ImportError:  # pragma: no cover
    _HAVE_CRYPTOGRAPHY = False

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results/reports"
ARCHIVE_DIR = OUTPUT_DIR / "audit_archive"

# ─────────────────────────────────────────────────────────────────────────
# Audit Log Retention Policy
# ─────────────────────────────────────────────────────────────────────────
# Default retention: 365 days. Override per deployment via the
# IOMT_AUDIT_RETENTION_DAYS environment variable or the constructor
# argument `retention_days`.
#
# Jurisdictional minima (informational — pick the strictest that applies):
#   HIPAA §164.530(j)        : 6 years from creation OR last effective date
#   FDA 21 CFR Part 11       : as long as the predicate record is required
#   EU AI Act Annex IV §4    : 6 months minimum for high-risk AI logs
#                              unless EU/national law specifies longer
#   GDPR Art. 5(1)(e)        : data minimization — no fixed number;
#                              hospital DPO determines proportionality
#   Vietnam Decree 13/2023   : duration set in DPIA, registered with MPS
#
# This default does NOT make the system compliant with any of the above
# by itself. It is a sensible technical default that the deployment
# owner must verify against their jurisdiction.
DEFAULT_RETENTION_DAYS = 365

# Default key locations. The private key is auto-bootstrapped on first
# run if no operator-provided key is available via IOMT_AUDIT_SIGNING_KEY.
# The public key is written next to the audit log so verifiers find it
# without configuration.
DEFAULT_PRIVATE_KEY_PATH = Path.home() / ".iomt-ids" / "audit_signing_key.pem"
DEFAULT_PUBLIC_KEY_PATH = OUTPUT_DIR / "audit_signing_key.pub.pem"

SIGNATURE_ALG = "ECDSA_P256_SHA256"

from common.phi import BIOMETRIC_COLUMNS as BIOMETRIC_FEATURES  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════
# 5.1  Response Policy Config
# ═══════════════════════════════════════════════════════════════════════

RESPONSE_POLICY = {
    "version": "1.0",
    "description": "Maps (alert_tier, device_tier, patient_acuity_level) to response action sets",
    "action_catalogue": {
        "log_event":           {"cost": 0.1, "reversible": True,  "requires_approval": False},
        "enhanced_monitoring": {"cost": 0.2, "reversible": True,  "requires_approval": False},
        "re_authenticate":     {"cost": 0.3, "reversible": True,  "requires_approval": False},
        "restrict_traffic":    {"cost": 0.5, "reversible": True,  "requires_approval": False},
        "isolate_device":      {"cost": 0.8, "reversible": True,  "requires_approval": True},
        "forensic_snapshot":   {"cost": 0.4, "reversible": True,  "requires_approval": False},
        "escalate_clinical":   {"cost": 0.7, "reversible": False, "requires_approval": False},
        "escalate_incident":   {"cost": 1.0, "reversible": False, "requires_approval": False},
    },
    "tier_policies": {
        "CRITICAL": {
            "default_actions": ["log_event", "isolate_device", "forensic_snapshot", "escalate_incident", "escalate_clinical"],
            "max_response_min": 5,
            "recommended_for_auto_execution": True,
        },
        "HIGH": {
            "default_actions": ["log_event", "isolate_device", "forensic_snapshot", "enhanced_monitoring"],
            "max_response_min": 15,
            "recommended_for_auto_execution": True,
        },
        "MEDIUM": {
            "default_actions": ["log_event", "restrict_traffic", "enhanced_monitoring"],
            "max_response_min": 60,
            "recommended_for_auto_execution": False,
        },
        "LOW": {
            "default_actions": ["log_event", "enhanced_monitoring"],
            "max_response_min": 480,
            "recommended_for_auto_execution": False,
        },
    },
    "device_constraints": {
        "life_sustaining":  {"max_action_cost": 0.5, "isolation_blocked": True,  "clinical_approval_required": True},
        "vital_monitoring": {"max_action_cost": 0.8, "isolation_blocked": False, "clinical_approval_required": True},
        "diagnostic":       {"max_action_cost": 0.8, "isolation_blocked": False, "clinical_approval_required": False},
        "auxiliary":        {"max_action_cost": 0.8, "isolation_blocked": False, "clinical_approval_required": False},
    },
    "acuity_overrides": {
        "elevated_acuity_threshold": 0.25,
        "action_on_elevated": "Add escalate_clinical if not present; require clinical confirmation before isolation",
    },
    "attack_routing": {
        "Spoofing":        {"add_actions": ["re_authenticate"], "primary_notify": "IT Security", "secondary_notify": "Biomedical Engineering"},
        "Data Alteration": {"add_actions": ["forensic_snapshot", "escalate_clinical"], "primary_notify": "IT Security", "secondary_notify": "Charge Nurse"},
    },
}


def export_response_policy() -> None:
    """Task 5.1: Export standalone response policy config."""
    path = OUTPUT_DIR / "response_policy.json"
    path.write_text(json.dumps(RESPONSE_POLICY, indent=2), encoding="utf-8")
    logger.info("5.1 Saved: response_policy.json")


# ═══════════════════════════════════════════════════════════════════════
# 5.2  PolicyEngine
# ═══════════════════════════════════════════════════════════════════════

class PolicyEngine:
    """Rule-based engine: reads policy config, returns recommended actions."""

    def __init__(self, policy: dict = RESPONSE_POLICY):
        self.policy = policy
        self.catalogue = policy["action_catalogue"]

    def recommend(
        self,
        alert_tier: str,
        device_tier: str = "vital_monitoring",
        attack_category: str = "unknown",
        patient_acuity: float = 0.0,
    ) -> dict:
        tier_policy = self.policy["tier_policies"].get(alert_tier, self.policy["tier_policies"]["LOW"])
        actions = list(tier_policy["default_actions"])

        # Attack-specific actions
        routing = self.policy["attack_routing"].get(attack_category, {})
        for a in routing.get("add_actions", []):
            if a not in actions:
                actions.append(a)

        # Device constraints
        constraint = self.policy["device_constraints"].get(device_tier, {})
        max_cost = constraint.get("max_action_cost", 1.0)
        if constraint.get("isolation_blocked") and "isolate_device" in actions:
            actions.remove("isolate_device")
            if "restrict_traffic" not in actions:
                actions.append("restrict_traffic")

        actions = [a for a in actions if self.catalogue.get(a, {}).get("cost", 0) <= max_cost
                   or a in ("log_event", "escalate_clinical")]

        # Clinical safety override (Task 5.3)
        override = clinical_safety_check(
            alert_tier, device_tier, patient_acuity, actions,
        )

        # Sort by cost
        actions = sorted(set(actions), key=lambda a: self.catalogue.get(a, {}).get("cost", 0))

        return {
            "actions": actions,
            "max_response_min": tier_policy["max_response_min"],
            "recommended_for_auto_execution": tier_policy["recommended_for_auto_execution"],
            "primary_notify": routing.get("primary_notify", "IT Security"),
            "secondary_notify": routing.get("secondary_notify"),
            "clinical_override": override,
            "requires_approval": any(
                self.catalogue.get(a, {}).get("requires_approval", False) for a in actions
            ),
        }

    def recommend_structured(
        self,
        alert_tier: str,
        device_tier: str = "vital_monitoring",
        attack_category: str = "unknown",
        patient_acuity: float = 0.0,
    ):
        """Structured ARCHITECTURE.md Step [15] recommendation.

        Wraps :meth:`recommend` and projects its dict into the
        :class:`src.data_models.ResponseRecommendation` dataclass —
        the canonical action contract per the doc. Lets new callers
        (M6 dashboard, audit log) use the doc-shaped object while
        legacy consumers keep their dict.
        """
        from src.data_models import ResponseRecommendation

        legacy = self.recommend(
            alert_tier=alert_tier,
            device_tier=device_tier,
            attack_category=attack_category,
            patient_acuity=patient_acuity,
        )
        actions = legacy["actions"] or ["log_event"]
        primary_code = actions[0]
        catalogue_entry = self.catalogue.get(primary_code, {})
        primary_human = catalogue_entry.get("name") or primary_code.replace("_", " ").title()

        # Estimated clinical impact: cheap actions are minimal; isolation
        # of life-sustaining devices is high; otherwise moderate.
        cost = float(catalogue_entry.get("cost", 0.0))
        if primary_code == "isolate_device" and device_tier in ("life_sustaining", "vital_monitoring"):
            impact = "high"
        elif cost >= 0.6:
            impact = "moderate"
        else:
            impact = "minimal"

        # Suggested priority: 1 (CRITICAL) → 5 (LOW); fall back to MEDIUM.
        priority_map = {"CRITICAL": 1, "HIGH": 2, "MEDIUM": 3, "LOW": 4}
        priority = priority_map.get(alert_tier.upper(), 3)

        # do_not_actions: clinical override (downgrade isolation on
        # critical devices) explicitly forbids isolation; otherwise
        # carry the device-tier constraint forward.
        do_not = []
        clinical_override = legacy.get("clinical_override", {})
        if clinical_override.get("triggered"):
            do_not.append("isolate_device")
        if device_tier in ("life_sustaining", "vital_monitoring"):
            do_not.append("power_cycle_device")

        rationale = (
            f"{alert_tier} {attack_category} on {device_tier} → "
            f"{primary_human} (max response: {legacy['max_response_min']} min)"
        )

        return ResponseRecommendation(
            primary_action=primary_human,
            primary_action_code=primary_code,
            rationale=rationale,
            estimated_clinical_impact=impact,
            operator_decision_required=True,   # INVARIANT 3 — always
            suggested_priority=priority,
            do_not_actions=sorted(set(do_not)),
        )


# ═══════════════════════════════════════════════════════════════════════
# 5.3  Clinical Safety Override
# ═══════════════════════════════════════════════════════════════════════

def clinical_safety_check(
    alert_tier: str,
    device_tier: str,
    patient_acuity: float,
    actions: list,
) -> dict:
    """Check if device is safety-critical AND patient acuity elevated → override."""
    override = {
        "triggered": False,
        "reason": None,
        "original_actions": list(actions),
        "clinical_confirmation_required": False,
    }

    is_critical_device = device_tier in ("life_sustaining", "vital_monitoring")
    acuity_elevated = patient_acuity >= RESPONSE_POLICY["acuity_overrides"]["elevated_acuity_threshold"]

    if is_critical_device and acuity_elevated:
        override["triggered"] = True
        override["clinical_confirmation_required"] = True

        if "isolate_device" in actions:
            override["reason"] = (
                f"Device tier '{device_tier}' with elevated patient acuity "
                f"({patient_acuity:.2f}) — isolation downgraded to restrict_traffic. "
                "Clinical confirmation required before any disruptive action."
            )
            actions.remove("isolate_device")
            if "restrict_traffic" not in actions:
                actions.append("restrict_traffic")
        else:
            override["reason"] = (
                f"Device tier '{device_tier}' with elevated acuity ({patient_acuity:.2f}) — "
                "clinical confirmation required."
            )

        if "escalate_clinical" not in actions:
            actions.append("escalate_clinical")

    return override


# ═══════════════════════════════════════════════════════════════════════
# 5.4  ActionExecutor (simulated)
# ═══════════════════════════════════════════════════════════════════════

class ActionExecutor:
    """Simulated executor: logs actions to audit trail instead of real changes."""

    def __init__(self):
        self.execution_log = []

    def execute(
        self,
        alert_id: str,
        sample_index: int,
        actions: list,
        recommendation: dict,
        ground_truth: str,
        timestamp: datetime,
    ) -> dict:
        has_mitigation = any(a in actions for a in
                            ("isolate_device", "restrict_traffic", "re_authenticate"))
        is_attack = ground_truth == "attack"

        if is_attack and has_mitigation:
            outcome = "threat_contained"
            effective = True
        elif is_attack and not has_mitigation:
            outcome = "threat_logged_not_mitigated"
            effective = False
        elif not is_attack and has_mitigation:
            outcome = "false_positive_isolated"
            effective = False
        else:
            outcome = "benign_logged"
            effective = True

        record = {
            "alert_id": alert_id,
            "sample_index": sample_index,
            "timestamp": timestamp.isoformat(),
            "actions_executed": actions,
            "recommended_for_auto_execution": recommendation.get("recommended_for_auto_execution", False),
            "clinical_override": recommendation.get("clinical_override", {}).get("triggered", False),
            "requires_approval": recommendation.get("requires_approval", False),
            "outcome": outcome,
            "effective": effective,
            "ground_truth": ground_truth,
        }
        self.execution_log.append(record)
        return record


# ═══════════════════════════════════════════════════════════════════════
# 5.5  NotificationService
# ═══════════════════════════════════════════════════════════════════════

class NotificationService:
    """Generate structured alert messages per stakeholder."""

    def __init__(self):
        self.notifications = []

    def notify(
        self,
        sample_index: int,
        alert_tier: str,
        recommendation: dict,
        clinician_summary: str,
        analyst_top_features: list,
        risk_score: float,
    ) -> list:
        msgs = []

        # Security analyst notification
        msgs.append({
            "recipient": recommendation["primary_notify"],
            "channel": "SIEM + Dashboard",
            "priority": alert_tier,
            "message": (
                f"[{alert_tier}] Alert #{sample_index}: "
                f"Risk={risk_score:.2f}. Actions: {', '.join(recommendation['actions'])}. "
                f"Top features: {', '.join(f['feature'] for f in analyst_top_features[:3])}."
            ),
        })

        # Clinical notification (if escalation required)
        if "escalate_clinical" in recommendation["actions"]:
            msgs.append({
                "recipient": "Clinical Staff",
                "channel": "Page / Dashboard",
                "priority": alert_tier,
                "message": clinician_summary[:300] if clinician_summary else "Clinical review requested.",
            })

        # Secondary notification
        if recommendation.get("secondary_notify"):
            msgs.append({
                "recipient": recommendation["secondary_notify"],
                "channel": "Email / Ticket",
                "priority": alert_tier,
                "message": f"[{alert_tier}] Sample #{sample_index}: {', '.join(recommendation['actions'])}",
            })

        self.notifications.extend(msgs)
        return msgs


# ═══════════════════════════════════════════════════════════════════════
# 5.6  Audit Logger (JSONL) — hash-chained, ECDSA-signed, rotatable
# ═══════════════════════════════════════════════════════════════════════

# ── Key management helpers ──────────────────────────────────────────────

def _require_cryptography() -> None:
    if not _HAVE_CRYPTOGRAPHY:
        raise RuntimeError(
            "audit log signing requires the `cryptography` package. "
            "Install it with `pip install cryptography>=42.0`."
        )


def _bootstrap_local_key(private_path: Path, public_path: Path) -> None:
    """Generate a fresh ECDSA P-256 keypair on first run.

    Private key is written with 0600 permissions to a user-local directory
    (default: ~/.iomt-ids). The public key is written next to the audit
    log so verifiers find it without extra configuration.

    SECURITY WARNING: an auto-generated key is convenient for development
    but offers no protection against an attacker who already has shell
    access on the host. For production, set IOMT_AUDIT_SIGNING_KEY to a
    key issued by your operator (HSM, KMS, or operator-provisioned PEM)
    so the private key never lives next to the data it signs.
    """
    _require_cryptography()
    private_path.parent.mkdir(parents=True, exist_ok=True)
    public_path.parent.mkdir(parents=True, exist_ok=True)

    private_key = ec.generate_private_key(ec.SECP256R1())
    pem_priv = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    pem_pub = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    private_path.write_bytes(pem_priv)
    try:
        os.chmod(private_path, 0o600)
    except OSError:
        # Best effort on platforms without POSIX perms
        pass
    public_path.write_bytes(pem_pub)
    logger.warning(
        "AuditLogger: bootstrapped a local ECDSA P-256 signing key at %s. "
        "Replace with an operator-provisioned key for production.",
        private_path,
    )


def _load_signing_key(
    private_path: Path | None = None,
    public_path: Path | None = None,
):
    """Load (or bootstrap) the ECDSA private key used to sign records.

    Resolution order:
      1. IOMT_AUDIT_SIGNING_KEY environment variable (operator override)
      2. Explicit `private_path` argument
      3. DEFAULT_PRIVATE_KEY_PATH (~/.iomt-ids/audit_signing_key.pem)
      4. Bootstrap a fresh key at the default path
    """
    _require_cryptography()
    env_path = os.environ.get("IOMT_AUDIT_SIGNING_KEY")
    if env_path:
        private_path = Path(env_path)
    elif private_path is None:
        private_path = DEFAULT_PRIVATE_KEY_PATH

    public_path = public_path or DEFAULT_PUBLIC_KEY_PATH

    if not private_path.exists():
        _bootstrap_local_key(private_path, public_path)

    private_key = serialization.load_pem_private_key(
        private_path.read_bytes(), password=None
    )

    # Always (re)export the matching public key next to the audit log so
    # verification works without operator intervention. Idempotent.
    pem_pub = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    public_path.parent.mkdir(parents=True, exist_ok=True)
    if not public_path.exists() or public_path.read_bytes() != pem_pub:
        public_path.write_bytes(pem_pub)

    key_id = "ecdsa-p256-" + hashlib.sha256(pem_pub).hexdigest()[:16]
    return private_key, public_path, key_id


def _canonical_json(record: dict) -> bytes:
    """Deterministic JSON encoding used for hashing and signing."""
    return json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")


# ── AuditLogger ─────────────────────────────────────────────────────────

class AuditLogger:
    """Hash-chained, ECDSA-signed append-only JSONL audit log.

    Each record carries:
      - `prev_hash`        : sha256 of the previous record (hash chain)
      - `integrity_hash`   : sha256 of the current record (covers prev_hash)
      - `signature`        : ECDSA P-256 signature over the canonical JSON
                             of the record (covers integrity_hash and
                             everything below it)
      - `signing_key_id`   : stable id derived from the public key
      - `signature_alg`    : "ECDSA_P256_SHA256"

    Optional reviewer attribution: when callers pass `reviewer_id` /
    `reviewer_role` to `log()`, a `reviewer` block is added to the record
    *before* signing, so reviewer attribution is bound to the signature.

    Restart safety: if the target file already exists, the constructor
    walks the last record and continues the chain from its
    `integrity_hash`, so multiple invocations of the same pipeline do
    not produce a fake chain break.

    Retention: `rotate_and_purge(days)` archives the active log into a
    sealed file under `audit_archive/`, then starts a new active log
    whose first `prev_hash` points back at the last archived record so
    the cross-rotation chain remains walkable for forensics.
    """

    def __init__(
        self,
        path: Path,
        *,
        signing_key_path: Path | None = None,
        public_key_path: Path | None = None,
        retention_days: int | None = None,
        sign: bool = True,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.sign_enabled = sign and _HAVE_CRYPTOGRAPHY
        if sign and not _HAVE_CRYPTOGRAPHY:
            logger.warning(
                "AuditLogger: cryptography not installed; signing disabled. "
                "Records will be hash-chained only."
            )

        if self.sign_enabled:
            self._private_key, self.public_key_path, self.signing_key_id = (
                _load_signing_key(signing_key_path, public_key_path)
            )
        else:
            self._private_key = None
            self.public_key_path = public_key_path or DEFAULT_PUBLIC_KEY_PATH
            self.signing_key_id = "unsigned"

        # Retention policy: env var > constructor arg > default
        env_days = os.environ.get("IOMT_AUDIT_RETENTION_DAYS")
        if retention_days is not None:
            self.retention_days = int(retention_days)
        elif env_days:
            self.retention_days = int(env_days)
        else:
            self.retention_days = DEFAULT_RETENTION_DAYS

        # Recover chain from existing file (genesis-on-restart fix)
        self.prev_hash = self._recover_prev_hash()

    # ── chain recovery ─────────────────────────────────────────────

    def _recover_prev_hash(self) -> str:
        """Read the last record's integrity_hash to continue the chain.

        M5-6: reads only the last 4 KB of the file instead of streaming
        the entire JSONL from the beginning — O(1) disk I/O regardless of
        log size.
        """
        if not self.path.exists() or self.path.stat().st_size == 0:
            return "0" * 64
        try:
            with open(self.path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 4096))
                tail = f.read()
        except OSError:
            return "0" * 64

        lines = [ln for ln in tail.split(b"\n") if ln.strip()]
        if not lines:
            return "0" * 64
        last_line = lines[-1].decode("utf-8", errors="ignore").strip()
        try:
            last_record = json.loads(last_line)
            recovered = last_record.get("integrity_hash")
            if isinstance(recovered, str) and len(recovered) == 64:
                return recovered
        except json.JSONDecodeError:
            logger.warning(
                "AuditLogger: tail of %s is unparseable; starting new "
                "chain at genesis.",
                self.path,
            )
        return "0" * 64

    # ── log ────────────────────────────────────────────────────────

    def log(
        self,
        record: dict,
        *,
        reviewer_id: str | None = None,
        reviewer_role: str | None = None,
        review_timestamp: str | None = None,
        review_action: str | None = None,
        mve_audit: dict | None = None,
    ) -> dict:
        """Append a hash-chained, signed record to the audit log.

        Args:
            record: arbitrary JSON-serializable event payload.
            reviewer_id: optional human reviewer identifier (e.g. P03).
            reviewer_role: optional role (Security Analyst / Clinician
                / Administrator).
            review_timestamp: ISO-8601 timestamp; defaults to now() in
                UTC if any other reviewer field is provided.
            review_action: optional free-text action label
                (confirm / reject / acknowledge / ...).
            mve_audit: optional ARCHITECTURE.md Step [16] explanation
                context — a dict with keys ``mve_mode`` (``A_llm`` /
                ``B_rule``), ``mve_text_shown``, ``shap_top_features``,
                ``shap_stability``, and for Mode A: ``llm_provider``,
                ``llm_model_version``, ``llm_full_prompt``,
                ``llm_full_response``. Stored verbatim under
                ``record["mve_audit"]`` so the operator decision is
                replayable.

        Returns:
            The record as it was written (with all envelope fields).
        """
        record = dict(record)  # do not mutate caller's dict

        # Reviewer block (only present when at least one field supplied)
        if any(x is not None for x in (reviewer_id, reviewer_role, review_action)):
            if review_timestamp is None:
                review_timestamp = datetime.now(timezone.utc).isoformat()
            record["reviewer"] = {
                "reviewer_id": reviewer_id,
                "reviewer_role": reviewer_role,
                "review_timestamp": review_timestamp,
                "review_action": review_action,
            }

        # ARCHITECTURE.md Step [16] explanation context (Mode A LLM
        # reproducibility): persist the full prompt + response +
        # model_version on the audit record so any LLM-generated MVE
        # is replayable post-hoc.
        if mve_audit is not None:
            record["mve_audit"] = dict(mve_audit)

        # ARCHITECTURE.md Step [16] forward-compatibility placeholders.
        # Step [17] (outcome tracking) and Step [18] (continuous
        # improvement) are post-defense work; we reserve the schema
        # slots NOW so the chain doesn't have to be retroactively
        # extended when those phases land.
        record.setdefault("ground_truth_label", None)
        record.setdefault("decision_quality", None)
        record.setdefault("feedback_loop_consumed", False)

        # 1. Chain
        record["prev_hash"] = self.prev_hash

        # 2. Integrity hash over (record + prev_hash)
        record["integrity_hash"] = hashlib.sha256(_canonical_json(record)).hexdigest()

        # 3. Signature over the record including integrity_hash
        if self.sign_enabled:
            sig_payload = _canonical_json(record)
            signature_der = self._private_key.sign(
                sig_payload, ec.ECDSA(hashes.SHA256())
            )
            record["signature"] = base64.b64encode(signature_der).decode("ascii")
            record["signing_key_id"] = self.signing_key_id
            record["signature_alg"] = SIGNATURE_ALG

        # 4. Advance chain and persist
        self.prev_hash = record["integrity_hash"]
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return record

    # ── verification ───────────────────────────────────────────────

    @classmethod
    def verify(
        cls,
        path: Path,
        public_key_path: Path | None = None,
        *,
        legacy_ok: bool = True,
    ) -> dict:
        """Walk an audit log and verify hash chain + signatures.

        Args:
            path: path to the audit log JSONL file.
            public_key_path: PEM file containing the ECDSA P-256 public
                key. Defaults to DEFAULT_PUBLIC_KEY_PATH.
            legacy_ok: if True, records without a `signature` field are
                accepted as `legacy` (chain still verified). If False,
                they are reported as `unsigned`. The migration default
                is True; flip to False after a clean rotation.

        Returns:
            Dict with totals, the line number of the first break (if
            any), and a list of per-broken-line reasons.
        """
        path = Path(path)
        public_key_path = Path(public_key_path or DEFAULT_PUBLIC_KEY_PATH)

        result: dict = {
            "path": str(path),
            "public_key": str(public_key_path),
            "total": 0,
            "valid_signed": 0,
            "valid_legacy": 0,
            "broken": [],
            "first_break_at": None,
        }

        if not path.exists():
            result["broken"].append({"line": 0, "reason": "file does not exist"})
            return result

        public_key = None
        if _HAVE_CRYPTOGRAPHY and public_key_path.exists():
            try:
                public_key = serialization.load_pem_public_key(
                    public_key_path.read_bytes()
                )
            except Exception as exc:  # noqa: BLE001
                result["broken"].append(
                    {"line": 0, "reason": f"failed to load public key: {exc}"}
                )
                return result

        prev_hash_expected = "0" * 64
        result["legacy_chain_restarts"] = 0
        with open(path, "r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue
                result["total"] += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    cls._mark_break(result, line_no, f"json parse: {exc}")
                    return result

                is_unsigned = "signature" not in record

                # 1. chain
                if record.get("prev_hash") != prev_hash_expected:
                    # Legacy migration: the pre-hardening AuditLogger
                    # reset the chain to genesis on every process start.
                    # In legacy mode, accept a fresh genesis block as a
                    # known-good restart marker rather than tampering.
                    if (
                        legacy_ok
                        and is_unsigned
                        and record.get("prev_hash") == "0" * 64
                        and line_no > 1
                    ):
                        result["legacy_chain_restarts"] += 1
                        prev_hash_expected = "0" * 64
                    else:
                        cls._mark_break(
                            result,
                            line_no,
                            f"prev_hash mismatch (expected "
                            f"{prev_hash_expected[:12]}..., got "
                            f"{str(record.get('prev_hash'))[:12]}...)",
                        )
                        return result

                # 2. integrity hash
                signature_b64 = record.pop("signature", None)
                signing_key_id = record.pop("signing_key_id", None)
                signature_alg = record.pop("signature_alg", None)
                stored_integrity = record.get("integrity_hash")
                # Recompute integrity hash from record minus integrity_hash
                without_hash = {k: v for k, v in record.items() if k != "integrity_hash"}
                computed = hashlib.sha256(_canonical_json(without_hash)).hexdigest()
                # Legacy records (no signature) used the default-separator
                # form of json.dumps; accept that variant during the
                # migration window.
                if computed != stored_integrity and signature_b64 is None:
                    legacy_payload = json.dumps(
                        without_hash, sort_keys=True
                    ).encode("utf-8")
                    legacy_hash = hashlib.sha256(legacy_payload).hexdigest()
                    if legacy_hash == stored_integrity:
                        computed = stored_integrity
                if computed != stored_integrity:
                    cls._mark_break(
                        result,
                        line_no,
                        "integrity_hash mismatch (record body tampered)",
                    )
                    return result

                # 3. signature (if present)
                if signature_b64 is None:
                    if legacy_ok:
                        result["valid_legacy"] += 1
                        prev_hash_expected = stored_integrity
                        continue
                    cls._mark_break(result, line_no, "record is unsigned")
                    return result

                if public_key is None:
                    cls._mark_break(
                        result,
                        line_no,
                        "signature present but no public key available",
                    )
                    return result

                try:
                    # Re-add the integrity_hash to the dict so the payload
                    # we verify matches what was signed.
                    sig_record = dict(record)
                    sig_payload = _canonical_json(sig_record)
                    public_key.verify(
                        base64.b64decode(signature_b64),
                        sig_payload,
                        ec.ECDSA(hashes.SHA256()),
                    )
                    result["valid_signed"] += 1
                except InvalidSignature:
                    cls._mark_break(result, line_no, "invalid signature")
                    return result
                except Exception as exc:  # noqa: BLE001
                    cls._mark_break(result, line_no, f"signature verify error: {exc}")
                    return result

                prev_hash_expected = stored_integrity

        return result

    @staticmethod
    def _mark_break(result: dict, line_no: int, reason: str) -> None:
        result["broken"].append({"line": line_no, "reason": reason})
        if result["first_break_at"] is None:
            result["first_break_at"] = line_no

    # ── rotation + purge ───────────────────────────────────────────

    def rotate_and_purge(
        self,
        retention_days: int | None = None,
        archive_dir: Path | None = None,
    ) -> dict:
        """Archive the current active log if it contains records older
        than the retention cutoff, then start a new active log whose
        first chain link points back at the archived log's tail.

        Returns a dict describing what happened.
        """
        days = retention_days if retention_days is not None else self.retention_days
        archive_dir = Path(archive_dir or ARCHIVE_DIR)
        archive_dir.mkdir(parents=True, exist_ok=True)

        report: dict = {
            "rotated": False,
            "reason": None,
            "archived_path": None,
            "manifest_path": None,
            "retention_days": days,
            "verify_before_rotate": None,
        }

        if not self.path.exists() or self.path.stat().st_size == 0:
            report["reason"] = "active log empty or missing"
            return report

        # Step 1: verify before any destructive action
        verify_report = self.verify(self.path, self.public_key_path, legacy_ok=True)
        report["verify_before_rotate"] = {
            "total": verify_report["total"],
            "valid_signed": verify_report["valid_signed"],
            "valid_legacy": verify_report["valid_legacy"],
            "first_break_at": verify_report["first_break_at"],
        }
        if verify_report["first_break_at"] is not None:
            report["reason"] = (
                f"refusing to rotate a tampered log (first break at "
                f"line {verify_report['first_break_at']})"
            )
            # Emit a SECURITY_INCIDENT marker into the active log so the
            # event of refusing to rotate is itself audited.
            self.log(
                {
                    "event_type": "SECURITY_INCIDENT",
                    "subtype": "rotate_refused_chain_broken",
                    "first_break_at": verify_report["first_break_at"],
                    "broken_count": len(verify_report["broken"]),
                }
            )
            return report

        # Step 2: read first/last records to compute the time window
        first_record = None
        last_record = None
        with open(self.path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if first_record is None:
                    first_record = rec
                last_record = rec

        if first_record is None or last_record is None:
            report["reason"] = "active log has no parseable records"
            return report

        # Try to find a usable timestamp on each record. We support a
        # couple of common shapes; falls back to mtime if neither exists.
        def _record_ts(rec: dict) -> datetime:
            for key in ("timestamp", "review_timestamp"):
                v = rec.get(key)
                if isinstance(v, str):
                    try:
                        return datetime.fromisoformat(v.replace("Z", "+00:00"))
                    except ValueError:
                        pass
            r = rec.get("reviewer", {})
            if isinstance(r, dict):
                v = r.get("review_timestamp")
                if isinstance(v, str):
                    try:
                        return datetime.fromisoformat(v.replace("Z", "+00:00"))
                    except ValueError:
                        pass
            return datetime.fromtimestamp(self.path.stat().st_mtime, tz=timezone.utc)

        first_ts = _record_ts(first_record)
        last_ts = _record_ts(last_record)
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        # Make naive timestamps tz-aware (assume UTC) so the comparison
        # below doesn't blow up on legacy records.
        if first_ts.tzinfo is None:
            first_ts = first_ts.replace(tzinfo=timezone.utc)
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=timezone.utc)

        if first_ts >= cutoff:
            report["reason"] = (
                f"oldest record ({first_ts.isoformat()}) is within "
                f"the {days}-day retention window; nothing to rotate"
            )
            return report

        # Step 3: archive
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        archived_path = archive_dir / f"{self.path.stem}.{stamp}.jsonl"
        shutil.move(str(self.path), str(archived_path))

        manifest = {
            "archived_path": str(archived_path),
            "first_record_ts": first_ts.isoformat(),
            "last_record_ts": last_ts.isoformat(),
            "n_records": verify_report["total"],
            "first_integrity_hash": first_record.get("integrity_hash"),
            "last_integrity_hash": last_record.get("integrity_hash"),
            "signing_key_id": last_record.get("signing_key_id"),
            "sealed_at": datetime.now(timezone.utc).isoformat(),
            "verifier_summary": report["verify_before_rotate"],
        }
        manifest_path = archived_path.with_suffix(".manifest.json")
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        # Step 4: start a fresh chain in the new active log. The
        # cross-rotation forensic link is preserved in two places:
        #   - the sealed manifest sidecar (archive_dir/*.manifest.json)
        #   - the AUDIT_LOG_ROTATED marker's *payload*, which carries
        #     the archived tail's integrity_hash and is itself signed
        # Each active file therefore verifies independently from
        # genesis ("0"*64), and a forensic walker reconstructs the
        # full history by following the marker → manifest → archive.
        self.prev_hash = "0" * 64
        self.log(
            {
                "event_type": "AUDIT_LOG_ROTATED",
                "archived_path": str(archived_path),
                "archived_first_ts": first_ts.isoformat(),
                "archived_last_ts": last_ts.isoformat(),
                "archived_n_records": verify_report["total"],
                "archived_last_integrity_hash": last_record.get("integrity_hash"),
                "archived_first_integrity_hash": first_record.get("integrity_hash"),
                "manifest_path": str(manifest_path),
                "retention_days": days,
                "rotated_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        report["rotated"] = True
        report["archived_path"] = str(archived_path)
        report["manifest_path"] = str(manifest_path)
        report["reason"] = (
            f"archived {verify_report['total']} records spanning "
            f"{first_ts.isoformat()} → {last_ts.isoformat()}"
        )
        return report


# ═══════════════════════════════════════════════════════════════════════
# 5.8  Feedback Loop Stub
# ═══════════════════════════════════════════════════════════════════════

class FeedbackLoop:
    """Record TP/FP labels; suggest weight/threshold adjustments."""

    def __init__(self):
        self.records = []

    def record(self, alert_id: str, ground_truth: str, predicted_tier: str,
               risk_score: float, actions: list) -> None:
        self.records.append({
            "alert_id": alert_id,
            "ground_truth": ground_truth,
            "predicted_tier": predicted_tier,
            "risk_score": risk_score,
            "actions": actions,
            "is_tp": ground_truth == "attack" and predicted_tier in ("MEDIUM", "HIGH", "CRITICAL"),
            "is_fp": ground_truth == "benign" and predicted_tier in ("MEDIUM", "HIGH", "CRITICAL"),
            "is_fn": ground_truth == "attack" and predicted_tier == "LOW",
        })

    def compute_adjustments(self, current_thresholds: dict | None = None) -> dict:
        """Return numeric threshold adjustments based on TP/FP/FN rates.

        Rules
        -----
        * FPR > 10 %  →  raise MEDIUM threshold by  0.05 × (FPR − 0.10) / 0.10
                         raise HIGH   threshold by  0.03 × (FPR − 0.10) / 0.10
        * FNR >  5 %  →  lower MEDIUM threshold by  0.05 × (FNR − 0.05) / 0.05
                         lower HIGH   threshold by  0.03 × (FNR − 0.05) / 0.05

        Returns a structured dict with metrics and suggested_threshold_change.
        """
        if not self.records:
            return {}

        # Default thresholds mirror risk_config.json
        if current_thresholds is None:
            current_thresholds = {"CRITICAL": 0.80, "HIGH": 0.60, "MEDIUM": 0.40}

        tp = sum(1 for r in self.records if r["is_tp"])
        fp = sum(1 for r in self.records if r["is_fp"])
        fn = sum(1 for r in self.records if r["is_fn"])
        total = len(self.records)

        fpr = fp / total if total > 0 else 0.0
        fnr = fn / total if total > 0 else 0.0

        # Compute numeric adjustments
        suggested = dict(current_thresholds)
        adjustments = []

        if fpr > 0.10:
            # Raise thresholds to reduce false positives
            delta_med  = 0.05 * (fpr - 0.10) / 0.10
            delta_high = 0.03 * (fpr - 0.10) / 0.10
            suggested["MEDIUM"]   += delta_med
            suggested["HIGH"]     += delta_high
            suggested["CRITICAL"] += delta_high * 0.5
            adjustments.append({
                "metric": "fpr", "current_value": round(fpr, 4),
                "target": 0.10, "direction": "raise",
            })

        if fnr > 0.05:
            # Lower thresholds to catch more attacks
            delta_med  = 0.05 * (fnr - 0.05) / 0.05
            delta_high = 0.03 * (fnr - 0.05) / 0.05
            suggested["MEDIUM"]   -= delta_med
            suggested["HIGH"]     -= delta_high
            suggested["CRITICAL"] -= delta_high * 0.5
            adjustments.append({
                "metric": "fnr", "current_value": round(fnr, 4),
                "target": 0.05, "direction": "lower",
            })

        if fpr <= 0.10 and fnr <= 0.05:
            adjustments.append({
                "metric": "calibrated", "current_value": None,
                "target": None, "direction": "none",
            })

        # Round suggested thresholds
        suggested = {k: round(v, 4) for k, v in suggested.items()}

        # Risk score distributions
        fp_scores = [r["risk_score"] for r in self.records if r["is_fp"]]
        tp_scores = [r["risk_score"] for r in self.records if r["is_tp"]]

        return {
            "total_evaluated": total,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "fpr": round(fpr, 4),
            "fnr": round(fnr, 4),
            "mean_fp_risk_score": round(float(np.mean(fp_scores)), 4) if fp_scores else None,
            "mean_tp_risk_score": round(float(np.mean(tp_scores)), 4) if tp_scores else None,
            "current_thresholds": current_thresholds,
            "suggested_threshold_change": suggested,
            "adjustments": adjustments,
        }


# ═══════════════════════════════════════════════════════════════════════
# 5.7  End-to-End Worked Examples
# ═══════════════════════════════════════════════════════════════════════

def run_worked_examples(
    risk_data: dict,
    attack_cats: np.ndarray,
    analyst_by_idx: dict,
    clinician_by_idx: dict,
) -> list:
    """Run 3 end-to-end scenarios: CRITICAL, HIGH, LOW."""
    R = risk_data["R"]
    levels = risk_data["risk_levels"]
    y_true = risk_data["y_true"]

    engine = PolicyEngine()
    executor = ActionExecutor()
    notifier = NotificationService()

    scenarios = []

    # Find one sample per tier
    target_tiers = ["CRITICAL", "HIGH", "LOW"]
    for tier in target_tiers:
        mask = (levels == tier) & (y_true == 1)  # prefer true attacks
        if not mask.any():
            mask = levels == tier
        if not mask.any():
            continue

        idx = int(np.where(mask)[0][np.argmax(R[mask])])
        cat = str(attack_cats[idx])
        gt = "attack" if y_true[idx] == 1 else "benign"
        a_pat = float(risk_data["d_clinical_tier"][idx])

        # Step 1: Policy recommendation
        rec = engine.recommend(
            alert_tier=tier,
            device_tier="vital_monitoring",
            attack_category=cat,
            patient_acuity=a_pat,
        )

        # Step 2: Execute (simulated)
        ts = datetime(2026, 4, 3, 12, 0, 0) + timedelta(seconds=idx)
        exec_result = executor.execute(
            f"ALERT-{idx:05d}", idx, rec["actions"], rec, gt, ts,
        )

        # Step 3: Notify
        clin_summary = clinician_by_idx.get(idx, {}).get("summary", "")
        analyst_feats = []
        if idx in analyst_by_idx:
            analyst_feats = analyst_by_idx[idx].get("models", {}).get("xgboost", {}).get("top_features", [])

        notifications = notifier.notify(
            idx, tier, rec, clin_summary, analyst_feats, float(R[idx]),
        )

        scenario = {
            "scenario": f"{tier} alert — {cat} on vital_monitoring device",
            "sample_index": idx,
            "ground_truth": gt,
            "attack_category": cat,
            "risk_score": round(float(R[idx]), 4),
            "risk_level": tier,
            "components": {
                "C_detect": round(float(risk_data["c_detect"][idx]), 4),
                "D_crit": round(float(risk_data["d_crit"][idx]), 4),
                "S_data": round(float(risk_data["s_data"][idx]), 4),
                "D_clinical_tier": round(float(risk_data["d_clinical_tier"][idx]), 4),
            },
            "policy_recommendation": rec,
            "execution_result": exec_result,
            "notifications": notifications,
            "clinical_override": rec["clinical_override"],
        }
        scenarios.append(scenario)
        logger.info("  %s: sample %d, R=%.4f, actions=%s, outcome=%s",
                    tier, idx, float(R[idx]), rec["actions"], exec_result["outcome"])

    return scenarios


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    sep = "=" * 72
    t0 = time.perf_counter()

    logger.info(sep)
    logger.info("MODULE 5 — RESPONSE PIPELINE INTEGRATION (Tasks 5.1-5.8)")
    logger.info(sep)

    # Load data
    # Strategy 1: demo path is the dashboard data source.
    risk_data = {k: v for k, v in
                 np.load(PROJECT_ROOT / "results/reports/demo_scores.npz",
                         allow_pickle=True).items()}
    with open(PROJECT_ROOT / "results/reports/analyst_report.json") as f:
        analyst_by_idx = {a["sample_index"]: a for a in json.load(f)}
    with open(PROJECT_ROOT / "results/reports/clinician_summaries.json") as f:
        clinician_by_idx = {s["sample_index"]: s for s in json.load(f)}
    attack_cats = pd.read_parquet(
        PROJECT_ROOT / "data/processed/demo_phase1.parquet",
        columns=["Attack Category"],
    )["Attack Category"].values

    n_samples = len(risk_data["R"])
    logger.info("Loaded: %d samples", n_samples)

    # 5.1 Export policy config
    export_response_policy()

    # 5.7 End-to-end worked examples
    logger.info("")
    logger.info("── 5.7 End-to-End Worked Examples ──")
    scenarios = run_worked_examples(risk_data, attack_cats, analyst_by_idx, clinician_by_idx)
    (OUTPUT_DIR / "worked_examples.json").write_text(
        json.dumps(scenarios, indent=2, default=str), encoding="utf-8")
    logger.info("  Saved: worked_examples.json (%d scenarios)", len(scenarios))

    # 5.6 + 5.8 Full pipeline run with audit logger + feedback loop
    logger.info("")
    logger.info("── 5.6/5.8 Full Pipeline Run (audit + feedback) ──")

    engine = PolicyEngine()
    executor = ActionExecutor()
    notifier = NotificationService()
    audit = AuditLogger(OUTPUT_DIR / "audit_log.jsonl")
    feedback = FeedbackLoop()

    R = risk_data["R"]
    levels = risk_data["risk_levels"]
    y_true = risk_data["y_true"]

    alert_count = 0
    for idx in range(n_samples):
        tier = str(levels[idx])
        if tier == "LOW" and R[idx] < 0.25:
            continue  # skip lowest-risk LOW alerts for efficiency

        cat = str(attack_cats[idx])
        gt = "attack" if y_true[idx] == 1 else "benign"
        a_pat = float(risk_data["d_clinical_tier"][idx])

        rec = engine.recommend(tier, "vital_monitoring", cat, a_pat)
        ts = datetime(2026, 4, 3, 12, 0, 0) + timedelta(seconds=idx)
        alert_id = f"ALERT-{idx:05d}"

        exec_result = executor.execute(alert_id, idx, rec["actions"], rec, gt, ts)
        audit.log(exec_result)
        feedback.record(alert_id, gt, tier, float(R[idx]), rec["actions"])
        alert_count += 1

    logger.info("  Processed %d alerts through pipeline", alert_count)
    logger.info("  Audit log: %s (%d records)", OUTPUT_DIR / "audit_log.jsonl", alert_count)

    # 5.8 Feedback analysis
    adjustments = feedback.compute_adjustments()
    (OUTPUT_DIR / "feedback_analysis.json").write_text(
        json.dumps(adjustments, indent=2), encoding="utf-8")
    logger.info("")
    logger.info("── 5.8 Feedback Loop Analysis ──")
    logger.info("  TP=%d, FP=%d, FN=%d", adjustments["true_positives"],
                adjustments["false_positives"], adjustments["false_negatives"])
    logger.info("  FP rate: %.1f%%, FN rate: %.1f%%",
                adjustments["fpr"] * 100, adjustments["fnr"] * 100)
    logger.info("  Current thresholds: %s", adjustments.get("current_thresholds"))
    logger.info("  Suggested thresholds: %s", adjustments.get("suggested_threshold_change"))
    for adj in adjustments.get("adjustments", []):
        logger.info("  Adjustment: %s", adj)
    logger.info("  Saved: feedback_analysis.json")

    # Notification stats
    logger.info("")
    logger.info("  Notifications generated: %d", len(notifier.notifications))

    elapsed = round(time.perf_counter() - t0, 1)
    logger.info("")
    logger.info(sep)
    logger.info("RESPONSE PIPELINE COMPLETE — %.1fs", elapsed)
    logger.info(sep)
    logger.info("  5.1 response_policy.json")
    logger.info("  5.6 audit_log.jsonl (%d records)", alert_count)
    logger.info("  5.7 worked_examples.json (%d scenarios)", len(scenarios))
    logger.info("  5.8 feedback_analysis.json")
    logger.info("  Output: %s", OUTPUT_DIR)
    logger.info(sep)


def _cli_verify(args: argparse.Namespace) -> int:
    path = Path(args.path or (OUTPUT_DIR / "audit_log.jsonl"))
    pubkey = Path(args.public_key) if args.public_key else None
    report = AuditLogger.verify(path, pubkey, legacy_ok=not args.strict)
    print(json.dumps(report, indent=2))
    return 0 if report["first_break_at"] is None else 1


def _cli_rotate(args: argparse.Namespace) -> int:
    path = Path(args.path or (OUTPUT_DIR / "audit_log.jsonl"))
    audit = AuditLogger(
        path,
        retention_days=args.retention_days,
        sign=not args.no_sign,
    )
    report = audit.rotate_and_purge(retention_days=args.retention_days)
    print(json.dumps(report, indent=2))
    return 0 if report["verify_before_rotate"] is None or report[
        "verify_before_rotate"]["first_break_at"] is None else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="python -m module5_responses.module5_pipeline",
        description="Module 5 — response pipeline + audit log management",
    )
    parser.add_argument(
        "--verify-audit-log",
        dest="verify_audit_log",
        action="store_true",
        help="Verify hash chain + signatures of an audit log JSONL file.",
    )
    parser.add_argument(
        "--rotate-audit-log",
        dest="rotate_audit_log",
        action="store_true",
        help="Rotate the active audit log if its oldest record is "
             "older than the retention window. Refuses to rotate a "
             "tampered log.",
    )
    parser.add_argument(
        "--path",
        default=None,
        help="Audit log path (default: results/reports/audit_log.jsonl)",
    )
    parser.add_argument(
        "--public-key",
        default=None,
        help="Public key PEM for verification "
             "(default: results/reports/audit_signing_key.pub.pem)",
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=None,
        help=f"Retention window in days (default: {DEFAULT_RETENTION_DAYS}; "
             f"env: IOMT_AUDIT_RETENTION_DAYS)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat unsigned (legacy) records as verification failures.",
    )
    parser.add_argument(
        "--no-sign",
        action="store_true",
        help="Disable signing for the rotate marker (testing only).",
    )

    args = parser.parse_args()

    if args.verify_audit_log:
        sys.exit(_cli_verify(args))
    if args.rotate_audit_log:
        sys.exit(_cli_rotate(args))

    main()
