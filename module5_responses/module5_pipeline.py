"""Back-compat shim for ``module5_responses.module5_pipeline``.

The original 1.5k-LOC monolith was decomposed into:

* ``module5_responses.config``           — unified taxonomy
* ``module5_responses.policy``           — PolicyEngine + clinical_safety_check
* ``module5_responses.executor``         — ActionExecutor + NotificationService
* ``module5_responses.feedback``         — FeedbackLoop
* ``module5_responses.worked_examples``  — run_worked_examples
* ``module5_responses.audit``            — signing, logger, retention, verify
* ``module5_responses.pipeline_cli``     — main() + CLI subcommands

External consumers continue to import from this module path. Run as
``python -m module5_responses.module5_pipeline [--verify-audit-log ...]``.
"""
from __future__ import annotations

# Audit-log primitives (legacy underscore names preserved).
from .audit.logger import (
    ARCHIVE_DIR,
    AuditLogger,
    DEFAULT_RETENTION_DAYS,
)
from .audit.signing import (
    DEFAULT_PRIVATE_KEY_PATH,
    DEFAULT_PUBLIC_KEY_PATH,
    OUTPUT_DIR,
    SIGNATURE_ALG,
    _bootstrap_local_key,
    _canonical_json,
    _HAVE_CRYPTOGRAPHY,
    _load_signing_key,
    _require_cryptography,
)

# Engine + executor + feedback.
from .config import export_response_policy_dict as _build_response_policy
from .executor import ActionExecutor, NotificationService
from .feedback import FeedbackLoop
from .policy import (
    PolicyEngine,
    clinical_safety_check,
    export_response_policy,
)
from .pipeline_cli import _cli_rotate, _cli_verify, cli_entry, main
from .worked_examples import run_worked_examples

# Legacy public dict (1.x shape). External readers (dashboard,
# integration tests) historically inspected this constant; preserved via
# the unified config so consumers don't break.
RESPONSE_POLICY = _build_response_policy()


__all__ = [
    "AuditLogger",
    "ARCHIVE_DIR",
    "DEFAULT_RETENTION_DAYS",
    "DEFAULT_PRIVATE_KEY_PATH",
    "DEFAULT_PUBLIC_KEY_PATH",
    "OUTPUT_DIR",
    "SIGNATURE_ALG",
    "_HAVE_CRYPTOGRAPHY",
    "_canonical_json",
    "_load_signing_key",
    "_bootstrap_local_key",
    "_require_cryptography",
    "PolicyEngine",
    "clinical_safety_check",
    "export_response_policy",
    "ActionExecutor",
    "NotificationService",
    "FeedbackLoop",
    "run_worked_examples",
    "main",
    "_cli_verify",
    "_cli_rotate",
    "RESPONSE_POLICY",
]


if __name__ == "__main__":
    cli_entry()
