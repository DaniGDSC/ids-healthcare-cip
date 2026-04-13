# XAI-IDS-Healthcare prototype — src package
from __future__ import annotations

import re

_PHI_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN-REDACTED]"),
    (re.compile(r"\bMRN[\s:]*\S+", re.IGNORECASE), "[MRN-REDACTED]"),
    (re.compile(r"\b(?:patient|pt)\s*(?:name|id)?[\s:]+[A-Z][a-z]+", re.IGNORECASE),
     "[PATIENT-REDACTED]"),
    (re.compile(r"\b\d{2}/\d{2}/\d{4}\b"), "[DOB-REDACTED]"),
]


def sanitize_for_log(value: object) -> str:
    """Strip potential PHI patterns from a value before logging.

    Applies pattern-based redaction for SSNs, MRNs, patient names,
    and date-of-birth formats. Designed to be called on any dynamic
    data interpolated into log messages in alert-processing code.

    Args:
        value: The value to sanitize (converted to str).

    Returns:
        Sanitized string safe for logging.
    """
    text = str(value)
    for pattern, replacement in _PHI_PATTERNS:
        text = pattern.sub(replacement, text)
    return text
