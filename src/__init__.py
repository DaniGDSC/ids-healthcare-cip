# XAI-IDS-Healthcare prototype — src package
from __future__ import annotations

import re

# Tier 1 F6: pattern set expanded so the redaction is not US-English
# only. The signature of each entry stays (compiled regex, replacement).
_PHI_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # US Social Security Number — 123-45-6789
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN-REDACTED]"),
    # Common MRN labels
    (re.compile(r"\bMRN[\s:]*\S+", re.IGNORECASE), "[MRN-REDACTED]"),
    # UK NHS number — 10 digits with mandatory space pattern (3-3-4)
    (re.compile(r"\b\d{3}\s?\d{3}\s?\d{4}\b"), "[NHS-NUMBER-REDACTED]"),
    # Patient/pt name/id labels followed by a capitalised token
    (re.compile(r"\b(?:patient|pt)\s*(?:name|id)?[\s:]+[A-Z][a-z]+", re.IGNORECASE),
     "[PATIENT-REDACTED]"),
    # Date of birth — US-style MM/DD/YYYY
    (re.compile(r"\b\d{2}/\d{2}/\d{4}\b"), "[DOB-REDACTED]"),
    # Date of birth — ISO-8601 YYYY-MM-DD (defensive — biometric
    # timestamps in this codebase are emitted as ISO-8601; a stray
    # plain-text date inside an LLM hallucination is the case worth
    # redacting).
    (re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b"), "[DATE-REDACTED]"),
    # Generic NNN-NN-NNNN-like national identifier (non-US healthcare
    # IDs, e.g. some EU formats, EHR system IDs).
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[ID-REDACTED]"),
]


def sanitize_for_log(value: object) -> str:
    """Strip potential PHI patterns from a value before logging.

    Applies pattern-based redaction for SSNs, MRNs (US + UK NHS),
    patient names, and several DOB formats. Designed to be called on
    any dynamic data interpolated into log messages in alert-processing
    code — and on every LLM response before it is placed into the
    signed alert envelope (tier 1 F6).

    Args:
        value: The value to sanitize (converted to str).

    Returns:
        Sanitized string safe for logging / envelope storage.
    """
    text = str(value)
    for pattern, replacement in _PHI_PATTERNS:
        text = pattern.sub(replacement, text)
    return text
