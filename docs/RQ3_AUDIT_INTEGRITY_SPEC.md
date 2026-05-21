# RQ3 Track 2 — Audit Log Integrity (Hash Chain + Schema)

**Project:** XAI-IDS-Healthcare
**Scope:** RQ3.2 — Audit trail completeness (Invariant 4); tamper-evident hash chain verification; schema compliance per `RQ3_expected_outputs.md §3.2`.
**Purpose:** Single, self-contained spec for the audit log integrity pipeline. Hand to Claude Code.
**Status of design:** All decisions locked. Five `DO NOT GUESS` checkpoints (audit log location/rotation, audit_logger.py existence, mve_mode_used field name, schema versioning, deployment fingerprint for genesis).

---

## 0. How to use this spec

1. Phase 0 is mandatory — Claude Code must verify which audit log files exist and confirm or create the `audit_logger.py` module before any verifier code runs.
2. Phases 1–6 are sequential.
3. Markers:
   - **DO NOT GUESS** — stop and ask the developer
   - **DEFENSE-CRITICAL** — directly defends the "tamper-evident audit log" claim
   - **HIPAA-ADJACENT** — directly defends the HIPAA/audit-trail compliance story
4. Total expected size: 1 shared canonicalization module, 1 audit logger (may already exist — verify), 2 new analysis scripts, 1 new test file. Runtime: verifier sub-second per ~1000 entries.

---

## 1. Background: what Track 2 produces

| Component | Question | Output | Defense weight |
|---|---|---|---|
| **Canonicalization module** | How is entry body serialized? | `common/audit_canonicalization.py` | Foundational |
| **Audit logger** | How are entries written? | `src/audit_logger.py` (may exist; verify) | Required by Invariant 4 |
| **Hash chain verifier** | Is the chain intact? | `analysis/verify_audit_log_integrity.py` | DEFENSE-CRITICAL |
| **Schema completeness auditor** | Does every entry have its required fields? | `analysis/audit_log_schema_completeness.py` | HIPAA-ADJACENT |
| **Integrity report** | Combined chain + schema status | `results/rq3_audit_integrity.json` | Defense reference |
| **CI gate test** | Hard fail on chain break or schema violation | `tests/test_step16_audit_integrity.py` | Required by Invariant 4 |

The defining property of Track 2: it makes a falsifiable claim ("the chain is intact"). A reviewer can run the verifier and see for themselves.

---

## 2. Locked design decisions

| Decision | Resolution |
|---|---|
| Hash construction | `entry_hash = SHA256(previous_hash_hex || canonical_json(body_without_hashes))` |
| Genesis entry | `previous_hash = "0" * 64` (all-zero, Bitcoin-style) |
| Canonicalization | `json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True)`. Documented as RFC 8785-compatible *for this schema* (ASCII keys, primitive values). |
| Schema validation | Hard strict per mode: Mode A and Mode B have different required field sets |
| Mode A/B asymmetry | Conditional schema: verifier branches on `mve_mode_used` |
| Chain break behavior | Detect ALL breaks; report each with ±3-entry forensic context |
| Empty log behavior | Test skips with clear message; separate gate enforces "audit log exists when study data exists" |
| Hash algorithm | SHA256 (NIST-approved; specified by `RQ3_expected_outputs.md`) |
| File format | JSON Lines (`.jsonl`), one entry per line. Matches `logs/llm_audit.jsonl` per RQ2 Compliance spec. |
| Tamper claim | "Tamper-**evident**" (detection), not "tamper-**resistant**" (prevention). Honestly framed per senior engineer review. |

---

## 3. Phase 0 — Audit infrastructure discovery (DO NOT GUESS)

### 3.1 Discovery script

```python
# scripts/discover_audit_infrastructure.py — TRANSIENT, delete after Phase 0
"""
Inventory existing audit log files, audit_logger module, and rotation policy.
"""
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
findings = {}

# 1. Candidate audit log paths
candidates = [
    "logs/llm_audit.jsonl",
    "logs/decision_audit.jsonl",
    "logs/audit.jsonl",
    "results/llm_audit_log.jsonl",
    "results/reports/alert_responses.json",
    "audit/mode_a.jsonl",
]
findings["audit_log_candidates"] = []
for p in candidates:
    full = REPO_ROOT / p
    if full.exists():
        with full.open() as f:
            first_line = f.readline().strip()
        sample_keys = []
        if first_line:
            try:
                sample_keys = list(json.loads(first_line).keys())
            except json.JSONDecodeError:
                pass
        findings["audit_log_candidates"].append({
            "path": p,
            "exists": True,
            "size_bytes": full.stat().st_size,
            "format_guess": "jsonl" if p.endswith(".jsonl") else "json",
            "sample_top_keys": sample_keys,
            "n_lines": sum(1 for _ in full.open()) if full.suffix == ".jsonl" else None,
        })
    else:
        findings["audit_log_candidates"].append({"path": p, "exists": False})

# 2. audit_logger module
logger_candidates = [
    "src/audit_logger.py",
    "module5_responses/audit_logger.py",
    "src/audit.py",
]
findings["audit_logger_module"] = None
for p in logger_candidates:
    full = REPO_ROOT / p
    if full.exists():
        findings["audit_logger_module"] = {
            "path": p,
            "size_bytes": full.stat().st_size,
        }
        text = full.read_text()
        for sym in ["AuditLogger", "log_entry", "previous_hash",
                    "entry_hash", "compute_hash", "sha256"]:
            findings["audit_logger_module"][f"has_{sym}"] = sym in text
        break

# 3. Existing schema documentation
schema_candidates = [
    "config/audit_log_schema.yaml",
    "docs/AUDIT_LOG_SCHEMA.md",
]
findings["schema_doc"] = None
for p in schema_candidates:
    if (REPO_ROOT / p).exists():
        findings["schema_doc"] = p
        break

# 4. Rotation/retention policy
findings["log_rotation"] = "DO NOT GUESS — confirm with developer"

print(json.dumps(findings, indent=2, default=str))
print("\n" + "="*60)
print("DEVELOPER ACTION:")
print("  1. Confirm which audit log file is canonical (logs/llm_audit.jsonl most likely)")
print("  2. Does src/audit_logger.py exist? If yes, confirm it implements SHA256 chain")
print("  3. If not, Phase 2 creates it. Confirm before proceeding")
print("  4. Confirm log rotation policy (single file? daily? per-run?)")
print("  5. Confirm mve_mode_used values exact strings ('A_llm' / 'B_rule' per RQ3 spec)")
print("="*60)
```

### 3.2 What to confirm before Phase 1

1. **Canonical audit log path.** `logs/llm_audit.jsonl` is the most likely default (matches RQ2 Compliance spec). Confirm.
2. **Existing `src/audit_logger.py`.** If it exists, does it implement SHA256 chaining? If yes, Phase 2 only writes the verifier; if no, Phase 2 also writes the logger.
3. **Log rotation policy.** Single growing file? Daily rotation? Per-run? The verifier needs to know whether to scan one file or many.
4. **`mve_mode_used` field exact values.** RQ3 spec says `A_llm | B_rule`. Verifier branches on these strings.
5. **Schema versioning.** If entries don't currently have a `schema_version` field, Phase 1 adds it. Defense gain: future schema changes won't silently break the verifier.

### 3.3 Verification

```bash
python scripts/discover_audit_infrastructure.py > /tmp/audit_discovery.json
# Developer reviews; confirms paths, logger existence, rotation policy
```

**DO NOT GUESS** any of these. The verifier's correctness depends on knowing exactly what file format and field schema it's reading.

---

## 4. Phase 1 — Canonicalization module (foundational)

### 4.1 Create `common/audit_canonicalization.py`

This is the **single source of truth** for how entries are serialized before hashing. Both the audit logger (writer) and the verifier (reader) import from here. If they ever drift, the chain breaks for the wrong reason.

```python
"""
common/audit_canonicalization.py

Canonical serialization for audit log entries.

This module is the SINGLE SOURCE OF TRUTH for:
  - How entries are serialized for hashing
  - How the entry_hash is computed
  - How the previous_hash is initialized

Both src/audit_logger.py (writer) and analysis/verify_audit_log_integrity.py
(reader) import from here. Modifications must be coordinated.

CANONICALIZATION CHOICE:
  We use json.dumps with sort_keys=True, separators=(',', ':'),
  ensure_ascii=True. This is RFC 8785-compatible FOR THIS SCHEMA because:
    - All keys are ASCII strings (no Unicode normalization issues)
    - All values are JSON primitives (no embedding ambiguity)
    - sort_keys=True ensures deterministic key ordering
    - separators removes whitespace ambiguity
  Should the schema evolve to include non-ASCII keys or types requiring
  Unicode normalization, upgrade to a full RFC 8785 implementation.
"""

import hashlib
import json
from typing import Any, Dict

# Genesis previous_hash for chain initialization (Bitcoin convention).
GENESIS_PREVIOUS_HASH = "0" * 64

# Fields that are part of the chain metadata, not the body that gets hashed.
HASH_METADATA_FIELDS = frozenset(["previous_hash", "entry_hash"])


def canonical_json(obj: Dict[str, Any]) -> str:
    """
    Serialize a dict to canonical JSON.

    Documented behavior:
      - Keys sorted lexicographically (sort_keys=True)
      - No whitespace between separators (separators=(',', ':'))
      - ASCII-encoded; non-ASCII chars become \\uXXXX (ensure_ascii=True)
      - No trailing newline

    Returns the canonical UTF-8-encodable string. Caller is responsible for
    encoding to bytes before hashing.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def compute_entry_hash(entry_body: Dict[str, Any], previous_hash: str) -> str:
    """
    Compute the entry_hash for an audit log entry.

    Hash construction:
        entry_hash = SHA256(previous_hash_hex || canonical_json(body_without_hashes))

    Where:
      - previous_hash_hex is the 64-character hex SHA256 of the previous entry
        (or GENESIS_PREVIOUS_HASH for the first entry)
      - body_without_hashes is entry_body with the keys "previous_hash" and
        "entry_hash" removed (they are not part of the data being authenticated)
      - canonical_json is defined above

    Args:
        entry_body: The full entry dict, possibly including previous_hash/entry_hash
        previous_hash: The previous entry's entry_hash, or GENESIS_PREVIOUS_HASH

    Returns:
        64-character lowercase hex SHA256.
    """
    if not isinstance(previous_hash, str) or len(previous_hash) != 64:
        raise ValueError(
            f"previous_hash must be 64-char hex string; got "
            f"{type(previous_hash).__name__} of length "
            f"{len(previous_hash) if isinstance(previous_hash, str) else 'N/A'}"
        )

    # Strip hash metadata to get the body that's authenticated.
    body = {k: v for k, v in entry_body.items() if k not in HASH_METADATA_FIELDS}

    serialized = canonical_json(body)
    digest_input = previous_hash.encode("ascii") + serialized.encode("utf-8")
    return hashlib.sha256(digest_input).hexdigest()


def verify_entry_hash(entry: Dict[str, Any], expected_previous_hash: str) -> Dict[str, Any]:
    """
    Recompute and verify an entry's hash against its stored entry_hash and
    expected_previous_hash.

    Returns a diagnostic dict:
      {
        "is_valid": bool,
        "stored_entry_hash": str,
        "computed_entry_hash": str,
        "stored_previous_hash": str,
        "expected_previous_hash": str,
        "previous_hash_match": bool,
        "entry_hash_match": bool,
        "failure_reason": str | None,
      }

    Does NOT raise — returns diagnostic for the verifier to interpret.
    """
    stored_prev = entry.get("previous_hash")
    stored_self = entry.get("entry_hash")

    prev_match = stored_prev == expected_previous_hash
    if not prev_match:
        return {
            "is_valid": False,
            "stored_entry_hash": stored_self,
            "computed_entry_hash": None,
            "stored_previous_hash": stored_prev,
            "expected_previous_hash": expected_previous_hash,
            "previous_hash_match": False,
            "entry_hash_match": False,
            "failure_reason": (
                f"previous_hash mismatch: stored={stored_prev!r}, "
                f"expected={expected_previous_hash!r}"
            ),
        }

    computed = compute_entry_hash(entry, expected_previous_hash)
    self_match = stored_self == computed

    return {
        "is_valid": prev_match and self_match,
        "stored_entry_hash": stored_self,
        "computed_entry_hash": computed,
        "stored_previous_hash": stored_prev,
        "expected_previous_hash": expected_previous_hash,
        "previous_hash_match": prev_match,
        "entry_hash_match": self_match,
        "failure_reason": (
            None if (prev_match and self_match)
            else f"entry_hash mismatch: stored={stored_self!r}, computed={computed!r}"
        ),
    }
```

### 4.2 Unit tests for canonicalization

Create `tests/test_audit_canonicalization.py`:

```python
"""Unit tests for the canonicalization module."""
import pytest

from common.audit_canonicalization import (
    GENESIS_PREVIOUS_HASH,
    canonical_json,
    compute_entry_hash,
    verify_entry_hash,
)


def test_canonical_json_key_order_invariant():
    """Same data, different key order in source → same output."""
    a = {"b": 1, "a": 2}
    b = {"a": 2, "b": 1}
    assert canonical_json(a) == canonical_json(b)


def test_canonical_json_no_whitespace():
    """No spaces in separators."""
    out = canonical_json({"a": 1, "b": [2, 3]})
    assert " " not in out


def test_canonical_json_ascii_only():
    """Non-ASCII characters are escaped."""
    out = canonical_json({"text": "café"})
    assert "café" not in out  # raw bytes not in output
    assert "caf\\u00e9" in out  # escaped form is


def test_genesis_hash_format():
    """Genesis previous_hash is 64 zero chars."""
    assert GENESIS_PREVIOUS_HASH == "0" * 64
    assert len(GENESIS_PREVIOUS_HASH) == 64


def test_compute_entry_hash_excludes_metadata():
    """Hashes of {body} and {body + previous_hash + entry_hash} are equal."""
    body = {"alert_id": "x", "value": 42}
    body_with_meta = dict(body, previous_hash="ab" * 32, entry_hash="cd" * 32)
    h1 = compute_entry_hash(body, GENESIS_PREVIOUS_HASH)
    h2 = compute_entry_hash(body_with_meta, GENESIS_PREVIOUS_HASH)
    assert h1 == h2


def test_compute_entry_hash_rejects_bad_previous():
    """Non-64-char previous_hash raises."""
    with pytest.raises(ValueError):
        compute_entry_hash({"x": 1}, "not_hex")
    with pytest.raises(ValueError):
        compute_entry_hash({"x": 1}, "0" * 63)
    with pytest.raises(ValueError):
        compute_entry_hash({"x": 1}, 12345)


def test_compute_entry_hash_deterministic():
    """Same body + previous → same hash."""
    body = {"alert_id": "x", "value": 42}
    h1 = compute_entry_hash(body, GENESIS_PREVIOUS_HASH)
    h2 = compute_entry_hash(body, GENESIS_PREVIOUS_HASH)
    assert h1 == h2


def test_chain_link_changes_hash():
    """Different previous_hash → different entry_hash for same body."""
    body = {"alert_id": "x"}
    h1 = compute_entry_hash(body, GENESIS_PREVIOUS_HASH)
    h2 = compute_entry_hash(body, "1" * 64)
    assert h1 != h2


def test_verify_entry_hash_valid():
    body = {"alert_id": "x", "value": 42}
    h = compute_entry_hash(body, GENESIS_PREVIOUS_HASH)
    entry = dict(body, previous_hash=GENESIS_PREVIOUS_HASH, entry_hash=h)
    result = verify_entry_hash(entry, GENESIS_PREVIOUS_HASH)
    assert result["is_valid"]


def test_verify_entry_hash_detects_body_tamper():
    body = {"alert_id": "x", "value": 42}
    h = compute_entry_hash(body, GENESIS_PREVIOUS_HASH)
    tampered = dict(body, value=999, previous_hash=GENESIS_PREVIOUS_HASH, entry_hash=h)
    result = verify_entry_hash(tampered, GENESIS_PREVIOUS_HASH)
    assert not result["is_valid"]
    assert result["entry_hash_match"] is False


def test_verify_entry_hash_detects_chain_break():
    body = {"alert_id": "x"}
    h = compute_entry_hash(body, GENESIS_PREVIOUS_HASH)
    entry = dict(body, previous_hash=GENESIS_PREVIOUS_HASH, entry_hash=h)
    result = verify_entry_hash(entry, "1" * 64)
    assert not result["is_valid"]
    assert result["previous_hash_match"] is False
```

### 4.3 Verification

```bash
pytest tests/test_audit_canonicalization.py -v
# Expected: 10 tests pass
```

---

## 5. Phase 2 — Audit logger (DO NOT GUESS existence)

### 5.1 Decision tree

Phase 0 discovery reveals one of three states:

1. **`src/audit_logger.py` exists and uses SHA256 chain compatible with our canonicalization.** No code changes; just import from `common.audit_canonicalization` instead of inlined functions if it was duplicated.
2. **`src/audit_logger.py` exists but uses a different hashing scheme.** Migration step required (see §5.3 below).
3. **`src/audit_logger.py` does not exist.** Create per the template in §5.2.

Claude Code must determine which state applies before proceeding.

### 5.2 Template: `src/audit_logger.py` (if needed)

```python
"""
src/audit_logger.py
Append-only hash-chained audit logger for IDS alert decisions.

Tamper-EVIDENT (not tamper-resistant): production deployment should route
to WORM storage / SIEM. The hash chain makes filesystem tampering detectable.

Schema versioned: see config/audit_log_schema.yaml.

Thread safety: uses fcntl.flock for write-side mutual exclusion on POSIX.
On Windows, falls back to msvcrt.locking.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from common.audit_canonicalization import (
    GENESIS_PREVIOUS_HASH,
    canonical_json,
    compute_entry_hash,
)

DEFAULT_LOG_PATH = Path("logs/llm_audit.jsonl")
SCHEMA_VERSION = "1.0"


class AuditLogger:
    """
    Append-only audit logger with SHA256 hash chain.

    Usage:
        logger = AuditLogger()
        logger.append(entry_body)   # raises on chain corruption
    """

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = Path(log_path) if log_path else DEFAULT_LOG_PATH
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _last_entry_hash(self) -> str:
        """Return the entry_hash of the last entry, or GENESIS for empty log."""
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            return GENESIS_PREVIOUS_HASH

        # Read the last line. For very large files, read backward;
        # for typical thesis-scale logs, reading from the end is fine.
        with self.log_path.open("rb") as f:
            try:
                f.seek(-2, os.SEEK_END)
                while f.read(1) != b"\n":
                    f.seek(-2, os.SEEK_CUR)
            except OSError:
                # File too short — read all of it
                f.seek(0)
                last_line = f.readlines()[-1] if f.readlines() else b""
                f.seek(0)
            last_line = f.readline()

        if not last_line.strip():
            return GENESIS_PREVIOUS_HASH

        try:
            last_entry = json.loads(last_line)
            return last_entry["entry_hash"]
        except (json.JSONDecodeError, KeyError) as e:
            raise RuntimeError(
                f"Cannot extract entry_hash from last log entry: {e}. "
                f"Log may be corrupted."
            )

    def append(self, entry_body: Dict[str, Any]) -> Dict[str, Any]:
        """
        Append an entry to the log with chain link.

        Args:
            entry_body: dict matching the audit log schema. Must NOT include
                previous_hash or entry_hash (they are added by this method).
                schema_version is added if not present.

        Returns:
            The full entry dict that was written.

        Raises:
            ValueError if entry_body contains hash metadata fields.
        """
        forbidden = {"previous_hash", "entry_hash"}
        if forbidden & entry_body.keys():
            raise ValueError(
                f"entry_body must not contain hash metadata fields: "
                f"{forbidden & entry_body.keys()}"
            )

        # Add schema_version + timestamp if not already present
        entry = dict(entry_body)
        entry.setdefault("schema_version", SCHEMA_VERSION)
        entry.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

        previous_hash = self._last_entry_hash()
        entry_hash = compute_entry_hash(entry, previous_hash)

        # Assemble final entry with hashes
        full_entry = dict(entry,
                          previous_hash=previous_hash,
                          entry_hash=entry_hash)

        # Append-atomic write
        line = canonical_json(full_entry) + "\n"
        self._atomic_append(line)

        return full_entry

    def _atomic_append(self, line: str):
        """Atomic line append with cross-platform locking."""
        # POSIX path
        if sys.platform != "win32":
            import fcntl
            with self.log_path.open("a", encoding="utf-8") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(line)
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            return

        # Windows fallback
        import msvcrt
        with self.log_path.open("a", encoding="utf-8") as f:
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
            try:
                f.write(line)
                f.flush()
                os.fsync(f.fileno())
            finally:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def get_default_logger() -> AuditLogger:
    """Module-level convenience for the default log path."""
    return AuditLogger()
```

### 5.3 Migration path (if existing logger uses different hashing)

If Phase 0 reveals an existing `audit_logger.py` with a different hash construction:

1. **Document the legacy scheme** in `docs/AUDIT_LOG_LEGACY_MIGRATION.md`.
2. **Add a `schema_version` field** to distinguish legacy from new entries.
3. **The verifier handles both:** entries with `schema_version: "0.x"` use the legacy hash function; entries with `schema_version: "1.0"+` use the canonical one.
4. **Rotate the log:** start a new file (`logs/llm_audit_v1.jsonl`); preserve the old one read-only.

This avoids retroactively rehashing — which would itself look like tampering.

**DO NOT GUESS** whether migration is needed. Phase 0 inventory determines this; surface to developer if unclear.

---

## 6. Phase 3 — Schema specification

### 6.1 Create `config/audit_log_schema.yaml`

```yaml
# config/audit_log_schema.yaml
# Audit log entry schema for XAI-IDS-Healthcare.
# Read by analysis/audit_log_schema_completeness.py.
#
# Hard strict validation per mode: Mode A and Mode B have different
# required field sets in explanation_context.

schema_version: "1.0"
preregistered_date: "2025-08-14"   # DO NOT GUESS — set to actual lock date
last_updated: "2026-MM-DD"

# Section: fields that must always be present regardless of mode.
sections:
  alert_context:
    required_always:
      - {field: alert_id, type: string}
      - {field: fusion_class, type: string,
         enum: [KNOWN_ATTACK, CONFIRMED_ANOMALY, NOVEL_ANOMALY, BENIGN]}
      - {field: risk_tier, type: string, enum: [CRITICAL, HIGH, MEDIUM, LOW]}
      - {field: recommended_action, type: string}
      - {field: primary_action_code, type: string}

  operator_context:
    required_always:
      - {field: operator_role, type: string,
         enum: [IT_GENERALIST, BIOMED_ENGINEER, NURSE_MANAGER]}
      - {field: view_role_rendered, type: string,
         enum: [IT_GENERALIST, BIOMED_ENGINEER, NURSE_MANAGER]}
    required_when_study: # required only when in study mode
      - {field: participant_id, type: string}
      - {field: group, type: string, enum: [A, B]}

  decision_capture:
    required_always:
      - {field: operator_action_taken, type: string}
      - {field: decision_time_seconds, type: [integer, float, "null"]}
      - {field: operator_confidence, type: [integer, "null"], range: [1, 5]}
      - {field: operator_rationale, type: [string, "null"]}

  explanation_context:
    required_always:
      - {field: mve_mode_used, type: string, enum: [A_llm, B_rule]}
      - {field: mve_text_shown, type: string}
      - {field: shap_features_shown, type: array, items_type: string,
         length: 3}
      - {field: shap_stability_score, type: [number, "null"]}
    required_when_mode_a:
      - {field: llm_provider, type: string}
      - {field: llm_model_version, type: string}
      - {field: full_prompt, type: string}
      - {field: full_response, type: string}

  tamper_evidence:
    required_always:
      - {field: previous_hash, type: string, length: 64, pattern: "^[0-9a-f]{64}$"}
      - {field: entry_hash, type: string, length: 64, pattern: "^[0-9a-f]{64}$"}

  meta:
    required_always:
      - {field: timestamp, type: string, format: iso8601}
      - {field: schema_version, type: string}
```

### 6.2 Verification

```bash
python -c "
import yaml
from pathlib import Path
doc = yaml.safe_load(Path('config/audit_log_schema.yaml').read_text())
print(f'Schema {doc[\"schema_version\"]}, preregistered {doc[\"preregistered_date\"]}')
for section, body in doc['sections'].items():
    n_always = len(body.get('required_always', []))
    n_cond = sum(len(body.get(k, [])) for k in body if k != 'required_always')
    print(f'  {section}: {n_always} always, {n_cond} conditional')
"
```

---

## 7. Phase 4 — Chain verifier (DEFENSE-CRITICAL)

### 7.1 Create `analysis/verify_audit_log_integrity.py`

**Contract:**
- **Input:** `logs/llm_audit.jsonl` (or path passed via env var / CLI).
- **Output:** `results/rq3_audit_chain_verification.json`.
- **Runtime:** sub-second per ~1000 entries; linear in chain length.
- **Behavior:** detects ALL chain breaks; reports each with ±3-entry forensic context.

### 7.2 Algorithm

```
1. Open log file. If missing or empty: emit "no_log_to_verify" status, exit cleanly.
2. expected_prev = GENESIS_PREVIOUS_HASH
3. For each line, parse JSON. On parse error: record break (entry malformed).
4. Compute expected hash via verify_entry_hash(entry, expected_prev).
   - If valid: expected_prev = entry["entry_hash"]; continue.
   - If invalid: record break with ±3-entry context; set expected_prev to
     the broken entry's stored entry_hash (continue forensics from there).
5. Emit summary: n_entries, n_breaks, breaks[]
```

### 7.3 Output schema

```json
{
  "_meta": {
    "schema_version": "1.0",
    "generated_at": "<ISO-8601>",
    "generated_by": "analysis/verify_audit_log_integrity.py",
    "log_path": "logs/llm_audit.jsonl",
    "log_sha256": "<hash of file>",
    "n_entries_scanned": 384,
    "canonicalization_module": "common.audit_canonicalization v1.0",
    "hash_algorithm": "SHA256",
    "genesis_previous_hash": "0000000000000000000000000000000000000000000000000000000000000000"
  },
  "headline": {
    "chain_intact": true,
    "n_entries": 384,
    "n_breaks": 0,
    "first_entry_timestamp": "2026-05-01T10:00:00Z",
    "last_entry_timestamp": "2026-05-19T14:23:00Z",
    "verification_complete": true
  },
  "breaks": [
    {
      "entry_line_number": 87,
      "entry_alert_id": "alert_00087",
      "reason": "entry_hash mismatch: stored vs computed",
      "stored_entry_hash": "abc123...",
      "computed_entry_hash": "def456...",
      "stored_previous_hash": "789xyz...",
      "expected_previous_hash": "789xyz...",
      "forensic_context": {
        "preceding_entries": [
          {"line": 84, "alert_id": "alert_00084", "timestamp": "..."},
          {"line": 85, "alert_id": "alert_00085", "timestamp": "..."},
          {"line": 86, "alert_id": "alert_00086", "timestamp": "..."}
        ],
        "broken_entry": {"line": 87, "alert_id": "alert_00087", ...},
        "following_entries": [
          {"line": 88, "alert_id": "alert_00088", "timestamp": "..."},
          {"line": 89, "alert_id": "alert_00089", "timestamp": "..."},
          {"line": 90, "alert_id": "alert_00090", "timestamp": "..."}
        ]
      }
    }
  ],
  "segments": [
    {
      "start_line": 1, "end_line": 86, "n_entries": 86,
      "first_alert_id": "alert_00001", "last_alert_id": "alert_00086",
      "_status": "intact"
    },
    {
      "start_line": 88, "end_line": 384, "n_entries": 297,
      "first_alert_id": "alert_00088", "last_alert_id": "alert_00384",
      "_status": "intact"
    }
  ]
}
```

### 7.4 Implementation

```python
"""
analysis/verify_audit_log_integrity.py
Verify the SHA256 hash chain of the audit log.

Detects ALL chain breaks (not just the first). For each break, captures
±3 entries of forensic context so a reviewer can identify which entry
was tampered with.

Output: results/rq3_audit_chain_verification.json
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from common.audit_canonicalization import (
    GENESIS_PREVIOUS_HASH,
    verify_entry_hash,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG = REPO_ROOT / "logs/llm_audit.jsonl"   # DO NOT GUESS — verify
OUT = REPO_ROOT / "results/rq3_audit_chain_verification.json"

FORENSIC_CONTEXT_WIDTH = 3


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _entry_summary(entry: dict, line: int) -> dict:
    """Minimal info per entry for forensic context."""
    return {
        "line": line,
        "alert_id": entry.get("alert_id"),
        "timestamp": entry.get("timestamp"),
        "entry_hash": entry.get("entry_hash", "")[:16] + "...",  # truncate
    }


def main(log_path: Optional[Path] = None):
    log_path = log_path or DEFAULT_LOG

    out_meta = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": "analysis/verify_audit_log_integrity.py",
        "log_path": str(log_path.relative_to(REPO_ROOT)),
        "canonicalization_module": "common.audit_canonicalization v1.0",
        "hash_algorithm": "SHA256",
        "genesis_previous_hash": GENESIS_PREVIOUS_HASH,
    }

    if not log_path.exists() or log_path.stat().st_size == 0:
        result = {
            "_meta": out_meta,
            "headline": {
                "chain_intact": None,
                "n_entries": 0,
                "n_breaks": 0,
                "verification_complete": False,
                "_note": "Audit log missing or empty — nothing to verify.",
            },
            "breaks": [],
            "segments": [],
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(result, indent=2, default=str))
        print(f"Wrote {OUT.relative_to(REPO_ROOT)} (no-op: empty log)")
        return

    out_meta["log_sha256"] = _file_sha256(log_path)

    # Load all entries once for forensic context windows
    entries: List[dict] = []
    parse_errors: List[dict] = []
    with log_path.open("r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                entries.append(json.loads(raw))
            except json.JSONDecodeError as e:
                parse_errors.append({
                    "line_number": line_num,
                    "error": str(e),
                    "raw_excerpt": raw[:120],
                })

    # Walk chain
    expected_prev = GENESIS_PREVIOUS_HASH
    breaks = []
    segments = [{"start_line": 1, "n_entries": 0}]

    for i, entry in enumerate(entries):
        line_num = i + 1  # 1-based
        result = verify_entry_hash(entry, expected_prev)
        if result["is_valid"]:
            segments[-1]["n_entries"] += 1
            segments[-1]["end_line"] = line_num
            expected_prev = result["stored_entry_hash"]
        else:
            preceding = [
                _entry_summary(entries[j], j + 1)
                for j in range(max(0, i - FORENSIC_CONTEXT_WIDTH), i)
            ]
            following = [
                _entry_summary(entries[j], j + 1)
                for j in range(i + 1, min(len(entries), i + 1 + FORENSIC_CONTEXT_WIDTH))
            ]
            breaks.append({
                "entry_line_number": line_num,
                "entry_alert_id": entry.get("alert_id"),
                "reason": result["failure_reason"],
                "stored_entry_hash": result["stored_entry_hash"],
                "computed_entry_hash": result["computed_entry_hash"],
                "stored_previous_hash": result["stored_previous_hash"],
                "expected_previous_hash": result["expected_previous_hash"],
                "forensic_context": {
                    "preceding_entries": preceding,
                    "broken_entry": _entry_summary(entry, line_num),
                    "following_entries": following,
                },
            })

            # Close current segment, start new one from after the break
            segments[-1]["end_line"] = line_num - 1
            segments.append({
                "start_line": line_num + 1, "n_entries": 0
            })
            # Continue verification using the broken entry's stored hash
            # (forensics continues even if chain is broken)
            expected_prev = entry.get("entry_hash", expected_prev)

    # Annotate segments
    for seg in segments:
        seg["_status"] = "intact" if seg["n_entries"] > 0 else "empty"
    # Drop trailing empty segments
    segments = [s for s in segments if s["n_entries"] > 0]

    headline = {
        "chain_intact": len(breaks) == 0 and len(parse_errors) == 0,
        "n_entries": len(entries),
        "n_breaks": len(breaks),
        "n_parse_errors": len(parse_errors),
        "first_entry_timestamp": entries[0].get("timestamp") if entries else None,
        "last_entry_timestamp": entries[-1].get("timestamp") if entries else None,
        "verification_complete": True,
    }
    out_meta["n_entries_scanned"] = len(entries)

    result = {
        "_meta": out_meta,
        "headline": headline,
        "breaks": breaks,
        "parse_errors": parse_errors,
        "segments": segments,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, default=str))
    print(f"Wrote {OUT.relative_to(REPO_ROOT)}")
    print(f"Chain: {'INTACT' if headline['chain_intact'] else 'BROKEN'} "
          f"({len(entries)} entries, {len(breaks)} breaks)")


if __name__ == "__main__":
    main()
```

### 7.5 Verification

```bash
python -m analysis.verify_audit_log_integrity
cat results/rq3_audit_chain_verification.json | python -m json.tool | head -30
# Expected if log exists: chain_intact: true, n_entries > 0
# Expected if log missing: chain_intact: null, n_entries: 0
```

---

## 8. Phase 5 — Schema completeness auditor (HIPAA-ADJACENT)

### 8.1 Create `analysis/audit_log_schema_completeness.py`

**Contract:**
- **Input:** audit log file + `config/audit_log_schema.yaml`.
- **Output:** `results/rq3_audit_schema_audit.json`.
- **Runtime:** sub-second per ~1000 entries.
- **Behavior:** validates every entry against the schema. Hard strict, conditional on `mve_mode_used`.

### 8.2 Algorithm

```
1. Load schema.
2. For each entry:
   a. Collect required_always fields from each section.
   b. If mve_mode_used == "A_llm": add required_when_mode_a from explanation_context.
   c. If participant_id is present anywhere: add required_when_study from operator_context.
   d. Check each required field: present, correct type, range/enum constraint.
3. Emit per-entry verdict + aggregate summary.
```

### 8.3 Output schema

```json
{
  "_meta": { ... },
  "headline": {
    "all_entries_pass_schema": true,
    "n_entries_validated": 384,
    "n_entries_failing": 0,
    "by_mode": {
      "A_llm": {"n_validated": 312, "n_failing": 0},
      "B_rule": {"n_validated": 72, "n_failing": 0}
    }
  },
  "failures": [
    {
      "line_number": 47,
      "alert_id": "alert_00047",
      "mode": "A_llm",
      "missing_required_fields": ["full_prompt"],
      "type_violations": [],
      "enum_violations": [],
      "range_violations": []
    }
  ]
}
```

### 8.4 Implementation outline

```python
"""
analysis/audit_log_schema_completeness.py
Validate every audit log entry against config/audit_log_schema.yaml.

Hard strict per mode. Mode A and Mode B have different required field sets.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = REPO_ROOT / "logs/llm_audit.jsonl"  # DO NOT GUESS
SCHEMA_PATH = REPO_ROOT / "config/audit_log_schema.yaml"
OUT = REPO_ROOT / "results/rq3_audit_schema_audit.json"


def _required_fields_for_entry(schema, entry):
    """Return the set of required field specs for this entry, conditional on mode."""
    required = []
    sections = schema["sections"]

    for sect_name, sect_body in sections.items():
        # required_always
        for f in sect_body.get("required_always", []):
            required.append((sect_name, f))

        # required_when_mode_a — only for explanation_context, only if Mode A
        if sect_name == "explanation_context":
            if entry.get("mve_mode_used") == "A_llm":
                for f in sect_body.get("required_when_mode_a", []):
                    required.append((sect_name, f))

        # required_when_study — only for operator_context, only if in study
        if sect_name == "operator_context":
            if entry.get("participant_id") is not None:
                for f in sect_body.get("required_when_study", []):
                    required.append((sect_name, f))

    return required


def _validate_field(field_spec, value) -> List[Dict[str, Any]]:
    """Return list of violations for a single field. Empty list = pass."""
    violations = []
    field_name = field_spec["field"]

    # Type check
    expected_type = field_spec.get("type")
    if isinstance(expected_type, list):
        # Multi-type allowed (e.g., [number, null])
        type_names = expected_type
    else:
        type_names = [expected_type]

    if not _type_matches(value, type_names):
        violations.append({
            "field": field_name,
            "kind": "type",
            "expected": expected_type,
            "got": type(value).__name__,
        })
        return violations  # other checks moot if type wrong

    # Enum check
    if "enum" in field_spec and value not in field_spec["enum"]:
        violations.append({
            "field": field_name,
            "kind": "enum",
            "expected": field_spec["enum"],
            "got": value,
        })

    # Range check
    if "range" in field_spec and value is not None:
        lo, hi = field_spec["range"]
        if not (lo <= value <= hi):
            violations.append({
                "field": field_name,
                "kind": "range",
                "expected": f"[{lo}, {hi}]",
                "got": value,
            })

    # Length check
    if "length" in field_spec and isinstance(value, (str, list)):
        if len(value) != field_spec["length"]:
            violations.append({
                "field": field_name,
                "kind": "length",
                "expected": field_spec["length"],
                "got": len(value),
            })

    # Pattern check (for hashes etc.)
    if "pattern" in field_spec and isinstance(value, str):
        if not re.match(field_spec["pattern"], value):
            violations.append({
                "field": field_name,
                "kind": "pattern",
                "expected": field_spec["pattern"],
                "got": value[:20] + "...",
            })

    return violations


def _type_matches(value, type_names) -> bool:
    """Check if value's type matches any of the named types."""
    type_map = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "float": float,
        "array": list,
        "object": dict,
        "boolean": bool,
        "null": type(None),
    }
    for tn in type_names:
        py_type = type_map.get(tn)
        if py_type and isinstance(value, py_type):
            # Special case: bool is subclass of int — exclude
            if tn == "integer" and isinstance(value, bool):
                continue
            return True
    return False


def _validate_entry(entry, schema):
    """Validate one entry. Returns dict with verdict + violations."""
    required = _required_fields_for_entry(schema, entry)
    missing = []
    violations = []

    for sect_name, field_spec in required:
        # The schema is structured as sections, but the JSONL entries are flat.
        # Lookup is by field name, regardless of which section it's documented in.
        field_name = field_spec["field"]
        if field_name not in entry:
            missing.append(field_name)
            continue
        violations.extend(_validate_field(field_spec, entry[field_name]))

    return {
        "missing_required_fields": missing,
        "type_violations": [v for v in violations if v["kind"] == "type"],
        "enum_violations": [v for v in violations if v["kind"] == "enum"],
        "range_violations": [v for v in violations if v["kind"] == "range"],
        "length_violations": [v for v in violations if v["kind"] == "length"],
        "pattern_violations": [v for v in violations if v["kind"] == "pattern"],
        "is_valid": not missing and not violations,
    }


def main():
    schema = yaml.safe_load(SCHEMA_PATH.read_text())
    if not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0:
        result = {
            "_meta": {
                "schema_version": "1.0",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "log_path": str(LOG_PATH.relative_to(REPO_ROOT)),
            },
            "headline": {
                "all_entries_pass_schema": None,
                "n_entries_validated": 0,
                "n_entries_failing": 0,
                "_note": "Audit log missing or empty — nothing to validate.",
            },
            "failures": [],
        }
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(result, indent=2, default=str))
        return

    failures = []
    by_mode = {"A_llm": {"n_validated": 0, "n_failing": 0},
               "B_rule": {"n_validated": 0, "n_failing": 0}}

    with LOG_PATH.open("r") as f:
        for line_num, raw in enumerate(f, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                failures.append({
                    "line_number": line_num,
                    "_status": "json_parse_error",
                })
                continue

            mode = entry.get("mve_mode_used", "unknown")
            if mode in by_mode:
                by_mode[mode]["n_validated"] += 1

            verdict = _validate_entry(entry, schema)
            if not verdict["is_valid"]:
                if mode in by_mode:
                    by_mode[mode]["n_failing"] += 1
                failures.append({
                    "line_number": line_num,
                    "alert_id": entry.get("alert_id"),
                    "mode": mode,
                    **{k: v for k, v in verdict.items() if k != "is_valid"},
                })

    n_total = by_mode["A_llm"]["n_validated"] + by_mode["B_rule"]["n_validated"]
    n_failing = len(failures)
    result = {
        "_meta": {
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generated_by": "analysis/audit_log_schema_completeness.py",
            "log_path": str(LOG_PATH.relative_to(REPO_ROOT)),
            "schema_path": str(SCHEMA_PATH.relative_to(REPO_ROOT)),
        },
        "headline": {
            "all_entries_pass_schema": n_failing == 0,
            "n_entries_validated": n_total,
            "n_entries_failing": n_failing,
            "by_mode": by_mode,
        },
        "failures": failures[:50],
        "failures_truncated_at": 50,
        "failures_total_count": n_failing,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, default=str))
    print(f"Wrote {OUT.relative_to(REPO_ROOT)}")
    print(f"Schema audit: {'PASS' if n_failing == 0 else f'FAIL ({n_failing} entries)'}")


if __name__ == "__main__":
    main()
```

### 8.5 Verification

```bash
python -m analysis.audit_log_schema_completeness
cat results/rq3_audit_schema_audit.json | python -m json.tool | head -30
```

---

## 9. Phase 6 — CI gate test

### 9.1 Create `tests/test_step16_audit_integrity.py`

This is the **test mapped to Invariant 4** in the Track 1 manifest. Once it exists, Invariant 4 transitions from `pending` to `enforced`.

```python
"""
Audit log integrity tests (Invariant 4).

Three CI gates:
  1. Chain intact (no SHA256 mismatches)
  2. Schema complete (all required fields per mode)
  3. Audit log exists when study data exists

Test skips if no audit log yet (acceptable: no Mode A runs have occurred).
"""

import json
from pathlib import Path

import pytest

CHAIN_JSON = Path("results/rq3_audit_chain_verification.json")
SCHEMA_JSON = Path("results/rq3_audit_schema_audit.json")
SURVEY_DIR = Path("survey")


def _run_chain_verifier():
    """Run the chain verifier; skip if it can't be invoked."""
    import subprocess
    try:
        subprocess.run(
            ["python", "-m", "analysis.verify_audit_log_integrity"],
            check=True, capture_output=True
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Chain verifier failed to run: {e.stderr.decode()}")


def _run_schema_auditor():
    import subprocess
    try:
        subprocess.run(
            ["python", "-m", "analysis.audit_log_schema_completeness"],
            check=True, capture_output=True
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Schema auditor failed to run: {e.stderr.decode()}")


# ─── Test 1: Hash chain integrity ──────────────────────────────

def test_audit_chain_intact():
    """SHA256 hash chain must be intact end-to-end."""
    if not CHAIN_JSON.exists():
        _run_chain_verifier()
    if not CHAIN_JSON.exists():
        pytest.skip("Chain verifier produced no output")

    result = json.loads(CHAIN_JSON.read_text())
    h = result["headline"]

    # Case 1: no audit log to verify
    if h.get("n_entries") == 0:
        pytest.skip("No audit log entries to verify yet")

    # Case 2: audit log exists, must be intact
    assert h["chain_intact"], (
        f"Audit log chain BROKEN: {h['n_breaks']} break(s), "
        f"{h.get('n_parse_errors', 0)} parse error(s). "
        f"See {CHAIN_JSON} for forensic context."
    )


# ─── Test 2: Schema completeness ───────────────────────────────

def test_audit_schema_complete():
    """Every audit entry must satisfy its mode-conditional schema."""
    if not SCHEMA_JSON.exists():
        _run_schema_auditor()
    if not SCHEMA_JSON.exists():
        pytest.skip("Schema auditor produced no output")

    result = json.loads(SCHEMA_JSON.read_text())
    h = result["headline"]

    if h.get("n_entries_validated", 0) == 0:
        pytest.skip("No audit log entries to validate yet")

    assert h["all_entries_pass_schema"], (
        f"{h['n_entries_failing']}/{h['n_entries_validated']} audit entries "
        f"violate schema. See {SCHEMA_JSON}."
    )


# ─── Test 3: Audit log presence when study data exists ─────────

def test_audit_log_exists_when_study_present():
    """
    HIPAA-ADJACENT: if survey/study_responses_*.json files exist (user study
    in progress or complete), the audit log must exist as well.

    This is a separate test from "chain intact" — it catches the case where
    a study runs but the logger silently failed to write.
    """
    study_files = list(SURVEY_DIR.glob("study_responses_*.json")) \
        if SURVEY_DIR.exists() else []
    if not study_files:
        pytest.skip("No study data yet — audit log presence not required")

    audit_log = Path("logs/llm_audit.jsonl")
    assert audit_log.exists() and audit_log.stat().st_size > 0, (
        f"Study data present ({len(study_files)} files) but audit log "
        f"missing or empty at {audit_log}. "
        f"Mode A runs without audit logging is a compliance violation."
    )


# ─── Test 4: Canonicalization module loadable ──────────────────

def test_canonicalization_module_importable():
    """Hard dependency: canonicalization must be importable from both writer and reader."""
    from common.audit_canonicalization import (
        GENESIS_PREVIOUS_HASH, canonical_json, compute_entry_hash, verify_entry_hash
    )
    assert GENESIS_PREVIOUS_HASH == "0" * 64
    # Round-trip smoke test
    body = {"alert_id": "test"}
    h = compute_entry_hash(body, GENESIS_PREVIOUS_HASH)
    assert len(h) == 64
```

### 9.2 Verification

```bash
pytest tests/test_step16_audit_integrity.py -v
# Expected:
#   - test_canonicalization_module_importable: PASS
#   - test_audit_chain_intact: PASS or SKIP (if no log yet)
#   - test_audit_schema_complete: PASS or SKIP
#   - test_audit_log_exists_when_study_present: PASS or SKIP (no study yet)
```

---

## 10. Execution order

```bash
# ─── PHASE 0: DISCOVERY ───────────────────────────────────────
python scripts/discover_audit_infrastructure.py > /tmp/audit_discovery.json
# DEVELOPER CONFIRMS: log path, audit_logger.py existence, mode field values

# ─── PHASE 1: CANONICALIZATION MODULE ─────────────────────────
# Create common/audit_canonicalization.py
# Create tests/test_audit_canonicalization.py
pytest tests/test_audit_canonicalization.py -v
# Expected: 10 tests pass

# ─── PHASE 2: AUDIT LOGGER ────────────────────────────────────
# Decision tree per §5.1:
#   - exists + compatible: import from common.audit_canonicalization
#   - exists + incompatible: migrate per §5.3
#   - missing: create per §5.2

# ─── PHASE 3: SCHEMA SPEC ─────────────────────────────────────
# Create config/audit_log_schema.yaml

# ─── PHASE 4: CHAIN VERIFIER ──────────────────────────────────
python -m analysis.verify_audit_log_integrity
cat results/rq3_audit_chain_verification.json | python -m json.tool | head -30

# ─── PHASE 5: SCHEMA AUDITOR ──────────────────────────────────
python -m analysis.audit_log_schema_completeness
cat results/rq3_audit_schema_audit.json | python -m json.tool | head -30

# ─── PHASE 6: CI GATE ─────────────────────────────────────────
# Create tests/test_step16_audit_integrity.py
pytest tests/test_step16_audit_integrity.py -v

# ─── FINAL VERIFICATION ───────────────────────────────────────
ls common/audit_canonicalization.py \
   src/audit_logger.py \
   config/audit_log_schema.yaml \
   results/rq3_audit_chain_verification.json \
   results/rq3_audit_schema_audit.json
```

---

## 11. Integration with Track 1 (Invariant Evidence)

Once Phase 6 of this spec is complete, **Invariant 4** in `config/invariants_manifest.yaml` transitions from `status: pending` to `status: enforced`:

```yaml
  - id: 4
    title: "Audit trail complete"
    # ... existing fields ...
    verification_method: pytest
    test_files:
      - "tests/test_step16_audit_integrity.py"
    status: enforced   # was: pending
```

Re-run `python -m analysis.compile_invariant_evidence` to refresh the Track 1 evidence JSON. The headline `_overall_status` updates to reflect Invariant 4 now being enforced.

---

## 12. Integration with `compute_rq3_metrics.py`

When the Phase 6 merge spec is written, it folds Track 2 output in:

```python
def _load_audit_integrity_subfile():
    chain_p = REPO_ROOT / "results/rq3_audit_chain_verification.json"
    schema_p = REPO_ROOT / "results/rq3_audit_schema_audit.json"

    block = {"_status": "pending", "_merged_at": None}
    if chain_p.exists() and schema_p.exists():
        chain = json.loads(chain_p.read_text())
        schema = json.loads(schema_p.read_text())

        # Determine combined status
        chain_ok = chain["headline"].get("chain_intact")
        schema_ok = schema["headline"].get("all_entries_pass_schema")

        if chain_ok and schema_ok:
            status = "complete"
        elif chain_ok is None or schema_ok is None:
            status = "partial — no audit data yet"
        else:
            status = "failed"

        block = {
            "_status": status,
            "_merged_at": datetime.now(timezone.utc).isoformat(),
            "chain_verification": chain["headline"],
            "schema_audit": schema["headline"],
            "n_breaks": chain["headline"].get("n_breaks", 0),
            "n_schema_failures": schema["headline"].get("n_entries_failing", 0),
            "tamper_evidence_claim": "tamper-evident (detection); not tamper-resistant (prevention)",
        }
    return block
```

In the aggregator: `out["audit_integrity"] = _load_audit_integrity_subfile()`.

---

## 13. Open questions to surface (DO NOT GUESS)

Claude Code must pause and ask:

1. **Phase 0 — canonical audit log location.** Likely `logs/llm_audit.jsonl` per RQ2 Compliance spec, but confirm. Phase 0 discovery checks 6 candidate paths.
2. **Phase 0 — existing `audit_logger.py`.** Does it exist? If yes, is its hashing compatible with `common/audit_canonicalization.py`? Determines whether Phase 2 is "import refactor" or "new module."
3. **Phase 0 — `mve_mode_used` exact values.** RQ3 schema says `A_llm | B_rule`. Confirm these are the literal strings in the data.
4. **Phase 1 — Windows compatibility.** Spec includes Windows file locking fallback. Confirm whether development environment is Linux-only (simplifies the audit_logger module).
5. **Phase 3 — preregistered_date for schema.** Set to the actual date the schema was locked. Defense-critical for proving the schema wasn't backfilled.

---

## 14. Defense talking points this enables

- **"Q: Audit log on filesystem isn't truly tamper-evident, is it?"** *(Senior reviewer's exact question)*
  *"Correct — file-based JSON is not WORM storage. We use SHA256 hash chain for **detectable** tampering: `analysis/verify_audit_log_integrity.py` walks the chain and reports any break with forensic context. The architecture explicitly distinguishes 'tamper-evident' (detection) from 'tamper-resistant' (prevention). Production deployment would route to SIEM-style audit store. The verification is reproducible — anyone with read access to `logs/llm_audit.jsonl` can run the verifier."*

- **"How do you know the chain implementation is correct?"**
  *"`common/audit_canonicalization.py` is the single source of truth for hash construction. Both the writer (`src/audit_logger.py`) and reader (`analysis/verify_audit_log_integrity.py`) import from it. `tests/test_audit_canonicalization.py` includes 10 unit tests covering key ordering invariance, ASCII normalization, deterministic hashing, chain link sensitivity, and tamper detection."*

- **"How do you handle entries that don't have LLM fields (Mode B)?"**
  *"The schema is conditional: Mode A entries require `llm_provider`, `llm_model_version`, `full_prompt`, `full_response`; Mode B entries don't. The schema auditor branches on `mve_mode_used`. The defense framing is that schema completeness is mode-relative — fields that didn't apply aren't artificially synthesized just to fit a uniform schema."*

- **"What if the audit log file is missing entirely?"**
  *"Three tests in `tests/test_step16_audit_integrity.py` handle this. If no log exists at all, the chain and schema tests skip (acceptable — no Mode A runs have occurred yet). But a separate test asserts that **if user study data is present** (`survey/study_responses_*.json` exists), then the audit log **must** exist. This catches the case where Mode A was used in a study but the logger silently failed."*

- **"How do you prevent the audit log itself from being silently rotated or replaced?"**
  *"The `_meta.log_sha256` field in `results/rq3_audit_chain_verification.json` records the file hash at verification time. Each verification run produces a new entry. A reviewer can compare across runs to detect file replacement. This is not cryptographic file-system protection — it's detection at the filesystem-state level."*

---

## 15. What this track deliberately does NOT do

- **Provide tamper-resistance.** The claim is tamper-evidence (detection). Production WORM/SIEM is out of scope.
- **Encrypt the audit log.** Hashes authenticate; they don't conceal. If PHI is in the log, encryption is a separate concern (covered by RQ2_COMPLIANCE_SPEC PHI flow control).
- **Cross-validate against external timestamps.** Each entry has its own timestamp, but we don't verify against an external time source (RFC 3161, trusted timestamps, etc.). Out of scope.
- **Support log rotation across multiple files.** Spec assumes single growing file. If you adopt daily rotation, the verifier needs extension — flagged as future work.

---

## End of spec

Implementation order: Phase 0 (discovery) → Phase 1 (canonicalization) → Phase 2 (logger: import or create) → Phase 3 (schema) → Phase 4 (chain verifier) → Phase 5 (schema auditor) → Phase 6 (CI gate). Phase 1 must complete before Phase 2; everything after is independent ordering.

After this track is implemented:
- Invariant 4 in Track 1 manifest flips from `pending` to `enforced`
- The "tamper-evident audit log" claim is defense-strong
- `results/rq3_audit_chain_verification.json` is the single reference for chain status
- `results/rq3_audit_schema_audit.json` is the single reference for schema status