"""Module 5 coverage boost — loaders, worked_examples, retention, verify, CLI."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from common.risk_scores_loader import RiskScoresArtefact
from module5_responses.audit.logger import AuditLogger
from module5_responses.audit.retention import rotate_and_purge
from module5_responses.audit.verify import verify_audit_log
from module5_responses.loaders import (
    _paths,
    load_attack_categories,
    load_explanations,
    load_risk_scores,
)
from module5_responses.pipeline_cli import _strict_json_default
from module5_responses.worked_examples import run_worked_examples


# ── loaders ────────────────────────────────────────────────────────────


def test_paths_test_split_no_suffix():
    p = _paths("test")
    assert p["suffix"] == ""
    assert p["out_alert_responses"].name == "alert_responses.json"


def test_paths_demo_split_suffix():
    p = _paths("demo")
    assert p["suffix"] == "_demo"
    assert p["out_alert_responses"].name == "alert_responses_demo.json"


def test_paths_unknown_split_raises():
    with pytest.raises(ValueError, match="not a valid Split"):
        _paths("staging")


def test_load_risk_scores(monkeypatch, tmp_path):
    def fake_verified_load(path: Path) -> RiskScoresArtefact:
        assert path == tmp_path / "scores.npz"
        return RiskScoresArtefact(
            R=np.array([0.1, 0.5, 0.9]),
            c_detect=np.array([0.1, 0.5, 0.9]),
            c_track_a=np.array([0.1, 0.5, 0.9]),
            c_track_b=np.array([0.0, 0.0, 0.0]),
            d_crit=np.array([0.2, 0.2, 0.2]),
            s_data=np.array([0.3, 0.3, 0.3]),
            d_clinical_tier=np.array([0.4, 0.4, 0.4]),
            y_true=np.array([0, 1, 1]),
            risk_level_codes=np.array([1, 2, 3], dtype=np.int8),
            risk_levels=np.array(["LOW", "MEDIUM", "HIGH"]),
            schema_version="2.0",
            formula_version="v2",
        )

    monkeypatch.setattr("common.risk_scores_loader.load_risk_scores", fake_verified_load)

    out = load_risk_scores(tmp_path / "scores.npz")
    assert "R" in out
    np.testing.assert_array_almost_equal(out["R"], [0.1, 0.5, 0.9])


def test_load_explanations_both_present(tmp_path):
    a = tmp_path / "a.json"
    c = tmp_path / "c.json"
    a.write_text(json.dumps([{"sample_index": 0, "x": 1}]))
    c.write_text(json.dumps([{"sample_index": 1, "summary": "ok"}]))
    analyst, clin = load_explanations(a, c)
    assert analyst[0]["x"] == 1
    assert clin[1]["summary"] == "ok"


def test_load_explanations_both_missing(tmp_path):
    analyst, clin = load_explanations(tmp_path / "nope_a.json", tmp_path / "nope_c.json")
    assert analyst == {}
    assert clin == {}


def test_load_attack_categories(tmp_path):
    parquet = tmp_path / "test.parquet"
    pd.DataFrame({"Attack Category": ["Spoofing", "normal", "Data Alteration"]}).to_parquet(parquet)
    out = load_attack_categories(parquet)
    assert list(out) == ["Spoofing", "normal", "Data Alteration"]


# ── strict JSON default for worked_examples (Y1) ───────────────────────


def test_strict_json_default_datetime():
    s = _strict_json_default(datetime(2026, 5, 1, 12, 0, 0))
    assert "2026-05-01" in s


def test_strict_json_default_numpy_int():
    out = _strict_json_default(np.int64(7))
    assert out == 7
    assert isinstance(out, int)


def test_strict_json_default_numpy_float():
    out = _strict_json_default(np.float64(3.5))
    assert out == 3.5
    assert isinstance(out, float)


def test_strict_json_default_numpy_array():
    out = _strict_json_default(np.array([1, 2, 3]))
    assert out == [1, 2, 3]


def test_strict_json_default_unknown_type_raises():
    class Weird:
        pass
    with pytest.raises(TypeError, match="non-JSON-serialisable"):
        _strict_json_default(Weird())


# ── worked_examples smoke ──────────────────────────────────────────────


def test_run_worked_examples_returns_three_scenarios():
    n = 12
    risk_data = {
        "R": np.array([0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05, 0.85, 0.5]),
        "risk_levels": np.array(["CRITICAL", "CRITICAL", "HIGH", "HIGH", "MEDIUM",
                                  "MEDIUM", "LOW", "LOW", "LOW", "LOW", "HIGH", "MEDIUM"]),
        "y_true": np.array([1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1]),
        "c_detect": np.linspace(0, 0.9, n),
        "d_crit": np.linspace(0, 0.9, n),
        "s_data": np.linspace(0, 0.9, n),
        "d_clinical_tier": np.linspace(0, 0.9, n),
    }
    attack_cats = np.array(["Spoofing"] * 6 + ["Data Alteration"] * 4 + ["normal"] * 2)
    scenarios = run_worked_examples(risk_data, attack_cats, {}, {})
    # 3 tiers (CRITICAL, HIGH, LOW) all have samples present.
    assert len(scenarios) == 3
    tiers = {s["risk_level"] for s in scenarios}
    assert tiers == {"CRITICAL", "HIGH", "LOW"}


def test_run_worked_examples_skips_missing_tier():
    n = 4
    risk_data = {
        "R": np.array([0.95, 0.9, 0.85, 0.8]),
        "risk_levels": np.array(["CRITICAL", "CRITICAL", "CRITICAL", "CRITICAL"]),
        "y_true": np.array([1, 1, 1, 0]),
        "c_detect": np.zeros(n), "d_crit": np.zeros(n),
        "s_data": np.zeros(n), "d_clinical_tier": np.zeros(n),
    }
    attack_cats = np.array(["Spoofing"] * 4)
    scenarios = run_worked_examples(risk_data, attack_cats, {}, {})
    # Only CRITICAL present.
    assert len(scenarios) == 1
    assert scenarios[0]["risk_level"] == "CRITICAL"


# ── audit retention/rotation ───────────────────────────────────────────


def _make_logger(tmp_path):
    from module5_responses.audit import signing as signing_mod

    signing_mod._artefacts_present = lambda: []
    signing_mod._read_pinned_key_id = lambda: None
    return AuditLogger(
        tmp_path / "audit.jsonl",
        signing_key_path=tmp_path / "priv.pem",
        public_key_path=tmp_path / "pub.pem",
        retention_days=365,
    )


def test_rotate_skips_when_oldest_within_retention(tmp_path):
    al = _make_logger(tmp_path)
    al.log({"event": "today", "timestamp": datetime.now(timezone.utc).isoformat()})
    report = rotate_and_purge(al, retention_days=365, archive_dir=tmp_path / "arch")
    assert report["rotated"] is False


def test_rotate_archives_when_oldest_beyond_retention(tmp_path):
    al = _make_logger(tmp_path)
    # Inject an "old" record by writing directly through al.log with old ts.
    al.log({"event": "ancient", "timestamp": (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()})
    report = rotate_and_purge(al, retention_days=30, archive_dir=tmp_path / "arch")
    assert report["rotated"] is True
    assert report["archived_path"] is not None
    archived = list((tmp_path / "arch").glob("audit.*.jsonl"))
    assert len(archived) == 1
    manifest = list((tmp_path / "arch").glob("audit.*.manifest.json"))
    assert len(manifest) == 1


def test_rotate_refuses_tampered_log(tmp_path):
    al = _make_logger(tmp_path)
    al.log({"event": "ok", "timestamp": (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()})
    al.log({"event": "ok2", "timestamp": (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()})
    # Tamper line 2.
    lines = al.path.read_text().strip().split("\n")
    rec = json.loads(lines[1])
    rec["event"] = "TAMPERED"
    lines[1] = json.dumps(rec)
    al.path.write_text("\n".join(lines) + "\n")

    report = rotate_and_purge(al, retention_days=30, archive_dir=tmp_path / "arch")
    assert report["rotated"] is False
    assert "tampered" in report["reason"]


# ── audit verify strict mode + legacy migration ────────────────────────


def test_verify_strict_unsigned_breaks(tmp_path):
    al = AuditLogger(
        tmp_path / "ns.jsonl",
        signing_key_path=tmp_path / "k.pem",
        public_key_path=tmp_path / "kpub.pem",
        sign=False,
    )
    al.log({"event": "unsigned"})
    report = verify_audit_log(al.path, legacy_ok=False)
    assert report["first_break_at"] == 1
    assert any("unsigned" in b["reason"] for b in report["broken"])


def test_verify_legacy_migration_accepts_chain_restart(tmp_path):
    """Legacy AuditLogger reset chain to genesis on every process start.
    With legacy_ok=True, the verifier tolerates a fresh genesis after line 1.
    """
    path = tmp_path / "legacy.jsonl"
    # Manually craft two legacy (unsigned) records, both with prev_hash="0"*64.
    import hashlib
    from module5_responses.audit.signing import _canonical_json

    rec1 = {"event": "first", "prev_hash": "0" * 64}
    rec1["integrity_hash"] = hashlib.sha256(_canonical_json(rec1)).hexdigest()

    rec2 = {"event": "after_restart", "prev_hash": "0" * 64}  # genesis again!
    rec2["integrity_hash"] = hashlib.sha256(_canonical_json(rec2)).hexdigest()

    path.write_text(json.dumps(rec1) + "\n" + json.dumps(rec2) + "\n")
    report = verify_audit_log(path, legacy_ok=True)
    assert report["first_break_at"] is None
    assert report["legacy_chain_restarts"] == 1


def test_verify_bad_json_breaks(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text("not json\n")
    report = verify_audit_log(path)
    assert report["first_break_at"] == 1
    assert any("json parse" in b["reason"] for b in report["broken"])


def test_verify_chain_break_on_mismatched_prev_hash(tmp_path):
    import hashlib
    from module5_responses.audit.signing import _canonical_json

    path = tmp_path / "broken.jsonl"
    rec1 = {"event": "a", "prev_hash": "0" * 64}
    rec1["integrity_hash"] = hashlib.sha256(_canonical_json(rec1)).hexdigest()
    # Wrong prev_hash AND not "0"*64, so legacy migration cannot rescue it.
    rec2 = {"event": "b", "prev_hash": "f" * 64}
    rec2["integrity_hash"] = hashlib.sha256(_canonical_json(rec2)).hexdigest()
    path.write_text(json.dumps(rec1) + "\n" + json.dumps(rec2) + "\n")
    report = verify_audit_log(path, legacy_ok=True)
    assert report["first_break_at"] == 2
    assert any("prev_hash mismatch" in b["reason"] for b in report["broken"])


# ── __main__ dispatcher (Y5) ───────────────────────────────────────────


def test_main_dispatcher_routes_verify_audit_log(monkeypatch, tmp_path):
    audit_log = tmp_path / "a.jsonl"
    audit_log.write_text("")  # empty file
    called = {}

    def fake_cli_entry():
        called["argv"] = list(sys.argv)

    monkeypatch.setattr("module5_responses.pipeline_cli.cli_entry", fake_cli_entry)
    monkeypatch.setattr(sys, "argv", ["m", "verify-audit-log", "--path", str(audit_log)])
    from module5_responses.__main__ import main
    main()
    assert "--verify-audit-log" in called["argv"]
    assert str(audit_log) in called["argv"]


def test_main_dispatcher_routes_worked_examples(monkeypatch):
    called = {}

    def fake_main():
        called["fired"] = True

    monkeypatch.setattr("module5_responses.pipeline_cli.main", fake_main)
    monkeypatch.setattr(sys, "argv", ["m", "worked-examples"])
    from module5_responses.__main__ import main
    main()
    assert called.get("fired") is True


def test_main_dispatcher_default_routes_responses_cli(monkeypatch):
    called = {}

    def fake_main():
        called["fired"] = True

    monkeypatch.setattr("module5_responses.responses_cli.main", fake_main)
    monkeypatch.setattr(sys, "argv", ["m"])
    from module5_responses.__main__ import main
    main()
    assert called.get("fired") is True


def test_pipeline_cli_uses_shared_loaders(monkeypatch, tmp_path):
    import module5_responses.pipeline_cli as cli

    risk_data = {
        "R": np.array([0.8]),
        "risk_levels": np.array(["HIGH"]),
        "y_true": np.array([1]),
        "d_clinical_tier": np.array([0.7]),
    }
    analyst = {0: {"sample_index": 0}}
    clinician = {0: {"sample_index": 0, "summary": "ok"}}
    attack_cats = np.array(["Spoofing"])
    scenarios = [{"risk_level": "HIGH"}]
    feedback_output = {
        "true_positives": 1,
        "false_positives": 0,
        "false_negatives": 0,
        "fpr": 0.0,
        "fnr": 0.0,
        "current_thresholds": {},
        "suggested_threshold_change": {},
        "adjustments": [],
    }

    monkeypatch.setattr(cli, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(cli, "load_risk_scores", lambda path: risk_data)
    monkeypatch.setattr(cli, "load_explanations", lambda a, c: (analyst, clinician))
    monkeypatch.setattr(cli, "load_attack_categories", lambda path: attack_cats)
    monkeypatch.setattr(cli, "export_response_policy", lambda: None)
    monkeypatch.setattr(cli, "run_worked_examples", lambda *args: scenarios)

    class FakePolicyEngine:
        def recommend(self, *args):
            return {"actions": ["notify"]}

    class FakeExecutor:
        def execute(self, *args):
            return {"event": "ok"}

    class FakeAudit:
        def __init__(self, path):
            self.path = path

        def log(self, _record):
            return None

    class FakeFeedback:
        def record(self, *args):
            return None

        def compute_adjustments(self):
            return feedback_output

    class FakeNotifier:
        def __init__(self):
            self.notifications = []

    monkeypatch.setattr(cli, "PolicyEngine", FakePolicyEngine)
    monkeypatch.setattr(cli, "ActionExecutor", FakeExecutor)
    monkeypatch.setattr(cli, "AuditLogger", FakeAudit)
    monkeypatch.setattr(cli, "FeedbackLoop", FakeFeedback)
    monkeypatch.setattr(cli, "NotificationService", FakeNotifier)

    cli.main()

    assert (tmp_path / "worked_examples.json").exists()
    assert (tmp_path / "feedback_analysis.json").exists()


# ── policy.export_response_policy artifact write ───────────────────────


def test_export_response_policy_writes_artifact(tmp_path):
    from module5_responses.policy import export_response_policy
    out = export_response_policy(tmp_path / "policy.json")
    assert out.exists()
    payload = json.loads(out.read_text())
    assert payload["version"] == "2.0"
    assert "CRITICAL" in payload["tier_policies"]
