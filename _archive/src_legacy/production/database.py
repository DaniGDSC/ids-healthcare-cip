"""SQLite persistence for IoMT IDS production data.

Stores predictions, alerts, access logs, feedback, and calibration
snapshots in a local SQLite database. Thread-safe for concurrent
reads/writes from simulator thread + dashboard thread.

Database file: data/production/iomt_ids.db (configurable).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    sample_index INTEGER,
    anomaly_score REAL,
    risk_level TEXT,
    clinical_severity INTEGER,
    attention_flag INTEGER DEFAULT 0,
    patient_safety_flag INTEGER DEFAULT 0,
    device_id TEXT,
    device_action TEXT,
    latency_ms REAL,
    ground_truth INTEGER DEFAULT -1,
    inference_failed INTEGER DEFAULT 0,
    explanation TEXT
);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    sample_index INTEGER,
    risk_level TEXT NOT NULL,
    clinical_severity INTEGER NOT NULL,
    patient_safety_flag INTEGER DEFAULT 0,
    device_id_hash TEXT,
    cia_scores TEXT,
    alert_emit INTEGER DEFAULT 1,
    alert_reason TEXT,
    acknowledged INTEGER DEFAULT 0,
    acknowledged_by TEXT,
    acknowledged_at TEXT
);

CREATE TABLE IF NOT EXISTS access_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    username TEXT NOT NULL,
    role TEXT NOT NULL,
    action TEXT NOT NULL,
    resource_id TEXT,
    ip_address TEXT
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    alert_id INTEGER,
    analyst_hash TEXT NOT NULL,
    ground_truth INTEGER NOT NULL,
    confidence REAL DEFAULT 0.8,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS calibration_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    time TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    mode TEXT NOT NULL,
    config TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_predictions_time ON predictions(time);
CREATE INDEX IF NOT EXISTS idx_predictions_risk ON predictions(risk_level);
CREATE INDEX IF NOT EXISTS idx_alerts_time ON alerts(time);
CREATE INDEX IF NOT EXISTS idx_alerts_risk ON alerts(risk_level);
CREATE INDEX IF NOT EXISTS idx_access_time ON access_log(time);
"""


class Database:
    """SQLite persistence for IoMT IDS production data.

    Thread-safe: all operations acquire a lock before executing.
    Uses WAL mode for concurrent read/write performance.

    Args:
        db_path: Path to SQLite database file.
    """

    def __init__(self, db_path: str | Path = "data/production/iomt_ids.db") -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        logger.info("Database initialized: %s", self._path)

    # ── Inserts ────────────────────────────────────────────────────────

    def insert_prediction(self, p: Dict[str, Any]) -> int:
        """Insert a prediction record. Returns row id."""
        explanation = p.get("explanation")
        exp_json = json.dumps(explanation) if explanation else None

        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO predictions
                   (sample_index, anomaly_score, risk_level, clinical_severity,
                    attention_flag, patient_safety_flag, device_id, device_action,
                    latency_ms, ground_truth, inference_failed, explanation)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    p.get("sample_index"),
                    p.get("anomaly_score"),
                    p.get("risk_level"),
                    p.get("clinical_severity"),
                    int(p.get("attention_flag", False)),
                    int(p.get("patient_safety_flag", False)),
                    p.get("device_id", ""),
                    p.get("device_action", "none"),
                    p.get("latency_ms"),
                    p.get("ground_truth", -1),
                    int(p.get("inference_failed", False)),
                    exp_json,
                ),
            )
            self._conn.commit()
            return cur.lastrowid

    def insert_alert(self, a: Dict[str, Any]) -> int:
        """Insert an alert record. Returns row id."""
        cia = a.get("cia_scores")
        cia_json = json.dumps(cia) if cia else None
        device_hash = ""
        device_id = a.get("device_id", "")
        if device_id:
            device_hash = hashlib.sha256(device_id.encode()).hexdigest()[:12]

        with self._lock:
            cur = self._conn.execute(
                """INSERT INTO alerts
                   (sample_index, risk_level, clinical_severity,
                    patient_safety_flag, device_id_hash, cia_scores,
                    alert_emit, alert_reason)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (
                    a.get("sample_index"),
                    a.get("risk_level"),
                    a.get("clinical_severity"),
                    int(a.get("patient_safety_flag", False)),
                    device_hash,
                    cia_json,
                    int(a.get("alert_emit", True)),
                    a.get("alert_reason", ""),
                ),
            )
            self._conn.commit()
            return cur.lastrowid

    def insert_access(
        self,
        username: str,
        role: str,
        action: str,
        resource_id: Optional[str] = None,
    ) -> None:
        """Log a dashboard access event (HIPAA audit)."""
        with self._lock:
            self._conn.execute(
                """INSERT INTO access_log (username, role, action, resource_id)
                   VALUES (?,?,?,?)""",
                (username, role, action, resource_id),
            )
            self._conn.commit()

    def insert_feedback(
        self,
        alert_id: int,
        analyst: str,
        ground_truth: int,
        confidence: float = 0.8,
        notes: str = "",
    ) -> None:
        """Record analyst TP/FP feedback on an alert."""
        analyst_hash = hashlib.sha256(analyst.encode()).hexdigest()[:12]
        with self._lock:
            self._conn.execute(
                """INSERT INTO feedback
                   (alert_id, analyst_hash, ground_truth, confidence, notes)
                   VALUES (?,?,?,?,?)""",
                (alert_id, analyst_hash, ground_truth, confidence, notes),
            )
            self._conn.commit()

    def insert_calibration(self, config: Dict[str, Any]) -> None:
        """Snapshot calibration state for audit trail."""
        mode = config.get("mode", "unknown")
        with self._lock:
            self._conn.execute(
                "INSERT INTO calibration_snapshots (mode, config) VALUES (?,?)",
                (mode, json.dumps(config)),
            )
            self._conn.commit()

    # ── Queries ────────────────────────────────────────────────────────

    def query_alerts(
        self,
        since: Optional[str] = None,
        until: Optional[str] = None,
        risk_level: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query alerts with optional filters."""
        sql = "SELECT * FROM alerts WHERE 1=1"
        params: list = []
        if since:
            sql += " AND time >= ?"
            params.append(since)
        if until:
            sql += " AND time <= ?"
            params.append(until)
        if risk_level:
            sql += " AND risk_level = ?"
            params.append(risk_level)
        sql += " ORDER BY time DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def query_predictions(
        self,
        since: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Query recent predictions."""
        sql = "SELECT * FROM predictions WHERE 1=1"
        params: list = []
        if since:
            sql += " AND time >= ?"
            params.append(since)
        sql += " ORDER BY time DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def query_access_log(
        self,
        username: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Query access log entries."""
        sql = "SELECT * FROM access_log WHERE 1=1"
        params: list = []
        if username:
            sql += " AND username = ?"
            params.append(username)
        sql += " ORDER BY time DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def get_risk_distribution(self, since: Optional[str] = None) -> Dict[str, int]:
        """Get cumulative risk level counts."""
        sql = "SELECT risk_level, COUNT(*) as cnt FROM predictions"
        params: list = []
        if since:
            sql += " WHERE time >= ?"
            params.append(since)
        sql += " GROUP BY risk_level"

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return {r["risk_level"]: r["cnt"] for r in rows}

    def get_alert_count(self, since: Optional[str] = None) -> int:
        """Count alerts."""
        sql = "SELECT COUNT(*) as cnt FROM alerts"
        params: list = []
        if since:
            sql += " WHERE time >= ?"
            params.append(since)

        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
        return row["cnt"] if row else 0

    def acknowledge_alert(self, alert_id: int, user: str) -> bool:
        """Mark an alert as acknowledged."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cur = self._conn.execute(
                """UPDATE alerts SET acknowledged=1, acknowledged_by=?,
                   acknowledged_at=? WHERE id=?""",
                (user, now, alert_id),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def get_prediction_count(self) -> int:
        """Total predictions stored."""
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) as cnt FROM predictions").fetchone()
        return row["cnt"] if row else 0

    def get_overridden_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get alerts marked safe by clinical staff (clinical overrides)."""
        sql = """SELECT a.*, f.analyst_hash, f.time as override_time
                 FROM alerts a JOIN feedback f ON a.id = f.alert_id
                 WHERE f.ground_truth = 0
                 ORDER BY f.time DESC LIMIT ?"""
        with self._lock:
            rows = self._conn.execute(sql, (limit,)).fetchall()
        return [dict(r) for r in rows]

    # ── Feedback queries (Human-in-the-Loop) ─────────────────────────

    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get feedback statistics for dashboard display."""
        with self._lock:
            total = self._conn.execute("SELECT COUNT(*) as cnt FROM feedback").fetchone()["cnt"]
            tp = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM feedback WHERE ground_truth = 1"
            ).fetchone()["cnt"]
            fp = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM feedback WHERE ground_truth = 0"
            ).fetchone()["cnt"]
        return {
            "total_feedback": total,
            "confirmed_attacks": tp,
            "marked_safe": fp,
            "tp_rate": round(tp / max(total, 1), 3),
            "fp_rate": round(fp / max(total, 1), 3),
        }

    def get_feedback_by_analyst(self) -> List[Dict[str, Any]]:
        """Get per-analyst feedback statistics for quality tracking."""
        sql = """SELECT analyst_hash,
                        COUNT(*) as total,
                        SUM(CASE WHEN ground_truth = 1 THEN 1 ELSE 0 END) as tp_count,
                        SUM(CASE WHEN ground_truth = 0 THEN 1 ELSE 0 END) as fp_count,
                        ROUND(AVG(confidence), 2) as avg_confidence
                 FROM feedback
                 GROUP BY analyst_hash
                 ORDER BY total DESC"""
        with self._lock:
            rows = self._conn.execute(sql).fetchall()
        return [dict(r) for r in rows]

    def get_feedback_disagreements(self) -> List[Dict[str, Any]]:
        """Find alerts where analysts disagree (one says TP, another FP)."""
        sql = """SELECT f1.alert_id,
                        f1.analyst_hash as analyst_a,
                        f1.ground_truth as verdict_a,
                        f2.analyst_hash as analyst_b,
                        f2.ground_truth as verdict_b
                 FROM feedback f1
                 JOIN feedback f2 ON f1.alert_id = f2.alert_id
                 WHERE f1.analyst_hash < f2.analyst_hash
                   AND f1.ground_truth != f2.ground_truth
                 ORDER BY f1.alert_id DESC
                 LIMIT 50"""
        with self._lock:
            rows = self._conn.execute(sql).fetchall()
        return [dict(r) for r in rows]

    def get_feedback_for_recalibration(self, limit: int = 500) -> List[Dict[str, Any]]:
        """Get feedback entries with prediction scores for recalibration."""
        sql = """SELECT p.anomaly_score, f.ground_truth, f.confidence
                 FROM feedback f
                 JOIN predictions p ON p.sample_index = (
                     SELECT a.sample_index FROM alerts a WHERE a.id = f.alert_id
                 )
                 WHERE f.ground_truth IN (0, 1)
                 ORDER BY f.time DESC
                 LIMIT ?"""
        with self._lock:
            rows = self._conn.execute(sql, (limit,)).fetchall()
        return [dict(r) for r in rows]

    def get_feedback_count_since(self, since_id: int = 0) -> int:
        """Count feedback entries since a given ID (for auto-trigger)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) as cnt FROM feedback WHERE id > ?", (since_id,),
            ).fetchone()
        return row["cnt"] if row else 0

    def purge_old_data(self, retention_days: int = 90) -> Dict[str, int]:
        """Delete records older than retention_days.

        Unacknowledged alerts are NOT purged (safety requirement).

        Returns:
            Dict with count of purged records per table.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
        with self._lock:
            pred = self._conn.execute(
                "DELETE FROM predictions WHERE time < ?", (cutoff,),
            ).rowcount
            alerts = self._conn.execute(
                "DELETE FROM alerts WHERE time < ? AND acknowledged = 1", (cutoff,),
            ).rowcount
            access = self._conn.execute(
                "DELETE FROM access_log WHERE time < ?", (cutoff,),
            ).rowcount
            self._conn.commit()
        logger.info(
            "Purged data older than %d days: %d predictions, %d alerts, %d access logs",
            retention_days, pred, alerts, access,
        )
        return {"predictions": pred, "alerts": alerts, "access_log": access}

    def close(self) -> None:
        """Close database connection."""
        self._conn.close()
