"""Security monitoring — canary inputs, heartbeat, config integrity.

Provides 6 security methods suitable for research/pilot deployment:
  1. Config hash verification — detect tampered production.yaml
  2. Heartbeat monitoring — detect flow ingestion failure
  3. Canary inputs — periodic known-attack test vectors
  4. Calibration validation — test calibrator against known attacks
  5. Feedback rollback — revert calibration if degraded
  6. Dependency scanning — check for known vulnerabilities
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SecurityMonitor:
    """Monitors system integrity and detects security anomalies.

    Args:
        project_root: Project root path for config/data files.
    """

    def __init__(self, project_root: Path) -> None:
        self._root = project_root
        self._config_path = project_root / "config" / "production.yaml"
        self._config_hash: Optional[str] = None
        self._last_flow_time: float = 0.0
        self._heartbeat_timeout: float = 60.0  # seconds
        self._canary_results: List[Dict[str, Any]] = []
        self._calibration_snapshots: List[Dict[str, Any]] = []

        # Compute initial config hash
        if self._config_path.exists():
            self._config_hash = self._hash_file(self._config_path)
            logger.info("Config hash initialized: %s", self._config_hash[:12])

    # ═══════════════════════════════════════════════════════════════
    # Method 1: Config Hash Verification
    # ═══════════════════════════════════════════════════════════════

    def verify_config(self) -> Dict[str, Any]:
        """Check if production.yaml has been modified since startup."""
        if not self._config_path.exists():
            return {"valid": False, "reason": "Config file missing"}

        current = self._hash_file(self._config_path)
        if self._config_hash is None:
            self._config_hash = current
            return {"valid": True, "reason": "First check — hash stored"}

        if current != self._config_hash:
            logger.warning(
                "CONFIG TAMPERED: hash changed from %s to %s",
                self._config_hash[:12], current[:12],
            )
            return {
                "valid": False,
                "reason": "Config modified since startup",
                "expected": self._config_hash[:12],
                "actual": current[:12],
            }

        return {"valid": True, "hash": current[:12]}

    # ═══════════════════════════════════════════════════════════════
    # Method 2: Heartbeat Monitoring
    # ═══════════════════════════════════════════════════════════════

    def record_flow(self) -> None:
        """Record that a flow was received (call on every ingestion)."""
        self._last_flow_time = time.time()

    def check_heartbeat(self) -> Dict[str, Any]:
        """Check if flows are still being ingested.

        Returns alert if no flow received within heartbeat_timeout.
        """
        if self._last_flow_time == 0:
            return {"alive": True, "reason": "No flows yet (waiting for first)"}

        elapsed = time.time() - self._last_flow_time
        if elapsed > self._heartbeat_timeout:
            logger.warning(
                "HEARTBEAT FAILURE: No flow for %.0fs (timeout: %.0fs)",
                elapsed, self._heartbeat_timeout,
            )
            return {
                "alive": False,
                "reason": f"No flow received for {elapsed:.0f}s",
                "last_flow_seconds_ago": round(elapsed, 1),
                "timeout": self._heartbeat_timeout,
            }

        return {
            "alive": True,
            "last_flow_seconds_ago": round(elapsed, 1),
        }

    # ═══════════════════════════════════════════════════════════════
    # Method 3: Canary Inputs
    # ═══════════════════════════════════════════════════════════════

    def run_canary_test(
        self,
        inference_fn: Any,
        n_features: int = 24,
    ) -> Dict[str, Any]:
        """Feed known-attack test vectors through the pipeline.

        Verifies the model still detects known attack patterns.
        If it doesn't, model weights may be corrupted or replaced.

        Args:
            inference_fn: Function that takes (window, features, category)
                and returns a result dict with risk_level.
            n_features: Number of features.

        Returns:
            Dict with test results.
        """
        # Generate canary: high-anomaly features (all 3.0 z-scores)
        canary_attack = np.full((1, 20, n_features), 3.0, dtype=np.float32)
        canary_features = np.full(n_features, 3.0, dtype=np.float32)

        # Generate canary: zero features (should be normal)
        canary_normal = np.zeros((1, 20, n_features), dtype=np.float32)
        canary_normal_features = np.zeros(n_features, dtype=np.float32)

        results = {"tests": [], "passed": 0, "failed": 0}

        try:
            # Test 1: High-anomaly should NOT be NORMAL
            attack_result = inference_fn(
                window=canary_attack,
                raw_features=canary_features,
                attack_category="unknown",
            )
            attack_risk = attack_result.get("risk_level", "NORMAL")
            test1_pass = attack_risk != "NORMAL"
            results["tests"].append({
                "name": "High-anomaly detection",
                "expected": "NOT NORMAL",
                "actual": attack_risk,
                "passed": test1_pass,
            })

            # Test 2: Zero features should be NORMAL or LOW
            normal_result = inference_fn(
                window=canary_normal,
                raw_features=canary_normal_features,
                attack_category="normal",
            )
            normal_risk = normal_result.get("risk_level", "CRITICAL")
            test2_pass = normal_risk in ("NORMAL", "LOW")
            results["tests"].append({
                "name": "Zero-feature normality",
                "expected": "NORMAL or LOW",
                "actual": normal_risk,
                "passed": test2_pass,
            })

            results["passed"] = sum(1 for t in results["tests"] if t["passed"])
            results["failed"] = sum(1 for t in results["tests"] if not t["passed"])

            if results["failed"] > 0:
                logger.error(
                    "CANARY FAILED: %d/%d tests failed — model integrity compromised?",
                    results["failed"], len(results["tests"]),
                )
            else:
                logger.info("Canary test passed: %d/%d", results["passed"], len(results["tests"]))

        except Exception as exc:
            results["error"] = str(exc)
            results["failed"] = 1
            logger.error("Canary test error: %s", exc)

        self._canary_results.append(results)
        return results

    # ═══════════════════════════════════════════════════════════════
    # Method 4: Calibration Validation
    # ═══════════════════════════════════════════════════════════════

    def validate_calibration(
        self,
        calibrator: Any,
        known_attack_scores: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Test calibrator against known attack scores.

        Verifies that known-high scores still map to MEDIUM+ risk.

        Args:
            calibrator: ScoreCalibrator instance.
            known_attack_scores: List of scores that SHOULD be detected.
                Defaults to [0.90, 0.93, 0.95, 0.97, 0.99].

        Returns:
            Validation result with pass/fail.
        """
        if known_attack_scores is None:
            known_attack_scores = [0.90, 0.93, 0.95, 0.97, 0.99]

        detected_levels = {"MEDIUM", "HIGH", "CRITICAL"}
        passed = 0
        failed = 0
        details = []

        for score in known_attack_scores:
            result = calibrator.calibrate(score)
            risk = result.get("risk_level", "NORMAL")
            is_detected = risk in detected_levels
            details.append({
                "score": score,
                "risk_level": risk,
                "detected": is_detected,
            })
            if is_detected:
                passed += 1
            else:
                failed += 1

        detection_rate = passed / max(len(known_attack_scores), 1)
        valid = detection_rate >= 0.8  # At least 80% of known attacks caught

        if not valid:
            logger.warning(
                "CALIBRATION VALIDATION FAILED: %.0f%% detection rate (need ≥80%%)",
                detection_rate * 100,
            )

        return {
            "valid": valid,
            "detection_rate": round(detection_rate, 2),
            "passed": passed,
            "failed": failed,
            "details": details,
        }

    # ═══════════════════════════════════════════════════════════════
    # Method 5: Feedback Rollback
    # ═══════════════════════════════════════════════════════════════

    def snapshot_calibration(self, calibrator: Any) -> None:
        """Save current calibrator state for potential rollback."""
        config = calibrator.get_config()
        self._calibration_snapshots.append({
            "time": time.time(),
            "config": config,
        })
        # Keep last 5 snapshots
        if len(self._calibration_snapshots) > 5:
            self._calibration_snapshots = self._calibration_snapshots[-5:]

    def should_rollback(
        self,
        calibrator: Any,
        known_attack_scores: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Check if calibration should be rolled back.

        Tests current calibrator against known attacks. If detection
        rate dropped below 60%, recommend rollback.
        """
        validation = self.validate_calibration(calibrator, known_attack_scores)

        if validation["valid"]:
            return {"rollback": False, "reason": "Calibration is healthy"}

        if not self._calibration_snapshots:
            return {
                "rollback": False,
                "reason": "No previous snapshot to rollback to",
                "current_detection_rate": validation["detection_rate"],
            }

        return {
            "rollback": True,
            "reason": f"Detection rate {validation['detection_rate']:.0%} < 80% — rollback recommended",
            "snapshots_available": len(self._calibration_snapshots),
            "current_detection_rate": validation["detection_rate"],
        }

    # ═══════════════════════════════════════════════════════════════
    # Method 6: Dependency Scanning
    # ═══════════════════════════════════════════════════════════════

    @staticmethod
    def scan_dependencies() -> Dict[str, Any]:
        """Check installed packages for known vulnerabilities.

        Uses pip-audit if available, otherwise returns advisory.
        """
        try:
            import subprocess
            result = subprocess.run(
                ["pip-audit", "--format", "json", "--progress-spinner", "off"],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0:
                import json
                vulns = json.loads(result.stdout)
                return {
                    "scanned": True,
                    "vulnerabilities": len(vulns),
                    "details": vulns[:10],
                    "status": "CLEAN" if len(vulns) == 0 else "VULNERABLE",
                }
            return {
                "scanned": True,
                "vulnerabilities": -1,
                "status": "SCAN_ERROR",
                "error": result.stderr[:200],
            }
        except FileNotFoundError:
            return {
                "scanned": False,
                "status": "TOOL_MISSING",
                "recommendation": "Install pip-audit: pip install pip-audit",
            }
        except Exception as exc:
            return {"scanned": False, "status": "ERROR", "error": str(exc)}

    # ═══════════════════════════════════════════════════════════════
    # Combined Status
    # ═══════════════════════════════════════════════════════════════

    def get_security_status(self) -> Dict[str, Any]:
        """Get overall security status."""
        config = self.verify_config()
        heartbeat = self.check_heartbeat()

        issues = []
        if not config.get("valid", True):
            issues.append(f"Config: {config.get('reason', 'unknown')}")
        if not heartbeat.get("alive", True):
            issues.append(f"Heartbeat: {heartbeat.get('reason', 'unknown')}")

        last_canary = self._canary_results[-1] if self._canary_results else None
        if last_canary and last_canary.get("failed", 0) > 0:
            issues.append("Canary test failed — model integrity check needed")

        return {
            "status": "SECURE" if not issues else "WARNING",
            "issues": issues,
            "config": config,
            "heartbeat": heartbeat,
            "canary_tests_run": len(self._canary_results),
            "calibration_snapshots": len(self._calibration_snapshots),
        }

    @staticmethod
    def _hash_file(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
