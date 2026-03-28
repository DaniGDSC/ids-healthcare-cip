"""Explainability (XAI) Validation Tests.

Three categories:
  1. Perturbation Analysis — masking top features should reduce prediction
  2. Explanation Stability — same input produces same explanation
  3. Role-Based View Consistency — each stakeholder gets appropriate content
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest
import tensorflow as tf


# ── Helpers ────────────────────────────────────────────────────────

def _build_mock_model(n_features: int = 24, timesteps: int = 5) -> tf.keras.Model:
    """Build a minimal CNN model with attention-like structure for testing."""
    inp = tf.keras.Input(shape=(timesteps, n_features), name="input")
    x = tf.keras.layers.Conv1D(8, 3, activation="relu", padding="same", name="conv1")(inp)
    x = tf.keras.layers.GlobalAveragePooling1D(name="attention")(x)
    x = tf.keras.layers.Dense(4, activation="relu", name="dense_head")(x)
    x = tf.keras.layers.Dropout(0.1, name="drop_head")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="output")(x)
    model = tf.keras.Model(inp, out, name="test_model")
    model.compile(optimizer="adam", loss="binary_crossentropy")

    # Train enough so gradients are non-trivial
    np.random.seed(42)
    X_train = np.random.randn(100, timesteps, n_features).astype(np.float32)
    y_train = np.concatenate([np.zeros(50), np.ones(50)]).astype(np.float32)
    model.fit(X_train, y_train, epochs=10, verbose=0)
    return model


N_FEATURES = 24
TIMESTEPS = 5
FEATURE_NAMES = [
    "SrcBytes", "DstBytes", "SrcLoad", "DstLoad",
    "SIntPkt", "DIntPkt", "SIntPktAct",
    "sMaxPktSz", "dMaxPktSz", "sMinPktSz",
    "Dur", "TotBytes", "Load", "pSrcLoss", "pDstLoss", "Packet_num",
    "Temp", "SpO2", "Pulse_Rate", "SYS", "DIA", "Heart_rate",
    "Resp_Rate", "ST",
]


@pytest.fixture(scope="module")
def model() -> tf.keras.Model:
    """Shared trained mock model for all XAI tests."""
    tf.random.set_seed(42)
    np.random.seed(42)
    return _build_mock_model(N_FEATURES, TIMESTEPS)


@pytest.fixture
def sample() -> np.ndarray:
    """Single test sample (1, timesteps, features)."""
    np.random.seed(123)
    return np.random.randn(1, TIMESTEPS, N_FEATURES).astype(np.float32)


@pytest.fixture
def batch() -> np.ndarray:
    """Batch of test samples (10, timesteps, features)."""
    np.random.seed(456)
    return np.random.randn(10, TIMESTEPS, N_FEATURES).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════
# 1. PERTURBATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════

class TestPerturbationAnalysis:
    """Verify that explanations are faithful to the model.

    If a feature is identified as important, masking it should
    change the prediction. If it doesn't, the explanation is
    unfaithful (decorative, not diagnostic).
    """

    def test_masking_top_feature_changes_prediction(
        self, model: tf.keras.Model, sample: np.ndarray,
    ) -> None:
        """Zeroing the top gradient-attributed feature should shift prediction."""
        # Get gradient-based importance
        sample_tensor = tf.constant(sample, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(sample_tensor)
            pred_original = model(sample_tensor, training=False)
        grads = tape.gradient(pred_original, sample_tensor)
        assert grads is not None

        feat_importance = np.mean(np.abs(grads.numpy().squeeze()), axis=0)
        top_idx = int(np.argmax(feat_importance))

        # Mask the top feature
        perturbed = sample.copy()
        perturbed[:, :, top_idx] = 0.0
        pred_perturbed = model.predict(perturbed, verbose=0)

        original_score = float(pred_original.numpy().ravel()[0])
        perturbed_score = float(pred_perturbed.ravel()[0])

        # Prediction should change (we don't know direction, just magnitude)
        assert abs(original_score - perturbed_score) > 1e-6, (
            f"Masking top feature {FEATURE_NAMES[top_idx]} did not change prediction: "
            f"{original_score:.6f} vs {perturbed_score:.6f}"
        )

    def test_masking_bottom_feature_changes_less(
        self, model: tf.keras.Model, sample: np.ndarray,
    ) -> None:
        """Masking the least important feature should change prediction less
        than masking the most important feature."""
        sample_tensor = tf.constant(sample, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(sample_tensor)
            pred_original = model(sample_tensor, training=False)
        grads = tape.gradient(pred_original, sample_tensor)
        feat_importance = np.mean(np.abs(grads.numpy().squeeze()), axis=0)

        top_idx = int(np.argmax(feat_importance))
        bottom_idx = int(np.argmin(feat_importance))

        if top_idx == bottom_idx:
            pytest.skip("All features have equal importance")

        original_score = float(pred_original.numpy().ravel()[0])

        # Mask top
        perturbed_top = sample.copy()
        perturbed_top[:, :, top_idx] = 0.0
        delta_top = abs(original_score - float(model.predict(perturbed_top, verbose=0).ravel()[0]))

        # Mask bottom
        perturbed_bot = sample.copy()
        perturbed_bot[:, :, bottom_idx] = 0.0
        delta_bot = abs(original_score - float(model.predict(perturbed_bot, verbose=0).ravel()[0]))

        assert delta_top >= delta_bot, (
            f"Top feature ({FEATURE_NAMES[top_idx]}, delta={delta_top:.6f}) should matter "
            f"more than bottom ({FEATURE_NAMES[bottom_idx]}, delta={delta_bot:.6f})"
        )

    def test_masking_all_features_collapses_prediction(
        self, model: tf.keras.Model, sample: np.ndarray,
    ) -> None:
        """Zeroing all features should push prediction toward base rate."""
        pred_original = float(model.predict(sample, verbose=0).ravel()[0])
        zeroed = np.zeros_like(sample)
        pred_zeroed = float(model.predict(zeroed, verbose=0).ravel()[0])

        # Should be different from the original
        assert abs(pred_original - pred_zeroed) > 1e-4, (
            "Model gives same prediction on zeroed input — features have no effect"
        )


# ═══════════════════════════════════════════════════════════════════
# 2. EXPLANATION STABILITY
# ═══════════════════════════════════════════════════════════════════

class TestExplanationStability:
    """Verify that explanations are deterministic and stable.

    Same input should always produce the same explanation.
    Small perturbations should produce similar explanations.
    """

    def test_same_input_same_gradient_importance(
        self, model: tf.keras.Model, sample: np.ndarray,
    ) -> None:
        """Gradient-based importance should be identical for same input."""
        importances = []
        for _ in range(3):
            sample_t = tf.constant(sample, dtype=tf.float32)
            with tf.GradientTape() as tape:
                tape.watch(sample_t)
                pred = model(sample_t, training=False)
            grads = tape.gradient(pred, sample_t)
            feat_imp = np.mean(np.abs(grads.numpy().squeeze()), axis=0)
            importances.append(feat_imp)

        np.testing.assert_array_almost_equal(importances[0], importances[1], decimal=6)
        np.testing.assert_array_almost_equal(importances[1], importances[2], decimal=6)

    def test_same_input_same_top_k_ranking(
        self, model: tf.keras.Model, sample: np.ndarray,
    ) -> None:
        """Top-5 feature ranking should be identical for repeated calls."""
        rankings = []
        for _ in range(3):
            sample_t = tf.constant(sample, dtype=tf.float32)
            with tf.GradientTape() as tape:
                tape.watch(sample_t)
                pred = model(sample_t, training=False)
            grads = tape.gradient(pred, sample_t)
            feat_imp = np.mean(np.abs(grads.numpy().squeeze()), axis=0)
            top5 = list(np.argsort(feat_imp)[::-1][:5])
            rankings.append(top5)

        assert rankings[0] == rankings[1] == rankings[2]

    def test_small_perturbation_preserves_top_feature(
        self, model: tf.keras.Model, sample: np.ndarray,
    ) -> None:
        """Adding small noise should preserve the #1 most important feature."""
        # Original importance
        sample_t = tf.constant(sample, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(sample_t)
            pred = model(sample_t, training=False)
        grads = tape.gradient(pred, sample_t)
        feat_imp_orig = np.mean(np.abs(grads.numpy().squeeze()), axis=0)
        top_orig = int(np.argmax(feat_imp_orig))

        # Perturbed importance (tiny noise)
        noise = np.random.randn(*sample.shape).astype(np.float32) * 0.001
        perturbed = sample + noise
        perturbed_t = tf.constant(perturbed, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(perturbed_t)
            pred = model(perturbed_t, training=False)
        grads = tape.gradient(pred, perturbed_t)
        feat_imp_pert = np.mean(np.abs(grads.numpy().squeeze()), axis=0)
        top_pert = int(np.argmax(feat_imp_pert))

        assert top_orig == top_pert, (
            f"Top feature changed from {FEATURE_NAMES[top_orig]} to "
            f"{FEATURE_NAMES[top_pert]} with 0.001 noise"
        )

    def test_batch_explanations_independent(
        self, model: tf.keras.Model, batch: np.ndarray,
    ) -> None:
        """Each sample in a batch should get its own explanation,
        not contaminated by other samples."""
        # Get importance for sample 0 alone
        single = batch[0:1]
        single_t = tf.constant(single, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(single_t)
            pred = model(single_t, training=False)
        grads = tape.gradient(pred, single_t)
        imp_single = np.mean(np.abs(grads.numpy().squeeze()), axis=0)

        # Get importance for sample 0 in a batch
        batch_t = tf.constant(batch, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(batch_t)
            pred = model(batch_t, training=False)
        grads = tape.gradient(pred, batch_t)
        imp_batch = np.mean(np.abs(grads.numpy()[0]), axis=0)

        # Should be identical
        np.testing.assert_array_almost_equal(imp_single, imp_batch, decimal=5)


# ═══════════════════════════════════════════════════════════════════
# 3. ROLE-BASED VIEW CONSISTENCY
# ═══════════════════════════════════════════════════════════════════

class TestRoleBasedViewConsistency:
    """Verify role-based cognitive translation produces appropriate content."""

    @pytest.fixture
    def translator(self):
        from src.phase4_risk_engine.phase4.cognitive_translator import CognitiveTranslator
        return CognitiveTranslator()

    @pytest.fixture
    def high_risk_alert(self) -> Dict[str, Any]:
        return {
            "clinical_severity": 4,
            "risk_level": "HIGH",
            "attack_category": "Spoofing",
            "attention_flag": False,
            "cia_scores": {"C": 0.24, "I": 0.9, "A": 0.3},
            "cia_max_dimension": "I",
            "scenario": "high_threat",
            "device_action": "isolate_network",
            "patient_safety_flag": True,
            "response_time_minutes": 15,
            "explanation": {
                "level": "attention_and_shap",
                "top_features": [{"feature": "DIntPkt", "importance": 0.08}],
                "timestep_importance": [0.1, 0.2, 0.3, 0.15, 0.25],
            },
            "device_type": "infusion_pump",
            "clinical_rationale": "Base: HIGH on infusion_pump | PATIENT SAFETY FLAG",
        }

    @pytest.fixture
    def novel_threat_alert(self) -> Dict[str, Any]:
        return {
            "clinical_severity": 3,
            "risk_level": "MEDIUM",
            "attack_category": "unknown",
            "attention_flag": True,
            "cia_scores": {"C": 0.2, "I": 0.4, "A": 0.35},
            "cia_max_dimension": "I",
            "scenario": "high_threat",
            "device_action": "restrict_network",
            "patient_safety_flag": False,
            "response_time_minutes": 60,
            "explanation": {"level": "attention_only", "timestep_importance": [0.2] * 5},
            "device_type": "pulse_oximeter",
            "clinical_rationale": "Novel threat on safety-critical device",
        }

    @pytest.fixture
    def normal_alert(self) -> Dict[str, Any]:
        return {"clinical_severity": 1, "risk_level": "NORMAL"}

    # ── SOC Analyst View ───────────────────────────────────────────

    def test_soc_view_contains_threat_type(
        self, translator, high_risk_alert: Dict,
    ) -> None:
        translator.translate([high_risk_alert])
        soc = high_risk_alert["stakeholder_views"]["soc_analyst"]
        assert soc["threat_type"] == "Spoofing"

    def test_soc_view_flags_novel_threat(
        self, translator, novel_threat_alert: Dict,
    ) -> None:
        translator.translate([novel_threat_alert])
        soc = novel_threat_alert["stakeholder_views"]["soc_analyst"]
        assert "NOVEL" in soc["threat_type"] or "ZERO-DAY" in soc["threat_type"]

    def test_soc_view_has_cia_impact(
        self, translator, high_risk_alert: Dict,
    ) -> None:
        translator.translate([high_risk_alert])
        soc = high_risk_alert["stakeholder_views"]["soc_analyst"]
        assert "Integrity" in soc["primary_cia_impact"]

    def test_soc_view_has_recommended_actions(
        self, translator, high_risk_alert: Dict,
    ) -> None:
        translator.translate([high_risk_alert])
        soc = high_risk_alert["stakeholder_views"]["soc_analyst"]
        assert len(soc["recommended_actions"]) > 0

    # ── Clinician View ─────────────────────────────────────────────

    def test_clinician_view_uses_plain_language(
        self, translator, high_risk_alert: Dict,
    ) -> None:
        translator.translate([high_risk_alert])
        clin = high_risk_alert["stakeholder_views"]["clinician"]
        # Should not contain technical jargon
        message = clin["message"]
        assert "MAD" not in message
        assert "sigmoid" not in message
        assert "threshold" not in message

    def test_clinician_view_has_safety_guidance(
        self, translator, high_risk_alert: Dict,
    ) -> None:
        translator.translate([high_risk_alert])
        clin = high_risk_alert["stakeholder_views"]["clinician"]
        assert "safety_guidance" in clin
        assert "DO NOT power off" in clin["safety_guidance"]

    def test_clinician_view_shows_device_status(
        self, translator, high_risk_alert: Dict,
    ) -> None:
        translator.translate([high_risk_alert])
        clin = high_risk_alert["stakeholder_views"]["clinician"]
        assert "device_status" in clin
        assert "restricted" in clin["device_status"].lower() or "isolated" in clin["device_status"].lower()

    def test_clinician_view_has_urgency(
        self, translator, high_risk_alert: Dict,
    ) -> None:
        translator.translate([high_risk_alert])
        clin = high_risk_alert["stakeholder_views"]["clinician"]
        assert "urgency" in clin
        assert "15 minutes" in clin["urgency"] or "Emergent" in clin["urgency"]

    # ── CISO View ──────────────────────────────────────────────────

    def test_ciso_view_has_compliance_impacts(
        self, translator, high_risk_alert: Dict,
    ) -> None:
        translator.translate([high_risk_alert])
        ciso = high_risk_alert["stakeholder_views"]["ciso"]
        assert "compliance_impacts" in ciso

    def test_ciso_view_has_reporting_requirements(
        self, translator, high_risk_alert: Dict,
    ) -> None:
        translator.translate([high_risk_alert])
        ciso = high_risk_alert["stakeholder_views"]["ciso"]
        assert "recommended_reporting" in ciso
        assert len(ciso["recommended_reporting"]) > 0

    def test_ciso_view_tracks_patient_safety(
        self, translator, high_risk_alert: Dict,
    ) -> None:
        translator.translate([high_risk_alert])
        ciso = high_risk_alert["stakeholder_views"]["ciso"]
        assert ciso["patient_safety_event"] is True

    # ── Biomed Engineer View ───────────────────────────────────────

    def test_biomed_view_has_device_type(
        self, translator, high_risk_alert: Dict,
    ) -> None:
        translator.translate([high_risk_alert])
        biomed = high_risk_alert["stakeholder_views"]["biomed_engineer"]
        assert biomed["device_type"] == "infusion_pump"

    def test_biomed_view_has_recommended_checks(
        self, translator, high_risk_alert: Dict,
    ) -> None:
        translator.translate([high_risk_alert])
        biomed = high_risk_alert["stakeholder_views"]["biomed_engineer"]
        assert "recommended_checks" in biomed
        assert len(biomed["recommended_checks"]) > 0
        # Should include firmware CVE check
        checks_text = " ".join(biomed["recommended_checks"]).lower()
        assert "firmware" in checks_text or "cve" in checks_text

    def test_biomed_view_has_anomaly_indicators(
        self, translator, high_risk_alert: Dict,
    ) -> None:
        translator.translate([high_risk_alert])
        biomed = high_risk_alert["stakeholder_views"]["biomed_engineer"]
        assert "anomaly_indicators" in biomed

    # ── Cross-Role Consistency ─────────────────────────────────────

    def test_normal_alert_gets_no_views(
        self, translator, normal_alert: Dict,
    ) -> None:
        translator.translate([normal_alert])
        assert normal_alert["stakeholder_views"] is None

    def test_all_four_views_present_for_actionable(
        self, translator, high_risk_alert: Dict,
    ) -> None:
        translator.translate([high_risk_alert])
        views = high_risk_alert["stakeholder_views"]
        assert "soc_analyst" in views
        assert "clinician" in views
        assert "ciso" in views
        assert "biomed_engineer" in views

    def test_views_contain_no_raw_scores(
        self, translator, high_risk_alert: Dict,
    ) -> None:
        """Clinician view should not leak raw anomaly scores or thresholds."""
        translator.translate([high_risk_alert])
        clin = high_risk_alert["stakeholder_views"]["clinician"]
        clin_text = str(clin).lower()
        assert "0.75" not in clin_text  # anomaly_score
        assert "0.5" not in clin_text   # threshold
        assert "mad" not in clin_text
