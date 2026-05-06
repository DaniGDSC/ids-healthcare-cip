"""Layer 2 detector — per-alert canonical entry point.

Implements the redesigned Layer 2 from system_architecture_final.md:

    Step 1: feature sanitization (NaN/Inf → BENIGN_MEDIAN, NOT 0.0)
    Step 2a: Track A (XGB + RF + DT, calibrated, with diversity score)
    Step 2b: Track B (cascaded DAE on [25 raw || P_xgb, P_rf, P_dt])
            with multi-threshold (p80/p95/p99) and per-dim error breakdown

Returns a single ``Layer2Output`` dataclass with every field the diagram
specifies as "COMBINED OUTPUT TO LAYER 3". Downstream code in Layer 3
(fusion + risk scoring) consumes exactly that shape.

Multi-threshold DAE
-------------------
At construction time, three percentile thresholds (``p80``, ``p95``,
``p99``) are computed from the DAE's persisted benign training-error
distribution. At inference, a row's reconstruction error is bucketed:

    err >= p99   → "strong"
    p95 <= err   → "moderate"
    p80 <= err   → "weak"
    err <  p80   → "below_threshold"   (no anomaly)

Operationally, "below_threshold" = no DAE flag; the three above-noise
buckets feed Layer 3's confidence-aware routing.

Per-dimension DAE errors
------------------------
``DAEDetector.reconstruction_error_decomposed`` returns per-feature
weighted squared errors. At construction time, per-dim p95 thresholds
are computed from a benign reference set (val benign parquet, falling
back to a synthetic re-pass over the cached train errors). At inference,
each dim with ``per_dim_error[i] >= per_dim_p95[i]`` is added to
``anomalous_dims``. ``per_dim_errors`` is the full (28,) array so
Layer 4 explanations can rank features without re-running the DAE.

Both behaviours fully replace the previous stubs (Pre-Redesign Tasks
4 + 5). Downstream API surface is unchanged: callers see the same
``Layer2Output`` field names, but ``threshold_level`` now varies and
``anomalous_dims`` / ``per_dim_errors`` are populated.
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from common import loads_signed  # noqa: E402
from src.data_models import DataQuality  # noqa: E402
from src.preprocessing import FEATURE_NAMES_25, sanitize_features  # noqa: E402
from src.risk_scorer import get_track_a_surfacing_threshold  # noqa: E402

logger = logging.getLogger(__name__)


# Multi-threshold level vocabulary (Task 4 implementation).
# "below_threshold" means the row's reconstruction error sits inside the
# bulk of the benign training distribution (below the p80 quantile);
# the three above-noise buckets feed Layer 3's confidence routing.
THRESHOLD_LEVELS: tuple[str, ...] = (
    "below_threshold", "weak", "moderate", "strong",
)
THRESHOLD_PERCENTILES: tuple[float, ...] = (80.0, 95.0, 99.0)
PER_DIM_PERCENTILE: float = 95.0   # per-feature anomaly cutoff


# ── Output dataclass ────────────────────────────────────────────────────

@dataclass
class Layer2Output:
    """Canonical Layer 2 output shape (matches system_architecture_final.md).

    Fields populated unconditionally:
      - ``p_xgb`` / ``p_rf`` / ``p_dt``       : calibrated P(attack)
      - ``c_track_a``                          : ``max(p_xgb, p_rf, p_dt)``
      - ``diversity_score``                    : std(p_xgb, p_rf, p_dt)
      - ``dae_score``                          : DAE anomaly score in [0, 1]
      - ``dae_score_raw_error``                : raw reconstruction error
      - ``c_track_b``                          : alias for ``dae_score``
      - ``device_class_threshold``             : per-device surfacing threshold
      - ``data_quality_flag``                  : OK / DEGRADED / FAILED
      - ``nan_rate``                           : fraction of NaN/Inf cells
      - ``calibration_used``                   : True if calibrated artefacts loaded
      - ``threshold_level``                    : one of THRESHOLD_LEVELS
                                                  (below_threshold/weak/moderate/strong)
      - ``anomalous_dims``                     : list of int — feature indices whose
                                                  per-dim error exceeded the per-dim p95
      - ``anomalous_dim_names``                : list of str — feature names matching
                                                  ``anomalous_dims`` for explainability
      - ``per_dim_errors``                     : ndarray (n_features,) — full per-dim
                                                  weighted squared errors
    """

    p_xgb: float
    p_rf: float
    p_dt: float
    c_track_a: float
    diversity_score: float
    dae_score: float
    dae_score_raw_error: float
    c_track_b: float
    device_class_threshold: float
    data_quality_flag: str
    nan_rate: float
    calibration_used: bool

    threshold_level: str = "below_threshold"
    anomalous_dims: list[int] = field(default_factory=list)
    anomalous_dim_names: list[str] = field(default_factory=list)
    per_dim_errors: np.ndarray | None = None

    def as_dict(self) -> dict[str, Any]:
        d = {
            "p_xgb": float(self.p_xgb),
            "p_rf": float(self.p_rf),
            "p_dt": float(self.p_dt),
            "c_track_a": float(self.c_track_a),
            "diversity_score": float(self.diversity_score),
            "dae_score": float(self.dae_score),
            "dae_score_raw_error": float(self.dae_score_raw_error),
            "c_track_b": float(self.c_track_b),
            "device_class_threshold": float(self.device_class_threshold),
            "data_quality_flag": self.data_quality_flag,
            "nan_rate": float(self.nan_rate),
            "calibration_used": bool(self.calibration_used),
            "threshold_level": self.threshold_level,
            "anomalous_dims": [int(i) for i in self.anomalous_dims],
            "anomalous_dim_names": list(self.anomalous_dim_names),
        }
        if self.per_dim_errors is not None:
            d["per_dim_errors"] = self.per_dim_errors.tolist()
        return d


# ── Detector ────────────────────────────────────────────────────────────

class Layer2Detector:
    """Per-alert Layer 2 detector — Step 1 + Step 2a + Step 2b in one call.

    Loads every artefact at construction time; ``score_alert`` is then a
    pure function over a single 25-feature flow vector. Designed for
    online inference where the same detector is reused across thousands
    of alerts.

    Args:
        models_dir: where ``{xgboost,random_forest,decision_tree}_final_pipeline.pkl``,
            their ``*_calibrator.pkl`` (if present), and DAE artefacts live.
            Defaults to ``results/models``.
        prefer_calibrated: when True (default), wrap each tree with its
            calibrator if the artefact exists. Falls back transparently
            to raw probas when a calibrator is missing — matches the
            ``module3_risk_scoring/module3_risk_scores.py::load_xgboost_proba``
            convention from Pre-Redesign Task 1.
        feature_names: 25-feature ordering matching the trained models.
            Defaults to ``FEATURE_NAMES_25`` from src/preprocessing.py.
    """

    def __init__(
        self,
        models_dir: Path | None = None,
        *,
        prefer_calibrated: bool = True,
        feature_names: Sequence[str] | None = None,
    ) -> None:
        self._models_dir = Path(models_dir) if models_dir else PROJECT_ROOT / "results/models"
        self._feature_names = tuple(feature_names) if feature_names else tuple(FEATURE_NAMES_25)
        self._calibration_used: dict[str, bool] = {
            "xgboost": False, "random_forest": False, "decision_tree": False,
        }

        # ── Step 1 prereqs (sanitizer is module-level; nothing to load) ──

        # ── Step 2a prereqs: 3 calibrated trees ──
        self._track_a: dict[str, Any] = {}
        for name in ("xgboost", "random_forest", "decision_tree"):
            base_pkl = self._models_dir / f"{name}_final_pipeline.pkl"
            cal_pkl = self._models_dir / f"{name}_calibrator.pkl"
            if not base_pkl.exists():
                raise FileNotFoundError(
                    f"Missing {base_pkl}. Run module2_detection/module2_train_models.py first."
                )
            if prefer_calibrated and cal_pkl.exists():
                self._track_a[name] = joblib.load(cal_pkl)
                self._calibration_used[name] = True
                logger.info("Track A: loaded calibrated %s", name)
            else:
                self._track_a[name] = loads_signed(base_pkl)
                logger.info("Track A: loaded raw %s (no calibrator at %s)", name, cal_pkl)

        # ── Step 2b prereqs: DAE ──
        from module2_detection.models.DAE import DAEDetector  # lazy: TF heavy
        dae_json = self._models_dir / "dae_detector.json"
        dae_weights = self._models_dir / "dae_model.weights.h5"
        if not (dae_json.exists() and dae_weights.exists()):
            raise FileNotFoundError(
                f"Missing DAE artefacts at {self._models_dir}. "
                f"Run module2_detection/module2_train_models.py."
            )
        self._dae = DAEDetector.from_artefacts(
            json_path=dae_json, weights_path=dae_weights,
        )
        logger.info("Track B: loaded DAE (threshold=%.6f)", self._dae.threshold)

        # ── Multi-threshold (Task 4): compute p80/p95/p99 from DAE's
        # persisted benign training-error distribution. Cheap (one np call). ──
        train_errs = self._load_train_errors()
        self._mt_p80, self._mt_p95, self._mt_p99 = (
            float(np.percentile(train_errs, p)) for p in THRESHOLD_PERCENTILES
        )
        logger.info(
            "Multi-threshold: p80=%.6f, p95=%.6f, p99=%.6f",
            self._mt_p80, self._mt_p95, self._mt_p99,
        )

        # ── Per-dim percentiles (Task 5): one DAE pass over a benign
        # reference set to derive per-feature p95 cutoffs. Reference
        # priority: val benign parquet (post-L1-2) → train benign parquet. ──
        self._per_dim_thresholds, self._cascade_feature_names = \
            self._compute_per_dim_thresholds()
        logger.info(
            "Per-dim percentiles (p%.0f) computed for %d cascade features",
            PER_DIM_PERCENTILE, len(self._per_dim_thresholds),
        )

    # ── Helpers for Tasks 4 + 5 ────────────────────────────────────────

    def _load_train_errors(self) -> np.ndarray:
        """Read the DAE's persisted train_errors array from the JSON sidecar."""
        sidecar = self._models_dir / "dae_detector.json"
        body = json.loads(sidecar.read_text(encoding="utf-8"))
        train_errs = body.get("train_errors")
        if not train_errs:
            raise RuntimeError(
                f"DAE sidecar at {sidecar} has no 'train_errors' — "
                "the DAE artefact predates the multi-threshold support; "
                "retrain via module2_detection/module2_train_models.py."
            )
        return np.asarray(train_errs, dtype=np.float64)

    def _compute_per_dim_thresholds(self) -> tuple[np.ndarray, list[str]]:
        """Compute per-dimension p95 cutoff over a benign reference set.

        Output ordering matches the cascade input layout:
            ``[25 raw features || track_a_xgb, track_a_rf, track_a_dt]``

        Returns:
            (per_dim_p95, cascade_feature_names) — both length 28.
        """
        import pandas as pd  # local: pandas import is heavy enough to defer
        # Pick the largest benign reference available without loading attacks.
        candidates = [
            self._models_dir.parent.parent / "data/processed/val_benign_phase1.parquet",
            self._models_dir.parent.parent / "data/processed/train_benign_phase1.parquet",
        ]
        reference_path = next((p for p in candidates if p.exists()), None)
        if reference_path is None:
            raise FileNotFoundError(
                "No benign reference parquet found for per-dim percentile "
                f"calibration. Searched: {[str(p) for p in candidates]}. "
                "Run module1_preprocessing/phase1 first."
            )
        df = pd.read_parquet(reference_path)
        drop = [c for c in (
            "Label", "Attack Category", "row_id",
            "device_class", "attack_category",
        ) if c in df.columns]
        feat_names = [c for c in df.columns if c not in drop]
        X_benign = df.drop(columns=drop).values.astype(np.float32)

        # Run Track A on benigns to construct the cascade input — same
        # transform as score_alert(). Probas mean-pooled across the 3
        # models? No — Track A in the cascade emits 3 separate columns.
        p_xgb = self._track_a["xgboost"].predict_proba(X_benign)[:, 1]
        p_rf = self._track_a["random_forest"].predict_proba(X_benign)[:, 1]
        p_dt = self._track_a["decision_tree"].predict_proba(X_benign)[:, 1]
        proba_cols = np.column_stack([p_xgb, p_rf, p_dt]).astype(np.float32)
        X_aug = np.column_stack([X_benign, proba_cols])

        _, per_feat_err = self._dae.reconstruction_error_decomposed(X_aug)
        per_dim_p95 = np.percentile(per_feat_err, PER_DIM_PERCENTILE,
                                      axis=0).astype(np.float64)

        cascade_names = list(feat_names) + ["track_a_xgb", "track_a_rf", "track_a_dt"]
        if len(cascade_names) != X_aug.shape[1]:
            raise RuntimeError(
                f"Feature-name count {len(cascade_names)} != cascade dim "
                f"{X_aug.shape[1]}; refusing to proceed with mismatched names."
            )
        return per_dim_p95, cascade_names

    def _bucket_threshold_level(self, recon_err: float) -> str:
        """Map a single reconstruction error to one of THRESHOLD_LEVELS."""
        if recon_err >= self._mt_p99:
            return "strong"
        if recon_err >= self._mt_p95:
            return "moderate"
        if recon_err >= self._mt_p80:
            return "weak"
        return "below_threshold"

    # ── Public API ──────────────────────────────────────────────────────

    def score_alert(
        self,
        raw_features: np.ndarray | Sequence[float],
        device_class: str | None = None,
    ) -> Layer2Output:
        """Run Step 1 + Step 2a + Step 2b on a single flow record.

        Args:
            raw_features: shape (25,) or (1, 25). May contain NaN/Inf.
                Order MUST match ``feature_names`` (i.e. FEATURE_NAMES_25).
            device_class: optional, e.g. "infusion_pump" / "ehr_workstation".
                Used to look up the per-device surfacing threshold (Layer 3
                consumes this; the detector itself does not gate).

        Returns:
            ``Layer2Output`` populated with all fields specified in the
            system architecture diagram.
        """
        # ── Step 1: sanitization ──
        x_clean, flag, nan_rate = sanitize_features(
            raw_features, feature_names=self._feature_names,
        )
        x_2d = x_clean.reshape(1, -1) if x_clean.ndim == 1 else x_clean
        # Note: scaling is already applied in Phase 1's processed parquet;
        # for live inference, the caller is responsible for scaling. The
        # detector does NOT re-scale to avoid double-transform footguns.
        # Phase-1 artefacts → directly in scaled space; live alerts must
        # be transformed via the persisted scaler before reaching here.

        # ── Step 2a: Track A (3 calibrated probas + diversity) ──
        p_xgb = float(self._track_a["xgboost"].predict_proba(x_2d)[0, 1])
        p_rf = float(self._track_a["random_forest"].predict_proba(x_2d)[0, 1])
        p_dt = float(self._track_a["decision_tree"].predict_proba(x_2d)[0, 1])
        c_track_a = max(p_xgb, p_rf, p_dt)
        diversity = float(np.std([p_xgb, p_rf, p_dt]))
        device_threshold = get_track_a_surfacing_threshold(device_class)

        # ── Step 2b: Track B (cascaded DAE + multi-threshold + per-dim) ──
        proba_columns = np.array([[p_xgb, p_rf, p_dt]], dtype=np.float32)
        x_aug = np.column_stack([x_2d.astype(np.float32), proba_columns])
        # Single forward pass yields BOTH per-sample and per-feature error
        # (reconstruction_error_decomposed). This is the reason the DAE
        # exposes a decomposed call — halves compute per alert.
        per_sample_err, per_feat_err = self._dae.reconstruction_error_decomposed(x_aug)
        recon_err = float(per_sample_err[0])
        per_dim_errors = per_feat_err[0].astype(np.float64)

        dae_threshold = float(self._dae.threshold)
        if recon_err <= dae_threshold:
            dae_score = 0.5 * (recon_err / max(dae_threshold, 1e-12))
        else:
            # Saturate above threshold so the score grows but never exceeds 1.
            dae_score = 0.5 + 0.5 * min(
                (recon_err - dae_threshold) / max(dae_threshold, 1e-12), 1.0
            )

        # ── Task 4: multi-threshold bucket (below_threshold/weak/moderate/strong) ──
        threshold_level = self._bucket_threshold_level(recon_err)

        # ── Task 5: anomalous-dim selection (per-dim error >= per-dim p95) ──
        anomalous_mask = per_dim_errors >= self._per_dim_thresholds
        anomalous_dims = [int(i) for i in np.where(anomalous_mask)[0]]
        anomalous_dim_names = [self._cascade_feature_names[i] for i in anomalous_dims]

        any_calibrated = any(self._calibration_used.values())

        return Layer2Output(
            p_xgb=p_xgb,
            p_rf=p_rf,
            p_dt=p_dt,
            c_track_a=c_track_a,
            diversity_score=diversity,
            dae_score=float(np.clip(dae_score, 0.0, 1.0)),
            dae_score_raw_error=recon_err,
            c_track_b=float(np.clip(dae_score, 0.0, 1.0)),
            device_class_threshold=device_threshold,
            data_quality_flag=flag,
            nan_rate=nan_rate,
            calibration_used=any_calibrated,
            threshold_level=threshold_level,
            anomalous_dims=anomalous_dims,
            anomalous_dim_names=anomalous_dim_names,
            per_dim_errors=per_dim_errors,
        )

    @property
    def calibration_status(self) -> dict[str, bool]:
        """Per-model calibration use status (debug / introspection)."""
        return dict(self._calibration_used)

    @property
    def multi_thresholds(self) -> dict[str, float]:
        """Multi-threshold cutoffs (debug / introspection / docs export)."""
        return {"p80": self._mt_p80, "p95": self._mt_p95, "p99": self._mt_p99}

    @property
    def per_dim_thresholds(self) -> np.ndarray:
        """Per-dimension p95 cutoffs (read-only copy)."""
        return self._per_dim_thresholds.copy()

    @property
    def cascade_feature_names(self) -> list[str]:
        """Names of the 28 cascade dimensions (25 raw + 3 Track A probas)."""
        return list(self._cascade_feature_names)


__all__ = [
    "Layer2Detector",
    "Layer2Output",
    "THRESHOLD_LEVELS",
    "THRESHOLD_PERCENTILES",
    "PER_DIM_PERCENTILE",
]
