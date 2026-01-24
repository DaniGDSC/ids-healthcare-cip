"""Ensemble classifier combining multiple models."""

import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from typing import Dict
import joblib
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class EnsembleClassifier:
    """Train and predict with an ensemble of classifiers."""

    def __init__(self, config: Dict):
        self.config = config
        self.models = {}
        self.ensemble = None

    def _build_models(self):
        cfg = self.config['classification']['models']

        if cfg['svm']['enabled']:
            self.models['svm'] = SVC(
                kernel=cfg['svm']['kernel'],
                C=cfg['svm']['C'],
                gamma=cfg['svm']['gamma'],
                probability=cfg['svm']['probability'],
                cache_size=cfg['svm']['cache_size']
            )

        if cfg['decision_tree']['enabled']:
            self.models['decision_tree'] = DecisionTreeClassifier(
                max_depth=cfg['decision_tree']['max_depth'],
                min_samples_split=cfg['decision_tree']['min_samples_split'],
                min_samples_leaf=cfg['decision_tree']['min_samples_leaf'],
                criterion=cfg['decision_tree']['criterion']
            )

        estimators = [(name, model) for name, model in self.models.items()]

        ensemble_cfg = self.config['classification']['ensemble']
        if ensemble_cfg['enabled']:
            self.ensemble = VotingClassifier(
                estimators=estimators,
                voting=ensemble_cfg['voting_type'],
                weights=ensemble_cfg.get('weights')
            )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._build_models()

        # Fit individual models
        for name, model in self.models.items():
            logger.info(f"Training {name} model...")
            model.fit(X, y)

        # Fit ensemble
        if self.ensemble is not None:
            logger.info("Training ensemble model...")
            self.ensemble.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.ensemble is not None:
            return self.ensemble.predict(X)
        # Fallback: use first model
        first_model = next(iter(self.models.values()))
        return first_model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.ensemble is not None:
            return self.ensemble.predict_proba(X)
        first_model = next(iter(self.models.values()))
        if hasattr(first_model, 'predict_proba'):
            return first_model.predict_proba(X)
        raise ValueError("Model does not support probability predictions")

    def save(self, model_dir: Path):
        model_dir.mkdir(parents=True, exist_ok=True)
        for name, model in self.models.items():
            joblib.dump(model, model_dir / f"{name}_model.joblib")
        if self.ensemble is not None:
            joblib.dump(self.ensemble, model_dir / "ensemble_model.joblib")

    def load(self, model_dir: Path):
        for name in ['svm', 'decision_tree', 'ensemble']:
            path = model_dir / f"{name}_model.joblib"
            if path.exists():
                self.models[name] = joblib.load(path)
        ensemble_path = model_dir / "ensemble_model.joblib"
        if ensemble_path.exists():
            self.ensemble = joblib.load(ensemble_path)