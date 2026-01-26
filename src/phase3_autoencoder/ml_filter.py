"""Lightweight ML filtering to reduce benign traffic load before Phase 3.

Trains a RandomForest to classify benign vs attack, calibrates a probability
threshold to achieve target recall on attacks (≥ 99.5%), and filters
high-confidence benign samples. Suspicious samples continue to autoencoder.
"""

from typing import Tuple, Dict, Any
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import recall_score
import joblib
from pathlib import Path
import os


class MLFilter:
    def __init__(self,
                 n_estimators: int = 100,
                 max_depth: int = 12,
                 min_samples_split: int = 8,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 benign_label: int = 0,
                 attack_label: int = 1):
        self.rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=n_jobs
        )
        self.threshold_: float = 0.5
        self.benign_label = benign_label
        self.attack_label = attack_label

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.rf.fit(X, y)
        return self

    def calibrate_threshold(self, X_val: np.ndarray, y_val: np.ndarray,
                            recall_target: float = 0.995) -> float:
        """Choose probability threshold for benign classification ensuring
        attack recall ≥ recall_target on validation.

        We predict attack probability p_attack = 1 - p_benign. We then choose
        threshold t on p_attack to classify attack if p_attack ≥ t.
        """
        proba = self.rf.predict_proba(X_val)
        # Assume classes are [benign_label, attack_label] order
        # Find index of attack class in RF classes_
        cls_attack_idx = int(np.where(self.rf.classes_ == self.attack_label)[0][0])
        p_attack = proba[:, cls_attack_idx]

        # Grid search thresholds from 0.1 → 0.9
        grid = np.linspace(0.1, 0.9, 33)
        best_t = 0.5
        best_recall = -1.0
        for t in grid:
            y_pred_attack = (p_attack >= t).astype(int)
            # Map to labels {benign_label, attack_label}
            y_pred = np.where(y_pred_attack == 1, self.attack_label, self.benign_label)
            rec = recall_score(y_val == self.attack_label, y_pred == self.attack_label)
            if rec >= recall_target:
                best_t = t
                best_recall = rec
                break
            if rec > best_recall:
                best_recall = rec
                best_t = t
        self.threshold_ = float(best_t)
        return self.threshold_

    def predict_attack_proba(self, X: np.ndarray) -> np.ndarray:
        cls_attack_idx = int(np.where(self.rf.classes_ == self.attack_label)[0][0])
        return self.rf.predict_proba(X)[:, cls_attack_idx]

    def filter_benign(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter out high-confidence benign samples using calibrated threshold.

        Returns (X_pass, y_pass, mask_pass) where mask_pass selects samples to pass.
        """
        p_attack = self.predict_attack_proba(X)
        pass_mask = p_attack >= self.threshold_
        return X[pass_mask], y[pass_mask], pass_mask

    def save(self, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'model': self.rf, 'threshold': self.threshold_}, str(p))
        try:
            os.chmod(p, 0o600)
        except Exception:
            pass

    @staticmethod
    def load(path: str) -> 'MLFilter':
        p = Path(path)
        blob = joblib.load(str(p))
        mf = MLFilter()
        mf.rf = blob['model']
        mf.threshold_ = blob['threshold']
        return mf
