"""
Phase 5: Ensemble Classification (SVM + Decision Tree)

Combines a Support Vector Machine (60% weight) and Decision Tree (40% weight)
using soft probability voting for robust attack classification on hybrid features
(35 original + 8 latent dimensions).

Features:
- Balanced class weights to handle WebAttack minority (0.15% of training data)
- Probability-based soft voting ensemble
- Feature importance extraction from decision tree
- Confidence score calculation for predictions
"""

import logging
import threading
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)


class EnsembleClassifier:
    """
    Soft voting ensemble combining SVM and Decision Tree classifiers.
    
    The ensemble uses weighted probability voting:
    - SVM weight: 0.6 (60%)
    - Decision Tree weight: 0.4 (40%)
    
    Both base classifiers use balanced class weights to handle severe imbalance
    (WebAttack: 0.15% of training data).
    
    Performance Optimization:
    - Uses Nystroem kernel approximation for RBF kernel (500 components)
    - Reduces SVM training complexity from O(n²-n³) to O(n·k) where k=500
    - Enables scalability to datasets >500K samples (50-100x faster training)
    - Maintains similar accuracy to exact RBF kernel SVM
    
    Parameters
    ----------
    svm_weight : float
        Weight for SVM predictions in ensemble (default: 0.6)
    dt_weight : float
        Weight for Decision Tree predictions in ensemble (default: 0.4)
    random_state : int
        Random seed for reproducibility (default: 42)
    n_jobs : int
        Number of parallel jobs for SVM (default: -1, use all cores)
    use_kernel_approximation : bool
        If True, use Nystroem kernel approximation for efficient training (default: True)
        Recommended for datasets >100K samples (50-100x faster than exact RBF kernel)
    n_components : int
        Number of components for Nystroem approximation (default: 500)
        Higher values increase accuracy but slow down training
    """
    
    def __init__(
        self,
        svm_weight: float = 0.6,
        dt_weight: float = 0.4,
        random_state: int = 42,
        n_jobs: int = -1,
        use_kernel_approximation: bool = True,
        n_components: int = 500,
        sgd_alpha: float = 0.0001,
        sgd_max_iter: int = 1000,
        sgd_tol: float = 1e-3,
        dt_max_depth: int = 30,
        dt_min_samples_split: int = 10,
        dt_min_samples_leaf: int = 5,
        dt_criterion: str = "gini"
    ):
        assert abs(svm_weight + dt_weight - 1.0) < 1e-6, \
            f"Weights must sum to 1.0, got {svm_weight + dt_weight}"
        
        self.svm_weight = svm_weight
        self.dt_weight = dt_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.use_kernel_approximation = use_kernel_approximation
        self.n_components = n_components
        self.sgd_alpha = sgd_alpha
        self.sgd_max_iter = sgd_max_iter
        self.sgd_tol = sgd_tol
        self.dt_max_depth = dt_max_depth
        self.dt_min_samples_split = dt_min_samples_split
        self.dt_min_samples_leaf = dt_min_samples_leaf
        self.dt_criterion = dt_criterion
        self._lock = threading.RLock()
        
        # Base classifiers
        if use_kernel_approximation:
            # Optimized: Nystroem kernel approximation + SGD (O(n·k) complexity)
            # 50-100x faster than exact RBF SVM for large datasets
            # Note: gamma=None uses 1/(n_features) as default (similar to 'scale')
            self.svm = Pipeline([
                ('feature_map', Nystroem(
                    kernel='rbf',
                    gamma=None,  # Uses 1/n_features (equivalent to 'scale' for normalized data)
                    n_components=n_components,
                    random_state=random_state
                )),
                ('sgd', SGDClassifier(
                    loss='log_loss',  # For probability estimates
                    penalty='l2',
                    alpha=self.sgd_alpha,
                    max_iter=self.sgd_max_iter,
                    tol=self.sgd_tol,
                    class_weight='balanced',
                    random_state=random_state,
                    n_jobs=n_jobs,
                    verbose=0
                ))
            ])
            logger.info(
                f"Using Nystroem kernel approximation (n_components={n_components}) "
                f"for efficient SVM training (O(n·k) vs O(n²-n³))"
            )
        else:
            # Original: Exact RBF kernel SVM (O(n²-n³) complexity)
            # Only use for small datasets (<100K samples)
            self.svm = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced',
                probability=True,
                max_iter=10000,
                random_state=random_state,
                verbose=0
            )
            logger.warning(
                "Using exact RBF kernel SVM - may be slow for large datasets. "
                "Consider setting use_kernel_approximation=True for >100K samples."
            )
        
        self.decision_tree = DecisionTreeClassifier(
            criterion=self.dt_criterion,
            max_depth=self.dt_max_depth,
            min_samples_split=self.dt_min_samples_split,
            min_samples_leaf=self.dt_min_samples_leaf,
            max_features=None,
            class_weight='balanced',
            random_state=random_state
        )
        
        # Scaler for SVM (RBF kernel benefits from normalization)
        self.scaler = StandardScaler()
        
        self.classes_ = None
        self.is_fitted = False
        
        logger.info(
            f"Initialized EnsembleClassifier with weights "
            f"SVM={svm_weight}, DT={dt_weight}"
        )
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> 'EnsembleClassifier':
        """
        Fit both base classifiers on training data.
        
        Parameters
        ----------
        X_train : np.ndarray, shape (n_samples, n_features)
            Training features (43D: 35 original + 8 latent)
        y_train : np.ndarray, shape (n_samples,)
            Training labels (0=Benign, 1=BruteForce, 2=WebAttack)
        
        Returns
        -------
        self
        """
        with self._lock:
            logger.info(f"Training ensemble on {X_train.shape[0]} samples, {X_train.shape[1]} features")
            
            self.classes_ = np.unique(y_train)
            
            # Scale features for SVM
            X_scaled = self.scaler.fit_transform(X_train)
            
            # Train SVM
            logger.info("Training SVM classifier...")
            self.svm.fit(X_scaled, y_train)
            if self.use_kernel_approximation:
                logger.info(f"SVM trained with Nystroem approximation ({self.n_components} components)")
            else:
                logger.info(f"SVM trained. Support vectors: {len(self.svm.support_vectors_)}")
            
            # Train Decision Tree
            logger.info("Training Decision Tree classifier...")
            self.decision_tree.fit(X_train, y_train)
            logger.info(f"Decision Tree trained. Depth: {self.decision_tree.get_depth()}")
            
            self.is_fitted = True
            logger.info("Ensemble training complete")
            
            return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using weighted voting.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input features
        
        Returns
        -------
        proba : np.ndarray, shape (n_samples, n_classes)
            Probability predictions from ensemble
        """
        with self._lock:
            if not self.is_fitted:
                raise ValueError("Ensemble must be fitted before prediction")
            
            X_scaled = self.scaler.transform(X)
            
            # Get probabilities from both classifiers
            svm_proba = self.svm.predict_proba(X_scaled)
            dt_proba = self.decision_tree.predict_proba(X)
            
            # Weighted combination
            ensemble_proba = (
                self.svm_weight * svm_proba + 
                self.dt_weight * dt_proba
            )
            
            return ensemble_proba
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using weighted voting.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input features
        
        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
            Predicted class labels
        """
        with self._lock:
            proba = self.predict_proba(X)
            return self.classes_[np.argmax(proba, axis=1)]
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Extract feature importance from decision tree.
        
        Returns
        -------
        importance : np.ndarray, shape (n_features,)
            Feature importance scores from decision tree
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before extracting importance")
        
        return self.decision_tree.feature_importances_
    
    def get_confidence_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Get maximum probability (confidence) for each prediction.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input features
        
        Returns
        -------
        confidence : np.ndarray, shape (n_samples,)
            Maximum prediction probability for each sample
        """
        proba = self.predict_proba(X)
        return np.max(proba, axis=1)
    
    def save(self, model_dir: Path) -> Dict[str, Path]:
        """
        Save ensemble models and scaler.
        
        Parameters
        ----------
        model_dir : Path
            Directory to save models
        
        Returns
        -------
        saved_files : Dict[str, Path]
            Paths to saved model files
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted ensemble")
        
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        svm_path = model_dir / "svm_model.joblib"
        dt_path = model_dir / "decision_tree_model.joblib"
        scaler_path = model_dir / "feature_scaler.joblib"
        ensemble_path = model_dir / "ensemble_model.joblib"
        
        # Save individual models
        joblib.dump(self.svm, svm_path)
        joblib.dump(self.decision_tree, dt_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Save ensemble metadata
        # Handle both pipeline and SVC types
        if self.use_kernel_approximation:
            n_features = self.svm.named_steps['feature_map'].n_features_in_
        else:
            n_features = self.svm.n_features_in_
        
        ensemble_config = {
            'svm_weight': self.svm_weight,
            'dt_weight': self.dt_weight,
            'classes': self.classes_.tolist(),
            'n_features': n_features,
            'random_state': self.random_state,
            'use_kernel_approximation': self.use_kernel_approximation,
            'n_components': self.n_components
        }
        joblib.dump(ensemble_config, ensemble_path)
        
        # Set restrictive permissions (0o600)
        for path in [svm_path, dt_path, scaler_path, ensemble_path]:
            path.chmod(0o600)
        
        logger.info(f"Saved ensemble to {model_dir}")
        
        return {
            'svm': svm_path,
            'decision_tree': dt_path,
            'scaler': scaler_path,
            'ensemble': ensemble_path
        }
    
    @staticmethod
    def load(model_dir: Path) -> 'EnsembleClassifier':
        """
        Load ensemble from saved models.
        
        Parameters
        ----------
        model_dir : Path
            Directory containing saved models
        
        Returns
        -------
        ensemble : EnsembleClassifier
            Loaded ensemble classifier
        """
        model_dir = Path(model_dir)
        
        # Load metadata
        ensemble_path = model_dir / "ensemble_model.joblib"
        ensemble_config = joblib.load(ensemble_path)
        
        # Create ensemble instance
        ensemble = EnsembleClassifier(
            svm_weight=ensemble_config['svm_weight'],
            dt_weight=ensemble_config['dt_weight'],
            random_state=ensemble_config['random_state'],
            use_kernel_approximation=ensemble_config.get('use_kernel_approximation', False),
            n_components=ensemble_config.get('n_components', 500)
        )
        
        # Load individual models
        ensemble.svm = joblib.load(model_dir / "svm_model.joblib")
        ensemble.decision_tree = joblib.load(model_dir / "decision_tree_model.joblib")
        ensemble.scaler = joblib.load(model_dir / "feature_scaler.joblib")
        ensemble.classes_ = np.array(ensemble_config['classes'])
        ensemble.is_fitted = True
        
        logger.info(f"Loaded ensemble from {model_dir}")
        
        return ensemble


def optimize_ensemble_weights(
    X_val: np.ndarray,
    y_val: np.ndarray,
    svm_model,  # Can be SVC or Pipeline
    dt_model: DecisionTreeClassifier,
    scaler: StandardScaler,
    weight_range: Tuple[float, float, float] = (0.5, 0.9, 0.1)
) -> Dict[str, Any]:
    """
    Optimize ensemble weights using validation set.
    
    Grid-searches SVM weight in [weight_range[0], weight_range[1], weight_range[2]]
    and reports F1-scores to find optimal combination.
    
    Parameters
    ----------
    X_val : np.ndarray
        Validation features
    y_val : np.ndarray
        Validation labels
    svm_model : SVC or Pipeline
        Fitted SVM model (can be exact RBF SVC or Nystroem approximation pipeline)
    dt_model : DecisionTreeClassifier
        Fitted Decision Tree model
    scaler : StandardScaler
        Fitted feature scaler
    weight_range : Tuple[float, float, float]
        (start, stop, step) for svm_weight grid search
    
    Returns
    -------
    optimization_result : Dict[str, Any]
        Dictionary with 'best_svm_weight', 'best_dt_weight', 'best_f1', and 'results'
    """
    from sklearn.metrics import f1_score
    
    logger.info("Optimizing ensemble weights on validation set...")
    
    X_val_scaled = scaler.transform(X_val)
    
    results = []
    svm_weights = np.arange(weight_range[0], weight_range[1] + weight_range[2], weight_range[2])
    
    svm_proba = svm_model.predict_proba(X_val_scaled)
    dt_proba = dt_model.predict_proba(X_val)
    classes = svm_model.classes_
    
    best_f1 = -1
    best_svm_weight = 0.5
    best_dt_weight = 0.5
    
    for svm_w in svm_weights:
        dt_w = 1.0 - svm_w
        
        # Weighted ensemble
        ensemble_proba = svm_w * svm_proba + dt_w * dt_proba
        y_pred = classes[np.argmax(ensemble_proba, axis=1)]
        
        # Calculate weighted F1
        f1_weighted = f1_score(y_val, y_pred, average='weighted', zero_division=0)
        f1_macro = f1_score(y_val, y_pred, average='macro', zero_division=0)
        
        results.append({
            'svm_weight': svm_w,
            'dt_weight': dt_w,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro
        })
        
        if f1_weighted > best_f1:
            best_f1 = f1_weighted
            best_svm_weight = svm_w
            best_dt_weight = dt_w
    
    logger.info(
        f"Best weights: SVM={best_svm_weight:.1f}, DT={best_dt_weight:.1f}, "
        f"F1={best_f1:.4f}"
    )
    
    return {
        'best_svm_weight': best_svm_weight,
        'best_dt_weight': best_dt_weight,
        'best_f1': best_f1,
        'results': results
    }