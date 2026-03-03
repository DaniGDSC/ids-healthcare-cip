"""
Phase 5: Final Classification Pipeline

Complete ensemble classification workflow:
1. Load Phase 4 outputs (latent features, cluster labels, anomaly predictions)
2. Load original Phase 2 selected features
3. Combine features (35D original + 8D latent = 43D hybrid)
4. Train SVM + Decision Tree ensemble with balanced class weights
5. Optimize ensemble weights on validation set
6. Evaluate on validation and test sets
7. Generate visualizations and save all outputs
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import json
import time
from datetime import datetime
import yaml

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.phase5_classification.ensemble_classifier import (
    EnsembleClassifier, optimize_ensemble_weights
)
from src.phase5_classification.evaluator import ClassificationEvaluator
from src.phase5_classification.visualizations import generate_all_visualizations
from src.utils.logger import get_logger, log_alert
from src.utils.checkpoint_manager import CheckpointManager

logger = logging.getLogger(__name__)


def load_config() -> Dict:
    """Load Phase 5 configuration."""
    config_path = project_root / 'config' / 'phase5_config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def _resolve_within_root(filepath: str) -> Path:
    """Safely resolve file path within project root."""
    p = Path(filepath).expanduser()
    if p.is_absolute():
        p = Path(*p.parts[1:])
    resolved = (project_root / p).resolve()
    resolved.relative_to(project_root.resolve())
    return resolved


def load_original_features() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load original selected features from Phase 2 (35D).
    
    Returns
    -------
    X_train, X_val, X_test : np.ndarray
        Original features [n_samples × 35]
    """
    logger.info("Loading original selected features (35D)...")
    
    features_dir = _resolve_within_root('data/features')
    
    train_path = features_dir / 'train_35features.npz'
    val_path = features_dir / 'val_35features.npz'
    test_path = features_dir / 'test_35features.npz'
    
    X_train = np.load(train_path)['X']
    X_val = np.load(val_path)['X']
    X_test = np.load(test_path)['X']
    
    logger.info(f"  Train: {X_train.shape}")
    logger.info(f"  Val:   {X_val.shape}")
    logger.info(f"  Test:  {X_test.shape}")
    
    return X_train, X_val, X_test


def load_latent_features() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load latent features from Phase 3 (8D).
    
    Returns
    -------
    X_train, X_val, X_test : np.ndarray
        Latent features [n_samples × 8]
    """
    logger.info("Loading latent features from autoencoder (8D)...")
    
    latent_dir = _resolve_within_root('data/latent')
    
    train_data = np.load(latent_dir / 'train_latent_8d.npz')
    val_data = np.load(latent_dir / 'val_latent_8d.npz')
    test_data = np.load(latent_dir / 'test_latent_8d.npz')
    
    X_train = train_data['latent']
    X_val = val_data['latent']
    X_test = test_data['latent']
    
    logger.info(f"  Train: {X_train.shape}")
    logger.info(f"  Val:   {X_val.shape}")
    logger.info(f"  Test:  {X_test.shape}")
    
    return X_train, X_val, X_test


def load_labels() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load ground truth labels from Phase 2 features.
    
    Returns
    -------
    y_train, y_val, y_test : np.ndarray
        Labels [0=Benign, 1=BruteForce, 2=WebAttack]
    """
    logger.info("Loading ground truth labels...")
    
    features_dir = _resolve_within_root('data/features')
    
    y_train = np.load(features_dir / 'train_35features.npz')['y']
    y_val = np.load(features_dir / 'val_35features.npz')['y']
    y_test = np.load(features_dir / 'test_35features.npz')['y']
    
    logger.info(f"  Train shape: {y_train.shape}")
    logger.info(f"  Train class distribution: {np.bincount(y_train)}")
    logger.info(f"  Val shape: {y_val.shape}")
    logger.info(f"  Val class distribution: {np.bincount(y_val)}")
    logger.info(f"  Test shape: {y_test.shape}")
    logger.info(f"  Test class distribution: {np.bincount(y_test)}")
    
    return y_train, y_val, y_test


def combine_features(
    original: Tuple[np.ndarray, np.ndarray, np.ndarray],
    latent: Tuple[np.ndarray, np.ndarray, np.ndarray]
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], list]:
    """
    Combine original features (35D) and latent features (8D) into hybrid (43D).
    
    Parameters
    ----------
    original : Tuple[np.ndarray, np.ndarray, np.ndarray]
        Original features
    latent : Tuple[np.ndarray, np.ndarray, np.ndarray]
        Latent features
    
    Returns
    -------
    combined : Tuple
        Combined features (X_train, X_val, X_test)
    feature_names : list
        Names of all 43 features
    """
    logger.info("Combining original (35D) and latent (8D) features...")
    
    X_orig_train, X_orig_val, X_orig_test = original
    X_lat_train, X_lat_val, X_lat_test = latent
    
    X_train = np.concatenate([X_orig_train, X_lat_train], axis=1)
    X_val = np.concatenate([X_orig_val, X_lat_val], axis=1)
    X_test = np.concatenate([X_orig_test, X_lat_test], axis=1)
    
    logger.info(f"  Combined train: {X_train.shape}")
    logger.info(f"  Combined val:   {X_val.shape}")
    logger.info(f"  Combined test:  {X_test.shape}")
    
    # Feature names
    original_names = [f"Orig_{i}" for i in range(X_orig_train.shape[1])]
    latent_names = [f"Latent_{i}" for i in range(X_lat_train.shape[1])]
    all_names = original_names + latent_names
    
    return (X_train, X_val, X_test), all_names


def train_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict
) -> Tuple[EnsembleClassifier, Dict]:
    """
    Train SVM + Decision Tree ensemble with balanced class weights.
    
    Parameters
    ----------
    X_train : np.ndarray, shape (n_samples, 43)
        Hybrid training features
    y_train : np.ndarray, shape (n_samples,)
        Training labels
    X_val : np.ndarray, shape (n_samples, 43)
        Hybrid validation features
    y_val : np.ndarray, shape (n_samples,)
        Validation labels
    Returns
    -------
    ensemble : EnsembleClassifier
        Trained ensemble
    weight_opt : Dict
        Weight optimization results
    """
    ensemble_cfg = config.get('classification', {}).get('ensemble', {})
    svm_weight = ensemble_cfg.get('svm_weight', 0.6)
    dt_weight = ensemble_cfg.get('dt_weight', 0.4)
    use_kernel_approx = ensemble_cfg.get('use_kernel_approximation', True)
    n_components = ensemble_cfg.get('n_components', 500)
    sgd_alpha = ensemble_cfg.get('svm_alpha', 0.0001)
    sgd_max_iter = ensemble_cfg.get('svm_max_iter', 1000)
    sgd_tol = ensemble_cfg.get('svm_tol', 1e-3)
    n_jobs = ensemble_cfg.get('n_jobs', -1)

    dt_cfg = config.get('classification', {}).get('models', {}).get('decision_tree', {})
    dt_max_depth = dt_cfg.get('max_depth', 30)
    dt_min_samples_split = dt_cfg.get('min_samples_split', 10)
    dt_min_samples_leaf = dt_cfg.get('min_samples_leaf', 5)
    dt_criterion = dt_cfg.get('criterion', 'gini')

    weight_opt_cfg = ensemble_cfg.get('weight_opt', {})
    weight_opt_enabled = weight_opt_cfg.get('enabled', True)
    weight_min = weight_opt_cfg.get('min', 0.5)
    weight_max = weight_opt_cfg.get('max', 0.9)
    weight_step = weight_opt_cfg.get('step', 0.1)

    logger.info("=" * 80)
    logger.info("STEP 1: TRAINING ENSEMBLE")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    # Create and fit ensemble
    ensemble = EnsembleClassifier(
        svm_weight=svm_weight,
        dt_weight=dt_weight,
        random_state=42,
        n_jobs=n_jobs,
        use_kernel_approximation=use_kernel_approx,
        n_components=n_components,
        sgd_alpha=sgd_alpha,
        sgd_max_iter=sgd_max_iter,
        sgd_tol=sgd_tol,
        dt_max_depth=dt_max_depth,
        dt_min_samples_split=dt_min_samples_split,
        dt_min_samples_leaf=dt_min_samples_leaf,
        dt_criterion=dt_criterion
    )
    
    ensemble.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    logger.info(f"Training complete in {training_time:.2f} seconds")
    
    weight_opt = {}
    if weight_opt_enabled:
        logger.info("\nOptimizing ensemble weights on validation set...")
        weight_opt = optimize_ensemble_weights(
            X_val, y_val,
            ensemble.svm,
            ensemble.decision_tree,
            ensemble.scaler,
            weight_range=(weight_min, weight_max, weight_step)
        )
        
        # Update ensemble with optimal weights
        best_svm_w = weight_opt['best_svm_weight']
        best_dt_w = weight_opt['best_dt_weight']
        ensemble.svm_weight = best_svm_w
        ensemble.dt_weight = best_dt_w
        
        logger.info(f"Optimized weights: SVM={best_svm_w:.1f}, DT={best_dt_w:.1f}")
        logger.info(f"Best validation F1: {weight_opt['best_f1']:.4f}")
    else:
        logger.info("Weight optimization disabled; using configured weights")
    
    return ensemble, weight_opt


def evaluate_ensemble(
    ensemble: EnsembleClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list
) -> Dict:
    """
    Comprehensive evaluation on all splits.
    
    Parameters
    ----------
    ensemble : EnsembleClassifier
        Trained ensemble
    X_train, y_train : np.ndarray
        Training data
    X_val, y_val : np.ndarray
        Validation data
    X_test, y_test : np.ndarray
        Test data
    feature_names : list
        Feature names for importance analysis
    
    Returns
    -------
    results : Dict
        All evaluation results
    """
    logger.info("=" * 80)
    logger.info("STEP 2: EVALUATING ENSEMBLE")
    logger.info("=" * 80)
    
    evaluator = ClassificationEvaluator(
        class_names=['Benign', 'BruteForce', 'WebAttack']
    )
    
    results = {}
    
    # Validation evaluation
    logger.info("\nValidation Set Evaluation:")
    y_val_pred = ensemble.predict(X_val)
    y_val_proba = ensemble.predict_proba(X_val)
    results['validation'] = evaluator.evaluate(
        y_val, y_val_pred, y_val_proba, "validation"
    )
    
    # Test evaluation
    logger.info("\nTest Set Evaluation:")
    y_test_pred = ensemble.predict(X_test)
    y_test_proba = ensemble.predict_proba(X_test)
    results['test'] = evaluator.evaluate(
        y_test, y_test_pred, y_test_proba, "test"
    )
    
    # Feature importance
    logger.info("\nExtracting feature importance...")
    importances = ensemble.get_feature_importance()
    top_indices = np.argsort(importances)[-10:][::-1]
    results['feature_importance'] = {
        feature_names[idx]: float(importances[idx])
        for idx in top_indices
    }
    logger.info("Top 10 important features:")
    for fname, imp in results['feature_importance'].items():
        logger.info(f"  {fname}: {imp:.4f}")
    
    # ROC/PR curves
    logger.info("\nGenerating ROC and PR curves...")
    results['roc_curves'] = evaluator.get_roc_curves(y_test, y_test_proba)
    results['pr_curves'] = evaluator.get_precision_recall_curves(y_test, y_test_proba)
    
    # Confidence statistics
    logger.info("\nAnalyzing confidence distributions...")
    results['confidence_stats'] = evaluator.get_confidence_statistics(y_test_proba)
    logger.info(f"  Mean confidence: {results['confidence_stats']['mean']:.4f}")
    logger.info(f"  High confidence (≥99.5%): {results['confidence_stats']['high_confidence_pct']:.1f}%")
    
    # Store predictions
    results['predictions'] = {
        'train_pred': y_train,
        'val_pred': y_val_pred,
        'test_pred': y_test_pred,
        'test_proba': y_test_proba
    }
    
    return results


def save_outputs(
    ensemble: EnsembleClassifier,
    results: Dict,
    feature_names: list,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    checkpoint_mgr: CheckpointManager = None
) -> Dict[str, Path]:
    """
    Save all models, predictions, and reports with versioning.
    
    Parameters
    ----------
    ensemble : EnsembleClassifier
        Trained ensemble
    results : Dict
        Evaluation results
    feature_names : list
        Feature names
    y_train, y_val, y_test : np.ndarray
        Ground truth labels
    checkpoint_mgr : CheckpointManager, optional
        Checkpoint manager for model versioning
    
    Returns
    -------
    saved_paths : Dict[str, Path]
        Dictionary of all saved file paths
    """
    logger.info("=" * 80)
    logger.info("STEP 3: SAVING OUTPUTS")
    logger.info("=" * 80)
    
    saved_paths = {}
    
    # Create output directories
    model_dir = _resolve_within_root('models/phase5')
    result_dir = _resolve_within_root('results/phase5')
    
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Save models (legacy path)
    logger.info("Saving trained models...")
    model_files = ensemble.save(model_dir)
    saved_paths['models'] = model_files
    
    # Save versioned models with checkpoint manager if provided
    if checkpoint_mgr:
        logger.info("Saving versioned model checkpoint...")
        model_metrics = {
            'test_accuracy': float(results['test']['accuracy']),
            'test_f1_weighted': float(results['test']['f1_weighted']),
            'val_accuracy': float(results['validation']['accuracy']),
            'val_f1_weighted': float(results['validation']['f1_weighted']),
        }
        checkpoint_mgr.save_model(
            ensemble,
            phase="phase5",
            model_name="ensemble_classifier",
            config={'ensemble': {
                'svm_weight': float(ensemble.svm_weight),
                'dt_weight': float(ensemble.dt_weight),
            }},
            metrics=model_metrics
        )
        logger.info("Model checkpoint saved with version tracking")
    
    # Save predictions
    logger.info("Saving predictions...")
    np.save(result_dir / 'train_predictions.npy', results['predictions']['train_pred'])
    np.save(result_dir / 'val_predictions.npy', results['predictions']['val_pred'])
    np.save(result_dir / 'test_predictions.npy', results['predictions']['test_pred'])
    np.save(result_dir / 'prediction_probabilities.npy', results['predictions']['test_proba'])
    saved_paths['predictions'] = [
        result_dir / 'train_predictions.npy',
        result_dir / 'val_predictions.npy',
        result_dir / 'test_predictions.npy',
        result_dir / 'prediction_probabilities.npy'
    ]
    
    # Save evaluation reports (JSON)
    logger.info("Saving evaluation reports...")
    
    # Remove non-serializable items for JSON
    val_report = {k: v for k, v in results['validation'].items() if k != 'confusion_matrix'}
    test_report = {k: v for k, v in results['test'].items() if k != 'confusion_matrix'}
    
    with open(result_dir / 'validation_report.json', 'w') as f:
        json.dump(val_report, f, indent=2)
    
    with open(result_dir / 'test_report.json', 'w') as f:
        json.dump(test_report, f, indent=2)
    
    # Save confusion matrices
    np.save(result_dir / 'val_confusion_matrix.npy', 
            np.array(results['validation']['confusion_matrix']))
    np.save(result_dir / 'test_confusion_matrix.npy',
            np.array(results['test']['confusion_matrix']))
    
    # Save feature importance
    with open(result_dir / 'feature_importance.json', 'w') as f:
        json.dump(results['feature_importance'], f, indent=2)
    
    # Save confidence statistics
    with open(result_dir / 'confidence_statistics.json', 'w') as f:
        json.dump(results['confidence_stats'], f, indent=2)
    
    # Save classification report (text)
    from src.phase5_classification.evaluator import ClassificationEvaluator
    evaluator = ClassificationEvaluator()
    with open(result_dir / 'classification_report.txt', 'w') as f:
        f.write(evaluator.get_classification_report(
            y_test, results['predictions']['test_pred']
        ))
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'dataset_sizes': {
            'train': int(len(y_train)),
            'validation': int(len(y_val)),
            'test': int(len(y_test))
        },
        'class_distribution': {
            'train': np.bincount(y_train).tolist(),
            'validation': np.bincount(y_val).tolist(),
            'test': np.bincount(y_test).tolist()
        },
        'ensemble_config': {
            'svm_weight': float(ensemble.svm_weight),
            'dt_weight': float(ensemble.dt_weight),
            'svm_kernel': 'rbf',
            'use_kernel_approximation': bool(ensemble.use_kernel_approximation),
            'n_components': int(ensemble.n_components) if ensemble.use_kernel_approximation else None,
            'svm_alpha': float(getattr(ensemble, 'sgd_alpha', 0.0001)),
            'svm_max_iter': int(getattr(ensemble, 'sgd_max_iter', 1000)),
            'svm_tol': float(getattr(ensemble, 'sgd_tol', 1e-3)),
            'dt_max_depth': int(getattr(ensemble, 'dt_max_depth', 30)),
            'dt_min_samples_split': int(getattr(ensemble, 'dt_min_samples_split', 10)),
            'dt_min_samples_leaf': int(getattr(ensemble, 'dt_min_samples_leaf', 5)),
            'dt_criterion': getattr(ensemble, 'dt_criterion', 'gini'),
            'feature_count': 43
        },
        'performance': {
            'val_accuracy': float(results['validation']['accuracy']),
            'test_accuracy': float(results['test']['accuracy']),
            'test_f1_weighted': float(results['test']['f1_weighted'])
        }
    }
    
    with open(result_dir / 'phase5_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"\nOutputs saved to:")
    logger.info(f"  Models: {model_dir}")
    logger.info(f"  Results: {result_dir}")
    
    return saved_paths


def main():
    """Main Phase 5 execution."""
    logger.info("=" * 80)
    logger.info("PHASE 5: FINAL CLASSIFICATION")
    logger.info("=" * 80)
    
    try:
        # Load configuration
        config = load_config()
        
        # Initialize checkpoint manager
        checkpoint_mgr = CheckpointManager(config.get('checkpointing', {}).get('checkpoint_dir', 'results/checkpoints'))
        
        # Load data
        logger.info("\n" + "=" * 80)
        logger.info("LOADING DATA")
        logger.info("=" * 80)
        
        orig_features = load_original_features()
        latent_features = load_latent_features()
        y_train, y_val, y_test = load_labels()
        
        # Combine features
        (X_train, X_val, X_test), feature_names = combine_features(
            orig_features, latent_features
        )
        
        logger.info(f"\nTotal features: {len(feature_names)}")
        
        # Train ensemble
        ensemble, weight_opt = train_ensemble(X_train, y_train, X_val, y_val, config)
        
        # Evaluate
        results = evaluate_ensemble(
            ensemble, X_train, y_train, X_val, y_val, X_test, y_test,
            feature_names
        )
        
        # Save outputs with versioning
        saved_paths = save_outputs(
            ensemble, results, feature_names, y_train, y_val, y_test,
            checkpoint_mgr=checkpoint_mgr
        )
        
        # Generate visualizations
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("=" * 80)
        generate_all_visualizations(
            results,
            feature_names,
            _resolve_within_root('results/phase5')
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 5 COMPLETE ✅")
        logger.info("=" * 80)
        logger.info(f"\nFinal Test Metrics:")
        logger.info(f"  Accuracy:  {results['test']['accuracy']:.4f}")
        logger.info(f"  F1-Score:  {results['test']['f1_weighted']:.4f}")
        logger.info(f"  Benign Recall:     {results['test']['per_class']['Benign']['recall']:.4f}")
        logger.info(f"  BruteForce Recall: {results['test']['per_class']['BruteForce']['recall']:.4f}")
        logger.info(f"  WebAttack Recall:  {results['test']['per_class']['WebAttack']['recall']:.4f}")
        logger.info("=" * 80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Phase 5 failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    log_file = project_root / 'logs' / 'phase5.log'
    logger = get_logger(__name__, level='INFO', log_file=str(log_file), structured=True)
    try:
        exit(main())
    except Exception as exc:
        log_alert(logger, "phase5_failed", error=str(exc))
        raise
