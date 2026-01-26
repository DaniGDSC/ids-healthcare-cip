"""Run Phase 3: ML filtering, Autoencoder training, and evaluation."""

import sys
import logging
from pathlib import Path
import yaml
import numpy as np
import json
import hashlib
import os
from typing import Tuple, Dict

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.settings import RANDOM_STATE
from src.phase3_autoencoder import (
    build_autoencoder,
    AutoencoderTrainer,
    AutoencoderEvaluator,
    ThresholdOptimizer
)
from src.phase3_autoencoder.ml_filter import MLFilter


def setup_logging(log_file: str):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def load_config():
    config_path = project_root / 'config' / 'phase3_config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_features() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    input_dir = project_root / 'data' / 'features'
    train = np.load(input_dir / 'train_35features.npz')
    val = np.load(input_dir / 'val_35features.npz')
    test = np.load(input_dir / 'test_35features.npz')
    return train['X'], train['y'], val['X'], val['y'], test['X'], test['y']


def save_models(autoencoder, encoder, decoder, model_dir: Path):
    model_dir.mkdir(parents=True, exist_ok=True)
    auto_path = model_dir / 'autoencoder.keras'
    enc_path = model_dir / 'encoder.keras'
    dec_path = model_dir / 'decoder.keras'
    autoencoder.save(auto_path)
    encoder.save(enc_path)
    decoder.save(dec_path)
    for p in [auto_path, enc_path, dec_path]:
        try:
            os.chmod(p, 0o600)
        except Exception:
            pass


def save_json(path: Path, data: Dict[str, any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass

def _resolve_within_root(path: Path) -> Path:
    """Resolve path and ensure it is within project_root to prevent traversal/symlink escape."""
    resolved = path.resolve()
    # Ensure resolved path is inside project_root
    resolved.relative_to(project_root)
    return resolved

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    config = load_config()
    log_file = project_root / config['logging']['file']
    setup_logging(log_file)
    logger = logging.getLogger(__name__)

    logger.info('=' * 80)
    logger.info('PHASE 3: AUTOENCODER')
    logger.info('=' * 80)

    try:
        X_train, y_train, X_val, y_val, X_test, y_test = load_features()

        # STEP 0: ML FILTERING (reduce benign by ~80% while maintaining recall)
        logger.info('\n' + '=' * 80)
        logger.info('STEP 0: ML FILTERING (RandomForest)')
        logger.info('=' * 80)
        ml_cfg = config['ml_filter']
        mlf = MLFilter(
            n_estimators=ml_cfg.get('n_estimators', 100),
            max_depth=ml_cfg.get('max_depth', 12),
            min_samples_split=ml_cfg.get('min_samples_split', 8),
            random_state=ml_cfg.get('random_state', 42),
            n_jobs=ml_cfg.get('n_jobs', -1)
        )
        mlf.fit(X_train, y_train)
        thr = mlf.calibrate_threshold(X_val, y_val, recall_target=ml_cfg.get('recall_target', 0.995))
        logger.info(f"Calibrated attack probability threshold: {thr:.4f}")

        X_train_pass, y_train_pass, mask_train_pass = mlf.filter_benign(X_train, y_train)
        X_val_pass, y_val_pass, mask_val_pass = mlf.filter_benign(X_val, y_val)
        X_test_pass, y_test_pass, mask_test_pass = mlf.filter_benign(X_test, y_test)
        logger.info(f"Train pass: {X_train_pass.shape[0]}/{X_train.shape[0]} samples")
        logger.info(f"Val pass:   {X_val_pass.shape[0]}/{X_val.shape[0]} samples")
        logger.info(f"Test pass:  {X_test_pass.shape[0]}/{X_test.shape[0]} samples")

        # Save ML filter model and config
        model_dir = _resolve_within_root(project_root / config['data']['model_dir'])
        model_dir.mkdir(parents=True, exist_ok=True)
        mlf.save(str(model_dir / 'ml_filter.pkl'))
        save_json(model_dir / 'ml_filter_config.json', {
            'benign_reduction': float(np.mean(y_train[~mask_train_pass] == 0)),
            'recall_threshold': ml_cfg.get('recall_target', 0.995),
            'samples_filtered': int(np.sum(~mask_train_pass)),
            'samples_passed': int(np.sum(mask_train_pass)),
            'threshold': float(thr)
        })

        arch = config['architecture']
        model, encoder, decoder = build_autoencoder(
            input_dim=arch['input_dim'],
            encoder_layers=arch['encoder_layers'],
            decoder_layers=arch['decoder_layers'],
            latent_dim=arch['latent_dim'],
            activation=arch['activation'],
            output_activation=arch['output_activation'],
            dropout_rate=arch['dropout_rate'],
            use_batch_norm=arch['use_batch_normalization'],
            l2_reg=arch['l2_regularization']
        )

        trainer_cfg = config['training']
        trainer = AutoencoderTrainer(
            optimizer=trainer_cfg['optimizer'],
            learning_rate=trainer_cfg['learning_rate'],
            batch_size=trainer_cfg['batch_size'],
            epochs=trainer_cfg['epochs'],
            early_stopping=trainer_cfg['early_stopping'],
            reduce_lr=trainer_cfg['reduce_lr'],
            validation_split=trainer_cfg['validation_split'],
            shuffle=trainer_cfg['shuffle']
        )

        trainer.compile_model(model)

        tensorboard_dir = None
        if config['logging'].get('tensorboard', False):
            tensorboard_dir = project_root / config['logging']['tensorboard_dir']

        # STEP 1: Train autoencoder on BENIGN ONLY from passed set
        benign_mask_train = (y_train_pass == 0)
        benign_mask_val = (y_val_pass == 0)
        X_train_benign = X_train_pass[benign_mask_train]
        X_val_benign = X_val_pass[benign_mask_val]
        logger.info(f"Train benign: {X_train_benign.shape[0]} samples")
        logger.info(f"Val benign:   {X_val_benign.shape[0]} samples")

        history = trainer.train(model, X_train_benign, X_val_benign, tensorboard_dir=tensorboard_dir)

        evaluator = AutoencoderEvaluator()
        metrics = evaluator.evaluate(model, X_val_benign, metrics={m: True for m in config['evaluation']['metrics']})

        # STEP 5/6: Threshold optimization on validation (passed set)
        errors_val = evaluator.reconstruction_errors(model, X_val_pass)
        threshold_cfg = config['anomaly_detection']
        threshold_opt = ThresholdOptimizer(
            method=threshold_cfg['threshold_method'],
            percentile=threshold_cfg.get('percentile', 95),
            n_std=threshold_cfg.get('n_std', 3)
        )
        # Grid search F1 across percentiles P50..P99 on val set
        percentiles = np.linspace(50, 99, 50)
        best_thr, thr_metrics = threshold_opt.grid_search_f1(errors_val, (y_val_pass == 1).astype(int), percentiles)
        threshold = best_thr

        # Save models and outputs
        model_dir = _resolve_within_root(project_root / config['data']['model_dir'])
        save_models(model, encoder, decoder, model_dir)

        latent_dir = _resolve_within_root(project_root / config['data']['output_dir'])
        latent_dir.mkdir(parents=True, exist_ok=True)
        # STEP 4: Extract latent features for PASSED sets
        latent_train = encoder.predict(X_train_pass, verbose=0)
        latent_val = encoder.predict(X_val_pass, verbose=0)
        latent_test = encoder.predict(X_test_pass, verbose=0)

        train_latent_path = latent_dir / 'train_latent_8d.npz'
        val_latent_path = latent_dir / 'val_latent_8d.npz'
        test_latent_path = latent_dir / 'test_latent_8d.npz'
        np.savez_compressed(train_latent_path, X=latent_train)
        np.savez_compressed(val_latent_path, X=latent_val)
        np.savez_compressed(test_latent_path, X=latent_test)
        for p in [train_latent_path, val_latent_path, test_latent_path]:
            try:
                os.chmod(p, 0o600)
            except Exception:
                pass

        # STEP 7/8/9: Predictions, evaluation, and save metadata
        # Compute anomaly scores on passed sets
        errors_train = evaluator.reconstruction_errors(model, X_train_pass)
        errors_test = evaluator.reconstruction_errors(model, X_test_pass)
        # Predictions
        y_pred_train = (errors_train > threshold).astype(int)
        y_pred_val = (errors_val > threshold).astype(int)
        y_pred_test = (errors_test > threshold).astype(int)

        # Save anomaly scores and predictions
        results_dir = _resolve_within_root(project_root / 'results' / 'phase3')
        results_dir.mkdir(parents=True, exist_ok=True)
        paths = {
            'train_anomaly_scores': results_dir / 'train_anomaly_scores.npy',
            'val_anomaly_scores': results_dir / 'val_anomaly_scores.npy',
            'test_anomaly_scores': results_dir / 'test_anomaly_scores.npy',
            'train_predictions': results_dir / 'train_predictions.npy',
            'val_predictions': results_dir / 'val_predictions.npy',
            'test_predictions': results_dir / 'test_predictions.npy'
        }
        np.save(paths['train_anomaly_scores'], errors_train)
        np.save(paths['val_anomaly_scores'], errors_val)
        np.save(paths['test_anomaly_scores'], errors_test)
        np.save(paths['train_predictions'], y_pred_train)
        np.save(paths['val_predictions'], y_pred_val)
        np.save(paths['test_predictions'], y_pred_test)
        for p in paths.values():
            try:
                os.chmod(p, 0o600)
            except Exception:
                pass

        # Save threshold config with metrics
        threshold_path = model_dir / 'threshold_config.json'
        save_json(threshold_path, {
            'optimal_threshold': float(threshold),
            'f1_score': float(thr_metrics.get('f1', 0.0)),
            'precision': float(thr_metrics.get('precision', 0.0)),
            'recall': float(thr_metrics.get('recall', 0.0)),
            'calibration_set': 'validation',
            'ml_filter_reduction': float(np.mean(y_train == 0) - np.mean(y_train_pass == 0))
        })

        # Evaluation metrics on validation passed set
        val_metrics = evaluator.binary_metrics((y_val_pass == 1).astype(int), y_pred_val)
        metadata_path = results_dir / 'phase3_metadata.json'
        save_json(metadata_path, {
            'latent_shapes': {
                'train': list(latent_train.shape),
                'val': list(latent_val.shape),
                'test': list(latent_test.shape)
            },
            'validation_metrics': val_metrics,
            'threshold_metrics': thr_metrics,
            'ml_filter': {
                'train_passed': int(X_train_pass.shape[0]),
                'val_passed': int(X_val_pass.shape[0]),
                'test_passed': int(X_test_pass.shape[0])
            }
        })

        # Integrity manifest (SHA256) for key outputs
        manifest = {
            'models': {
                'autoencoder.keras': _sha256_file(model_dir / 'autoencoder.keras'),
                'encoder.keras': _sha256_file(model_dir / 'encoder.keras'),
                'decoder.keras': _sha256_file(model_dir / 'decoder.keras'),
                'ml_filter.pkl': _sha256_file(model_dir / 'ml_filter.pkl'),
                'threshold_config.json': _sha256_file(threshold_path)
            },
            'latent': {
                'train_latent_8d.npz': _sha256_file(train_latent_path),
                'val_latent_8d.npz': _sha256_file(val_latent_path),
                'test_latent_8d.npz': _sha256_file(test_latent_path)
            },
            'results': {k + '.npy': _sha256_file(p) for k, p in paths.items()},
            'metadata': {
                'phase3_metadata.json': _sha256_file(metadata_path)
            }
        }
        save_json(results_dir / 'manifest.json', manifest)

        logger.info('\n' + '=' * 80)
        logger.info('PHASE 3 COMPLETE!')
        logger.info('=' * 80)

    except Exception as e:
        logger.error(f"Error in Phase 3: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()