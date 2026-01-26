"""Run Phase 4: Clustering on anomalies only (DBSCAN)."""

import sys
import logging
from pathlib import Path
import yaml
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import json

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.phase4_clustering import DBSCANClustering, ClusterAnalyzer


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
    config_path = project_root / 'config' / 'phase4_config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_latent_and_labels():
    latent_dir = project_root / 'data' / 'latent'
    features_dir = project_root / 'data' / 'features'
    preds_dir = project_root / 'results' / 'phase3'
    train_latent = np.load(latent_dir / 'train_latent_8d.npz')['X']
    val_latent = np.load(latent_dir / 'val_latent_8d.npz')['X']
    test_latent = np.load(latent_dir / 'test_latent_8d.npz')['X']

    y_train = np.load(features_dir / 'train_35features.npz')['y']
    y_val = np.load(features_dir / 'val_35features.npz')['y']
    y_test = np.load(features_dir / 'test_35features.npz')['y']

    pred_train = np.load(preds_dir / 'train_predictions.npy')
    pred_val = np.load(preds_dir / 'val_predictions.npy')
    pred_test = np.load(preds_dir / 'test_predictions.npy')

    return (
        train_latent, val_latent, test_latent,
        y_train, y_val, y_test,
        pred_train, pred_val, pred_test
    )


def main():
    config = load_config()
    log_file = project_root / config['logging']['file']
    setup_logging(log_file)
    logger = logging.getLogger(__name__)

    logger.info('=' * 80)
    logger.info('PHASE 4: CLUSTERING')
    logger.info('=' * 80)

    try:
        (X_train_latent, X_val_latent, X_test_latent,
         y_train, y_val, y_test,
         pred_train, pred_val, pred_test) = load_latent_and_labels()

        # STEP 1: Filter anomalies only (use validation set for clustering)
        logger.info('\n' + '=' * 80)
        logger.info('STEP 1: FILTER ANOMALIES (validation set)')
        logger.info('=' * 80)
        anomaly_mask_val = pred_val.reshape(-1) == 1
        X_val_anom = X_val_latent[anomaly_mask_val]
        y_val_anom = y_val[anomaly_mask_val]
        logger.info(f"Validation anomalies: {X_val_anom.shape[0]}/{X_val_latent.shape[0]}")

        # STEP 2: Normalize latent features to [0,1]
        logger.info('\n' + '=' * 80)
        logger.info('STEP 2: NORMALIZE LATENT (MinMaxScaler)')
        logger.info('=' * 80)
        scaler = MinMaxScaler()
        X_val_norm = scaler.fit_transform(X_val_anom)

        db_cfg = config['clustering']['dbscan']
        clusterer = DBSCANClustering(
            eps=db_cfg['eps'],
            min_samples=db_cfg['min_samples'],
            metric=db_cfg['metric'],
            algorithm=db_cfg['algorithm'],
            leaf_size=db_cfg['leaf_size']
        )

        logger.info('\n' + '=' * 80)
        logger.info('STEP 3: DBSCAN CLUSTERING (validation anomalies)')
        logger.info('=' * 80)
        labels_val = clusterer.fit_predict(X_val_norm)

        analyzer = ClusterAnalyzer(
            compute_silhouette=config['analysis']['compute_silhouette'],
            compute_db=config['analysis']['compute_davies_bouldin'],
            compute_ch=config['analysis']['compute_calinski_harabasz']
        )

        # STEP 4/5: Analysis & metrics (purity, dominant label, centroids)
        metrics = analyzer.evaluate(X_val_norm, labels_val)
        stats, centroids = analyzer.cluster_stats(X_val_norm, labels_val, y_val_anom)

        # STEP 6: Dimensionality reduction for visualization (optional) skipped here

        output_dir = project_root / config['data']['output_dir']
        output_dir.mkdir(parents=True, exist_ok=True)

        # STEP 7: Save outputs
        np.save(output_dir / 'val_clusters.npy', labels_val)
        np.save(output_dir / 'cluster_centroids_8d.npy', centroids)

        # Assignments CSV
        import csv
        assignments_path = output_dir / 'cluster_assignments.csv'
        with open(assignments_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['index', 'cluster', 'true_label'])
            for idx, (c, t) in enumerate(zip(labels_val, y_val_anom)):
                writer.writerow([idx, c, t])

        # Cluster statistics CSV
        stats_path = output_dir / 'cluster_statistics.csv'
        with open(stats_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['cluster', 'size', 'purity', 'dominant_label'])
            for row in stats:
                writer.writerow([row['cluster'], row['size'], row['purity'], row['dominant_label']])

        with open(output_dir / 'cluster_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        with open(output_dir / 'cluster_mapping.json', 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info('\n' + '=' * 80)
        logger.info('PHASE 4 COMPLETE!')
        logger.info('=' * 80)

    except Exception as e:
        logger.error(f"Error in Phase 4: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()