"""
Unit tests for pydantic data contract schemas.

Tests schema validation, constraint enforcement, and inter-phase contract validation.
"""

import pytest
import numpy as np
from datetime import datetime
from src.schemas import (
    Phase1Output, Phase2Output, Phase3Output, Phase4Output, Phase5Output,
    DataContractValidator
)


class TestPhase1Output:
    """Tests for Phase 1 output schema."""
    
    def test_valid_phase1_output(self):
        """Valid Phase 1 output passes validation."""
        output = Phase1Output(
            X_train_normalized=np.random.randn(1000, 52),
            X_val_normalized=np.random.randn(250, 52),
            X_test_normalized=np.random.randn(250, 52),
            y_train=np.random.randint(0, 7, 1000),
            y_val=np.random.randint(0, 7, 250),
            y_test=np.random.randint(0, 7, 250),
            feature_names=[f"feature_{i}" for i in range(52)],
            feature_count=52,
            train_size=1000,
            val_size=250,
            test_size=250
        )
        assert output.feature_count == 52
        assert len(output.feature_names) == 52
    
    def test_feature_count_mismatch(self):
        """Feature count must match length of feature_names."""
        with pytest.raises(ValueError, match="feature_count"):
            Phase1Output(
                X_train_normalized=np.random.randn(1000, 52),
                X_val_normalized=np.random.randn(250, 52),
                X_test_normalized=np.random.randn(250, 52),
                y_train=np.random.randint(0, 7, 1000),
                y_val=np.random.randint(0, 7, 250),
                y_test=np.random.randint(0, 7, 250),
                feature_names=[f"feature_{i}" for i in range(50)],  # Mismatch
                feature_count=52,
                train_size=1000,
                val_size=250,
                test_size=250
            )
    
    def test_data_shape_mismatch_x_y(self):
        """X and y must have same number of samples."""
        with pytest.raises(ValueError, match="same number of samples"):
            Phase1Output(
                X_train_normalized=np.random.randn(1000, 52),
                X_val_normalized=np.random.randn(250, 52),
                X_test_normalized=np.random.randn(250, 52),
                y_train=np.random.randint(0, 7, 999),  # Wrong size
                y_val=np.random.randint(0, 7, 250),
                y_test=np.random.randint(0, 7, 250),
                feature_names=[f"feature_{i}" for i in range(52)],
                feature_count=52,
                train_size=1000,
                val_size=250,
                test_size=250
            )
    
    def test_feature_count_mismatch_x_arrays(self):
        """All X arrays must have same number of features."""
        with pytest.raises(ValueError, match="same number of features"):
            Phase1Output(
                X_train_normalized=np.random.randn(1000, 52),
                X_val_normalized=np.random.randn(250, 51),  # Wrong number of features
                X_test_normalized=np.random.randn(250, 52),
                y_train=np.random.randint(0, 7, 1000),
                y_val=np.random.randint(0, 7, 250),
                y_test=np.random.randint(0, 7, 250),
                feature_names=[f"feature_{i}" for i in range(52)],
                feature_count=52,
                train_size=1000,
                val_size=250,
                test_size=250
            )
    
    def test_negative_split_size(self):
        """Split sizes must be positive."""
        with pytest.raises(ValueError, match="must be positive"):
            Phase1Output(
                X_train_normalized=np.random.randn(1000, 52),
                X_val_normalized=np.random.randn(250, 52),
                X_test_normalized=np.random.randn(250, 52),
                y_train=np.random.randint(0, 7, 1000),
                y_val=np.random.randint(0, 7, 250),
                y_test=np.random.randint(0, 7, 250),
                feature_names=[f"feature_{i}" for i in range(52)],
                feature_count=52,
                train_size=-100,  # Invalid
                val_size=250,
                test_size=250
            )


class TestPhase2Output:
    """Tests for Phase 2 output schema."""
    
    def test_valid_phase2_output(self):
        """Valid Phase 2 output passes validation."""
        output = Phase2Output(
            X_train_selected=np.random.randn(1000, 35),
            X_val_selected=np.random.randn(250, 35),
            X_test_selected=np.random.randn(250, 35),
            y_train=np.random.randint(0, 7, 1000),
            y_val=np.random.randint(0, 7, 250),
            y_test=np.random.randint(0, 7, 250),
            selected_feature_names=[f"feature_{i}" for i in range(35)],
            selected_feature_indices=list(range(35)),
            n_selected=35
        )
        assert output.n_selected == 35
    
    def test_n_selected_mismatch(self):
        """n_selected must match length of feature lists."""
        with pytest.raises(ValueError, match="n_selected"):
            Phase2Output(
                X_train_selected=np.random.randn(1000, 35),
                X_val_selected=np.random.randn(250, 35),
                X_test_selected=np.random.randn(250, 35),
                y_train=np.random.randint(0, 7, 1000),
                y_val=np.random.randint(0, 7, 250),
                y_test=np.random.randint(0, 7, 250),
                selected_feature_names=[f"feature_{i}" for i in range(35)],
                selected_feature_indices=list(range(35)),
                n_selected=40  # Mismatch
            )
    
    def test_indices_names_mismatch(self):
        """Feature indices and names must have same length."""
        with pytest.raises(ValueError, match="same length"):
            Phase2Output(
                X_train_selected=np.random.randn(1000, 35),
                X_val_selected=np.random.randn(250, 35),
                X_test_selected=np.random.randn(250, 35),
                y_train=np.random.randint(0, 7, 1000),
                y_val=np.random.randint(0, 7, 250),
                y_test=np.random.randint(0, 7, 250),
                selected_feature_names=[f"feature_{i}" for i in range(35)],
                selected_feature_indices=list(range(40)),  # Mismatch
                n_selected=35
            )


class TestPhase3Output:
    """Tests for Phase 3 output schema."""
    
    def test_valid_phase3_output(self):
        """Valid Phase 3 output passes validation."""
        output = Phase3Output(
            X_train_latent=np.random.randn(1000, 8),
            X_val_latent=np.random.randn(250, 8),
            X_test_latent=np.random.randn(250, 8),
            train_predictions=np.random.randint(0, 2, 1000),
            val_predictions=np.random.randint(0, 2, 250),
            test_predictions=np.random.randint(0, 2, 250),
            anomaly_threshold=2.5
        )
        assert output.X_train_latent.shape[1] == 8
    
    def test_latent_dimension_not_8(self):
        """Latent features must be 8-dimensional."""
        with pytest.raises(ValueError, match="8-dimensional"):
            Phase3Output(
                X_train_latent=np.random.randn(1000, 10),  # Wrong
                X_val_latent=np.random.randn(250, 8),
                X_test_latent=np.random.randn(250, 8),
                train_predictions=np.random.randint(0, 2, 1000),
                val_predictions=np.random.randint(0, 2, 250),
                test_predictions=np.random.randint(0, 2, 250),
                anomaly_threshold=2.5
            )
    
    def test_predictions_not_binary(self):
        """Predictions must be binary (0 or 1)."""
        with pytest.raises(ValueError, match="only 0|1"):
            Phase3Output(
                X_train_latent=np.random.randn(1000, 8),
                X_val_latent=np.random.randn(250, 8),
                X_test_latent=np.random.randn(250, 8),
                train_predictions=np.random.randint(0, 3, 1000),  # Contains 2
                val_predictions=np.random.randint(0, 2, 250),
                test_predictions=np.random.randint(0, 2, 250),
                anomaly_threshold=2.5
            )


class TestPhase4Output:
    """Tests for Phase 4 output schema."""
    
    def test_valid_phase4_output(self):
        """Valid Phase 4 output passes validation."""
        output = Phase4Output(
            val_cluster_assignments=np.random.randint(-1, 5, 100),
            cluster_centroids_latent=np.random.randn(5, 8),
            cluster_stats=[
                {"cluster": 0, "size": 25, "purity": 0.8},
                {"cluster": 1, "size": 20, "purity": 0.75},
            ],
            n_clusters=5
        )
        assert output.cluster_centroids_latent.shape[1] == 8
    
    def test_centroid_not_8d(self):
        """Cluster centroids must be 8-dimensional."""
        with pytest.raises(ValueError, match="8-dimensional"):
            Phase4Output(
                val_cluster_assignments=np.random.randint(-1, 5, 100),
                cluster_centroids_latent=np.random.randn(5, 10),  # Wrong
                cluster_stats=[],
                n_clusters=5
            )


class TestPhase5Output:
    """Tests for Phase 5 output schema."""
    
    def test_valid_phase5_output(self):
        """Valid Phase 5 output passes validation."""
        output = Phase5Output(
            val_predictions=np.random.randint(0, 3, 250),
            test_predictions=np.random.randint(0, 3, 250),
            svm_weight=0.6,
            tree_weight=0.4,
            val_accuracy=0.92,
            test_accuracy=0.90,
            val_f1_macro=0.85,
            test_f1_macro=0.84,
            val_precision_weighted=0.88,
            test_precision_weighted=0.87,
            val_recall_weighted=0.85,
            test_recall_weighted=0.84,
            n_features=43,
            feature_names=[f"f_{i}" for i in range(43)],
            class_labels=[0, 1, 2]
        )
        assert output.n_features == 43
    
    def test_feature_count_not_43(self):
        """Phase 5 expects exactly 43 features (35 + 8 latent)."""
        with pytest.raises(ValueError, match="43 features"):
            Phase5Output(
                val_predictions=np.random.randint(0, 3, 250),
                test_predictions=np.random.randint(0, 3, 250),
                svm_weight=0.6,
                tree_weight=0.4,
                val_accuracy=0.92,
                test_accuracy=0.90,
                val_f1_macro=0.85,
                test_f1_macro=0.84,
                val_precision_weighted=0.88,
                test_precision_weighted=0.87,
                val_recall_weighted=0.85,
                test_recall_weighted=0.84,
                n_features=42,  # Wrong
                feature_names=[f"f_{i}" for i in range(42)],
                class_labels=[0, 1, 2]
            )
    
    def test_weights_dont_sum_to_one(self):
        """Ensemble weights must sum to ~1.0."""
        with pytest.raises(ValueError, match="sum to"):
            Phase5Output(
                val_predictions=np.random.randint(0, 3, 250),
                test_predictions=np.random.randint(0, 3, 250),
                svm_weight=0.6,
                tree_weight=0.5,  # Sums to 1.1
                val_accuracy=0.92,
                test_accuracy=0.90,
                val_f1_macro=0.85,
                test_f1_macro=0.84,
                val_precision_weighted=0.88,
                test_precision_weighted=0.87,
                val_recall_weighted=0.85,
                test_recall_weighted=0.84,
                n_features=43,
                feature_names=[f"f_{i}" for i in range(43)],
                class_labels=[0, 1, 2]
            )
    
    def test_metrics_out_of_range(self):
        """Metrics must be in [0, 1] range."""
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            Phase5Output(
                val_predictions=np.random.randint(0, 3, 250),
                test_predictions=np.random.randint(0, 3, 250),
                svm_weight=0.6,
                tree_weight=0.4,
                val_accuracy=1.5,  # Invalid
                test_accuracy=0.90,
                val_f1_macro=0.85,
                test_f1_macro=0.84,
                val_precision_weighted=0.88,
                test_precision_weighted=0.87,
                val_recall_weighted=0.85,
                test_recall_weighted=0.84,
                n_features=43,
                feature_names=[f"f_{i}" for i in range(43)],
                class_labels=[0, 1, 2]
            )


class TestDataContractValidator:
    """Tests for inter-phase contract validation."""
    
    def test_phase1_to_phase2_valid(self):
        """Valid Phase 1 output satisfies Phase 2 contract."""
        phase1_output = Phase1Output(
            X_train_normalized=np.random.randn(1000, 52),
            X_val_normalized=np.random.randn(250, 52),
            X_test_normalized=np.random.randn(250, 52),
            y_train=np.random.randint(0, 7, 1000),
            y_val=np.random.randint(0, 7, 250),
            y_test=np.random.randint(0, 7, 250),
            feature_names=[f"feature_{i}" for i in range(52)],
            feature_count=52,
            train_size=1000,
            val_size=250,
            test_size=250
        )
        assert DataContractValidator.validate_phase1_to_phase2(phase1_output)
    
    def test_phase1_to_phase2_too_few_features(self):
        """Phase 1 must provide at least 1 feature."""
        phase1_output = Phase1Output(
            X_train_normalized=np.random.randn(1000, 0),  # No features
            X_val_normalized=np.random.randn(250, 0),
            X_test_normalized=np.random.randn(250, 0),
            y_train=np.random.randint(0, 7, 1000),
            y_val=np.random.randint(0, 7, 250),
            y_test=np.random.randint(0, 7, 250),
            feature_names=[],
            feature_count=0,
            train_size=1000,
            val_size=250,
            test_size=250
        )
        with pytest.raises(ValueError, match="at least 1 feature"):
            DataContractValidator.validate_phase1_to_phase2(phase1_output)
    
    def test_phase2_to_phase3_requires_35_features(self):
        """Phase 3 expects exactly 35 selected features."""
        phase2_output = Phase2Output(
            X_train_selected=np.random.randn(1000, 40),  # Wrong
            X_val_selected=np.random.randn(250, 40),
            X_test_selected=np.random.randn(250, 40),
            y_train=np.random.randint(0, 7, 1000),
            y_val=np.random.randint(0, 7, 250),
            y_test=np.random.randint(0, 7, 250),
            selected_feature_names=[f"feature_{i}" for i in range(40)],
            selected_feature_indices=list(range(40)),
            n_selected=40
        )
        with pytest.raises(ValueError, match="35 selected features"):
            DataContractValidator.validate_phase2_to_phase3(phase2_output)
    
    def test_phase3_to_phase4_requires_8d_latent(self):
        """Phase 4 expects 8D latent features."""
        phase3_output = Phase3Output(
            X_train_latent=np.random.randn(1000, 10),  # Wrong (should be 8)
            X_val_latent=np.random.randn(250, 10),
            X_test_latent=np.random.randn(250, 10),
            train_predictions=np.random.randint(0, 2, 1000),
            val_predictions=np.random.randint(0, 2, 250),
            test_predictions=np.random.randint(0, 2, 250),
            anomaly_threshold=2.5
        )
        # Schema validation will catch this before DataContractValidator
        with pytest.raises(ValueError, match="8-dimensional"):
            DataContractValidator.validate_phase3_to_phase4(phase3_output)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
