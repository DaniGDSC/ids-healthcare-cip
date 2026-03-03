"""
Data contract schemas for inter-phase validation.

Defines pydantic models for each phase's output to prevent silent data
contract failures between phases. Each schema documents what a phase
produces and what the next phase expects.
"""

from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field, validator, root_validator
import numpy as np
from datetime import datetime


class Phase1Output(BaseModel):
    """
    Phase 1 (Data Preprocessing) output contract.
    
    Guarantees:
    - Data is normalized
    - No missing values
    - Features are selected and correlated pairs removed
    - Data is split into train/val/test
    """
    
    # Core data artifacts
    X_train_normalized: np.ndarray = Field(..., description="Normalized training features [n_samples × n_features]")
    X_val_normalized: np.ndarray = Field(..., description="Normalized validation features")
    X_test_normalized: np.ndarray = Field(..., description="Normalized test features")
    
    y_train: np.ndarray = Field(..., description="Training labels [n_samples]")
    y_val: np.ndarray = Field(..., description="Validation labels")
    y_test: np.ndarray = Field(..., description="Test labels")
    
    # Metadata
    feature_names: List[str] = Field(..., description="Names of selected features")
    feature_count: int = Field(..., description="Number of features (len(feature_names))")
    
    # Statistics
    train_size: int = Field(..., description="Number of training samples")
    val_size: int = Field(..., description="Number of validation samples")
    test_size: int = Field(..., description="Number of test samples")
    
    # Audit
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="When output was created")
    config_hash: Optional[str] = Field(None, description="Hash of phase1_config.yaml for resume validation")
    
    class Config:
        arbitrary_types_allowed = True  # Allow numpy arrays
    
    @validator('feature_count')
    def feature_count_matches(cls, v, values):
        if 'feature_names' in values and v != len(values['feature_names']):
            raise ValueError(f"feature_count ({v}) must equal len(feature_names) ({len(values['feature_names'])})")
        return v
    
    @validator('train_size', 'val_size', 'test_size')
    def validate_split_sizes(cls, v):
        if v <= 0:
            raise ValueError("Split sizes must be positive")
        return v
    
    @root_validator
    def validate_shapes(cls, values):
        """Validate that data arrays have consistent shapes."""
        X_train = values.get('X_train_normalized')
        X_val = values.get('X_val_normalized')
        X_test = values.get('X_test_normalized')
        y_train = values.get('y_train')
        y_val = values.get('y_val')
        y_test = values.get('y_test')
        
        if X_train is not None and y_train is not None:
            if X_train.shape[0] != y_train.shape[0]:
                raise ValueError(f"X_train and y_train must have same number of samples: {X_train.shape[0]} vs {y_train.shape[0]}")
        
        if X_val is not None and y_val is not None:
            if X_val.shape[0] != y_val.shape[0]:
                raise ValueError(f"X_val and y_val must have same number of samples")
        
        if X_test is not None and y_test is not None:
            if X_test.shape[0] != y_test.shape[0]:
                raise ValueError(f"X_test and y_test must have same number of samples")
        
        # All X arrays should have same feature count
        if X_train is not None and X_val is not None:
            if X_train.shape[1] != X_val.shape[1]:
                raise ValueError(f"X_train and X_val must have same number of features")
        
        if X_train is not None and X_test is not None:
            if X_train.shape[1] != X_test.shape[1]:
                raise ValueError(f"X_train and X_test must have same number of features")
        
        return values


class Phase2Output(BaseModel):
    """
    Phase 2 (Feature Selection) output contract.
    
    Guarantees:
    - Features are selected (typically 35 for CIC-IDS2018)
    - Data shape matches Phase 1 (same train/val/test split sizes)
    - Feature names correspond to Phase 1 selected features
    """
    
    # Core data artifacts
    X_train_selected: np.ndarray = Field(..., description="Selected training features [n_samples × n_selected_features]")
    X_val_selected: np.ndarray = Field(..., description="Selected validation features")
    X_test_selected: np.ndarray = Field(..., description="Selected test features")
    
    y_train: np.ndarray = Field(..., description="Training labels (same as Phase 1)")
    y_val: np.ndarray = Field(..., description="Validation labels (same as Phase 1)")
    y_test: np.ndarray = Field(..., description="Test labels (same as Phase 1)")
    
    # Feature selection metadata
    selected_feature_names: List[str] = Field(..., description="Names of selected features")
    selected_feature_indices: List[int] = Field(..., description="Indices in Phase 1 feature list")
    n_selected: int = Field(..., description="Number of selected features")
    
    # Selection stages (optional, for debugging)
    selection_stages: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Metadata from each selection stage (IG, RF, RFE, etc.)"
    )
    
    # Audit
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    config_hash: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    @validator('n_selected')
    def n_selected_matches(cls, v, values):
        if 'selected_feature_names' in values and v != len(values['selected_feature_names']):
            raise ValueError(f"n_selected must equal len(selected_feature_names)")
        return v
    
    @root_validator
    def validate_selection_consistency(cls, values):
        """Validate feature indices match names."""
        indices = values.get('selected_feature_indices')
        names = values.get('selected_feature_names')
        
        if indices and names and len(indices) != len(names):
            raise ValueError("selected_feature_indices and selected_feature_names must have same length")
        
        return values


class Phase3Output(BaseModel):
    """
    Phase 3 (Autoencoder) output contract.
    
    Guarantees:
    - Latent features are 8-dimensional
    - Predictions are binary (0=benign, 1=anomaly)
    - Data shape matches Phase 2 (same train/val/test split sizes)
    """
    
    # Core latent artifacts
    X_train_latent: np.ndarray = Field(..., description="Training latent features [n_samples × 8]")
    X_val_latent: np.ndarray = Field(..., description="Validation latent features [n_samples × 8]")
    X_test_latent: np.ndarray = Field(..., description="Test latent features [n_samples × 8]")
    
    # Predictions
    train_predictions: np.ndarray = Field(..., description="Anomaly predictions for training [n_samples] (0=benign, 1=anomaly)")
    val_predictions: np.ndarray = Field(..., description="Anomaly predictions for validation")
    test_predictions: np.ndarray = Field(..., description="Anomaly predictions for test")
    
    # Reconstruction errors (optional, for analysis)
    train_reconstruction_errors: Optional[np.ndarray] = Field(None, description="Reconstruction errors per sample")
    val_reconstruction_errors: Optional[np.ndarray] = None
    test_reconstruction_errors: Optional[np.ndarray] = None
    
    # Anomaly threshold
    anomaly_threshold: float = Field(..., description="Reconstruction error threshold for anomaly detection")
    
    # Audit
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    config_hash: Optional[str] = None
    encoder_model_path: Optional[str] = Field(None, description="Path to saved encoder model")
    autoencoder_model_path: Optional[str] = Field(None, description="Path to saved autoencoder model")
    
    class Config:
        arbitrary_types_allowed = True
    
    @root_validator
    def validate_latent_dimensions(cls, values):
        """Latent features must be 8-dimensional."""
        X_train = values.get('X_train_latent')
        X_val = values.get('X_val_latent')
        X_test = values.get('X_test_latent')
        
        for name, X in [('X_train_latent', X_train), ('X_val_latent', X_val), ('X_test_latent', X_test)]:
            if X is not None and X.shape[1] != 8:
                raise ValueError(f"{name} must be 8-dimensional, got {X.shape[1]}")
        
        return values
    
    @root_validator
    def validate_predictions_binary(cls, values):
        """Predictions must be binary (0 or 1)."""
        for key in ['train_predictions', 'val_predictions', 'test_predictions']:
            preds = values.get(key)
            if preds is not None:
                unique = np.unique(preds)
                if not set(unique).issubset({0, 1}):
                    raise ValueError(f"{key} must contain only 0 (benign) or 1 (anomaly)")
        
        return values


class Phase4Output(BaseModel):
    """
    Phase 4 (Clustering) output contract.
    
    Guarantees:
    - Clustering applied to anomalies only
    - Cluster assignments are integers (negative values are noise points in DBSCAN)
    - Centroids are 8-dimensional (latent space)
    """
    
    # Cluster assignments
    val_cluster_assignments: np.ndarray = Field(..., description="Cluster labels for validation anomalies [n_anomalies] (>=0 is cluster, -1 is noise)")
    
    # Cluster centroids in latent space
    cluster_centroids_latent: np.ndarray = Field(..., description="Cluster centroids [n_clusters × 8]")
    
    # Cluster statistics
    cluster_stats: List[Dict[str, Any]] = Field(..., description="Per-cluster statistics (size, purity, dominant_label, etc.)")
    
    # Cluster metrics
    n_clusters: int = Field(..., description="Number of clusters found")
    silhouette_score: Optional[float] = Field(None, description="Silhouette coefficient (if computed)")
    davies_bouldin_score: Optional[float] = Field(None, description="Davies-Bouldin index (if computed)")
    calinski_harabasz_score: Optional[float] = Field(None, description="Calinski-Harabasz index (if computed)")
    
    # Audit
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    config_hash: Optional[str] = None
    dbscan_params: Dict[str, Any] = Field(default_factory=dict, description="DBSCAN hyperparameters (eps, min_samples, etc.)")
    
    class Config:
        arbitrary_types_allowed = True
    
    @validator('n_clusters')
    def n_clusters_positive(cls, v):
        if v < 1:
            raise ValueError("n_clusters must be >= 1")
        return v
    
    @root_validator
    def validate_centroid_dimensions(cls, values):
        """Cluster centroids must be 8-dimensional."""
        centroids = values.get('cluster_centroids_latent')
        if centroids is not None and centroids.shape[1] != 8:
            raise ValueError(f"Cluster centroids must be 8-dimensional, got {centroids.shape[1]}")
        return values


class Phase5Output(BaseModel):
    """
    Phase 5 (Classification) output contract.
    
    Guarantees:
    - Ensemble classifier with SVM and Decision Tree
    - Predictions are multi-class (0=Benign, 1=BruteForce, 2=WebAttack, etc.)
    - Features are 43-dimensional (35D original + 8D latent)
    """
    
    # Predictions
    val_predictions: np.ndarray = Field(..., description="Ensemble predictions for validation [n_samples]")
    test_predictions: np.ndarray = Field(..., description="Ensemble predictions for test [n_samples]")
    
    # Prediction probabilities (if available)
    val_probabilities: Optional[np.ndarray] = Field(None, description="Prediction probabilities [n_samples × n_classes]")
    test_probabilities: Optional[np.ndarray] = Field(None, description="Test probabilities")
    
    # Ensemble weights
    svm_weight: float = Field(..., ge=0, le=1, description="Weight of SVM in ensemble")
    tree_weight: float = Field(..., ge=0, le=1, description="Weight of Decision Tree in ensemble")
    
    # Performance metrics
    val_accuracy: float = Field(..., description="Validation accuracy")
    test_accuracy: float = Field(..., description="Test accuracy")
    
    val_f1_macro: float = Field(..., description="Validation F1-macro")
    test_f1_macro: float = Field(..., description="Test F1-macro")
    
    val_precision_weighted: float = Field(..., description="Validation precision (weighted)")
    test_precision_weighted: float = Field(..., description="Test precision (weighted)")
    
    val_recall_weighted: float = Field(..., description="Validation recall (weighted)")
    test_recall_weighted: float = Field(..., description="Test recall (weighted)")
    
    # Per-class metrics (optional)
    per_class_metrics: Optional[Dict[str, Dict[str, float]]] = Field(
        None,
        description="Per-class precision, recall, F1 scores"
    )
    
    # Feature information
    n_features: int = Field(..., description="Total features (should be 43)")
    feature_names: List[str] = Field(..., description="Names of features used")
    
    # Class information
    class_labels: List[int] = Field(..., description="Unique class labels found")
    class_names: Optional[List[str]] = Field(None, description="Human-readable class names")
    
    # Audit
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    config_hash: Optional[str] = None
    svm_model_path: Optional[str] = Field(None, description="Path to saved SVM model")
    tree_model_path: Optional[str] = Field(None, description="Path to saved Decision Tree model")
    
    class Config:
        arbitrary_types_allowed = True
    
    @validator('n_features')
    def validate_feature_count(cls, v):
        if v != 43:
            raise ValueError(f"Expected 43 features (35 original + 8 latent), got {v}")
        return v
    
    @validator('svm_weight', 'tree_weight')
    def validate_weights_sum(cls, v, values):
        """Weights should sum to 1.0 (or close to it)."""
        if 'svm_weight' in values and 'tree_weight' in values:
            total = values['svm_weight'] + values['tree_weight']
            if not (0.99 <= total <= 1.01):  # Allow small floating point errors
                raise ValueError(f"Ensemble weights must sum to ~1.0, got {total}")
        return v
    
    @root_validator
    def validate_metrics_in_range(cls, values):
        """All metrics should be in [0, 1] range."""
        metric_fields = [
            'val_accuracy', 'test_accuracy',
            'val_f1_macro', 'test_f1_macro',
            'val_precision_weighted', 'test_precision_weighted',
            'val_recall_weighted', 'test_recall_weighted'
        ]
        for field in metric_fields:
            value = values.get(field)
            if value is not None and not (0.0 <= value <= 1.0):
                raise ValueError(f"{field} must be in [0, 1], got {value}")
        
        return values


class DataContractValidator:
    """
    Utility class for validating data contracts between phases.
    
    Usage:
        # Validate Phase 1 output
        output = Phase1Output(
            X_train_normalized=X_train,
            X_val_normalized=X_val,
            X_test_normalized=X_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            feature_names=feature_names,
            feature_count=len(feature_names),
            train_size=X_train.shape[0],
            val_size=X_val.shape[0],
            test_size=X_test.shape[0]
        )
        
        # Validate that Phase 2 will accept this data
        DataContractValidator.validate_phase1_to_phase2(output)
    """
    
    @staticmethod
    def validate_phase1_to_phase2(phase1_output: Phase1Output) -> bool:
        """
        Validate that Phase 1 output meets Phase 2 input requirements.
        
        Returns:
            True if valid
            
        Raises:
            ValueError if validation fails
        """
        # Phase 2 expects normalized data with specific shape
        if phase1_output.feature_count < 1:
            raise ValueError("Phase 1 must output at least 1 feature")
        
        if phase1_output.feature_count > 100:
            raise ValueError(f"Phase 1 output too many features ({phase1_output.feature_count}); expected <100")
        
        # Validate that data is present
        if phase1_output.X_train_normalized is None or phase1_output.X_train_normalized.size == 0:
            raise ValueError("Phase 1 must provide X_train_normalized data")
        
        return True
    
    @staticmethod
    def validate_phase2_to_phase3(phase2_output: Phase2Output) -> bool:
        """Validate that Phase 2 output meets Phase 3 input requirements."""
        # Phase 3 expects selected features (typically 35 for CIC-IDS2018)
        if phase2_output.n_selected != 35:
            raise ValueError(f"Phase 3 expects 35 selected features, Phase 2 provided {phase2_output.n_selected}")
        
        if phase2_output.X_train_selected is None or phase2_output.X_train_selected.shape[1] != 35:
            raise ValueError("Phase 2 must provide X_train_selected with 35 features")
        
        return True
    
    @staticmethod
    def validate_phase3_to_phase4(phase3_output: Phase3Output) -> bool:
        """Validate that Phase 3 output meets Phase 4 input requirements."""
        # Phase 4 expects 8D latent features
        if phase3_output.X_train_latent.shape[1] != 8:
            raise ValueError("Phase 4 expects 8D latent features from Phase 3")
        
        # Phase 4 needs anomaly predictions
        if phase3_output.train_predictions is None:
            raise ValueError("Phase 3 must provide anomaly predictions")
        
        return True
    
    @staticmethod
    def validate_phase4_to_phase5(phase4_output: Phase4Output) -> bool:
        """Validate that Phase 4 output meets Phase 5 input requirements."""
        # Phase 5 doesn't directly depend on Phase 4 outputs
        # Phase 4 outputs are used for analysis/visualization
        # But good to validate structure
        
        if phase4_output.n_clusters < 1:
            raise ValueError("Phase 4 must find at least 1 cluster")
        
        if phase4_output.cluster_centroids_latent.shape[1] != 8:
            raise ValueError("Phase 4 cluster centroids must be 8-dimensional")
        
        return True
