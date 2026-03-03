"""Checkpoint and artifact management with versioning and recovery support."""

import json
import logging
import hashlib
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple, Type
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manage phase checkpoints, versioning, and recovery.
    
    Saves intermediate artifacts with metadata (timestamp, phase, config hash, stats).
    Supports resuming from last good checkpoint and version selection.
    """
    
    def __init__(self, checkpoint_dir: str = "results/checkpoints"):
        """
        Initialize CheckpointManager.
        
        Args:
            checkpoint_dir: Root directory for storing checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _compute_config_hash(self, config: Dict[str, Any]) -> str:
        """Compute a stable hash of config dict for artifact versioning."""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def _build_checkpoint_path(
        self,
        phase: str,
        artifact_type: str,
        version: Optional[str] = None
    ) -> Tuple[Path, str]:
        """
        Build checkpoint directory path and filename.
        
        Args:
            phase: Phase name (e.g., "phase1", "phase3")
            artifact_type: Type of artifact (e.g., "data", "model", "encoder")
            version: Optional version string (defaults to timestamp)
            
        Returns:
            Tuple of (directory_path, filename_stem)
        """
        phase_dir = self.checkpoint_dir / phase
        phase_dir.mkdir(parents=True, exist_ok=True)
        
        if version is None:
            version = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        filename_stem = f"{artifact_type}_{version}"
        return phase_dir, filename_stem
    
    def save_dataframe(
        self,
        df: pd.DataFrame,
        phase: str,
        artifact_name: str = "data",
        config: Optional[Dict[str, Any]] = None,
        stats: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None,
        format: str = "feather"
    ) -> Dict[str, Any]:
        """
        Save a DataFrame checkpoint with metadata.
        
        Args:
            df: DataFrame to checkpoint
            phase: Phase name
            artifact_name: Artifact identifier (e.g., "data", "features")
            config: Configuration dict for hashing
            stats: Optional statistics dict to store in metadata
            version: Optional version string
            format: "feather" (fast, binary) or "parquet" (columnar)
            
        Returns:
            Metadata dict with checkpoint info
        """
        phase_dir, filename_stem = self._build_checkpoint_path(phase, artifact_name, version)
        
        # Save artifact
        if format == "feather":
            data_path = phase_dir / f"{filename_stem}.feather"
            df.to_feather(data_path)
        elif format == "parquet":
            data_path = phase_dir / f"{filename_stem}.parquet"
            df.to_parquet(data_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        # Build metadata
        metadata = {
            "timestamp": datetime.utcnow().isoformat(),
            "version": filename_stem.split("_", 1)[1] if "_" in filename_stem else "",
            "phase": phase,
            "artifact_type": artifact_name,
            "format": format,
            "file_path": str(data_path.relative_to(self.checkpoint_dir)),
            "dataframe_shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1_000_000, 3),
        }
        
        if config:
            metadata["config_hash"] = self._compute_config_hash(config)
        
        if stats:
            metadata["stats"] = stats
        
        # Save metadata
        meta_path = phase_dir / f"{filename_stem}.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(json.dumps({
            "event": "checkpoint",
            "action": "save",
            "phase": phase,
            "artifact": artifact_name,
            "shape": metadata["dataframe_shape"],
            "memory_mb": metadata["memory_mb"],
            "file": str(data_path.relative_to(self.checkpoint_dir))
        }))
        
        return metadata
    
    def load_dataframe(
        self,
        phase: str,
        artifact_name: str = "data",
        version: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load a DataFrame checkpoint.
        
        Args:
            phase: Phase name
            artifact_name: Artifact identifier
            version: Optional version string; if None, loads latest
            
        Returns:
            Tuple of (DataFrame, metadata_dict)
            
        Raises:
            FileNotFoundError: If checkpoint not found
        """
        phase_dir = self.checkpoint_dir / phase
        
        if not phase_dir.exists():
            raise FileNotFoundError(f"No checkpoint directory for {phase}")
        
        # Find checkpoint files
        if version:
            filename_stem = f"{artifact_name}_{version}"
        else:
            # Find latest version
            candidates = list(phase_dir.glob(f"{artifact_name}_*.feather")) + \
                         list(phase_dir.glob(f"{artifact_name}_*.parquet"))
            if not candidates:
                raise FileNotFoundError(f"No checkpoint for {phase}/{artifact_name}")
            # Sort by modification time, get latest
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            data_path = candidates[0]
            filename_stem = data_path.stem
        
        # Load data
        feather_path = phase_dir / f"{filename_stem}.feather"
        parquet_path = phase_dir / f"{filename_stem}.parquet"
        
        if feather_path.exists():
            df = pd.read_feather(feather_path)
        elif parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        else:
            raise FileNotFoundError(f"Checkpoint not found: {filename_stem}")
        
        # Load metadata
        meta_path = phase_dir / f"{filename_stem}.json"
        metadata = {}
        if meta_path.exists():
            with open(meta_path, "r") as f:
                metadata = json.load(f)
        
        logger.info(json.dumps({
            "event": "checkpoint",
            "action": "load",
            "phase": phase,
            "artifact": artifact_name,
            "shape": df.shape,
            "version": filename_stem.split("_", 1)[1] if "_" in filename_stem else ""
        }))
        
        return df, metadata
    
    def save_model(
        self,
        model: Any,
        phase: str,
        model_name: str = "model",
        config: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Save a model (sklearn, xgboost, custom) with metadata.
        
        Args:
            model: Model object (must be pickleable)
            phase: Phase name
            model_name: Model identifier (e.g., "svm", "classifier")
            config: Configuration dict
            metrics: Performance metrics dict
            version: Optional version string
            
        Returns:
            Metadata dict
        """
        phase_dir, filename_stem = self._build_checkpoint_path(phase, model_name, version)
        
        # Save model
        model_path = phase_dir / f"{filename_stem}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        # Build metadata
        metadata = {
            "timestamp": datetime.utcnow().isoformat(),
            "version": filename_stem.split("_", 1)[1] if "_" in filename_stem else "",
            "phase": phase,
            "model_name": model_name,
            "model_type": type(model).__name__,
            "file_path": str(model_path.relative_to(self.checkpoint_dir)),
        }
        
        if config:
            metadata["config_hash"] = self._compute_config_hash(config)
        
        if metrics:
            metadata["metrics"] = metrics
        
        # Save metadata
        meta_path = phase_dir / f"{filename_stem}.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(json.dumps({
            "event": "checkpoint",
            "action": "save_model",
            "phase": phase,
            "model": model_name,
            "type": metadata["model_type"],
            "file": str(model_path.relative_to(self.checkpoint_dir))
        }))
        
        return metadata
    
    def load_model(
        self,
        phase: str,
        model_name: str = "model",
        version: Optional[str] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Load a pickled model checkpoint.
        
        Args:
            phase: Phase name
            model_name: Model identifier
            version: Optional version string; if None, loads latest
            
        Returns:
            Tuple of (model_object, metadata_dict)
        """
        phase_dir = self.checkpoint_dir / phase
        
        if not phase_dir.exists():
            raise FileNotFoundError(f"No checkpoint directory for {phase}")
        
        # Find checkpoint files
        if version:
            filename_stem = f"{model_name}_{version}"
        else:
            # Find latest version
            candidates = list(phase_dir.glob(f"{model_name}_*.pkl"))
            if not candidates:
                raise FileNotFoundError(f"No model checkpoint for {phase}/{model_name}")
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            model_path = candidates[0]
            filename_stem = model_path.stem
        
        # Load model
        model_path = phase_dir / f"{filename_stem}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {filename_stem}")
        
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        # Load metadata
        meta_path = phase_dir / f"{filename_stem}.json"
        metadata = {}
        if meta_path.exists():
            with open(meta_path, "r") as f:
                metadata = json.load(f)
        
        logger.info(json.dumps({
            "event": "checkpoint",
            "action": "load_model",
            "phase": phase,
            "model": model_name,
            "type": metadata.get("model_type", "unknown"),
            "version": filename_stem.split("_", 1)[1] if "_" in filename_stem else ""
        }))
        
        return model, metadata
    
    def list_checkpoints(self, phase: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        List all available checkpoints.
        
        Args:
            phase: Optional phase to filter by; if None, lists all phases
            
        Returns:
            Dict mapping phase -> list of checkpoint metadata dicts
        """
        results = {}
        
        phases = [phase] if phase else [d.name for d in self.checkpoint_dir.iterdir() if d.is_dir()]
        
        for p in phases:
            phase_dir = self.checkpoint_dir / p
            if not phase_dir.exists():
                continue
            
            checkpoints = []
            for meta_file in sorted(phase_dir.glob("*.json"), reverse=True):
                try:
                    with open(meta_file, "r") as f:
                        meta = json.load(f)
                    checkpoints.append(meta)
                except Exception as e:
                    logger.warning(f"Failed to read checkpoint metadata {meta_file}: {e}")
            
            results[p] = checkpoints
        
        return results
    
    def get_latest_checkpoint(self, phase: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for the latest checkpoint in a phase.
        
        Args:
            phase: Phase name
            
        Returns:
            Metadata dict or None if no checkpoints found
        """
        checkpoints = self.list_checkpoints(phase)
        if phase in checkpoints and checkpoints[phase]:
            return checkpoints[phase][0]  # Already sorted by mtime desc
        return None
    
    def has_checkpoint(self, phase: str, artifact_name: str = "data") -> bool:
        """Check if a checkpoint exists for a phase/artifact."""
        try:
            phase_dir = self.checkpoint_dir / phase
            candidates = list(phase_dir.glob(f"{artifact_name}_*.feather")) + \
                         list(phase_dir.glob(f"{artifact_name}_*.parquet"))
            return len(candidates) > 0
        except Exception:
            return False
    
    def clear_checkpoints(self, phase: Optional[str] = None, keep_latest: int = 0):
        """
        Clear checkpoints.
        
        Args:
            phase: If provided, clear only this phase; else clear all
            keep_latest: Keep N latest checkpoints per phase
        """
        if phase:
            phases = [phase]
        else:
            phases = [d.name for d in self.checkpoint_dir.iterdir() if d.is_dir()]
        
        for p in phases:
            phase_dir = self.checkpoint_dir / p
            if not phase_dir.exists():
                continue
            
            # Group checkpoints by artifact type/name
            artifacts = {}
            for meta_file in phase_dir.glob("*.json"):
                try:
                    with open(meta_file, "r") as f:
                        meta = json.load(f)
                    artifact_key = f"{meta.get('artifact_type', meta.get('model_name', 'unknown'))}"
                    if artifact_key not in artifacts:
                        artifacts[artifact_key] = []
                    artifacts[artifact_key].append((meta_file, meta_file.stat().st_mtime))
                except Exception:
                    pass
            
            # Remove old checkpoints, keep N latest
            for artifact_key, files in artifacts.items():
                files.sort(key=lambda x: x[1], reverse=True)
                for meta_file, _ in files[keep_latest:]:
                    try:
                        stem = meta_file.stem
                        meta_file.unlink()
                        (meta_file.parent / f"{stem}.feather").unlink(missing_ok=True)
                        (meta_file.parent / f"{stem}.parquet").unlink(missing_ok=True)
                        (meta_file.parent / f"{stem}.pkl").unlink(missing_ok=True)
                        logger.info(f"Cleared checkpoint {meta_file.stem}")
                    except Exception as e:
                        logger.warning(f"Failed to clear checkpoint {meta_file}: {e}")
    
    def validate_and_save_pydantic(
        self,
        phase: str,
        schema_obj: Any,
        artifact_name: str = "data",
        config: Optional[Dict[str, Any]] = None,
        version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate data using pydantic schema and save checkpoint.
        
        This method validates that phase output meets its data contract
        before saving, preventing silent data corruption propagation.
        
        Args:
            phase: Phase name (e.g., "phase1", "phase2")
            schema_obj: Pydantic BaseModel instance with validated data
            artifact_name: Artifact identifier
            config: Configuration dict for hashing
            version: Optional version string
            
        Returns:
            Metadata dict with checkpoint info
            
        Raises:
            ValueError: If schema validation fails
            
        Example:
            from src.schemas import Phase1Output
            
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
                test_size=X_test.shape[0],
                config_hash=config_hash
            )
            
            # Validates and saves
            metadata = checkpoint_mgr.validate_and_save_pydantic(
                "phase1",
                output,
                config=config
            )
        """
        # Pydantic model is already validated by __init__, but we can add
        # additional checks here if needed
        phase_dir, filename_stem = self._build_checkpoint_path(phase, artifact_name, version)
        
        # Save schema validation record
        schema_metadata = {
            "timestamp": datetime.utcnow().isoformat(),
            "phase": phase,
            "artifact_type": artifact_name,
            "schema_class": type(schema_obj).__name__,
            "schema_module": type(schema_obj).__module__,
            "validated": True
        }
        
        # Extract numpy arrays from schema for serialization
        artifact_data = {}
        for field_name, field_value in schema_obj.__dict__.items():
            if isinstance(field_value, np.ndarray):
                # Save numpy arrays separately
                array_path = phase_dir / f"{filename_stem}_{field_name}.npy"
                np.save(array_path, field_value)
                artifact_data[field_name] = {
                    "type": "numpy_array",
                    "shape": field_value.shape,
                    "dtype": str(field_value.dtype),
                    "file": str(array_path.relative_to(self.checkpoint_dir))
                }
            elif isinstance(field_value, datetime):
                artifact_data[field_name] = field_value.isoformat()
            else:
                artifact_data[field_name] = field_value
        
        # Save schema data
        schema_data_path = phase_dir / f"{filename_stem}_schema.json"
        with open(schema_data_path, "w") as f:
            json.dump(artifact_data, f, indent=2, default=str)
        
        # Save metadata
        if config:
            schema_metadata["config_hash"] = self._compute_config_hash(config)
        
        meta_path = phase_dir / f"{filename_stem}.json"
        with open(meta_path, "w") as f:
            json.dump(schema_metadata, f, indent=2, default=str)
        
        logger.info(json.dumps({
            "event": "checkpoint_validated",
            "action": "save",
            "phase": phase,
            "artifact": artifact_name,
            "schema": type(schema_obj).__name__,
            "file": str(schema_data_path.relative_to(self.checkpoint_dir))
        }))
        
        return schema_metadata
    
    def load_and_validate_pydantic(
        self,
        phase: str,
        schema_class: Type,
        artifact_name: str = "data",
        version: Optional[str] = None,
        allow_stale: bool = False,
        config: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Load checkpoint and reconstruct pydantic schema object.
        
        Validates that checkpoint matches expected schema and optionally
        validates config hash to prevent stale data reuse.
        
        Args:
            phase: Phase name
            schema_class: Pydantic BaseModel class to reconstruct
            artifact_name: Artifact identifier
            version: Optional version string
            allow_stale: If False, reject checkpoints with stale config
            config: Current configuration (for staleness check)
            
        Returns:
            Tuple of (schema_obj, metadata_dict)
            
        Raises:
            FileNotFoundError: If checkpoint not found
            ValueError: If schema validation fails or config is stale
            
        Example:
            from src.schemas import Phase1Output
            
            output, metadata = checkpoint_mgr.load_and_validate_pydantic(
                "phase1",
                Phase1Output,
                allow_stale=False,
                config=current_config
            )
        """
        phase_dir = self.checkpoint_dir / phase
        
        if not phase_dir.exists():
            raise FileNotFoundError(f"No checkpoint directory for {phase}")
        
        # Find checkpoint
        if version:
            filename_stem = f"{artifact_name}_{version}"
        else:
            # Find latest version
            candidates = list(phase_dir.glob(f"{artifact_name}_schema.json"))
            if not candidates:
                raise FileNotFoundError(f"No schema checkpoint for {phase}/{artifact_name}")
            candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            filename_stem = candidates[0].stem.replace("_schema", "")
        
        # Load metadata
        meta_path = phase_dir / f"{filename_stem}.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {filename_stem}.json")
        
        with open(meta_path, "r") as f:
            metadata = json.load(f)
        
        # Check config staleness
        if not allow_stale and config:
            saved_config_hash = metadata.get("config_hash")
            current_config_hash = self._compute_config_hash(config)
            if saved_config_hash and saved_config_hash != current_config_hash:
                logger.warning(
                    f"Checkpoint for {phase} has stale config hash "
                    f"(saved={saved_config_hash}, current={current_config_hash}); ignoring"
                )
                raise ValueError(
                    f"Checkpoint config is stale (saved hash={saved_config_hash}, "
                    f"current={current_config_hash}). Re-run phase with current config or set allow_stale=True"
                )
        
        # Load schema data
        schema_data_path = phase_dir / f"{filename_stem}_schema.json"
        if not schema_data_path.exists():
            raise FileNotFoundError(f"Schema data not found: {filename_stem}_schema.json")
        
        with open(schema_data_path, "r") as f:
            artifact_data = json.load(f)
        
        # Reconstruct numpy arrays
        for field_name, field_value in artifact_data.items():
            if isinstance(field_value, dict) and field_value.get("type") == "numpy_array":
                array_path = phase_dir / field_value["file"]
                if not array_path.exists():
                    raise FileNotFoundError(f"Numpy array not found: {array_path}")
                artifact_data[field_name] = np.load(array_path)
        
        # Reconstruct schema object
        try:
            schema_obj = schema_class(**artifact_data)
        except Exception as e:
            raise ValueError(f"Failed to reconstruct {schema_class.__name__}: {e}")
        
        logger.info(json.dumps({
            "event": "checkpoint_validated",
            "action": "load",
            "phase": phase,
            "artifact": artifact_name,
            "schema": schema_class.__name__,
            "version": filename_stem.split("_", 1)[1] if "_" in filename_stem else ""
        }))
        
        return schema_obj, metadata
