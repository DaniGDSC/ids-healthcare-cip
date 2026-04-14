"""Phase 1 configuration — pydantic-validated settings.

Loads from ``config.yaml`` and validates all fields at construction time.
Paths are resolved relative to the project root and routed through the
Phase 0 ``PathValidator`` so any path that escapes the workspace is
rejected at config-load time, before the pipeline touches the disk.

The schema is **strict**: ``from_yaml`` rejects any top-level section
not in ``ALLOWED_TOP_LEVEL`` (see the constant below). The previous
permissive loader silently fell back to defaults whenever a CI YAML
disagreed with the production YAML, so the CI was exercising a
fictional pipeline. Strict-mode loading makes that class of bug a
hard failure at config-load time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import yaml
from pydantic import BaseModel, field_validator, model_validator


# Strict allowlist of top-level YAML sections. The loader refuses any
# section not in this set, so a CI config that uses ``hipaa:`` instead
# of ``identifier_removal:`` (the previous bug) becomes a fail-loud
# error rather than a silent fall-back to defaults.
ALLOWED_TOP_LEVEL: frozenset[str] = frozenset(
    {
        "data",
        "identifier_removal",
        "encoding",
        "cleaning",
        "variance_filtering",
        "correlation_removal",
        "splitting",
        "normalization",
        "track_a",
        "track_b",
        "output",
        "logging",  # operator-only; not consumed by from_yaml but tolerated
    }
)


class ConfigError(Exception):
    """Raised when ``phase1_config.yaml`` is structurally invalid.

    Distinct from ``pydantic.ValidationError`` so callers can tell a
    YAML-shape failure (operator typo) apart from a semantic failure
    in a field validator.
    """


class Phase1Config(BaseModel):
    """Validated configuration for the Phase 1 preprocessing pipeline."""

    # Data
    input_dir: Path
    output_dir: Path
    file_pattern: str = "*.csv"
    label_column: str = "Label"
    multi_label_column: str = "Attack Category"

    # Step 1: Identifier removal
    id_removal_enabled: bool = True
    id_removal_columns: List[str]

    # Step 2: Encoding
    label_encode_columns: List[str] = []
    parse_numeric_columns: List[str] = []
    # Sentinel for unparseable strings. -99999 sits well outside any
    # valid port (0–65535) or flag-count range, so the model cannot
    # accidentally learn ``port == -1`` as a meaningful event from a
    # one-step discontinuity. See finding #23.
    parse_numeric_sentinel: int = -99999

    # Step 3: Cleaning
    biometric_columns: List[str]
    biometric_strategy: str = "median"  # patient-safe default
    network_strategy: str = "dropna"  # missing ≠ zero
    session_column: str | None = None

    # Step 4a: Variance filtering
    variance_enabled: bool = True
    variance_max_unique: int = 1

    # Step 4b: Correlation removal
    correlation_enabled: bool = True
    correlation_threshold: float = 0.95
    phase0_corr_file: Path

    # Step 5a: Split
    train_ratio: float = 0.70
    test_ratio: float = 0.30
    random_state: int = 42
    stratify: bool = True

    # Step 6: Scaling
    scaling_method: str = "robust"

    # Track A: SMOTE config (applied inside CV, not standalone)
    smote_enabled: bool = True
    smote_strategy: str = "auto"
    smote_k_neighbors: int = 5

    # Track B: Novelty detection
    track_b_enabled: bool = True

    # Phase 0 artifacts
    phase0_stats_file: Path = Path("results/phase0_analysis/stats_report.json")
    phase0_integrity_file: Path = Path("results/phase0_analysis/dataset_integrity.json")

    # Output filenames
    train_parquet: str = "train_phase1.parquet"
    test_parquet: str = "test_phase1.parquet"
    train_benign_parquet: str = "train_benign_phase1.parquet"
    scaler_file: str = "robust_scaler.json"
    encoder_file: str = "categorical_encoder.json"
    report_file: str = "preprocessing_report.json"

    model_config = {"arbitrary_types_allowed": True}

    @field_validator("correlation_threshold")
    @classmethod
    def _threshold_in_range(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError(f"correlation_threshold must be in (0, 1], got {v}")
        return v

    @field_validator("smote_k_neighbors")
    @classmethod
    def _k_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"smote_k_neighbors must be ≥ 1, got {v}")
        return v

    @field_validator("random_state")
    @classmethod
    def _random_state_canonical(cls, v: int) -> int:
        # Allowlist of vetted seeds. Any deviation produces a WARNING in
        # the logs because an attacker (or an over-eager researcher)
        # who can edit the YAML can otherwise pin the train/test split
        # to a particularly favourable seed and inflate every reported
        # metric. See finding #17.
        canonical = {0, 7, 42}
        if v not in canonical:
            import logging

            logging.getLogger(__name__).warning(
                "Phase1Config.random_state=%d is outside the canonical "
                "vetted set %s. The integer is logged into the report "
                "and report renderer, but a non-canonical seed is a "
                "research-integrity smell — see security review #17.",
                v,
                sorted(canonical),
            )
        return v

    @model_validator(mode="after")
    def _ratios_sum_to_one(self) -> Phase1Config:
        total = round(self.train_ratio + self.test_ratio, 4)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"train_ratio + test_ratio must equal 1.0, got {total}")
        return self

    @classmethod
    def from_yaml(
        cls,
        path: Path,
        *,
        workspace_root: Path | None = None,
    ) -> Phase1Config:
        """Load and validate configuration from a YAML file.

        Strict-mode loading: any top-level section not in
        ``ALLOWED_TOP_LEVEL`` raises ``ConfigError`` rather than being
        silently ignored. This prevents the class of bug where a CI
        config writes ``hipaa:`` while the loader expects
        ``identifier_removal:``, with the loader silently falling back
        to defaults and the CI passing on a fictional pipeline.

        Both ``data.input_dir`` and ``data.output_dir`` are routed
        through ``PathValidator`` so a hostile YAML cannot point at
        ``/etc`` or escape the workspace via ``../``.

        Raises:
            FileNotFoundError: if *path* does not exist.
            ConfigError: if YAML is unparseable or contains an unknown
                top-level section.
            PermissionError: if input/output paths escape the workspace.
        """
        if not path.exists():
            raise FileNotFoundError(f"Phase 1 config not found: {path}")

        try:
            raw: Dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"Failed to parse YAML at {path}: {exc}") from exc

        if not isinstance(raw, dict):
            raise ConfigError(
                f"{path} must contain a YAML mapping at the top level, " f"got {type(raw).__name__}"
            )

        unknown = set(raw) - ALLOWED_TOP_LEVEL
        if unknown:
            raise ConfigError(
                f"{path}: unknown top-level section(s) {sorted(unknown)}. "
                f"Allowed: {sorted(ALLOWED_TOP_LEVEL)}. "
                f"This is most often caused by a CI config that uses an "
                f"old field name (e.g. 'hipaa' instead of "
                f"'identifier_removal') — silent fall-back is intentionally "
                f"removed."
            )

        data = raw.get("data", {})
        idr = raw.get("identifier_removal", {})
        encoding = raw.get("encoding", {})
        cl = raw.get("cleaning", {})
        corr = raw.get("correlation_removal", {})
        var = raw.get("variance_filtering", {})
        split = raw.get("splitting", {})
        norm = raw.get("normalization", {})
        track_a = raw.get("track_a", {})
        smote = track_a.get("smote", {})
        track_b = raw.get("track_b", {})
        output = raw.get("output", {})

        # Phase 1 lives at module1_preprocessing/phase1/config.py
        # → workspace root is four parents up.
        root = workspace_root or Path(__file__).resolve().parents[2]
        # Lazy import: keep config.py importable even if a developer
        # builds a venv without the Phase 0 package on the path.
        from module0_analysis.phase0.security import PathValidator

        validator = PathValidator(root)
        # validate_input_path requires existence; data dirs may not
        # exist in CI runs that only exercise config parsing, so we
        # only require workspace containment here. The pipeline's
        # _ingest_with_integrity step re-validates with existence.
        try:
            validator._resolve_inside_workspace(Path(data.get("input_dir", "data/raw/WUSTL-EHMS")))
        except PermissionError:
            raise
        validator.validate_output_dir(Path(data.get("output_dir", "data/processed")))

        return cls(
            input_dir=Path(data.get("input_dir", "data/raw/WUSTL-EHMS")),
            output_dir=Path(data.get("output_dir", "data/processed")),
            file_pattern=data.get("file_pattern", "*.csv"),
            label_column=data.get("label_column", "Label"),
            multi_label_column=data.get("multi_label_column", "Attack Category"),
            id_removal_enabled=idr.get("enabled", True),
            id_removal_columns=idr.get("remove_columns", []),
            label_encode_columns=encoding.get("label_encode", []),
            parse_numeric_columns=encoding.get("parse_numeric", []),
            parse_numeric_sentinel=encoding.get("parse_numeric_sentinel", -99999),
            biometric_columns=cl.get("biometric_columns", []),
            biometric_strategy=cl.get("biometric_strategy", "median"),
            network_strategy=cl.get("network_strategy", "dropna"),
            session_column=cl.get("session_column"),
            correlation_enabled=corr.get("enabled", True),
            correlation_threshold=corr.get("threshold", 0.95),
            phase0_corr_file=Path(
                corr.get(
                    "phase0_corr_file",
                    "results/phase0_analysis/high_correlations.csv",
                )
            ),
            variance_enabled=var.get("enabled", True),
            variance_max_unique=var.get("max_unique", 1),
            train_ratio=split.get("train_ratio", 0.70),
            test_ratio=split.get("test_ratio", 0.30),
            random_state=split.get("random_state", 42),
            stratify=split.get("stratify", True),
            scaling_method=norm.get("method", "robust"),
            smote_enabled=smote.get("enabled", True),
            smote_strategy=smote.get("sampling_strategy", "auto"),
            smote_k_neighbors=smote.get("k_neighbors", 5),
            track_b_enabled=track_b.get("enabled", True),
            train_parquet=output.get("train_parquet", "train_phase1.parquet"),
            test_parquet=output.get("test_parquet", "test_phase1.parquet"),
            train_benign_parquet=output.get("train_benign_parquet", "train_benign_phase1.parquet"),
            scaler_file=output.get("scaler_file", "robust_scaler.json"),
            encoder_file=output.get("encoder_file", "categorical_encoder.json"),
            report_file=output.get("report_file", "preprocessing_report.json"),
        )
