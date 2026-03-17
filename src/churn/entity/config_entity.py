"""
Typed configuration dataclasses passed between pipeline components.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    raw_data_path: Path
    interim_data_dir: Path
    processed_data_dir: Path
    test_size: float
    val_size: float
    random_state: int
    stratify: bool
    target_column: str
    id_column: str


@dataclass(frozen=True)
class DataValidationConfig:
    required_columns: list[str]
    min_rows: int
    target_column: str
    numeric_features: list[str]
    categorical_features: list[str]
    validation_report_path: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    processed_data_dir: Path
    encoders_dir: Path
    numeric_features: list[str]
    categorical_features: list[str]
    target_column: str
    pipeline_save_path: Path


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    processed_data_dir: Path
    target_column: str
    numeric_features: list[str]
    categorical_features: list[str]


@dataclass(frozen=True)
class ModelTrainerConfig:
    models_dir: Path
    reports_dir: Path
    random_state: int
    cv_folds: int
    primary_metric: str
    models_params: dict


@dataclass(frozen=True)
class ModelEvaluationConfig:
    models_dir: Path
    reports_dir: Path
    threshold: float
    primary_metric: str


@dataclass(frozen=True)
class ModelPusherConfig:
    models_dir: Path
    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    registered_model_name: str
