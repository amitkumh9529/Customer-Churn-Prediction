"""
Typed artifact dataclasses — outputs produced by each pipeline component.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataIngestionArtifact:
    train_file_path: Path
    test_file_path: Path
    val_file_path: Path
    n_train: int
    n_test: int
    n_val: int
    is_ingested: bool
    message: str


@dataclass
class DataValidationArtifact:
    is_validated: bool
    message: str
    report_path: Path
    missing_columns: list[str] = field(default_factory=list)
    n_rows: int = 0
    n_cols: int = 0


@dataclass
class DataTransformationArtifact:
    pipeline_path: Path
    transformed_train_path: Path
    transformed_test_path: Path
    transformed_val_path: Path
    feature_names: list[str] = field(default_factory=list)
    is_transformed: bool = True
    message: str = ""


@dataclass
class FeatureEngineeringArtifact:
    feature_names: list[str]
    n_features: int
    message: str


@dataclass
class ModelTrainerArtifact:
    model_path: Path
    best_model_name: str
    train_metrics: dict
    val_metrics: dict
    is_trained: bool
    message: str


@dataclass
class ModelEvaluationArtifact:
    model_path: Path
    test_metrics: dict
    confusion_matrix: list
    classification_report: str
    feature_importance: dict
    is_evaluated: bool
    message: str


@dataclass
class ModelPusherArtifact:
    mlflow_run_id: str
    model_uri: str
    is_pushed: bool
    message: str
