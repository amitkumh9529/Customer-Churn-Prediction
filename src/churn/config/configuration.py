"""
Configuration manager — reads YAML configs and returns typed config entities.
"""

from pathlib import Path

from churn.constants.constants import (
    CONFIG_FILE_PATH,
    MODEL_CONFIG_FILE_PATH,
    MLFLOW_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_REGISTERED_MODEL_NAME,
    MODELS_DIR,
    ENCODERS_DIR,
    REPORTS_DIR,
    PROCESSED_DATA_DIR,
    INTERIM_DATA_DIR,
    RAW_DATA_FILE,
)
from churn.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    FeatureEngineeringConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelPusherConfig,
)
from churn.utils.common import read_yaml
from churn.utils.exception import ConfigurationException
from churn.utils.logger import logger


class ConfigurationManager:
    """Loads config YAML files and exposes typed config objects."""

    def __init__(
        self,
        config_path: Path = CONFIG_FILE_PATH,
        model_config_path: Path = MODEL_CONFIG_FILE_PATH,
    ):
        try:
            self._cfg = read_yaml(config_path)
            self._mcfg = read_yaml(model_config_path)
            logger.info("Configuration loaded successfully.")
        except Exception as e:
            raise ConfigurationException("Failed to load configuration", e)

    # ── Data Ingestion ───────────────────────────────────────────────────────
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        data_cfg = self._cfg["data"]
        return DataIngestionConfig(
            raw_data_path=RAW_DATA_FILE,
            interim_data_dir=INTERIM_DATA_DIR,
            processed_data_dir=PROCESSED_DATA_DIR,
            test_size=data_cfg["test_size"],
            val_size=data_cfg["validation_size"],
            random_state=data_cfg["random_state"],
            stratify=data_cfg["stratify"],
            target_column=data_cfg["target_column"],
            id_column=data_cfg["id_column"],
        )

    # ── Data Validation ──────────────────────────────────────────────────────
    def get_data_validation_config(self) -> DataValidationConfig:
        data_cfg = self._cfg["data"]
        val_schema = data_cfg["validation_schema"]
        return DataValidationConfig(
            required_columns=val_schema["required_columns"],
            min_rows=val_schema["min_rows"],
            target_column=data_cfg["target_column"],
            numeric_features=data_cfg["numeric_features"],
            categorical_features=data_cfg["categorical_features"],
            validation_report_path=REPORTS_DIR / "data_validation_report.json",
        )

    # ── Data Transformation ──────────────────────────────────────────────────
    def get_data_transformation_config(self) -> DataTransformationConfig:
        data_cfg = self._cfg["data"]
        return DataTransformationConfig(
            processed_data_dir=PROCESSED_DATA_DIR,
            encoders_dir=ENCODERS_DIR,
            numeric_features=data_cfg["numeric_features"],
            categorical_features=data_cfg["categorical_features"],
            target_column=data_cfg["target_column"],
            pipeline_save_path=ENCODERS_DIR / "preprocessing_pipeline.pkl",
        )

    # ── Feature Engineering ──────────────────────────────────────────────────
    def get_feature_engineering_config(self) -> FeatureEngineeringConfig:
        data_cfg = self._cfg["data"]
        return FeatureEngineeringConfig(
            processed_data_dir=PROCESSED_DATA_DIR,
            target_column=data_cfg["target_column"],
            numeric_features=data_cfg["numeric_features"],
            categorical_features=data_cfg["categorical_features"],
        )

    # ── Model Trainer ────────────────────────────────────────────────────────
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        eval_cfg = self._mcfg["evaluation"]
        return ModelTrainerConfig(
            models_dir=MODELS_DIR,
            reports_dir=REPORTS_DIR,
            random_state=self._cfg["data"]["random_state"],
            cv_folds=self._mcfg["model_selection"]["cross_validation_folds"],
            primary_metric=eval_cfg["primary_metric"],
            models_params=self._mcfg["models"],
        )

    # ── Model Evaluation ─────────────────────────────────────────────────────
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        eval_cfg = self._mcfg["evaluation"]
        return ModelEvaluationConfig(
            models_dir=MODELS_DIR,
            reports_dir=REPORTS_DIR,
            threshold=eval_cfg["threshold"],
            primary_metric=eval_cfg["primary_metric"],
        )

    # ── Model Pusher ─────────────────────────────────────────────────────────
    def get_model_pusher_config(self) -> ModelPusherConfig:
        return ModelPusherConfig(
            models_dir=MODELS_DIR,
            mlflow_tracking_uri=str(MLFLOW_DIR),
            mlflow_experiment_name=MLFLOW_EXPERIMENT_NAME,
            registered_model_name=MLFLOW_REGISTERED_MODEL_NAME,
        )
