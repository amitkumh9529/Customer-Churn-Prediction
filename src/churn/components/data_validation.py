"""
Stage 2 — Data Validation
Checks schema, null rates, and basic distribution sanity.
"""

from pathlib import Path

import pandas as pd

from churn.entity.config_entity import DataValidationConfig
from churn.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from churn.utils.common import ensure_dir, save_json
from churn.utils.exception import DataValidationException
from churn.utils.logger import logger


class DataValidation:
    def __init__(self, config: DataValidationConfig, ingestion_artifact: DataIngestionArtifact):
        self.config = config
        self.ingestion_artifact = ingestion_artifact

    def _load(self, path: Path) -> pd.DataFrame:
        return pd.read_csv(path)

    def _validate_schema(self, df: pd.DataFrame) -> tuple[bool, list[str]]:
        """Ensure all required columns are present."""
        missing = [c for c in self.config.required_columns if c not in df.columns]
        return len(missing) == 0, missing

    def _validate_rows(self, df: pd.DataFrame) -> bool:
        return len(df) >= self.config.min_rows

    def _null_report(self, df: pd.DataFrame) -> dict:
        null_counts = df.isnull().sum()
        return {col: int(cnt) for col, cnt in null_counts.items() if cnt > 0}

    def _dtype_report(self, df: pd.DataFrame) -> dict:
        return {col: str(dtype) for col, dtype in df.dtypes.items()}

    def _target_distribution(self, df: pd.DataFrame) -> dict:
        if self.config.target_column in df.columns:
            counts = df[self.config.target_column].value_counts().to_dict()
            return {str(k): int(v) for k, v in counts.items()}
        return {}

    def run(self) -> DataValidationArtifact:
        logger.info("=" * 60)
        logger.info("STAGE 2: Data Validation — START")
        try:
            ensure_dir(self.config.validation_report_path.parent)

            train_df = self._load(self.ingestion_artifact.train_file_path)
            test_df = self._load(self.ingestion_artifact.test_file_path)

            schema_ok, missing_cols = self._validate_schema(train_df)
            rows_ok = self._validate_rows(train_df)
            nulls = self._null_report(train_df)

            report = {
                "train_shape": list(train_df.shape),
                "test_shape": list(test_df.shape),
                "schema_valid": schema_ok,
                "missing_columns": missing_cols,
                "sufficient_rows": rows_ok,
                "null_counts": nulls,
                "dtypes": self._dtype_report(train_df),
                "target_distribution": self._target_distribution(train_df),
            }

            save_json(report, self.config.validation_report_path)

            is_validated = schema_ok and rows_ok

            if not schema_ok:
                logger.error(f"Schema validation failed. Missing: {missing_cols}")
            if not rows_ok:
                logger.error(f"Insufficient rows: {len(train_df)} < {self.config.min_rows}")
            if nulls:
                logger.warning(f"Null values detected: {nulls}")

            message = "Validation passed." if is_validated else "Validation FAILED — check report."
            logger.info(f"STAGE 2: Data Validation — {'DONE' if is_validated else 'FAILED'}")

            return DataValidationArtifact(
                is_validated=is_validated,
                message=message,
                report_path=self.config.validation_report_path,
                missing_columns=missing_cols,
                n_rows=len(train_df),
                n_cols=len(train_df.columns),
            )

        except Exception as e:
            raise DataValidationException("Data validation failed", e)
