"""
Stage 4 — Feature Engineering
Domain-specific features derived from raw columns before transformation.
"""

import pandas as pd

from churn.entity.config_entity import FeatureEngineeringConfig
from churn.entity.artifact_entity import DataIngestionArtifact, FeatureEngineeringArtifact
from churn.utils.exception import FeatureEngineeringException
from churn.utils.logger import logger


class FeatureEngineering:
    """
    Adds business-domain features that improve model signal:
      - tenure_group:      bucket tenure into loyalty categories
      - avg_monthly_cost:  TotalCharges / (tenure + 1)
      - has_streaming:     any streaming service active
      - services_count:    total number of active add-on services
      - is_month_to_month: contract type binary flag
    """

    def __init__(
        self,
        config: FeatureEngineeringConfig,
        ingestion_artifact: DataIngestionArtifact,
    ):
        self.config = config
        self.ingestion_artifact = ingestion_artifact

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # --- tenure group ---
        if "tenure" in df.columns:
            df["tenure_group"] = pd.cut(
                df["tenure"],
                bins=[-1, 12, 24, 48, 60, 72],
                labels=["0-12m", "13-24m", "25-48m", "49-60m", "61-72m"],
            ).astype(str)

        # --- average monthly cost ---
        if {"TotalCharges", "tenure"}.issubset(df.columns):
            df["avg_monthly_cost"] = df["TotalCharges"] / (df["tenure"] + 1)

        # --- has_streaming ---
        streaming_cols = [c for c in ["StreamingTV", "StreamingMovies"] if c in df.columns]
        if streaming_cols:
            df["has_streaming"] = (
                df[streaming_cols].apply(lambda c: c == "Yes").any(axis=1).astype(int)
            )

        # --- services_count ---
        service_cols = [
            c for c in [
                "PhoneService", "MultipleLines", "InternetService",
                "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies",
            ]
            if c in df.columns
        ]
        if service_cols:
            df["services_count"] = (
                df[service_cols].apply(lambda c: c == "Yes").sum(axis=1)
            )

        # --- is_month_to_month ---
        if "Contract" in df.columns:
            df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)

        return df

    def _process_split(self, path, label: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        logger.info(f"Engineering features for {label} set ({df.shape[0]} rows)...")
        df = self._add_features(df)
        df.to_csv(path, index=False)  # overwrite in-place
        logger.info(f"{label} set saved back with {df.shape[1]} columns.")
        return df

    def run(self) -> FeatureEngineeringArtifact:
        logger.info("=" * 60)
        logger.info("STAGE 4: Feature Engineering — START")
        try:
            train_df = self._process_split(
                self.ingestion_artifact.train_file_path, "train"
            )
            self._process_split(self.ingestion_artifact.val_file_path, "val")
            self._process_split(self.ingestion_artifact.test_file_path, "test")

            feature_names = [
                c for c in train_df.columns if c != self.config.target_column
            ]

            artifact = FeatureEngineeringArtifact(
                feature_names=feature_names,
                n_features=len(feature_names),
                message="Feature engineering completed successfully.",
            )
            logger.info(
                f"STAGE 4: Feature Engineering — DONE. {len(feature_names)} features."
            )
            return artifact

        except Exception as e:
            raise FeatureEngineeringException("Feature engineering failed", e)
