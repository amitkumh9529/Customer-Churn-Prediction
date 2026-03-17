"""
Stage 1 — Data Ingestion
Loads raw CSV, performs train/val/test split, saves to processed/.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.entity.config_entity import DataIngestionConfig
from churn.entity.artifact_entity import DataIngestionArtifact
from churn.utils.common import ensure_dir
from churn.utils.exception import DataIngestionException
from churn.utils.helpers import encode_target
from churn.utils.logger import logger


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def _load_raw_data(self) -> pd.DataFrame:
        logger.info(f"Loading raw data from: {self.config.raw_data_path}")
        try:
            df = pd.read_csv(self.config.raw_data_path)
            logger.info(f"Raw data loaded — shape: {df.shape}")
            return df
        except Exception as e:
            raise DataIngestionException("Failed to load raw CSV", e)

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic cleaning: fix TotalCharges dtype, encode target."""
        logger.info("Cleaning raw data...")

        # Drop customerID — it's not a feature
        if self.config.id_column in df.columns:
            df = df.drop(columns=[self.config.id_column])

        # TotalCharges is often read as object due to spaces
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            n_missing = df["TotalCharges"].isna().sum()
            if n_missing > 0:
                logger.warning(f"Imputing {n_missing} missing TotalCharges with median.")
                df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

        # Encode target
        target = self.config.target_column
        if df[target].dtype == object:
            df[target] = encode_target(df[target])

        logger.info(f"Clean data shape: {df.shape}")
        return df

    def _split(self, df: pd.DataFrame):
        """Split into train / val / test sets."""
        target = self.config.target_column
        stratify_col = df[target] if self.config.stratify else None

        # First split: (train+val) vs test
        train_val, test = train_test_split(
            df,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=stratify_col,
        )

        # Second split: train vs val (from train_val pool)
        val_ratio = self.config.val_size / (1 - self.config.test_size)
        stratify_col2 = train_val[target] if self.config.stratify else None

        train, val = train_test_split(
            train_val,
            test_size=val_ratio,
            random_state=self.config.random_state,
            stratify=stratify_col2,
        )

        logger.info(
            f"Split sizes — train: {len(train)}, val: {len(val)}, test: {len(test)}"
        )
        return train, val, test

    def run(self) -> DataIngestionArtifact:
        logger.info("=" * 60)
        logger.info("STAGE 1: Data Ingestion — START")
        try:
            ensure_dir(self.config.processed_data_dir)
            ensure_dir(self.config.interim_data_dir)

            df = self._load_raw_data()
            df = self._clean_data(df)
            train, val, test = self._split(df)

            # Save splits
            train_path = self.config.processed_data_dir / "train.csv"
            val_path = self.config.processed_data_dir / "val.csv"
            test_path = self.config.processed_data_dir / "test.csv"

            train.to_csv(train_path, index=False)
            val.to_csv(val_path, index=False)
            test.to_csv(test_path, index=False)

            artifact = DataIngestionArtifact(
                train_file_path=train_path,
                test_file_path=test_path,
                val_file_path=val_path,
                n_train=len(train),
                n_test=len(test),
                n_val=len(val),
                is_ingested=True,
                message="Data ingestion completed successfully.",
            )
            logger.info(f"STAGE 1: Data Ingestion — DONE. Artifact: {artifact}")
            return artifact

        except Exception as e:
            raise DataIngestionException("Data ingestion failed", e)
