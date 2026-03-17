"""
Unit tests for DataIngestion component.
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from churn.components.data_ingestion import DataIngestion
from churn.entity.config_entity import DataIngestionConfig


@pytest.fixture
def ingestion_config(tmp_path, sample_raw_df) -> DataIngestionConfig:
    # Save sample data to a temp CSV
    raw_path = tmp_path / "raw" / "churn.csv"
    raw_path.parent.mkdir(parents=True)
    sample_raw_df.to_csv(raw_path, index=False)

    return DataIngestionConfig(
        raw_data_path=raw_path,
        interim_data_dir=tmp_path / "interim",
        processed_data_dir=tmp_path / "processed",
        test_size=0.2,
        val_size=0.1,
        random_state=42,
        stratify=False,   # small sample → disable stratify
        target_column="Churn",
        id_column="customerID",
    )


def test_ingestion_produces_artifact(ingestion_config):
    ingestion = DataIngestion(ingestion_config)
    artifact = ingestion.run()

    assert artifact.is_ingested
    assert artifact.n_train > 0
    assert artifact.n_test > 0
    assert artifact.train_file_path.exists()
    assert artifact.test_file_path.exists()
    assert artifact.val_file_path.exists()


def test_ingestion_splits_sum_to_total(ingestion_config, sample_raw_df):
    ingestion = DataIngestion(ingestion_config)
    artifact = ingestion.run()

    total = artifact.n_train + artifact.n_val + artifact.n_test
    # customerID is dropped, Churn target is encoded — row count preserved
    assert total == len(sample_raw_df)


def test_ingestion_target_is_binary(ingestion_config):
    ingestion = DataIngestion(ingestion_config)
    artifact = ingestion.run()

    train_df = pd.read_csv(artifact.train_file_path)
    assert set(train_df["Churn"].unique()).issubset({0, 1})


def test_ingestion_drops_customer_id(ingestion_config):
    ingestion = DataIngestion(ingestion_config)
    artifact = ingestion.run()

    train_df = pd.read_csv(artifact.train_file_path)
    assert "customerID" not in train_df.columns


def test_total_charges_coerced_to_numeric(ingestion_config):
    ingestion = DataIngestion(ingestion_config)
    artifact = ingestion.run()

    train_df = pd.read_csv(artifact.train_file_path)
    assert pd.api.types.is_numeric_dtype(train_df["TotalCharges"])
