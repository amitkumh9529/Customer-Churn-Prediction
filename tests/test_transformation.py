"""
Unit tests for DataTransformation component.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from churn.components.data_ingestion import DataIngestion
from churn.components.data_transformation import DataTransformation
from churn.entity.config_entity import DataIngestionConfig, DataTransformationConfig

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]


@pytest.fixture
def ingestion_artifact(tmp_path, sample_raw_df):
    raw_path = tmp_path / "churn.csv"
    sample_raw_df.to_csv(raw_path, index=False)

    cfg = DataIngestionConfig(
        raw_data_path=raw_path,
        interim_data_dir=tmp_path / "interim",
        processed_data_dir=tmp_path / "processed",
        test_size=0.2,
        val_size=0.1,
        random_state=42,
        stratify=False,
        target_column="Churn",
        id_column="customerID",
    )
    return DataIngestion(cfg).run()


@pytest.fixture
def transformation_config(tmp_path):
    return DataTransformationConfig(
        processed_data_dir=tmp_path / "processed",
        encoders_dir=tmp_path / "encoders",
        numeric_features=NUMERIC,
        categorical_features=CATEGORICAL,
        target_column="Churn",
        pipeline_save_path=tmp_path / "encoders" / "pipeline.pkl",
    )


def test_transformation_produces_artifact(ingestion_artifact, transformation_config):
    t = DataTransformation(transformation_config, ingestion_artifact)
    artifact = t.run()

    assert artifact.is_transformed
    assert artifact.pipeline_path.exists()
    assert artifact.transformed_train_path.exists()
    assert artifact.transformed_test_path.exists()


def test_transformed_data_has_no_nulls(ingestion_artifact, transformation_config):
    t = DataTransformation(transformation_config, ingestion_artifact)
    artifact = t.run()

    for path in [
        artifact.transformed_train_path,
        artifact.transformed_val_path,
        artifact.transformed_test_path,
    ]:
        df = pd.read_csv(path)
        assert df.isnull().sum().sum() == 0, f"Nulls found in {path}"


def test_pipeline_is_sklearn_pipeline(ingestion_artifact, transformation_config):
    from churn.utils.common import load_object

    t = DataTransformation(transformation_config, ingestion_artifact)
    artifact = t.run()

    pipeline = load_object(artifact.pipeline_path)
    assert isinstance(pipeline, Pipeline)


def test_feature_names_populated(ingestion_artifact, transformation_config):
    t = DataTransformation(transformation_config, ingestion_artifact)
    artifact = t.run()

    assert len(artifact.feature_names) > 0
    # numeric features should appear verbatim
    for feat in NUMERIC:
        assert feat in artifact.feature_names
