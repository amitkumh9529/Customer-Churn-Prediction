"""
Stage 3 — Data Transformation
Builds a Scikit-learn Pipeline with imputation, encoding, and scaling.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.entity.config_entity import DataTransformationConfig
from churn.entity.artifact_entity import DataIngestionArtifact, DataTransformationArtifact
from churn.utils.common import ensure_dir, save_object
from churn.utils.exception import DataTransformationException
from churn.utils.logger import logger


class DataTransformation:
    def __init__(
        self,
        config: DataTransformationConfig,
        ingestion_artifact: DataIngestionArtifact,
    ):
        self.config = config
        self.ingestion_artifact = ingestion_artifact

    def _build_preprocessing_pipeline(self) -> Pipeline:
        """Construct the full sklearn preprocessing pipeline."""
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, self.config.numeric_features),
                ("cat", categorical_pipeline, self.config.categorical_features),
            ],
            remainder="drop",
        )

        return Pipeline([("preprocessor", preprocessor)])

    def _load(self, path) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(path)
        X = df.drop(columns=[self.config.target_column])
        y = df[self.config.target_column].values
        return X, y

    def _get_feature_names(self, pipeline: Pipeline) -> list[str]:
        """Extract feature names after preprocessing."""
        try:
            preprocessor = pipeline.named_steps["preprocessor"]
            num_names = self.config.numeric_features
            cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
            cat_names = cat_encoder.get_feature_names_out(self.config.categorical_features).tolist()
            return num_names + cat_names
        except Exception:
            return []

    def run(self) -> DataTransformationArtifact:
        logger.info("=" * 60)
        logger.info("STAGE 3: Data Transformation — START")
        try:
            ensure_dir(self.config.processed_data_dir)
            ensure_dir(self.config.encoders_dir)

            X_train, y_train = self._load(self.ingestion_artifact.train_file_path)
            X_val, y_val = self._load(self.ingestion_artifact.val_file_path)
            X_test, y_test = self._load(self.ingestion_artifact.test_file_path)

            pipeline = self._build_preprocessing_pipeline()
            logger.info("Fitting preprocessing pipeline on train set...")
            X_train_t = pipeline.fit_transform(X_train)
            X_val_t = pipeline.transform(X_val)
            X_test_t = pipeline.transform(X_test)

            feature_names = self._get_feature_names(pipeline)
            logger.info(f"Transformed features: {len(feature_names)}")

            # Save transformed arrays as CSVs with target
            def _save(X, y, name):
                df_out = pd.DataFrame(X, columns=feature_names)
                df_out[self.config.target_column] = y
                path = self.config.processed_data_dir / f"transformed_{name}.csv"
                df_out.to_csv(path, index=False)
                return path

            train_path = _save(X_train_t, y_train, "train")
            val_path = _save(X_val_t, y_val, "val")
            test_path = _save(X_test_t, y_test, "test")

            save_object(pipeline, self.config.pipeline_save_path)

            artifact = DataTransformationArtifact(
                pipeline_path=self.config.pipeline_save_path,
                transformed_train_path=train_path,
                transformed_test_path=test_path,
                transformed_val_path=val_path,
                feature_names=feature_names,
                is_transformed=True,
                message="Data transformation completed successfully.",
            )
            logger.info("STAGE 3: Data Transformation — DONE")
            return artifact

        except Exception as e:
            raise DataTransformationException("Data transformation failed", e)
