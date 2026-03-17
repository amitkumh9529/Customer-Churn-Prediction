"""
Stage 6 — Model Evaluation
Evaluates the best model on the hold-out test set.
"""

import numpy as np
import pandas as pd

from churn.entity.config_entity import ModelEvaluationConfig
from churn.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataTransformationArtifact,
    ModelEvaluationArtifact,
)
from churn.utils.common import ensure_dir, load_object, save_json
from churn.utils.exception import ModelEvaluationException
from churn.utils.helpers import (
    compute_metrics,
    get_classification_report,
    get_confusion_matrix,
    get_feature_importance,
)
from churn.utils.logger import logger


class ModelEvaluation:
    def __init__(
        self,
        config: ModelEvaluationConfig,
        trainer_artifact: ModelTrainerArtifact,
        transformation_artifact: DataTransformationArtifact,
    ):
        self.config = config
        self.trainer_artifact = trainer_artifact
        self.transformation_artifact = transformation_artifact

    def run(self) -> ModelEvaluationArtifact:
        logger.info("=" * 60)
        logger.info("STAGE 6: Model Evaluation — START")
        try:
            ensure_dir(self.config.reports_dir)

            # Load model and test data
            model = load_object(self.trainer_artifact.model_path)
            df_test = pd.read_csv(self.transformation_artifact.transformed_test_path)
            X_test = df_test.drop(columns=["Churn"]).values
            y_test = df_test["Churn"].values

            # Predictions
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= self.config.threshold).astype(int)

            # Metrics
            test_metrics = compute_metrics(y_test, y_pred, y_prob)
            conf_matrix = get_confusion_matrix(y_test, y_pred)
            clf_report = get_classification_report(y_test, y_pred)
            feature_names = self.transformation_artifact.feature_names
            feat_importance = get_feature_importance(model, feature_names)

            # Save reports
            full_report = {
                "model_name": self.trainer_artifact.best_model_name,
                "threshold": self.config.threshold,
                "test_metrics": test_metrics,
                "train_metrics": self.trainer_artifact.train_metrics,
                "val_metrics": self.trainer_artifact.val_metrics,
                "confusion_matrix": conf_matrix,
                "feature_importance": feat_importance,
            }
            save_json(full_report, self.config.reports_dir / "evaluation_metrics.json")
            save_json(feat_importance, self.config.reports_dir / "feature_importance.json")

            logger.info(f"Test metrics: {test_metrics}")
            logger.info(f"\n{clf_report}")
            logger.info("STAGE 6: Model Evaluation — DONE")

            return ModelEvaluationArtifact(
                model_path=self.trainer_artifact.model_path,
                test_metrics=test_metrics,
                confusion_matrix=conf_matrix,
                classification_report=clf_report,
                feature_importance=feat_importance,
                is_evaluated=True,
                message="Evaluation completed.",
            )

        except Exception as e:
            raise ModelEvaluationException("Model evaluation failed", e)
