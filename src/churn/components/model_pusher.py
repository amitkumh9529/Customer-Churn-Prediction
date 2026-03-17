"""
Stage 7 — Model Pusher / Registry
Logs all artifacts to MLflow and registers the best model.
"""

from pathlib import Path
from typing import Any

import mlflow
from mlflow import sklearn as mlflow_sklearn
from mlflow.tracking import MlflowClient

from churn.entity.config_entity import ModelPusherConfig
from churn.entity.artifact_entity import (
    ModelEvaluationArtifact,
    ModelTrainerArtifact,
    DataTransformationArtifact,
    ModelPusherArtifact,
)
from churn.utils.common import load_object
from churn.utils.exception import ModelPusherException
from churn.utils.logger import logger


class ModelPusher:
    def __init__(
        self,
        config: ModelPusherConfig,
        evaluation_artifact: ModelEvaluationArtifact,
        trainer_artifact: ModelTrainerArtifact,
        transformation_artifact: DataTransformationArtifact,
    ):
        self.config = config
        self.evaluation_artifact = evaluation_artifact
        self.trainer_artifact = trainer_artifact
        self.transformation_artifact = transformation_artifact

    @staticmethod
    def _make_mlflow_uri(path: str | Path) -> str:
        """
        Build a file:/// URI that works on both Windows and Linux.
        Windows:  D:\\foo\\bar  →  file:///D:/foo/bar
        Linux:    /foo/bar      →  file:///foo/bar
        """
        forward = Path(path).resolve().as_posix()
        if not forward.startswith("/"):
            forward = "/" + forward
        return f"file://{forward}"

    def run(self) -> ModelPusherArtifact:
        logger.info("=" * 60)
        logger.info("STAGE 7: Model Pusher — START")
        try:
            tracking_uri = self._make_mlflow_uri(self.config.mlflow_tracking_uri)
            logger.info(f"MLflow tracking URI: {tracking_uri}")
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(self.config.mlflow_experiment_name)

            model: Any = load_object(self.trainer_artifact.model_path)
            pipeline: Any = load_object(self.transformation_artifact.pipeline_path)

            with mlflow.start_run(run_name=self.trainer_artifact.best_model_name) as run:
                run_id = run.info.run_id

                # Log params
                mlflow.log_param("model_type", self.trainer_artifact.best_model_name)
                mlflow.log_param("threshold", self.evaluation_artifact.test_metrics.get("threshold", 0.5))

                # Log train / val / test metrics
                for prefix, metrics in [
                    ("train", self.trainer_artifact.train_metrics),
                    ("val",   self.trainer_artifact.val_metrics),
                    ("test",  self.evaluation_artifact.test_metrics),
                ]:
                    for k, v in metrics.items():
                        mlflow.log_metric(f"{prefix}_{k}", v)

                # Log pipeline artifact
                mlflow.log_artifact(str(self.transformation_artifact.pipeline_path))

                # Log report JSONs
                reports_dir = self.config.models_dir.parent / "reports"
                for report_file in reports_dir.glob("*.json"):
                    mlflow.log_artifact(str(report_file))

                # Register model
                mlflow_sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=self.config.registered_model_name,
                )

                model_uri = f"runs:/{run_id}/model"
                logger.info(f"MLflow run_id: {run_id}")
                logger.info(f"Model URI:     {model_uri}")

            logger.info("STAGE 7: Model Pusher — DONE")
            return ModelPusherArtifact(
                mlflow_run_id=run_id,
                model_uri=model_uri,
                is_pushed=True,
                message="Model logged and registered in MLflow.",
            )

        except Exception as e:
            raise ModelPusherException("Model pusher failed", e)