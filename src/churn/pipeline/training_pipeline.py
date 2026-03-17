"""
End-to-End Training Pipeline
Orchestrates all 7 stages in sequence.
"""

import time

from churn.config.configuration import ConfigurationManager
from churn.components.data_ingestion import DataIngestion
from churn.components.data_validation import DataValidation
from churn.components.data_transformation import DataTransformation
from churn.components.feature_engineering import FeatureEngineering
from churn.components.model_trainer import ModelTrainer
from churn.components.model_evaluation import ModelEvaluation
from churn.components.model_pusher import ModelPusher
from churn.utils.exception import ChurnBaseException
from churn.utils.logger import logger


class TrainingPipeline:
    """Runs the full training pipeline end-to-end."""

    def __init__(self):
        self.config_manager = ConfigurationManager()

    def run(self) -> None:
        start = time.time()
        logger.info("╔══════════════════════════════════════╗")
        logger.info("║  CUSTOMER CHURN — TRAINING PIPELINE  ║")
        logger.info("╚══════════════════════════════════════╝")

        try:
            # Stage 1: Data Ingestion
            ingestion_cfg = self.config_manager.get_data_ingestion_config()
            ingestion = DataIngestion(ingestion_cfg)
            ingestion_artifact = ingestion.run()

            # Stage 2: Data Validation
            validation_cfg = self.config_manager.get_data_validation_config()
            validation = DataValidation(validation_cfg, ingestion_artifact)
            validation_artifact = validation.run()

            if not validation_artifact.is_validated:
                raise ChurnBaseException(
                    f"Data validation failed: {validation_artifact.message}"
                )

            # Stage 3: Feature Engineering (before transformation)
            fe_cfg = self.config_manager.get_feature_engineering_config()
            fe = FeatureEngineering(fe_cfg, ingestion_artifact)
            fe_artifact = fe.run()

            # Stage 4: Data Transformation
            transformation_cfg = self.config_manager.get_data_transformation_config()
            transformation = DataTransformation(transformation_cfg, ingestion_artifact)
            transformation_artifact = transformation.run()

            # Stage 5: Model Training
            trainer_cfg = self.config_manager.get_model_trainer_config()
            trainer = ModelTrainer(trainer_cfg, transformation_artifact)
            trainer_artifact = trainer.run()

            # Stage 6: Model Evaluation
            evaluation_cfg = self.config_manager.get_model_evaluation_config()
            evaluator = ModelEvaluation(
                evaluation_cfg, trainer_artifact, transformation_artifact
            )
            evaluation_artifact = evaluator.run()

            # Stage 7: Model Pusher (MLflow)
            pusher_cfg = self.config_manager.get_model_pusher_config()
            pusher = ModelPusher(
                pusher_cfg,
                evaluation_artifact,
                trainer_artifact,
                transformation_artifact,
            )
            pusher_artifact = pusher.run()

            elapsed = round(time.time() - start, 2)
            logger.info("=" * 60)
            logger.info(f"✅ Pipeline completed in {elapsed}s")
            logger.info(f"   Best model   : {trainer_artifact.best_model_name}")
            logger.info(f"   Test ROC-AUC : {evaluation_artifact.test_metrics.get('roc_auc')}")
            logger.info(f"   Test F1      : {evaluation_artifact.test_metrics.get('f1')}")
            logger.info(f"   MLflow run   : {pusher_artifact.mlflow_run_id}")
            logger.info("=" * 60)

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise
