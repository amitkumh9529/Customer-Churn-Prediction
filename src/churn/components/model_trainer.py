"""
Stage 5 — Model Training
Trains multiple classifiers, selects the best by CV ROC-AUC.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from churn.entity.config_entity import ModelTrainerConfig
from churn.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from churn.utils.common import ensure_dir, save_object
from churn.utils.exception import ModelTrainingException
from churn.utils.helpers import compute_metrics
from churn.utils.logger import logger


class ModelTrainer:
    def __init__(
        self,
        config: ModelTrainerConfig,
        transformation_artifact: DataTransformationArtifact,
    ):
        self.config = config
        self.transformation_artifact = transformation_artifact

    def _build_models(self) -> dict:
        """Instantiate classifiers from config params."""
        models = {}
        params = self.config.models_params

        if params.get("logistic_regression", {}).get("enabled", True):
            p = params["logistic_regression"]["params"]
            models["LogisticRegression"] = LogisticRegression(**p)

        if params.get("random_forest", {}).get("enabled", True):
            p = params["random_forest"]["params"]
            models["RandomForest"] = RandomForestClassifier(**p)

        if params.get("xgboost", {}).get("enabled", True):
            p = {k: v for k, v in params["xgboost"]["params"].items()
                 if k != "use_label_encoder"}
            models["XGBoost"] = XGBClassifier(**p, verbosity=0)

        return models

    def _load_split(self, path) -> tuple[np.ndarray, np.ndarray]:
        df = pd.read_csv(path)
        target = "Churn"
        X = df.drop(columns=[target]).values
        y = df[target].values
        return X, y

    def _cross_validate(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
    ) -> float:
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )
        scores = cross_val_score(
            model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        mean_auc = float(scores.mean())
        logger.info(
            f"CV ROC-AUC: {mean_auc:.4f} ± {scores.std():.4f}"
        )
        return mean_auc

    def run(self) -> ModelTrainerArtifact:
        logger.info("=" * 60)
        logger.info("STAGE 5: Model Training — START")
        try:
            ensure_dir(self.config.models_dir)
            ensure_dir(self.config.reports_dir)

            X_train, y_train = self._load_split(
                self.transformation_artifact.transformed_train_path
            )
            X_val, y_val = self._load_split(
                self.transformation_artifact.transformed_val_path
            )

            models = self._build_models()
            cv_results: dict[str, float] = {}

            for name, model in models.items():
                logger.info(f"Cross-validating {name}...")
                auc = self._cross_validate(model, X_train, y_train)
                cv_results[name] = auc

            # Select best model
            best_name = max(cv_results, key=cv_results.get)
            best_model = models[best_name]
            logger.info(f"Best model: {best_name} (AUC={cv_results[best_name]:.4f})")

            # Final fit on full train set
            logger.info(f"Training {best_name} on full training set...")
            best_model.fit(X_train, y_train)

            # Train metrics
            y_pred_train = best_model.predict(X_train)
            y_prob_train = best_model.predict_proba(X_train)[:, 1]
            train_metrics = compute_metrics(y_train, y_pred_train, y_prob_train)

            # Val metrics
            y_pred_val = best_model.predict(X_val)
            y_prob_val = best_model.predict_proba(X_val)[:, 1]
            val_metrics = compute_metrics(y_val, y_pred_val, y_prob_val)

            logger.info(f"Train metrics: {train_metrics}")
            logger.info(f"Val metrics:   {val_metrics}")

            # Save model
            model_path = self.config.models_dir / "best_model.pkl"
            save_object(best_model, model_path)

            artifact = ModelTrainerArtifact(
                model_path=model_path,
                best_model_name=best_name,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                is_trained=True,
                message=f"Training completed. Best model: {best_name}",
            )
            logger.info("STAGE 5: Model Training — DONE")
            return artifact

        except Exception as e:
            raise ModelTrainingException("Model training failed", e)
