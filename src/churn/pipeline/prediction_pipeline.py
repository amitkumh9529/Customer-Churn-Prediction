"""
Real-time Prediction Pipeline
Loads model + preprocessing pipeline and scores a single customer record.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from churn.constants.constants import MODEL_FILE, PIPELINE_FILE
from churn.utils.common import load_object
from churn.utils.exception import PredictionException
from churn.utils.logger import logger


class PredictionPipeline:
    """
    Singleton-style pipeline that lazily loads model and preprocessor.
    Thread-safe for concurrent API requests.
    """

    _model = None
    _preprocessor = None

    def __init__(
        self,
        model_path: Path = MODEL_FILE,
        pipeline_path: Path = PIPELINE_FILE,
    ):
        self.model_path = model_path
        self.pipeline_path = pipeline_path

    def _load_artifacts(self) -> None:
        if PredictionPipeline._model is None:
            logger.info("Loading model artifact...")
            PredictionPipeline._model = load_object(self.model_path)
        if PredictionPipeline._preprocessor is None:
            logger.info("Loading preprocessing pipeline...")
            PredictionPipeline._preprocessor = load_object(self.pipeline_path)

    def predict(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Accepts a raw customer dict, returns prediction + probability.

        Returns:
            {
              "prediction": 0 | 1,
              "prediction_label": "No Churn" | "Churn",
              "churn_probability": float,
              "no_churn_probability": float,
            }
        """
        # AFTER:
        try:
            self._load_artifacts()

            df = pd.DataFrame([data])

            # Clean raw inputs — mirrors DataIngestion._clean_data
            # TotalCharges can be ' ' (blank string) in raw CSV rows
            if "TotalCharges" in df.columns:
                df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
                if df["TotalCharges"].isna().any():
                    fallback = float(df["MonthlyCharges"].iloc[0]) if "MonthlyCharges" in df.columns else 0.0
                    df["TotalCharges"] = df["TotalCharges"].fillna(fallback)

            # Drop any columns the preprocessor was not trained on
            for col in ["customerID", "Churn"]:
                if col in df.columns:
                    df = df.drop(columns=[col])

            logger.debug(f"Input shape after cleaning: {df.shape}")

            X_transformed = PredictionPipeline._preprocessor.transform(df)
            prob = PredictionPipeline._preprocessor  # just reference for type checker

            probs = PredictionPipeline._model.predict_proba(X_transformed)[0]
            no_churn_prob, churn_prob = float(probs[0]), float(probs[1])
            prediction = int(churn_prob >= 0.5)

            result = {
                "prediction": prediction,
                "prediction_label": "Churn" if prediction == 1 else "No Churn",
                "churn_probability": round(churn_prob, 4),
                "no_churn_probability": round(no_churn_prob, 4),
            }
            logger.info(f"Prediction result: {result}")
            return result

        except Exception as e:
            raise PredictionException("Prediction failed", e)

    def predict_batch(self, records: list[dict]) -> list[dict]:
        """Score a list of customer records."""
        return [self.predict(r) for r in records]
