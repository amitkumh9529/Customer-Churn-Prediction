#!/usr/bin/env python
"""
Entry-point script: run a single prediction from the command line.

Usage:
    python scripts/predict_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churn.pipeline.prediction_pipeline import PredictionPipeline
from churn.utils.logger import logger

SAMPLE_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


def main():
    pipeline = PredictionPipeline()
    result = pipeline.predict(SAMPLE_CUSTOMER)
    logger.info(f"Prediction Result: {result}")
    print("\n── Prediction Result ──────────────────────")
    for k, v in result.items():
        print(f"  {k:25s}: {v}")
    print("──────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
