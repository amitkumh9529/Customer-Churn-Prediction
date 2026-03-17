"""
Shared fixtures for pytest test suite.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Make src importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture
def sample_raw_df() -> pd.DataFrame:
    """Minimal raw Telco-style dataframe."""
    return pd.DataFrame({
        "customerID":      ["A001", "A002", "A003", "A004", "A005"],
        "gender":          ["Female", "Male", "Male", "Female", "Male"],
        "SeniorCitizen":   [0, 0, 1, 0, 1],
        "Partner":         ["Yes", "No", "No", "Yes", "No"],
        "Dependents":      ["No", "No", "No", "Yes", "No"],
        "tenure":          [1, 34, 2, 45, 10],
        "PhoneService":    ["No", "Yes", "Yes", "No", "Yes"],
        "MultipleLines":   ["No phone service", "No", "No", "No phone service", "Yes"],
        "InternetService": ["DSL", "DSL", "DSL", "DSL", "Fiber optic"],
        "OnlineSecurity":  ["No", "Yes", "Yes", "Yes", "No"],
        "OnlineBackup":    ["Yes", "No", "Yes", "No", "No"],
        "DeviceProtection":["No", "Yes", "No", "Yes", "No"],
        "TechSupport":     ["No", "No", "No", "Yes", "No"],
        "StreamingTV":     ["No", "No", "No", "No", "No"],
        "StreamingMovies": ["No", "No", "No", "No", "No"],
        "Contract":        ["Month-to-month", "One year", "Month-to-month", "One year", "Month-to-month"],
        "PaperlessBilling":["Yes", "No", "Yes", "No", "Yes"],
        "PaymentMethod":   ["Electronic check", "Mailed check", "Mailed check",
                            "Bank transfer (automatic)", "Electronic check"],
        "MonthlyCharges":  [29.85, 56.95, 53.85, 42.30, 70.70],
        "TotalCharges":    ["29.85", "1889.5", "108.15", "1840.75", "151.65"],
        "Churn":           ["No", "No", "Yes", "No", "Yes"],
    })


@pytest.fixture
def sample_customer_dict() -> dict:
    return {
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


@pytest.fixture
def tiny_X_y():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 5))
    y = (X[:, 0] + rng.standard_normal(100) > 0).astype(int)
    return X, y


@pytest.fixture
def trained_lr(tiny_X_y):
    X, y = tiny_X_y
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    return model
