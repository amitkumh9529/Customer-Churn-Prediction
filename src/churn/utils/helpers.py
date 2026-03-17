"""
Helper functions for data processing and model evaluation.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churn.utils.logger import logger


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute a comprehensive set of classification metrics."""
    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_prob is not None:
        metrics["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4)

    logger.info(f"Metrics computed: {metrics}")
    return metrics


def get_classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Return a formatted classification report string."""
    return classification_report(y_true, y_pred, target_names=["No Churn", "Churn"])


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> list[list[int]]:
    """Return confusion matrix as a nested list."""
    return confusion_matrix(y_true, y_pred).tolist()


def get_feature_importance(
    model: Any,
    feature_names: list[str],
    top_n: int = 20,
) -> dict[str, float]:
    """Extract feature importances from a trained model or pipeline."""
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            # Pipeline — try to get the last estimator
            estimator = model[-1] if hasattr(model, "__getitem__") else model
            if hasattr(estimator, "feature_importances_"):
                importances = estimator.feature_importances_
            elif hasattr(estimator, "coef_"):
                importances = np.abs(estimator.coef_[0])
            else:
                logger.warning("Model has no feature importances attribute.")
                return {}

        pairs = sorted(
            zip(feature_names, importances.tolist()),
            key=lambda x: x[1],
            reverse=True,
        )[:top_n]
        return {name: round(val, 6) for name, val in pairs}
    except Exception as e:
        logger.warning(f"Could not extract feature importances: {e}")
        return {}


def encode_target(series: pd.Series) -> pd.Series:
    """Map 'Yes'/'No' target column to 1/0."""
    return series.map({"Yes": 1, "No": 0}).astype(int)


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memory usage by downcasting numeric columns."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df
