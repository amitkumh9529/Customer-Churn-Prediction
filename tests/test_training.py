"""
Unit tests for ModelTrainer and ModelEvaluation components.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from churn.utils.helpers import compute_metrics, get_feature_importance


# ── helpers unit tests ───────────────────────────────────────────────────────

def test_compute_metrics_perfect():
    y = np.array([0, 1, 0, 1])
    metrics = compute_metrics(y, y, y.astype(float))
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_compute_metrics_no_prob():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0])
    metrics = compute_metrics(y_true, y_pred)
    assert "accuracy" in metrics
    assert "roc_auc" not in metrics


def test_compute_metrics_range():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, 200)
    y_pred = rng.integers(0, 2, 200)
    y_prob = rng.uniform(0, 1, 200)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    for v in metrics.values():
        assert 0.0 <= v <= 1.0


def test_feature_importance_from_lr(trained_lr):
    names = [f"feat_{i}" for i in range(5)]
    importance = get_feature_importance(trained_lr, names)
    assert len(importance) == 5
    assert all(v >= 0 for v in importance.values())


def test_feature_importance_sorted(trained_lr):
    names = [f"feat_{i}" for i in range(5)]
    importance = get_feature_importance(trained_lr, names)
    values = list(importance.values())
    assert values == sorted(values, reverse=True)


# ── ModelTrainer smoke test with synthetic data ───────────────────────────────

def test_model_trainer_smoke(tmp_path):
    """Smoke test: trainer runs end-to-end on tiny synthetic data."""
    from sklearn.linear_model import LogisticRegression
    from churn.components.model_trainer import ModelTrainer
    from churn.entity.config_entity import ModelTrainerConfig
    from churn.entity.artifact_entity import DataTransformationArtifact

    rng = np.random.default_rng(42)
    n = 120
    X = rng.standard_normal((n, 5))
    y = (X[:, 0] > 0).astype(int)
    cols = [f"feat_{i}" for i in range(5)]

    def _save_split(X_part, y_part, name):
        df = pd.DataFrame(X_part, columns=cols)
        df["Churn"] = y_part
        p = tmp_path / f"{name}.csv"
        df.to_csv(p, index=False)
        return p

    idx = np.arange(n)
    train_p = _save_split(X[idx[:80]], y[idx[:80]], "transformed_train")
    val_p   = _save_split(X[idx[80:100]], y[idx[80:100]], "transformed_val")
    test_p  = _save_split(X[idx[100:]], y[idx[100:]], "transformed_test")

    transformation_artifact = DataTransformationArtifact(
        pipeline_path=tmp_path / "pipe.pkl",
        transformed_train_path=train_p,
        transformed_test_path=test_p,
        transformed_val_path=val_p,
        feature_names=cols,
    )

    models_params = {
        "logistic_regression": {
            "enabled": True,
            "params": {"C": 1.0, "max_iter": 200, "random_state": 42},
        },
        "random_forest": {"enabled": False, "params": {}},
        "xgboost":       {"enabled": False, "params": {}},
    }

    cfg = ModelTrainerConfig(
        models_dir=tmp_path / "models",
        reports_dir=tmp_path / "reports",
        random_state=42,
        cv_folds=3,
        primary_metric="roc_auc",
        models_params=models_params,
    )

    trainer = ModelTrainer(cfg, transformation_artifact)
    artifact = trainer.run()

    assert artifact.is_trained
    assert artifact.model_path.exists()
    assert "roc_auc" in artifact.val_metrics
    assert artifact.best_model_name == "LogisticRegression"
