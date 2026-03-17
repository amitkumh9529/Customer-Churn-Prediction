"""
Global constants for the customer churn prediction system.
"""

from pathlib import Path

# ─── Project root ───────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[3]  # src/churn/constants/ → project root

# ─── Config paths ────────────────────────────────────────────────────────────
CONFIG_FILE_PATH = ROOT_DIR / "configs" / "config.yaml"
MODEL_CONFIG_FILE_PATH = ROOT_DIR / "configs" / "model_config.yaml"
LOGGING_CONFIG_FILE_PATH = ROOT_DIR / "configs" / "logging_config.yaml"

# ─── Data paths ──────────────────────────────────────────────────────────────
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
INTERIM_DATA_DIR = ROOT_DIR / "data" / "interim"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"

RAW_DATA_FILE = RAW_DATA_DIR / "Telco-Customer-Churn.csv"

TRAIN_FILE = PROCESSED_DATA_DIR / "train.csv"
TEST_FILE = PROCESSED_DATA_DIR / "test.csv"
VAL_FILE = PROCESSED_DATA_DIR / "val.csv"

# ─── Artifact paths ───────────────────────────────────────────────────────────
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
ENCODERS_DIR = ARTIFACTS_DIR / "encoders"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

MODEL_FILE = MODELS_DIR / "best_model.pkl"
PIPELINE_FILE = ENCODERS_DIR / "preprocessing_pipeline.pkl"
METRICS_FILE = REPORTS_DIR / "evaluation_metrics.json"
FEATURE_IMPORTANCE_FILE = REPORTS_DIR / "feature_importance.json"

# ─── MLflow ───────────────────────────────────────────────────────────────────
MLFLOW_DIR = ROOT_DIR / "experiments" / "mlruns"
MLFLOW_EXPERIMENT_NAME = "customer-churn-prediction"
MLFLOW_REGISTERED_MODEL_NAME = "ChurnClassifier"

# ─── Data columns ────────────────────────────────────────────────────────────
TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ─── Model ────────────────────────────────────────────────────────────────────
DEFAULT_THRESHOLD = 0.5
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1
CV_FOLDS = 5

# ─── API ─────────────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
API_V1_PREFIX = "/api/v1"

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_FILE = ROOT_DIR / "logs" / "app.log"
LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
