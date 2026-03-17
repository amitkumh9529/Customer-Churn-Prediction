# ChurnScope — Production ML System for Customer Churn Prediction

A fully production-grade, end-to-end machine learning system for predicting customer churn, built on the Telco Customer Churn dataset. The system follows real-world ML engineering practices: typed config entities, staged pipelines, MLflow experiment tracking, a FastAPI prediction service, and a React + Tailwind UI.

---

## Architecture Overview

```
Raw CSV ──► Data Ingestion ──► Data Validation ──► Feature Engineering
              ──► Data Transformation ──► Model Training (CV)
                ──► Model Evaluation ──► MLflow Registry
                                            │
                                    FastAPI API  ◄──  React UI
```

---

## Project Structure

```
customer_churn_ml/
├── configs/
│   ├── config.yaml            # Data paths, split ratios, feature lists
│   ├── model_config.yaml      # Model hyperparameters, evaluation config
│   └── logging_config.yaml    # Log rotation, format, retention
├── data/
│   ├── raw/                   # Original Telco CSV
│   ├── interim/               # Intermediate outputs
│   └── processed/             # Train / val / test splits
├── artifacts/
│   ├── models/                # best_model.pkl
│   ├── encoders/              # preprocessing_pipeline.pkl
│   └── reports/               # evaluation_metrics.json, feature_importance.json
├── experiments/
│   └── mlruns/                # MLflow tracking directory
├── logs/
│   └── app.log                # Rotating structured log file
├── scripts/
│   ├── train_pipeline.py      # Run full training pipeline
│   ├── predict_pipeline.py    # Score a single sample
│   └── batch_inference.py     # Batch scoring from CSV
├── src/churn/
│   ├── constants/             # Global path and column constants
│   ├── config/                # ConfigurationManager
│   ├── entity/                # Typed config + artifact dataclasses
│   ├── components/            # 7 pipeline stages (ingestion → pusher)
│   ├── pipeline/              # Training, prediction, batch pipelines
│   ├── models/                # ModelLoader utility
│   ├── api/                   # FastAPI app, router, schemas, service
│   └── utils/                 # logger, exception, common, helpers
├── frontend/                  # React + Tailwind UI
│   ├── src/
│   │   ├── App.tsx
│   │   ├── pages/Predict.tsx
│   │   ├── components/PredictionForm.tsx
│   │   └── services/api.ts
│   ├── Dockerfile
│   └── nginx.conf
├── tests/                     # 23 pytest unit tests
├── Dockerfile                 # Backend Docker image
├── docker-compose.yml         # Orchestrates API + MLflow + Frontend
└── Makefile                   # One-command shortcuts
```

---

## Quick Start (Local, without Docker)

### 1. Prerequisites

- Python ≥ 3.10
- Node.js ≥ 18 (for the frontend only)

### 2. Install Python dependencies

```bash
cd customer_churn_ml
pip install -r requirements.txt
```

### 3. Train the full ML pipeline

```bash
PYTHONPATH=src python scripts/train_pipeline.py
# or with Make:
make train
```

This runs all 7 stages in sequence and prints a summary:
```
╔══════════════════════════════════════╗
║  CUSTOMER CHURN — TRAINING PIPELINE  ║
╚══════════════════════════════════════╝
Stage 1: Data Ingestion     — train: 4929 | val: 705 | test: 1409
Stage 2: Data Validation    — schema OK, no nulls
Stage 3: Feature Engineering — 24 features (incl. tenure_group, services_count…)
Stage 4: Data Transformation — 46 features after OHE + scaling
Stage 5: Model Training      — CV: LR 0.843 | RF 0.840 | XGB 0.824
Stage 6: Model Evaluation    — Test AUC: 0.842 | F1: 0.613 | Recall: 0.786
Stage 7: Model Pusher        — Registered ChurnClassifier v1 in MLflow
✅ Pipeline completed in ~38s
```

### 4. Start the FastAPI prediction server

```bash
make api
# or manually:
PYTHONPATH=src uvicorn churn.api.main:app --host 0.0.0.0 --port 8000 --reload
```

API is now live at:
- Swagger UI: http://localhost:8000/docs
- ReDoc:       http://localhost:8000/redoc
- Health:      http://localhost:8000/api/v1/health

### 5. Run a single prediction

```bash
make predict
# or:
PYTHONPATH=src python scripts/predict_pipeline.py
```

### 6. Run batch inference

```bash
make batch-infer
# or:
PYTHONPATH=src python scripts/batch_inference.py \
    --input data/raw/Telco-Customer-Churn.csv \
    --output artifacts/reports/batch_results.csv
```

### 7. View MLflow experiments

```bash
make mlflow
# or:
mlflow ui --backend-store-uri experiments/mlruns --port 5000
```

Open http://localhost:5000 to browse:
- Experiment runs and metrics
- Hyperparameters logged per run
- Registered model: `ChurnClassifier`
- Artifact downloads (model, preprocessing pipeline, reports)

### 8. Start the React frontend

```bash
make frontend-install   # once
make frontend           # starts dev server at http://localhost:3000
```

The UI features:
- Sectioned customer input form (Demographics / Account / Services / Charges)
- Real-time churn probability gauge
- Colour-coded risk level (Low / Medium / High)
- Action recommendation panel
- API health indicator

---

## Run with Docker Compose

```bash
# Build all images and start all services
make docker-up
# or:
docker-compose up -d --build

# View logs
make docker-logs

# Stop everything
make docker-down
```

Services started:
| Service  | URL                    | Description                  |
|----------|------------------------|------------------------------|
| API      | http://localhost:8000  | FastAPI prediction backend   |
| MLflow   | http://localhost:5000  | Experiment tracking server   |
| Frontend | http://localhost:3000  | React prediction UI          |

---

## Run Tests

```bash
make test
# with coverage:
make test-cov
```

Test summary (23 tests):
```
tests/test_api.py              8 passed  — health, predict, batch, validation
tests/test_data_ingestion.py   5 passed  — splits, target encoding, schema
tests/test_transformation.py   4 passed  — nulls, pipeline type, feature names
tests/test_training.py         6 passed  — metrics, feature importance, smoke
```

---

## API Reference

### `POST /api/v1/predict`

Accepts a single customer record and returns churn prediction.

**Request body** (all fields required):
```json
{
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
  "TotalCharges": 29.85
}
```

**Response**:
```json
{
  "prediction": 1,
  "prediction_label": "Churn",
  "churn_probability": 0.7823,
  "no_churn_probability": 0.2177,
  "model_version": "1.0.0"
}
```

### `POST /api/v1/predict/batch`

Accepts up to 1000 records:
```json
{ "records": [ {…}, {…} ] }
```

Returns aggregate stats + per-record results including `churn_rate`.

### `GET /api/v1/health`

Returns model load status and API version.

---

## ML Pipeline Details

| Stage | Component | Key actions |
|-------|-----------|-------------|
| 1 | `DataIngestion` | Load CSV, clean TotalCharges, encode target, stratified 70/10/20 split |
| 2 | `DataValidation` | Schema check, row count threshold, null audit, JSON report |
| 3 | `FeatureEngineering` | tenure_group, avg_monthly_cost, has_streaming, services_count, is_month_to_month |
| 4 | `DataTransformation` | Sklearn Pipeline: median imputer + StandardScaler (numeric), mode imputer + OneHotEncoder (categorical) |
| 5 | `ModelTrainer` | 5-fold stratified CV: LogisticRegression, RandomForest, XGBoost — selects best by ROC-AUC |
| 6 | `ModelEvaluation` | Full test-set metrics, confusion matrix, classification report, feature importances |
| 7 | `ModelPusher` | Logs params/metrics/artifacts to MLflow, registers model as `ChurnClassifier` |

---

## Model Performance

| Metric    | Train  | Val    | Test   |
|-----------|--------|--------|--------|
| ROC-AUC   | 0.848  | 0.858  | 0.842  |
| F1 Score  | 0.629  | 0.645  | 0.613  |
| Recall    | 0.804  | 0.829  | 0.786  |
| Accuracy  | 0.748  | 0.757  | 0.737  |

> Model is tuned for **high recall** — important in churn detection where missing an at-risk customer is costlier than a false alarm.

---

## Configuration

All pipeline behaviour is controlled via `configs/`:

- **`config.yaml`** — feature lists, split ratios, data paths, MLflow settings
- **`model_config.yaml`** — per-model hyperparameters, CV folds, primary metric, threshold
- **`logging_config.yaml`** — log file rotation, retention, format

To add a new model, enable it in `model_config.yaml` and add its instantiation to `ModelTrainer._build_models()`.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTHONPATH` | `src` | Must include `src/` for imports to resolve |
| `VITE_API_URL` | `http://localhost:8000` | Backend URL used by the frontend |
| `MLFLOW_TRACKING_URI` | `file://experiments/mlruns` | Can be set to a remote MLflow server |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Core | Scikit-learn, XGBoost, imbalanced-learn |
| Experiment Tracking | MLflow |
| API | FastAPI + Uvicorn |
| Frontend | React 18, Tailwind CSS, Vite |
| Containerization | Docker, Docker Compose |
| Testing | pytest, pytest-asyncio, httpx |
| Logging | Loguru |
| Config | PyYAML, Pydantic v2 |
