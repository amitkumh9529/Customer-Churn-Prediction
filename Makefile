.PHONY: install train api test lint docker-build docker-up docker-down clean mlflow frontend

# ─── Setup ────────────────────────────────────────────────────────────────────
install:
	pip install -e . && pip install -r requirements.txt

# ─── Training ─────────────────────────────────────────────────────────────────
train:
	PYTHONPATH=src python scripts/train_pipeline.py

predict:
	PYTHONPATH=src python scripts/predict_pipeline.py

batch-infer:
	PYTHONPATH=src python scripts/batch_inference.py \
		--input data/raw/Telco-Customer-Churn.csv \
		--output artifacts/reports/batch_results.csv

# ─── API ──────────────────────────────────────────────────────────────────────
api:
	PYTHONPATH=src uvicorn churn.api.main:app \
		--host 0.0.0.0 --port 8000 --reload

api-prod:
	PYTHONPATH=src uvicorn churn.api.main:app \
		--host 0.0.0.0 --port 8000 --workers 4

# ─── MLflow ───────────────────────────────────────────────────────────────────
mlflow:
	mlflow ui --backend-store-uri experiments/mlruns --port 5000

# ─── Frontend ─────────────────────────────────────────────────────────────────
frontend-install:
	cd frontend && npm install

frontend:
	cd frontend && npm run dev

frontend-build:
	cd frontend && npm run build

# ─── Tests ───────────────────────────────────────────────────────────────────
test:
	PYTHONPATH=src pytest tests/ -v --tb=short

test-cov:
	PYTHONPATH=src pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

# ─── Lint ────────────────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/

format:
	black src/ tests/ scripts/

# ─── Docker ───────────────────────────────────────────────────────────────────
docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# ─── Clean ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info/
	rm -f logs/app.log

clean-artifacts:
	rm -f artifacts/models/*.pkl
	rm -f artifacts/encoders/*.pkl
	rm -f artifacts/reports/*.json
