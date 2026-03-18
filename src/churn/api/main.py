"""
FastAPI application entry point.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from churn.api.router import router
from churn.api.prediction_service import get_predictor
from churn.models.model_loader import ModelLoader
from churn.utils.logger import logger




@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load model artifacts on startup."""
    logger.info("🚀 Starting Churn Prediction API...")
    try:
        predictor = get_predictor()
        predictor._load_artifacts()
        logger.info("✅ Model artifacts loaded successfully.")
    except Exception as e:
        logger.warning(f"⚠️  Could not pre-load model: {e}. Train first.")
    yield
    logger.info("🛑 API shutting down.")


app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "Production-grade REST API for predicting customer churn "
        "using a trained ML model on the Telco dataset."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

loader = ModelLoader()

@app.on_event("startup")
def load_model():
    loader.model
    loader.pipeline

def get_model():
    return loader.model

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["Root"])
def root():
    return {
        "service": "Customer Churn Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }

