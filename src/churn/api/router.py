"""
API Router — prediction endpoints.
"""

from fastapi import APIRouter, HTTPException, status

from churn.api.schemas import (
    CustomerFeatures,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
)
from churn.api.prediction_service import get_predictor, predict_single, predict_batch
from churn.utils.exception import PredictionException
from churn.utils.logger import logger

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Check API health and model availability."""
    try:
        predictor = get_predictor()
        predictor._load_artifacts()
        model_loaded = True
    except Exception:
        model_loaded = False

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        version="1.0.0",
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Predictions"],
    summary="Predict churn for a single customer",
)
def predict(customer: CustomerFeatures) -> PredictionResponse:
    """
    Accepts customer feature data and returns:
    - **prediction**: 0 (no churn) or 1 (churn)
    - **prediction_label**: human-readable label
    - **churn_probability**: probability of churning
    - **no_churn_probability**: probability of staying
    """
    try:
        return predict_single(customer)
    except PredictionException as exc:
        logger.error(f"Prediction error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error(f"Unexpected error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during prediction.",
        )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Predictions"],
    summary="Predict churn for multiple customers",
)
def predict_batch_endpoint(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Batch prediction endpoint — accepts up to 1000 records."""
    if len(request.records) > 1000:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Batch size exceeds maximum of 1000 records.",
        )
    try:
        return predict_batch(request.records)
    except PredictionException as exc:
        logger.error(f"Batch prediction error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
