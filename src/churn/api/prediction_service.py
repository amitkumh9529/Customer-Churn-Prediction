"""
Prediction service layer — sits between the router and the ML pipeline.
"""

from functools import lru_cache

from churn.pipeline.prediction_pipeline import PredictionPipeline
from churn.api.schemas import CustomerFeatures, PredictionResponse, BatchPredictionResponse
from churn.utils.logger import logger


@lru_cache(maxsize=1)
def get_predictor() -> PredictionPipeline:
    """Return a cached singleton prediction pipeline."""
    return PredictionPipeline()


def predict_single(customer: CustomerFeatures) -> PredictionResponse:
    """Score one customer record."""
    predictor = get_predictor()
    raw = customer.model_dump()
    result = predictor.predict(raw)
    logger.info(f"Single prediction: label={result['prediction_label']}, "
                f"prob={result['churn_probability']}")
    return PredictionResponse(**result)


def predict_batch(records: list[CustomerFeatures]) -> BatchPredictionResponse:
    """Score a batch of customer records."""
    predictor = get_predictor()
    raw_records = [r.model_dump() for r in records]
    results = predictor.predict_batch(raw_records)
    responses = [PredictionResponse(**r) for r in results]
    churn_count = sum(r.prediction for r in responses)
    return BatchPredictionResponse(
        total=len(responses),
        results=responses,
        churn_count=churn_count,
        churn_rate=round(churn_count / len(responses), 4) if responses else 0.0,
    )
