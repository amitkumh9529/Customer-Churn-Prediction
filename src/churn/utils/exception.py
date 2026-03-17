"""
Custom exception classes for the churn prediction system.
"""

import sys
import traceback


def _get_error_details(error: Exception) -> str:
    _, _, exc_tb = sys.exc_info()
    if exc_tb is None:
        return str(error)
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    return f"Error in [{file_name}] at line [{line_number}]: {str(error)}"


class ChurnBaseException(Exception):
    """Base exception for the churn prediction system."""

    def __init__(self, message: str, error: Exception | None = None):
        self.message = message
        if error:
            self.detail = _get_error_details(error)
        else:
            self.detail = message
        super().__init__(self.detail)

    def __str__(self) -> str:
        return self.detail


class DataIngestionException(ChurnBaseException):
    """Raised when data ingestion fails."""
    pass


class DataValidationException(ChurnBaseException):
    """Raised when data validation fails."""
    pass


class DataTransformationException(ChurnBaseException):
    """Raised when data transformation fails."""
    pass


class FeatureEngineeringException(ChurnBaseException):
    """Raised when feature engineering fails."""
    pass


class ModelTrainingException(ChurnBaseException):
    """Raised when model training fails."""
    pass


class ModelEvaluationException(ChurnBaseException):
    """Raised when model evaluation fails."""
    pass


class ModelPusherException(ChurnBaseException):
    """Raised when saving/registering a model fails."""
    pass


class PredictionException(ChurnBaseException):
    """Raised when prediction fails."""
    pass


class ConfigurationException(ChurnBaseException):
    """Raised when configuration loading fails."""
    pass
