"""
Model Loader
Utility for loading and inspecting persisted model artifacts.
"""

from pathlib import Path
from typing import Any

from churn.constants.constants import MODEL_FILE, PIPELINE_FILE
from churn.utils.common import load_object
from churn.utils.logger import logger


class ModelLoader:
    """Loads and caches model + preprocessing pipeline."""

    def __init__(
        self,
        model_path: Path = MODEL_FILE,
        pipeline_path: Path = PIPELINE_FILE,
    ):
        self.model_path = model_path
        self.pipeline_path = pipeline_path
        self._model: Any = None
        self._pipeline: Any = None

    @property
    def model(self) -> Any:
        if self._model is None:
            logger.info(f"Loading model from: {self.model_path}")
            self._model = load_object(self.model_path)
        return self._model

    @property
    def pipeline(self) -> Any:
        if self._pipeline is None:
            logger.info(f"Loading preprocessing pipeline from: {self.pipeline_path}")
            self._pipeline = load_object(self.pipeline_path)
        return self._pipeline

    def model_info(self) -> dict:
        m = self.model
        return {
            "model_type": type(m).__name__,
            "model_path": str(self.model_path),
            "pipeline_path": str(self.pipeline_path),
            "model_params": m.get_params() if hasattr(m, "get_params") else {},
        }
