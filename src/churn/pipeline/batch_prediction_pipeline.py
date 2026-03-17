"""
Batch Prediction Pipeline
Scores an entire CSV of customers and outputs results to artifacts/reports/.
"""

from pathlib import Path

import pandas as pd

from churn.constants.constants import MODEL_FILE, PIPELINE_FILE, REPORTS_DIR
from churn.pipeline.prediction_pipeline import PredictionPipeline
from churn.utils.common import ensure_dir
from churn.utils.exception import PredictionException
from churn.utils.logger import logger


class BatchPredictionPipeline:
    def __init__(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
        model_path: Path = MODEL_FILE,
        pipeline_path: Path = PIPELINE_FILE,
    ):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path) if output_path else REPORTS_DIR / "batch_predictions.csv"
        self.predictor = PredictionPipeline(model_path, pipeline_path)

    def run(self) -> Path:
        logger.info("=" * 60)
        logger.info("BATCH PREDICTION PIPELINE — START")
        try:
            # AFTER:
            df = pd.read_csv(self.input_path)
            logger.info(f"Loaded {len(df)} records from {self.input_path}")

            # Clean TotalCharges — raw CSV has blank strings for some rows
            if "TotalCharges" in df.columns:
                df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
                n_bad = df["TotalCharges"].isna().sum()
                if n_bad > 0:
                    logger.warning(f"Imputing {n_bad} blank TotalCharges with MonthlyCharges value")
                    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"])

            # Drop target & ID if present
            for col in ["Churn", "customerID"]:
                if col in df.columns:
                    df = df.drop(columns=[col])

            records = df.to_dict(orient="records")
            results = self.predictor.predict_batch(records)

            df_out = df.copy()
            df_out["prediction"] = [r["prediction"] for r in results]
            df_out["prediction_label"] = [r["prediction_label"] for r in results]
            df_out["churn_probability"] = [r["churn_probability"] for r in results]

            ensure_dir(self.output_path.parent)
            df_out.to_csv(self.output_path, index=False)

            churn_rate = df_out["prediction"].mean()
            logger.info(f"Batch scoring done — {len(df_out)} rows | churn rate: {churn_rate:.2%}")
            logger.info(f"Results saved to: {self.output_path}")
            return self.output_path

        except Exception as e:
            raise PredictionException("Batch prediction failed", e)
