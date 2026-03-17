#!/usr/bin/env python
"""
Entry-point script: run batch inference on a CSV file.

Usage:
    python scripts/batch_inference.py --input data/raw/Telco-Customer-Churn.csv
    python scripts/batch_inference.py --input data/raw/Telco-Customer-Churn.csv \
                                      --output artifacts/reports/results.csv
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churn.pipeline.batch_prediction_pipeline import BatchPredictionPipeline
from churn.utils.logger import logger


def main():
    parser = argparse.ArgumentParser(description="Batch churn inference")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", default=None, help="Path to output CSV")
    args = parser.parse_args()

    pipeline = BatchPredictionPipeline(
        input_path=args.input,
        output_path=args.output,
    )
    output_path = pipeline.run()
    print(f"\n✅ Batch inference complete. Results saved to: {output_path}\n")


if __name__ == "__main__":
    main()
