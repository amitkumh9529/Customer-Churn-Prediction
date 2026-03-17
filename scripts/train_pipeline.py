#!/usr/bin/env python
"""
Entry-point script: run the full training pipeline.

Usage:
    python scripts/train_pipeline.py
"""

import sys
from pathlib import Path

# Make src importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churn.pipeline.training_pipeline import TrainingPipeline
from churn.utils.logger import logger


def main():
    logger.info("Launching training pipeline...")
    pipeline = TrainingPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
