"""
Centralized logging configuration using Loguru.
"""

import sys
from pathlib import Path
from loguru import logger

from churn.constants.constants import LOG_FILE, LOG_FORMAT


def setup_logger(log_file: str | Path = LOG_FILE, level: str = "INFO") -> None:
    """Configure loguru logger with file and stdout sinks."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove default handler
    logger.remove()

    # Stdout handler (colorized)
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )

    # File handler (structured)
    logger.add(
        str(log_path),
        format=LOG_FORMAT,
        level=level,
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        enqueue=True,  # thread-safe
    )


# Initialize on import
setup_logger()

__all__ = ["logger", "setup_logger"]
