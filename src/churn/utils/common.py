"""
Common utility functions shared across pipeline components.
"""

import json
import os
import pickle
from pathlib import Path
from typing import Any

import yaml

from churn.utils.logger import logger
from churn.utils.exception import ChurnBaseException


def read_yaml(path: str | Path) -> dict:
    """Read a YAML file and return as dictionary."""
    try:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"YAML file not found: {path}")
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise ChurnBaseException(f"Failed to read YAML: {path}", e)


def ensure_dir(path: str | Path) -> Path:
    """Create directory if it doesn't exist, return Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_object(obj: Any, path: str | Path) -> None:
    """Serialize and save a Python object using pickle."""
    try:
        path = Path(path)
        ensure_dir(path.parent)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Object saved to: {path}")
    except Exception as e:
        raise ChurnBaseException(f"Failed to save object to {path}", e)


def load_object(path: str | Path) -> Any:
    """Load a pickled Python object."""
    try:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"Object loaded from: {path}")
        return obj
    except Exception as e:
        raise ChurnBaseException(f"Failed to load object from {path}", e)


def save_json(data: dict, path: str | Path) -> None:
    """Save a dictionary as a JSON file."""
    try:
        path = Path(path)
        ensure_dir(path.parent)
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"JSON saved to: {path}")
    except Exception as e:
        raise ChurnBaseException(f"Failed to save JSON to {path}", e)


def load_json(path: str | Path) -> dict:
    """Load a JSON file as a dictionary."""
    try:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise ChurnBaseException(f"Failed to load JSON from {path}", e)


def get_size_mb(path: str | Path) -> float:
    """Return the file size in megabytes."""
    return round(os.path.getsize(path) / (1024 * 1024), 3)
