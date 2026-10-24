"""
Robust, recursive JSON-safe serialization utility for AI-MADAC.
Converts all Pandas, NumPy, Decimal, Datetime, and UUID objects to native JSON primitives.
"""

from datetime import date, datetime, time
from decimal import Decimal
import math
from typing import Any
import uuid

import numpy as np
import pandas as pd


def to_json_safe(obj: Any) -> Any:
    """
    Recursively converts arbitrary Python data structures into clean JSON-serializable primitives.
    Handles:
    - pd.Timestamp, np.datetime64, datetime, date, time -> ISO string or formatted string
    - Decimal, np.floating -> float (or None if NaN/Inf)
    - np.integer -> int
    - np.bool_ -> bool
    - float -> None if NaN or Infinite
    - pd.DataFrame -> list of JSON-safe dicts
    - pd.Series -> list of JSON-safe values
    - np.ndarray -> list of JSON-safe values
    - UUID -> str
    - Pydantic BaseModel -> model_dump(mode="json")
    - dict, list, tuple, set -> recursively processed
    """
    if obj is None:
        return None

    # Pydantic models
    if hasattr(obj, "model_dump"):
        return to_json_safe(obj.model_dump(mode="json"))

    # Pandas DataFrame
    if isinstance(obj, pd.DataFrame):
        return [to_json_safe(record) for record in obj.to_dict(orient="records")]

    # Pandas Series or Numpy array
    if isinstance(obj, (pd.Series, np.ndarray)):
        return [to_json_safe(item) for item in obj.tolist()]

    # Temporal objects
    if isinstance(obj, (pd.Timestamp, datetime)):
        if hasattr(obj, "hour") and obj.hour == 0 and obj.minute == 0 and obj.second == 0 and obj.microsecond == 0:
            return obj.strftime("%Y-%m-%d")
        return obj.isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, time):
        return obj.isoformat()
    if isinstance(obj, np.datetime64):
        return str(pd.to_datetime(obj))

    # Numbers
    if isinstance(obj, Decimal):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, (np.floating, float)):
        return None if (math.isnan(obj) or math.isinf(obj)) else float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)

    # UUID
    if isinstance(obj, uuid.UUID):
        return str(obj)

    # Dictionaries
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}

    # Collections
    if isinstance(obj, (list, tuple, set)):
        return [to_json_safe(item) for item in obj]

    # Catch-all for pandas/numpy NA
    if pd.isna(obj):
        return None

    return obj
