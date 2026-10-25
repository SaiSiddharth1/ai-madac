"""
Temporal Detection and Validation Layer for AI-MADAC.
Provides rigorous, schema-aware temporal classification to prevent non-temporal
columns (e.g., OverTime, MonthlyIncome, DailyRate) from being falsely treated as dates/timestamps.
"""

import re
from typing import Any, Optional
import pandas as pd


# Explicit non-temporal terms in column names that must NEVER be treated as datetime
NON_TEMPORAL_KEYWORDS = {
    "income", "salary", "rate", "cost", "price", "fee", "expense", "charges",
    "amount", "profit", "balance", "limit", "wage", "hike", "satisfaction",
    "involvement", "level", "percent", "percentage", "overtime", "parttime",
    "fulltime", "standardhours", "count", "num", "score", "distance",
    "trainingtimes", "workingyears", "yearsatcompany", "yearsincurrentrole",
    "yearssincelastpromotion", "yearswithcurrmanager", "duration", "tenure"
}

GENUINE_DATE_COLUMN_PATTERNS = [
    r"^(?:order|ship|invoice|transaction|sale|purchase|created|updated|event|registration)?_?(?:date|timestamp|time|datetime|at)$",
    r"^(?:date|timestamp|datetime|invoicedate|orderdate|shipdate|saledate|purchasedate)$",
    r"^(?:year|month|day|quarter)$",
]


def is_temporal_column(
    col_name: str,
    dtype: str = "",
    sample_values: Optional[list[Any]] = None,
    schema: Optional[dict[str, Any]] = None,
) -> bool:
    """
    Determine if a column is genuinely temporal (datetime / date).
    Strictly checks:
    1. Negative keywords (OverTime, MonthlyIncome, DailyRate -> NEVER temporal)
    2. Data types (datetime64, timestamp, date)
    3. Positive naming patterns (order_date, invoicedate, date, timestamp)
    4. Parseability of sample values as real dates (years 1900-2100)
    """
    if not col_name:
        return False

    name_lower = col_name.strip().lower()
    name_clean = re.sub(r"[\s_]+", "_", name_lower)

    if schema and (not dtype or sample_values is None):
        for col_info in schema.get("columns", []):
            if isinstance(col_info, dict) and col_info.get("name", "").lower() == name_lower:
                if not dtype:
                    dtype = str(col_info.get("dtype", ""))
                if sample_values is None:
                    sample_values = col_info.get("sample_values") or col_info.get("sample") or []
                break

    dtype_lower = str(dtype).strip().lower()

    # 1. Negative checks: Non-temporal metric and category columns
    # 'overtime', 'monthlyincome', 'dailyrate', 'hourlyrate', etc.
    if name_clean in ("overtime", "parttime", "fulltime", "over_time", "part_time"):
        return False

    # If column name contains rate, income, salary, etc.
    for kw in NON_TEMPORAL_KEYWORDS:
        if kw in name_clean:
            return False

    # 2. Check schema dtype
    if any(dt in dtype_lower for dt in ("datetime", "timestamp")):
        return True
    if dtype_lower == "date":
        return True

    # 3. Check positive name patterns
    is_date_name = any(re.match(pat, name_clean) for pat in GENUINE_DATE_COLUMN_PATTERNS)
    if not is_date_name:
        # Check if ends with _date, _timestamp, or _at
        if name_clean.endswith(("_date", "date", "_timestamp", "timestamp", "_at")) and not any(kw in name_clean for kw in NON_TEMPORAL_KEYWORDS):
            is_date_name = True

    if not is_date_name:
        return False

    # 4. Check sample values if available
    if sample_values:
        valid_samples = [s for s in sample_values if s is not None and str(s).strip() != ""]
        if valid_samples:
            # If samples are boolean or simple small integers (e.g. 1, 2, 3) or Yes/No
            str_samples = [str(s).strip() for s in valid_samples[:5]]
            if any(s.lower() in ("yes", "no", "true", "false", "y", "n") for s in str_samples):
                return False

            parsed_count = 0
            for s in str_samples:
                try:
                    # Avoid parsing single small numbers like 1, 2, 3 as dates
                    if s.isdigit() and len(s) < 4:
                        continue
                    dt = pd.to_datetime(s, errors="raise")
                    if 1900 <= dt.year <= 2100:
                        parsed_count += 1
                except Exception:
                    pass

            # If none of the samples could be parsed as dates between 1900 and 2100
            if parsed_count == 0 and not any(dt in dtype_lower for dt in ("datetime", "timestamp", "date")):
                return False

    return True


def find_temporal_column(
    schema: dict[str, Any] | list[Any],
    prefer_column: Optional[str] = None,
) -> Optional[str]:
    """
    Find the genuine temporal date/time column in the dataset schema.
    Returns None if dataset does not have a genuine temporal column.
    """
    cols_list = schema.get("columns", []) if isinstance(schema, dict) else schema
    valid_temporal_cols = []

    for col in cols_list:
        if isinstance(col, dict):
            name = col.get("name", "")
            dtype = col.get("dtype", "")
            samples = col.get("sample_values") or col.get("sample") or []
        else:
            name = str(col)
            dtype = ""
            samples = []

        if is_temporal_column(name, dtype=dtype, sample_values=samples):
            valid_temporal_cols.append(name)

    if not valid_temporal_cols:
        return None

    if prefer_column:
        for c in valid_temporal_cols:
            if c.lower() == prefer_column.lower():
                return c

    # Prioritize order_date, date, invoicedate over year/day
    for c in valid_temporal_cols:
        low = c.lower()
        if any(k in low for k in ("order_date", "orderdate", "invoicedate", "date", "timestamp", "created_at")):
            return c

    return valid_temporal_cols[0]


def is_temporal_intent(question: str, has_temporal_column: bool) -> bool:
    """
    Check if the user query is asking for a time-series / temporal trend.
    Requires:
    1. The user asks for temporal progression (over time, month by month, sales trend, etc.)
    2. Has a genuine temporal column in the dataset (or is asking for dates)
    3. Not a rate/salary metric (e.g., 'monthly income', 'monthly rate', 'daily rate' are not trends!)
    """
    q_lower = question.lower()

    # Rule: If "monthly", "daily", "hourly" is immediately followed by a metric term, it's not a trend indicator
    cleaned_q = re.sub(
        r"\b(?:monthly|daily|hourly|weekly|yearly|annual)\s+(?:income|salary|rate|cost|expense|charges|revenue|sales|profit|wage|fee|amount|balance)\b",
        " ",
        q_lower
    )

    explicit_temporal = bool(re.search(
        r"\b(?:over time|change over time|changed over time|trend|trend over time|historical trend|month by month|day by day|year by year|sales trend|revenue trend|quantity trend|timeline)\b",
        cleaned_q
    )) or bool(re.search(
        r"\b(?:from|between)\s+\d{4}\s+(?:to|and|-)\s+\d{4}\b",
        cleaned_q
    ))

    periodic = bool(re.search(r"\b(?:by month|by year|by day|by quarter|by date|per month|per year|monthly total|yearly total)\b", cleaned_q))

    return bool(has_temporal_column and (explicit_temporal or periodic))
