"""
SQL Result Validator.
Authoritative validator ensuring SQL execution results conform to the canonical AnalysisSpec:
- Grouped queries MUST contain group_by columns and multiple rows (not scalar totals).
- Scalar queries MUST return a single scalar row without accidental grouping.
- Distribution queries MUST return raw numeric observations (not premature aggregations).
- Multi-series queries MUST preserve series dimension and X-axis columns.
"""

from typing import Any, Optional
import logging
import pandas as pd

from app.schemas.query_contract import QueryContract

logger = logging.getLogger(__name__)


class SQLResultValidator:
    """
    Validates executed SQL DataFrame against the canonical AnalysisSpec (QueryContract).
    """

    @classmethod
    def validate(
        cls,
        df: pd.DataFrame,
        contract: QueryContract,
        schema: Optional[dict[str, Any]] = None,
        dataset_row_count: Optional[int] = None,
    ) -> tuple[bool, Optional[str]]:
        """
        Validate DataFrame shape, columns, and cardinality against AnalysisSpec.
        Returns: (is_valid: bool, failure_reason: Optional[str])
        """
        if df is None:
            return False, "SQL result DataFrame is None."

        df_cols_lower = [str(c).lower() for c in df.columns]

        # 1. GROUPED RESULT VALIDATION
        is_grouped = bool(contract.group_by) or contract.answer_type == "GROUPED"
        if is_grouped:
            # All group_by columns must be present in result
            for g in contract.group_by:
                if g.lower() not in df_cols_lower:
                    return (
                        False,
                        f"EXPECTED GROUPED RESULT: Missing grouping column '{g}' in result columns {list(df.columns)}.",
                    )

            # If question didn't ask for LIMIT 1 (like 'which role has the highest...'),
            # a grouped query on a multi-row dataset must NOT return a single row global total!
            is_limit_one = (contract.limit == 1)
            if not is_limit_one and len(df) <= 1:
                # Check if dataset has multiple rows
                total_rows = dataset_row_count or (schema.get("row_count") if isinstance(schema, dict) else None) or 100
                if total_rows > 1:
                    return (
                        False,
                        f"EXPECTED GROUPED RESULT BUT RECEIVED SCALAR RESULT: "
                        f"AnalysisSpec expects group_by={contract.group_by}, but SQL returned only {len(df)} row.",
                    )

        # 2. SCALAR RESULT VALIDATION
        elif contract.answer_type == "SCALAR" and not contract.calculation:
            if len(df) > 1:
                return (
                    False,
                    f"EXPECTED SCALAR RESULT BUT RECEIVED MULTIPLE ROWS: "
                    f"AnalysisSpec expects a single scalar total, but SQL returned {len(df)} rows.",
                )
            if len(df.columns) > 1:
                return (
                    False,
                    f"EXPECTED SCALAR RESULT BUT RECEIVED MULTIPLE COLUMNS: "
                    f"AnalysisSpec expects a single scalar metric, but SQL returned columns {list(df.columns)}.",
                )

        # 3. DISTRIBUTION RESULT VALIDATION (Numeric)
        elif contract.answer_type == "DISTRIBUTION" or contract.suggested_chart_type == "histogram":
            target = contract.x_axis or contract.measure_column or (contract.dimensions[0] if contract.dimensions else None)
            if target and target.lower() not in df_cols_lower:
                return (
                    False,
                    f"EXPECTED DISTRIBUTION RESULT: Target column '{target}' missing from SQL result columns {list(df.columns)}.",
                )
            total_rows = dataset_row_count or (schema.get("row_count") if isinstance(schema, dict) else None) or 100
            if total_rows > 1 and len(df) <= 1:
                return (
                    False,
                    f"EXPECTED DISTRIBUTION OBSERVATIONS BUT RECEIVED SCALAR RESULT: "
                    f"Histogram requires continuous observation rows, but SQL returned {len(df)} row.",
                )

        # 4. MULTI-SERIES RESULT VALIDATION
        if contract.series_dimension:
            if contract.series_dimension.lower() not in df_cols_lower:
                return (
                    False,
                    f"EXPECTED MULTI-SERIES RESULT: Series dimension '{contract.series_dimension}' missing from result columns {list(df.columns)}.",
                )

        return True, None
