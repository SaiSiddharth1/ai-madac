"""
Deterministic Calculation Engine for AI-MADAC.
Executes and validates derived mathematical metrics (differences, ratios, profit margins, percentages).
Guarantees numerical truth and eliminates LLM arithmetic errors.
"""

from decimal import Decimal
import logging
import math
from typing import Any, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from app.schemas.query_contract import CalculationSpec, OperandSpec

logger = logging.getLogger(__name__)


class CalculationResult(BaseModel):
    calculated: bool = True
    operator: str
    label: str
    expression: str
    value: Any = None
    formatted_value: str = ""
    operands: dict[str, Any] = Field(default_factory=dict)
    is_finite: bool = True
    error: Optional[str] = None
    grouped: bool = False
    group_dimension: Optional[str] = None
    results_by_group: Optional[list[dict[str, Any]]] = None


class CalculationEngine:
    """
    Deterministic calculation engine that performs verified mathematical computations
    on data retrieved by SQL/Pandas.
    """

    @classmethod
    def compute_scalar(
        cls,
        val_a: Any,
        val_b: Any,
        operator: str,
        label: Optional[str] = None,
        name_a: str = "A",
        name_b: str = "B",
    ) -> CalculationResult:
        """
        Deterministically calculate scalar metrics with strict zero-division and NaN protection.
        """
        op = operator.upper()

        # Coerce values to float safely
        try:
            a = float(val_a) if val_a is not None else 0.0
        except (ValueError, TypeError):
            a = 0.0

        try:
            b = float(val_b) if val_b is not None else 0.0
        except (ValueError, TypeError):
            b = 0.0

        res_val = None
        error_msg = None
        expr_str = ""

        if op in ("SUBTRACT", "DIFFERENCE", "MINUS"):
            res_val = a - b
            expr_str = f"{a:,.2f} - {b:,.2f}"
            lbl = label or "Difference"

        elif op in ("ADD", "SUM", "PLUS"):
            res_val = a + b
            expr_str = f"{a:,.2f} + {b:,.2f}"
            lbl = label or "Sum"

        elif op in ("MULTIPLY", "PRODUCT"):
            res_val = a * b
            expr_str = f"{a:,.2f} * {b:,.2f}"
            lbl = label or "Product"

        elif op in ("DIVIDE", "RATIO"):
            if b == 0:
                res_val = None
                error_msg = "Division by zero"
            else:
                res_val = a / b
            expr_str = f"{a:,.2f} / {b:,.2f}"
            lbl = label or ("Ratio" if op == "RATIO" else "Division")

        elif op in ("MARGIN", "PROFIT_MARGIN"):
            # Profit Margin = (Profit / Sales) * 100
            # Identify if A or B is profit vs sales by label/name
            if "profit" in name_a.lower() and "sale" in name_b.lower():
                num, den = a, b
            elif "profit" in name_b.lower() and "sale" in name_a.lower():
                num, den = b, a
            else:
                num, den = b, a  # default B=profit, A=sales

            if den == 0:
                res_val = None
                error_msg = "Division by zero (denominator is zero)"
            else:
                res_val = (num / den) * 100.0
            expr_str = f"({num:,.2f} / {den:,.2f}) * 100"
            lbl = label or "Profit Margin"

        elif op in ("PERCENTAGE", "PERCENT"):
            # Percentage = (A / B) * 100
            if b == 0:
                res_val = None
                error_msg = "Division by zero"
            else:
                res_val = (a / b) * 100.0
            expr_str = f"({a:,.2f} / {b:,.2f}) * 100"
            lbl = label or "Percentage"

        elif op in ("PERCENTAGE_CHANGE", "GROWTH"):
            # Change = (New - Old) / Old * 100 (where a is new, b is old)
            if b == 0:
                res_val = None
                error_msg = "Division by zero (baseline value is zero)"
            else:
                res_val = ((a - b) / b) * 100.0
            expr_str = f"(({a:,.2f} - {b:,.2f}) / {b:,.2f}) * 100"
            lbl = label or "Percentage Change"

        else:
            res_val = a - b
            expr_str = f"{a:,.2f} - {b:,.2f}"
            lbl = label or "Difference"

        # Validate finiteness
        is_finite = True
        if res_val is None or math.isnan(res_val) or math.isinf(res_val):
            is_finite = False
            res_val = None
            if not error_msg:
                error_msg = "Calculation resulted in an undefined or non-finite value"

        # Format display string
        if res_val is not None:
            if op in ("MARGIN", "PROFIT_MARGIN", "PERCENTAGE", "PERCENT", "PERCENTAGE_CHANGE", "GROWTH"):
                formatted = f"{res_val:,.2f}%"
            elif op == "RATIO":
                formatted = f"{res_val:,.4f}"
            else:
                formatted = f"{res_val:,.2f}"
        else:
            formatted = "N/A"

        return CalculationResult(
            calculated=True,
            operator=op,
            label=lbl,
            expression=expr_str,
            value=round(res_val, 4) if res_val is not None else None,
            formatted_value=formatted,
            operands={
                name_a: a,
                name_b: b,
            },
            is_finite=is_finite,
            error=error_msg,
        )

    @classmethod
    def compute_from_dataframe(
        cls,
        df: pd.DataFrame,
        calc_spec: CalculationSpec,
    ) -> tuple[pd.DataFrame, Optional[CalculationResult]]:
        """
        Validate and compute calculations over a DataFrame (both scalar and grouped).
        """
        if df is None or df.empty or not calc_spec:
            return df, None

        op = calc_spec.operator.upper()
        cols = df.columns.tolist()

        if len(calc_spec.operands) < 2:
            return df, None

        op_a = calc_spec.operands[0]
        op_b = calc_spec.operands[1]

        # Find matching column names for A and B
        meas_a = (op_a.measure or op_a.measure_column or "").lower()
        meas_b = (op_b.measure or op_b.measure_column or "").lower()
        name_a = (op_a.name or "").lower()
        name_b = (op_b.name or "").lower()

        col_a = next((c for c in cols if (name_a and name_a in c.lower()) or (meas_a and meas_a in c.lower())), None)
        col_b = next((c for c in cols if c != col_a and ((name_b and name_b in c.lower()) or (meas_b and meas_b in c.lower()))), None)

        if not col_a or not col_b or col_a == col_b:
            logger.warning(f"Calculation operands could not be resolved from DataFrame columns: {cols}. Required: A={name_a or meas_a}, B={name_b or meas_b}")
            error_calc = CalculationResult(
                calculated=False,
                operator=op,
                label=calc_spec.label or "Calculation Error",
                expression=calc_spec.label or "",
                value=None,
                formatted_value="N/A",
                is_finite=False,
                error=f"Required operand columns ({name_a or meas_a}, {name_b or meas_b}) not found in query results.",
            )
            return df, error_calc

        # If scalar row (1 row):
        if len(df) == 1:
            val_a = df[col_a].iloc[0] if col_a in df else 0.0
            val_b = df[col_b].iloc[0] if col_b in df else 0.0

            calc_res = cls.compute_scalar(
                val_a, val_b, op, calc_spec.label,
                name_a=op_a.measure or col_a,
                name_b=op_b.measure or col_b,
            )
            return df, calc_res

        # If grouped (multiple rows):
        # Identify group dimension column (usually the non-numeric string column)
        group_dim = next((c for c in cols if c not in (col_a, col_b)), None)
        target_col = calc_spec.target_column_name or (calc_spec.label.lower().replace(" ", "_") if calc_spec.label else "calculated_result")

        df_out = df.copy()
        if op in ("SUBTRACT", "DIFFERENCE", "MINUS"):
            df_out[target_col] = (df_out[col_a] - df_out[col_b]).round(2)
        elif op in ("ADD", "SUM", "PLUS"):
            df_out[target_col] = (df_out[col_a] + df_out[col_b]).round(2)
        elif op in ("MULTIPLY", "PRODUCT"):
            df_out[target_col] = (df_out[col_a] * df_out[col_b]).round(2)
        elif op in ("DIVIDE", "RATIO"):
            df_out[target_col] = np.where(df_out[col_b] != 0, (df_out[col_a] / df_out[col_b]).round(4), np.nan)
        elif op in ("MARGIN", "PROFIT_MARGIN"):
            # Profit margin = (Profit / Sales) * 100
            if "profit" in col_a.lower() and "sale" in col_b.lower():
                num_c, den_c = col_a, col_b
            elif "profit" in col_b.lower() and "sale" in col_a.lower():
                num_c, den_c = col_b, col_a
            else:
                num_c, den_c = col_b, col_a
            df_out[target_col] = np.where(df_out[den_c] != 0, ((df_out[num_c] / df_out[den_c]) * 100.0).round(2), np.nan)
        elif op in ("PERCENTAGE", "PERCENT"):
            df_out[target_col] = np.where(df_out[col_b] != 0, ((df_out[col_a] / df_out[col_b]) * 100.0).round(2), np.nan)
        else:
            df_out[target_col] = (df_out[col_a] - df_out[col_b]).round(2)

        # Build group summary list
        grouped_results = []
        for _, row in df_out.iterrows():
            grp_name = str(row[group_dim]) if group_dim else "Group"
            res_val = row[target_col]
            grouped_results.append({
                "group": grp_name,
                "value": float(res_val) if pd.notnull(res_val) else None,
                "formatted_value": f"{float(res_val):,.2f}" if pd.notnull(res_val) else "N/A",
            })

        calc_res = CalculationResult(
            calculated=True,
            operator=op,
            label=calc_spec.label or "Derived Calculation",
            expression=f"{col_a} {op} {col_b}",
            value=None,
            formatted_value=f"Computed for {len(df_out)} groups",
            grouped=True,
            group_dimension=group_dim,
            results_by_group=grouped_results,
            operands={"A": col_a, "B": col_b},
            is_finite=bool(df_out[target_col].notnull().all()),
        )

        return df_out, calc_res
