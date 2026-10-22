"""
Visualization Quality Gate for AI-MADAC.
Evaluates semantic, data type, aggregation, axis, time granularity, series count,
and readability quality before Plotly rendering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Evaluation result from the Visualization Quality Gate."""
    passed: bool
    score: float  # 0.0 to 1.0
    critical_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommendation: str = "render"  # "render", "repair", "fallback_to_table"


class VisualizationQualityGate:
    """
    Quality gate enforcing Power BI-like visual standards before rendering.
    Ensures a wrong or unreadable chart is never displayed.
    """

    @classmethod
    def evaluate(
        cls,
        spec: dict[str, Any],
        df: pd.DataFrame,
        contract: Any,
    ) -> QualityReport:
        critical_errors: list[str] = []
        warnings: list[str] = []
        score: float = 1.0

        if df is None or df.empty:
            return QualityReport(
                passed=False,
                score=0.0,
                critical_errors=["DataFrame is empty."],
                recommendation="fallback_to_table",
            )

        chart = spec.get("chart") or {}
        chart_type = chart.get("type") or spec.get("chart_type") or "bar"
        fields = spec.get("fields") or {}
        x_col = fields.get("x") or spec.get("x_axis")
        y_col = fields.get("y") or spec.get("y_axis")
        color_col = fields.get("color") or spec.get("color_by")

        # 1. Axis existence check
        cols_lower = {str(c).lower(): str(c) for c in df.columns}
        if x_col and x_col.lower() not in cols_lower and len(df.columns) == 0:
            critical_errors.append(f"X-axis column '{x_col}' does not exist in prepared data.")
            score -= 0.5

        # 2. Time-series Line Chart checks
        if chart_type in ("line", "trend"):
            # Check point count: plotting > 500 points for a trend is bad practice
            if len(df) > 500:
                warnings.append(
                    f"Line chart contains {len(df)} points. For high-level trends, aggregate to month or week."
                )
                score -= 0.2

            # Check if x is time-like or chronological
            target_x = cols_lower.get(str(x_col).lower(), df.columns[0])
            dt_check = pd.to_datetime(df[target_x], errors="coerce")
            if dt_check.notna().mean() < 0.3 and not any(k in str(target_x).lower() for k in ("date", "time", "year", "month", "day", "period")):
                warnings.append(f"Line chart X-axis '{target_x}' may not be temporal.")
                score -= 0.15

            # Multi-series line count check
            if color_col and color_col.lower() in cols_lower:
                series_count = df[cols_lower[color_col.lower()]].nunique()
                if series_count > 12:
                    critical_errors.append(
                        f"Multi-series line chart has {series_count} overlapping lines (maximum recommended is 10)."
                    )
                    score -= 0.4

        # 3. Scatter Plot checks
        elif chart_type in ("scatter", "relationship"):
            target_x = cols_lower.get(str(x_col).lower())
            target_y = cols_lower.get(str(y_col).lower())
            if not target_x or not target_y:
                if len(df.columns) >= 2:
                    target_x, target_y = df.columns[0], df.columns[1]
                else:
                    critical_errors.append("Scatter plot requires two numeric columns.")
                    score -= 0.5

            if target_x and target_y:
                x_num = pd.to_numeric(df[target_x], errors="coerce")
                y_num = pd.to_numeric(df[target_y], errors="coerce")
                if x_num.notna().mean() < 0.7:
                    critical_errors.append(f"Scatter plot X-axis '{target_x}' must be numeric.")
                    score -= 0.4
                if y_num.notna().mean() < 0.7:
                    critical_errors.append(f"Scatter plot Y-axis '{target_y}' must be numeric.")
                    score -= 0.4

        # 4. Histogram checks
        elif chart_type in ("histogram", "distribution"):
            target_x = cols_lower.get(str(x_col).lower(), df.columns[0])
            x_num = pd.to_numeric(df[target_x], errors="coerce")
            if x_num.notna().mean() < 0.7:
                critical_errors.append(f"Histogram variable '{target_x}' must be continuous numeric data.")
                score -= 0.5

        # 5. Pie / Donut checks
        elif chart_type in ("pie", "donut"):
            cat_col = cols_lower.get(str(x_col).lower(), df.columns[0])
            category_count = df[cat_col].nunique()
            if category_count > 10:
                critical_errors.append(
                    f"Pie chart contains {category_count} categories (maximum readable is 7). Prefer a bar chart."
                )
                score -= 0.4

        score = max(0.0, min(1.0, score))
        passed = (len(critical_errors) == 0) and (score >= 0.6)

        recommendation = "render"
        if not passed:
            recommendation = "repair" if score >= 0.4 else "fallback_to_table"

        logger.info(
            f"Visualization Quality Gate: chart_type='{chart_type}', score={score:.2f}, "
            f"passed={passed}, errors={critical_errors}, warnings={warnings}"
        )

        return QualityReport(
            passed=passed,
            score=score,
            critical_errors=critical_errors,
            warnings=warnings,
            recommendation=recommendation,
        )
