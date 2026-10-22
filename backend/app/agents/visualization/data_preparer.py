"""
Pandas Data Preparation and Statistical Insight Calculation for AI-MADAC.

Adheres to:
1. RAW DATA RULE: Python/Pandas calculates counts, means, rankings, bins, and metrics.
2. HISTOGRAM RULE: Uses raw observations directly; sum of frequencies == total valid rows.
3. MATHEMATICAL INSIGHTS: Calculates true statistical extremes in Python (never invented).
4. STRICT FILTERING & SORTING: Applies current query filters and ordering in Pandas.
"""

from __future__ import annotations

import re
import logging
from typing import Any, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataPreparer:
    """Prepares tabular data for Plotly rendering and calculates true mathematical insights."""

    @staticmethod
    def parse_range_midpoint(val: Any) -> Optional[float]:
        """Parse range category like '10-20%' or '20-30%' into transparent numeric midpoint."""
        if val is None or pd.isna(val):
            return None
        s = str(val).strip().replace("%", "")
        m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(\d+(?:\.\d+)?)\s*$", s)
        if m:
            return (float(m.group(1)) + float(m.group(2))) / 2.0
        try:
            return float(s)
        except ValueError:
            return None

    @staticmethod
    def apply_filters(df: pd.DataFrame, filters: list[dict[str, Any]]) -> pd.DataFrame:
        """Apply filters specified in the validated specification to the DataFrame."""
        filtered_df = df.copy()
        for f in filters:
            col = f.get("column") or f.get("field")
            op = str(f.get("op") or f.get("operator") or "==").lower()
            val = f.get("value")
            if not col or col not in filtered_df.columns:
                continue

            try:
                # 1. Equals / Equal
                if op in ("==", "=", "equals", "equal"):
                    val_str = str(val).strip().lower()
                    match_mask = filtered_df[col].astype(str).str.strip().str.lower() == val_str
                    # If no direct match and value looks like a range '30-40%'
                    if not match_mask.any() and ("-" in val_str or "%" in val_str):
                        clean_range = val_str.replace("%", "").strip()
                        parts = re.split(r"[-–—to]", clean_range)
                        if len(parts) == 2:
                            pat = rf"{parts[0].strip()}\s*[-–—to]\s*{parts[1].strip()}"
                            match_mask = filtered_df[col].astype(str).str.contains(pat, regex=True, case=False)
                    filtered_df = filtered_df[match_mask]

                # 2. Not equals
                elif op in ("!=", "<>", "not_equals", "not_equal"):
                    val_str = str(val).strip().lower()
                    filtered_df = filtered_df[filtered_df[col].astype(str).str.strip().str.lower() != val_str]

                # 3. Greater than / Greater or equal
                elif op in (">", "greater_than"):
                    filtered_df = filtered_df[pd.to_numeric(filtered_df[col], errors="coerce") > float(val)]
                elif op in (">=", "greater_or_equal", "greater_than_or_equal"):
                    filtered_df = filtered_df[pd.to_numeric(filtered_df[col], errors="coerce") >= float(val)]

                # 4. Less than / Less or equal
                elif op in ("<", "less_than"):
                    filtered_df = filtered_df[pd.to_numeric(filtered_df[col], errors="coerce") < float(val)]
                elif op in ("<=", "less_or_equal", "less_than_or_equal"):
                    filtered_df = filtered_df[pd.to_numeric(filtered_df[col], errors="coerce") <= float(val)]

                # 5. Between
                elif op == "between":
                    if isinstance(val, (tuple, list)) and len(val) == 2:
                        low, high = float(val[0]), float(val[1])
                    elif isinstance(val, str) and "-" in val:
                        parts = re.split(r"[-–—to]", val.replace("%", "").strip())
                        low, high = float(parts[0]), float(parts[1])
                    else:
                        continue

                    num_series = pd.to_numeric(filtered_df[col], errors="coerce")
                    if num_series.notna().sum() > 0:
                        filtered_df = filtered_df[(num_series >= low) & (num_series <= high)]
                    else:
                        pat = rf"{int(low)}\s*[-–—to]\s*{int(high)}"
                        filtered_df = filtered_df[filtered_df[col].astype(str).str.contains(pat, regex=True, case=False)]

                # 6. Contains
                elif op == "contains":
                    filtered_df = filtered_df[filtered_df[col].astype(str).str.contains(str(val), case=False, na=False)]

                # 7. In
                elif op in ("in", "is_in"):
                    val_list = [str(x).strip().lower() for x in (val if isinstance(val, (list, tuple, set)) else [val])]
                    filtered_df = filtered_df[filtered_df[col].astype(str).str.strip().str.lower().isin(val_list)]

            except Exception:
                continue
        return filtered_df

    @classmethod
    def prepare(
        cls,
        df: pd.DataFrame,
        spec: dict[str, Any],
    ) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
        """
        Prepare DataFrame according to the validated specification.
        Returns: (prepared_df, applied_parameters, key_insights)
        """
        fields = spec.get("fields") or {}
        x_col = fields.get("x")
        y_col = fields.get("y")
        color_col = fields.get("color")
        chart = spec.get("chart") or {}
        chart_type = chart.get("type", "bar")
        measure = spec.get("measure") or {}
        measure_type = measure.get("type", "count")
        filters = spec.get("filters") or []
        sorting = spec.get("sorting") or {}
        limit = spec.get("limit")
        binning = spec.get("binning") or spec.get("bins") or {}

        # 1. Apply Filters strictly before any aggregation
        work_df = cls.apply_filters(df, filters)
        total_filtered_rows = len(work_df)

        insights: list[str] = []
        parameters_used: list[str] = []

        if filters:
            for f in filters:
                col = f.get("column") or f.get("field")
                op = f.get("op") or f.get("operator")
                val = f.get("value")
                parameters_used.append(f"Filter: {col} {op} {val}")
            parameters_used.append(f"Filtered Records: {total_filtered_rows} rows")

        # 2. Check for empty filtered dataframe
        if work_df.empty:
            return work_df, {"chart_type": chart_type, "filters": filters}, ["No records match the requested filter criteria."]

        # Case A: Cross-Tabulation / Grouped Bar
        if color_col and color_col in work_df.columns and x_col and x_col in work_df.columns:
            prepared_df = (
                work_df.groupby([x_col, color_col])
                .size()
                .reset_index(name="count")
            )
            if sorting.get("direction") == "desc":
                prepared_df = prepared_df.sort_values(by="count", ascending=False)
            elif sorting.get("direction") == "asc":
                prepared_df = prepared_df.sort_values(by="count", ascending=True)

            parameters_used.append(f"X-Axis: {x_col}")
            parameters_used.append(f"Group/Split: {color_col}")
            parameters_used.append("Aggregation: Count")

            if not prepared_df.empty:
                top_row = prepared_df.loc[prepared_df["count"].idxmax()]
                total_investors = prepared_df["count"].sum()
                insights.append(
                    f"Highest segment: {top_row[color_col]} with objective '{top_row[x_col]}' ({top_row['count']} investors, {round(100 * top_row['count'] / max(total_investors, 1), 1)}% of total)."
                )
                low_row = prepared_df.loc[prepared_df["count"].idxmin()]
                if low_row[x_col] != top_row[x_col] or low_row[color_col] != top_row[color_col]:
                    insights.append(
                        f"Lowest segment: {low_row[color_col]} with objective '{low_row[x_col]}' ({low_row['count']} investors)."
                    )

            applied_params = {
                "chart_type": "Grouped Bar",
                "x_axis": x_col,
                "split_by": color_col,
                "aggregation": "Count",
                "filters": [f"{f.get('column') or f.get('field')} {f.get('op') or f.get('operator')} {f.get('value')}" for f in filters],
                "sorting": sorting.get("direction"),
                "limit": limit,
            }
            return prepared_df, applied_params, insights

        # Case B: Histogram / Distribution
        if chart_type == "histogram":
            # Verify data type: Categorical data must NOT be histogrammed!
            num_check = pd.to_numeric(work_df[x_col], errors="coerce")
            if num_check.notna().mean() < 0.8:
                # Column is categorical: route to Bar chart of counts
                prepared_df = work_df[x_col].value_counts().reset_index()
                prepared_df.columns = [x_col, "investor_count"]
                parameters_used.append(f"Category: {x_col}")
                parameters_used.append("Note: Categorical data represented via Bar chart frequency")
                applied_params = {
                    "chart_type": "Bar chart (Categorical Distribution)",
                    "x_axis": x_col,
                    "y_axis": "investor_count",
                    "observation_count": total_filtered_rows,
                }
                insights.append(f"Distribution of categorical field '{x_col}' across {total_filtered_rows} observations.")
                return prepared_df, applied_params, insights

            # HISTOGRAM RULE: Keep raw observations
            raw_obs = num_check.dropna()
            total_obs = len(raw_obs)
            bin_size = binning.get("bin_size") or binning.get("size")

            # Bin edge calculation and validation
            if bin_size:
                min_edge = float(np.floor(raw_obs.min() / bin_size) * bin_size)
                max_edge = float(np.ceil(raw_obs.max() / bin_size) * bin_size)
                edges = np.arange(min_edge, max_edge + bin_size, bin_size)
                counts, _ = np.histogram(raw_obs, bins=edges)
                # BIN VALIDATION: adjacent edges differ by bin_size and counts sum to total_obs
                edge_diffs = np.diff(edges)
                assert np.allclose(edge_diffs, bin_size), "Adjacent bin edges must differ by bin_size."
                assert counts.sum() == total_obs, f"Histogram count sum ({counts.sum()}) must equal total observations ({total_obs})."
                parameters_used.append(f"Bin Size: {bin_size}")
            else:
                counts, edges = np.histogram(raw_obs, bins="auto")
                assert counts.sum() == total_obs, f"Histogram count sum ({counts.sum()}) must equal total observations ({total_obs})."

            prepared_df = pd.DataFrame({x_col: raw_obs})
            parameters_used.append(f"Variable: {x_col}")
            parameters_used.append(f"Raw Observations: {total_obs}")

            if total_obs > 0:
                mean_val = float(np.mean(raw_obs))
                median_val = float(np.median(raw_obs))
                min_val = float(np.min(raw_obs))
                max_val = float(np.max(raw_obs))
                insights.append(
                    f"Distribution of {x_col} across {total_obs} valid observations spans from {min_val:g} to {max_val:g} with a mean of {mean_val:.1f} and median of {median_val:g}."
                )

            applied_params = {
                "chart_type": "Histogram",
                "x_axis": x_col,
                "observation_count": total_obs,
                "bin_size": bin_size,
                "filters": [f"{f.get('column') or f.get('field')} {f.get('op') or f.get('operator')} {f.get('value')}" for f in filters],
            }
            return prepared_df, applied_params, insights

        # Case C: Box Plot (Numerical vs Categorical / Range)
        if chart_type in ("box", "box_plot") and x_col and y_col and x_col in work_df.columns and y_col in work_df.columns:
            prepared_df = work_df[[x_col, y_col]].dropna()
            total_obs = len(prepared_df)
            parameters_used.append(f"Category Group: {x_col}")
            parameters_used.append(f"Numerical Metric: {y_col}")
            parameters_used.append(f"Observations: {total_obs}")

            groups = prepared_df.groupby(x_col)[y_col]
            medians = groups.median().sort_values(ascending=False)
            if not medians.empty:
                top_group = medians.index[0]
                insights.append(f"Highest median {y_col} is in category '{top_group}' ({medians.iloc[0]:g}).")

            applied_params = {
                "chart_type": "Box Plot",
                "x_axis": x_col,
                "y_axis": y_col,
                "observation_count": total_obs,
                "filters": [f"{f.get('column') or f.get('field')} {f.get('op') or f.get('operator')} {f.get('value')}" for f in filters],
            }
            return prepared_df, applied_params, insights

        # Case D: Scatter / Relationship
        if chart_type == "scatter" and x_col and y_col and x_col in work_df.columns and y_col in work_df.columns:
            plot_df = work_df.copy()
            actual_y = y_col
            # Check if y_col is categorical range (e.g. '10-20%', '20-30%', '30-40%')
            y_is_range = (
                plot_df[y_col].astype(str).str.contains(r"\d+\s*[-–—to]\s*\d+").mean() >= 0.6
                or "expect" in y_col.lower()
                or "return" in y_col.lower()
            ) and pd.to_numeric(plot_df[y_col], errors="coerce").isna().sum() > 0

            if y_is_range:
                midpoint_col = "Expected Return Midpoint (%)"
                plot_df[midpoint_col] = plot_df[y_col].apply(cls.parse_range_midpoint)
                actual_y = midpoint_col
                parameters_used.append(
                    "Derived Numerical Field: 'Expected Return Midpoint (%)' calculated transparently from range categories (10-20% → 15%, 20-30% → 25%, 30-40% → 35%)"
                )

            # Check if x_col is categorical range
            actual_x = x_col
            x_is_range = (
                plot_df[x_col].astype(str).str.contains(r"\d+\s*[-–—to]\s*\d+").mean() >= 0.6
            ) and pd.to_numeric(plot_df[x_col], errors="coerce").isna().sum() > 0

            if x_is_range:
                midpoint_x = f"{x_col} Midpoint"
                plot_df[midpoint_x] = plot_df[x_col].apply(cls.parse_range_midpoint)
                actual_x = midpoint_x
                parameters_used.append(f"Derived Numerical Field: '{midpoint_x}' from categorical ranges")

            prepared_df = plot_df[[actual_x, actual_y]].dropna()
            total_obs = len(prepared_df)
            parameters_used.append(f"X-Axis: {actual_x}")
            parameters_used.append(f"Y-Axis: {actual_y}")
            parameters_used.append(f"Observations: {total_obs}")

            if total_obs > 1:
                try:
                    corr = float(prepared_df[actual_x].corr(prepared_df[actual_y]))
                    direction = "positive" if corr > 0.1 else ("negative" if corr < -0.1 else "neutral / weak")
                    insights.append(
                        f"Correlation between {actual_x} and {actual_y} is {corr:.2f} ({direction} relationship across {total_obs} observations)."
                    )
                except Exception:
                    insights.append(f"Plotted {total_obs} observations comparing {actual_x} and {actual_y}.")

            applied_params = {
                "chart_type": "Scatter Plot",
                "x_axis": actual_x,
                "y_axis": actual_y,
                "observation_count": total_obs,
                "filters": [f"{f.get('column') or f.get('field')} {f.get('op') or f.get('operator')} {f.get('value')}" for f in filters],
            }
            return prepared_df, applied_params, insights

        # Case E: Line Chart / Time Series
        if (chart_type in ("line", "trend") or "time" in chart_type) and x_col and x_col in work_df.columns:
            other_cols = [c for c in work_df.columns if c != x_col]
            val_col = y_col if (y_col and y_col in work_df.columns) else (other_cols[0] if other_cols else x_col)

            if val_col in work_df.columns and len(work_df.columns) >= 2:
                prepared_df = work_df[[x_col, val_col]].dropna().copy()
            else:
                prepared_df = work_df.groupby(x_col).size().reset_index(name="count")
                val_col = "count"

            # Chronological ascending sort
            try:
                sort_key = pd.to_datetime(prepared_df[x_col], errors="coerce")
                if sort_key.notna().mean() >= 0.5:
                    prepared_df["_sort_key"] = sort_key
                    prepared_df = prepared_df.sort_values(by="_sort_key", ascending=True).drop(columns=["_sort_key"])
                else:
                    prepared_df = prepared_df.sort_values(by=x_col, ascending=True)
            except Exception:
                prepared_df = prepared_df.sort_values(by=x_col, ascending=True)

            parameters_used.append(f"Time Dimension (X-Axis): {x_col}")
            parameters_used.append(f"Metric (Y-Axis): {val_col}")
            parameters_used.append("Sort: Chronological Ascending")

            if not prepared_df.empty:
                first_pt = prepared_df.iloc[0]
                last_pt = prepared_df.iloc[-1]
                v_first = first_pt[val_col]
                v_last = last_pt[val_col]
                vf_str = f"{v_first:,.2f}" if isinstance(v_first, (int, float)) else str(v_first)
                vl_str = f"{v_last:,.2f}" if isinstance(v_last, (int, float)) else str(v_last)
                insights.append(
                    f"{val_col.replace('_', ' ').title()} trajectory from {first_pt[x_col]} ({vf_str}) to {last_pt[x_col]} ({vl_str}) across {len(prepared_df)} time points."
                )

            applied_params = {
                "chart_type": "Line Chart (Time Series)",
                "x_axis": x_col,
                "y_axis": val_col,
                "data_points": len(prepared_df),
                "sorting": "Chronological Ascending",
                "filters": [f"{f.get('column') or f.get('field')} {f.get('op') or f.get('operator')} {f.get('value')}" for f in filters],
            }
            return prepared_df, applied_params, insights

        # Case F: Categorical Aggregation (Bar or Pie)
        if x_col and x_col in work_df.columns:
            other_cols = [c for c in work_df.columns if c != x_col]
            is_pre_aggregated = (
                len(other_cols) >= 1
                and any("count" in c.lower() or "percentage" in c.lower() or "total" in c.lower() or "average" in c.lower() for c in other_cols)
            )

            # Check if measure is average on a range category column (e.g. average expected return by avenue)
            if y_col and y_col in work_df.columns and measure_type in ("mean", "avg", "average"):
                y_is_range = (
                    work_df[y_col].astype(str).str.contains(r"\d+\s*[-–—to]\s*\d+").mean() >= 0.6
                    or "expect" in y_col.lower()
                    or "return" in y_col.lower()
                ) and pd.to_numeric(work_df[y_col], errors="coerce").isna().sum() > 0

                if y_is_range:
                    temp_mid_col = "midpoint_val"
                    work_df = work_df.copy()
                    work_df[temp_mid_col] = work_df[y_col].apply(cls.parse_range_midpoint)
                    metric_name = "average_expected_return_pct"
                    prepared_df = (
                        work_df.groupby(x_col)[temp_mid_col]
                        .mean()
                        .reset_index(name=metric_name)
                    )
                    val_col = metric_name
                    agg_label = "Calculated MEAN from Expected Return Midpoints"
                    parameters_used.append(f"X-Axis: {x_col}")
                    parameters_used.append("Metric: Average Expected Return Midpoint (%)")
                    parameters_used.append("Derived Field: 'Expected Return Midpoint' (10-20% → 15%, 20-30% → 25%, 30-40% → 35%)")
                else:
                    agg_fn = "mean"
                    metric_name = f"{agg_fn}_{y_col}"
                    prepared_df = (
                        work_df.groupby(x_col)[y_col]
                        .agg(agg_fn)
                        .reset_index(name=metric_name)
                    )
                    val_col = metric_name
                    agg_label = f"Calculated {agg_fn.upper()}"
                    parameters_used.append(f"X-Axis: {x_col}")
                    parameters_used.append(f"Metric: {agg_fn.upper()}({y_col})")

            elif is_pre_aggregated:
                val_col = other_cols[0]
                prepared_df = work_df[[x_col, val_col]].copy()
                agg_label = "Already aggregated by source agent"
                parameters_used.append(f"Category: {x_col}")
                parameters_used.append(f"Value: {val_col}")
                parameters_used.append(f"Aggregation: {agg_label}")
            elif y_col and y_col in work_df.columns and measure_type in ("sum", "min", "max"):
                agg_fn = measure_type
                metric_name = f"{agg_fn}_{y_col}"
                prepared_df = (
                    work_df.groupby(x_col)[y_col]
                    .agg(agg_fn)
                    .reset_index(name=metric_name)
                )
                val_col = metric_name
                agg_label = f"Calculated {agg_fn.upper()}"
                parameters_used.append(f"X-Axis: {x_col}")
                parameters_used.append(f"Metric: {agg_fn.upper()}({y_col})")
            else:
                prepared_df = (
                    work_df.groupby(x_col)
                    .size()
                    .reset_index(name="investor_count")
                )
                val_col = "investor_count"
                agg_label = "Count"
                parameters_used.append(f"Category: {x_col}")
                parameters_used.append("Aggregation: Count")

            # Sorting
            sort_dir = sorting.get("direction")
            if sort_dir == "desc" or not sort_dir:
                prepared_df = prepared_df.sort_values(by=val_col, ascending=False)
            elif sort_dir == "asc":
                prepared_df = prepared_df.sort_values(by=val_col, ascending=True)

            # Limit (Top-N)
            if limit and isinstance(limit, int) and limit > 0:
                prepared_df = prepared_df.head(limit)
                parameters_used.append(f"Limit: Top {limit}")

            if not prepared_df.empty:
                top_row = prepared_df.iloc[0]
                if val_col == "investor_count" and total_filtered_rows > 0:
                    pct = round(100.0 * top_row[val_col] / total_filtered_rows, 1)
                    filter_clause = f" among the {total_filtered_rows} filtered records" if filters else ""
                    insights.append(
                        f"Top category is '{top_row[x_col]}' with {top_row[val_col]} investors ({pct}% of total{filter_clause})."
                    )
                else:
                    val_display = f"{top_row[val_col]:.1f}%" if "pct" in val_col or "return" in val_col else str(top_row[val_col])
                    insights.append(
                        f"Top category is '{top_row[x_col]}' with {val_display}."
                    )
                if len(prepared_df) > 1:
                    low_row = prepared_df.iloc[-1]
                    low_display = f"{low_row[val_col]:.1f}%" if "pct" in val_col or "return" in val_col else str(low_row[val_col])
                    insights.append(
                        f"Lowest category is '{low_row[x_col]}' with {low_display}."
                    )

            applied_params = {
                "chart_type": "Pie chart" if chart_type == "pie" else "Bar chart",
                "category": x_col,
                "value": val_col,
                "x_axis": x_col,
                "y_axis": val_col,
                "aggregation": agg_label,
                "sorting": "Descending" if (sort_dir == "desc" or not sort_dir) else "Ascending",
                "limit": limit,
            }
            if filters:
                filter_strs = [
                    f"{f.get('column') or f.get('field')} = '{f.get('value')}'" if f.get('op') in ('=', '==') else f"{f.get('column') or f.get('field')} {f.get('op')} {f.get('value')}"
                    for f in filters
                ]
                applied_params["filters"] = ", ".join(filter_strs)

            return prepared_df, applied_params, insights

        # Fallback raw data
        return work_df, {"chart_type": "Raw Preview"}, ["Data displayed without aggregation."]
