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


def ensure_1d(val: Any, col_name: str = "") -> Optional[pd.Series]:
    """
    Universally guarantees that axis data is a valid 1-D pandas Series.
    Rejects or squeezes 2-D DataFrames/arrays, flattens series containing arrays,
    and returns a clean pd.Series with a reset index.
    """
    if val is None:
        return None

    # Handle DataFrame
    if isinstance(val, pd.DataFrame):
        if val.empty:
            return pd.Series(dtype=object)
        if val.shape[1] == 1:
            val = val.iloc[:, 0]
        else:
            # If multiple columns have the exact same name (duplicate columns in DataFrame), take first
            if len(set(val.columns)) == 1:
                val = val.iloc[:, 0]
            else:
                raise ValueError(
                    f"Expected 1-D series for '{col_name}', but received DataFrame with {val.shape[1]} columns: {list(val.columns)}"
                )

    # Handle numpy ndarray
    if isinstance(val, np.ndarray):
        squeezed = np.asarray(val).squeeze()
        if squeezed.ndim == 0:
            squeezed = np.array([squeezed])
        elif squeezed.ndim != 1:
            raise ValueError(
                f"Expected 1-D array for '{col_name}', but received array with shape {val.shape}"
            )
        return pd.Series(squeezed).reset_index(drop=True)

    # Handle list or tuple
    if isinstance(val, (list, tuple)):
        if val and isinstance(val[0], (list, tuple, np.ndarray)):
            flat = [item[0] if len(item) == 1 else item for item in val]
            if any(isinstance(x, (list, tuple, np.ndarray)) for x in flat):
                raise ValueError(f"Expected 1-D sequence for '{col_name}', received nested sequence")
            return pd.Series(flat).reset_index(drop=True)
        return pd.Series(val).reset_index(drop=True)

    # Handle pandas Series
    if isinstance(val, pd.Series):
        if not val.empty and isinstance(val.iloc[0], (np.ndarray, list, tuple)):
            if all(len(x) == 1 for x in val.dropna()):
                val = val.apply(lambda x: x[0] if isinstance(x, (np.ndarray, list, tuple)) and len(x) == 1 else x)
            else:
                raise ValueError(f"Series for '{col_name}' contains multi-element array entries")
        return val.reset_index(drop=True)

    # Scalar / single element
    return pd.Series([val])


class DataPreparer:
    """Prepares tabular data for Plotly rendering and calculates true mathematical insights."""

    ensure_1d = staticmethod(ensure_1d)

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
        if color_col and color_col in work_df.columns and x_col and x_col in work_df.columns and chart_type not in ("line", "multi_line", "multiline", "trend", "scatter", "relationship"):
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
        if chart_type in ("histogram", "distribution"):
            # Check if range category (e.g. Expect '10-20%', '20-30%')
            is_range_cat = (
                work_df[x_col].astype(str).str.contains(r"\d+\s*[-–—to]\s*\d+").mean() >= 0.6
                or "expect" in str(x_col).lower()
                or "return" in str(x_col).lower()
            ) and pd.to_numeric(work_df[x_col], errors="coerce").isna().sum() > 0

            num_check = pd.to_numeric(work_df[x_col], errors="coerce")
            if is_range_cat:
                raw_obs = work_df[x_col].apply(cls.parse_range_midpoint).dropna()
                parameters_used.append(f"Derived Numerical Midpoints for '{x_col}' (e.g. 10-20% → 15%, 20-30% → 25%)")
            elif num_check.notna().mean() >= 0.7:
                raw_obs = num_check.dropna()
            else:
                # Column is purely categorical: route to Bar chart of counts
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

            # HISTOGRAM RULE: Keep raw 1-D observations
            raw_obs = ensure_1d(raw_obs, str(x_col)).dropna()
            total_obs = len(raw_obs)
            bin_size = binning.get("bin_size") or binning.get("size")

            if total_obs > 0 and not bin_size:
                # Dynamic bin calculation
                min_v = float(raw_obs.min())
                max_v = float(raw_obs.max())
                val_range = max_v - min_v
                n_uniq = raw_obs.nunique()
                if val_range == 0 or n_uniq <= 5:
                    bin_size = 1.0
                else:
                    iqr = float(raw_obs.quantile(0.75) - raw_obs.quantile(0.25))
                    h = (2.0 * iqr / (total_obs ** (1.0 / 3.0))) if iqr > 0 else (val_range / max(1.0 + np.log2(total_obs), 1.0))
                    candidates = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
                    bin_size = min(candidates, key=lambda c: abs(c - h))

            if bin_size:
                parameters_used.append(f"Bin Size: {bin_size}")

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
        if chart_type in ("scatter", "relationship") and x_col and y_col and x_col in work_df.columns and y_col in work_df.columns:
            # 1-D normalizer ensures strictly 1D series
            s_x = ensure_1d(work_df[x_col], str(x_col))
            s_y = ensure_1d(work_df[y_col], str(y_col))

            actual_y = y_col
            # Check if y_col is categorical range (e.g. '10-20%', '20-30%', '30-40%')
            y_is_range = (
                s_y.astype(str).str.contains(r"\d+\s*[-–—to]\s*\d+").mean() >= 0.6
                or "expect" in str(y_col).lower()
                or "return" in str(y_col).lower()
            ) and pd.to_numeric(s_y, errors="coerce").isna().sum() > 0

            if y_is_range:
                actual_y = "Expected Return Midpoint (%)"
                s_y = s_y.apply(cls.parse_range_midpoint)
                parameters_used.append(
                    "Derived Numerical Field: 'Expected Return Midpoint (%)' calculated transparently from range categories (10-20% → 15%, 20-30% → 25%, 30-40% → 35%)"
                )

            # Check if x_col is categorical range
            actual_x = x_col
            x_is_range = (
                s_x.astype(str).str.contains(r"\d+\s*[-–—to]\s*\d+").mean() >= 0.6
            ) and pd.to_numeric(s_x, errors="coerce").isna().sum() > 0

            if x_is_range:
                actual_x = f"{x_col} Midpoint"
                s_x = s_x.apply(cls.parse_range_midpoint)
                parameters_used.append(f"Derived Numerical Field: '{actual_x}' from categorical ranges")

            # Numeric coercion for X and Y
            s_x = pd.to_numeric(s_x, errors="coerce")
            s_y = pd.to_numeric(s_y, errors="coerce")

            df_dict = {actual_x: s_x, actual_y: s_y}
            if color_col and color_col in work_df.columns and color_col not in (x_col, y_col):
                df_dict[color_col] = ensure_1d(work_df[color_col], str(color_col))

            clean_df = pd.DataFrame(df_dict).dropna(subset=[actual_x, actual_y])
            total_obs = len(clean_df)

            # Strict 1-D & equal length validation
            assert ensure_1d(clean_df[actual_x]).ndim == 1, "Scatter X axis must be 1-D Series"
            assert ensure_1d(clean_df[actual_y]).ndim == 1, "Scatter Y axis must be 1-D Series"
            assert len(clean_df[actual_x]) == len(clean_df[actual_y]), "Scatter X and Y must have equal length"

            parameters_used.append(f"X-Axis: {actual_x}")
            parameters_used.append(f"Y-Axis: {actual_y}")
            if color_col and color_col in clean_df.columns:
                parameters_used.append(f"Color Series: {color_col}")
            parameters_used.append(f"Observations: {total_obs}")

            if total_obs > 1:
                try:
                    corr = float(clean_df[actual_x].corr(clean_df[actual_y]))
                    direction = "positive" if corr > 0.1 else ("negative" if corr < -0.1 else "neutral / weak")
                    insights.append(
                        f"Correlation between {actual_x} and {actual_y} is {corr:.2f} ({direction} relationship across {total_obs} observations)."
                    )
                except Exception:
                    insights.append(f"Plotted {total_obs} observations comparing {actual_x} and {actual_y}.")

            # Deterministic visualization sampling for very large datasets (> 5000 rows)
            if total_obs > 5000:
                prepared_df = clean_df.sample(n=5000, random_state=42).reset_index(drop=True)
                parameters_used.append(f"Rendering: Deterministic visual sample of 5,000 points from {total_obs} total rows")
            else:
                prepared_df = clean_df

            applied_params = {
                "chart_type": "Scatter Plot",
                "category": actual_x,
                "value": actual_y,
                "x_axis": actual_x,
                "y_axis": actual_y,
                "series": color_col if (color_col and color_col in prepared_df.columns) else None,
                "observation_count": total_obs,
                "sorting": f"{actual_x} ascending",
                "filters": ", ".join([f"{f.get('column') or f.get('field')} {f.get('op') or f.get('operator')} {f.get('value')}" for f in filters]) if filters else "None",
            }
            return prepared_df, applied_params, insights

        # Case E: Line Chart / Multi-Series Line (Numeric or Chronological Sort)
        if (chart_type in ("line", "trend", "multi_line") or "time" in str(chart_type).lower()) and x_col and x_col in work_df.columns:
            other_cols = [c for c in work_df.columns if c not in (x_col, color_col)]
            val_col = y_col if (y_col and y_col in work_df.columns and y_col != x_col and y_col != color_col) else (other_cols[0] if other_cols else x_col)
            
            meas_raw = str(measure_type or "avg").lower()
            if meas_raw in ("mean", "average", "avg"):
                meas_type = "avg"
                agg_fn = "mean"
                agg_display = "AVG"
            elif meas_raw in ("sum", "total"):
                meas_type = "sum"
                agg_fn = "sum"
                agg_display = "SUM"
            elif meas_raw in ("max", "maximum", "highest"):
                meas_type = "max"
                agg_fn = "max"
                agg_display = "MAX"
            elif meas_raw in ("min", "minimum", "lowest"):
                meas_type = "min"
                agg_fn = "min"
                agg_display = "MIN"
            elif meas_raw in ("count", "distinct"):
                meas_type = "count"
                agg_fn = "count"
                agg_display = "COUNT"
            else:
                meas_type = "avg"
                agg_fn = "mean"
                agg_display = "AVG"

            plot_df = work_df.copy()

            # 1. Determine if X is numeric (e.g. Age, Tenor, Year number) or Datetime
            x_num = pd.to_numeric(plot_df[x_col], errors="coerce")
            x_is_numeric = (
                pd.api.types.is_numeric_dtype(plot_df[x_col])
                or (x_num.notna().mean() >= 0.85 and not any(k in str(x_col).lower() for k in ("date", "time", "invoicedate")))
            )

            is_valid_datetime = False
            time_grain = spec.get("time_granularity") or "month"

            if x_is_numeric:
                plot_df["_x_clean"] = x_num
                plot_df = plot_df.dropna(subset=["_x_clean"])
            else:
                # Check for datetime
                dt_series = pd.to_datetime(plot_df[x_col], errors="coerce")
                if dt_series.notna().mean() >= 0.5:
                    is_valid_datetime = True
                    plot_df["_dt"] = dt_series
                    plot_df = plot_df.dropna(subset=["_dt"])
                    time_grain = str(time_grain).lower()
                    if time_grain == "year":
                        plot_df["_x_clean"] = plot_df["_dt"].dt.strftime("%Y")
                    elif time_grain == "quarter":
                        plot_df["_x_clean"] = plot_df["_dt"].dt.to_period("Q").astype(str)
                    elif time_grain == "week":
                        plot_df["_x_clean"] = plot_df["_dt"].dt.strftime("%Y-W%W")
                    elif time_grain == "day":
                        plot_df["_x_clean"] = plot_df["_dt"].dt.strftime("%Y-%m-%d")
                    else:
                        plot_df["_x_clean"] = plot_df["_dt"].dt.strftime("%Y-%m")
                else:
                    plot_df["_x_clean"] = plot_df[x_col].astype(str)

            sort_display = "Chronological Ascending" if is_valid_datetime else f"{x_col} ascending"

            # 2. Determine and parse value column (val_col)
            # Check if val_col is a range category (e.g. Expect '10-20%', '20-30%')
            val_is_range = (
                plot_df[val_col].astype(str).str.contains(r"\d+\s*[-–—to]\s*\d+").mean() >= 0.6
                or "expect" in str(val_col).lower()
                or "return" in str(val_col).lower()
            ) and pd.to_numeric(plot_df[val_col], errors="coerce").isna().sum() > 0

            if val_is_range:
                plot_df["_val_clean"] = plot_df[val_col].apply(cls.parse_range_midpoint)
                parameters_used.append("Calculated numeric midpoints for range categories (e.g. 10-20% → 15%, 20-30% → 25%)")
            else:
                plot_df["_val_clean"] = pd.to_numeric(plot_df[val_col], errors="coerce")

            plot_df = plot_df.dropna(subset=["_val_clean"])

            # 3. Multi-series vs Single-line
            has_multi_series = False
            if color_col and color_col in plot_df.columns and color_col not in (x_col, val_col):
                if not pd.api.types.is_numeric_dtype(plot_df[color_col]):
                    n_uniq = plot_df[color_col].nunique()
                    if 1 < n_uniq <= 12:
                        has_multi_series = True

            filter_str = ", ".join([f"{f.get('column') or f.get('field')} {f.get('op') or f.get('operator')} {f.get('value')}" for f in filters]) if filters else "None"

            if has_multi_series:
                # Group by [X, series_dim] and aggregate
                prepared_df = (
                    plot_df.groupby(["_x_clean", color_col])["_val_clean"]
                    .agg(agg_fn)
                    .reset_index(name=val_col)
                )
                prepared_df = prepared_df.rename(columns={"_x_clean": x_col})
                
                # Sort numerically if X is numeric, chronologically if datetime, else standard
                if x_is_numeric:
                    prepared_df[x_col] = pd.to_numeric(prepared_df[x_col], errors="coerce")
                    prepared_df = prepared_df.sort_values(by=[x_col, color_col], ascending=[True, True])
                else:
                    prepared_df = prepared_df.sort_values(by=[x_col, color_col], ascending=[True, True])

                # Validate: exactly 1-D series for all Plotly axes
                assert ensure_1d(prepared_df[x_col]).ndim == 1, "X axis must be 1-D Series"
                assert ensure_1d(prepared_df[val_col]).ndim == 1, "Y axis must be 1-D Series"
                assert ensure_1d(prepared_df[color_col]).ndim == 1, "Color series must be 1-D Series"

                series_vals = prepared_df[color_col].unique()
                parameters_used.append(f"X-Axis: {x_col}")
                parameters_used.append(f"Metric (Y-Axis): {val_col}")
                parameters_used.append(f"Aggregation: {agg_display}")
                parameters_used.append(f"Series (Lines): {color_col} ({len(series_vals)} lines: {', '.join(str(s) for s in series_vals[:5])})")
                parameters_used.append(f"Sort: {sort_display}")

                applied_params = {
                    "chart_type": "Multi-Line",
                    "category": x_col,
                    "x_axis": x_col,
                    "value": val_col,
                    "y_axis": val_col,
                    "aggregation": agg_display,
                    "series": color_col,
                    "sorting": sort_display,
                    "filters": filter_str,
                }
                insights.append(
                    f"Multi-line trajectory of {val_col.replace('_', ' ').title()} across {x_col} grouped by {color_col}."
                )
                return prepared_df, applied_params, insights

            # Single line chart
            prepared_df = (
                plot_df.groupby("_x_clean")["_val_clean"]
                .agg(agg_fn)
                .reset_index(name=val_col)
            )
            prepared_df = prepared_df.rename(columns={"_x_clean": x_col})
            if x_is_numeric:
                prepared_df[x_col] = pd.to_numeric(prepared_df[x_col], errors="coerce")
                prepared_df = prepared_df.sort_values(by=x_col, ascending=True)
            else:
                prepared_df = prepared_df.sort_values(by=x_col, ascending=True)

            # Validate: strictly 1-D
            assert ensure_1d(prepared_df[x_col]).ndim == 1, "X axis must be 1-D Series"
            assert ensure_1d(prepared_df[val_col]).ndim == 1, "Y axis must be 1-D Series"

            parameters_used.append(f"X-Axis: {x_col}")
            parameters_used.append(f"Metric (Y-Axis): {val_col}")
            parameters_used.append(f"Aggregation: {agg_display}")
            parameters_used.append(f"Sort: {sort_display}")

            if not prepared_df.empty:
                first_pt = prepared_df.iloc[0]
                last_pt = prepared_df.iloc[-1]
                v_first = first_pt[val_col]
                v_last = last_pt[val_col]
                vf_str = f"{v_first:,.2f}" if isinstance(v_first, (int, float)) else str(v_first)
                vl_str = f"{v_last:,.2f}" if isinstance(v_last, (int, float)) else str(v_last)
                insights.append(
                    f"{val_col.replace('_', ' ').title()} trajectory from {first_pt[x_col]} ({vf_str}) to {last_pt[x_col]} ({vl_str}) across {len(prepared_df)} observations."
                )

            applied_params = {
                "chart_type": "Line Chart",
                "category": x_col,
                "x_axis": x_col,
                "value": val_col,
                "y_axis": val_col,
                "aggregation": agg_display,
                "sorting": sort_display,
                "filters": filter_str,
            }
            return prepared_df, applied_params, insights

        # Case F: Clustered Multi-Measure Bar (e.g. Sales and Profit by Category)
        if isinstance(y_col, list) and x_col and x_col in work_df.columns:
            matched_measures = []
            for m in y_col:
                if not isinstance(m, str):
                    continue
                if m in work_df.columns:
                    matched_measures.append(m)
                else:
                    alias = next((c for c in work_df.columns if c.lower() == f"total_{m.lower()}" or c.lower() == m.lower() or m.lower() in c.lower()), None)
                    if alias:
                        matched_measures.append(alias)
            if len(matched_measures) >= 2:
                valid_measures = matched_measures
                plot_df = work_df.copy()
                for m in valid_measures:
                    plot_df[m] = pd.to_numeric(plot_df[m], errors="coerce")
                prepared_df = plot_df.groupby(x_col)[valid_measures].sum().reset_index()

                parameters_used.append(f"Category (X-Axis): {x_col}")
                parameters_used.append(f"Measures: {', '.join(valid_measures)}")
                parameters_used.append("Aggregation: SUM")

                applied_params = {
                    "chart_type": "Clustered Bar Chart",
                    "x_axis": x_col,
                    "category": x_col,
                    "measures": valid_measures,
                    "aggregation": "SUM",
                    "filters": [f"{f.get('column') or f.get('field')} {f.get('op') or f.get('operator')} {f.get('value')}" for f in filters],
                }
                insights.append(f"Comparison of {', '.join(valid_measures)} across categories of {x_col}.")
                return prepared_df, applied_params, insights
            elif len(valid_measures) == 1:
                y_col = valid_measures[0]

        # Case G: Categorical Aggregation (Bar, Horizontal Bar, or Pie)
        if x_col and x_col in work_df.columns:
            other_cols = [c for c in work_df.columns if c != x_col]
            is_pre_aggregated = (
                len(other_cols) >= 1
                and any("count" in c.lower() or "percentage" in c.lower() or "total" in c.lower() or "average" in c.lower() for c in other_cols)
            )

            # Check if measure is average on a range category column (e.g. average expected return by avenue)
            if y_col and isinstance(y_col, str) and y_col in work_df.columns and measure_type in ("mean", "avg", "average"):
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
            elif y_col and isinstance(y_col, str) and y_col in work_df.columns and measure_type in ("sum", "min", "max"):
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

            # Pie Chart category limit: max 6 categories + "Other"
            if chart_type in ("pie", "donut") and len(prepared_df) > 7:
                top_slice = prepared_df.iloc[:6]
                other_sum = prepared_df.iloc[6:][val_col].sum()
                other_row = pd.DataFrame([{x_col: "Other", val_col: other_sum}])
                prepared_df = pd.concat([top_slice, other_row], ignore_index=True)

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

            is_horizontal = (chart_type == "horizontal_bar" or (spec.get("chart") or {}).get("orientation") == "horizontal")
            chart_label = "Horizontal Bar Chart" if is_horizontal else ("Pie chart" if chart_type in ("pie", "donut") else "Bar chart")

            applied_params = {
                "chart_type": chart_label,
                "category": x_col,
                "value": val_col,
                "x_axis": x_col if not is_horizontal else val_col,
                "y_axis": val_col if not is_horizontal else x_col,
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

    @classmethod
    def prepare_visualization_data(
        cls,
        df: pd.DataFrame,
        spec: dict[str, Any],
        question: str = "",
        query_id: str = "",
        dataset_id: str = "",
    ) -> tuple[pd.DataFrame, dict[str, Any], list[str], dict[str, Any]]:
        """
        Dedicated Chart Data Preparation Layer adhering to AI-MADAC Specification:
        1. Validates tabular query result.
        2. Normalizes column types, resolves midpoints, applies filters, sorts.
        3. Enforces strictly 1-D structures for all Plotly axes (x, y, color, size).
        4. Logs full Section 46 debugging metrics.
        Returns: (prepared_df, applied_parameters, key_insights, chart_data)
        """
        prepared_df, applied_params, insights = cls.prepare(df, spec)

        fields = spec.get("fields") or {}
        chart = spec.get("chart") or {}
        chart_type = chart.get("type", "bar")
        x_col = fields.get("x")
        y_col = fields.get("y")
        color_col = fields.get("color")

        # Build normalized 1-D chart_data structure (Section 29)
        chart_data: dict[str, Any] = {
            "x": None,
            "y": None,
            "color": None,
            "size": None,
            "hover_data": None,
        }

        if not prepared_df.empty:
            cols = prepared_df.columns.tolist()
            real_x = x_col if (x_col and x_col in cols) else (cols[0] if cols else None)
            if real_x:
                chart_data["x"] = ensure_1d(prepared_df[real_x], str(real_x))

            real_y = y_col if (isinstance(y_col, str) and y_col in cols) else (
                applied_params.get("value") if applied_params.get("value") in cols else (
                    cols[1] if len(cols) > 1 and cols[1] != real_x else None
                )
            )
            if real_y:
                chart_data["y"] = ensure_1d(prepared_df[real_y], str(real_y))

            real_color = color_col if (color_col and color_col in cols) else (
                applied_params.get("series") if applied_params.get("series") in cols else None
            )
            if real_color:
                chart_data["color"] = ensure_1d(prepared_df[real_color], str(real_color))

        x_s = chart_data.get("x")
        y_s = chart_data.get("y")
        c_s = chart_data.get("color")

        x_type = type(x_s).__name__ if x_s is not None else "None"
        x_shape = str(x_s.shape) if x_s is not None else "None"
        y_type = type(y_s).__name__ if y_s is not None else "None"
        y_shape = str(y_s.shape) if y_s is not None else "None"
        c_type = type(c_s).__name__ if c_s is not None else "None"
        c_shape = str(c_s.shape) if c_s is not None else "None"

        nulls_removed = int(df.isna().any(axis=1).sum() - prepared_df.isna().any(axis=1).sum()) if df is not None and not df.empty else 0
        nulls_removed = max(0, nulls_removed)

        trace_cnt = len(c_s.unique()) if (c_s is not None and hasattr(c_s, "unique")) else 1

        logger.info(
            f"\n========== VISUALIZATION PIPELINE DEBUG ==========\n"
            f"QUERY_ID: {query_id or spec.get('query_id', 'N/A')}\n"
            f"QUESTION: {question}\n"
            f"DATASET_ID: {dataset_id}\n"
            f"INTENT: {spec.get('query_interpretation', {}).get('intent', 'VISUALIZATION')}\n"
            f"CHART_TYPE: {chart_type}\n"
            f"X_COLUMN: {x_col}\n"
            f"Y_COLUMN: {y_col}\n"
            f"SERIES_COLUMN: {color_col}\n"
            f"MEASURE: {spec.get('measure', {}).get('column')}\n"
            f"AGGREGATION: {spec.get('measure', {}).get('type')}\n"
            f"GROUP_BY: {[x_col, color_col] if color_col else [x_col]}\n"
            f"RESULT_COLUMNS: {prepared_df.columns.tolist()}\n"
            f"RESULT_ROW_COUNT: {len(prepared_df)}\n"
            f"PREPARED_X_TYPE: {x_type}\n"
            f"PREPARED_X_SHAPE: {x_shape}\n"
            f"PREPARED_Y_TYPE: {y_type}\n"
            f"PREPARED_Y_SHAPE: {y_shape}\n"
            f"PREPARED_COLOR_TYPE: {c_type}\n"
            f"PREPARED_COLOR_SHAPE: {c_shape}\n"
            f"NULL_ROWS_REMOVED: {nulls_removed}\n"
            f"FINAL_PLOTLY_TRACE_COUNT: {trace_cnt}\n"
            f"=================================================="
        )
        prepared_df.attrs["chart_data"] = chart_data
        return prepared_df, applied_params, insights, chart_data


def prepare_visualization_data(
    df: pd.DataFrame,
    spec: dict[str, Any],
    question: str = "",
    query_id: str = "",
    dataset_id: str = "",
) -> tuple[pd.DataFrame, dict[str, Any], list[str], dict[str, Any]]:
    """Standalone chart data preparation layer."""
    return DataPreparer.prepare_visualization_data(df, spec, question, query_id, dataset_id)

