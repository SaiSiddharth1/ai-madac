"""
Pandas Data Preparation and Statistical Insight Calculation for AI-MADAC.

Adheres to:
1. RAW DATA RULE: Python/Pandas calculates counts, means, rankings, bins, and metrics.
2. HISTOGRAM RULE: Uses raw observations directly; sum of frequencies == total valid rows.
3. MATHEMATICAL INSIGHTS: Calculates true statistical extremes in Python (never invented).
4. STRICT FILTERING & SORTING: Applies current query filters and ordering in Pandas.
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
import pandas as pd


class DataPreparer:
    """Prepares tabular data for Plotly rendering and calculates true mathematical insights."""

    @staticmethod
    def apply_filters(df: pd.DataFrame, filters: list[dict[str, Any]]) -> pd.DataFrame:
        """Apply filters specified in the validated specification to the DataFrame."""
        filtered_df = df.copy()
        for f in filters:
            col = f.get("column")
            op = f.get("op")
            val = f.get("value")
            if not col or col not in filtered_df.columns:
                continue

            try:
                if op == ">":
                    filtered_df = filtered_df[pd.to_numeric(filtered_df[col], errors="coerce") > float(val)]
                elif op == "<":
                    filtered_df = filtered_df[pd.to_numeric(filtered_df[col], errors="coerce") < float(val)]
                elif op == ">=":
                    filtered_df = filtered_df[pd.to_numeric(filtered_df[col], errors="coerce") >= float(val)]
                elif op == "<=":
                    filtered_df = filtered_df[pd.to_numeric(filtered_df[col], errors="coerce") <= float(val)]
                elif op == "between" and isinstance(val, (tuple, list)) and len(val) == 2:
                    num_series = pd.to_numeric(filtered_df[col], errors="coerce")
                    filtered_df = filtered_df[(num_series >= float(val[0])) & (num_series <= float(val[1]))]
                elif op in ("==", "="):
                    filtered_df = filtered_df[filtered_df[col].astype(str).str.strip().str.lower() == str(val).strip().lower()]
                elif op in ("!=", "<>"):
                    filtered_df = filtered_df[filtered_df[col].astype(str).str.strip().str.lower() != str(val).strip().lower()]
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
        binning = spec.get("binning")

        # 1. Apply Filters
        work_df = cls.apply_filters(df, filters)

        insights: list[str] = []
        parameters_used: list[str] = []

        if filters:
            for f in filters:
                col = f.get("column")
                op = f.get("op")
                val = f.get("value")
                parameters_used.append(f"Filter: {col} {op} {val}")

        # 2. Prepare by Chart Type & Intent

        # Case A: Cross-Tabulation / Grouped Bar
        if color_col and color_col in work_df.columns and x_col and x_col in work_df.columns:
            prepared_df = (
                work_df.groupby([x_col, color_col])
                .size()
                .reset_index(name="count")
            )
            # Sort within group if requested
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
                "filters": [f"{f['column']} {f['op']} {f['value']}" for f in filters],
                "sorting": sorting.get("direction"),
                "limit": limit,
            }
            return prepared_df, applied_params, insights

        # Case B: Histogram / Distribution
        if chart_type == "histogram":
            # HISTOGRAM RULE: Keep raw observations
            raw_obs = work_df[x_col].dropna()
            prepared_df = pd.DataFrame({x_col: raw_obs})
            total_obs = len(raw_obs)

            parameters_used.append(f"Variable: {x_col}")
            parameters_used.append(f"Raw Observations: {total_obs}")
            if binning and binning.get("bin_size"):
                parameters_used.append(f"Bin Size: {binning['bin_size']}")

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
                "bin_size": binning.get("bin_size") if binning else None,
                "filters": [f"{f['column']} {f['op']} {f['value']}" for f in filters],
            }
            return prepared_df, applied_params, insights

        # Case C: Scatter / Relationship
        if chart_type == "scatter":
            prepared_df = work_df[[x_col, y_col]].dropna()
            total_obs = len(prepared_df)
            parameters_used.append(f"X-Axis: {x_col}")
            parameters_used.append(f"Y-Axis: {y_col}")
            parameters_used.append(f"Observations: {total_obs}")

            if total_obs > 1:
                try:
                    corr = float(prepared_df[x_col].corr(prepared_df[y_col]))
                    direction = "positive" if corr > 0.1 else ("negative" if corr < -0.1 else "neutral / weak")
                    insights.append(
                        f"Correlation between {x_col} and {y_col} is {corr:.2f} ({direction} relationship across {total_obs} observations)."
                    )
                except Exception:
                    insights.append(f"Plotted {total_obs} observations comparing {x_col} and {y_col}.")

            applied_params = {
                "chart_type": "Scatter Plot",
                "x_axis": x_col,
                "y_axis": y_col,
                "observation_count": total_obs,
                "filters": [f"{f['column']} {f['op']} {f['value']}" for f in filters],
            }
            return prepared_df, applied_params, insights

        # Case D: Categorical Aggregation (Bar or Pie)
        if x_col and x_col in work_df.columns:
            # Detect if source dataframe is already aggregated
            other_cols = [c for c in work_df.columns if c != x_col]
            is_pre_aggregated = (
                len(other_cols) >= 1
                and any("count" in c.lower() or "percentage" in c.lower() or "total" in c.lower() or "average" in c.lower() for c in other_cols)
            )

            if is_pre_aggregated:
                val_col = other_cols[0]
                prepared_df = work_df[[x_col, val_col]].copy()
                agg_label = "Already aggregated by source agent"
                parameters_used.append(f"Category: {x_col}")
                parameters_used.append(f"Value: {val_col}")
                parameters_used.append(f"Aggregation: {agg_label}")
            elif y_col and y_col in work_df.columns and measure_type in ("mean", "avg", "average", "sum", "min", "max"):
                agg_fn = "mean" if measure_type in ("mean", "avg", "average") else measure_type
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
                insights.append(
                    f"Top category is '{top_row[x_col]}' with {top_row[val_col]}."
                )
                if len(prepared_df) > 1:
                    low_row = prepared_df.iloc[-1]
                    insights.append(
                        f"Lowest category is '{low_row[x_col]}' with {low_row[val_col]}."
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
                    f"{f['column']} = '{f['value']}'" if f.get('op') in ('=', '==') else f"{f['column']} {f['op']} {f['value']}"
                    for f in filters
                ]
                applied_params["filters"] = ", ".join(filter_strs)

            return prepared_df, applied_params, insights

        # Fallback raw data
        return work_df, {"chart_type": "Raw Preview"}, ["Data displayed without aggregation."]
