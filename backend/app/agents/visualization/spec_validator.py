"""
Strict Visualization Specification Validator and Auto-Repair.

Validates Gemini specification against:
1. QueryContract (ensuring all requested fields are present and unrequested fields are rejected)
2. Dataset schema (ensuring no hallucinated columns)
3. Chart type compatibility (ensuring cross-tabulations produce grouped bars, distributions produce histograms, etc.)
4. Deterministic auto-repair fallback when Gemini output is invalid.
"""

from __future__ import annotations

from typing import Any, Optional
from app.agents.visualization.contract import QueryContract


SUPPORTED_CHART_TYPES = {
    "bar", "horizontal_bar", "grouped_bar", "stacked_bar",
    "clustered_column", "clustered_bar", "stacked_column", "stacked_bar",
    "line", "pie", "donut", "histogram", "scatter", "box", "box_plot", "heatmap", "area", "kpi", "card"
}


class SpecValidator:
    """Strict validator for visualization specifications."""

    @staticmethod
    def normalize_spec(spec: dict[str, Any]) -> dict[str, Any]:
        """Normalize flat or nested specification into canonical internal format."""
        if not isinstance(spec, dict):
            return spec

        normalized = dict(spec)
        visual_type = spec.get("visual_type") or spec.get("chart_type")
        chart_obj = dict(spec.get("chart") or {})

        if visual_type:
            v_lower = str(visual_type).lower().replace(" ", "_")
            if v_lower in ("clustered_column", "grouped_column"):
                chart_obj["type"] = "bar"
                chart_obj["mode"] = "grouped"
                chart_obj["orientation"] = "vertical"
            elif v_lower in ("clustered_bar", "grouped_bar"):
                chart_obj["type"] = "horizontal_bar"
                chart_obj["mode"] = "grouped"
                chart_obj["orientation"] = "horizontal"
            elif v_lower in ("stacked_column",):
                chart_obj["type"] = "bar"
                chart_obj["mode"] = "stacked"
                chart_obj["orientation"] = "vertical"
            elif v_lower in ("stacked_bar",):
                chart_obj["type"] = "horizontal_bar"
                chart_obj["mode"] = "stacked"
                chart_obj["orientation"] = "horizontal"
            else:
                chart_obj["type"] = v_lower

        if "type" not in chart_obj:
            chart_obj["type"] = "bar"
        normalized["chart"] = chart_obj

        fields = dict(spec.get("fields") or {})
        if "x" in spec and "x" not in fields:
            fields["x"] = spec["x"]
        if "y" in spec and "y" not in fields:
            fields["y"] = spec["y"]
        if "legend" in spec and "color" not in fields:
            fields["color"] = spec["legend"]
        if "color" in spec and "color" not in fields:
            fields["color"] = spec["color"]
        normalized["fields"] = fields

        measure = dict(spec.get("measure") or {})
        if "aggregation" in spec and "type" not in measure:
            measure["type"] = spec["aggregation"]
        if "y" in spec and spec.get("aggregation") and "column" not in measure:
            measure["column"] = spec["y"]
        normalized["measure"] = measure

        return normalized

    @staticmethod
    def validate(
        spec: dict[str, Any],
        contract: QueryContract,
        available_columns: list[str],
    ) -> tuple[bool, Optional[str]]:
        """
        Validate the visualization specification against contract and schema.
        Returns: (is_valid, failure_reason)
        """
        if not isinstance(spec, dict):
            return False, "Specification must be a JSON object."

        spec = SpecValidator.normalize_spec(spec)

        if not spec.get("visualization_available", True):
            return True, None

        chart = spec.get("chart") or {}
        chart_type = chart.get("type") or spec.get("chart_type")
        if not chart_type or chart_type not in SUPPORTED_CHART_TYPES:
            return False, f"Unsupported chart type: '{chart_type}'."

        fields = spec.get("fields") or {}
        spec_cols = set()
        for k in ("x", "y", "color", "facet"):
            val = fields.get(k)
            if val and isinstance(val, str):
                spec_cols.add(val)
            elif val and isinstance(val, list):
                for item in val:
                    if isinstance(item, str):
                        spec_cols.add(item)

        measure_col = (spec.get("measure") or {}).get("column")
        if measure_col and isinstance(measure_col, str):
            spec_cols.add(measure_col)
        elif measure_col and isinstance(measure_col, list):
            for item in measure_col:
                if isinstance(item, str):
                    spec_cols.add(item)

        # 1. Verify all spec columns exist in dataset schema (no hallucinated columns)
        MEASURE_ALIASES = {"count", "investor_count", "frequency", "percentage", "share", "records", "total_investors", "value"}

        for col in spec_cols:
            if col.lower() in MEASURE_ALIASES:
                continue
            if col not in available_columns:
                return False, f"Column '{col}' does not exist in dataset columns."

        # 2. Strict Query Contract Validation:
        # Every requested field MUST appear in spec_cols!
        for req in contract.requested_fields:
            if req not in spec_cols:
                return False, f"Field '{req}' was explicitly requested in user query, but is missing from specification."

        # 3. Reject unrequested columns (e.g. injecting 'age' when user asked for 'gender vs objective')
        for col in spec_cols:
            if col.lower() in MEASURE_ALIASES:
                continue
            is_requested = col in contract.requested_fields
            is_filter_col = any(((f.get("column") or f.get("field")) if isinstance(f, dict) else getattr(f, "column", None)) == col for f in contract.filters)
            meas_col_val = contract.measure.get("column") if isinstance(contract.measure, dict) else (getattr(contract, "measure_column", None) or getattr(contract, "measure", None))
            is_measure_col = (meas_col_val == col)
            is_series_col = (col == getattr(contract, "series_dimension", None)) or (col == getattr(contract, "color_by", None))
            is_sec_meas_col = (col == getattr(contract, "secondary_measure_column", None)) or (col == getattr(contract, "secondary_measure", None))
            if not (is_requested or is_filter_col or is_measure_col or is_series_col or is_sec_meas_col):
                return False, f"Column '{col}' was not requested in user query and cannot replace requested fields."

        # 4. Critical Cross-Tabulation Validation:
        # For multi-dimensional comparisons (e.g. 'gender vs investment objective')
        if contract.intent == "cross_tabulation" or len(contract.dimensions) >= 2:
            x_field = fields.get("x")
            color_field = fields.get("color")
            if not x_field or not color_field:
                return False, "Cross-tabulation queries require both an x dimension and a color grouping dimension."
            if x_field == color_field:
                return False, "x dimension and color grouping dimension must be distinct."
            if chart_type not in ("bar", "grouped_bar", "stacked_bar", "heatmap"):
                return False, f"Cross-tabulation requires a grouped bar or heatmap, not '{chart_type}'."

        # 5. Histogram Validation:
        if contract.intent == "distribution" or contract.explicit_chart_type == "histogram":
            if chart_type != "histogram" and chart_type != "box":
                return False, f"Distribution query requires a histogram or box plot, not '{chart_type}'."

        # 6. Scatter Validation:
        if contract.intent == "relationship" or contract.explicit_chart_type == "scatter":
            if chart_type != "scatter" and chart_type != "heatmap":
                return False, f"Relationship query between numerical variables requires a scatter plot, not '{chart_type}'."
            if not fields.get("x") or not fields.get("y"):
                return False, "Scatter plot requires both x and y numerical fields."

        # 7. Box Plot Validation:
        if contract.explicit_chart_type in ("box", "box_plot"):
            if chart_type not in ("box", "box_plot"):
                return False, f"Box plot was explicitly requested, but specification returned '{chart_type}'."

        return True, None

    @staticmethod
    def build_deterministic_spec(
        contract: QueryContract,
        available_columns: list[str],
    ) -> dict[str, Any]:
        """
        Build an infallible, mathematically correct visualization specification
        directly from the QueryContract. Used as auto-repair fallback.
        Works generically for any dataset schema without hardcoded column names.
        """
        requested = [str(f) for f in contract.requested_fields if f]
        dims = [str(d) for d in contract.dimensions if d]
        explicit_chart = str(contract.explicit_chart_type or "").lower()
        intent = str(contract.intent or "").lower()
        avail = [str(c) for c in available_columns if c]

        # Case 0: Time Series / Line Chart (Single or Multi-Series)
        is_time_series = (
            explicit_chart in ("line", "trend")
            or intent in ("trend", "time_series")
            or bool(contract.time_granularity)
            or any("date" in str(f).lower() or "time" in str(f).lower() or "year" in str(f).lower() for f in requested + dims)
        )
        if is_time_series:
            # Detect temporal column for X
            time_col = next((
                c for c in dims + requested + avail
                if any(k in c.lower() for k in ("invoicedate", "order_date", "orderdate", "date", "timestamp", "time", "year", "month", "day", "created_at", "saledate"))
            ), (dims[0] if dims else (requested[0] if requested else (avail[0] if avail else "Date"))))

            # Detect measure column for Y
            meas_col = contract.measure.get("column") if isinstance(contract.measure, dict) else None
            if not meas_col:
                meas_col = next((
                    c for c in requested + avail
                    if c != time_col and any(k in c.lower() for k in ("revenue", "total", "quantity", "price", "amount", "sales", "profit", "value", "unitprice"))
                ), (requested[0] if requested and requested[0] != time_col else (avail[1] if len(avail) > 1 else "Value")))

            meas_type = contract.measure.get("type", "sum") if isinstance(contract.measure, dict) else "sum"
            meas_label = str(meas_col).replace("_", " ").title()

            # Multi-series line chart check
            series_dim = getattr(contract, "series_dimension", None)
            if not series_dim and len(dims) > 1:
                series_dim = next((d for d in dims if d != time_col), None)

            chart_title = f"{meas_label} by {str(series_dim).replace('_', ' ').title()} Over Time" if series_dim else f"{meas_label} Over Time"

            return {
                "visualization_available": True,
                "query_interpretation": {
                    "intent": "time_series",
                    "requested_fields": [time_col, meas_col] + ([series_dim] if series_dim else []),
                },
                "fields": {
                    "x": time_col,
                    "y": meas_col,
                    "color": series_dim,
                    "facet": None,
                },
                "measure": {
                    "type": meas_type,
                    "column": meas_col,
                },
                "filters": contract.filters,
                "sorting": {"field": time_col, "direction": "asc"},
                "limit": contract.limit,
                "binning": None,
                "time_granularity": getattr(contract, "time_granularity", None) or "month",
                "chart": {
                    "type": "line",
                    "orientation": "vertical",
                    "mode": "lines+markers",
                },
                "title": chart_title,
                "axis_labels": {
                    "x": str(time_col).replace("_", " ").title(),
                    "y": meas_label,
                },
                "legend": {
                    "show": bool(series_dim),
                    "title": str(series_dim).replace("_", " ").title() if series_dim else None,
                },
            }

        # Case 1: Relationship / Scatter Plot (raw observations between two numeric columns)
        if intent == "relationship" or explicit_chart == "scatter":
            num_cols = [
                c for c in requested + avail
                if any(k in c.lower() for k in ("quantity", "price", "unitprice", "age", "income", "amount", "sales", "profit", "cost", "value", "return", "midpoint"))
            ]
            x_col = num_cols[0] if len(num_cols) > 0 else (requested[0] if requested else (avail[0] if avail else "X"))
            y_col = num_cols[1] if len(num_cols) > 1 else (requested[1] if len(requested) > 1 else (avail[1] if len(avail) > 1 else x_col))

            color_col = getattr(contract, "color_by", None)
            if not color_col and len(dims) > 0 and dims[0] not in (x_col, y_col):
                color_col = dims[0]

            title_text = f"{str(y_col).replace('_', ' ').title()} vs {str(x_col).replace('_', ' ').title()}"
            if color_col:
                title_text += f" by {str(color_col).replace('_', ' ').title()}"

            return {
                "visualization_available": True,
                "query_interpretation": {
                    "intent": "relationship",
                    "requested_fields": [x_col, y_col] + ([color_col] if color_col else []),
                },
                "fields": {
                    "x": x_col,
                    "y": y_col,
                    "color": color_col,
                    "facet": None,
                },
                "measure": {
                    "type": "raw",
                    "column": y_col,
                },
                "filters": contract.filters,
                "sorting": contract.sorting,
                "limit": contract.limit or 5000,
                "binning": None,
                "chart": {
                    "type": "scatter",
                    "orientation": "vertical",
                    "mode": "markers",
                },
                "title": title_text,
                "axis_labels": {
                    "x": str(x_col).replace("_", " ").title(),
                    "y": str(y_col).replace("_", " ").title(),
                },
                "legend": {
                    "show": bool(color_col),
                    "title": str(color_col).replace("_", " ").title() if color_col else None,
                },
            }

        # Case 2: Distribution / Histogram (raw observations of one numeric column)
        if intent == "distribution" or explicit_chart == "histogram":
            target_col = requested[0] if requested else (dims[0] if dims else (avail[0] if avail else "Value"))
            return {
                "visualization_available": True,
                "query_interpretation": {
                    "intent": "distribution",
                    "requested_fields": [target_col],
                },
                "fields": {
                    "x": target_col,
                    "y": None,
                    "color": None,
                    "facet": None,
                },
                "measure": {
                    "type": "raw",
                    "column": target_col,
                },
                "filters": contract.filters,
                "sorting": contract.sorting,
                "limit": contract.limit or 5000,
                "binning": contract.binning,
                "chart": {
                    "type": "histogram",
                    "orientation": "vertical",
                    "mode": "overlay",
                },
                "title": f"Distribution of {str(target_col).replace('_', ' ').title()}",
                "axis_labels": {
                    "x": str(target_col).replace("_", " ").title(),
                    "y": "Count of Observations",
                },
                "legend": {
                    "show": False,
                    "title": None,
                },
            }

        # Case 3: Box Plot
        if explicit_chart in ("box", "box_plot"):
            if len(requested) >= 2:
                x_col, y_col = requested[0], requested[1]
            else:
                x_col = dims[0] if dims else (avail[0] if avail else "Category")
                y_col = requested[0] if requested else (avail[1] if len(avail) > 1 else x_col)

            return {
                "visualization_available": True,
                "query_interpretation": {
                    "intent": "distribution",
                    "requested_fields": [y_col, x_col],
                },
                "fields": {
                    "x": x_col,
                    "y": y_col,
                    "color": None,
                    "facet": None,
                },
                "measure": {
                    "type": "raw",
                    "column": y_col,
                },
                "filters": contract.filters,
                "sorting": contract.sorting,
                "limit": contract.limit,
                "binning": None,
                "chart": {
                    "type": "box",
                    "orientation": "vertical",
                },
                "title": f"{str(y_col).replace('_', ' ').title()} by {str(x_col).replace('_', ' ').title()}",
                "axis_labels": {
                    "x": str(x_col).replace("_", " ").title(),
                    "y": str(y_col).replace("_", " ").title(),
                },
                "legend": {
                    "show": False,
                    "title": None,
                },
            }

        # Case 4: Proportion / Pie chart
        if explicit_chart in ("pie", "donut") or intent == "proportion":
            x_col = dims[0] if dims else (requested[0] if requested else (avail[0] if avail else "Category"))
            meas_col = contract.measure.get("column") if isinstance(contract.measure, dict) else None
            if not meas_col:
                meas_candidates = [f for f in requested if f != x_col]
                if meas_candidates:
                    meas_col = meas_candidates[0]
            meas_type = contract.measure.get("type", "sum") if isinstance(contract.measure, dict) else "sum"
            meas_title = f"{str(meas_col).replace('_', ' ').title()} " if (meas_col and meas_type != 'count') else ""
            title_text = f"Distribution: {x_col}" if not meas_title else f"{meas_title}Contribution by {str(x_col).replace('_', ' ').title()}"

            return {
                "visualization_available": True,
                "query_interpretation": {
                    "intent": "proportion",
                    "requested_fields": [x_col] + ([meas_col] if meas_col else []),
                },
                "fields": {
                    "x": x_col,
                    "y": meas_col,
                    "color": None,
                    "facet": None,
                },
                "measure": {
                    "type": meas_type,
                    "column": meas_col,
                },
                "filters": contract.filters,
                "sorting": contract.sorting,
                "limit": contract.limit,
                "binning": None,
                "chart": {
                    "type": "pie",
                    "orientation": "vertical",
                    "mode": "donut" if explicit_chart == "donut" else "pie",
                },
                "title": title_text,
                "axis_labels": {
                    "x": str(x_col).replace("_", " ").title(),
                    "y": "Share",
                },
                "legend": {
                    "show": True,
                    "title": str(x_col).replace("_", " ").title(),
                },
            }

        # Case 5: Multi-measure comparison (e.g. Sales and Profit by Category)
        sec_meas = getattr(contract, "secondary_measure_column", None) or getattr(contract, "secondary_measure", None)
        if sec_meas:
            x_col = dims[0] if dims else (requested[0] if requested else (avail[0] if avail else "Category"))
            meas_col = contract.measure.get("column") if isinstance(contract.measure, dict) else None
            if not meas_col:
                meas_col = next((f for f in requested + avail if f not in (x_col, sec_meas)), "Sales")
            meas_label = str(meas_col).replace("_", " ").title()
            sec_label = str(sec_meas).replace("_", " ").title()
            return {
                "visualization_available": True,
                "query_interpretation": {
                    "intent": "multi_measure_comparison",
                    "requested_fields": [x_col, meas_col, sec_meas],
                },
                "fields": {
                    "x": x_col,
                    "y": [meas_col, sec_meas],
                    "color": None,
                    "facet": None,
                },
                "measure": {
                    "type": "sum",
                    "column": meas_col,
                },
                "filters": contract.filters,
                "sorting": contract.sorting,
                "limit": contract.limit,
                "binning": None,
                "chart": {
                    "type": "bar",
                    "orientation": "vertical",
                    "mode": "grouped",
                },
                "title": f"{meas_label} and {sec_label} by {str(x_col).replace('_', ' ').title()}",
                "axis_labels": {
                    "x": str(x_col).replace("_", " ").title(),
                    "y": "Values",
                },
                "legend": {
                    "show": True,
                    "title": "Measures",
                },
            }

        # Case 6: Cross-tabulation / 2 categorical dimensions
        if (intent == "cross_tabulation" or len(dims) >= 2) and intent not in ("relationship", "distribution", "time_series"):
            d1 = dims[0] if dims else (requested[0] if requested else (avail[0] if avail else "Category"))
            d2 = dims[1] if len(dims) > 1 else (requested[1] if len(requested) > 1 else (avail[1] if len(avail) > 1 else "Group"))
            x_col, color_col = d1, d2

            return {
                "visualization_available": True,
                "query_interpretation": {
                    "intent": "cross_tabulation",
                    "requested_fields": [d1, d2],
                },
                "fields": {
                    "x": x_col,
                    "y": None,
                    "color": color_col,
                    "facet": None,
                },
                "measure": {
                    "type": "count",
                    "column": None,
                },
                "filters": contract.filters,
                "sorting": contract.sorting,
                "limit": contract.limit,
                "binning": None,
                "chart": {
                    "type": "bar",
                    "orientation": "vertical",
                    "mode": "grouped",
                },
                "title": f"Breakdown by {str(color_col).replace('_', ' ').title()} and {str(x_col).replace('_', ' ').title()}",
                "axis_labels": {
                    "x": str(x_col).replace("_", " ").title(),
                    "y": "Count",
                },
                "legend": {
                    "show": True,
                    "title": str(color_col).replace("_", " ").title(),
                },
            }

        # Case 7: Single category aggregation / Ranking Bar chart (Generic Default)
        x_col = dims[0] if dims else (requested[0] if requested else (avail[0] if avail else "Category"))
        meas_type = contract.measure.get("type", "count") if isinstance(contract.measure, dict) else "count"
        meas_col = (
            contract.measure.get("column") if isinstance(contract.measure, dict) else None
        ) or next((c for c in avail if c != x_col), None)

        y_label = f"Total {str(meas_col).replace('_', ' ').title()}" if (meas_col and meas_type != "count") else "Count"

        # Determine orientation: Ranking / Top-N queries use horizontal bar for readability
        is_horizontal = (
            intent == "ranking"
            or explicit_chart in ("horizontal_bar", "bar_horizontal")
            or (contract.limit is not None and contract.limit <= 15)
        )

        return {
            "visualization_available": True,
            "query_interpretation": {
                "intent": intent,
                "requested_fields": requested,
            },
            "fields": {
                "x": x_col,
                "y": meas_col,
                "color": None,
                "facet": None,
            },
            "measure": {
                "type": meas_type,
                "column": meas_col,
            },
            "filters": contract.filters,
            "sorting": contract.sorting,
            "limit": contract.limit,
            "binning": None,
            "chart": {
                "type": "bar",
                "orientation": "horizontal" if is_horizontal else "vertical",
                "mode": "standard",
            },
            "title": f"{y_label} by {str(x_col).replace('_', ' ').title()}",
            "axis_labels": {
                "x": str(x_col).replace("_", " ").title() if not is_horizontal else y_label,
                "y": y_label if not is_horizontal else str(x_col).replace("_", " ").title(),
            },
            "legend": {
                "show": False,
                "title": None,
            },
        }
