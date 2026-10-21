"""
Visualization Agent.
Autonomous intelligence layer for AI-MADAC visualizations.
Analyzes user query and current analysis data, determines visualization utility,
selects the optimal chart type, and outputs a machine-readable Plotly specification.
Python/FastAPI backend generates the interactive Plotly figure and static preview.
"""

import json
import logging
import os
import re
import uuid
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from app.agents.state import AgentState
from app.config import settings
from app.core.llm_provider import get_llm

logger = logging.getLogger(__name__)

VISUALIZATION_AGENT_PROMPT = """You are the Visualization Agent of AI-MADAC, an autonomous multi-agent data science analyst.

Your job is to analyze the CURRENT USER QUERY and the CURRENT DATA/ANALYSIS RESULT, determine whether a visualization is useful, select the most appropriate visualization, and return a structured visualization specification that the AI-MADAC backend will use to generate the ACTUAL interactive chart using Plotly.

You are NOT the chart renderer. Plotly is the chart renderer.
Your responsibility is to make the correct visualization decision and provide an accurate, machine-readable specification for Plotly.

CRITICAL STATE ISOLATION RULE:
Every request is independent unless the CURRENT USER QUERY explicitly refers to a previous result.
NEVER reuse previous chart type, columns, data, aggregation, filters, sorting, title, explanation, or insights.

CHART SELECTION:
- BAR: Category comparisons, rankings, counts by category, aggregated values, top-N. (Prefer horizontal bar if > 10 categories).
- LINE: Time trends, chronological data, continuous time-series.
- PIE / DONUT: Part-to-whole relationships, proportions, percentages, small number of categories (2-10). Use donut if explicitly requested.
- HISTOGRAM: Numerical distributions (age, salary, income, etc.). Use raw observations whenever possible.
- SCATTER PLOT: Relationship between two continuous numerical variables, correlation.
- BOX PLOT: Numerical distribution spread, median, quartiles, outliers across categories.
- HEATMAP: Correlation matrices, 2D frequency/intensity matrices.
- AREA CHART: Cumulative volume/magnitude trends over time.

HISTOGRAM RULE:
Use RAW numerical observations whenever available.
Validate: sum of histogram bin frequencies == number of valid raw observations.

AGGREGATION:
If data is already aggregated, set aggregation = "already_aggregated". Do NOT aggregate again.

OUTPUT FORMAT (JSON only):
{
  "visualization_available": true,
  "chart": {
    "chart_type": "bar",
    "title": "...",
    "x_axis": "...",
    "y_axis": "...",
    "data": [ ... ],
    "plotly_options": { ... }
  },
  "parameters": {
    "aggregation": "...",
    "group_by": "...",
    "filters": [],
    "sorting": "...",
    "limit": null
  },
  "explanation": {
    "what_this_shows": "...",
    "parameters_used": [ ... ],
    "key_insights": [ ... ]
  }
}
"""


class ExplanationDict(dict):
    """
    Structured explanation dictionary adhering to the AI-MADAC Step 19 contract
    (what_this_shows, parameters_used, key_insights) while providing text formatting
    for backwards compatibility.
    """

    def __contains__(self, item: Any) -> bool:
        if super().__contains__(item):
            return True
        return str(item).lower() in self.as_text().lower()

    def as_text(self) -> str:
        lines = []
        if self.get("what_this_shows"):
            lines.append(str(self["what_this_shows"]))
        params = self.get("parameters_used", [])
        if params:
            lines.append("\nParameters used:")
            for p in params:
                lines.append(f"- {p}")
        insights = self.get("key_insights", [])
        if insights:
            lines.append("\nKey Insights:")
            for ins in insights:
                lines.append(f"- {ins}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.as_text()


class VisualizationAgent:
    """
    Visualization Agent responsible for chart specification decisions,
    data validation, Plotly figure generation, and structured explanation delivery.
    """

    def __init__(self):
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm(temperature=0.0)
        return self._llm

    def run(self, state: AgentState) -> dict[str, Any]:
        """
        Execute the visualization decision pipeline and produce the Plotly figure.
        """
        sql_result = state.get("sql_result")
        analysis_result = state.get("analysis_result")
        question = state.get("question", "")
        structured_intent = state.get("structured_intent") or {}

        # Follow-up visualization: reuse previous result only if explicit
        if not sql_result and state.get("previous_validated_result"):
            previous = state["previous_validated_result"]
            if previous.get("validated") and str(previous.get("dataset_id")) == str(state.get("dataset_id")):
                sql_result = {
                    "rows": previous.get("result", []),
                    "columns": previous.get("columns_used", []),
                    "row_count": previous.get("row_count", 0),
                    "query": previous.get("query"),
                    "filters": previous.get("filters", {}),
                }

        df = pd.DataFrame()
        is_grouped_comparison = False

        # Support Data Analyst Agent relationship/impact findings
        if analysis_result and analysis_result.get("analysis_type") == "relationship_impact_analysis":
            with_grp = analysis_result.get("group_with_event", {})
            without_grp = analysis_result.get("group_without_event", {})
            pred_col = analysis_result.get("predictor_column", "Factor")
            target_col = analysis_result.get("target_column", "Impact")

            df = pd.DataFrame([
                {"Group": f"With {pred_col}", f"Average {target_col}": with_grp.get("mean_target", 0)},
                {"Group": f"Without {pred_col}", f"Average {target_col}": without_grp.get("mean_target", 0)},
            ])
            is_grouped_comparison = True

        elif sql_result and sql_result.get("rows"):
            rows = sql_result.get("rows", [])
            df = pd.DataFrame(rows)

        # STEP 17: No Data / Insufficient Data handling
        if df.empty:
            logger.warning("Visualization Agent: No tabular data available.")
            failure_result = self._empty_visualization_result()
            return {
                "visualization_result": failure_result,
                "agents_used": state.get("agents_used", []) + ["visualization_agent"],
            }

        # STEP 16: Single Value Results
        asks_visual_explicit = bool(re.search(r"\b(?:chart|plot|graph|pie|bar|visualize|histogram|scatter)\b", question.lower()))
        if len(df) == 1 and not is_grouped_comparison and not asks_visual_explicit:
            logger.info("Visualization Agent: Single value KPI result, skipping chart.")
            failure_result = self._empty_visualization_result(
                reason="Single value results are better communicated directly as a KPI rather than a chart."
            )
            return {
                "visualization_result": failure_result,
                "agents_used": state.get("agents_used", []) + ["visualization_agent"],
            }

        logger.info(f"Visualization Agent processing {len(df)} rows for query: '{question}'")

        try:
            # STEP 1 to 11: Decide visualization specification (Intelligence Layer)
            spec = self._decide_specification(
                df=df,
                question=question,
                structured_intent=structured_intent,
                sql_result=sql_result,
                is_grouped_comparison=is_grouped_comparison,
            )

            # STEP 12 & 13: Render actual Plotly Figure (Backend Rendering Layer)
            plotly_fig, chart_title = self._render_plotly_figure(spec, df, question)

            # Persist interactive Plotly JSON & static PNG preview
            settings.ensure_directories()
            chart_id = uuid.uuid4().hex[:10]
            json_filename = f"chart_{chart_id}.json"
            png_filename = f"chart_{chart_id}.png"
            json_filepath = os.path.join(settings.CHARTS_DIR, json_filename)
            png_filepath = os.path.join(settings.CHARTS_DIR, png_filename)

            chart_json_str = pio.to_json(plotly_fig)
            with open(json_filepath, "w", encoding="utf-8") as f:
                f.write(chart_json_str)

            chart_json_dict = json.loads(chart_json_str)

            # Static PNG preview
            self._write_png_preview(
                df=df,
                spec=spec,
                path=png_filepath,
                title=chart_title,
                question=question,
            )

            chart_paths = [f"/static/charts/{png_filename}"]
            chart_jsons = [chart_json_dict]
            titles = [chart_title]

            # Build final structured visualization result
            viz_data = {
                "visualization_available": True,
                "chart": {
                    "chart_type": spec["chart_type"],
                    "title": chart_title,
                    "x_axis": spec.get("x_axis"),
                    "y_axis": spec.get("y_axis"),
                    "category": spec.get("category"),
                    "value": spec.get("value"),
                    "group_by": spec.get("group_by"),
                    "parameters": spec["parameters"],
                    "data": df.head(50).to_dict(orient="records"),
                    "plotly_options": spec.get("plotly_options", {}),
                },
                "parameters": spec["parameters"],
                "explanation": spec["explanation"],
                "key_insights": spec["explanation"].get("key_insights", []),
                # Backward compatibility & rendering metadata
                "chart_type": spec["chart_type"],
                "chart_path": chart_paths[0],
                "chart_paths": chart_paths,
                "chart_json": chart_jsons[0],
                "chart_jsons": chart_jsons,
                "titles": titles,
            }

            agents_used = state.get("agents_used", []) + ["visualization_agent"]

            return {
                "visualization_result": viz_data,
                "agents_used": agents_used,
            }

        except Exception as e:
            logger.error(f"Visualization Agent error: {e}", exc_info=True)
            failure_result = self._empty_visualization_result(reason=str(e))
            return {
                "visualization_result": failure_result,
                "agents_used": state.get("agents_used", []) + ["visualization_agent"],
            }

    def _decide_specification(
        self,
        df: pd.DataFrame,
        question: str,
        structured_intent: dict[str, Any],
        sql_result: dict[str, Any] | None,
        is_grouped_comparison: bool,
    ) -> dict[str, Any]:
        """
        Determine the optimal chart type, axes, parameters, and explanations.
        """
        cols = df.columns.tolist()
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        q_lower = question.lower()

        # Step 4: Select Chart Type
        chart_type = "bar"
        x_axis = None
        y_axis = None
        plotly_opts: dict[str, Any] = {}

        if is_grouped_comparison:
            chart_type = "bar"
            x_axis = "Group"
            y_axis = cols[1]
            plotly_opts = {"orientation": "vertical"}

        elif "pie" in q_lower or "donut" in q_lower or "proportion" in q_lower or "share" in q_lower:
            chart_type = "pie"
            x_axis = cat_cols[0] if cat_cols else cols[0]
            y_axis = num_cols[0] if num_cols else (cols[1] if len(cols) > 1 else None)
            if "donut" in q_lower:
                plotly_opts["hole"] = 0.4

        elif "histogram" in q_lower or ("distribution" in q_lower and num_cols and not cat_cols):
            chart_type = "histogram"
            x_axis = num_cols[0] if num_cols else cols[0]
            y_axis = None

        elif "scatter" in q_lower or "correlation" in q_lower or ("relationship" in q_lower and len(num_cols) >= 2):
            chart_type = "scatter"
            if len(num_cols) >= 2:
                x_axis = num_cols[0]
                y_axis = num_cols[1]
            else:
                x_axis = cols[0]
                y_axis = cols[1] if len(cols) > 1 else None

        elif "box plot" in q_lower or "boxplot" in q_lower:
            chart_type = "box"
            x_axis = cat_cols[0] if cat_cols else cols[0]
            y_axis = num_cols[0] if num_cols else None

        elif "area" in q_lower:
            chart_type = "area"
            x_axis = cat_cols[0] if cat_cols else cols[0]
            y_axis = num_cols[0] if num_cols else (cols[1] if len(cols) > 1 else None)

        elif "trend" in q_lower or "over time" in q_lower or "line" in q_lower or any(token in q_lower for token in ("year", "month", "date")):
            chart_type = "line"
            x_axis = cols[0]
            y_axis = num_cols[0] if num_cols else (cols[1] if len(cols) > 1 else None)

        elif "heatmap" in q_lower:
            chart_type = "heatmap"

        else:
            # Default categorical comparison
            if cat_cols and len(cat_cols) > 0:
                x_axis = cat_cols[0]
                y_axis = num_cols[0] if num_cols else (cols[1] if len(cols) > 1 else None)
            else:
                x_axis = cols[0]
                y_axis = cols[1] if len(cols) > 1 else None

            # Horizontal bar preference if many categories (> 10)
            if len(df) > 10:
                plotly_opts["orientation"] = "horizontal"
            else:
                plotly_opts["orientation"] = "vertical"

        # Step 6 & 7 & 8: Determine Parameters
        applied_filters = (sql_result or {}).get("filters", {})
        filter_list = []
        if applied_filters and isinstance(applied_filters, dict):
            filter_list = [f"{col} = '{val}'" for col, val in applied_filters.items()]

        # Aggregation determination
        if is_grouped_comparison:
            aggregation = "Average"
        elif y_axis and any(tok in y_axis.lower() for tok in ("count", "total", "average", "mean", "sum", "percentage")):
            aggregation = "already_aggregated"
        elif structured_intent.get("aggregation"):
            aggregation = structured_intent.get("aggregation")
        elif chart_type == "histogram":
            aggregation = "bin_count"
        else:
            aggregation = "already_aggregated" if len(cols) >= 2 else "none"

        sorting = "none"
        if y_axis and y_axis in df.columns and pd.api.types.is_numeric_dtype(df[y_axis]):
            if df[y_axis].is_monotonic_decreasing:
                sorting = "descending"
            elif df[y_axis].is_monotonic_increasing:
                sorting = "ascending"

        limit = structured_intent.get("limit_n") or None

        chart_display = {
            "bar": "Bar chart",
            "line": "Line chart",
            "pie": "Pie chart",
            "histogram": "Histogram",
            "scatter": "Scatter plot",
            "box": "Box plot",
            "heatmap": "Heatmap",
            "area": "Area chart",
        }.get(chart_type, chart_type.capitalize() + " chart")

        agg_display = "Already aggregated by source agent" if aggregation == "already_aggregated" else aggregation

        parameters = {
            "chart_type": chart_display,
            "x_axis": x_axis,
            "y_axis": y_axis,
            "category": x_axis if chart_type in {"bar", "pie"} else None,
            "value": y_axis,
            "aggregation": agg_display,
            "group_by": x_axis if chart_type in {"bar", "pie", "box"} else None,
            "filters": ", ".join(filter_list) if filter_list else None,
            "sorting": sorting if sorting != "none" else None,
            "limit": limit,
        }
        # Prune None keys
        parameters = {k: v for k, v in parameters.items() if v is not None}

        # Step 15: Generate grounded explanation and key insights
        chart_names = {
            "bar": "bar chart",
            "line": "line chart",
            "pie": "pie chart",
            "histogram": "histogram",
            "scatter": "scatter plot",
            "box": "box plot",
            "heatmap": "heatmap",
            "area": "area chart",
        }
        display_name = chart_names.get(chart_type, "chart")

        if chart_type == "pie":
            what_this_shows = f"The {display_name} shows the proportion of records across different {x_axis} categories."
        elif chart_type == "histogram":
            what_this_shows = f"The {display_name} displays the frequency distribution of {x_axis} across numeric bins."
        elif chart_type == "scatter":
            what_this_shows = f"The {display_name} illustrates the correlation and relationship between {x_axis} and {y_axis}."
        elif chart_type == "line":
            what_this_shows = f"The {display_name} tracks the trend of {y_axis} over {x_axis}."
        else:
            what_this_shows = f"The {display_name} compares {y_axis or 'counts'} across different {x_axis} categories."

        params_used = [
            f"Category: {x_axis}" if chart_type in {"bar", "pie", "box"} else f"X-axis: {x_axis}",
        ]
        if y_axis:
            params_used.append(f"Value: {y_axis}" if chart_type == "pie" else f"Y-axis: {y_axis}")
        if aggregation:
            params_used.append(f"Aggregation: {aggregation}")
        if filter_list:
            params_used.append(f"Filters: {', '.join(filter_list)}")
        if sorting != "none":
            params_used.append(f"Sorting: {sorting.capitalize()}")
        if limit:
            params_used.append(f"Limit: Top {limit}")

        # Grounded Key Insights
        key_insights = []
        if y_axis and y_axis in df.columns and pd.api.types.is_numeric_dtype(df[y_axis]) and len(df) > 0:
            max_idx = df[y_axis].idxmax()
            max_row = df.loc[max_idx]
            total_sum = df[y_axis].sum()

            if chart_type == "pie" and total_sum > 0:
                pct = (max_row[y_axis] / total_sum) * 100
                key_insights.append(f"{max_row[x_axis]} represents the largest share at {pct:.1f}% ({max_row[y_axis]}).")
            else:
                key_insights.append(f"{max_row[x_axis]} has the highest value with {max_row[y_axis]}.")

            if len(df) >= 2:
                min_idx = df[y_axis].idxmin()
                if min_idx != max_idx:
                    min_row = df.loc[min_idx]
                    key_insights.append(f"{min_row[x_axis]} has the lowest value with {min_row[y_axis]}.")

            key_insights.append(f"A total of {len(df)} distinct {x_axis} values are represented.")
        elif chart_type == "histogram" and x_axis in df.columns:
            valid_obs = int(df[x_axis].dropna().count())
            mean_val = df[x_axis].dropna().mean()
            key_insights.append(f"Histogram represents {valid_obs} total valid raw observations.")
            if not np.isnan(mean_val):
                key_insights.append(f"The average {x_axis} is {mean_val:.2f}.")
        else:
            key_insights.append(f"Displaying {len(df)} rows across {x_axis}.")

        explanation_payload = ExplanationDict({
            "what_this_shows": what_this_shows,
            "parameters_used": params_used,
            "key_insights": key_insights[:3],
        })

        title = f"{y_axis or 'Distribution'} by {x_axis}" if y_axis else f"Distribution of {x_axis}"
        if chart_type == "pie":
            title = f"Distribution: {x_axis}"

        return {
            "chart_type": chart_type,
            "title": title,
            "x_axis": x_axis,
            "y_axis": y_axis,
            "category": x_axis if chart_type in {"bar", "pie"} else None,
            "value": y_axis,
            "group_by": x_axis if chart_type in {"bar", "pie"} else None,
            "plotly_options": plotly_opts,
            "parameters": parameters,
            "explanation": explanation_payload,
        }

    def _render_plotly_figure(
        self,
        spec: dict[str, Any],
        df: pd.DataFrame,
        question: str,
    ) -> tuple[go.Figure, str]:
        """
        Construct the actual interactive Plotly figure based on the structured specification.
        """
        chart_type = spec["chart_type"]
        title = spec["title"]
        x_axis = spec["x_axis"]
        y_axis = spec.get("y_axis")
        opts = spec.get("plotly_options", {})

        palette = ["#6366f1", "#06b6d4", "#ec4899", "#8b5cf6", "#10b981", "#f59e0b", "#3b82f6", "#14b8a6"]

        fig = None

        if chart_type == "pie":
            hole = opts.get("hole", 0.0)
            fig = px.pie(
                df,
                names=x_axis,
                values=y_axis,
                hole=hole,
                title=title,
                color_discrete_sequence=palette,
                template="plotly_dark",
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")

        elif chart_type == "bar":
            orientation = opts.get("orientation", "vertical")
            if orientation == "horizontal":
                fig = px.bar(
                    df,
                    x=y_axis,
                    y=x_axis,
                    orientation="h",
                    title=title,
                    color_discrete_sequence=["#6366f1"],
                    template="plotly_dark",
                )
            else:
                fig = px.bar(
                    df,
                    x=x_axis,
                    y=y_axis,
                    title=title,
                    color_discrete_sequence=["#6366f1"],
                    template="plotly_dark",
                )

        elif chart_type == "line":
            fig = px.line(
                df,
                x=x_axis,
                y=y_axis,
                title=title,
                color_discrete_sequence=["#06b6d4"],
                template="plotly_dark",
            )

        elif chart_type == "histogram":
            # HISTOGRAM RULE: uses raw observations
            fig = px.histogram(
                df,
                x=x_axis,
                title=title,
                color_discrete_sequence=["#818cf8"],
                template="plotly_dark",
            )

        elif chart_type == "scatter":
            fig = px.scatter(
                df,
                x=x_axis,
                y=y_axis,
                title=title,
                color_discrete_sequence=["#22d3ee"],
                template="plotly_dark",
            )

        elif chart_type == "box":
            fig = px.box(
                df,
                x=x_axis,
                y=y_axis,
                title=title,
                color_discrete_sequence=["#a855f7"],
                template="plotly_dark",
            )

        elif chart_type == "area":
            fig = px.area(
                df,
                x=x_axis,
                y=y_axis,
                title=title,
                color_discrete_sequence=["#22d3ee"],
                template="plotly_dark",
            )

        elif chart_type == "heatmap":
            numeric_df = df.select_dtypes(include=["number"])
            if len(numeric_df.columns) >= 2:
                corr = numeric_df.corr()
                fig = px.imshow(
                    corr,
                    text_auto=True,
                    aspect="auto",
                    title=title or "Correlation Heatmap",
                    color_continuous_scale="RdBu_r",
                    template="plotly_dark",
                )
            else:
                fig = px.bar(df.head(15), x=cols_or_default(df, 0), y=cols_or_default(df, 1), template="plotly_dark")

        else:
            fig = px.bar(
                df.head(15),
                x=x_axis,
                y=y_axis,
                title=title,
                template="plotly_dark",
            )

        # Style chart layout for dark UI integration
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, -apple-system, sans-serif", color="#f1f5f9"),
            margin=dict(l=40, r=40, t=50, b=40),
            hovermode="closest",
        )

        return fig, title

    def _write_png_preview(
        self,
        df: pd.DataFrame,
        spec: dict[str, Any],
        path: str,
        title: str,
        question: str,
    ) -> None:
        """Create a high-quality static preview image mirroring the interactive Plotly chart."""
        fig, ax = plt.subplots(figsize=(10, 5.8))
        fig.patch.set_facecolor("#101426")
        ax.set_facecolor("#161b33")
        ax.tick_params(colors="#dbeafe")
        for spine in ax.spines.values():
            spine.set_color("#475569")

        palette = ["#6366f1", "#06b6d4", "#ec4899", "#8b5cf6", "#10b981", "#f59e0b", "#3b82f6", "#14b8a6"]
        chart_type = spec["chart_type"]
        x_col = spec["x_axis"]
        y_col = spec.get("y_axis")
        opts = spec.get("plotly_options", {})

        if chart_type == "pie":
            pie_values = df[y_col] if y_col and y_col in df.columns else df[x_col].value_counts().values
            pie_labels = df[x_col].astype(str) if y_col and y_col in df.columns else df[x_col].value_counts().index.astype(str)
            colors = palette[:len(pie_values)] if len(pie_values) <= len(palette) else None
            wedges, texts, autotexts = ax.pie(
                pie_values,
                labels=pie_labels,
                autopct="%1.1f%%",
                colors=colors,
                startangle=140,
                wedgeprops=dict(edgecolor="#161b33", linewidth=2),
            )
            for t in texts:
                t.set_color("#f8fafc")
                t.set_fontsize(10)
            for at in autotexts:
                at.set_color("#ffffff")
                at.set_fontsize(9)
                at.set_weight("bold")

        elif chart_type == "histogram":
            ax.hist(df[x_col].dropna(), bins=min(20, max(5, len(df) // 2)), color="#818cf8", edgecolor="#0f172a")
            ax.set_ylabel("Records", color="#dbeafe")
            ax.set_xlabel(x_col, color="#dbeafe")

        elif chart_type == "scatter" and y_col:
            ax.scatter(df[x_col], df[y_col], color="#22d3ee", alpha=0.8)
            ax.set_ylabel(y_col, color="#dbeafe")
            ax.set_xlabel(x_col, color="#dbeafe")

        elif chart_type in {"line", "area"} and y_col:
            if chart_type == "area":
                ax.fill_between(range(len(df)), df[y_col], color="#22d3ee", alpha=0.35)
            ax.plot(df[x_col].astype(str), df[y_col], color="#22d3ee", marker="o")
            ax.set_ylabel(y_col, color="#dbeafe")
            ax.set_xlabel(x_col, color="#dbeafe")

        elif chart_type == "box" and y_col:
            ax.boxplot([group[y_col].dropna() for _, group in df.groupby(x_col)], tick_labels=[str(name) for name, _ in df.groupby(x_col)])
            ax.set_ylabel(y_col, color="#dbeafe")
            ax.set_xlabel(x_col, color="#dbeafe")

        else:
            # Bar chart
            orientation = opts.get("orientation", "vertical")
            if orientation == "horizontal" and y_col:
                plotted = df.head(15)
                ax.barh(plotted[x_col].astype(str), plotted[y_col], color="#6366f1")
                ax.set_xlabel(y_col, color="#dbeafe")
                ax.set_ylabel(x_col, color="#dbeafe")
            elif y_col:
                plotted = df.head(15)
                ax.bar(plotted[x_col].astype(str), plotted[y_col], color="#6366f1")
                ax.set_ylabel(y_col, color="#dbeafe")
                ax.set_xlabel(x_col, color="#dbeafe")
            else:
                counts = df[x_col].astype(str).value_counts().head(15)
                ax.bar(counts.index, counts.values, color="#6366f1")
                ax.set_ylabel("Records", color="#dbeafe")
                ax.set_xlabel(x_col, color="#dbeafe")

        ax.set_title(title, color="#f8fafc", fontweight="bold", pad=14)
        if chart_type != "pie":
            ax.grid(axis="y", color="#334155", alpha=0.45)
            plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        fig.tight_layout()
        fig.savefig(path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _empty_visualization_result(reason: str | None = None) -> dict[str, Any]:
        """Return standardized failure specification when data is unavailable or insufficient."""
        return {
            "visualization_available": False,
            "reason": reason or "The required tabular data was not available for visualization.",
            "chart": None,
            "parameters": {},
            "explanation": ExplanationDict({
                "what_this_shows": "There is not enough data available to generate this visualization.",
                "parameters_used": [],
                "key_insights": [],
            }),
            "key_insights": [],
            "chart_paths": [],
            "chart_jsons": [],
            "titles": [],
        }


def cols_or_default(df: pd.DataFrame, idx: int) -> str | None:
    return df.columns[idx] if len(df.columns) > idx else None


# Singleton instance
visualization_agent = VisualizationAgent()
