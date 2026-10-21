"""
Visualization Agent.
Generates interactive Plotly visualizations and portable previews based on tabular data or statistical analysis results.
Strictly adheres to user intent, avoids inventing parameters or data, and provides clear, grounded explanations.
"""

import json
import logging
import os
import re
import uuid
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.io as pio
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.state import AgentState
from app.config import settings
from app.core.llm_provider import get_llm

logger = logging.getLogger(__name__)

VISUALIZATION_AGENT_PROMPT = """You are the Visualization Agent in AI-MADAC.

Your job is to understand the user's visualization request, use the actual data/result returned by the analysis agents, generate the most appropriate visualization, and provide a short explanation of the visualization.

You MUST use the user's current query and the CURRENT agent result/data.
Do NOT reuse data, columns, filters, or results from previous questions.

CHART SELECTION RULES:
- BAR CHART: Comparing categories, rankings, counts, aggregated values by category, Top-N results.
- LINE CHART: Trends over time, year/month/day progression, ordered time-series data.
- PIE CHART: Part-to-whole composition, small number of meaningful categories (2-10), values represent parts of a meaningful total.
- SCATTER CHART: Relationship/correlation between two numeric variables.

CRITICAL RULES:
1. Current query > previous queries.
2. Current data > previous data.
3. Never invent data, counts, percentages, or categories.
4. Only claim parameters that were actually used (do not invent filters or aggregations).
5. If data was already aggregated by source agent, use: Aggregation = "Already aggregated by source agent".
6. Keep explanations simple, natural, and concise.
7. Provide 1-3 key insights directly grounded in the data.
"""


class VisualizationAgent:
    """
    Visualization Agent responsible for chart type selection, parameter determination,
    visual rendering (Plotly & static PNG), explanation generation, and key insights.
    """

    def __init__(self):
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = get_llm(temperature=0.1)
        return self._llm

    def run(self, state: AgentState) -> dict[str, Any]:
        """
        Generate visualization from sql_result or analysis_result data.
        """
        sql_result = state.get("sql_result")
        analysis_result = state.get("analysis_result")
        question = state.get("question", "")
        structured_intent = state.get("structured_intent") or {}

        # Fallback to previous validated result only when explicit follow-up visualization
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
        chart_title = "Data Visualization"
        chart_type = "bar"
        is_grouped_comparison = False

        # Check if analysis_result provides comparison groups
        if analysis_result and analysis_result.get("analysis_type") == "relationship_impact_analysis":
            with_grp = analysis_result.get("group_with_event", {})
            without_grp = analysis_result.get("group_without_event", {})
            pred_col = analysis_result.get("predictor_column", "Factor")
            target_col = analysis_result.get("target_column", "Impact")

            df = pd.DataFrame([
                {"Group": f"With {pred_col}", f"Average {target_col}": with_grp.get("mean_target", 0)},
                {"Group": f"Without {pred_col}", f"Average {target_col}": without_grp.get("mean_target", 0)},
            ])
            chart_title = f"Comparison: Average {target_col} by {pred_col} Presence"
            chart_type = "bar"
            is_grouped_comparison = True

        elif sql_result and sql_result.get("rows"):
            rows = sql_result.get("rows", [])
            df = pd.DataFrame(rows)

        # STEP 2: Verify data availability
        if df.empty:
            logger.warning("Visualization Agent: No tabular data available to visualize.")
            failure_result = {
                "visualization_available": False,
                "reason": "The required tabular data was not available for visualization.",
                "chart": None,
                "explanation": "There is not enough data available to generate this visualization.",
                "key_insights": [],
                "chart_paths": [],
                "chart_jsons": [],
                "titles": [],
            }
            return {
                "visualization_result": failure_result,
                "agents_used": state.get("agents_used", []) + ["visualization_agent"],
            }

        # A single KPI (1 row, 1 value) cannot form a meaningful comparative chart.
        if len(df) == 1 and not is_grouped_comparison:
            logger.info("Visualization Agent: Skipping chart for single-value result.")
            failure_result = {
                "visualization_available": False,
                "reason": "A single summary KPI is better communicated directly in text or table rather than a chart.",
                "chart": None,
                "explanation": "There is not enough comparative data (only 1 data point) to generate a visualization.",
                "key_insights": [],
                "chart_paths": [],
                "chart_jsons": [],
                "titles": [],
            }
            return {
                "visualization_result": failure_result,
                "agents_used": state.get("agents_used", []) + ["visualization_agent"],
            }

        logger.info(f"Visualization Agent generating chart for {len(df)} rows...")

        try:
            fig = None
            if is_grouped_comparison:
                x_col = "Group"
                y_col = df.columns[1]
                fig = px.bar(
                    df,
                    x="Group",
                    y=y_col,
                    title=chart_title,
                    color="Group",
                    color_discrete_sequence=["#6366f1", "#06b6d4"],
                    template="plotly_dark",
                )
            else:
                chart_type, x_col, y_col = self._determine_chart_specs(df, question)

                # Format title
                if chart_type == "pie":
                    chart_title = f"Distribution: {x_col}"
                    fig = px.pie(
                        df,
                        names=x_col,
                        values=y_col,
                        title=chart_title,
                        template="plotly_dark",
                    )
                    fig.update_traces(textposition="inside", textinfo="percent+label")
                elif chart_type == "bar":
                    chart_title = f"{y_col or 'Count'} by {x_col}"
                    fig = px.bar(
                        df,
                        x=x_col,
                        y=y_col,
                        title=chart_title,
                        color_discrete_sequence=["#6366f1"],
                        template="plotly_dark",
                    )
                elif chart_type == "line":
                    chart_title = f"Trend: {y_col} over {x_col}"
                    fig = px.line(
                        df,
                        x=x_col,
                        y=y_col,
                        title=chart_title,
                        color_discrete_sequence=["#06b6d4"],
                        template="plotly_dark",
                    )
                elif chart_type == "scatter":
                    chart_title = f"Scatter Plot: {x_col} vs {y_col}"
                    fig = px.scatter(
                        df,
                        x=x_col,
                        y=y_col,
                        title=chart_title,
                        color_discrete_sequence=["#22d3ee"],
                        template="plotly_dark",
                    )
                elif chart_type == "histogram":
                    chart_title = f"Histogram Distribution of {x_col}"
                    fig = px.histogram(
                        df,
                        x=x_col,
                        title=chart_title,
                        color_discrete_sequence=["#818cf8"],
                        template="plotly_dark",
                    )
                elif chart_type == "area":
                    chart_title = f"{y_col} over {x_col}"
                    fig = px.area(df, x=x_col, y=y_col, title=chart_title, template="plotly_dark")
                elif chart_type == "box":
                    chart_title = f"Distribution of {y_col} by {x_col}"
                    fig = px.box(df, x=x_col, y=y_col, title=chart_title, template="plotly_dark")
                elif chart_type == "violin":
                    chart_title = f"Distribution of {y_col} by {x_col}"
                    fig = px.violin(df, x=x_col, y=y_col, box=True, points="outliers", title=chart_title, template="plotly_dark")
                else:
                    chart_title = f"Data Preview ({df.columns[0]})"
                    fig = px.bar(
                        df.head(15),
                        x=df.columns[0],
                        y=df.columns[1] if len(df.columns) > 1 else None,
                        title=chart_title,
                        template="plotly_dark",
                    )

            # Style chart layout for dark UI integration
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color="#f1f5f9"),
                margin=dict(l=40, r=40, t=50, b=40),
            )

            settings.ensure_directories()
            chart_paths = []
            chart_jsons = []
            chart_titles = []

            chart_id = uuid.uuid4().hex[:10]
            json_filename = f"chart_{chart_id}.json"
            png_filename = f"chart_{chart_id}.png"
            json_filepath = os.path.join(settings.CHARTS_DIR, json_filename)
            png_filepath = os.path.join(settings.CHARTS_DIR, png_filename)
            chart_json_str = pio.to_json(fig)

            with open(json_filepath, "w", encoding="utf-8") as f:
                f.write(chart_json_str)

            # Static high-res PNG preview for browser rendering without requiring client-side Plotly bundle
            self._write_png_preview(
                df=df,
                chart_type=chart_type,
                question=question,
                path=png_filepath,
                title=chart_title,
            )
            chart_paths.append(f"/static/charts/{png_filename}")
            chart_jsons.append(json.loads(chart_json_str))
            chart_titles.append(chart_title)

            # STEP 4: Determine parameters
            applied_filters = (sql_result or {}).get("filters", {})
            parameters = self._extract_parameters(
                df=df,
                chart_type=chart_type,
                x_col=x_col,
                y_col=y_col,
                structured_intent=structured_intent,
                filters=applied_filters,
                is_grouped_comparison=is_grouped_comparison,
            )

            # STEP 6: Simple, grounded explanation and key insights
            explanation = self._generate_explanation(
                df=df,
                chart_type=chart_type,
                x_col=x_col,
                y_col=y_col,
                parameters=parameters,
                chart_title=chart_title,
            )

            key_insights = self._generate_insights(
                df=df,
                chart_type=chart_type,
                x_col=x_col,
                y_col=y_col,
            )

            chart_spec = {
                "chart_type": chart_type,
                "title": chart_title,
                "description": f"Visualizes {x_col}" + (f" and {y_col}" if y_col else ""),
                "parameters": parameters,
                "data": df.head(50).to_dict(orient="records"),
            }

            viz_data = {
                "visualization_available": True,
                "chart": chart_spec,
                "explanation": explanation,
                "key_insights": key_insights,
                "chart_type": chart_type,
                "chart_path": chart_paths[0],
                "chart_paths": chart_paths,
                "chart_json": chart_jsons[0],
                "chart_jsons": chart_jsons,
                "titles": chart_titles,
            }

            agents_used = state.get("agents_used", []) + ["visualization_agent"]

            return {
                "visualization_result": viz_data,
                "agents_used": agents_used,
            }

        except Exception as e:
            logger.error(f"Visualization generation error: {e}", exc_info=True)
            failure_result = {
                "visualization_available": False,
                "reason": f"An error occurred while rendering the visualization: {str(e)}",
                "chart": None,
                "explanation": "There is not enough data available to generate this visualization.",
                "key_insights": [],
                "chart_paths": [],
                "chart_jsons": [],
                "titles": [],
            }
            return {
                "visualization_result": failure_result,
                "agents_used": state.get("agents_used", []) + ["visualization_agent"],
            }

    def _determine_chart_specs(self, df: pd.DataFrame, question: str) -> tuple[str, str, str | None]:
        """
        Determine appropriate chart type and axes according to the user's intent.
        """
        cols = df.columns.tolist()
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        q_lower = question.lower()

        # Check explicit user requests first
        if "area" in q_lower:
            x_col = cat_cols[0] if cat_cols else cols[0]
            y_col = num_cols[0] if num_cols else (cols[1] if len(cols) > 1 else None)
            return "area", x_col, y_col

        if "box plot" in q_lower or "boxplot" in q_lower:
            x_col = cat_cols[0] if cat_cols else cols[0]
            y_col = num_cols[0] if num_cols else None
            return "box", x_col, y_col

        if "violin" in q_lower:
            x_col = cat_cols[0] if cat_cols else cols[0]
            y_col = num_cols[0] if num_cols else None
            return "violin", x_col, y_col

        if "pie" in q_lower or "share" in q_lower or "proportion" in q_lower:
            x_col = cat_cols[0] if cat_cols else cols[0]
            y_col = num_cols[0] if num_cols else (cols[1] if len(cols) > 1 else None)
            return "pie", x_col, y_col

        if "histogram" in q_lower:
            x_col = num_cols[0] if num_cols else cols[0]
            return "histogram", x_col, None

        if "trend" in q_lower or "over time" in q_lower or "line" in q_lower:
            x_col = cols[0]
            y_col = num_cols[0] if num_cols else (cols[1] if len(cols) > 1 else None)
            return "line", x_col, y_col

        if "scatter" in q_lower or "relationship" in q_lower or "correlation" in q_lower:
            if len(num_cols) >= 2:
                return "scatter", num_cols[0], num_cols[1]

        # Automatic heuristic selection based on data characteristics
        if "distribution" in q_lower:
            # If discrete categorical breakdown with <= 10 categories, pie chart is ideal
            if cat_cols and len(df) <= 10:
                x_col = cat_cols[0]
                y_col = num_cols[0] if num_cols else (cols[1] if len(cols) > 1 else None)
                return "pie", x_col, y_col
            elif num_cols and len(num_cols) == 1:
                return "histogram", num_cols[0], None

        if cat_cols and num_cols:
            return "bar", cat_cols[0], num_cols[0]
        elif len(cols) >= 2:
            return "bar", cols[0], cols[1]

        return "bar", cols[0], None

    def _extract_parameters(
        self,
        df: pd.DataFrame,
        chart_type: str,
        x_col: str,
        y_col: str | None,
        structured_intent: dict[str, Any],
        filters: dict[str, Any],
        is_grouped_comparison: bool,
    ) -> dict[str, Any]:
        """
        Extract only parameters that were actually used. Never claim unused filters or aggregations.
        """
        chart_names = {
            "bar": "Bar chart",
            "line": "Line chart",
            "pie": "Pie chart",
            "scatter": "Scatter plot",
            "histogram": "Histogram",
            "box": "Box plot",
            "violin": "Violin plot",
            "area": "Area chart",
        }
        params: dict[str, Any] = {
            "chart_type": chart_names.get(chart_type, chart_type.capitalize() + " chart"),
        }

        if chart_type == "pie":
            params["category"] = x_col
            if y_col:
                params["value"] = y_col
        else:
            params["x_axis"] = x_col
            if y_col:
                params["y_axis"] = y_col

        # Group by
        if is_grouped_comparison or chart_type in {"bar", "pie", "line"}:
            params["group_by"] = x_col

        # Aggregation: reflect true source aggregation without inventing
        if is_grouped_comparison:
            params["aggregation"] = "Average"
        elif y_col and any(token in y_col.lower() for token in ("count", "total", "average", "mean", "sum", "percentage")):
            params["aggregation"] = "Already aggregated by source agent"
        elif structured_intent.get("aggregation"):
            params["aggregation"] = structured_intent.get("aggregation")

        # Filters: only include if actually present and non-empty
        if filters and isinstance(filters, dict):
            filter_clauses = [f"{col} = '{val}'" for col, val in filters.items()]
            if filter_clauses:
                params["filters"] = ", ".join(filter_clauses)

        # Sort order: only if discernible
        if y_col and y_col in df.columns and pd.api.types.is_numeric_dtype(df[y_col]):
            if len(df) > 1:
                if df[y_col].is_monotonic_decreasing:
                    params["sort"] = "Descending"
                elif df[y_col].is_monotonic_increasing:
                    params["sort"] = "Ascending"

        return params

    def _generate_explanation(
        self,
        df: pd.DataFrame,
        chart_type: str,
        x_col: str,
        y_col: str | None,
        parameters: dict[str, Any],
        chart_title: str,
    ) -> str:
        """
        Generate a clear, human-understandable explanation answering:
        1. What does this chart show?
        2. Which parameters/data were used?
        3. What aggregation was applied?
        4. What filters were applied, if any?
        5. What are the important values shown?
        """
        chart_display = parameters.get("chart_type", "chart").lower()

        lines = []
        if chart_type == "pie":
            lines.append(f"The {chart_display} shows the distribution of records across different {x_col.lower()} categories.")
        elif chart_type == "line":
            lines.append(f"The {chart_display} shows the progression of {y_col} over {x_col}.")
        elif chart_type == "scatter":
            lines.append(f"The {chart_display} illustrates the relationship between {x_col} and {y_col}.")
        else:
            lines.append(f"The {chart_display} displays {y_col or 'counts'} grouped by {x_col}.")

        lines.append("\nParameters used:")
        if "category" in parameters:
            lines.append(f"- Category: {parameters['category']}")
        if "x_axis" in parameters:
            lines.append(f"- X-axis: {parameters['x_axis']}")
        if "y_axis" in parameters:
            lines.append(f"- Y-axis: {parameters['y_axis']}")
        if "value" in parameters:
            lines.append(f"- Value: {parameters['value']}")
        if "aggregation" in parameters:
            lines.append(f"- Aggregation: {parameters['aggregation']}")
        if "filters" in parameters:
            lines.append(f"- Filters: {parameters['filters']}")
        if "sort" in parameters:
            lines.append(f"- Sort: {parameters['sort']}")

        # Highlight important values
        if y_col and y_col in df.columns and len(df) > 0:
            top_row = df.iloc[0]
            cat_val = top_row[x_col]
            metric_val = top_row[y_col]
            if chart_type == "pie" and pd.api.types.is_numeric_dtype(df[y_col]):
                total = df[y_col].sum()
                pct = (metric_val / total * 100) if total > 0 else 0
                lines.append(f"\nThe chart shows that {cat_val} has the highest share with {metric_val} ({pct:.1f}%).")
            else:
                lines.append(f"\nThe chart highlights that {cat_val} leads with {metric_val}.")

        return "\n".join(lines)

    def _generate_insights(
        self,
        df: pd.DataFrame,
        chart_type: str,
        x_col: str,
        y_col: str | None,
    ) -> list[str]:
        """
        Provide 1–3 short insights directly determined from actual values in the data.
        """
        insights = []
        if df.empty:
            return insights

        if y_col and y_col in df.columns and pd.api.types.is_numeric_dtype(df[y_col]):
            # Highest / leading category
            max_idx = df[y_col].idxmax()
            max_row = df.loc[max_idx]
            total_val = df[y_col].sum()

            if chart_type == "pie" and total_val > 0:
                pct = (max_row[y_col] / total_val) * 100
                insights.append(f"{max_row[x_col]} represents the largest share at {pct:.1f}% ({max_row[y_col]}).")
            else:
                insights.append(f"{max_row[x_col]} has the highest value with {max_row[y_col]}.")

            # Second highest or lowest if multiple rows exist
            if len(df) >= 2:
                min_idx = df[y_col].idxmin()
                if min_idx != max_idx:
                    min_row = df.loc[min_idx]
                    insights.append(f"{min_row[x_col]} has the lowest value with {min_row[y_col]}.")

            # Category count
            insights.append(f"A total of {len(df)} distinct {x_col.lower()} categories are represented.")
        else:
            insights.append(f"{len(df)} records are displayed across {x_col}.")

        return insights[:3]

    def _write_png_preview(
        self,
        df: pd.DataFrame,
        chart_type: str,
        question: str,
        path: str,
        title: str,
        distribution: bool = False,
    ) -> None:
        """Create a high-quality portable report image that mirrors the interactive chart."""
        fig, ax = plt.subplots(figsize=(10, 5.8))
        fig.patch.set_facecolor("#101426")
        ax.set_facecolor("#161b33")
        ax.tick_params(colors="#dbeafe")
        for spine in ax.spines.values():
            spine.set_color("#475569")

        palette = ["#6366f1", "#06b6d4", "#ec4899", "#8b5cf6", "#10b981", "#f59e0b", "#3b82f6", "#14b8a6"]

        if distribution:
            numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
            column = numeric_columns[0]
            ax.hist(df[column].dropna(), bins=min(20, max(5, len(df) // 2)), color="#22d3ee", edgecolor="#0f172a")
            ax.set_xlabel(column, color="#dbeafe")
            ax.set_ylabel("Records", color="#dbeafe")
        elif "Group" in df.columns:
            value_column = df.columns[1]
            ax.bar(df["Group"].astype(str), df[value_column], color=["#6366f1", "#06b6d4"])
            ax.set_ylabel(value_column, color="#dbeafe")
        else:
            selected_type, x_column, y_column = self._determine_chart_specs(df, question)

            if selected_type == "pie":
                pie_values = df[y_column] if y_column and y_column in df.columns else df[x_column].value_counts().values
                pie_labels = df[x_column].astype(str) if y_column and y_column in df.columns else df[x_column].value_counts().index.astype(str)
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
            elif selected_type == "histogram":
                ax.hist(df[x_column].dropna(), bins=min(20, max(5, len(df) // 2)), color="#818cf8", edgecolor="#0f172a")
                ax.set_ylabel("Records", color="#dbeafe")
            elif selected_type == "scatter" and y_column:
                ax.scatter(df[x_column], df[y_column], color="#22d3ee", alpha=0.8)
                ax.set_ylabel(y_column, color="#dbeafe")
            elif selected_type in {"line", "area"} and y_column:
                if selected_type == "area":
                    ax.fill_between(range(len(df)), df[y_column], color="#22d3ee", alpha=0.35)
                ax.plot(df[x_column].astype(str), df[y_column], color="#22d3ee", marker="o")
                ax.set_ylabel(y_column, color="#dbeafe")
            elif selected_type == "box" and y_column:
                ax.boxplot([group[y_column].dropna() for _, group in df.groupby(x_column)], tick_labels=[str(name) for name, _ in df.groupby(x_column)])
                ax.set_ylabel(y_column, color="#dbeafe")
            elif selected_type == "violin" and y_column:
                groups = [group[y_column].dropna() for _, group in df.groupby(x_column)]
                ax.violinplot(groups, showmedians=True)
                ax.set_xticks(range(1, len(groups) + 1), [str(name) for name, _ in df.groupby(x_column)])
                ax.set_ylabel(y_column, color="#dbeafe")
            elif y_column:
                plotted = df.head(15)
                ax.bar(plotted[x_column].astype(str), plotted[y_column], color="#6366f1")
                ax.set_ylabel(y_column, color="#dbeafe")
            else:
                counts = df[x_column].astype(str).value_counts().head(15)
                ax.bar(counts.index, counts.values, color="#6366f1")
                ax.set_ylabel("Records", color="#dbeafe")

            if selected_type != "pie":
                ax.set_xlabel(x_column, color="#dbeafe")

        ax.set_title(title, color="#f8fafc", fontweight="bold", pad=14)
        if chart_type != "pie":
            ax.grid(axis="y", color="#334155", alpha=0.45)
            plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        fig.tight_layout()
        fig.savefig(path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)


# Singleton instance
visualization_agent = VisualizationAgent()
