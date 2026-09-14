"""
Visualization Agent.
Generates interactive Plotly visualizations based on retrieved tabular data or statistical analysis results.
"""

import json
import logging
import os
import uuid
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.io as pio

from app.agents.state import AgentState
from app.config import settings

logger = logging.getLogger(__name__)


class VisualizationAgent:
    """
    Visualization Agent responsible for chart type selection and Plotly chart generation.
    """

    def run(self, state: AgentState) -> dict[str, Any]:
        """
        Generate visualization from sql_result or analysis_result data.
        """
        sql_result = state.get("sql_result")
        analysis_result = state.get("analysis_result")
        question = state.get("question", "")

        df = pd.DataFrame()
        chart_title = "Data Visualization"
        chart_type = "bar"

        # Check if analysis_result provides comparison groups (e.g. event vs no event)
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

        elif sql_result and sql_result.get("rows"):
            rows = sql_result.get("rows", [])
            df = pd.DataFrame(rows)

        if df.empty:
            logger.warning("Visualization Agent: No data available to visualize.")
            return {"visualization_result": None}

        # A single KPI is best communicated directly in the answer and table.
        # Rendering it as a one-bar chart adds no information.
        if len(df) == 1 and "Group" not in df.columns:
            logger.info("Visualization Agent: Skipping chart for single-value result.")
            return {"visualization_result": None}

        logger.info(f"Visualization Agent generating chart for {len(df)} rows...")

        try:
            fig = None
            if "Group" in df.columns:
                fig = px.bar(
                    df,
                    x="Group",
                    y=df.columns[1],
                    title=chart_title,
                    color="Group",
                    color_discrete_sequence=["#6366f1", "#06b6d4"],
                    template="plotly_dark",
                )
            else:
                chart_type, x_col, y_col = self._determine_chart_specs(df, question)

                if chart_type == "bar":
                    fig = px.bar(
                        df,
                        x=x_col,
                        y=y_col,
                        title=f"{y_col or 'Count'} by {x_col}",
                        color_discrete_sequence=["#6366f1"],
                        template="plotly_dark",
                    )
                elif chart_type == "line":
                    fig = px.line(
                        df,
                        x=x_col,
                        y=y_col,
                        title=f"Line Chart: {y_col} over {x_col}",
                        color_discrete_sequence=["#06b6d4"],
                        template="plotly_dark",
                    )
                elif chart_type == "pie":
                    fig = px.pie(
                        df,
                        names=x_col,
                        values=y_col,
                        title=f"Distribution: {x_col}",
                        template="plotly_dark",
                    )
                elif chart_type == "histogram":
                    fig = px.histogram(
                        df,
                        x=x_col,
                        title=f"Histogram Distribution of {x_col}",
                        color_discrete_sequence=["#818cf8"],
                        template="plotly_dark",
                    )
                elif chart_type == "scatter":
                    fig = px.scatter(
                        df,
                        x=x_col,
                        y=y_col,
                        title=f"Scatter Plot: {x_col} vs {y_col}",
                        color_discrete_sequence=["#22d3ee"],
                        template="plotly_dark",
                    )
                elif chart_type == "area":
                    fig = px.area(df, x=x_col, y=y_col, title=f"{y_col} over {x_col}", template="plotly_dark")
                elif chart_type == "box":
                    fig = px.box(df, x=x_col, y=y_col, title=f"Distribution of {y_col} by {x_col}", template="plotly_dark")
                elif chart_type == "violin":
                    fig = px.violin(df, x=x_col, y=y_col, box=True, points="outliers", title=f"Distribution of {y_col} by {x_col}", template="plotly_dark")
                else:
                    fig = px.bar(
                        df.head(15),
                        x=df.columns[0],
                        y=df.columns[1] if len(df.columns) > 1 else None,
                        title=f"Data Preview ({df.columns[0]})",
                        template="plotly_dark",
                    )

            # Style chart layout for dark UI integration
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif", color="#f1f5f9"),
                margin=dict(l=40, r=40, t=50, b=40),
            )

            # Produce exactly the requested view. An automatic histogram of
            # aggregate values can change the meaning of a ranking chart.
            figures = [fig]

            settings.ensure_directories()
            chart_paths = []
            chart_jsons = []
            chart_titles = []

            for index, chart_figure in enumerate(figures):
                chart_id = uuid.uuid4().hex[:10]
                json_filename = f"chart_{chart_id}.json"
                png_filename = f"chart_{chart_id}.png"
                json_filepath = os.path.join(settings.CHARTS_DIR, json_filename)
                png_filepath = os.path.join(settings.CHARTS_DIR, png_filename)
                chart_json_str = pio.to_json(chart_figure)

                with open(json_filepath, "w", encoding="utf-8") as f:
                    f.write(chart_json_str)

                # PNG makes the visual available in the browser, PDFs, and
                # external report downloads without requiring Plotly in Next.js.
                # Matplotlib avoids requiring a browser process in the API
                # server (Plotly's JSON is still retained for future reuse).
                self._write_png_preview(
                    df=df,
                    chart_type=chart_type,
                    question=question,
                    path=png_filepath,
                    title=chart_figure.layout.title.text or "Data Visualization",
                    distribution=index > 0,
                )
                chart_paths.append(f"/static/charts/{png_filename}")
                chart_jsons.append(json.loads(chart_json_str))
                chart_titles.append(
                    chart_figure.layout.title.text if chart_figure.layout.title else "Data Visualization"
                )

            logger.info("Saved %s chart(s) for the report", len(chart_paths))

            viz_data = {
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
            logger.error(f"Visualization generation error: {e}")
            return {"visualization_result": None}

    def _write_png_preview(
        self,
        df: pd.DataFrame,
        chart_type: str,
        question: str,
        path: str,
        title: str,
        distribution: bool = False,
    ) -> None:
        """Create a portable report image that mirrors the interactive chart."""
        fig, ax = plt.subplots(figsize=(11, 6.2))
        fig.patch.set_facecolor("#101426")
        ax.set_facecolor("#161b33")
        ax.tick_params(colors="#dbeafe")
        for spine in ax.spines.values():
            spine.set_color("#475569")

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
            if selected_type == "histogram":
                ax.hist(df[x_column].dropna(), bins=min(20, max(5, len(df) // 2)), color="#818cf8", edgecolor="#0f172a")
                ax.set_ylabel("Records", color="#dbeafe")
            elif selected_type == "scatter" and y_column:
                ax.scatter(df[x_column], df[y_column], color="#22d3ee", alpha=0.8)
                ax.set_ylabel(y_column, color="#dbeafe")
            elif selected_type == "pie" and y_column:
                ax.pie(df[y_column], labels=df[x_column].astype(str), autopct="%1.1f%%")
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
            ax.set_xlabel(x_column, color="#dbeafe")

        ax.set_title(title, color="#f8fafc", fontweight="bold", pad=14)
        ax.grid(axis="y", color="#334155", alpha=0.45)
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        fig.tight_layout()
        fig.savefig(path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
        plt.close(fig)

    def _determine_chart_specs(self, df: pd.DataFrame, question: str) -> tuple[str, str, str | None]:
        """
        Determine appropriate chart type and axes.
        """
        cols = df.columns.tolist()
        num_cols = df.select_dtypes(include=["number"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        q_lower = question.lower()

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
            y_col = num_cols[0] if num_cols else None
            return "pie", x_col, y_col

        if "distribution" in q_lower or "histogram" in q_lower:
            x_col = num_cols[0] if num_cols else cols[0]
            return "histogram", x_col, None

        if "trend" in q_lower or "over time" in q_lower or "line" in q_lower:
            x_col = cols[0]
            y_col = num_cols[0] if num_cols else (cols[1] if len(cols) > 1 else None)
            return "line", x_col, y_col

        if "scatter" in q_lower or "relationship" in q_lower:
            if len(num_cols) >= 2:
                return "scatter", num_cols[0], num_cols[1]

        if cat_cols and num_cols:
            return "bar", cat_cols[0], num_cols[0]
        elif len(cols) >= 2:
            return "bar", cols[0], cols[1]

        return "bar", cols[0], None


# Singleton instance
visualization_agent = VisualizationAgent()
