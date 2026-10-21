"""
Plotly Figure Generation with Centralized AI-MADAC Theme.

Enforces:
1. Centralized AI-MADAC color system (dark theme #101426, consistent categorical palette).
2. Stable category colors across charts (e.g. Male/Female).
3. Professional layouts, readable labels, hover tooltips, responsive configurations.
4. Generates both interactive Plotly JSON figures and high-resolution PNG previews.
"""

from __future__ import annotations

import os
import uuid
import json
import logging
from typing import Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from app.config import settings

logger = logging.getLogger(__name__)

# Centralized AI-MADAC Theme Constants
AI_MADAC_PALETTE = [
    "#6366F1",  # Indigo (Primary Brand)
    "#8B5CF6",  # Purple
    "#10B981",  # Emerald Green
    "#F59E0B",  # Amber
    "#EF4444",  # Coral Red
    "#06B6D4",  # Cyan
    "#EC4899",  # Pink
    "#14B8A6",  # Teal
    "#3B82F6",  # Blue
    "#F97316",  # Orange
]

# Consistent categorical mappings
STABLE_CATEGORY_COLORS = {
    "Male": "#6366F1",
    "male": "#6366F1",
    "Female": "#EC4899",
    "female": "#EC4899",
    "Other": "#94A3B8",
}


class PlotlyRenderer:
    """Renders validated and prepared data into interactive Plotly figures."""

    @classmethod
    def render(
        cls,
        df: pd.DataFrame,
        spec: dict[str, Any],
        question: str,
    ) -> tuple[go.Figure, str, str, str]:
        """
        Build Plotly figure, apply theme, save JSON and PNG artifacts.
        Returns: (fig, title, chart_path, chart_json_path)
        """
        settings.ensure_directories()
        chart = spec.get("chart") or {}
        chart_type = chart.get("type", "bar")
        mode = chart.get("mode")
        orientation = chart.get("orientation", "vertical")
        fields = spec.get("fields") or {}
        x_col = fields.get("x")
        y_col = fields.get("y")
        color_col = fields.get("color")
        title = spec.get("title") or f"Analysis for: {question}"

        # Select columns in DataFrame
        cols = df.columns.tolist()

        # 1. Grouped / Stacked Bar Chart
        if color_col and color_col in cols and x_col and x_col in cols and "count" in cols:
            barmode = "stack" if mode == "stacked" else "group"
            fig = px.bar(
                df,
                x=x_col,
                y="count",
                color=color_col,
                barmode=barmode,
                color_discrete_map=STABLE_CATEGORY_COLORS,
                color_discrete_sequence=AI_MADAC_PALETTE,
                title=title,
            )
            fig.update_layout(
                xaxis_title=x_col.replace("_", " ").title(),
                yaxis_title="Number of Investors",
                legend_title_text=color_col.replace("_", " ").title(),
            )

        # 2. Histogram / Distribution
        elif chart_type == "histogram" and x_col and x_col in cols:
            binning = spec.get("binning") or {}
            bin_size = binning.get("bin_size")
            fig = px.histogram(
                df,
                x=x_col,
                title=title,
                color_discrete_sequence=[AI_MADAC_PALETTE[0]],
            )
            if bin_size:
                fig.update_traces(xbins=dict(size=bin_size))
            fig.update_layout(
                xaxis_title=x_col.replace("_", " ").title(),
                yaxis_title="Number of Investors (Raw Observations)",
                bargap=0.08,
            )

        # 3. Scatter Plot
        elif chart_type == "scatter" and x_col and y_col and x_col in cols and y_col in cols:
            fig = px.scatter(
                df,
                x=x_col,
                y=y_col,
                title=title,
                color_discrete_sequence=[AI_MADAC_PALETTE[5]],
            )
            fig.update_traces(marker=dict(size=9, opacity=0.85, line=dict(width=1, color="#1e293b")))
            fig.update_layout(
                xaxis_title=x_col.replace("_", " ").title(),
                yaxis_title=y_col.replace("_", " ").title(),
            )

        # 4. Pie / Donut
        elif chart_type == "pie" and x_col and x_col in cols:
            val_col = cols[1] if len(cols) > 1 else "investor_count"
            is_donut = (mode == "donut" or "donut" in str(chart.get("type", "")).lower())
            fig = px.pie(
                df,
                names=x_col,
                values=val_col if val_col in df.columns else None,
                title=title,
                hole=0.4 if is_donut else 0.0,
                color_discrete_map=STABLE_CATEGORY_COLORS,
                color_discrete_sequence=AI_MADAC_PALETTE,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")

        # 5. Standard Bar (Vertical or Horizontal)
        elif x_col and x_col in cols:
            val_col = cols[1] if len(cols) > 1 else cols[0]
            # Auto switch to horizontal if many categories
            if len(df) > 7 or orientation == "horizontal":
                fig = px.bar(
                    df,
                    x=val_col,
                    y=x_col,
                    orientation="h",
                    title=title,
                    color_discrete_sequence=[AI_MADAC_PALETTE[0]],
                )
                fig.update_layout(
                    xaxis_title=val_col.replace("_", " ").title(),
                    yaxis_title=x_col.replace("_", " ").title(),
                )
            else:
                fig = px.bar(
                    df,
                    x=x_col,
                    y=val_col,
                    title=title,
                    color_discrete_sequence=[AI_MADAC_PALETTE[0]],
                )
                fig.update_layout(
                    xaxis_title=x_col.replace("_", " ").title(),
                    yaxis_title=val_col.replace("_", " ").title(),
                )

        # 6. Fallback
        else:
            fig = go.Figure()
            fig.update_layout(title=title)

        # Centralized AI-MADAC Theme Polish
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#101426",
            plot_bgcolor="#101426",
            font=dict(color="#f1f5f9", family="Inter, system-ui, sans-serif", size=12),
            title_font=dict(size=15, color="#f8fafc", family="Inter, system-ui, sans-serif"),
            margin=dict(l=40, r=40, t=55, b=45),
            xaxis=dict(
                gridcolor="#1e293b",
                zerolinecolor="#334155",
                tickfont=dict(color="#94a3b8"),
                showgrid=True,
            ),
            yaxis=dict(
                gridcolor="#1e293b",
                zerolinecolor="#334155",
                tickfont=dict(color="#94a3b8"),
                showgrid=True,
            ),
        )

        # Persist artifacts (JSON figure and PNG preview)
        chart_id = uuid.uuid4().hex[:10]
        json_filename = f"chart_{chart_id}.json"
        png_filename = f"chart_{chart_id}.png"
        json_filepath = os.path.join(settings.CHARTS_DIR, json_filename)
        png_filepath = os.path.join(settings.CHARTS_DIR, png_filename)

        chart_json_str = pio.to_json(fig)
        with open(json_filepath, "w", encoding="utf-8") as f:
            f.write(chart_json_str)

        # Persist static PNG preview using matplotlib (instant & non-blocking)
        cls._render_static_figure(df, spec, png_filepath, title)

        chart_path = f"/static/charts/{png_filename}"
        return fig, title, chart_path, json_filepath

    @staticmethod
    def _render_static_figure(df: pd.DataFrame, spec: dict[str, Any], path: str, title: str) -> None:
        """Create high-res static PNG preview using matplotlib adhering to dark theme."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8.5, 4.8), facecolor="#101426")
            ax.set_facecolor("#101426")

            fields = spec.get("fields") or {}
            x_col = fields.get("x")
            y_col = fields.get("y")
            color_col = fields.get("color")
            chart_type = (spec.get("chart") or {}).get("type", "bar")

            if color_col and color_col in df.columns and x_col and x_col in df.columns and "count" in df.columns:
                # Grouped bar
                pivot = df.pivot(index=x_col, columns=color_col, values="count").fillna(0)
                colors = ["#6366F1", "#EC4899", "#10B981", "#F59E0B"]
                pivot.plot(kind="bar", ax=ax, color=colors[:len(pivot.columns)], edgecolor="#1e293b")
                ax.set_ylabel("Number of Investors", color="#dbeafe")
                ax.legend(title=color_col.replace("_", " ").title(), facecolor="#161b33", labelcolor="#f8fafc")
            elif chart_type == "histogram" and x_col and x_col in df.columns:
                ax.hist(df[x_col].dropna(), bins=min(20, max(5, len(df) // 2)), color="#6366F1", edgecolor="#0f172a")
                ax.set_ylabel("Number of Investors", color="#dbeafe")
            elif chart_type == "scatter" and x_col and y_col and x_col in df.columns and y_col in df.columns:
                ax.scatter(df[x_col], df[y_col], color="#06B6D4", alpha=0.85, edgecolors="#1e293b")
                ax.set_ylabel(y_col.replace("_", " ").title(), color="#dbeafe")
            elif chart_type == "pie" and x_col and x_col in df.columns:
                val_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                values = df[val_col] if val_col != x_col else df[x_col].value_counts().values
                labels = df[x_col].astype(str) if val_col != x_col else df[x_col].value_counts().index.astype(str)
                wedges, texts, autotexts = ax.pie(
                    values, labels=labels, autopct="%1.1f%%",
                    colors=["#6366F1", "#EC4899", "#10B981", "#F59E0B", "#06B6D4"][:len(values)],
                    startangle=140, wedgeprops=dict(edgecolor="#161b33", linewidth=2),
                )
                for t in texts:
                    t.set_color("#f8fafc")
                for at in autotexts:
                    at.set_color("#ffffff")
            elif x_col and x_col in df.columns:
                val_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                plotted = df.head(15)
                ax.bar(plotted[x_col].astype(str), plotted[val_col], color="#6366F1", edgecolor="#1e293b")
                ax.set_ylabel(val_col.replace("_", " ").title(), color="#dbeafe")

            ax.set_title(title, color="#f8fafc", fontweight="bold", pad=12)
            if chart_type != "pie" and x_col:
                ax.set_xlabel(x_col.replace("_", " ").title(), color="#dbeafe")
                ax.tick_params(colors="#94a3b8", labelsize=9)
                plt.xticks(rotation=20, ha="right")
                ax.grid(color="#1e293b", linestyle="--", linewidth=0.6, alpha=0.7)
                for spine in ax.spines.values():
                    spine.set_color("#334155")

            fig.savefig(path, dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight")
            plt.close(fig)
        except Exception as err:
            logger.error(f"Matplotlib static preview generation failed: {err}")
