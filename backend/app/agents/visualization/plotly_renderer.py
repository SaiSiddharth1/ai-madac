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
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from app.config import settings
from app.agents.visualization.data_preparer import ensure_1d

logger = logging.getLogger(__name__)

# Centralized AI-MADAC Theme Constants
POWER_BI_PALETTE = [
    "#6366F1",  # Indigo
    "#8B5CF6",  # Purple
    "#06B6D4",  # Cyan
    "#10B981",  # Emerald Green
    "#F59E0B",  # Amber
    "#EF4444",  # Coral Red
    "#EC4899",  # Pink
    "#14B8A6",  # Teal
    "#3B82F6",  # Blue
    "#84CC16",  # Lime Green
]

AI_MADAC_PALETTE = POWER_BI_PALETTE

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
        chart_type = chart.get("type") or spec.get("suggested_chart_type") or "bar"
        mode = chart.get("mode")
        orientation = chart.get("orientation", "vertical")
        fields = spec.get("fields") or {}
        x_col = fields.get("x") or spec.get("x_axis")
        y_col = fields.get("y") or spec.get("y_axis")
        color_col = fields.get("color") or spec.get("color_by")
        title = spec.get("title") or spec.get("chart_title") or f"Analysis for: {question}"

        cols = df.columns.tolist()

        # If x_col or y_col is missing from cols, resolve case-insensitively
        col_map = {c.lower(): c for c in cols}
        if x_col and x_col.lower() in col_map:
            x_col = col_map[x_col.lower()]
        if y_col and isinstance(y_col, str) and y_col.lower() in col_map:
            y_col = col_map[y_col.lower()]
        if color_col and color_col.lower() in col_map:
            color_col = col_map[color_col.lower()]

        # 1. Clustered Multi-Measure Bar Chart (e.g. Sales and Profit side by side)
        is_multi_measure = isinstance(y_col, list) and len(y_col) >= 2
        if is_multi_measure and x_col and x_col in cols:
            measure_cols = []
            for m in y_col:
                if m in cols:
                    measure_cols.append(m)
                else:
                    alias = next((c for c in cols if c.lower() == f"total_{m.lower()}" or c.lower() == m.lower() or m.lower() in c.lower()), None)
                    if alias:
                        measure_cols.append(alias)
            fig = go.Figure()
            x_1d = ensure_1d(df[x_col], str(x_col))
            for idx, m in enumerate(measure_cols):
                y_1d = ensure_1d(df[m], str(m))
                fig.add_trace(go.Bar(
                    name=m.replace("_", " ").title(),
                    x=x_1d,
                    y=y_1d,
                    marker_color=POWER_BI_PALETTE[idx % len(POWER_BI_PALETTE)],
                ))
            fig.update_layout(
                barmode="group",
                title=title,
                xaxis_title=x_col.replace("_", " ").title(),
                yaxis_title="Values",
                legend_title_text="Measures",
            )

        # 2. Grouped / Stacked Bar Chart with explicit categorical color column
        elif color_col and color_col in cols and x_col and x_col in cols and len(cols) >= 3 and chart_type not in ("line", "trend", "multi_line", "scatter", "relationship"):
            val_col = next((c for c in cols if c not in (x_col, color_col)), "count")
            barmode = "stack" if mode == "stacked" else "group"
            clean_grouped_df = pd.DataFrame({
                x_col: ensure_1d(df[x_col], str(x_col)),
                val_col: ensure_1d(df[val_col], str(val_col)),
                color_col: ensure_1d(df[color_col], str(color_col)),
            })
            fig = px.bar(
                clean_grouped_df,
                x=x_col,
                y=val_col,
                color=color_col,
                barmode=barmode,
                color_discrete_map=STABLE_CATEGORY_COLORS,
                color_discrete_sequence=POWER_BI_PALETTE,
                title=title,
            )
            fig.update_layout(
                xaxis_title=x_col.replace("_", " ").title(),
                yaxis_title=val_col.replace("_", " ").title(),
                legend_title_text=color_col.replace("_", " ").title(),
            )

        # 3. Histogram / Distribution (Raw observations)
        elif chart_type in ("histogram", "distribution") and (x_col in cols or len(cols) >= 1):
            target_x = x_col if x_col in cols else cols[0]
            x_1d = ensure_1d(df[target_x], str(target_x)).dropna()
            clean_hist_df = pd.DataFrame({target_x: x_1d})

            bin_size = (
                spec.get("bin_width")
                or (spec.get("binning") or {}).get("bin_size")
                or (spec.get("binning") or {}).get("size")
                or (spec.get("bins") or {}).get("size")
            )
            if not bin_size and not clean_hist_df.empty:
                min_v = float(clean_hist_df[target_x].min())
                max_v = float(clean_hist_df[target_x].max())
                val_range = max_v - min_v
                n_obs = len(clean_hist_df)
                if val_range == 0 or clean_hist_df[target_x].nunique() <= 5:
                    bin_size = 1.0
                else:
                    iqr = float(clean_hist_df[target_x].quantile(0.75) - clean_hist_df[target_x].quantile(0.25))
                    h = (2.0 * iqr / (n_obs ** (1.0 / 3.0))) if iqr > 0 else (val_range / max(1.0 + np.log2(n_obs), 1.0))
                    candidates = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 25.0, 50.0, 100.0, 200.0, 500.0, 1000.0]
                    bin_size = min(candidates, key=lambda c: abs(c - h))

            fig = px.histogram(
                clean_hist_df,
                x=target_x,
                title=title,
                color_discrete_sequence=[POWER_BI_PALETTE[0]],
            )
            if bin_size and not clean_hist_df.empty:
                try:
                    min_v = float(clean_hist_df[target_x].min())
                    max_v = float(clean_hist_df[target_x].max())
                    start_edge = float(np.floor(min_v / bin_size) * bin_size)
                    end_edge = float(np.ceil(max_v / bin_size) * bin_size)
                    fig.update_traces(
                        xbins=dict(start=start_edge, end=end_edge + bin_size, size=bin_size),
                        marker=dict(color=POWER_BI_PALETTE[0], line=dict(width=1, color="#1e293b")),
                        hovertemplate="<b>Bin Range:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>",
                    )
                except Exception:
                    fig.update_traces(
                        marker=dict(color=POWER_BI_PALETTE[0], line=dict(width=1, color="#1e293b")),
                        hovertemplate="<b>Bin Range:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>",
                    )
            else:
                fig.update_traces(
                    marker=dict(color=POWER_BI_PALETTE[0], line=dict(width=1, color="#1e293b")),
                    hovertemplate="<b>Bin Range:</b> %{x}<br><b>Count:</b> %{y}<extra></extra>",
                )
            fig.update_layout(
                xaxis_title=target_x.replace("_", " ").title(),
                yaxis_title="Count of Observations",
                bargap=0.08,
            )

        # 4. Scatter Plot (Raw observations of two numerical columns, optional category color)
        elif chart_type in ("scatter", "relationship"):
            scatter_x = x_col if x_col in cols else (cols[0] if len(cols) >= 1 else None)
            scatter_y = y_col if y_col in cols else (cols[1] if len(cols) >= 2 else None)
            if scatter_x and scatter_y:
                x_1d = ensure_1d(df[scatter_x], str(scatter_x))
                y_1d = ensure_1d(df[scatter_y], str(scatter_y))
                has_color = bool(color_col and color_col in cols and color_col not in (scatter_x, scatter_y))
                if has_color:
                    color_1d = ensure_1d(df[color_col], str(color_col))
                    clean_scatter_df = pd.DataFrame({
                        scatter_x: x_1d,
                        scatter_y: y_1d,
                        color_col: color_1d,
                    }).dropna(subset=[scatter_x, scatter_y])

                    fig = px.scatter(
                        clean_scatter_df,
                        x=scatter_x,
                        y=scatter_y,
                        color=color_col,
                        title=title,
                        color_discrete_map=STABLE_CATEGORY_COLORS,
                        color_discrete_sequence=POWER_BI_PALETTE,
                        render_mode="webgl" if len(clean_scatter_df) > 3000 else "auto",
                    )
                    fig.update_layout(
                        legend_title_text=color_col.replace("_", " ").title(),
                    )
                else:
                    clean_scatter_df = pd.DataFrame({
                        scatter_x: x_1d,
                        scatter_y: y_1d,
                    }).dropna(subset=[scatter_x, scatter_y])

                    fig = px.scatter(
                        clean_scatter_df,
                        x=scatter_x,
                        y=scatter_y,
                        title=title,
                        color_discrete_sequence=[POWER_BI_PALETTE[2]],
                        render_mode="webgl" if len(clean_scatter_df) > 3000 else "auto",
                    )
                fig.update_traces(
                    marker=dict(size=8, opacity=0.85, line=dict(width=1, color="#1e293b")),
                    hovertemplate=f"<b>{scatter_x}:</b> %{{x}}<br><b>{scatter_y}:</b> %{{y}}<extra></extra>",
                )
                fig.update_layout(
                    xaxis_title=scatter_x.replace("_", " ").title(),
                    yaxis_title=scatter_y.replace("_", " ").title(),
                )
            else:
                fig = go.Figure()

        # 5. Line Chart (Single or Multi-Series Line)
        elif chart_type in ("line", "trend", "multi_line") and len(cols) >= 2:
            line_x = x_col if x_col in cols else cols[0]
            line_y = y_col if y_col in cols else (cols[1] if cols[1] != line_x else (cols[0] if len(cols) > 1 else line_x))
            has_series = bool(color_col and color_col in cols and color_col not in (line_x, line_y))

            x_1d = ensure_1d(df[line_x], str(line_x))
            y_1d = ensure_1d(df[line_y], str(line_y))

            if has_series:
                color_1d = ensure_1d(df[color_col], str(color_col))
                clean_line_df = pd.DataFrame({
                    line_x: x_1d,
                    line_y: y_1d,
                    color_col: color_1d,
                })
                # If X is numeric, sort numerically ascending
                if pd.api.types.is_numeric_dtype(clean_line_df[line_x]):
                    clean_line_df = clean_line_df.sort_values(by=[line_x, color_col], ascending=[True, True])

                fig = px.line(
                    clean_line_df,
                    x=line_x,
                    y=line_y,
                    color=color_col,
                    title=title,
                    markers=True,
                    color_discrete_map=STABLE_CATEGORY_COLORS,
                    color_discrete_sequence=POWER_BI_PALETTE,
                )
                fig.update_traces(line=dict(width=2.5), marker=dict(size=6))
                fig.update_layout(
                    xaxis_title=line_x.replace("_", " ").title(),
                    yaxis_title=str(line_y).replace("_", " ").title(),
                    legend_title_text=color_col.replace("_", " ").title(),
                )
            else:
                clean_line_df = pd.DataFrame({
                    line_x: x_1d,
                    line_y: y_1d,
                })
                if pd.api.types.is_numeric_dtype(clean_line_df[line_x]):
                    clean_line_df = clean_line_df.sort_values(by=line_x, ascending=True)

                fig = px.line(
                    clean_line_df,
                    x=line_x,
                    y=line_y,
                    title=title,
                    markers=True,
                    color_discrete_sequence=[POWER_BI_PALETTE[0]],
                )
                fig.update_traces(line=dict(width=3), marker=dict(size=6))
                fig.update_layout(
                    xaxis_title=line_x.replace("_", " ").title(),
                    yaxis_title=str(line_y).replace("_", " ").title(),
                )

        # 6. Box Plot
        elif chart_type in ("box", "box_plot") and (x_col in cols or len(cols) >= 1):
            target_x = x_col if x_col in cols else cols[0]
            box_y = y_col if y_col in cols else (cols[1] if len(cols) > 1 else None)
            if box_y:
                clean_box_df = pd.DataFrame({
                    target_x: ensure_1d(df[target_x], str(target_x)),
                    box_y: ensure_1d(df[box_y], str(box_y)),
                })
                fig = px.box(
                    clean_box_df,
                    x=target_x,
                    y=box_y,
                    title=title,
                    color=target_x,
                    color_discrete_sequence=POWER_BI_PALETTE,
                )
                fig.update_layout(
                    xaxis_title=target_x.replace("_", " ").title(),
                    yaxis_title=box_y.replace("_", " ").title(),
                    showlegend=False,
                )
            else:
                clean_box_df = pd.DataFrame({
                    target_x: ensure_1d(df[target_x], str(target_x))
                })
                fig = px.box(clean_box_df, y=target_x, title=title, color_discrete_sequence=[POWER_BI_PALETTE[0]])

        # 7. Pie / Donut
        elif chart_type in ("pie", "donut") and len(cols) >= 1:
            cat_col = x_col if x_col in cols else cols[0]
            val_col = y_col if y_col in cols else (cols[1] if len(cols) > 1 else None)
            is_donut = (mode == "donut" or "donut" in str(chart_type).lower())
            clean_pie_df = pd.DataFrame({
                cat_col: ensure_1d(df[cat_col], str(cat_col)),
                val_col: ensure_1d(df[val_col], str(val_col)),
            })
            fig = px.pie(
                clean_pie_df,
                names=cat_col,
                values=val_col,
                title=title,
                hole=0.4 if is_donut else 0.0,
                color_discrete_map=STABLE_CATEGORY_COLORS,
                color_discrete_sequence=POWER_BI_PALETTE,
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")

        # 8. Standard Bar Chart (Vertical or Horizontal)
        elif len(cols) >= 2 or (x_col and x_col in cols):
            cat_col = x_col if x_col in cols else cols[0]
            val_col = y_col if y_col in cols else (cols[1] if len(cols) > 1 else cols[0])

            is_horizontal = (len(df) > 8 or orientation == "horizontal" or chart_type == "horizontal_bar")
            clean_bar_df = pd.DataFrame({
                cat_col: ensure_1d(df[cat_col], str(cat_col)),
                val_col: ensure_1d(df[val_col], str(val_col)),
            })
            if is_horizontal:
                fig = px.bar(
                    clean_bar_df,
                    x=val_col,
                    y=cat_col,
                    orientation="h",
                    title=title,
                    color=cat_col,
                    color_discrete_sequence=POWER_BI_PALETTE,
                )
                fig.update_layout(
                    xaxis_title=str(val_col).replace("_", " ").title(),
                    yaxis_title=str(cat_col).replace("_", " ").title(),
                    yaxis=dict(autorange="reversed"),
                    showlegend=False,
                )
            else:
                fig = px.bar(
                    clean_bar_df,
                    x=cat_col,
                    y=val_col,
                    title=title,
                    color=cat_col,
                    color_discrete_sequence=POWER_BI_PALETTE,
                )
                fig.update_layout(
                    xaxis_title=str(cat_col).replace("_", " ").title(),
                    yaxis_title=str(val_col).replace("_", " ").title(),
                    showlegend=False,
                )

        # Fallback
        else:
            fig = go.Figure()
            fig.update_layout(title=title)

        # Centralized AI-MADAC Theme Polish (Power BI Dark Style)
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="#101426",
            plot_bgcolor="#101426",
            hovermode="closest",
            font=dict(color="#f1f5f9", family="Inter, system-ui, sans-serif", size=12),
            title_font=dict(size=15, color="#f8fafc", family="Inter, system-ui, sans-serif"),
            margin=dict(l=45, r=40, t=55, b=45),
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

        # Mirror to backend/charts if distinct so tests and relative lookups succeed
        from pathlib import Path
        backend_dir = str(Path(__file__).resolve().parents[3] / "charts")
        if os.path.abspath(settings.CHARTS_DIR) != os.path.abspath(backend_dir):
            os.makedirs(backend_dir, exist_ok=True)
            try:
                import shutil
                shutil.copy2(png_filepath, os.path.join(backend_dir, png_filename))
                shutil.copy2(json_filepath, os.path.join(backend_dir, json_filename))
            except Exception:
                pass

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

            if chart_type in ("line", "trend") and x_col and x_col in df.columns:
                val_col = y_col if (isinstance(y_col, str) and y_col in df.columns) else next((c for c in df.columns if c != x_col and c != color_col), (df.columns[1] if len(df.columns) > 1 else x_col))
                if color_col and color_col in df.columns and color_col != x_col:
                    for idx, (grp_name, grp_df) in enumerate(df.groupby(color_col)):
                        ax.plot(grp_df[x_col].astype(str), grp_df[val_col], marker='o', label=str(grp_name), color=POWER_BI_PALETTE[idx % len(POWER_BI_PALETTE)], linewidth=2)
                    ax.legend(title=color_col.replace("_", " ").title(), facecolor="#161b33", labelcolor="#f8fafc")
                else:
                    ax.plot(df[x_col].astype(str), df[val_col], marker='o', color="#6366F1", linewidth=2.5)
                ax.set_ylabel(str(val_col).replace("_", " ").title(), color="#dbeafe")
            elif isinstance(y_col, list) and x_col and x_col in df.columns:
                m_cols = [m for m in df.columns if m != x_col]
                x_indices = np.arange(len(df))
                width = 0.8 / max(1, len(m_cols))
                for idx, m in enumerate(m_cols):
                    ax.bar(x_indices + idx * width, df[m], width=width, label=m.replace("_", " ").title(), color=POWER_BI_PALETTE[idx % len(POWER_BI_PALETTE)], edgecolor="#1e293b")
                ax.set_xticks(x_indices + width * (len(m_cols) - 1) / 2)
                ax.set_xticklabels(df[x_col].astype(str))
                ax.legend(facecolor="#161b33", labelcolor="#f8fafc")
            elif color_col and color_col in df.columns and x_col and x_col in df.columns and "count" in df.columns:
                # Grouped bar
                pivot = df.pivot(index=x_col, columns=color_col, values="count").fillna(0)
                colors = ["#6366F1", "#EC4899", "#10B981", "#F59E0B"]
                pivot.plot(kind="bar", ax=ax, color=colors[:len(pivot.columns)], edgecolor="#1e293b")
                ax.set_ylabel("Number of Investors", color="#dbeafe")
                ax.legend(title=color_col.replace("_", " ").title(), facecolor="#161b33", labelcolor="#f8fafc")
            elif chart_type == "histogram" and x_col and x_col in df.columns:
                ax.hist(df[x_col].dropna(), bins=min(20, max(5, len(df) // 2)), color="#6366F1", edgecolor="#0f172a")
                ax.set_ylabel("Number of Investors", color="#dbeafe")
            elif chart_type in ("box", "box_plot") and x_col and x_col in df.columns:
                box_y = y_col if (isinstance(y_col, str) and y_col in df.columns) else next((c for c in df.columns if c != x_col), None)
                if box_y:
                    groups_data = []
                    labels = []
                    for grp_name, grp_df in df.groupby(x_col):
                        labels.append(str(grp_name))
                        groups_data.append(grp_df[box_y].dropna().values)
                    if groups_data:
                        bp = ax.boxplot(groups_data, tick_labels=labels, patch_artist=True)
                        for patch in bp['boxes']:
                            patch.set_facecolor("#6366F1")
                            patch.set_edgecolor("#94a3b8")
                        for median in bp['medians']:
                            median.set_color("#EC4899")
                            median.set_linewidth(2)
                    ax.set_ylabel(box_y.replace("_", " ").title(), color="#dbeafe")
            elif chart_type == "scatter" and x_col and x_col in df.columns:
                actual_y = y_col if (isinstance(y_col, str) and y_col in df.columns) else next((c for c in df.columns if c != x_col), None)
                if actual_y:
                    ax.scatter(df[x_col], df[actual_y], color="#06B6D4", alpha=0.85, edgecolors="#1e293b")
                    ax.set_ylabel(actual_y.replace("_", " ").title(), color="#dbeafe")
            elif chart_type in ("pie", "donut") and x_col and x_col in df.columns:
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
