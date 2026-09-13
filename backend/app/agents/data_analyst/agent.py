"""
Data Analyst Agent.
Handles automatic profiling, statistical analysis, data quality checks, EDA, and relationship/impact analysis.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import text

from app.agents.state import AgentState
from app.db.database import sync_engine
from app.utils.sql_validator import sanitize_table_name

logger = logging.getLogger(__name__)


class DataAnalystAgent:
    """
    Data Analyst Agent responsible for profiling datasets, computing statistics,
    identifying data quality issues, and executing in-depth relationship and impact analysis.
    """

    def run(self, state: AgentState) -> dict[str, Any]:
        """
        Execute deep statistical / relationship analysis on the dataset for a user query.
        """
        table_name = state.get("table_name", "")
        question = state.get("question", "")
        schema = state.get("schema", {})

        safe_table = sanitize_table_name(table_name)
        if not safe_table:
            return {"analysis_result": None}

        logger.info(f"Data Analyst Agent executing query analysis for table '{safe_table}'")

        try:
            df = pd.read_sql(f'SELECT * FROM "{safe_table}"', con=sync_engine)
        except Exception as e:
            logger.error(f"Data Analyst read table error: {e}")
            return {"analysis_result": {"error": str(e)}}

        if df.empty:
            return {"analysis_result": {"message": "Dataset table is empty."}}

        # Perform targeted relationship & impact analysis
        analysis_res = self.analyze_relationship_query(df, question, schema)
        agents_used = state.get("agents_used", []) + ["data_analyst_agent"]

        return {
            "analysis_result": analysis_res,
            "agents_used": agents_used,
        }

    def analyze_relationship_query(self, df: pd.DataFrame, question: str, schema: dict) -> dict[str, Any]:
        """
        Analyze relationships, impact, group differences, and missingness quality condition.
        """
        q_lower = question.lower()
        cols = df.columns.tolist()

        # Identify candidate predictor (e.g. event) and target metric (e.g. stock_impact)
        predictor_col = None
        target_col = None

        # Check explicit column mentions in question
        for c in cols:
            if c.lower() in q_lower or c.lower().replace("_", " ") in q_lower:
                if df[c].dtype == "object" or df[c].dtype.name == "category":
                    predictor_col = c
                elif pd.api.types.is_numeric_dtype(df[c]):
                    target_col = c

        # Fallback defaults if not matched directly from text
        if not predictor_col:
            # Look for columns with missing values or categorical names like 'event', 'category', 'status'
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            for c in cat_cols:
                if "event" in c.lower() or "type" in c.lower() or "group" in c.lower():
                    predictor_col = c
                    break
            if not predictor_col and cat_cols:
                predictor_col = cat_cols[0]

        if not target_col:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            for c in num_cols:
                if "impact" in c.lower() or "growth" in c.lower() or "revenue" in c.lower() or "score" in c.lower():
                    target_col = c
                    break
            if not target_col and num_cols:
                target_col = num_cols[-1]

        if not predictor_col or not target_col:
            return {
                "summary_analysis": f"Dataset has {len(df)} records and {len(cols)} columns.",
                "total_rows": len(df),
            }

        # 1. Total records
        total_rows = len(df)

        # 2. Missingness calculation for predictor column
        predictor_series = df[predictor_col]
        missing_count = int(predictor_series.isnull().sum() + (predictor_series == "").sum())
        missing_pct = round((missing_count / total_rows) * 100, 2)

        # 3. Group Segmentation: Group A (with predictor) vs Group B (without predictor)
        is_present_mask = predictor_series.notnull() & (predictor_series != "") & (predictor_series != "None")
        group_with = df[is_present_mask]
        group_without = df[~is_present_mask]

        count_with = len(group_with)
        count_without = len(group_without)

        target_with = group_with[target_col].dropna()
        target_without = group_without[target_col].dropna()

        mean_with = round(float(target_with.mean()), 4) if not target_with.empty else None
        median_with = round(float(target_with.median()), 4) if not target_with.empty else None
        std_with = round(float(target_with.std()), 4) if len(target_with) > 1 else None

        mean_without = round(float(target_without.mean()), 4) if not target_without.empty else None
        median_without = round(float(target_without.median()), 4) if not target_without.empty else None
        std_without = round(float(target_without.std()), 4) if len(target_without) > 1 else None

        diff_in_means = round(mean_with - mean_without, 4) if (mean_with is not None and mean_without is not None) else None

        # 4. Statistical significance test (Welch's t-test)
        stat_p_value = None
        stat_significant = False
        if len(target_with) > 2 and len(target_without) > 2:
            try:
                t_stat, p_val = stats.ttest_ind(target_with, target_without, equal_var=False)
                if not np.isnan(p_val):
                    stat_p_value = round(float(p_val), 5)
                    stat_significant = bool(p_val < 0.05)
            except Exception:
                pass

        # 5. Category breakdown for Group A
        category_breakdown = {}
        if count_with > 0:
            top_cats = group_with[predictor_col].value_counts().head(5)
            for cat_name, cat_cnt in top_cats.items():
                sub = group_with[group_with[predictor_col] == cat_name][target_col].dropna()
                category_breakdown[str(cat_name)] = {
                    "count": int(cat_cnt),
                    "mean_target": round(float(sub.mean()), 4) if not sub.empty else None,
                }

        # 6. Construct Data Quality Caveat & Limitations
        limitation_msg = (
            f"The predictor column '{predictor_col}' is missing for {missing_count} out of {total_rows} records "
            f"({missing_pct}% missingness). Because of this severe missingness, observational group differences "
            f"do not provide sufficient evidence to conclude a true causal effect."
        )

        return {
            "analysis_type": "relationship_impact_analysis",
            "predictor_column": predictor_col,
            "target_column": target_col,
            "total_records": total_rows,
            "missing_predictor_count": missing_count,
            "missing_predictor_percentage": missing_pct,
            "group_with_event": {
                "label": f"With {predictor_col}",
                "count": count_with,
                "mean_target": mean_with,
                "median_target": median_with,
                "std_target": std_with,
            },
            "group_without_event": {
                "label": f"Without {predictor_col}",
                "count": count_without,
                "mean_target": mean_without,
                "median_target": median_without,
                "std_target": std_without,
            },
            "difference_in_means": diff_in_means,
            "p_value": stat_p_value,
            "is_statistically_significant": stat_significant,
            "category_breakdown": category_breakdown,
            "data_quality_limitation": limitation_msg,
            "causation_warning": "Observational group differences or correlations MUST NOT be interpreted as causation.",
        }

    def profile_dataset(self, table_name: str) -> dict[str, Any]:
        """
        Generate a comprehensive statistical profile of a dataset table in PostgreSQL.
        """
        safe_table = sanitize_table_name(table_name)

        logger.info(f"Profiling table '{safe_table}'...")
        try:
            df = pd.read_sql(f'SELECT * FROM "{safe_table}"', con=sync_engine)
        except Exception as e:
            logger.error(f"Failed to read table '{safe_table}' for profiling: {e}")
            raise RuntimeError(f"Database read error during profiling: {e}")

        total_rows = len(df)
        total_cols = len(df.columns)
        duplicate_rows = int(df.duplicated().sum())
        memory_mb = round(float(df.memory_usage(deep=True).sum()) / (1024 * 1024), 3)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        datetime_cols = df.select_dtypes(include=["datetime", "datetimetz"]).columns.tolist()

        missing_summary = {}
        for col in df.columns:
            null_count = int(df[col].isnull().sum())
            if null_count > 0:
                missing_summary[col] = {
                    "count": null_count,
                    "percentage": round((null_count / total_rows) * 100, 2),
                }

        numeric_stats = {}
        if numeric_cols:
            desc = df[numeric_cols].describe().T
            for col in numeric_cols:
                row = desc.loc[col]
                numeric_stats[col] = {
                    "count": int(row["count"]),
                    "mean": round(float(row["mean"]), 3) if not np.isnan(row["mean"]) else None,
                    "std": round(float(row["std"]), 3) if not np.isnan(row["std"]) else None,
                    "min": round(float(row["min"]), 3) if not np.isnan(row["min"]) else None,
                    "q25": round(float(row["25%"]), 3) if not np.isnan(row["25%"]) else None,
                    "median": round(float(row["50%"]), 3) if not np.isnan(row["50%"]) else None,
                    "q75": round(float(row["75%"]), 3) if not np.isnan(row["75%"]) else None,
                    "max": round(float(row["max"]), 3) if not np.isnan(row["max"]) else None,
                    "skewness": round(float(df[col].skew()), 3) if not df[col].isnull().all() else None,
                }

        categorical_stats = {}
        for col in categorical_cols:
            val_counts = df[col].value_counts(dropna=False).head(10).to_dict()
            formatted_counts = {str(k): int(v) for k, v in val_counts.items()}
            top_val = str(df[col].mode().iloc[0]) if not df[col].mode().empty else None

            categorical_stats[col] = {
                "unique_count": int(df[col].nunique(dropna=True)),
                "top_value": top_val,
                "top_freq": int(df[col].value_counts().iloc[0]) if not df[col].empty else 0,
                "value_counts": formatted_counts,
            }

        correlations = {}
        if len(numeric_cols) >= 2:
            corr_df = df[numeric_cols].corr().round(3)
            for col in numeric_cols:
                correlations[col] = {
                    other: float(corr_df.loc[col, other])
                    for other in numeric_cols
                    if not np.isnan(corr_df.loc[col, other])
                }

        quality_issues = []
        if duplicate_rows > 0:
            quality_issues.append({
                "type": "duplicates",
                "severity": "medium",
                "message": f"Found {duplicate_rows} duplicate row(s) ({(duplicate_rows/total_rows)*100:.1f}% of dataset)."
            })

        for col, info in missing_summary.items():
            pct = info["percentage"]
            severity = "high" if pct > 30 else ("medium" if pct > 10 else "low")
            quality_issues.append({
                "type": "missing_values",
                "severity": severity,
                "column": col,
                "message": f"Column '{col}' has {info['count']} missing values ({pct}%)."
            })

        profile = {
            "summary": {
                "total_rows": total_rows,
                "total_columns": total_cols,
                "duplicate_rows": duplicate_rows,
                "memory_usage_mb": memory_mb,
                "numeric_columns_count": len(numeric_cols),
                "categorical_columns_count": len(categorical_cols),
                "datetime_columns_count": len(datetime_cols),
            },
            "column_types": {
                "numeric": numeric_cols,
                "categorical": categorical_cols,
                "datetime": datetime_cols,
            },
            "missing_summary": missing_summary,
            "numeric_stats": numeric_stats,
            "categorical_stats": categorical_stats,
            "correlations": correlations,
            "data_quality_issues": quality_issues,
        }

        logger.info(f"Dataset profiling complete for '{safe_table}'. Identified {len(quality_issues)} quality issue(s).")
        return profile


# Singleton instance
data_analyst_agent = DataAnalystAgent()
