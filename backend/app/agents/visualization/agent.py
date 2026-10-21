"""
Visualization Agent of AI-MADAC.

Autonomous multi-agent data science analyst visualization intelligence and rendering pipeline:

Pipeline:
CURRENT USER QUERY
        ↓
QUERY UNDERSTANDING / INTENT EXTRACTION (Query Contract)
        ↓
CURRENT DATASET SCHEMA & SEMANTIC COLUMN RESOLUTION
        ↓
VISUALIZATION SPECIFICATION (Gemini Intelligence Layer)
        ↓
STRICT SPEC VALIDATION (SpecValidator)
        ↓
REPAIR / RETRY IF INVALID (Auto-Repair)
        ↓
CURRENT DATAFRAME (Database or SQL result)
        ↓
PANDAS DATA PREPARATION (DataPreparer)
        ↓
DATA VALIDATION
        ↓
PLOTLY FIGURE GENERATION (PlotlyRenderer)
        ↓
FIGURE VALIDATION
        ↓
REACT UI

CRITICAL RULES:
1. THE CURRENT USER QUERY IS THE PRIMARY SOURCE OF TRUTH.
2. Explicit Field Rule: Explicitly mentioned dataset fields MUST appear in the final chart.
3. No shortcut from previous result -> chart unless explicitly referenced with anaphoric pronouns.
4. Gemini decides WHAT to visualize and selects configuration, but NEVER generates fake data or plotting code.
5. Pandas / Python calculates all actual counts, means, bins, filters, and rankings.
6. Histogram uses raw observations (sum of bin counts == total valid rows).
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.state import AgentState
from app.agents.visualization.contract import QueryContract, extract_query_contract, resolve_column_semantically
from app.agents.visualization.spec_validator import SpecValidator
from app.agents.visualization.data_preparer import DataPreparer
from app.agents.visualization.plotly_renderer import PlotlyRenderer
from app.core.llm_provider import get_llm
from app.db.database import sync_engine

logger = logging.getLogger(__name__)


VISUALIZATION_AGENT_SYSTEM_PROMPT = """You are the Visualization Agent of AI-MADAC, an autonomous multi-agent data science analyst.

Your job is to analyze the CURRENT USER QUERY, the QUERY CONTRACT, and the CURRENT DATASET SCHEMA, select the most appropriate visualization, and return a structured visualization specification that the AI-MADAC backend will use to generate the actual interactive chart using Plotly.

You are NOT the chart renderer.
Plotly is the chart renderer.
Your responsibility is to make the correct visualization decision and provide an accurate, machine-readable specification.

CRITICAL RULES:
1. THE CURRENT USER QUERY IS THE PRIMARY SOURCE OF TRUTH.
2. Every field explicitly requested by the user MUST appear in your specification.
   Example: If user asks "Show gender versus investment objective", you MUST use:
   x = "investment_objective", color = "gender", mode = "grouped", chart = "bar".
   You MUST NEVER select "age" or any unrequested column.
3. NEVER invent columns. Only select from CURRENT DATASET COLUMNS.
4. NEVER invent chart data or calculations. Python will calculate all values.
5. For cross-tabulations (two categorical dimensions), use chart = "bar", mode = "grouped", with x = dimension 1, color = dimension 2.
6. For numeric distributions, use chart = "histogram".
7. For relationships between two numeric variables, use chart = "scatter".
8. For small categorical part-to-whole, use chart = "pie".

RETURN ONLY VALID JSON WITH THIS EXACT STRUCTURE:
{
    "visualization_available": true,
    "query_interpretation": {
        "intent": "categorical_comparison",
        "requested_fields": ["investment_objective", "gender"]
    },
    "fields": {
        "x": "investment_objective",
        "y": null,
        "color": "gender",
        "facet": null
    },
    "measure": {
        "type": "count",
        "column": null
    },
    "filters": [],
    "sorting": {
        "field": null,
        "direction": null
    },
    "limit": null,
    "binning": null,
    "chart": {
        "type": "bar",
        "orientation": "vertical",
        "mode": "grouped"
    },
    "title": "Investors by Gender and Investment Objective",
    "axis_labels": {
        "x": "Investment Objective",
        "y": "Number of Investors"
    },
    "legend": {
        "show": true,
        "title": "Gender"
    }
}
"""


class ExplanationDict(dict):
    """
    Structured explanation dictionary adhering to AI-MADAC specifications
    (what_this_shows, parameters_used, key_insights) with text formatting.
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
    Visualization Agent orchestrating query contract creation, Gemini specification,
    strict validation, auto-repair, Pandas data preparation, and Plotly rendering.
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
        Execute the end-to-end visualization pipeline.
        """
        question = state.get("question", "").strip()
        schema = state.get("schema") or {}
        table_name = state.get("table_name", "")
        sql_result = state.get("sql_result")
        analysis_result = state.get("analysis_result")
        prev_result = state.get("previous_validated_result")

        logger.info(f"Visualization Agent processing query: '{question}'")

        # 1. Extract Available Columns from Schema or SQL Result
        available_cols = [str(c.get("name", "")) for c in schema.get("columns", []) if c.get("name")]
        if not available_cols:
            if sql_result and sql_result.get("columns"):
                available_cols = [str(c) for c in sql_result["columns"]]
            elif sql_result and sql_result.get("rows") and len(sql_result["rows"]) > 0:
                available_cols = list(sql_result["rows"][0].keys())

        # 2. Extract Query Contract (Intent Extraction & Semantic Column Resolution)
        contract = extract_query_contract(
            question=question,
            schema=schema if schema.get("columns") else available_cols,
            previous_result=prev_result if bool(prev_result) else None,
        )

        # Unpack previous_validated_result if this is an explicit follow-up and sql_result is empty
        if contract.is_follow_up and not sql_result and prev_result and prev_result.get("validated"):
            sql_result = {
                "rows": prev_result.get("result", []),
                "columns": prev_result.get("columns_used", []),
                "row_count": prev_result.get("row_count", len(prev_result.get("result", []))),
                "query": prev_result.get("query"),
                "filters": prev_result.get("filters", {}),
            }

        # Incorporate filters from sql_result if present
        if sql_result and sql_result.get("filters"):
            sql_filters = sql_result["filters"]
            if isinstance(sql_filters, dict):
                for k, v in sql_filters.items():
                    if not any(f.get("column") == k for f in contract.filters):
                        contract.filters.append({"column": k, "op": "==", "value": v})

        logger.info(
            f"Query Contract extracted: requested_fields={contract.requested_fields}, "
            f"dimensions={contract.dimensions}, intent={contract.intent}, is_follow_up={contract.is_follow_up}"
        )

        # 3. Handle Single Value Metric (KPI) questions (Section 16)
        asks_visual_explicit = bool(re.search(r"\b(?:chart|plot|graph|pie|bar|visualize|histogram|scatter)\b", question.lower()))
        if contract.intent == "single_metric" and not asks_visual_explicit:
            logger.info("Visualization Agent: Single value KPI result, skipping chart.")
            return {
                "visualization_result": self._empty_visualization_result(
                    reason="Single value results are better communicated directly as a KPI rather than a chart."
                ),
                "agents_used": state.get("agents_used", []) + ["visualization_agent"],
            }

        # 4. Obtain Current DataFrame
        df = self._load_current_dataframe(table_name, sql_result, analysis_result, contract)
        if df is None or df.empty:
            logger.warning("Visualization Agent: No tabular data available.")
            return {
                "visualization_result": self._empty_visualization_result(
                    reason="There is not enough data available to generate this visualization."
                ),
                "agents_used": state.get("agents_used", []) + ["visualization_agent"],
            }

        # Single value KPI check on loaded df
        is_rel_impact = bool(analysis_result and analysis_result.get("analysis_type") == "relationship_impact_analysis")
        if len(df) == 1 and not asks_visual_explicit and not is_rel_impact:
            logger.info("Visualization Agent: Single value KPI result, skipping chart.")
            return {
                "visualization_result": self._empty_visualization_result(
                    reason="Single value results are better communicated directly as a KPI rather than a chart."
                ),
                "agents_used": state.get("agents_used", []) + ["visualization_agent"],
            }

        # Add any runtime DataFrame columns to available_cols if not already present
        for col in df.columns:
            if col not in available_cols:
                available_cols.append(col)

        # 5. Gemini Specification Decision with Strict Validation & Auto-Repair
        spec = self._get_validated_specification(question, contract, schema, available_cols)

        # 6. Pandas Data Preparation & Mathematical Insights
        try:
            prepared_df, applied_params, insights = DataPreparer.prepare(df, spec)
        except Exception as e:
            logger.error(f"Data preparation error: {e}. Falling back to deterministic spec.", exc_info=True)
            fallback_spec = SpecValidator.build_deterministic_spec(contract, available_cols)
            spec = fallback_spec
            prepared_df, applied_params, insights = DataPreparer.prepare(df, spec)

        if prepared_df.empty:
            logger.warning("Visualization Agent: Prepared DataFrame is empty after filtering/aggregation.")
            return {
                "visualization_result": self._empty_visualization_result(
                    reason="No records remained after applying the requested filters."
                ),
                "agents_used": state.get("agents_used", []) + ["visualization_agent"],
            }

        # 7. Plotly Figure Generation (Rendering Layer)
        try:
            plotly_fig, chart_title, chart_path, json_path = PlotlyRenderer.render(
                prepared_df, spec, question
            )
        except Exception as e:
            logger.error(f"Plotly rendering failed: {e}", exc_info=True)
            return {
                "visualization_result": self._empty_visualization_result(reason=str(e)),
                "agents_used": state.get("agents_used", []) + ["visualization_agent"],
            }

        # 8. Build Structured Explanation
        chart_type_name = applied_params.get("chart_type", "Chart")
        what_shows = spec.get("title") or f"The {chart_type_name.lower()} visualizes {', '.join(contract.requested_fields)}."
        param_lines = [
            f"{k.replace('_', ' ').title()}: {v}"
            for k, v in applied_params.items()
            if v is not None and not (isinstance(v, list) and len(v) == 0)
        ]

        explanation_dict = ExplanationDict({
            "what_this_shows": what_shows,
            "parameters_used": param_lines,
            "key_insights": insights,
        })

        # Load figure JSON for frontend
        with open(json_path, "r", encoding="utf-8") as f:
            chart_json_dict = json.load(f)

        # Build chart object with backwards compatibility parameters
        chart_obj = {
            "chart_type": spec.get("chart", {}).get("type", "bar"),
            "title": chart_title,
            "x_axis": spec.get("fields", {}).get("x"),
            "y_axis": spec.get("fields", {}).get("y") or applied_params.get("value"),
            "category": applied_params.get("category") or spec.get("fields", {}).get("x"),
            "value": applied_params.get("value") or spec.get("fields", {}).get("y"),
            "parameters": applied_params,
            "data": prepared_df.head(50).to_dict(orient="records"),
            "plotly_options": spec.get("plotly_options", {}),
        }

        viz_result = {
            "visualization_available": True,
            "chart": chart_obj,
            "parameters": applied_params,
            "explanation": explanation_dict,
            "key_insights": insights,
            "chart_type": spec.get("chart", {}).get("type", "bar"),
            "chart_path": chart_path,
            "chart_paths": [chart_path],
            "chart_json": chart_json_dict,
            "chart_jsons": [chart_path.replace(".png", ".json")],
            "titles": [chart_title],
        }

        return {
            "visualization_result": viz_result,
            "agents_used": state.get("agents_used", []) + ["visualization_agent"],
        }

    def _load_current_dataframe(
        self,
        table_name: str,
        sql_result: Optional[dict[str, Any]],
        analysis_result: Optional[dict[str, Any]],
        contract: QueryContract,
    ) -> Optional[pd.DataFrame]:
        """
        Load the current DataFrame with all required columns.
        Priority:
        1. If table_name is available and query requires raw observations or cross-tabulation:
           read directly from the database table.
        2. If analysis_result contains relationship findings: format as comparative DataFrame.
        3. If sql_result contains rows with required columns: use sql_result rows.
        """
        # If database table is available, query it directly to guarantee access to raw observations
        # and all requested dimensions (e.g. for cross-tabulation or histograms)
        if table_name:
            try:
                safe_table = table_name.replace('"', '""')
                df = pd.read_sql(f'SELECT * FROM "{safe_table}"', con=sync_engine)
                if not df.empty:
                    # Check if all requested fields are in this DataFrame
                    if all(col in df.columns for col in contract.requested_fields):
                        return df
            except Exception as e:
                logger.warning(f"Failed to read table '{table_name}' from sync_engine: {e}")

        # Support relationship analysis from Data Analyst Agent
        if analysis_result and analysis_result.get("analysis_type") == "relationship_impact_analysis":
            with_grp = analysis_result.get("group_with_event", {})
            without_grp = analysis_result.get("group_without_event", {})
            pred_col = analysis_result.get("predictor_column", "Factor")
            target_col = analysis_result.get("target_column", "Impact")
            return pd.DataFrame([
                {"Group": f"With {pred_col}", f"Average {target_col}": with_grp.get("mean_target", 0)},
                {"Group": f"Without {pred_col}", f"Average {target_col}": without_grp.get("mean_target", 0)},
            ])

        # Use SQL result rows if available
        if sql_result and sql_result.get("rows"):
            return pd.DataFrame(sql_result["rows"])

        return None

    def _get_validated_specification(
        self,
        question: str,
        contract: QueryContract,
        schema: dict[str, Any],
        available_cols: list[str],
    ) -> dict[str, Any]:
        """
        Consult Gemini for the specification, strictly validate against QueryContract,
        and auto-repair if validation fails.
        """
        # Try LLM generation first
        try:
            prompt_data = {
                "CURRENT_USER_QUERY": question,
                "QUERY_CONTRACT": contract.to_dict(),
                "CURRENT_DATASET_COLUMNS": available_cols,
                "COLUMN_TYPES": {c.get("name"): c.get("dtype") for c in schema.get("columns", [])},
            }
            messages = [
                SystemMessage(content=VISUALIZATION_AGENT_SYSTEM_PROMPT),
                HumanMessage(content=json.dumps(prompt_data, indent=2)),
            ]
            response = self.llm.invoke(messages)
            content = self._extract_text(response.content)
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            candidate_spec = json.loads(content)

            # Strict Spec Validation
            is_valid, error_reason = SpecValidator.validate(candidate_spec, contract, available_cols)
            if is_valid:
                logger.info("Gemini visualization specification passed strict validation.")
                return candidate_spec

            logger.warning(
                f"Gemini specification failed validation: {error_reason}. Attempting repair retry."
            )
            # 1 LLM repair retry with explicit error
            repair_messages = messages + [
                response,
                HumanMessage(
                    content=(
                        f"VALIDATION ERROR: {error_reason}\n\n"
                        f"You MUST use the exact requested fields: {contract.requested_fields}. "
                        "Do not include any unrequested columns. Correct the JSON specification."
                    )
                ),
            ]
            repair_response = self.llm.invoke(repair_messages)
            repair_content = self._extract_text(repair_response.content)
            if repair_content.startswith("```"):
                repair_content = repair_content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            repaired_spec = json.loads(repair_content)
            is_valid, _ = SpecValidator.validate(repaired_spec, contract, available_cols)
            if is_valid:
                logger.info("Repaired Gemini visualization specification passed strict validation.")
                return repaired_spec

        except Exception as e:
            logger.warning(f"Gemini specification extraction failed: {e}. Using deterministic auto-repair.")

        # Fallback: Infallible Deterministic Auto-Repair from QueryContract
        logger.info("Using deterministic specification directly from QueryContract.")
        return SpecValidator.build_deterministic_spec(contract, available_cols)

    @staticmethod
    def _extract_text(content: Any) -> str:
        """Safely extract plain string from LLM response content blocks."""
        if isinstance(content, list):
            return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content).strip()
        return str(content).strip()

    @staticmethod
    def _empty_visualization_result(reason: str | None = None) -> dict[str, Any]:
        """Return standardized payload when visualization is unavailable."""
        msg = reason or "There is not enough data available to generate this visualization."
        return {
            "visualization_available": False,
            "reason": msg,
            "chart": None,
            "parameters": {},
            "explanation": ExplanationDict({
                "what_this_shows": msg,
                "parameters_used": [],
                "key_insights": [],
                "not enough data": msg,
            }),
            "key_insights": [],
            "chart_paths": [],
            "chart_jsons": [],
            "titles": [],
        }


# Singleton instance
visualization_agent = VisualizationAgent()
