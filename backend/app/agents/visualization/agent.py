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


VISUALIZATION_AGENT_SYSTEM_PROMPT = """You are the Visualization Agent of AI-MADAC, an autonomous multi-agent data science analyst behaving like a Power BI visual intelligence engine.

Your job is to analyze the CURRENT USER QUERY, the QUERY CONTRACT, and the CURRENT DATASET SCHEMA, select the most appropriate visualization, and return a structured visualization specification that the AI-MADAC backend will use to generate the actual interactive chart using Plotly.

You are NOT the chart renderer. Plotly is the chart renderer.
Your responsibility is to make the correct visualization decision and provide an accurate, machine-readable specification.
Python/Pandas/SQL remains the SOLE source of truth for all actual numbers, counts, bins, filters, and metrics.

CRITICAL RULES:
1. THE CURRENT USER QUERY IS THE PRIMARY SOURCE OF TRUTH.
2. Every field explicitly requested by the user MUST appear in your specification.
   Example: If user asks "Show gender versus investment objective", you MUST use:
   x = "objective" (or "investment_objective"), color = "gender", mode = "grouped", chart = "bar".
   You MUST NEVER select "age", "expected", or any unrequested column.
3. FILTERS MUST BE EXPLICITLY RETURNED AS FIRST-CLASS PARAMETERS:
   Example: For "Show gender distribution of investors expecting 30-40% returns", return:
   filters = [{"field": "Expect", "operator": "equals", "value": "30-40%"}],
   x = "gender", color = null, chart = "bar", aggregation = "count".
   Do NOT treat the filter column as a plot dimension!
4. HISTOGRAM & BINNING:
   - Histograms only apply to raw numerical observations.
   - Categorical columns (gender, objective, return ranges) MUST NOT be histogrammed (use bar chart).
   - If user asks "bins of 5", set bins = {"mode": "fixed", "size": 5}.
   - If auto bins, set bins = {"mode": "auto"}.
5. NUMERICAL VS CATEGORICAL/RANGE:
   - If comparing a continuous numerical variable (e.g. Age) against a range category (e.g. Expected Return '10-20%', '20-30%', '30-40%'), choose chart = "box" (Box plot) to show distribution across brackets.
   - If the user explicitly asks for scatter plot, set y = "Expected Return Midpoint (%)" with transparent midpoint transformation.
6. AGGREGATIONS:
   - "count" / "number of investors" -> count
   - "average" / "mean" -> mean
   Never substitute count when user asked for average, or average when user asked for count.
7. NEVER invent columns. Only select from CURRENT DATASET COLUMNS.
8. NEVER invent chart data or calculations. Python will calculate all values.

RETURN ONLY VALID JSON WITH THIS EXACT STRUCTURE:
{
    "visualization_available": true,
    "query_interpretation": {
        "intent": "categorical_comparison",
        "requested_fields": ["objective", "gender"]
    },
    "fields": {
        "x": "objective",
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

    def run_spec_generation(self, state: AgentState) -> dict[str, Any]:
        """Step 1: Extract Query Contract and formulate candidate specification."""
        question = state.get("question", "").strip()
        schema = state.get("schema") or {}
        table_name = state.get("table_name", "")
        sql_result = state.get("sql_result")
        analysis_result = state.get("analysis_result")
        prev_result = state.get("previous_validated_result")

        logger.info(f"Visualization Agent processing query: '{question}'")

        query_contract_dict = state.get("query_contract")
        if query_contract_dict and not query_contract_dict.get("is_related_to_dataset"):
            return {
                "visualization_result": self._empty_visualization_result(
                    query_contract_dict.get("unrelated_reason")
                    or "I can't generate a chart for this question because it is not related to the currently loaded dataset."
                ),
                "agents_used": list(set(state.get("agents_used", []) + ["visualization_agent"])),
            }

        available_cols = [str(c.get("name", "")) for c in schema.get("columns", []) if c.get("name")]
        if not available_cols:
            if sql_result and sql_result.get("columns"):
                available_cols = [str(c) for c in sql_result["columns"]]
            elif sql_result and sql_result.get("rows") and len(sql_result["rows"]) > 0:
                available_cols = list(sql_result["rows"][0].keys())

        contract = extract_query_contract(
            question=question,
            schema=schema if schema.get("columns") else available_cols,
            previous_result=prev_result if bool(prev_result) else None,
        )

        # Merge fields from state query_contract if present
        if query_contract_dict:
            if query_contract_dict.get("suggested_chart_type"):
                contract.requested_chart = query_contract_dict["suggested_chart_type"]
            if query_contract_dict.get("bin_width"):
                contract.bin = {"mode": "fixed", "size": query_contract_dict["bin_width"]}
            if query_contract_dict.get("group_by"):
                contract.requested_dimensions = query_contract_dict["group_by"]

        if contract.is_follow_up and not sql_result and prev_result and prev_result.get("validated"):
            sql_result = {
                "rows": prev_result.get("result", []),
                "columns": prev_result.get("columns_used", []),
                "row_count": prev_result.get("row_count", len(prev_result.get("result", []))),
                "query": prev_result.get("query"),
                "filters": prev_result.get("filters", {}),
            }

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

        df = self._load_current_dataframe(table_name, sql_result, analysis_result, contract)
        if df is not None and not df.empty:
            for col in df.columns:
                if col not in available_cols:
                    available_cols.append(col)

        candidate_spec = self._get_candidate_specification(question, contract, schema, available_cols)
        if query_contract_dict:
            if query_contract_dict.get("suggested_chart_type"):
                candidate_spec.setdefault("chart", {})["type"] = query_contract_dict["suggested_chart_type"]
            if query_contract_dict.get("x_axis"):
                candidate_spec.setdefault("fields", {})["x"] = query_contract_dict["x_axis"]
            if query_contract_dict.get("y_axis"):
                candidate_spec.setdefault("fields", {})["y"] = query_contract_dict["y_axis"]
            if query_contract_dict.get("bin_width"):
                candidate_spec["bin_width"] = query_contract_dict["bin_width"]

        return {
            "_query_contract": contract,
            "_raw_spec": candidate_spec,
            "_available_cols": available_cols,
            "_query_df": df,
            "agents_used": list(set(state.get("agents_used", []) + ["visualization_agent"])),
        }

    def run_validation(self, state: AgentState) -> dict[str, Any]:
        """Step 2: Strict Query-to-Spec Validation & Deterministic Auto-Repair."""
        contract = state.get("_query_contract")
        raw_spec = state.get("_raw_spec")
        available_cols = state.get("_available_cols", [])

        if not contract:
            question = state.get("question", "").strip()
            schema = state.get("schema") or {}
            contract = extract_query_contract(question, schema)

        if not raw_spec:
            validated_spec = SpecValidator.build_deterministic_spec(contract, available_cols)
        else:
            is_valid, _ = SpecValidator.validate(raw_spec, contract, available_cols)
            if is_valid:
                validated_spec = SpecValidator.normalize_spec(raw_spec)
            else:
                validated_spec = SpecValidator.build_deterministic_spec(contract, available_cols)

        return {
            "_validated_spec": validated_spec,
            "agents_used": list(set(state.get("agents_used", []) + ["visualization_validator"])),
        }

    def run_rendering(self, state: AgentState) -> dict[str, Any]:
        """Step 3: Prepare data via Pandas and render interactive Plotly figure."""
        contract = state.get("_query_contract")
        df = state.get("_query_df")
        spec = state.get("_validated_spec")
        question = state.get("question", "").strip()
        table_name = state.get("table_name", "")
        sql_result = state.get("sql_result")
        analysis_result = state.get("analysis_result")
        schema = state.get("schema") or {}

        if not contract:
            contract = extract_query_contract(question, schema)
        if df is None or df.empty:
            df = self._load_current_dataframe(table_name, sql_result, analysis_result, contract)
        if not spec:
            available_cols = state.get("_available_cols", [str(c.get("name", "")) for c in schema.get("columns", [])])
            spec = SpecValidator.build_deterministic_spec(contract, available_cols)

        asks_visual_explicit = bool(re.search(r"\b(?:chart|plot|graph|pie|bar|visualize|histogram|scatter)\b", question.lower()))
        if contract.intent == "single_metric" and not asks_visual_explicit:
            empty_res = self._empty_visualization_result("Single value results are better communicated directly as a KPI rather than a chart.")
            return {
                "visualization_result": empty_res,
                "agents_used": list(set(state.get("agents_used", []) + ["chart_renderer"])),
            }

        if df is None or df.empty:
            empty_res = self._empty_visualization_result("There is not enough data available to generate this visualization.")
            return {
                "visualization_result": empty_res,
                "agents_used": list(set(state.get("agents_used", []) + ["chart_renderer"])),
            }

        is_rel_impact = bool(analysis_result and analysis_result.get("analysis_type") == "relationship_impact_analysis")
        if len(df) == 1 and not asks_visual_explicit and not is_rel_impact:
            empty_res = self._empty_visualization_result("Single value results are better communicated directly as a KPI rather than a chart.")
            return {
                "visualization_result": empty_res,
                "agents_used": list(set(state.get("agents_used", []) + ["chart_renderer"])),
            }

        try:
            prepared_df, applied_params, insights = DataPreparer.prepare(df, spec)
        except Exception as e:
            fallback_spec = SpecValidator.build_deterministic_spec(contract, list(df.columns))
            spec = fallback_spec
            prepared_df, applied_params, insights = DataPreparer.prepare(df, spec)

        if prepared_df.empty:
            empty_res = self._empty_visualization_result("No records remained after applying the requested filters.")
            return {
                "visualization_result": empty_res,
                "agents_used": list(set(state.get("agents_used", []) + ["chart_renderer"])),
            }

        try:
            plotly_fig, chart_title, chart_path, json_path = PlotlyRenderer.render(
                prepared_df, spec, question
            )
        except Exception as e:
            logger.error(f"Plotly rendering failed: {e}", exc_info=True)
            return {
                "visualization_result": self._empty_visualization_result(reason=str(e)),
                "agents_used": list(set(state.get("agents_used", []) + ["chart_renderer"])),
            }

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

        with open(json_path, "r", encoding="utf-8") as f:
            chart_json_dict = json.load(f)

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
            "agents_used": list(set(state.get("agents_used", []) + ["chart_renderer"])),
        }

    def run(self, state: AgentState) -> dict[str, Any]:
        """
        Execute the end-to-end visualization pipeline (composite execution).
        """
        s1 = self.run_spec_generation(state)
        merged = {**state, **s1}
        s2 = self.run_validation(merged)
        merged = {**merged, **s2}
        s3 = self.run_rendering(merged)
        return {
            "visualization_result": s3.get("visualization_result"),
            "agents_used": list(set(state.get("agents_used", []) + ["visualization_agent"])),
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
        1. If sql_result contains rows with required columns: use sql_result rows.
        2. If analysis_result contains relationship findings: format as comparative DataFrame.
        3. If table_name is available and query requires raw observations: query table with LIMIT.
        """
        # 1. Use SQL result rows if available
        if sql_result and sql_result.get("rows"):
            df = pd.DataFrame(sql_result["rows"])
            if not df.empty:
                return df

        # 2. Support relationship analysis from Data Analyst Agent
        if analysis_result and analysis_result.get("analysis_type") == "relationship_impact_analysis":
            with_grp = analysis_result.get("group_with_event", {})
            without_grp = analysis_result.get("group_without_event", {})
            pred_col = analysis_result.get("predictor_column", "Factor")
            target_col = analysis_result.get("target_column", "Impact")
            return pd.DataFrame([
                {"Group": f"With {pred_col}", f"Average {target_col}": with_grp.get("mean_target", 0)},
                {"Group": f"Without {pred_col}", f"Average {target_col}": without_grp.get("mean_target", 0)},
            ])

        # 3. If database table is available, query only the required columns with safe LIMIT
        if table_name:
            try:
                safe_table = table_name.replace('"', '""')
                if contract.requested_fields:
                    quoted_cols = [f'"{f}"' for f in contract.requested_fields]
                    col_sql = ", ".join(quoted_cols)
                    df = pd.read_sql(f'SELECT {col_sql} FROM "{safe_table}" LIMIT 5000', con=sync_engine)
                else:
                    df = pd.read_sql(f'SELECT * FROM "{safe_table}" LIMIT 5000', con=sync_engine)
                if not df.empty:
                    return df
            except Exception as e:
                logger.warning(f"Failed to read table '{table_name}' from sync_engine: {e}")

        return None

    def _get_candidate_specification(
        self,
        question: str,
        contract: QueryContract,
        schema: dict[str, Any],
        available_cols: list[str],
    ) -> dict[str, Any]:
        """Obtain initial candidate specification from Gemini or deterministic spec."""
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
            return candidate_spec
        except Exception as e:
            logger.warning(f"Gemini candidate spec extraction failed: {e}. Using deterministic fallback.")
            return SpecValidator.build_deterministic_spec(contract, available_cols)

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
