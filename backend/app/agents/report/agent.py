"""
Report Agent.
Synthesizes query results, statistical analysis, charts, and predictions into a unified final report.
"""

import json
import logging
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from app.agents.state import AgentState
from app.core.llm_provider import get_llm

logger = logging.getLogger(__name__)

REPORT_PROMPT = """You are the Report Synthesis Agent for an enterprise data science platform.
Your task is to take the outputs from specialized agents (SQL Agent, Data Analyst Agent, ML Agent, etc.) and generate a clear, professional, direct answer to the user's question.

CRITICAL REPORTING RULES:
1. Provide a direct, data-backed answer in the first paragraph using the EXACT numbers calculated by the Data Analyst Agent or SQL Agent.
2. For relationship, impact, or comparison questions (e.g. "Did events affect stock impact?"):
   - Explicitly state the average metric for records WITH the factor (X) compared to records WITHOUT the factor (Y).
   - Explicitly state the EXACT missingness count and percentage for the factor (e.g. "the event field is missing for 97.87% of records").
   - Explicitly state that due to high missingness or observational design, the dataset DOES NOT provide sufficient evidence to conclude a true causal effect.
   - NEVER claim causation merely from correlation or group differences.
3. Provide 2-4 key insights as concise bullet points.
4. Output ONLY valid JSON matching this schema:
{
  "answer": "<direct data-backed answer>",
  "insights": ["<insight_1>", "<insight_2>", "<insight_3>"]
}
"""


class ReportAgent:
    """
    Report Agent responsible for synthesizing final structured answers and insights.
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
        Synthesize findings from state into final report.
        """
        question = state.get("question", "")
        sql_result = state.get("sql_result")
        analysis_result = state.get("analysis_result")
        ml_result = state.get("ml_result")
        viz_result = state.get("visualization_result")
        agents_used = state.get("agents_used", []) + ["report_agent"]

        logger.info(f"Report Agent synthesizing final response for query: '{question}'")

        context_parts = []

        if analysis_result:
            context_parts.append(f"Data Analyst Statistical Analysis: {json.dumps(analysis_result)}")

        if sql_result:
            sample_rows = sql_result.get("rows", [])[:10]
            context_parts.append(f"SQL Query Executed: {sql_result.get('query')}")
            context_parts.append(f"Data Retrieved ({sql_result.get('row_count')} total rows): {json.dumps(sample_rows)}")

        if ml_result:
            context_parts.append(f"ML Model Results: {json.dumps(ml_result)}")

        context_str = "\n\n".join(context_parts) if context_parts else "No tabular data returned."

        try:
            response = self.llm.invoke([
                SystemMessage(content=REPORT_PROMPT),
                HumanMessage(content=f"User Question: '{question}'\n\nAgent Analysis & Data Findings:\n{context_str}"),
            ])

            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            parsed = json.loads(content)
            answer = parsed.get("answer", "Data analysis completed.")
            insights = parsed.get("insights", [])

        except Exception as e:
            logger.warning(f"LLM report synthesis failed: {e}. Generating fallback report.")
            answer, insights = self._deterministic_fallback_report(question, analysis_result, sql_result)

        # Construct final report dictionary
        chart_paths = []
        if viz_result:
            chart_paths = viz_result.get("chart_paths", [])
            # Backwards compatibility for previously generated single-chart
            # visualization results.
            if not chart_paths and viz_result.get("chart_path"):
                chart_paths = [viz_result["chart_path"]]

        final_report = {
            "answer": answer,
            "insights": insights,
            "chart_paths": chart_paths,
            "data_used": sql_result,
            "agents_used": list(set(agents_used)),
        }

        return {
            "final_report": final_report,
            "agents_used": list(set(agents_used)),
        }

    def _deterministic_fallback_report(
        self, question: str, analysis_result: dict | None, sql_result: dict | None
    ) -> tuple[str, list[str]]:
        """
        Deterministic fallback if LLM synthesis fails.
        """
        if analysis_result and analysis_result.get("analysis_type") == "relationship_impact_analysis":
            with_grp = analysis_result.get("group_with_event", {})
            without_grp = analysis_result.get("group_without_event", {})
            missing_pct = analysis_result.get("missing_predictor_percentage", 0)
            pred_col = analysis_result.get("predictor_column", "event")
            target_col = analysis_result.get("target_column", "impact")

            mean_w = with_grp.get("mean_target", 0)
            mean_wo = without_grp.get("mean_target", 0)

            ans = (
                f"Based on the available records, stocks with recorded '{pred_col}' showed {mean_w} average {target_col} "
                f"compared with {mean_wo} for records without recorded '{pred_col}'. However, the '{pred_col}' field is missing "
                f"for {missing_pct}% of records, so this dataset does not provide sufficient evidence to conclude that "
                f"'{pred_col}' caused changes in {target_col}."
            )
            insights = [
                f"Records with recorded '{pred_col}' (N={with_grp.get('count')}) averaged {mean_w} {target_col}.",
                f"Records without recorded '{pred_col}' (N={without_grp.get('count')}) averaged {mean_wo} {target_col}.",
                f"Extreme missingness ({missing_pct}%) severe limits statistical generalizability.",
                "Observational group differences MUST NOT be interpreted as causal proof.",
            ]
            return ans, insights

        if sql_result and sql_result.get("message"):
            return sql_result["message"], ["Try a question using one of the dataset's available columns."]

        if sql_result and sql_result.get("rows"):
            rows = sql_result.get("rows", [])
            columns = sql_result.get("columns", [])

            # SQL fallback rules return compact, meaningful result sets. Turn
            # those into an answer rather than reporting only that data exists.
            if len(rows) == 1 and len(columns) == 1:
                metric = columns[0].replace("_", " ")
                value = rows[0].get(columns[0])
                ans = f"{metric.capitalize()}: {value}."
                insights = ["This is calculated from the full uploaded dataset, not just the table preview."]
                return ans, insights

            if len(columns) >= 2:
                group_column, metric_column = columns[0], columns[1]

                # SQL results are serialized for JSON with missing cells as an
                # empty string. Convert metric values safely before ranking so
                # a blank cannot be compared to a float at report time.
                def numeric_value(row: dict) -> float:
                    value = row.get(metric_column)
                    try:
                        return float(value) if value not in (None, "") else float("-inf")
                    except (TypeError, ValueError):
                        return float("-inf")

                highest = max(rows, key=numeric_value)
                ans = (
                    f"The result contains {len(rows)} {group_column.replace('_', ' ')} group(s). "
                    f"The highest {metric_column.replace('_', ' ')} is {highest.get(metric_column)} "
                    f"for {highest.get(group_column)}."
                )
                insights = [
                    f"The table below shows all {len(rows)} returned groups.",
                    f"The chart compares {metric_column.replace('_', ' ')} across {group_column.replace('_', ' ')}.",
                ]
                return ans, insights

            ans = f"Retrieved {sql_result.get('row_count')} matching record(s) from the dataset."
            insights = [f"Data contains {len(columns)} attributes."]
            return ans, insights

        return "Data analysis completed.", []


# Singleton instance
report_agent = ReportAgent()
