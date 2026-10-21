"""
Report Agent.
Synthesizes query results, statistical analysis, charts, and predictions into a unified final report.
"""

import json
import logging
from collections import Counter
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




RESPONSE_REWRITER_PROMPT = """You are the final response rewriter for AI-MADAC.

Your ONLY job is to rewrite the response produced by another AI-MADAC agent into a simple, clear, natural, user-friendly response.

The original agent has already:
- Understood the user's question
- Performed the required analysis
- Retrieved the required information
- Calculated the result
- Produced the answer

DO NOT redo the analysis.
DO NOT change the meaning of the original response.
DO NOT question or reinterpret the agent's result.
DO NOT add new facts, numbers, calculations, assumptions, or conclusions.

Your job is ONLY to improve how the existing answer is communicated.

## MAIN GOAL
Make the response sound like a helpful AI assistant having a normal conversation with the user.
The final response should be:
- Simple
- Clear
- Concise
- Natural
- Easy to understand
- Directly focused on the user's question

Use simple English.
Prefer short sentences.
Remove unnecessary technical language and internal processing details.

## IMPORTANT: PRESERVE INFORMATION
Never change:
- Numbers, percentages, dates, names, categories, calculated results, important facts.

## REMOVE INTERNAL INFORMATION
Do not expose internal AI-MADAC implementation details such as agent names, LangGraph, RAG, SQL execution, database queries, internal state, tool calls, or debug information.

## CRITICAL RULE
The original agent response is the SOURCE OF TRUTH.
AGENT RESPONSE -> SIMPLER LANGUAGE.
DO NOT hallucinate.
Return ONLY the final user-facing response.
Do not include wrappers like "Rewritten response:", "Here is the simplified version:", or quotes.
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
        structured_intent = state.get("structured_intent") or {}

        if structured_intent.get("mode") == "FOLLOW_UP_VISUALIZATION":
            previous = state.get("previous_validated_result") or {}
            viz = state.get("visualization_result") or {}
            if previous.get("validated") and previous.get("dataset_id") == state.get("dataset_id"):
                answer = (
                    viz.get("explanation")
                    if viz and viz.get("explanation")
                    else "Created the requested visualization from the previous validated result."
                )
                insights = (
                    viz.get("key_insights")
                    if viz and viz.get("key_insights")
                    else ["The chart uses the exact values from the preceding analysis; no new analysis was performed."]
                )
                prev_data_used = {
                    "query": previous.get("query"),
                    "columns": previous.get("columns_used", []),
                    "rows": previous.get("result", []),
                    "row_count": previous.get("row_count", 0),
                    "visualization": viz,
                    "parameters": viz.get("parameters"),
                    "chart_json": viz.get("chart_json"),
                }
                final_report = {
                    "answer": str(answer),
                    "insights": insights,
                    "chart_paths": viz.get("chart_paths", []),
                    "chart_json": viz.get("chart_json"),
                    "parameters": viz.get("parameters"),
                    "visualization_result": viz,
                    "data_used": prev_data_used,
                    "agents_used": list(set(agents_used)),
                }
                return {"final_report": final_report, "agents_used": agents_used}

        logger.info(f"Report Agent synthesizing final response for query: '{question}'")
        logger.info(
            "========== AI-MADAC DEBUG ==========\nREPORT INPUT:\n%s\n====================================",
            json.dumps({
                "question": question,
                "structured_intent": structured_intent,
                "executed_sql": (sql_result or {}).get("query"),
                "result_columns": (sql_result or {}).get("columns", []),
                "result_rows": (sql_result or {}).get("rows", [])[:10],
            }, default=str),
        )

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

        # Aggregations already contain the exact answer.  Rendering them
        # deterministically prevents an LLM from substituting unrelated
        # columns or stale generic language for the current result.
        if sql_result and structured_intent.get("operation") in {"AGGREGATE", "FREQUENCY", "PERCENTAGE"}:
            answer, insights = self._deterministic_fallback_report(
                question, analysis_result, sql_result, structured_intent
            )
        else:
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
                answer, insights = self._deterministic_fallback_report(
                    question, analysis_result, sql_result, structured_intent
                )

        # Construct final report dictionary
        chart_paths = []
        if viz_result:
            chart_paths = viz_result.get("chart_paths", [])
            # Backwards compatibility for previously generated single-chart
            # visualization results.
            if not chart_paths and viz_result.get("chart_path"):
                chart_paths = [viz_result["chart_path"]]

        # Rewrite the final answer into simple, clear, natural, conversational response for non-aggregation outputs
        if structured_intent.get("operation") not in {"AGGREGATE", "FREQUENCY", "PERCENTAGE"}:
            answer = self._rewrite_response(question, answer)

        # When a visualization was explicitly requested, use the visualization explanation and insights
        if viz_result:
            if viz_result.get("visualization_available") is False and not chart_paths:
                if any(w in question.lower() for w in ("chart", "plot", "pie", "graph", "visualiz")):
                    answer = viz_result.get("explanation", answer)
                    insights = viz_result.get("key_insights", insights)
            elif viz_result.get("explanation") and any(w in question.lower() for w in ("chart", "plot", "pie", "graph", "visualiz")):
                answer = viz_result["explanation"]
                if viz_result.get("key_insights"):
                    insights = viz_result["key_insights"]

        data_used_payload = sql_result or {}
        if isinstance(data_used_payload, dict) and viz_result:
            data_used_payload["visualization"] = viz_result
            data_used_payload["parameters"] = viz_result.get("parameters")
            data_used_payload["chart_json"] = viz_result.get("chart_json")

        final_report = {
            "answer": str(answer),
            "insights": insights,
            "chart_paths": chart_paths,
            "chart_json": viz_result.get("chart_json") if viz_result else None,
            "parameters": viz_result.get("parameters") if viz_result else None,
            "visualization_result": viz_result,
            "data_used": data_used_payload,
            "agents_used": list(set(agents_used)),
        }
        logger.info(
            "========== AI-MADAC DEBUG ==========\nFINAL RESPONSE:\n%s\n====================================",
            json.dumps({"answer": answer, "insights": insights}, default=str),
        )

        return {
            "final_report": final_report,
            "agents_used": list(set(agents_used)),
        }

    
    def _rewrite_response(self, question: str, answer: str) -> str:
        """
        Rewrite raw agent response using the Final Response Rewriter into a natural conversational reply.
        """
        if not answer or not answer.strip():
            return answer
        try:
            prompt = f"User Question: '{question}'\n\nAgent Response:\n{answer}"
            messages = [
                SystemMessage(content=RESPONSE_REWRITER_PROMPT),
                HumanMessage(content=prompt),
            ]
            response = self.llm.invoke(messages)
            rewritten = response.content.strip()
            if rewritten and len(rewritten) > 2:
                if (rewritten.startswith('"') and rewritten.endswith('"')) or (rewritten.startswith("'") and rewritten.endswith("'")):
                    rewritten = rewritten[1:-1].strip()
                return rewritten
        except Exception as e:
            logger.warning(f"Response rewriter failed: {e}. Preserving original answer.")
        return answer

    def _deterministic_fallback_report(
        self, question: str, analysis_result: dict | None, sql_result: dict | None,
        structured_intent: dict | None = None,
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
            intent = structured_intent or {}

            # SQL fallback rules return compact, meaningful result sets. Turn
            # those into an answer rather than reporting only that data exists.
            if len(rows) == 1 and len(columns) == 1:
                metric = columns[0].replace("_", " ")
                value = rows[0].get(columns[0])
                ans = f"{metric.capitalize()}: {value}."
                insights = ["This is calculated from the full uploaded dataset, not just the table preview."]
                return ans, insights

            is_grouped = bool(intent.get("group_by_needed")) or any(
                marker in (sql_result.get("query") or "").upper()
                for marker in ("GROUP BY", "COUNT(", "SUM(", "AVG(")
            )
            if is_grouped and len(columns) >= 2:
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
                group_label = group_column.replace("_", " ")
                metric_label = metric_column.replace("_", " ")
                asks_all = any(w in question.lower() for w in ("each", "all", "breakdown", "distribution", "both", "male and female", "female and male", "compare")) or len(rows) <= 3
                is_exclusive_top = any(w in question.lower() for w in ("most popular", "highest", "least popular", "lowest", "top "))
                if asks_all and not is_exclusive_top and len(rows) <= 5:
                    breakdown_str = ", ".join(f"{r.get(group_column)}: {r.get(metric_column)}" for r in rows)
                    ans = f"{group_label.capitalize()} breakdown: {breakdown_str}."
                    insights = [f"{r.get(group_column)} has {r.get(metric_column)} records." for r in rows]
                    return ans, insights

                if "popular" in question.lower() or intent.get("operation") == "FREQUENCY":
                    ans = (
                        f"The most popular {group_label} is {highest.get(group_column)} "
                        f"with {highest.get(metric_column)} investors."
                    )
                else:
                    ans = (
                        f"The highest {metric_label} is {highest.get(metric_column)} "
                        f"for {highest.get(group_column)}."
                    )
                insights = [
                    f"The ranked result contains {len(rows)} {group_label} value(s).",
                    f"The answer is calculated from the SQL result for this question.",
                ]
                return ans, insights

            if intent.get("commonality_analysis"):
                patterns = self._find_commonalities(rows, columns)
                return (
                    f"Retrieved {sql_result.get('row_count')} matching records. " + " ".join(patterns),
                    patterns,
                )

            # A compact two-column result is conventionally a category/value
            # ranking even when called outside the graph without intent state.
            if len(columns) == 2:
                group_column, metric_column = columns
                def numeric_value(row: dict) -> float:
                    try:
                        value = row.get(metric_column)
                        return float(value) if value not in (None, "") else float("-inf")
                    except (TypeError, ValueError):
                        return float("-inf")
                highest = max(rows, key=numeric_value)
                return (
                    f"The highest {metric_column.replace('_', ' ')} is {highest.get(metric_column)} for {highest.get(group_column)}.",
                    ["The ranking is calculated from the returned query rows."],
                )

            ans = f"Retrieved {sql_result.get('row_count')} matching records from the dataset."
            insights = ["These are raw records, not an aggregated ranking."]
            return ans, insights

        return "I couldn't find matching data for this question.", []

    @staticmethod
    def _find_commonalities(rows: list[dict], columns: list[str]) -> list[str]:
        """Describe dominant categorical values for ranked-record questions."""
        if not rows:
            return []
        findings = []
        for column in columns:
            values = [row.get(column) for row in rows if isinstance(row.get(column), str) and row.get(column)]
            if not values:
                continue
            value, count = Counter(values).most_common(1)[0]
            if count / len(rows) >= 0.6:
                findings.append(f"{value} appears in {count} out of {len(rows)} records for {column}.")
        return findings or ["The returned records are diverse and do not show a dominant categorical pattern."]


# Singleton instance
report_agent = ReportAgent()
