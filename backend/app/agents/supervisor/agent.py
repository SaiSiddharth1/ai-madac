"""
Supervisor Agent.
Classifies user question intent and constructs an execution plan routing to specialized agents.
"""

import json
import logging
import re
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from app.agents.state import AgentState
from app.agents.visualization.contract import is_visualization_requested
from app.core.llm_provider import get_llm

logger = logging.getLogger(__name__)

SUPERVISOR_PROMPT = """You are the Supervisor Agent of an autonomous multi-agent data science platform.
Your task is to analyze the user's question, column schema, and profile summary of a dataset, then determine:
1. The user's primary intent.
2. The exact ordered execution plan of specialized agents required to answer the question.

AVAILABLE INTENTS:
- DATA_RETRIEVAL: Requesting raw records, row filtering, top N rows (e.g. "Show me 10 records", "List raw entries").
- AGGREGATION: Group by totals, counts, min/max metrics (e.g. "How many rows per company?", "Total revenue by year").
- RELATIONSHIP_ANALYSIS: Investigating impact, effect, relationship, correlation, influence, group comparison, or factors (e.g. "Did events affect stock impact?", "Does R&D spending correlate with revenue?", "What factors influence performance?").
- PREDICTION: Machine learning prediction, modeling, forecasting (e.g. "Predict stock impact for OpenAI", "Train a model to classify sales").
- VISUALIZATION: Explicit request for charts, plots, or graphs (e.g. "Plot stock_impact over time", "Show a bar chart").
- GENERAL_DATASET_ANALYSIS: Overall dataset profiling, health summary, schema overview.

ANALYTICAL ROUTING RULES:
- Questions containing analytical keywords such as "affect", "impact", "influence", "relationship", "correlation", "associated", "related", "factors", "difference", "effect", or "why" MUST be classified as RELATIONSHIP_ANALYSIS or DATA_ANALYSIS (NOT simple DATA_RETRIEVAL).
- For RELATIONSHIP_ANALYSIS or STATISTICAL questions, the plan should include `data_analyst_agent` (for group comparisons, missingness checks, and statistical relationship analysis) and `visualization_agent` (if visual charts would enhance understanding), followed by `report_agent`. Do NOT include `sql_agent` in RELATIONSHIP_ANALYSIS execution plans, as the Data Analyst Agent loads and processes the dataset directly.
- Do NOT route every "why" question to `ml_agent` unless explicit predictive modeling or machine learning is requested.

AVAILABLE AGENTS:
- sql_agent: Generates and executes SQL queries for data fetching and group aggregation.
- data_analyst_agent: Performs statistical EDA, group comparisons, impact analysis, and missingness quality checks.
- ml_agent: Trains machine learning models or executes inference predictions.
- visualization_agent: Generates interactive Plotly charts from retrieved data.
- report_agent: Synthesizes findings into the final structured report (ALWAYS required as final agent).

OUTPUT FORMAT:
Respond with ONLY a valid JSON object matching this schema:
{
  "intent": "<INTENT>",
  "execution_plan": ["<agent_1>", "<agent_2>", ..., "report_agent"],
  "reasoning": "<short sentence explaining routing>"
}
"""


class SupervisorAgent:
    """
    Supervisor Agent responsible for intent classification and workflow routing.
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
        Classify query intent and determine agent execution plan using Gemini Semantic Planner.
        """
        question = state.get("question", "")
        schema = state.get("schema", {})
        profile = state.get("dataset_profile", {})
        query_id = state.get("query_id", "")
        dataset_id = state.get("dataset_id", "")

        logger.info(f"Supervisor Agent processing query: '{question}'")

        # Check explicit follow up ONLY if question refers directly to previous result
        # without introducing fresh domain queries
        follow_up = self._follow_up_intent(question, bool(state.get("previous_validated_result")))
        if follow_up:
            prev = state.get("previous_validated_result") or {}
            prev_rows = prev.get("result", [])
            can_direct_visualize = bool(prev.get("validated") and (len(prev_rows) >= 2 or "result" not in prev))
            if follow_up == "FOLLOW_UP_VISUALIZATION":
                plan = ["visualization_agent", "report_agent"] if can_direct_visualize else ["sql_agent", "visualization_agent", "report_agent"]
            else:
                plan = ["sql_agent", "report_agent"]
            return {
                "intent": follow_up,
                "structured_intent": {"mode": follow_up},
                "execution_plan": plan,
                "agents_used": ["supervisor_agent"],
            }

        # Produce rigorous Query Contract using Gemini
        from app.agents.supervisor.semantic_planner import semantic_planner
        contract = semantic_planner.plan_query(
            question=question,
            schema=schema,
            dataset_id=dataset_id,
            query_id=query_id,
            dataset_profile=profile,
        )

        contract_dict = contract.model_dump()
        intent = contract.intent

        # Check if user explicitly asked for a visualization
        asks_visual = bool(re.search(r"\b(?:visually|visualize|chart|plot|graph|pie|bar|histogram|scatter|line)\b", question.lower()))
        if asks_visual:
            contract.requested_visualization = True
            contract.should_visualize = True

        # Determine routing plan based on Query Contract
        if not contract.is_related_to_dataset or intent == "UNRELATED":
            # Unrelated question: do NOT invoke SQL or Visualization (no fake charts!)
            plan = ["report_agent"]
            intent = "UNRELATED"
        elif intent == "PREDICTION":
            plan = ["ml_agent", "report_agent"]
        elif intent == "RELATIONSHIP_ANALYSIS":
            if contract.should_visualize:
                plan = ["data_analyst_agent", "visualization_agent", "visualization_validator", "chart_renderer", "report_agent"]
            else:
                plan = ["data_analyst_agent", "report_agent"]
        elif contract.should_visualize or contract.requested_visualization:
            plan = ["sql_agent", "visualization_agent", "visualization_validator", "chart_renderer", "report_agent"]
            if contract.requested_visualization or any(tok in question.lower() for tok in ("visually", "chart", "plot", "graph", "histogram", "scatter")):
                intent = "VISUALIZATION"
        else:
            plan = ["sql_agent", "report_agent"]

        # Ensure report_agent is always at the end
        if "report_agent" not in plan:
            plan.append("report_agent")

        structured_intent = {
            "operation": "AGGREGATE" if intent == "AGGREGATION" else intent,
            "target_column": contract.measure_column,
            "measure": contract.measure,
            "measure_expression": contract.measure_expression,
            "aggregation": contract.aggregation,
            "group_by_needed": bool(contract.group_by),
            "group_by": contract.group_by,
            "dimensions": contract.dimensions,
            "answer_type": contract.answer_type,
            "should_visualize": contract.should_visualize,
            "suggested_chart_type": contract.suggested_chart_type,
            "is_related_to_dataset": contract.is_related_to_dataset,
            "unrelated_reason": contract.unrelated_reason,
            "bin_width": contract.bin_width,
            "limit": contract.limit,
            "sort": contract.sort.model_dump() if contract.sort else None,
        }

        logger.info(f"Supervisor intent classified: '{intent}', plan: {plan}")
        return {
            "intent": intent,
            "query_contract": contract_dict,
            "execution_plan": plan,
            "structured_intent": structured_intent,
            "agents_used": ["supervisor_agent"],
        }

    @staticmethod
    def _follow_up_intent(question: str, has_previous: bool) -> str | None:
        """Reuse context only for an explicit anaphoric reference, never because state exists."""
        if not has_previous:
            return None
        q = f" {question.lower().strip()} "
        chart_keywords = ("bar chart", "pie chart", "line chart", "scatter", "histogram", "box plot", "heatmap", "visualize", "plot", "graph", "chart")
        asks_chart = any(token in q for token in chart_keywords)

        # Explicit filter follow-ups modifying the prior query
        filter_refinement_phrases = (
            "only the results", "only for ", "filter for ", "filter by ",
            "now show only", "just the ", "just for ", "restricted to ", "for only"
        )
        is_filter_followup = any(token in q for token in filter_refinement_phrases)
        if is_filter_followup:
            return "FOLLOW_UP_FILTER"

        # Explicit sort follow-ups
        if any(token in q for token in ("sort by", "sort it", "highest to lowest", "lowest to highest", "order by")):
            return "FOLLOW_UP_SORT"

        # An explicit visualization follow-up must contain clear anaphoric reference to the previous result
        has_anaphoric = any(token in q for token in (" it ", " that ", " this ", " same result", " same data", " same as ", " the same ", " previous result"))

        # If user explicitly mentions domain entities/columns, it is an independent new query
        explicit_entities = (
            "gender", "male", "female", "objective", "avenue", "age", "return", "expected",
            "income", "duration", "department", "salary", "sales", "product", "region", "country"
        )
        matched_entities = [token for token in explicit_entities if token in q]
        if matched_entities and not any(phrase in q for phrase in ("same result", "same data", "the same")):
            return None

        if not has_anaphoric:
            return None

        if asks_chart:
            return "FOLLOW_UP_VISUALIZATION"
        if any(token in q for token in ("explain", "why ", "what does")):
            return "FOLLOW_UP_EXPLANATION"
        return None

    @staticmethod
    def _parse_structured_intent(question: str, schema: dict[str, Any]) -> dict[str, Any]:
        """Extract the analytical operation without relying on an LLM.

        This intentionally uses only schema names and question wording.  It is
        shared state, so every downstream agent reasons about the same current
        question rather than guessing from a prior result shape.
        """
        q = question.lower()
        columns = [str(c.get("name", "")) for c in schema.get("columns", [])]

        def score(column: str) -> int:
            words = column.lower().replace("_", " ")
            singular = words[:-1] if words.endswith("s") else words
            plural = words[:-1] + "ies" if words.endswith("y") else words + "s"
            # Exact column phrases beat partial component matches; this makes
            # "Avenue" win over the similarly named Investment_Avenues.
            if re.search(r"\b" + re.escape(words) + r"\b", q) or re.search(r"\b" + re.escape(singular) + r"\b", q) or re.search(r"\b" + re.escape(plural) + r"\b", q):
                return 100 + len(words)
            parts = [p for p in words.split() if p not in {"id", "of", "the"}]
            return sum(1 for p in parts if re.search(r"\b" + re.escape(p) + r"(?:s|ies)?\b", q)) if parts else 0

        matches = [(score(column), column) for column in columns]
        matches = [(value, column) for value, column in matches if value]
        matches.sort(reverse=True)
        numeric_names = {
            str(c.get("name", "")) for c in schema.get("columns", [])
            if str(c.get("dtype", "")).lower() in {"int", "int64", "integer", "float", "float64", "double", "number"}
        }
        numeric_words = ("total", "sum", "average", "mean")
        is_count = bool(re.search(r"\b(?:how many|count|number of)\b", q))
        has_numeric_match = any(column in numeric_names for _, column in matches)
        if is_count and not has_numeric_match:
            requested_aggregate = "COUNT"
        else:
            requested_aggregate = next(("AVG" if word in q and word in {"average", "mean"} else "SUM"
                                        for word in numeric_words if word in q), None)

        # For a grouped numeric total, prefer the mentioned categorical/date
        # dimension over the mentioned numeric measure.
        dimension_matches = [(value, column) for value, column in matches if column not in numeric_names]
        dimension = (dimension_matches[0][1] if requested_aggregate and dimension_matches
                     else matches[0][1] if matches else None)
        # This dataset contains both a multi-select Investment_Avenues field
        # and the requested primary Avenue field.  Prefer the exact dimension
        # when the user says "avenue" (or explicitly names the Avenue column).
        avenue_column = next((column for column in columns if column.lower() == "avenue"), None)
        if "avenue" in q and avenue_column:
            dimension = avenue_column
        gender_column = next((column for column in columns if column.lower() == "gender"), None)
        has_male = bool(re.search(r"\bmales?\b", q))
        has_female = bool(re.search(r"\bfemales?\b", q))
        has_gender_word = bool(re.search(r"\bgenders?\b", q))
        is_gender_both = bool(gender_column and has_male and has_female)
        if gender_column and (is_gender_both or (has_gender_word and not dimension)):
            dimension = gender_column
        limit_match = re.search(r"\b(?:top|bottom)\s+(\d+)", q)
        limit_n = int(limit_match.group(1)) if limit_match else 0
        direction = "ASC" if any(word in q for word in ("bottom", "least", "lowest")) else "DESC"
        is_percentage = any(word in q for word in ("percentage", "percent", "proportion", "share"))
        is_frequency = any(phrase in q for phrase in (
            "most popular", "most common", "frequency", "for each", "each ", "by each"
        ))
        is_rank = bool(re.search(r"\b(?:top|bottom|highest|lowest|most popular|least popular)\b", q))
        # A rank of individual records ("top transactions by sales") is not
        # a grouped aggregation.  An explicit total/average by a dimension is.
        entity_rank = bool(re.search(r"\b(?:transaction|record|project)s?\b", q))
        asks_visual = bool(re.search(r"\b(?:chart|plot|graph|visual(?:ize|ly|s)?|pie|bar|bargraph|distribution|histogram)\b", q))
        group_by_needed = bool(dimension and (
            is_frequency or is_percentage or (requested_aggregate in {"SUM", "AVG"}) or
            (is_rank and not entity_rank) or "each" in q or is_gender_both or asks_visual
        ))

        if is_percentage:
            operation = "PERCENTAGE"
        elif (is_frequency or is_gender_both or (asks_visual and dimension)) and not requested_aggregate:
            operation = "FREQUENCY"
        elif is_rank and not requested_aggregate:
            operation = "RANK"
        elif is_count or group_by_needed or requested_aggregate in {"SUM", "AVG"}:
            operation = "AGGREGATE"
        else:
            operation = "RETRIEVE"

        return {
            "operation": operation,
            "target_column": dimension,
            "group_by_needed": group_by_needed,
            "aggregation": requested_aggregate or ("COUNT" if operation in {"AGGREGATE", "FREQUENCY", "PERCENTAGE"} else None),
            "sort_direction": direction,
            "limit_n": limit_n,
            "commonality_analysis": "common" in q or "similar" in q,
            "entity": "transactions" if "transaction" in q else "projects" if "project" in q else "records",
        }

    def _heuristic_routing(self, question: str) -> dict[str, Any]:
        """Rule-based fallback routing if LLM fails or is unconfigured."""
        q_lower = question.lower()

        if any(w in q_lower for w in ["predict", "forecast", "classify", "model", "train"]):
            intent = "PREDICTION"
            plan = ["ml_agent", "report_agent"]
        elif is_visualization_requested(question):
            intent = "VISUALIZATION"
            plan = ["sql_agent", "visualization_agent", "visualization_validator", "chart_renderer", "report_agent"]
        elif any(w in q_lower for w in ["affect", "impact", "influence", "relationship", "correlation", "associated", "related", "factors", "difference", "effect", "why"]):
            intent = "RELATIONSHIP_ANALYSIS"
            plan = ["data_analyst_agent", "visualization_agent", "visualization_validator", "chart_renderer", "report_agent"]
        else:
            intent = "DATA_RETRIEVAL"
            plan = ["sql_agent", "report_agent"]

        return {
            "intent": intent,
            "execution_plan": plan,
            "agents_used": ["supervisor_agent"],
        }

    @staticmethod
    def _should_visualize(question: str, intent: str = "") -> bool:
        return is_visualization_requested(question, intent)


# Singleton instance
supervisor_agent = SupervisorAgent()
