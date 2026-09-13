"""
Supervisor Agent.
Classifies user question intent and constructs an execution plan routing to specialized agents.
"""

import json
import logging
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from app.agents.state import AgentState
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
        Classify query intent and determine agent execution plan.
        """
        question = state.get("question", "")
        schema = state.get("schema", {})
        profile = state.get("dataset_profile", {})

        logger.info(f"Supervisor Agent processing query: '{question}'")

        # First check heuristic analytical triggers for instant accuracy
        q_lower = question.lower()
        analytical_keywords = [
            "affect", "impact", "influence", "relationship", "correlation",
            "associated", "related", "factors", "difference", "effect", "why"
        ]

        # Try LLM classification first
        try:
            prompt = f"""Dataset Schema Columns: {json.dumps([c.get('name') for c in schema.get('columns', [])])}
Dataset Profile Summary: {json.dumps(profile.get('summary', {}))}

User Question: "{question}"
"""
            messages = [
                SystemMessage(content=SUPERVISOR_PROMPT),
                HumanMessage(content=prompt),
            ]
            response = self.llm.invoke(messages)
            content = response.content.strip()

            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            data = json.loads(content)
            intent = data.get("intent", "DATA_RETRIEVAL")
            plan = data.get("execution_plan", ["sql_agent", "report_agent"])

            # Ensure analytical queries get data_analyst_agent and bypass sql_agent
            if any(kw in q_lower for kw in analytical_keywords):
                if intent not in ["RELATIONSHIP_ANALYSIS", "DATA_ANALYSIS", "PREDICTION"]:
                    intent = "RELATIONSHIP_ANALYSIS"
                
                # For RELATIONSHIP_ANALYSIS, bypass SQL Agent and use Data Analyst Agent directly
                if intent == "RELATIONSHIP_ANALYSIS":
                    plan = ["data_analyst_agent", "visualization_agent", "report_agent"]

            # Ensure report_agent is always at the end
            if "report_agent" not in plan:
                plan.append("report_agent")

            # Visuals are valuable for comparisons and trends, but a single
            # number (such as a unique-country count) is clearer as a direct
            # answer and table than as a meaningless one-bar chart.
            if self._should_visualize(question, intent) and "visualization_agent" not in plan:
                plan.insert(plan.index("report_agent"), "visualization_agent")

            logger.info(f"Supervisor intent classified: '{intent}', plan: {plan}")
            return {
                "intent": intent,
                "execution_plan": plan,
                "agents_used": ["supervisor_agent"],
            }

        except Exception as e:
            logger.warning(f"LLM supervisor classification failed: {e}. Falling back to rule-based routing.")
            return self._heuristic_routing(question)

    def _heuristic_routing(self, question: str) -> dict[str, Any]:
        """Rule-based fallback routing if LLM fails or is unconfigured."""
        q_lower = question.lower()

        if any(w in q_lower for w in ["predict", "forecast", "classify", "model", "train"]):
            intent = "PREDICTION"
            plan = ["ml_agent", "report_agent"]
        elif any(w in q_lower for w in ["chart", "plot", "graph", "visualize", "bar", "pie", "histogram"]):
            intent = "VISUALIZATION"
            plan = ["sql_agent", "visualization_agent", "report_agent"]
        elif any(w in q_lower for w in ["affect", "impact", "influence", "relationship", "correlation", "associated", "related", "factors", "difference", "effect", "why"]):
            intent = "RELATIONSHIP_ANALYSIS"
            plan = ["data_analyst_agent", "visualization_agent", "report_agent"]
        elif self._should_visualize(question, "DATA_RETRIEVAL"):
            intent = "DATA_RETRIEVAL"
            plan = ["sql_agent", "visualization_agent", "report_agent"]

        else:
            intent = "DATA_RETRIEVAL"
            plan = ["sql_agent", "report_agent"]

        return {
            "intent": intent,
            "execution_plan": plan,
            "agents_used": ["supervisor_agent"],
        }

    @staticmethod
    def _should_visualize(question: str, intent: str) -> bool:
        q = question.lower()
        if intent in {"RELATIONSHIP_ANALYSIS", "VISUALIZATION"}:
            return True
        return any(word in q for word in (
            "chart", "plot", "graph", "visualize", "trend", "distribution",
            "by country", "by sector", "by status", "each country", "each sector",
            "compare", "comparison", "highest", "lowest", "top ",
        ))


# Singleton instance
supervisor_agent = SupervisorAgent()
