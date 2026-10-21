"""End-to-end LangGraph regression for the streaming request-state shape."""

from unittest.mock import patch
import unittest

import pandas as pd

from app.agents.sql.agent import sql_agent
from app.agents.supervisor.agent import supervisor_agent
from app.graphs.workflow import workflow_graph


QUESTION = "What is the most popular investment avenue based on the Avenue column?"
SCHEMA = {"columns": [
    {"name": "age", "dtype": "int64"},
    {"name": "gender", "dtype": "object"},
    {"name": "investment_avenues", "dtype": "object"},
    {"name": "avenue", "dtype": "object"},
    {"name": "source", "dtype": "object"},
]}


class _UnavailableLLM:
    def invoke(self, _messages):
        raise RuntimeError("test fallback: LLM unavailable")


class LiveWorkflowAggregationTests(unittest.IsolatedAsyncioTestCase):
    async def test_streaming_shape_reaches_report_with_aggregated_result(self):
        # This is intentionally the same shape built in api/queries.py, where
        # the browser's SSE endpoint starts the graph.
        initial_state = {
            "user_id": "test-user",
            "dataset_id": "test-dataset",
            "query_id": "test-query",
            "question": QUESTION,
            "table_name": "finance_investment_data",
            "schema": SCHEMA,
            "dataset_profile": {},
            "intent": "",
            "execution_plan": [],
            "structured_intent": {},
            "sql_query": None,
            "sql_result": None,
            "analysis_result": None,
            "ml_result": None,
            "visualization_result": None,
            "rag_result": None,
            "final_report": None,
            "errors": [],
            "agents_used": [],
        }
        sql_seen = []

        def execute(sql, con):
            sql_seen.append(sql)
            self.assertIn('SELECT "avenue", COUNT(*) AS investor_count', sql)
            self.assertIn('GROUP BY "avenue"', sql)
            self.assertIn("LIMIT 1", sql)
            self.assertNotIn("SELECT *", sql)
            return pd.DataFrame([{"avenue": "Mutual Funds", "investor_count": 14}])

        old_supervisor_llm = supervisor_agent._llm
        try:
            supervisor_agent._llm = _UnavailableLLM()
            with patch("app.agents.sql.agent.pd.read_sql", side_effect=execute):
                result = await workflow_graph.ainvoke(initial_state)
        finally:
            supervisor_agent._llm = old_supervisor_llm

        self.assertEqual(len(sql_seen), 1)
        self.assertEqual(result["intent"], "AGGREGATION")
        self.assertEqual(result["execution_plan"], ["sql_agent", "report_agent"])
        self.assertEqual(result["structured_intent"]["target_column"], "avenue")
        self.assertEqual(result["sql_result"]["columns"], ["avenue", "investor_count"])
        self.assertEqual(result["sql_result"]["rows"], [{"avenue": "Mutual Funds", "investor_count": 14}])
        self.assertEqual(
            result["final_report"]["answer"],
            "The most popular avenue is Mutual Funds with 14 investors.",
        )
        self.assertNotIn("gender", result["final_report"]["answer"].lower())
        self.assertNotIn("age", result["final_report"]["answer"].lower())


if __name__ == "__main__":
    unittest.main()
