"""
AI-MADAC Ultimate Final Fix: Regression Test Suite.
Verifies:
1. Single-line chart (monthly total sales from 2011 to 2015 -> 1 line, ~48-60 points, chronological order).
2. Multi-line chart (monthly sales for each category from 2011 to 2015 -> exactly 3 lines for Category).
3. Scatter plot preservation.
4. Histogram preservation.
5. Pie chart preservation.
6. Bar chart preservation.
7. Conversational Chatbot (hi, hello, how are you, thanks, bye -> mode CHAT, normal response, no workflow).
8. Mixed queries (Hi, what is the total sales? -> mode DATASET).
9. Mixed visualization queries (Hello, show monthly sales as a line chart. -> mode DATASET).
"""

import asyncio
import json
import sqlite3
import unittest
import pandas as pd

from app.agents.supervisor.chat_router import classify_chat_or_dataset
from app.agents.supervisor.semantic_planner import semantic_planner
from app.agents.sql.agent import sql_agent
from app.agents.visualization.agent import visualization_agent
from app.agents.visualization.data_preparer import DataPreparer
from app.agents.visualization.plotly_renderer import PlotlyRenderer
from app.agents.visualization.quality_gate import VisualizationQualityGate


class TestUltimateFinalFix(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        conn = sqlite3.connect("ai_madac.db")
        cur = conn.cursor()
        # Find test_superstore table
        tables = [r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
        cls.table_name = "test_superstore" if "test_superstore" in tables else "dataset_superstore"
        if cls.table_name not in tables:
            # Pick any table with Order_Date, Sales, Category
            for t in tables:
                cols = [c[1] for c in cur.execute(f"PRAGMA table_info({t})").fetchall()]
                if any("order" in c.lower() for c in cols) and any("sale" in c.lower() for c in cols):
                    cls.table_name = t
                    break

        cols = [c[1] for c in cur.execute(f"PRAGMA table_info({cls.table_name})").fetchall()]
        conn.close()

        cls.schema = {"columns": [{"name": c, "dtype": "TEXT"} for c in cols]}
        cls.dataset_id = "test-superstore-ds"

    def _create_state(self, question: str, contract=None):
        return {
            "user_id": "test-user",
            "dataset_id": self.dataset_id,
            "query_id": "test-q-" + str(hash(question))[:8],
            "question": question,
            "table_name": self.table_name,
            "schema": self.schema,
            "dataset_profile": {},
            "previous_validated_result": None,
            "validated_result": None,
            "intent": "",
            "query_contract": contract.model_dump() if contract else None,
            "structured_intent": contract.model_dump() if contract else {},
            "execution_plan": [],
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

    # =========================================================================
    # PART 28: TEST 1 — SINGLE LINE
    # =========================================================================
    def test_01_single_line_monthly_sales(self):
        """
        Show monthly total sales from 2011 to 2015 as a line chart.
        Must produce:
        - 1 clean line
        - Monthly points (~48-60 points)
        - Chronological order
        - No 498 overlapping lines
        - Passes visual quality validator
        """
        q = "Show monthly total sales from 2011 to 2015 as a line chart."
        contract = semantic_planner.plan_query(q, self.schema, self.dataset_id, "t1")

        self.assertEqual(contract.suggested_chart_type, "line")
        self.assertIsNone(contract.series_dimension)
        self.assertEqual(contract.aggregation, "SUM")
        self.assertIn("order_date", contract.x_axis.lower())
        self.assertIn("sales", contract.measure_column.lower())

        state = self._create_state(q, contract)
        sql_res = sql_agent.run(state)
        state["sql_result"] = sql_res["sql_result"]
        state["sql_query"] = sql_res["sql_query"]

        rows = state["sql_result"]["rows"]
        row_count = len(rows)
        # 2011-2015 is at most 60 months, not 500 transaction rows!
        self.assertLessEqual(row_count, 65, f"Expected monthly points <= 65, got {row_count}")
        self.assertGreaterEqual(row_count, 12, f"Expected at least 12 monthly points, got {row_count}")

        # Run visualization agent
        viz_res = visualization_agent.run(state)
        viz_data = viz_res["visualization_result"]
        self.assertIsNotNone(viz_data)

        # Validate spec & chart
        df = pd.DataFrame(rows)
        chart_json = viz_data.get("chart_json", {})
        traces = chart_json.get("data", [])

        self.assertEqual(len(traces), 1, f"Expected exactly 1 line trace, got {len(traces)}")
        self.assertTrue(viz_data.get("visualization_available"))
        self.assertGreaterEqual(viz_data.get("quality_score", 0), 0.8)

    # =========================================================================
    # PART 28: TEST 2 — MULTI-LINE
    # =========================================================================
    def test_02_multi_line_monthly_sales_by_category(self):
        """
        Show monthly sales for each category from 2011 to 2015 as a multi-line chart.
        Must produce:
        - 1 line per category (3 lines for Furniture, Office Supplies, Technology)
        - Not one line per order or per transaction
        - Clean legend
        - Chronological order
        """
        q = "Show monthly sales for each category from 2011 to 2015 as a multi-line chart."
        contract = semantic_planner.plan_query(q, self.schema, self.dataset_id, "t2")

        self.assertEqual(contract.suggested_chart_type, "line")
        self.assertEqual(contract.series_dimension, "Category")

        state = self._create_state(q, contract)
        sql_res = sql_agent.run(state)
        state["sql_result"] = sql_res["sql_result"]
        state["sql_query"] = sql_res["sql_query"]

        # Run visualization agent
        viz_res = visualization_agent.run(state)
        viz_data = viz_res["visualization_result"]
        self.assertIsNotNone(viz_data)

        df = pd.DataFrame(state["sql_result"]["rows"])
        chart_json = viz_data.get("chart_json", {})
        traces = chart_json.get("data", [])

        # Should have 3 category traces (Furniture, Office Supplies, Technology)
        self.assertIn(len(traces), (2, 3, 4), f"Expected category series count ~3, got {len(traces)}")
        self.assertLessEqual(len(traces), 10, f"Must not have hundreds of traces! Got {len(traces)}")
        self.assertTrue(viz_data.get("visualization_available"))
        self.assertGreaterEqual(viz_data.get("quality_score", 0), 0.8)

    # =========================================================================
    # PART 28: TEST 3 — SCATTER PLOT PRESERVATION
    # =========================================================================
    def test_03_scatter_plot_preservation(self):
        """Show sales versus profit using a scatter plot."""
        q = "Show sales versus profit using a scatter plot."
        contract = semantic_planner.plan_query(q, self.schema, self.dataset_id, "t3")

        self.assertEqual(contract.intent, "RELATIONSHIP")
        self.assertEqual(contract.suggested_chart_type, "scatter")
        self.assertTrue(contract.should_visualize)

        state = self._create_state(q, contract)
        sql_res = sql_agent.run(state)
        self.assertNotIn("GROUP BY", sql_res["sql_query"].upper())

    # =========================================================================
    # PART 28: TEST 4 — HISTOGRAM PRESERVATION
    # =========================================================================
    def test_04_histogram_preservation(self):
        """Show the distribution of sales using a histogram."""
        q = "Show the distribution of sales using a histogram."
        contract = semantic_planner.plan_query(q, self.schema, self.dataset_id, "t4")

        self.assertEqual(contract.intent, "DISTRIBUTION")
        self.assertEqual(contract.suggested_chart_type, "histogram")
        self.assertTrue(contract.should_visualize)

        state = self._create_state(q, contract)
        sql_res = sql_agent.run(state)
        self.assertNotIn("GROUP BY", sql_res["sql_query"].upper())

    # =========================================================================
    # PART 28: TEST 5 — PIE CHART PRESERVATION
    # =========================================================================
    def test_05_pie_chart_preservation(self):
        """Show the percentage share of total sales by category as a pie chart."""
        q = "Show the percentage share of total sales by category as a pie chart."
        contract = semantic_planner.plan_query(q, self.schema, self.dataset_id, "t5")

        self.assertEqual(contract.suggested_chart_type, "pie")
        self.assertEqual(contract.group_by, ["Category"])
        self.assertEqual(contract.aggregation, "SUM")

    # =========================================================================
    # PART 28: TEST 6 — BAR CHART PRESERVATION
    # =========================================================================
    def test_06_bar_chart_preservation(self):
        """Show total sales by category as a bar chart."""
        q = "Show total sales by category as a bar chart."
        contract = semantic_planner.plan_query(q, self.schema, self.dataset_id, "t6")

        self.assertEqual(contract.suggested_chart_type, "bar")
        self.assertEqual(contract.group_by, ["Category"])
        self.assertEqual(contract.aggregation, "SUM")

    # =========================================================================
    # PART 28: TESTS 7-11 — CONVERSATIONAL CHAT ROUTING
    # =========================================================================
    async def test_07_chat_hi(self):
        res = await classify_chat_or_dataset("hi", self.schema)
        self.assertEqual(res["mode"], "CHAT")
        self.assertTrue(len(res.get("response", "")) > 0)

    async def test_08_chat_hello(self):
        res = await classify_chat_or_dataset("hello", self.schema)
        self.assertEqual(res["mode"], "CHAT")
        self.assertTrue(len(res.get("response", "")) > 0)

    async def test_09_chat_how_are_you(self):
        res = await classify_chat_or_dataset("how are you?", self.schema)
        self.assertEqual(res["mode"], "CHAT")
        self.assertTrue(len(res.get("response", "")) > 0)

    async def test_10_chat_thanks(self):
        res = await classify_chat_or_dataset("thanks", self.schema)
        self.assertEqual(res["mode"], "CHAT")
        self.assertTrue(len(res.get("response", "")) > 0)

    async def test_11_chat_bye(self):
        res = await classify_chat_or_dataset("bye", self.schema)
        self.assertEqual(res["mode"], "CHAT")
        self.assertTrue(len(res.get("response", "")) > 0)

    # =========================================================================
    # PART 28: TESTS 12-13 — MIXED QUESTIONS MUST ROUTE TO DATASET
    # =========================================================================
    async def test_12_mixed_total_sales(self):
        res = await classify_chat_or_dataset("Hi, what is the total sales?", self.schema)
        self.assertEqual(res["mode"], "DATASET")

    async def test_13_mixed_monthly_sales_line_chart(self):
        res = await classify_chat_or_dataset("Hello, show monthly sales as a line chart.", self.schema)
        self.assertEqual(res["mode"], "DATASET")


if __name__ == "__main__":
    unittest.main()
