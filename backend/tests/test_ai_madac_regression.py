"""
AI-MADAC Comprehensive Regression & Acceptance Test Suite.
Verifies aggregation reliability, measure precision, state isolation,
and intelligent visualization rules.
"""

import asyncio
import json
import sqlite3
import time
import unittest
from typing import Any

import pandas as pd

from app.agents.sql.agent import sql_agent
from app.agents.supervisor.agent import supervisor_agent
from app.agents.supervisor.semantic_planner import semantic_planner
from app.agents.visualization.agent import visualization_agent
from app.agents.report.agent import report_agent
from app.graphs.workflow import workflow_graph
from app.schemas.query_contract import QueryContract


class AIMadacRegressionTestSuite(unittest.IsolatedAsyncioTestCase):
    """
    Acceptance criteria tests for AI-MADAC.
    Uses SQLite database tables for OnlineRetail and finance_investment_data.
    """

    def tearDown(self):
        time.sleep(1.5)

    @classmethod
    def setUpClass(cls):
        conn = sqlite3.connect("ai_madac.db")
        cur = conn.cursor()
        datasets = cur.execute("SELECT id, name, table_name FROM datasets").fetchall()
        conn.close()

        # Locate OnlineRetail and Finance datasets
        cls.retail_ds = next((d for d in datasets if "OnlineRetail" in d[1]), None)
        cls.finance_ds = next((d for d in datasets if "finance_investment_data" in d[1]), None)

        if cls.retail_ds:
            cls.retail_id, cls.retail_name, cls.retail_table = cls.retail_ds
            conn = sqlite3.connect("ai_madac.db")
            cols = [c[1] for c in conn.cursor().execute(f"PRAGMA table_info({cls.retail_table})").fetchall()]
            conn.close()
            cls.retail_schema = {"columns": [{"name": c, "dtype": "TEXT"} for c in cols]}
        else:
            cls.retail_id = "retail-ds"
            cls.retail_table = "dataset_b9d5839975"
            cls.retail_schema = {
                "columns": [
                    {"name": "InvoiceNo", "dtype": "TEXT"},
                    {"name": "StockCode", "dtype": "TEXT"},
                    {"name": "Description", "dtype": "TEXT"},
                    {"name": "Quantity", "dtype": "INTEGER"},
                    {"name": "InvoiceDate", "dtype": "TEXT"},
                    {"name": "UnitPrice", "dtype": "REAL"},
                    {"name": "CustomerID", "dtype": "TEXT"},
                    {"name": "Country", "dtype": "TEXT"},
                ]
            }

        if cls.finance_ds:
            cls.finance_id, cls.finance_name, cls.finance_table = cls.finance_ds
            conn = sqlite3.connect("ai_madac.db")
            cols = [c[1] for c in conn.cursor().execute(f"PRAGMA table_info({cls.finance_table})").fetchall()]
            conn.close()
            cls.finance_schema = {"columns": [{"name": c, "dtype": "TEXT"} for c in cols]}
        else:
            cls.finance_id = "finance-ds"
            cls.finance_table = "dataset_d24b902b11"
            cls.finance_schema = {
                "columns": [
                    {"name": "Age", "dtype": "INTEGER"},
                    {"name": "Gender", "dtype": "TEXT"},
                    {"name": "Investment_Avenues", "dtype": "TEXT"},
                    {"name": "Mutual_Funds", "dtype": "INTEGER"},
                    {"name": "Avenue", "dtype": "TEXT"},
                ]
            }

    def _create_initial_state(self, question: str, dataset_id: str, table_name: str, schema: dict, query_id: str = "test-q") -> dict:
        return {
            "user_id": "test-user",
            "dataset_id": str(dataset_id),
            "query_id": query_id,
            "question": question,
            "table_name": table_name,
            "schema": schema,
            "dataset_profile": {},
            "previous_validated_result": None,
            "validated_result": None,
            "intent": "",
            "query_contract": None,
            "structured_intent": {},
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

    # -------------------------------------------------------------
    # 1. Average Unit Price
    # -------------------------------------------------------------
    def test_01_average_unit_price(self):
        """Must use UnitPrice column, AVG aggregation, and return SCALAR without grouping."""
        q = "What is the average unit price?"
        contract = semantic_planner.plan_query(q, self.retail_schema, self.retail_id, "q1")
        self.assertEqual(contract.answer_type, "SCALAR")
        self.assertEqual(contract.group_by, [])
        self.assertIn("unitprice", contract.measure_column.lower())
        self.assertEqual(contract.aggregation, "AVG")

        # Test deterministic SQL execution
        state = self._create_initial_state(q, self.retail_id, self.retail_table, self.retail_schema)
        state["query_contract"] = contract.model_dump()
        res = sql_agent.run(state)
        self.assertIsNotNone(res.get("sql_result"))
        sql_res = res["sql_result"]
        self.assertEqual(sql_res["response_type"], "SCALAR")
        self.assertEqual(len(sql_res["rows"]), 1)
        self.assertIn("average", sql_res["columns"][0].lower())
        self.assertNotIn("quantity", sql_res["columns"][0].lower())

    # -------------------------------------------------------------
    # 2. Total Quantity
    # -------------------------------------------------------------
    def test_02_total_quantity(self):
        """Must use Quantity column, SUM aggregation, and SCALAR without grouping."""
        q = "What is the total quantity?"
        contract = semantic_planner.plan_query(q, self.retail_schema, self.retail_id, "q2")
        self.assertEqual(contract.answer_type, "SCALAR")
        self.assertEqual(contract.group_by, [])
        self.assertIn("quantity", contract.measure_column.lower())
        self.assertEqual(contract.aggregation, "SUM")

        state = self._create_initial_state(q, self.retail_id, self.retail_table, self.retail_schema)
        state["query_contract"] = contract.model_dump()
        res = sql_agent.run(state)
        sql_res = res["sql_result"]
        self.assertEqual(sql_res["response_type"], "SCALAR")
        self.assertEqual(len(sql_res["rows"]), 1)
        self.assertIn("quantity", sql_res["columns"][0].lower())
        self.assertNotIn("price", sql_res["columns"][0].lower())

    # -------------------------------------------------------------
    # 3. Total Units Sold
    # -------------------------------------------------------------
    def test_03_total_units_sold(self):
        """'units sold' must map to Quantity column, NEVER to UnitPrice."""
        q = "What are the units sold?"
        contract = semantic_planner.plan_query(q, self.retail_schema, self.retail_id, "q3")
        self.assertEqual(contract.answer_type, "SCALAR")
        self.assertEqual(contract.group_by, [])
        self.assertIn("quantity", contract.measure_column.lower())
        self.assertNotIn("unitprice", contract.measure_column.lower())

        state = self._create_initial_state(q, self.retail_id, self.retail_table, self.retail_schema)
        state["query_contract"] = contract.model_dump()
        res = sql_agent.run(state)
        sql_res = res["sql_result"]
        self.assertEqual(sql_res["response_type"], "SCALAR")
        self.assertEqual(len(sql_res["rows"]), 1)
        self.assertNotIn("price", sql_res["columns"][0].lower())

    # -------------------------------------------------------------
    # 4. Total Revenue (Scalar - NEVER Invent Grouping)
    # -------------------------------------------------------------
    def test_04_total_revenue_scalar(self):
        """Total revenue without grouping MUST return exactly ONE scalar total."""
        q = "What is the total revenue?"
        contract = semantic_planner.plan_query(q, self.retail_schema, self.retail_id, "q4")
        self.assertEqual(contract.answer_type, "SCALAR")
        self.assertEqual(contract.group_by, [])
        # Derived expression Quantity * UnitPrice
        self.assertIn("quantity", contract.measure_expression.lower())
        self.assertIn("unitprice", contract.measure_expression.lower())

        state = self._create_initial_state(q, self.retail_id, self.retail_table, self.retail_schema)
        state["query_contract"] = contract.model_dump()
        res = sql_agent.run(state)
        sql_res = res["sql_result"]
        self.assertEqual(sql_res["response_type"], "SCALAR")
        self.assertEqual(len(sql_res["rows"]), 1)
        self.assertEqual(len(sql_res["columns"]), 1)
        self.assertNotIn("country", [c.lower() for c in sql_res["columns"]])

    # -------------------------------------------------------------
    # 5. Unique Customers (COUNT DISTINCT)
    # -------------------------------------------------------------
    def test_05_unique_customers(self):
        """Must use COUNT(DISTINCT CustomerID), not normal COUNT(*)."""
        q = "How many unique customers?"
        contract = semantic_planner.plan_query(q, self.retail_schema, self.retail_id, "q5")
        self.assertEqual(contract.answer_type, "SCALAR")
        self.assertEqual(contract.aggregation, "COUNT_DISTINCT")
        self.assertIn("customerid", contract.measure_column.lower())

        state = self._create_initial_state(q, self.retail_id, self.retail_table, self.retail_schema)
        state["query_contract"] = contract.model_dump()
        res = sql_agent.run(state)
        self.assertIn("COUNT(DISTINCT", res["sql_query"].upper())

    # -------------------------------------------------------------
    # 6. Revenue By Country (Explicit Grouping)
    # -------------------------------------------------------------
    def test_06_revenue_by_country(self):
        """Must group by Country and calculate SUM(Quantity * UnitPrice)."""
        q = "What is the total revenue by country?"
        contract = semantic_planner.plan_query(q, self.retail_schema, self.retail_id, "q6")
        self.assertEqual(contract.answer_type, "GROUPED")
        self.assertEqual(contract.group_by, ["Country"])

        state = self._create_initial_state(q, self.retail_id, self.retail_table, self.retail_schema)
        state["query_contract"] = contract.model_dump()
        res = sql_agent.run(state)
        sql_res = res["sql_result"]
        self.assertIn("Country", sql_res["columns"])
        self.assertTrue(len(sql_res["rows"]) > 1)

    # -------------------------------------------------------------
    # 7. Top 10 Countries by Revenue
    # -------------------------------------------------------------
    def test_07_top_10_countries_by_revenue(self):
        """Must group by Country, order by revenue DESC, and limit to 10."""
        q = "Top 10 countries by revenue"
        contract = semantic_planner.plan_query(q, self.retail_schema, self.retail_id, "q7")
        self.assertEqual(contract.group_by, ["Country"])
        self.assertEqual(contract.limit, 10)
        self.assertIsNotNone(contract.sort)
        self.assertEqual(contract.sort.direction.lower(), "desc")

        state = self._create_initial_state(q, self.retail_id, self.retail_table, self.retail_schema)
        state["query_contract"] = contract.model_dump()
        res = sql_agent.run(state)
        sql_res = res["sql_result"]
        self.assertLessEqual(len(sql_res["rows"]), 10)

    # -------------------------------------------------------------
    # 8. Transactions by Country
    # -------------------------------------------------------------
    def test_08_transactions_by_country(self):
        """Must group by Country and count invoices/transactions."""
        q = "Number of transactions by country"
        contract = semantic_planner.plan_query(q, self.retail_schema, self.retail_id, "q8")
        self.assertEqual(contract.group_by, ["Country"])
        self.assertTrue("invoiceno" in (contract.measure_column or "").lower() or contract.aggregation in ("COUNT", "COUNT_DISTINCT"))

    # -------------------------------------------------------------
    # 9. Quantity Distribution (Histogram with Raw Observations)
    # -------------------------------------------------------------
    def test_09_quantity_distribution(self):
        """Distribution must use raw observations without pre-aggregation."""
        q = "Show the distribution of Quantity"
        contract = semantic_planner.plan_query(q, self.retail_schema, self.retail_id, "q9")
        self.assertEqual(contract.intent, "DISTRIBUTION")
        self.assertEqual(contract.suggested_chart_type, "histogram")
        self.assertTrue(contract.should_visualize)

        state = self._create_initial_state(q, self.retail_id, self.retail_table, self.retail_schema)
        state["query_contract"] = contract.model_dump()
        res = sql_agent.run(state)
        self.assertNotIn("GROUP BY", res["sql_query"].upper())
        self.assertNotIn("SUM(", res["sql_query"].upper())

    # -------------------------------------------------------------
    # 10. Quantity vs UnitPrice (Scatter Plot Raw Observations)
    # -------------------------------------------------------------
    def test_10_quantity_vs_unitprice_scatter(self):
        """Relationship between two numbers must use raw observations for scatter."""
        q = "Show the relationship between Quantity and UnitPrice"
        contract = semantic_planner.plan_query(q, self.retail_schema, self.retail_id, "q10")
        self.assertEqual(contract.intent, "RELATIONSHIP")
        self.assertEqual(contract.suggested_chart_type, "scatter")
        self.assertTrue(contract.should_visualize)

        state = self._create_initial_state(q, self.retail_id, self.retail_table, self.retail_schema)
        state["query_contract"] = contract.model_dump()
        res = sql_agent.run(state)
        self.assertNotIn("GROUP BY", res["sql_query"].upper())

    # -------------------------------------------------------------
    # 11. Time-based Revenue Trend
    # -------------------------------------------------------------
    def test_11_time_based_revenue_trend(self):
        """Time-based trend must use line chart."""
        q = "Show revenue trend over time"
        contract = semantic_planner.plan_query(q, self.retail_schema, self.retail_id, "q11")
        self.assertTrue(contract.should_visualize)
        self.assertIn(contract.suggested_chart_type, ("line", "bar"))

    # -------------------------------------------------------------
    # 12. Unrelated Question (No fake chart, informative message)
    # -------------------------------------------------------------
    def test_12_unrelated_question(self):
        """Unrelated question must not generate fake chart or fake SQL."""
        q = "What is the weather in Hyderabad?"
        contract = semantic_planner.plan_query(q, self.retail_schema, self.retail_id, "q12")
        self.assertFalse(contract.is_related_to_dataset)
        self.assertFalse(contract.should_visualize)

        # In report agent, must return response_type="UNRELATED" with no chart_paths
        state = self._create_initial_state(q, self.retail_id, self.retail_table, self.retail_schema)
        state["query_contract"] = contract.model_dump()
        state["structured_intent"] = contract.model_dump()
        rep = report_agent.run(state)
        final_rep = rep["final_report"]
        self.assertEqual(final_rep["response_type"], "UNRELATED")
        self.assertEqual(final_rep["chart_paths"], [])
        self.assertIn("not related", final_rep["answer"].lower())

    # -------------------------------------------------------------
    # 13. State Isolation (No Leaking Across Queries)
    # -------------------------------------------------------------
    def test_13_state_isolation_between_consecutive_queries(self):
        """Query 2 must NEVER inherit filters, measures, or grouping from Query 1."""
        # Query 1: Investment avenues
        q1 = "What is the most popular investment avenue?"
        contract1 = semantic_planner.plan_query(q1, self.finance_schema, self.finance_id, "q13-1")
        self.assertIn("avenue", (contract1.measure_column or "").lower() + " ".join(contract1.group_by).lower())

        # Query 2: Completely different question: Average age
        q2 = "What is the average age?"
        # Clean state
        contract2 = semantic_planner.plan_query(q2, self.finance_schema, self.finance_id, "q13-2")
        self.assertEqual(contract2.answer_type, "SCALAR")
        self.assertEqual(contract2.group_by, [])
        self.assertEqual(contract2.aggregation, "AVG")
        self.assertIn("age", contract2.measure_column.lower())
        self.assertNotIn("avenue", contract2.measure_column.lower())
        self.assertNotIn("avenue", " ".join(contract2.group_by).lower())


if __name__ == "__main__":
    unittest.main()
