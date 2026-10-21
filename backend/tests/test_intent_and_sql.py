"""
Comprehensive tests for intent classification, SQL generation, and report synthesis.
Tests the fixes for incorrect query interpretation in the multi-agent workflow.
"""

import unittest
from collections import Counter

from app.agents.supervisor.agent import SupervisorAgent
from app.agents.sql.agent import SQLAgent
from app.agents.report.agent import ReportAgent


# ── Test schema simulating a sales dataset ──
SALES_SCHEMA = {
    "columns": [
        {"name": "transaction_id", "dtype": "int64"},
        {"name": "sale_date", "dtype": "object"},
        {"name": "product_category", "dtype": "object"},
        {"name": "region", "dtype": "object"},
        {"name": "sales_amount", "dtype": "float64"},
        {"name": "quantity", "dtype": "int64"},
        {"name": "customer_name", "dtype": "object"},
    ]
}

# ── Test schema simulating a finance dataset ──
FINANCE_SCHEMA = {
    "columns": [
        {"name": "country", "dtype": "object"},
        {"name": "project_name", "dtype": "object"},
        {"name": "source_of_financing", "dtype": "object"},
        {"name": "commitment_in_u_a", "dtype": "float64"},
    ]
}

INVESTMENT_SCHEMA = {
    "columns": [
        {"name": "Age", "dtype": "int64"},
        {"name": "Gender", "dtype": "object"},
        {"name": "Investment_Avenues", "dtype": "object"},
        {"name": "Objective", "dtype": "object"},
        {"name": "Duration", "dtype": "object"},
        {"name": "Expect", "dtype": "object"},
    ]
}


class TestStructuredIntentParsing(unittest.TestCase):
    """Test that the Supervisor correctly parses structured intent from questions."""

    def test_top_10_transactions_is_rank_not_aggregate(self):
        """The original bug: 'top 10 transactions by sales amount' must be RANK, not AGGREGATE."""
        si = SupervisorAgent._parse_structured_intent(
            "Identify the top 10 transactions by sales amount and explain what they have in common.",
            SALES_SCHEMA,
        )
        self.assertEqual(si["operation"], "RANK")
        self.assertFalse(si["group_by_needed"])
        self.assertEqual(si["limit_n"], 10)
        self.assertEqual(si["sort_direction"], "DESC")
        self.assertTrue(si["commonality_analysis"])
        self.assertIn(si["entity"], ("transactions", "transaction"))

    def test_top_10_dates_by_total_sales_is_aggregate(self):
        """'top 10 dates by total sales' SHOULD be aggregation with GROUP BY."""
        si = SupervisorAgent._parse_structured_intent(
            "Show the top 10 dates by total sales amount.",
            SALES_SCHEMA,
        )
        self.assertEqual(si["aggregation"], "SUM")
        self.assertTrue(si["group_by_needed"])
        self.assertEqual(si["limit_n"], 10)

    def test_how_many_unique_regions(self):
        si = SupervisorAgent._parse_structured_intent(
            "How many unique regions are in the dataset?",
            SALES_SCHEMA,
        )
        self.assertEqual(si["aggregation"], "COUNT")

    def test_average_sales_per_category(self):
        si = SupervisorAgent._parse_structured_intent(
            "What is the average sales amount per product category?",
            SALES_SCHEMA,
        )
        self.assertEqual(si["aggregation"], "AVG")

    def test_bottom_5_transactions(self):
        si = SupervisorAgent._parse_structured_intent(
            "Show me the bottom 5 transactions by sales amount.",
            SALES_SCHEMA,
        )
        self.assertEqual(si["operation"], "RANK")
        self.assertEqual(si["sort_direction"], "ASC")
        self.assertEqual(si["limit_n"], 5)

    def test_list_all_records_no_limit(self):
        si = SupervisorAgent._parse_structured_intent(
            "List all records from the dataset.",
            SALES_SCHEMA,
        )
        self.assertEqual(si["limit_n"], 0)

    def test_commonality_detection_explicit(self):
        si = SupervisorAgent._parse_structured_intent(
            "What do the top transactions have in common?",
            SALES_SCHEMA,
        )
        self.assertTrue(si["commonality_analysis"])

    def test_commonality_detection_similar(self):
        si = SupervisorAgent._parse_structured_intent(
            "Find similar patterns among the highest sales.",
            SALES_SCHEMA,
        )
        self.assertTrue(si["commonality_analysis"])


class TestSQLGenerationHeuristic(unittest.TestCase):
    """Test that the SQL heuristic builder generates correct queries."""

    def test_top_10_transactions_generates_select_star_with_limit(self):
        """The core bug fix: individual record retrieval must NOT GROUP BY."""
        si = SupervisorAgent._parse_structured_intent(
            "Identify the top 10 transactions by sales amount",
            SALES_SCHEMA,
        )
        sql = SQLAgent._build_heuristic_query(
            "Identify the top 10 transactions by sales amount",
            SALES_SCHEMA,
            "sales_data",
            si,
        )
        self.assertIn("SELECT *", sql)
        self.assertIn("ORDER BY", sql)
        self.assertIn("LIMIT 10", sql)
        self.assertNotIn("GROUP BY", sql)
        self.assertNotIn("SUM", sql)

    def test_top_10_dates_by_total_sales_generates_group_by(self):
        """Grouped aggregation must still work correctly."""
        si = SupervisorAgent._parse_structured_intent(
            "Show the top 10 dates by total sales amount",
            SALES_SCHEMA,
        )
        # When si.operation != RANK, the heuristic falls through to the grouped branch
        sql = SQLAgent._build_heuristic_query(
            "Show the top 10 dates by total sales amount",
            SALES_SCHEMA,
            "sales_data",
            si,
        )
        self.assertIn("GROUP BY", sql)
        self.assertIn("SUM", sql)
        self.assertIn("LIMIT 10", sql)

    def test_top_countries_by_total_commitment_uses_sum(self):
        """Existing test: aggregation with SUM for financial commitments."""
        si = SupervisorAgent._parse_structured_intent(
            "Create a bar chart of the top 10 countries by total financial commitment (commitment_in_u_a)",
            FINANCE_SCHEMA,
        )
        sql = SQLAgent._build_heuristic_query(
            "Create a bar chart of the top 10 countries by total financial commitment (commitment_in_u_a)",
            FINANCE_SCHEMA,
            "finance1",
            si,
        )
        self.assertIn('SUM("commitment_in_u_a")', sql)
        self.assertIn('GROUP BY "country"', sql)
        self.assertNotIn('COUNT(*)', sql)

    def test_top_projects_by_commitment_sorts_individual(self):
        """Projects are treated as individual entities."""
        si = SupervisorAgent._parse_structured_intent(
            "Top 10 projects by commitment",
            FINANCE_SCHEMA,
        )
        sql = SQLAgent._build_heuristic_query(
            "Top 10 projects by commitment",
            FINANCE_SCHEMA,
            "finance1",
            si,
        )
        self.assertIn('ORDER BY', sql)
        self.assertIn('LIMIT 10', sql)

    def test_how_many_countries_gives_count_distinct(self):
        si = SupervisorAgent._parse_structured_intent(
            "How many countries are in the dataset?",
            FINANCE_SCHEMA,
        )
        sql = SQLAgent._build_heuristic_query(
            "How many countries are in the dataset?",
            FINANCE_SCHEMA,
            "finance1",
            si,
        )
        self.assertIn("COUNT(DISTINCT", sql)

    def test_bottom_5_sales_generates_asc_order(self):
        si = SupervisorAgent._parse_structured_intent(
            "Show me the bottom 5 transactions by sales amount",
            SALES_SCHEMA,
        )
        sql = SQLAgent._build_heuristic_query(
            "Show me the bottom 5 transactions by sales amount",
            SALES_SCHEMA,
            "sales_data",
            si,
        )
        self.assertIn("ASC", sql)
        self.assertIn("LIMIT 5", sql)
        self.assertNotIn("GROUP BY", sql)


class TestReportFallback(unittest.TestCase):
    """Test the deterministic fallback report for different result shapes."""

    def test_individual_records_not_called_groups(self):
        """Multi-column individual records should NOT say 'N groups'."""
        si = {"operation": "RANK", "entity": "transactions", "group_by_needed": False,
              "commonality_analysis": False, "limit_n": 10}
        answer, insights = ReportAgent()._deterministic_fallback_report(
            "Top 10 transactions by sales amount",
            None,
            {
                "columns": ["transaction_id", "sale_date", "product_category", "region", "sales_amount"],
                "rows": [
                    {"transaction_id": 1, "sale_date": "2023-10-20", "product_category": "Electronics", "region": "North", "sales_amount": 5000},
                    {"transaction_id": 2, "sale_date": "2023-10-21", "product_category": "Electronics", "region": "South", "sales_amount": 4500},
                ],
                "row_count": 2,
            },
            si,
        )
        self.assertNotIn("group", answer.lower())
        self.assertIn("records", answer.lower())

    def test_commonality_analysis_finds_patterns(self):
        """When asked about commonalities, the report should analyze categorical fields."""
        si = {"operation": "RANK", "entity": "transactions", "group_by_needed": False,
              "commonality_analysis": True, "limit_n": 5}
        rows = [
            {"id": 1, "category": "Electronics", "region": "North", "amount": 5000},
            {"id": 2, "category": "Electronics", "region": "South", "amount": 4500},
            {"id": 3, "category": "Electronics", "region": "North", "amount": 4000},
            {"id": 4, "category": "Electronics", "region": "East", "amount": 3500},
            {"id": 5, "category": "Clothing", "region": "North", "amount": 3000},
        ]
        answer, insights = ReportAgent()._deterministic_fallback_report(
            "Top 5 transactions by amount and what they have in common",
            None,
            {"columns": ["id", "category", "region", "amount"], "rows": rows, "row_count": 5},
            si,
        )
        # Should mention Electronics as a dominant pattern
        full_text = answer + " ".join(insights)
        self.assertTrue(
            "Electronics" in full_text or "4 out of 5" in full_text or "80%" in full_text,
            f"Expected commonality mention of Electronics, got: {full_text}"
        )

    def test_grouped_results_still_work(self):
        """2-column grouped results should still produce group-style report."""
        si = {"operation": "AGGREGATE", "group_by_needed": True}
        answer, insights = ReportAgent()._deterministic_fallback_report(
            "Total sales by region",
            None,
            {
                "columns": ["region", "total_sales"],
                "rows": [
                    {"region": "North", "total_sales": 15000},
                    {"region": "South", "total_sales": 12000},
                ],
                "row_count": 2,
            },
            si,
        )
        self.assertIn("15000", answer)
        self.assertIn("North", answer)

    def test_single_value_result(self):
        answer, _ = ReportAgent()._deterministic_fallback_report(
            "What is the total sales?",
            None,
            {"columns": ["total_sales"], "rows": [{"total_sales": 95000}], "row_count": 1},
            {},
        )
        self.assertIn("95000", answer)

    def test_no_data_friendly_message(self):
        answer, _ = ReportAgent()._deterministic_fallback_report(
            "Show me invalid things", None, None, {}
        )
        self.assertIn("couldn't", answer.lower())


class TestCommonalityFinder(unittest.TestCase):
    """Test the _find_commonalities static method directly."""

    def test_dominant_category_detected(self):
        rows = [
            {"name": "A", "type": "Gold", "price": 100},
            {"name": "B", "type": "Gold", "price": 200},
            {"name": "C", "type": "Gold", "price": 150},
            {"name": "D", "type": "Silver", "price": 50},
        ]
        patterns = ReportAgent._find_commonalities(rows, ["name", "type", "price"])
        found_gold = any("Gold" in p for p in patterns)
        self.assertTrue(found_gold, f"Expected Gold pattern, got: {patterns}")

    def test_no_dominant_pattern_reports_diversity(self):
        rows = [
            {"name": "A", "type": "X"},
            {"name": "B", "type": "Y"},
            {"name": "C", "type": "Z"},
            {"name": "D", "type": "W"},
        ]
        patterns = ReportAgent._find_commonalities(rows, ["name", "type"])
        self.assertTrue(any("diverse" in p.lower() or "don't show" in p.lower() for p in patterns))

    def test_empty_rows_returns_empty(self):
        patterns = ReportAgent._find_commonalities([], ["col1"])
        self.assertEqual(patterns, [])


class TestMetricGuardrail(unittest.TestCase):
    """Test that the metric guardrail interacts correctly with structured intent."""

    def test_guardrail_still_triggers_on_amount(self):
        """The word 'amount' should still trigger the guardrail (heuristic path)."""
        self.assertTrue(SQLAgent._requires_metric_guardrail(
            "Top 10 transactions by sales amount"
        ))

    def test_guardrail_triggers_on_total(self):
        self.assertTrue(SQLAgent._requires_metric_guardrail(
            "What is the total revenue?"
        ))

    def test_guardrail_does_not_trigger_on_simple_list(self):
        self.assertFalse(SQLAgent._requires_metric_guardrail(
            "Show me all records"
        ))


if __name__ == "__main__":
    unittest.main()
