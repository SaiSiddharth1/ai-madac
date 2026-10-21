"""Regression tests for strict new-question and visualization follow-up routing."""

import unittest
from pathlib import Path

import pandas as pd

from app.agents.supervisor.agent import SupervisorAgent
from app.agents.sql.agent import SQLAgent


SCHEMA = {"columns": [
    {"name": "Avenue", "dtype": "object"},
    {"name": "Objective", "dtype": "object"},
    {"name": "Expect", "dtype": "object"},
    {"name": "Age", "dtype": "int64"},
]}


class RequestLifecycleTests(unittest.TestCase):
    def test_new_question_does_not_become_follow_up_when_previous_exists(self):
        result = SupervisorAgent().run({
            "question": "What is the average age?", "schema": SCHEMA,
            "dataset_profile": {}, "previous_validated_result": {"validated": True},
        })
        self.assertNotIn(result["intent"], {"FOLLOW_UP_VISUALIZATION", "FOLLOW_UP_FILTER", "FOLLOW_UP_SORT"})
        self.assertEqual(result["structured_intent"].get("mode"), None)

    def test_chart_reference_reuses_only_explicitly(self):
        result = SupervisorAgent().run({
            "question": "Show it as a bar chart.", "schema": SCHEMA,
            "dataset_profile": {}, "previous_validated_result": {"validated": True},
        })
        self.assertEqual(result["intent"], "FOLLOW_UP_VISUALIZATION")
        self.assertEqual(result["execution_plan"], ["visualization_agent", "report_agent"])

    def test_only_the_results_for_male_is_a_recomputed_filter_follow_up(self):
        result = SupervisorAgent().run({
            "question": "Now show only the results for male investors.", "schema": SCHEMA,
            "dataset_profile": {}, "previous_validated_result": {"validated": True},
        })
        self.assertEqual(result["intent"], "FOLLOW_UP_FILTER")
        self.assertEqual(result["execution_plan"], ["sql_agent", "report_agent"])

    def test_expected_return_cross_tab_keeps_filter_and_both_dimensions(self):
        question = "Among investors with an expected return of 30–40%, what are their most common investment avenues and objectives?"
        intent = SupervisorAgent._parse_structured_intent(question, SCHEMA)
        sql = SQLAgent._build_heuristic_query(question, SCHEMA, "investment_data", intent)
        self.assertIn('"Expect" = \'30-40%\'', sql)
        self.assertIn('GROUP BY "Avenue", "Objective"', sql)
        self.assertIn('COUNT(*) AS investor_count', sql)
        data = pd.read_csv(Path(__file__).parents[2] / "data" / "sample" / "finance_investment_data.csv")
        computed = data.loc[data["Expect"].eq("30-40%")].groupby(["Avenue", "Objective"]).size()
        # Reproduces the reported sequence: the second request's result is a
        # filtered cross-tab, not Q1's unfiltered Mutual Funds = 14.
        self.assertNotIn(14, computed.tolist())
        self.assertEqual(int(computed.loc[("Equity Market", "Wealth Creation")]), 7)


if __name__ == "__main__":
    unittest.main()
