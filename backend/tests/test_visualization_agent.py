"""Unit tests for VisualizationAgent adhering to AI-MADAC visualization specifications."""

import os
import unittest
import pandas as pd

from app.agents.visualization.agent import VisualizationAgent


class VisualizationAgentTests(unittest.TestCase):
    def setUp(self):
        self.agent = VisualizationAgent()

    def test_empty_data_returns_unavailable_payload(self):
        state = {
            "question": "Can you show in pie chart how many genders are there?",
            "sql_result": {"rows": [], "columns": ["gender", "investor_count"]},
            "analysis_result": None,
        }
        output = self.agent.run(state)
        viz = output["visualization_result"]

        self.assertIsNotNone(viz)
        self.assertFalse(viz["visualization_available"])
        self.assertIsNone(viz["chart"])
        self.assertIn("not enough data", viz["explanation"])
        self.assertEqual(viz["key_insights"], [])

    def test_single_value_kpi_skips_chart(self):
        state = {
            "question": "Show count of genders",
            "sql_result": {"rows": [{"gender_count": 2}], "columns": ["gender_count"]},
            "analysis_result": None,
        }
        output = self.agent.run(state)
        viz = output["visualization_result"]

        self.assertIsNotNone(viz)
        self.assertFalse(viz["visualization_available"])
        self.assertIsNone(viz["chart"])
        self.assertEqual(viz["key_insights"], [])

    def test_gender_pie_chart_generation(self):
        state = {
            "question": "can u show in pie chart how many genders are there",
            "sql_result": {
                "rows": [
                    {"gender": "Male", "investor_count": 23},
                    {"gender": "Female", "investor_count": 17},
                ],
                "columns": ["gender", "investor_count"],
                "query": 'SELECT "gender", COUNT(*) AS investor_count FROM "finance" GROUP BY "gender"',
                "filters": {},
            },
            "analysis_result": None,
            "structured_intent": {"target_column": "gender", "aggregation": "COUNT"},
        }
        output = self.agent.run(state)
        viz = output["visualization_result"]

        self.assertIsNotNone(viz)
        self.assertTrue(viz["visualization_available"])
        self.assertIsNotNone(viz["chart"])
        self.assertEqual(viz["chart"]["chart_type"], "pie")
        self.assertEqual(viz["chart"]["title"], "Distribution: gender")

        # Verify parameters adhere to spec
        params = viz["chart"]["parameters"]
        self.assertEqual(params["chart_type"], "Pie chart")
        self.assertEqual(params["category"], "gender")
        self.assertEqual(params["value"], "investor_count")
        self.assertEqual(params["aggregation"], "Already aggregated by source agent")
        # Ensure no unused filters are claimed
        self.assertNotIn("filters", params)

        # Verify explanation
        self.assertIn("Parameters used:", viz["explanation"])
        self.assertIn("Category: gender", viz["explanation"])
        self.assertIn("Value: investor_count", viz["explanation"])
        self.assertIn("Male", viz["explanation"])

        # Verify insights
        self.assertGreater(len(viz["key_insights"]), 0)
        self.assertTrue(any("Male" in s for s in viz["key_insights"]))

        # Verify PNG file was generated on disk
        self.assertGreater(len(viz["chart_paths"]), 0)
        png_rel_path = viz["chart_paths"][0].replace("/static/charts/", "")
        charts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "charts")
        png_full_path = os.path.join(charts_dir, png_rel_path)
        self.assertTrue(os.path.exists(png_full_path))

    def test_bar_chart_generation_with_filter(self):
        state = {
            "question": "Show investment avenues for female investors",
            "sql_result": {
                "rows": [
                    {"avenue": "Mutual Funds", "investor_count": 5},
                    {"avenue": "Fixed Deposits", "investor_count": 4},
                ],
                "columns": ["avenue", "investor_count"],
                "filters": {"gender": "Female"},
            },
            "analysis_result": None,
            "structured_intent": {"target_column": "avenue", "aggregation": "COUNT"},
        }
        output = self.agent.run(state)
        viz = output["visualization_result"]

        self.assertTrue(viz["visualization_available"])
        self.assertEqual(viz["chart"]["chart_type"], "bar")
        params = viz["chart"]["parameters"]
        self.assertEqual(params["chart_type"], "Bar chart")
        self.assertEqual(params["x_axis"], "avenue")
        self.assertEqual(params["y_axis"], "investor_count")
        # Filter is properly included when applied
        self.assertIn("gender = 'Female'", params.get("filters", ""))

    def test_gender_versus_investment_objective_must_never_produce_age(self):
        schema = {
            "columns": [
                {"name": "age", "dtype": "int"},
                {"name": "gender", "dtype": "object"},
                {"name": "investment_objective", "dtype": "object"},
                {"name": "investment_avenue", "dtype": "object"},
            ]
        }
        df_rows = [
            {"age": 28, "gender": "Female", "investment_objective": "Capital Appreciation", "investment_avenue": "Mutual Funds"},
            {"age": 34, "gender": "Male", "investment_objective": "Growth", "investment_avenue": "Equity"},
            {"age": 42, "gender": "Female", "investment_objective": "Growth", "investment_avenue": "Mutual Funds"},
            {"age": 25, "gender": "Male", "investment_objective": "Capital Appreciation", "investment_avenue": "Equity"},
        ]
        state = {
            "question": "Show gender versus investment objective.",
            "schema": schema,
            "sql_result": {"rows": df_rows, "columns": ["age", "gender", "investment_objective", "investment_avenue"]},
        }
        output = self.agent.run(state)
        viz = output["visualization_result"]

        self.assertTrue(viz["visualization_available"])
        chart = viz["chart"]
        self.assertEqual(chart["chart_type"], "bar")
        params = chart["parameters"]

        # MUST be grouped bar with investment_objective and gender
        fields_in_chart = {params.get("x_axis"), params.get("split_by"), params.get("category")}
        self.assertIn("gender", fields_in_chart)
        self.assertIn("investment_objective", fields_in_chart)

        # MUST NEVER produce Age
        self.assertNotIn("age", fields_in_chart)
        self.assertNotIn("Age by Gender", chart["title"])
        self.assertNotIn("Age by Gender", viz["explanation"].as_text())

    def test_age_versus_expected_return_produces_scatter(self):
        schema = {
            "columns": [
                {"name": "age", "dtype": "int"},
                {"name": "gender", "dtype": "object"},
                {"name": "expected_return", "dtype": "float"},
                {"name": "investment_avenue", "dtype": "object"},
            ]
        }
        df_rows = [
            {"age": 28, "gender": "Female", "expected_return": 15.0, "investment_avenue": "Mutual Funds"},
            {"age": 34, "gender": "Male", "expected_return": 20.0, "investment_avenue": "Equity"},
            {"age": 45, "gender": "Male", "expected_return": 10.0, "investment_avenue": "Bonds"},
        ]
        state = {
            "question": "Show age versus expected return.",
            "schema": schema,
            "sql_result": {"rows": df_rows, "columns": ["age", "gender", "expected_return", "investment_avenue"]},
        }
        output = self.agent.run(state)
        viz = output["visualization_result"]

        self.assertTrue(viz["visualization_available"])
        chart = viz["chart"]
        self.assertEqual(chart["chart_type"], "scatter")
        params = chart["parameters"]
        self.assertEqual(params["x_axis"], "age")
        self.assertEqual(params["y_axis"], "expected_return")
        # Must not contain gender or avenue
        self.assertNotIn("gender", [params.get("x_axis"), params.get("y_axis")])
        self.assertNotIn("investment_avenue", [params.get("x_axis"), params.get("y_axis")])

    def test_age_distribution_histogram_raw_observations(self):
        schema = {
            "columns": [
                {"name": "age", "dtype": "int"},
                {"name": "gender", "dtype": "object"},
            ]
        }
        # 10 records with raw age values
        ages = [25, 25, 28, 30, 30, 30, 35, 40, 40, 50]
        df_rows = [{"age": a, "gender": "Male" if i % 2 == 0 else "Female"} for i, a in enumerate(ages)]
        state = {
            "question": "Show the age distribution.",
            "schema": schema,
            "sql_result": {"rows": df_rows, "columns": ["age", "gender"]},
        }
        output = self.agent.run(state)
        viz = output["visualization_result"]

        self.assertTrue(viz["visualization_available"])
        chart = viz["chart"]
        self.assertEqual(chart["chart_type"], "histogram")
        self.assertEqual(chart["parameters"]["x_axis"], "age")
        self.assertEqual(chart["parameters"]["observation_count"], 10)

    def test_age_distribution_with_bins_of_5_years(self):
        schema = {"columns": [{"name": "age", "dtype": "int"}]}
        df_rows = [{"age": a} for a in [21, 23, 26, 28, 31, 35, 42, 48]]
        state = {
            "question": "Show the age distribution using bins of 5 years.",
            "schema": schema,
            "sql_result": {"rows": df_rows, "columns": ["age"]},
        }
        output = self.agent.run(state)
        viz = output["visualization_result"]

        self.assertTrue(viz["visualization_available"])
        chart = viz["chart"]
        self.assertEqual(chart["chart_type"], "histogram")
        self.assertEqual(chart["parameters"]["bin_size"], 5)

    def test_rank_investment_avenues_by_number_of_investors(self):
        schema = {"columns": [{"name": "investment_avenue", "dtype": "object"}]}
        df_rows = [
            {"investment_avenue": "Mutual Funds"},
            {"investment_avenue": "Mutual Funds"},
            {"investment_avenue": "Mutual Funds"},
            {"investment_avenue": "Equity"},
            {"investment_avenue": "Equity"},
            {"investment_avenue": "PPF"},
        ]
        state = {
            "question": "Rank investment avenues by number of investors.",
            "schema": schema,
            "sql_result": {"rows": df_rows, "columns": ["investment_avenue"]},
        }
        output = self.agent.run(state)
        viz = output["visualization_result"]

        self.assertTrue(viz["visualization_available"])
        chart = viz["chart"]
        self.assertEqual(chart["chart_type"], "bar")
        self.assertEqual(chart["parameters"]["sorting"], "Descending")
        top_rec = chart["data"][0]
        self.assertEqual(top_rec["investment_avenue"], "Mutual Funds")
        self.assertEqual(top_rec["investor_count"], 3)

    def test_state_contamination_isolation(self):
        schema = {
            "columns": [
                {"name": "age", "dtype": "int"},
                {"name": "gender", "dtype": "object"},
                {"name": "investment_objective", "dtype": "object"},
                {"name": "investment_avenue", "dtype": "object"},
                {"name": "expected_return", "dtype": "float"},
            ]
        }
        dataset_rows = [
            {"age": 28, "gender": "Female", "investment_objective": "Growth", "investment_avenue": "Mutual Funds", "expected_return": 15.0},
            {"age": 35, "gender": "Male", "investment_objective": "Income", "investment_avenue": "PPF", "expected_return": 8.0},
            {"age": 42, "gender": "Female", "investment_objective": "Capital Appreciation", "investment_avenue": "Equity", "expected_return": 20.0},
        ]
        # Step 1: Follow-up test
        prev_avenue_result = {
            "validated": True,
            "columns_used": ["investment_avenue", "investor_count"],
            "result": [{"investment_avenue": "Mutual Funds", "investor_count": 14}, {"investment_avenue": "Equity", "investor_count": 7}],
        }
        # Q2: Show it as a bar chart (legitimate follow-up)
        state_followup = {
            "question": "Show it as a bar chart.",
            "schema": schema,
            "previous_validated_result": prev_avenue_result,
            "sql_result": None,
        }
        res_followup = self.agent.run(state_followup)["visualization_result"]
        self.assertTrue(res_followup["visualization_available"])
        self.assertEqual(res_followup["chart"]["parameters"]["category"], "investment_avenue")

        # Step 2: New query with explicit fields MUST NOT reuse avenue or age
        state_new = {
            "question": "Show gender versus investment objective.",
            "schema": schema,
            "previous_validated_result": prev_avenue_result,
            "sql_result": {"rows": dataset_rows, "columns": ["age", "gender", "investment_objective", "investment_avenue", "expected_return"]},
        }
        res_new = self.agent.run(state_new)["visualization_result"]
        self.assertTrue(res_new["visualization_available"])
        chart_new = res_new["chart"]
        self.assertIn("gender", [chart_new["parameters"].get("x_axis"), chart_new["parameters"].get("split_by")])
        self.assertIn("investment_objective", [chart_new["parameters"].get("x_axis"), chart_new["parameters"].get("split_by")])
        self.assertNotIn("age", [chart_new["parameters"].get("x_axis"), chart_new["parameters"].get("split_by")])
        self.assertNotIn("investment_avenue", [chart_new["parameters"].get("x_axis"), chart_new["parameters"].get("split_by")])


if __name__ == "__main__":
    unittest.main()
